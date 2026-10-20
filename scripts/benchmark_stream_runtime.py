"""Bounded synthetic CUDA timing; no dataset/checkpoint writes or quality claims.

Run from the repo root: python scripts/benchmark_stream_runtime.py --output result.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch

from maichart.form.codec import RationalCodec
from maichart.stream.model import ChartStreamTransformer, StreamConfig
from maichart.stream.symmetry import KINEMATIC_DIM
from maichart.stream.train import _loss
from maichart.stream.runtime import attention_backend
from maichart.stream.training_batch import pack_batch,transfer_batch


def batch(config, batch_size, device):
    b, length, nodes, queries, archive = batch_size, config.max_tokens, config.detail_audio_nodes, 160, 256
    codec = RationalCodec()
    rand = lambda *shape: torch.rand(*shape, device=device)
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.bool, device=device)
    clock = rand(b, queries, 20)
    clock[..., 19] = .5
    clock[..., 17] = .5
    target = torch.randint(7, len(codec), (b, length), device=device)
    target[:, ::10] = codec.at_id
    target[:, -1] = codec.eos_id
    decoder = torch.cat((target.new_full((b, 1), codec.bos_id), target[:, :-1]), 1)
    indices = torch.arange(length, device=device)[None].expand(b, -1) * queries // length
    times = torch.linspace(0, 160, nodes, device=device)[None].expand(b, -1)
    query_seconds = torch.linspace(15, 30, queries, device=device)[None].expand(b, -1)
    count = length // 10
    result = dict(audio=rand(b, 1024, 120), audio_padding_mask=zeros(b, 1024),
        audio_times_sec=torch.linspace(0, 160, 1024, device=device)[None].expand(b, -1),
        condition=rand(b, 16), condition_null=zeros(b),
        decoder_input=decoder, target=target, target_padding_mask=zeros(b, length),
        target_quality=rand(b, length), quality_weight=rand(b),
        history=target[:, :256], history_padding_mask=zeros(b, 256),
        history_clock=rand(b, 256, 20), decoder_clock=clock.gather(1, indices[..., None].expand(-1, -1, 20)),
        decoder_query_index=indices, query_clock=clock, query_seconds=query_seconds,
        query_audio=rand(b, queries, 120), query_actions=rand(b, queries, 51),
        query_kinematics=rand(b, queries, KINEMATIC_DIM),
        detail_audio=rand(b, nodes, 120), detail_times_sec=times, detail_audio_padding_mask=zeros(b, nodes),
        archive_audio=rand(b, archive, 120), archive_padding_mask=zeros(b, archive),
        archive_beats=rand(b, archive)*15, archive_seconds=rand(b, archive)*10,
        archive_event_index=torch.arange(archive, device=device)[None].expand(b, -1),
        archive_flat_tokens=target[:, :archive].reshape(-1),
        archive_token_event=torch.arange(b*archive, device=device),
        archive_token_position=rand(b*archive), archive_token_inverse_length=torch.ones(b*archive, device=device),
        plan_labels=rand(b, length, 10), plan_mask=(target == codec.at_id), plan_mask_padding_mask=zeros(b, length),
        curve_target=rand(b, 1024, 4), curve_target_padding_mask=zeros(b, 1024),
        curve_quality=rand(b, 1024), curve_time_weight=torch.ones(b, 1024, device=device)/1024,
        curve_example_weight=torch.ones(b, device=device),
        alignment_event_indices=(torch.arange(count, device=device)*10)[None].expand(b, -1),
        alignment_audio_index=torch.searchsorted(times.contiguous(), query_seconds[:, :count].contiguous())+1,
        alignment_event_mask=~zeros(b, count), alignment_event_weight=rand(b, count),
        position_alignment_target=rand(b, nodes), position_alignment_mask=~zeros(b, nodes),
        detail_time_weight=torch.ones(b, nodes, device=device)/nodes, alignment_audio_quality=rand(b, nodes),
        cycle_exit_mask=(target == codec.at_id))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--attention-backend', choices=('auto','cudnn','efficient'),default='auto')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(13)
    device = torch.device('cuda:0')
    # Leave several GiB for Windows, display and other applications.
    torch.cuda.set_per_process_memory_fraction(.68, device)
    config = StreamConfig(len(RationalCodec()), d_model=384, nhead=8, layers=6,
        dim_feedforward=1536, dropout=.1, grounding_profile='causal_audio_v1')
    model = ChartStreamTransformer(config).to(device)
    data = batch(config, args.batch_size, device)

    def training_step():
        model.train(); model.zero_grad(set_to_none=True)
        with attention_backend(args.attention_backend,device), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(data)
            loss, _ = _loss(output, data, cycle_exit_weight=.25)
        loss.backward()

    def timing(fn, count=5):
        values = []
        for _ in range(2): fn()
        torch.cuda.synchronize()
        for _ in range(count):
            start = time.perf_counter(); fn(); torch.cuda.synchronize()
            values.append(time.perf_counter()-start)
        return {'median_seconds': statistics.median(values), 'seconds': values}

    result = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'batch_size': args.batch_size, 'attention_backend':args.attention_backend,'training': timing(training_step)}
    host={key:value.cpu().pin_memory() for key,value in data.items()}
    packed=pack_batch(host)
    result['transfer_per_field']=timing(lambda:transfer_batch(host,device))
    result['transfer_packed']=timing(lambda:transfer_batch(packed,device))
    model.zero_grad(set_to_none=True); model.eval()
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            training_step()
        result['profile'] = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=22)
        result['sync_profile'] = prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=15)
    model.zero_grad(set_to_none=True); model.eval()
    for precision in ('fp32', 'bf16'):
        @torch.inference_mode()
        def decoding():
            with (torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()):
                memory = torch.zeros(1, 1280, config.d_model, device=device)
                mask = torch.zeros(1, 1280, dtype=torch.bool, device=device)
                clock = data['decoder_clock'][:1, :1]
                context = {key: torch.zeros(1, 1, config.d_model, device=device)
                           for key in ('local', 'archive', 'action')}
                context['global'] = torch.zeros(1, config.d_model, device=device)
                cache = None
                for i in range(40):
                    _, cache = model.decode_step(data['decoder_input'][:1, i:i+1], i, memory, mask,
                        cache, timing_features=clock, grounding_context=context)
        result['decode_'+precision] = timing(decoding, 3)
    result['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if 'profile' not in key}, indent=2))
    if args.profile: print(result['profile']); print(result['sync_profile'])


if __name__ == '__main__':
    main()
