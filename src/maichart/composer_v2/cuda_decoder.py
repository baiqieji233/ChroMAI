"""Replay a single FP32 decoder step with the same layers as eager inference."""
import json
import threading

import torch


class _StepGraph:
    """The graph owns scratch space. Every branch keeps its own authoritative KV."""

    def __init__(self, layers, x, condition, beats, audio, state, tile, capacity):
        self.layers = layers
        self.capacity = capacity
        self.x = torch.empty_like(x)
        self.condition = torch.empty_like(condition)
        self.beats = torch.empty_like(beats)
        self.mask = torch.zeros(1, capacity, dtype=torch.bool, device=x.device)
        attn = layers[0].self_attention
        self.history = x.new_zeros((len(layers), 2, 1, attn.heads, capacity, attn.dim))
        self.audio = {'local': self.x}
        self.memories = []
        for i, (layer, live) in enumerate(zip(layers, state['layers'])):
            memory = dict(self_kv_storage=tuple(self.history[i].unbind(0)))
            local = layer.local_memory(audio, live)
            memory['local_kv'] = tuple(torch.empty_like(v[tile:tile+1]) for v in local)
            if layer.song_attention is not None:
                memory['song_kv'] = tuple(torch.empty_like(v) for v in layer.song_memory(audio, live))
            self.memories.append(memory)
        for name in ('local_mask', 'local_beats', 'song_mask', 'song_beats'):
            if name in audio:
                value = audio[name]
                self.audio[name] = torch.empty_like(value[tile:tile+1] if name.startswith('local') else value)
        self.local_source = self.song_source = None
        self.local_mask = self.local_clock = self.song_mask = self.song_clock = None
        self.tile = None
        self._inputs(x, condition, beats, audio, state, tile)
        self.mask[:, -1] = True
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward()
        torch.cuda.current_stream(x.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self._forward()

    def _inputs(self, x, condition, beats, audio, state, tile):
        self.x.copy_(x)
        self.condition.copy_(condition)
        self.beats.copy_(beats)
        if (self.local_source is not audio['local'] or self.tile != tile
                or self.local_mask is not audio['local_mask'] or self.local_clock is not audio['local_beats']):
            for layer, fixed, live in zip(self.layers, self.memories, state['layers']):
                for dst, src in zip(fixed['local_kv'], layer.local_memory(audio, live)):
                    dst.copy_(src[tile:tile+1])
            for key in ('local_mask', 'local_beats'):
                self.audio[key].copy_(audio[key][tile:tile+1])
            self.local_source, self.tile = audio['local'], tile
            self.local_mask, self.local_clock = audio['local_mask'], audio['local_beats']
        if (self.song_source is not audio['song'] or self.song_clock is not audio.get('song_beats')
                or self.song_mask is not audio['song_mask']):
            for layer, fixed, live in zip(self.layers, self.memories, state['layers']):
                if layer.song_attention is not None:
                    for dst, src in zip(fixed['song_kv'], layer.song_memory(audio, live)):
                        dst.copy_(src)
            for key in ('song_mask', 'song_beats'):
                if key in self.audio:
                    self.audio[key].copy_(audio[key])
            self.song_source, self.song_clock = audio['song'], audio.get('song_beats')
            self.song_mask = audio['song_mask']

    def _forward(self):
        x = self.x
        for layer, fixed in zip(self.layers, self.memories):
            # The last slot is always the current token. True prefix positions,
            # including holes from masked reference views, are supplied at replay.
            state = dict(fixed, self_kv=tuple(v[..., :-1, :] for v in fixed['self_kv_storage']))
            x = layer(x, self.condition, self.mask, self.beats, self.audio, state=state, groups=0)
        return x

    def _live_storage(self, state, past):
        packed = state.get('packed_cuda_kv')
        first = state['layers'][0]
        if (packed is None or packed.size(-2) <= past
                or first['self_kv_storage'][0].data_ptr() != packed.data_ptr()):
            packed = torch.empty_like(self.history)
            for i, live in enumerate(state['layers']):
                buffers = tuple(packed[i].unbind(0))
                for buf, old in zip(buffers, live['self_kv']):
                    buf[..., :past, :].copy_(old)
                live['self_kv_storage'] = buffers
                live['self_kv'] = tuple(buf[..., :past, :] for buf in buffers)
            state['packed_cuda_kv'] = packed
        return packed

    def run(self, x, condition, beats, mask, audio, state, tile):
        past = state['length']
        self._inputs(x, condition, beats, audio, state, tile)
        packed = self._live_storage(state, past)
        # Copy from the actual branch, including after rewind, prefill or repair.
        self.history[..., :past, :].copy_(packed[..., :past, :])
        self.mask.zero_()
        self.mask[:, :past].copy_(mask[:, :past])
        self.mask[:, -1:].copy_(mask[:, -1:])
        self.graph.replay()
        packed[..., past:past+1, :].copy_(self.history[..., -1:, :])
        for live in state['layers']:
            live['self_kv'] = tuple(buf[..., :past+1, :] for buf in live['self_kv_storage'])
        return self.output


class CudaDecoderRuntime:
    def __init__(self, layers):
        self.layers = layers
        self.reset()

    def reset(self):
        self.graph = self.shape = None
        self.owner_thread = threading.get_ident()
        self.stream = None
        self.failed = False
        self.report = dict(requested='auto', selected='torch', precision='fp32',
                           scope='single_token_decoder_layers', captures=0, replayed_steps=0)

    def step(self, x, condition, beats, mask, audio, state, groups):
        if (self.failed or threading.get_ident() != self.owner_thread
                or not isinstance(groups, int) or x.shape[:2] != (1, 1)
                or not x.is_cuda or x.dtype != torch.float32 or torch.is_autocast_enabled('cuda')
                or not state['length'] or 'self_kv' not in state['layers'][0]):
            return None
        # Static graph IO belongs to one stream. Independent callers can read
        # the same model eagerly using their own decoder state.
        stream = torch.cuda.current_stream(x.device)
        if self.stream is None:
            self.stream = stream
        elif self.stream != stream:
            return None
        length = mask.size(1)
        shape = (x.device, audio['local_mask'].size(1), audio['song_mask'].size(1))
        build = self.graph is None or self.shape != shape or self.graph.capacity < length
        try:
            if build:
                bucket = max(128, 1 << (length-1).bit_length())
                capacity = max(length, min(bucket, self.layers[0].cache_capacity))
                self.graph = None
                with torch.cuda.device(x.device):
                    graph = _StepGraph(self.layers, x, condition, beats, audio, state, groups, capacity)
                # Check the first real query of each shape before enabling replay.
                ref = x
                ref_kv = []
                for layer, live in zip(self.layers, state['layers']):
                    trial = dict(live)
                    ref = layer(ref, condition, mask, beats, audio, state=trial, groups=groups)
                    ref_kv.extend(v[..., -1, :] for v in trial['self_kv'])
                ref_kv = torch.stack(ref_kv).reshape_as(graph.history[..., -1, :])
                self.graph, self.shape = graph, shape
            result = self.graph.run(x, condition, beats, mask, audio, state, groups)
            if build:
                if not (torch.allclose(result, ref, rtol=2e-5, atol=2e-5)
                        and torch.allclose(self.graph.history[..., -1, :], ref_kv, rtol=2e-5, atol=2e-5)):
                    raise ValueError('CUDA graph output differs from the eager decoder')
                self.report['captures'] += 1
                self.report['selected'] = 'cuda_graph'
                if self.report['captures'] == 1:
                    print('composer_v2_acceleration ' + json.dumps(self.report), flush=True)
        except (RuntimeError, ValueError) as exc:
            # Prefix storage may contain a trial append. Eager replay starts at
            # the original live length and rewrites every layer's next slot.
            for live in state['layers']:
                if 'self_kv' in live:
                    live['self_kv'] = tuple(v[..., :state['length'], :] for v in live['self_kv'])
            self.graph, self.failed = None, True
            self.report.update(selected='torch', fallback=str(exc))
            print('composer_v2_acceleration ' + json.dumps(self.report), flush=True)
            return None
        self.report['replayed_steps'] += 1
        return result
