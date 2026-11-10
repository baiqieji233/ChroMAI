"""Regression checks for optional Intel acceleration: python tools/check_intel_acceleration.py."""
from copy import deepcopy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from maichart.composer_v2.acceleration import resolve_audio_device
from maichart.composer_v2.model import ComposerV2Config, DecoderBlock
from maichart.composer_v2.openvino_decoder import DecoderStep, DecoderRuntime


class IntelAccelerationTests(unittest.TestCase):
    def test_phase_readout_preserves_values_and_gradients(self):
        from maichart.composer_v2.onset_readout import apply_phase_evidence
        torch.manual_seed(17)
        logits=torch.randn(2,4,11,requires_grad=True)
        scores=torch.randn(3,3,requires_grad=True)
        queries=[(0,2,0),(1,1,2)]
        ids=torch.tensor([1,4,7])
        rows=torch.tensor([0,1]);positions=torch.tensor([2,1]);tiles=torch.tensor([0,2])
        old=logits[rows,positions].index_select(1,ids)
        shifted=old+scores.index_select(0,tiles)
        updated=shifted-(torch.logsumexp(shifted,-1)-torch.logsumexp(old,-1))[:,None]
        expected=logits.clone()
        expected[rows[:,None],positions[:,None],ids[None,:]]=updated
        actual=apply_phase_evidence(logits,scores,queries,ids)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        wanted=torch.autograd.grad(expected.square().sum(),(logits,scores),retain_graph=True)
        got=torch.autograd.grad(actual.square().sum(),(logits,scores))
        for a,b in zip(got,wanted):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_audio_selection_and_explicit_cpu(self):
        with patch('maichart.composer_v2.acceleration.openvino_devices', return_value=('CPU','GPU','NPU')):
            self.assertEqual(resolve_audio_device('auto','cpu'),'openvino-gpu')
            self.assertEqual(resolve_audio_device('auto','cuda'),'cuda')
            self.assertEqual(resolve_audio_device('cpu','cuda'),'cpu')
        with patch('maichart.composer_v2.acceleration.openvino_devices', return_value=()):
            self.assertEqual(resolve_audio_device('auto'),'cpu')

    def test_fused_step_and_rewound_history(self):
        torch.manual_seed(42)
        config=ComposerV2Config(d_model=32,dim_feedforward=64,layers=2,nhead=4,
                               dropout=0.,audio_backend='native',music_context_policy='song_detail_v1')
        layers=torch.nn.ModuleList([DecoderBlock(config,song=False),DecoderBlock(config,song=True)]).eval()
        wrapper=DecoderStep(layers).eval()
        audio=dict(local=torch.randn(2,7,32),local_mask=torch.ones(2,7,dtype=torch.bool),
                   local_beats=torch.arange(7).float()[None].expand(2,-1),
                   song=torch.randn(1,5,32),song_mask=torch.ones(1,5,dtype=torch.bool),
                   song_beats=torch.arange(5).float()[None])
        condition=torch.randn(1,32)
        with torch.inference_mode():
            memories=[{} for _ in layers]
            x=torch.randn(1,6,32)
            for layer,memory in zip(layers,memories):
                x=layer(x,condition,torch.ones(1,6,dtype=torch.bool),torch.zeros(1,6),audio,
                        state=memory,groups=(torch.tensor([0]),torch.arange(6)[None],torch.ones(1,6,dtype=torch.bool)))
            for retained in (6,3,1):
                states=deepcopy(memories)
                for memory in states:
                    memory['self_kv']=tuple(v[:,:,:retained].clone() for v in memory['self_kv'])
                    memory.pop('self_kv_storage',None)
                x=torch.randn(1,1,32);beats=torch.ones(1,1)
                mask=torch.ones(1,retained+1,dtype=torch.bool)
                inputs=[x,condition,beats,mask,audio['local_mask'][1:2],audio['local_beats'][1:2],
                        audio['song_mask'],audio['song_beats']]
                for layer,memory in zip(layers,states):
                    inputs.extend((*memory['self_kv'],*(v[1:2] for v in memory['local_kv'])))
                    if layer.song_attention is not None:inputs.extend(memory['song_kv'])
                actual=wrapper(*inputs)
                original=deepcopy(states)
                expected=x
                for layer,memory in zip(layers,states):
                    expected=layer(expected,condition,mask,beats,audio,state=memory,groups=1)
                torch.testing.assert_close(actual[0],expected)
                for i,memory in enumerate(states):
                    for value,reference in zip(actual[1+2*i:3+2*i],memory['self_kv']):
                        torch.testing.assert_close(value,reference[:,:,-1:])
                # Fixed-shape GPU padding must preserve the exact live history.
                runtime=DecoderRuntime(SimpleNamespace(decoder=layers),SimpleNamespace(cache_dir='unused',cpu_threads=1))
                padded=runtime._gpu_inputs(inputs)
                padded_result=wrapper(*(torch.from_numpy(v) for v in padded))
                torch.testing.assert_close(padded_result[0],actual[0])
                if retained==3:
                    try:import openvino
                    except ImportError:continue
                    with TemporaryDirectory() as temporary:
                        model=SimpleNamespace(decoder=layers,config=config)
                        args=SimpleNamespace(cache_dir=temporary,cpu_threads=2,inference_backend='openvino-cpu')
                        runtime=DecoderRuntime(model,args)
                        output=runtime.step(x,condition,beats,mask,audio,{'layers':original},1)
                        torch.testing.assert_close(output,expected,rtol=2e-5,atol=2e-5)

    def test_edge_chunks_do_not_pad_or_compile(self):
        from maichart.composer_v2.openvino_audio import OpenVINOMERT
        runner=OpenVINOMERT.__new__(OpenVINOMERT)
        for device in ('NPU','GPU'):
            runner.device=device
            self.assertIsNone(runner.infer(np.zeros((1,400),dtype=np.float32)))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
