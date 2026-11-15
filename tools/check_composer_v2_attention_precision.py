"""Regression check for audio-attention gradients with large shared key offsets.

Run with the project's Python environment. CUDA checks skip on CPU-only hosts.
"""
from contextlib import contextmanager
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from maichart.composer_v2.model import Attention


@contextmanager
def reference_attention():
    def forward(query, key, value, *, attn_mask, dropout_p, is_causal):
        assert not dropout_p and not is_causal
        dtype = query.dtype
        with torch.autocast(query.device.type, enabled=False):
            scores = query.float() @ key.float().transpose(-1, -2)
            scores = scores / math.sqrt(query.size(-1)) + attn_mask.float()
            return (scores.softmax(-1) @ value.float()).to(dtype)

    with patch.object(F, 'scaled_dot_product_attention', forward):
        yield


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for fused-backend regression')
class AudioAttentionPrecision(unittest.TestCase):
    def test_large_key_offsets_keep_correct_gradients(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        for dtype in (torch.bfloat16, torch.float16):
            for kind in ('local', 'relative'):
                with self.subTest(dtype=dtype, kind=kind):
                    torch.manual_seed(42)
                    module = Attention(128, 2, local=kind == 'local', relative=kind == 'relative').cuda()
                    query = torch.randn(1, 2, 8, 64, device='cuda').to(dtype)
                    key = (torch.randn(1, 2, 1, 64, device='cuda') * 300
                           + torch.randn(1, 2, 32, 64, device='cuda') * 2).to(dtype)
                    value = (torch.randn(1, 2, 1, 64, device='cuda') * 300
                             + torch.randn(1, 2, 32, 64, device='cuda') * 10).to(dtype)
                    upstream = torch.randn(1, 8, 128, device='cuda').to(dtype)

                    def compute():
                        inputs = [tensor.detach().clone().requires_grad_() for tensor in (query, key, value)]
                        query_shape = torch.zeros(1, 8, 128, device='cuda')
                        mask = torch.arange(32, device='cuda')[None] < 27
                        with torch.autocast('cuda', dtype=dtype):
                            output = module(query_shape, projected_query=inputs[0], kv=tuple(inputs[1:]),
                                            key_mask=mask, query_beats=torch.arange(8, device='cuda')[None].float(),
                                            key_beats=torch.arange(32, device='cuda')[None].float())
                        parameters = ([module.past, module.future] if kind == 'local' else [module.relative_beat])
                        gradients = torch.autograd.grad(output, inputs + parameters, upstream)
                        return output.detach(), gradients

                    actual, actual_gradients = compute()
                    with reference_attention():
                        expected, expected_gradients = compute()
                    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
                    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
                        error = (actual_grad.float() - expected_grad.float()).norm()
                        relative_error = error / expected_grad.float().norm().clamp_min(1e-8)
                        self.assertLess(float(relative_error), .03)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
