"""Run explicitly with the root agent's GPU acceptance checks."""
import pytest
import torch
from torch.nn import functional as F
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer.event_set_codec import EventSetCodec


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA BF16 attention regression')
def test_stream_dropout_math_policy_preserves_single_key_zero_query_key_gradient():
    if not torch.cuda.is_bf16_supported():pytest.skip('BF16 unavailable')
    model=StreamComposer(StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,dropout=.1),
                         EventSetCodec.fit(['1'])).train()
    q=torch.randn(1,1,16,48,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    v=torch.full_like(q,2.,requires_grad=True)
    mask=torch.full((1,1,1,16),-torch.inf,device='cuda',dtype=torch.bfloat16);mask[...,0]=0
    with torch.autocast('cuda',dtype=torch.bfloat16):
        with model.attention_context('cuda'):
            result=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=.1)
    result.backward(torch.ones_like(result))
    assert q.grad.abs().max()<1e-6 and k.grad.abs().max()<1e-6
    assert torch.isfinite(v.grad).all() and torch.any(result!=2.)
