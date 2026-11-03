import torch
from maichart.composer.stream_timing import ternary_probability_stages

def test_timing_attribution_preserves_logits_and_separates_stage_effects():
    raw=torch.tensor([0.,0.,0.,0.]);mode=raw+torch.tensor([0.,0.,2.,2.]);curve=mode-torch.tensor([0.,0.,2.,2.])
    saved=[x.clone() for x in (raw,mode,curve)]
    result=ternary_probability_stages(raw,mode,curve,torch.tensor([0,12,6,18]))
    assert result[0]==.5 and result[2]==.5 and result[1]>.8
    assert all(torch.equal(a,b) for a,b in zip(saved,(raw,mode,curve)))
