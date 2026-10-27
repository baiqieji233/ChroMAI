"""Valid model widths, learnable horizon projection and retained audio context."""
import pytest
import torch
from torch.nn import functional as F

from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model
from test_composer_audio_grounded_v3 import inputs


ATOMS = ['1', '2', '3', '4', '5', '6', '7', '8', 'A1', 'C']


def config(width, heads):
    return ComposerConfig(
        atom_vocab_size=len(ATOMS)+1, architecture_profile=GROUNDED_PROFILE,
        d_model=width, nhead=heads, layers=1, dim_feedforward=width*2,
        dropout=0., history_events=8, action_slots=3, head_timing_dim=99,
        lens_width=8, lens_chunk_size=3, future_horizon_policy='candidate_horizon_v1',
    )


@pytest.mark.parametrize(('width','heads','horizon_heads'), [
    (30,6,3), (18,6,3), (198,6,3), (32,8,4), (24,3,3), (35,5,1), (30,1,1),
])
def test_horizon_keeps_valid_base_model_dimensions(width, heads, horizon_heads):
    model=build_model(config(width,heads),ATOMS)
    assert model.future_horizon.nhead==horizon_heads
    assert width % model.future_horizon.nhead==0


def test_six_head_model_with_non_four_divisible_width_can_learn_horizon():
    torch.manual_seed(81)
    model=build_model(config(30,6),ATOMS)
    batch=inputs()
    targets=torch.tensor([[1,2,0],[3,0,0]])
    out=model(**batch,candidate_indices=torch.tensor([1,2]),action_targets=targets)
    loss=F.cross_entropy(out['action_logits'].reshape(-1,len(ATOMS)+1),targets.reshape(-1))
    loss.backward()
    grad=model.future_horizon.output.weight.grad
    assert torch.isfinite(loss)
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_horizon_keeps_the_existing_global_audio_encoder_context():
    # The added direct readout window does not make the preceding audio
    # encoder causal or remove its already available whole-candidate context.
    torch.manual_seed(22)
    model=build_model(config(16,4),ATOMS).eval()
    with torch.no_grad():
        model.future_horizon.output.weight.normal_(0,.2)
    batch=inputs()
    beats=torch.tensor([[.5,1.,2.,4.,11.,12.]]).repeat(2,1)
    batch['candidate_features'][...,120]=beats
    batch['candidate_features'][...,121]=beats*.4
    captured=[]
    hook=model.future_horizon.register_forward_hook(
        lambda module,args,out:captured.append(out.detach().clone()))
    try:
        with torch.no_grad():
            model(**batch,candidate_indices=torch.tensor([1,1]))
            changed={key:value.clone() for key,value in batch.items()}
            changed['candidate_features'][:,4,:120]+=5.
            model(**changed,candidate_indices=torch.tensor([1,1]))
    finally:
        hook.remove()
    assert (captured[0][:,1]-captured[1][:,1]).abs().max()>1e-6
