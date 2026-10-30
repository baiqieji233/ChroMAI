"""Phase-aware song history: score/gradient equivalence and checkpoint wiring."""
import math
import pytest
import torch

from maichart.composer.quality_context import (
    QualityContext,SONG_RELATIVE_TIME_DIM,SONG_RELATIVE_TIME_PERIODS,song_relative_time_scores)
from maichart.composer.checkpoint import migrate_audio_grounded,save,load
from test_composer_echo_stage2_integration import _echo_fixture
from test_composer_audio_grounded_v3 import ATOMS


def dense_scores(query,offsets,ages):
    delta=offsets[...,None]+ages[:,None]
    features=[torch.log1p(delta.clamp_min(0)),torch.exp(-delta.clamp_min(0)/4)]
    for period in SONG_RELATIVE_TIME_PERIODS:
        features.extend(((delta*(2*math.pi/period)).sin(),(delta*(2*math.pi/period)).cos()))
    return torch.einsum('bchf,bcf->bch',torch.stack(features,-1),query)/math.sqrt(SONG_RELATIVE_TIME_DIM)


@pytest.mark.parametrize('scale',[1.,16.,128.])
def test_factorized_phase_scores_and_gradients_match_ten_feature_reference(scale):
    torch.manual_seed(27)
    query=torch.randn(3,6,10,requires_grad=True)
    offsets=torch.rand(3,6)*4
    ages=torch.rand(3,9)*scale-1
    expected=dense_scores(query,offsets,ages)
    actual=song_relative_time_scores(query,offsets,ages)
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=2e-4)
    weights=torch.randn_like(actual)
    old_gradient=torch.autograd.grad((expected*weights).sum(),query,retain_graph=True)[0]
    new_gradient=torch.autograd.grad((actual*weights).sum(),query)[0]
    torch.testing.assert_close(new_gradient,old_gradient,atol=1e-4,rtol=2e-4)


@pytest.mark.parametrize('index,period',list(enumerate(SONG_RELATIVE_TIME_PERIODS)))
def test_each_period_distinguishes_phase_and_matches_whole_cycles(index,period):
    query=torch.zeros(1,3,10);query[...,3+2*index]=math.sqrt(10)
    offsets=torch.tensor([[0.,period/2,period]])
    actual=song_relative_time_scores(query,offsets,torch.zeros(1,1))
    torch.testing.assert_close(actual.flatten(),torch.tensor([1.,-1.,1.]),atol=2e-6,rtol=0)


def test_paired_time_projection_has_live_gradients_and_checkpoint_recipe(tmp_path):
    model,data=_echo_fixture('song_paired_v3')
    assert model.quality_context.relative_time_query.out_features==10
    with torch.no_grad():
        model.quality_context.output.weight.normal_(std=.05)
        # The fixture's macro head starts exactly uniform. Use a learned-head
        # state so action NLL can supervise the context, as after warmup/v23.
        model.conserved_action_head.macro_head.weight.normal_(std=.1)
    result=model(**data)
    torch.nn.functional.cross_entropy(result['action_logits'][:,0],torch.tensor([1,2])).backward()
    gradient=model.quality_context.relative_time_query.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    path=tmp_path/'paired_echo.pt'
    save(path,model,None,epoch=0,args={},manifest=tmp_path/'unused',atom_catalog=ATOMS)
    restored,payload=load(path)
    assert restored.quality_context.relative_time_query.out_features==10
    assert payload['training_objective_recipe']['configuration_quality']['relative_time_features']=='log_decay_and_1_2_3_4_beat_sin_cos_10d'
    restored.eval()
    torch.testing.assert_close(restored(**data)['action_logits'],result['action_logits'],atol=0,rtol=0)


def test_migration_preserves_trained_echo_while_adding_song_time_features():
    old,_=_echo_fixture('cooperative_v2')
    new=migrate_audio_grounded(old,ATOMS,configuration_quality_policy='song_paired_v3')
    assert new.quality_context.relative_time_query.out_features==10
    for key,value in old.echo_prior.state_dict().items():
        torch.testing.assert_close(new.echo_prior.state_dict()[key],value,atol=0,rtol=0)
    assert not hasattr(new.quality_context,'anchor_audio')
    assert QualityContext(32,paired_v3=False).pair_time_query.out_features==6


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for BF16 integration')
def test_song_echo_bf16_forward_backward_is_finite():
    model,data=_echo_fixture('song_paired_v3');model=model.cuda().train()
    data={k:({n:t.cuda() for n,t in v.items()} if isinstance(v,dict) else v.cuda()) for k,v in data.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(**data,packed_actions=True)
        rows,slots=result['action_slot_indices'].unbind(-1)
        loss=torch.nn.functional.cross_entropy(result['action_logits'],data['action_targets'][rows,slots])
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:assert torch.isfinite(parameter.grad).all()
