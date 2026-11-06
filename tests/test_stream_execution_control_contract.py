"""Explicit source preferences and deployed completion are different inputs."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.stream_data import collate_stream
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer.stream_train import resolve_learning_policies
from test_stream_draft_integration import build


def model_batch(monkeypatch,fraction,*,rows=1):
    ds,old,_,_=build(monkeypatch)
    model=StreamComposer(replace(old.config,deployment_condition_fraction=fraction),ds.codec)
    batch=collate_stream([ds[1]]*rows)
    return model.train(),batch


def level_only(controls,only_level_probability=.5,**kwargs):
    known=torch.zeros_like(controls,dtype=torch.bool);known[:,0]=True
    return known


def test_new_stage_mixes_controls_but_old_direct_and_resume_contract_stays_unchanged():
    old=StreamComposerConfig()
    assert old.deployment_condition_fraction==1.
    assert resolve_learning_policies(SimpleNamespace())['deployment_condition_fraction']==.25
    assert resolve_learning_policies(SimpleNamespace(),old,resume=False)['deployment_condition_fraction']==.25
    assert resolve_learning_policies(SimpleNamespace(),old,resume=True)['deployment_condition_fraction']==1.
    with pytest.raises(ValueError,match='Resume inherits deployment_condition_fraction'):
        resolve_learning_policies(SimpleNamespace(deployment_condition_fraction=.25),old,resume=True)
    with pytest.raises(ValueError,match='fraction'):
        StreamComposerConfig(deployment_condition_fraction=1.1)


def test_explicit_source_preferences_execute_verbatim_without_disabling_profile_learning(monkeypatch):
    model,batch=model_batch(monkeypatch,0.,rows=2)
    # These are legacy preferences, not a feasible probability marginal:
    # a direct all-four type lock would incorrectly reject this chart.
    controls=batch['condition'].clone();controls[:,8:12]=.1
    batch['condition']=controls
    monkeypatch.setattr(model.song_profile,'sample_known_mask',level_only)
    output=model(batch)
    assert output['manual_condition_chart_mask'].tolist()==[True]
    assert output['unified_profile_evidence']['known_mask'].all()
    assert not output['profile_loss_evidence']['known_mask'][:,1:].any()
    torch.testing.assert_close(output['_causal_profile_condition'],controls,rtol=0,atol=0)
    loss,metrics=model.loss(output,batch)
    expected=(output['profile_loss_evidence']['joint_profile_nll']*batch['unified_profile_chart_weights']).sum()
    torch.testing.assert_close(metrics['song_profile_nll'],expected)
    assert expected>0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.song_profile.parameters())


def test_deployment_execution_never_reads_unknown_source_controls(monkeypatch):
    model,batch=model_batch(monkeypatch,1.)
    monkeypatch.setattr(model.song_profile,'sample_known_mask',level_only)
    torch.manual_seed(871);a=model(batch)
    changed={**batch,'condition':batch['condition'].clone()}
    changed['condition'][:,1:]=.2
    torch.manual_seed(871);b=model(changed)
    assert 'profile_loss_evidence' not in a
    torch.testing.assert_close(a['_causal_profile_condition'],b['_causal_profile_condition'],rtol=0,atol=0)
    torch.testing.assert_close(a['execution_kind_logits'],b['execution_kind_logits'],rtol=0,atol=0)


def test_recovery_choice_and_controls_override_source_row_sampling(monkeypatch):
    model,batch=model_batch(monkeypatch,0.)
    frozen=batch['condition'].clone();frozen[:,1:]=.2;frozen[:,0]=.8
    known=level_only(frozen)
    batch.update(committed_song_component=torch.tensor([2]),
        committed_execution_controls=frozen,committed_known_mask=known)
    output=model(batch)
    assert output['manual_condition_chart_mask'].tolist()==[False]
    assert output['selected_song_component'].tolist()==[2]
    torch.testing.assert_close(output['_causal_profile_condition'],frozen,rtol=0,atol=0)
    torch.testing.assert_close(output['unified_profile_evidence']['causal_condition'],frozen*known)
