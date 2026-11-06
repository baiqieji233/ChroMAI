"""The editor readout learns counts without changing the streaming actor."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from maichart.composer.stream_curve_preview import preview_count_loss
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer.stream_train import resolve_learning_policies,upgrade_learning_stage
from maichart.composer.stream_data import collate_stream
from test_stream_draft_integration import build


def test_preview_loss_reaches_only_the_small_head_with_real_executor_controls(monkeypatch):
    ds,old,_,_=build(monkeypatch)
    model=StreamComposer(replace(old.config,curve_preview_policy='audio_counts_v1'),ds.codec)
    model.load_state_dict(old.state_dict(),strict=False);model.train()
    batch=collate_stream([ds[1]])
    frozen=batch['condition'].clone();frozen[:,1:]=.2;frozen[:,0]=.8
    known=torch.zeros_like(frozen,dtype=torch.bool);known[:,0]=True
    batch.update(committed_song_component=torch.tensor([2]),committed_execution_controls=frozen,
        committed_known_mask=known)
    observed=[]
    model.curve_preview_head.register_forward_pre_hook(lambda module,args:observed.append(args[0]))
    output=model(batch)
    torch.testing.assert_close(output['_causal_profile_condition'],frozen,rtol=0,atol=0)
    assert not observed[0].requires_grad
    preview,_=preview_count_loss(output['curve_preview_logits'],batch,output['curve_preview_rows'],
        audio_grid_seconds=model.config.audio_grid_seconds)
    preview.backward()
    for name,parameter in model.named_parameters():
        if name.startswith('curve_preview_head.'):
            assert parameter.grad is not None and parameter.grad.abs().sum()>0,name
        else:assert parameter.grad is None,name


def test_preview_aggregation_preserves_counts_and_actual_seconds_once():
    # The last native cell is shorter; nominal bin edges cannot invent its
    # missing time or duplicate the count of a cell crossing that boundary.
    exposure=torch.tensor([[.7,.7,.7,.7,.2]])
    target=torch.tensor([[[0.,1.,1.],[0.,2.,1.],[0.,0.,0.],[0.,3.,2.],[0.,1.,1.]]])
    batch=dict(audio_plan_target=target,audio_plan_exposure=exposure,audio_plan_weight=exposure,
        song_inverse=torch.tensor([0]),song_seconds=torch.tensor([[0.,.7,1.4,2.1,2.8]]),
        unified_profile_chart_weights=torch.tensor([1.]))
    rates=torch.tensor([2.,3.]);logits=torch.log(torch.expm1(rates)).expand(1,5,2).clone().requires_grad_()
    loss,metrics=preview_count_loss(logits,batch,torch.tensor([0]),audio_grid_seconds=.7)
    widths=torch.tensor([2.1,.9]);counts=torch.tensor([[3.,2.],[4.,3.]])
    predicted=widths[:,None]*rates
    expected=(predicted-counts*predicted.log()+torch.lgamma(counts+1.)).sum()/3.
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(metrics['curve_preview_exposure_seconds'],torch.tensor(3.))
    torch.testing.assert_close(metrics['curve_preview_target_heads'],target[...,1].sum())
    torch.testing.assert_close(metrics['curve_preview_target_events'],target[...,2].sum())
    loss.backward();assert torch.isfinite(logits.grad).all()


def test_old_checkpoints_keep_preview_off_and_new_stage_can_add_it(monkeypatch):
    assert StreamComposerConfig.from_dict({}).curve_preview_policy=='off'
    old=StreamComposerConfig()
    assert resolve_learning_policies(SimpleNamespace(),old,resume=True)['curve_preview_policy']=='off'
    assert resolve_learning_policies(SimpleNamespace())['curve_preview_policy']=='audio_counts_v1'
    assert resolve_learning_policies(SimpleNamespace(),old,resume=False)['curve_preview_policy']=='audio_counts_v1'
    ds,model,_,_=build(monkeypatch)
    assert not hasattr(model,'curve_preview_head')
    upgraded=upgrade_learning_stage(model,dict(curve_preview_policy='audio_counts_v1',curve_preview_weight=.05))
    assert upgraded.curve_preview_head.out_features==2
    assert upgraded.curve_preview_trained_updates==0
    for key,value in model.state_dict().items():
        torch.testing.assert_close(upgraded.state_dict()[key],value,rtol=0,atol=0)
