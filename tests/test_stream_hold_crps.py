"""Physical-duration proper scores on the final regular-Hold mode law."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from maichart.composer.stream_data import collate_stream
from maichart.composer.stream_duration import bounded_duration_crps
from maichart.composer.stream_duration import HoldDurationCalibration,_duration_coefficients
from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
from maichart.composer.stream_train import resolve_learning_policies
from test_stream_draft_integration import build


def test_crps_distinguishes_far_tail_errors_without_forbidding_true_long_holds():
    seconds=torch.tensor([[.5,1.,14.]])
    target=torch.tensor([1])
    near=torch.tensor([[.4,.6,0.]])
    far=torch.tensor([[0.,.6,.4]])
    a=bounded_duration_crps(near,seconds,target)
    b=bounded_duration_crps(far,seconds,target)
    assert b>a>0  # Identical gold-class probability; different wrong tails.
    torch.testing.assert_close(a,torch.tensor([.16*(.5-1/3)]))
    long=torch.tensor([[0.,0.,1.]])
    assert bounded_duration_crps(long,seconds,torch.tensor([2]))==0
    assert bounded_duration_crps(near,seconds,torch.tensor([2]))>0


def test_equal_duration_modifier_variants_share_cdf_mass_exactly():
    split=torch.tensor([[.2,.1,.4,.3]],requires_grad=True)
    score=bounded_duration_crps(split,torch.tensor([[.5,1.,1.,14.]]),torch.tensor([1]))
    combined=bounded_duration_crps(torch.tensor([[.2,.5,.3]]),torch.tensor([[.5,1.,14.]]),torch.tensor([1]))
    torch.testing.assert_close(score,combined)
    score.sum().backward()
    torch.testing.assert_close(split.grad[:,1],split.grad[:,2])


def test_regular_hold_score_uses_final_mode_and_draft_detail_probabilities(monkeypatch):
    ds,old,_,_=build(monkeypatch)
    model=StreamComposer(replace(old.config,duration_audio_policy='off',duration_objective_policy='hold_crps_v2'),ds.codec)
    event=ds.codec.encode('3h[4:1]')
    context=torch.randn(1,model.config.d_model,requires_grad=True)
    mode=torch.randn(1,model.config.phrase_modes,8,requires_grad=True)
    duration=dict(seconds_per_beat=torch.tensor([60/190]),valid=torch.tensor([True]))
    base,mode_nll,score,mode_score=model.regular_detail_log_probs(context,event['shape'],
        torch.tensor([event['phase']]),torch.tensor([event['details']]),duration_context=duration,
        return_modes=True,mode_context=mode,return_calibration=True)
    assert torch.isfinite(mode_score).all() and mode_score.shape==(1,4)
    assert (mode_score-mode_score.mean(-1,keepdim=True)).abs().sum()>0
    mode_score[0,0].backward()
    assert mode.grad.abs().sum()>0 and context.grad.abs().sum()>0
    assert model.delta_head.weight.grad is None
    assert model.detail_mode_control.features.weight.grad.abs().sum()>0


def test_full_loss_contains_mode_duration_score_and_respects_configuration_mask(monkeypatch):
    ds,old,_,_=build(monkeypatch)
    model=StreamComposer(replace(old.config,duration_objective_policy='hold_crps_v2'),ds.codec).eval()
    batch=collate_stream([ds[len(ds)-2]])
    with torch.no_grad():output=model(batch)
    model.config.duration_calibration_weight=0.
    categorical,_=model.loss(output,batch)
    model.config.duration_calibration_weight=.1
    calibrated,metrics=model.loss(output,batch)
    assert calibrated>categorical
    assert metrics['duration_calibration_target_count']>0
    assert metrics['duration_calibration_mean_mode_score']>0
    batch={**batch,'configuration_mask':torch.zeros_like(batch['configuration_mask'])}
    model.config.duration_calibration_weight=0.;without,_=model.loss(output,batch)
    model.config.duration_calibration_weight=.1;with_score,_=model.loss(output,batch)
    torch.testing.assert_close(with_score,without,rtol=0,atol=0)


def test_old_duration_contract_stays_categorical_while_fresh_and_finetune_enable_score():
    old=StreamComposerConfig.from_dict({})
    assert old.duration_objective_policy=='categorical_v1'
    assert resolve_learning_policies(SimpleNamespace(),old,resume=True)['duration_objective_policy']=='categorical_v1'
    assert resolve_learning_policies(SimpleNamespace())['duration_objective_policy']=='hold_crps_v2'
    assert resolve_learning_policies(SimpleNamespace(),old,resume=False)['duration_objective_policy']=='hold_crps_v2'


def test_zero_short_and_long_hold_catalog_stays_fully_calibrated_without_legacy_change():
    atoms=['1h[1:0]','1hb[1:0]','1hx[1:0]','1h[8:1]','1h[2:23]']
    codec=EventSetCodec.fit(atoms)
    calibration=HoldDurationCalibration(codec)
    shape=codec.encode(atoms[0])['shape']
    assert (shape,0) in calibration._tables
    assert calibration.coverage()['catalog_candidates']==5
    assert _duration_coefficients('1h[1:0]') is None  # Existing v50 scorer unchanged.
    table=getattr(calibration,calibration._tables[shape,0])
    torch.testing.assert_close(table.sum(-1).sort().values,torch.tensor([0.,0.,0.,.5,46.]))
    probabilities=torch.tensor([[.1,.1,.1,.4,.3]],requires_grad=True)
    context=dict(seconds_per_beat=torch.tensor([1.]),valid=torch.tensor([True]))
    scores=calibration(probabilities,shape,0,context,torch.tensor([codec.encode('1h[8:1]')['details'][0]]))
    assert torch.isfinite(scores).all() and scores>0
    scores.sum().backward();assert torch.isfinite(probabilities.grad).all()
