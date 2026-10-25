from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch

from maichart.composer.acoustics import AcousticContext, native_dynamics
from maichart.composer.model import ChartComposer, ComposerConfig
from maichart.composer.train import _loss
from maichart.composer.guard import EventGuard
from maichart.event_timing import EventPoint, EventTempo


def test_native_short_windows_match_waveform_measurement(monkeypatch):
    from maichart.form import rational_audio
    sr=8000
    signal=np.zeros(sr,dtype=np.float32)
    signal[1000:2000]=np.sin(np.arange(1000)*.3)*.1
    monkeypatch.setattr(rational_audio,'waveform',lambda *_:(signal,sr))
    audio=SimpleNamespace(source='virtual-native-fixture',duration_sec=1.,
        metadata={'normalization_scale':[1.]*95},cache_dir=None)
    seconds=np.asarray([.1,.126,.2,.3,.8])
    measured=native_dynamics(audio,seconds)
    reference=rational_audio.query_spectra(signal,sr,seconds)
    np.testing.assert_allclose(measured[:,:3],reference[:,92:95],atol=1e-6,rtol=1e-4)
    assert measured[1,0]>measured[-1,0]


def test_multiscale_context_is_candidate_independent_and_bounded():
    times=np.linspace(0.,10.,1001)
    values=np.ones((1001,120),dtype=np.float32)
    bank=AcousticContext(values,times,10.)
    first=bank.at(np.asarray([2.,3.]))
    second=bank.at(np.asarray([1.,2.,3.,9.]))
    assert first.shape==(2,104)
    np.testing.assert_allclose(first,second[1:3])
    np.testing.assert_allclose(first[0],first[1])


def test_multiscale_context_deduplicates_clipped_tail_indices():
    # A duration longer than the final acoustic node used to append an
    # out-of-range search index, clip it to the last row, and leave duplicate
    # timestamps in the integration grid.
    times=np.asarray([0.,1.],dtype=np.float64)
    values=np.ones((2,120),dtype=np.float32)
    bank=AcousticContext(values,times,3.)
    assert np.all(np.diff(bank.times)>0)
    result=bank.at(np.asarray([0.,1.,2.9]))
    assert result.shape==(3,104)
    assert np.isfinite(result).all()


def test_curve_gradient_and_conditional_slots_share_train_inference_contract():
    torch.manual_seed(5)
    model=ChartComposer(ComposerConfig(atom_vocab_size=5,d_model=16,nhead=4,layers=1,
        dim_feedforward=32,action_slots=2,dropout=0.)).eval()
    context=torch.randn(2,16)
    context[1]=context[0]
    labels=torch.tensor([[1,0],[2,0]])
    decision=model.decode_candidate(context,action_targets=labels)
    torch.testing.assert_close(decision['action_logits'][0,0],decision['action_logits'][1,0])
    assert not torch.allclose(decision['action_logits'][0,1],decision['action_logits'][1,1])
    torch.testing.assert_close(decision['action_logits'][0,1],model.action_slot(context[0],1,[1]))
    features=torch.randn(2,4,141)
    audio=torch.randn(2,6,104)
    output=model(features,None,torch.zeros(2,16),candidate_indices=torch.tensor([1,2]),
        workload_audio=audio,ledger=torch.zeros(2,4,6),action_targets=labels)
    batch={'candidate_target':torch.tensor([1,2]),'action_target':labels,
        'eos_target':torch.zeros(2),'future_target':torch.zeros(2,8),
        'workload_target':torch.zeros(2,6,5),'workload_weight':torch.ones(2,6),
        'workload_seconds':torch.arange(6)[None].expand(2,-1).float()}
    loss,metrics=_loss(output,batch,5)
    assert metrics['workload_loss']>0
    loss.backward()
    assert model.workload_head[-1].weight.grad.abs().sum()>0
    empty=dict(batch,sample_weight=torch.zeros(2))
    zero,_=_loss(output,empty,5)
    assert zero==0


def test_guard_rejects_new_overlap_without_mutating_prefix_and_allows_release():
    tempo=[EventTempo(Fraction(0),120.)]
    guard=EventGuard(tempo,0.,8.)
    point=EventPoint(Fraction(0),'1h[2:1]/2h[2:1]')
    assert guard.accepts(point)
    guard.commit(point)
    before=list(guard.prefix)
    assert not guard.accepts(EventPoint(Fraction(1),'3'))
    assert guard.prefix==before
    assert guard.accepts(EventPoint(Fraction(3),'3'))
    assert not guard.accepts(EventPoint(Fraction(15),'4h[1:2]'))
