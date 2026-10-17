from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.stream import audio_bounds
from maichart.stream.train import _loss
from maichart.form.codec import RationalCodec


def test_quiet_nonzero_audio_is_never_trimmed_as_digital_silence():
    values=np.r_[np.ones(800,dtype=np.float32),np.full(200,1e-12,dtype=np.float32)]
    assert audio_bounds.exact_signal_end(values,1000) == 1.
    values[-100:]=0
    assert audio_bounds.exact_signal_end(values,1000) == .9
    assert audio_bounds.exact_signal_end(np.zeros(1000),1000) == 0.


def test_explicit_end_is_an_audio_second_and_zero_trim_can_be_disabled(monkeypatch):
    audio=SimpleNamespace(duration_sec=1.,source='unused',metadata={'audio_sha256':'x'},cache_dir='unused')
    values=np.r_[np.ones(800),np.zeros(200)]
    monkeypatch.setattr(audio_bounds,'mapped_waveform',lambda *_:(values,1000))
    assert audio_bounds.generation_bounds(audio)['effective_end_sec'] == .8
    assert audio_bounds.generation_bounds(audio,trim_exact_silence=False)['effective_end_sec'] == 1.
    assert audio_bounds.generation_bounds(audio,requested_end=.7)['effective_end_sec'] == .7
    with pytest.raises(ValueError,match='生成结束时间'):
        audio_bounds.generation_bounds(audio,requested_end=2.)


def test_event_decision_objective_only_adds_supervision_at_real_boundaries():
    codec=RationalCodec()
    logits=torch.zeros(1,3,len(codec),requires_grad=True)
    output={'logits':logits,'plan':torch.zeros(1,3,8),'curve':torch.zeros(1,1,4)}
    batch={'decoder_input':torch.tensor([[codec.bos_id,codec.ids['N:1'],codec.end_id]]),
        'target':torch.tensor([[codec.at_id,codec.end_id,codec.eos_id]]),
        'target_padding_mask':torch.zeros(1,3,dtype=torch.bool),
        'plan_mask':torch.zeros(1,3,dtype=torch.bool),'plan_mask_padding_mask':torch.zeros(1,3,dtype=torch.bool),
        'plan_labels':torch.zeros(1,3,8),'curve_target':torch.zeros(1,1,4),
        'curve_target_padding_mask':torch.zeros(1,1,dtype=torch.bool),'audio_padding_mask':torch.zeros(1,1,dtype=torch.bool)}
    plain,_=_loss(output,batch,event_weight=0.)
    enhanced,metrics=_loss(output,batch,event_weight=.25)
    delta=enhanced-plain
    assert float(delta) == pytest.approx(.25*np.log(2))
    gradients=torch.autograd.grad(delta,logits)[0]
    assert gradients[0,1].abs().sum() == 0
    assert gradients[0,0,codec.at_id] < 0 and gradients[0,2,codec.eos_id] < 0
