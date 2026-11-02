"""Speculative modes are local; committed metadata alone survives replay."""
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import torch

from maichart.composer.stream_data import StreamStep
from maichart.composer.stream_phrase_runtime import execution_options,execution_metadata,execution_report


class Model:
    config=SimpleNamespace(phrase_execution_policy='persistent_modes_v1',phrase_modes=4)
    calls=0

    def execution_prior(self,output,batch):
        self.calls+=1
        return torch.tensor([[[0.,0.,8.,0.]]])

    def execution_proposal(self,action,output,batch,mode):
        return dict(execution_lane_bias=torch.full((8,),float(mode)))


def batch(start):
    return dict(phrase_start_beat=torch.tensor([[float(start)]]),
                candidate_phrase_start_beat=torch.tensor([[float(start)]]),step_beat=torch.tensor([[0.]]))


def test_retries_share_mode_and_replay_uses_only_committed_metadata():
    model=Model();context=SimpleNamespace(_committed=[]);generator=torch.Generator().manual_seed(1)
    for beat in (1,2):
        execution_options(model,context,{},batch(0),torch.zeros(3),generator,temperature=0,beat=beat)
    assert model.calls==1
    step=StreamStep(2,Fraction(0),Fraction(1),0.,1.,'1',**execution_metadata(context,1))
    replay=SimpleNamespace(_committed=[step])
    execution_options(model,replay,{},batch(0),torch.zeros(3),generator,temperature=0,beat=2)
    assert model.calls==1
    assert execution_metadata(replay,2)==dict(phrase_mode=2,phrase_mode_start=0.)


def test_failed_future_candidate_does_not_survive_commit_or_truncation():
    model=Model();context=SimpleNamespace(_committed=[]);generator=torch.Generator().manual_seed(1)
    execution_options(model,context,{},batch(8),torch.zeros(3),generator,temperature=0,beat=9)
    assert model.calls==1
    context._committed.append(StreamStep(2,Fraction(0),Fraction(1),0.,1.,'1',phrase_mode=1,phrase_mode_start=0.))
    execution_options(model,context,{},batch(8),torch.zeros(3),generator,temperature=0,beat=9)
    assert model.calls==2
    truncated=SimpleNamespace(_committed=[])
    execution_options(model,truncated,{},batch(0),torch.zeros(3),generator,temperature=0,beat=1)
    assert model.calls==3


def test_report_counts_committed_choices_once_per_audio_phrase():
    steps=[SimpleNamespace(phrase_mode=2,phrase_mode_start=0.),
           SimpleNamespace(phrase_mode=2,phrase_mode_start=0.),
           SimpleNamespace(phrase_mode=1,phrase_mode_start=8.)]
    result=execution_report(steps)
    assert result['phrase_choices']==2
    assert result['mode_note_counts']=={2:2,1:1}


def test_reset_old_context_clears_pending_metadata():
    from maichart.composer import stream_data
    context=SimpleNamespace(_execution_runtime={'pending':{8.:2}},
                            phrase_context_policy='off',continuation_policy='off',
                            song=SimpleNamespace(tempo=[],first=0.))
    with patch.object(stream_data,'IncrementalState',return_value=object()):
        stream_data.StreamInferenceContext._reset_prefix(context)
    assert context._execution_runtime is None
    assert execution_metadata(context,9)=={}
    assert context._committed==[]


def test_candidate_landing_does_not_change_departure_mode_group():
    model=Model();context=SimpleNamespace(_committed=[]);generator=torch.Generator().manual_seed(1)
    fields=batch(0)
    fields['candidate_phrase_start_beat']=torch.tensor([[8.]])
    execution_options(model,context,{},fields,torch.zeros(3),generator,temperature=0,beat=9)
    assert execution_metadata(context,9)['phrase_mode_start']==0.
    assert model.calls==1
