"""Boundary-plan alignment and unsaturated causal clocks over long songs."""
from fractions import Fraction as F
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.codec import RationalCodec
from maichart.generation.conditions import GenerationCondition
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream import data
from maichart.stream.grounding import clock_inputs, masked_clock_inputs
from maichart.stream.model import ChartStreamTransformer, StreamConfig


@pytest.fixture
def source():
    beats=[F(0),F(129,2),F(65),F(1025,2),F(513),F(4099,4)]
    points=[EventPoint(beat,str(1+i%8)) for i,beat in enumerate(beats)]
    song=SimpleNamespace(first=0.,tempo=[EventTempo(F(0),120.)],meters=[],
        spans=[SimpleNamespace(start=F(0),end=F(1026))],
        quality_profile='musical_evidence_v1',curve_bin_seconds=1.,
        audio=SimpleNamespace(duration_sec=514.),
        load_curve_times_sec=np.array([.5,256.,513.5]),
        load_curve=np.zeros((3,4),dtype=np.float32))
    return song,points


def test_dataset_plan_rows_align_with_bos_end_and_model_intent_boundaries(monkeypatch,tmp_path,source):
    song,points=source
    record=(song,points,13.,1.,SustainStateIndex(points,song.tempo,0.),GenerationCondition(),[],None)
    monkeypatch.setattr(data,'FormDataset',lambda *_args,**_kwargs:SimpleNamespace(
        songs=[record],audit={'charts':[{'audio_sha256':'boundary-fixture'}]}))
    dataset=data.StreamDataset('unused',cache_dir=tmp_path,max_tokens=36,input_cache_mode='legacy')
    codec=dataset.codec
    assert len(dataset.chunks)>1
    model=ChartStreamTransformer(StreamConfig(len(codec),d_model=16,nhead=2,layers=1,
        dim_feedforward=32,dropout=0.,max_tokens=36,grounding_profile='causal_audio_v1',
        detail_audio_nodes=256,local_audio_neighbors=8,archive_events=64,archive_retrievals=4)).eval()
    captured=[]
    hook=model.grounding.intent_projection.register_forward_pre_hook(
        lambda _module,args:captured.append(args[0].detach().clone()))
    try:
        for chunk in dataset.chunks:
            decoder=torch.tensor(chunk.decoder_input)[None]
            target=torch.tensor(chunk.target)
            at_rows=target==codec.at_id
            boundaries=(decoder[0]==codec.bos_id)|(decoder[0]==codec.end_id)
            assert torch.equal(torch.tensor(chunk.plan_mask),at_rows)
            assert boundaries[at_rows].all()
            # END also predicts the final EOS, which has no next-event plan label.
            assert torch.equal(boundaries,at_rows|(target==codec.eos_id))
            previous=points[chunk.start_event-1].beat if chunk.start_event else F(0)
            before=points[chunk.start_event-2].beat if chunk.start_event>1 else F(0)
            rows=clock_inputs(chunk.decoder_input,codec,song,previous=previous,
                previous_previous=before,committed=chunk.start_event)
            length=decoder.size(1)
            context={key:torch.zeros(1,length,16) for key in ('local','archive','action')}
            context['global']=torch.zeros(1,16)
            with torch.inference_mode():
                result=model.decode(decoder,torch.zeros(1,2,16),torch.zeros(1,2,dtype=torch.bool),
                    timing_features=rows['decoder_clock'][None],grounding_context=context)
            active=0
            expected=[]
            for row,is_boundary in enumerate(boundaries.tolist()):
                if is_boundary:active=row
                expected.append(result['plan'][0,active])
            torch.testing.assert_close(captured[-1][0],torch.stack(expected))
    finally:
        hook.remove()


@pytest.mark.parametrize('dropout',[0.,1.])
def test_long_beat_clock_preserves_actual_position_through_masking(source,dropout):
    song,points=source
    codec=RationalCodec();ids=codec.encode(points);decoder=ids[:-1];target=ids[1:]
    rows=clock_inputs(decoder,codec,song)
    # Check every consumed token: a new position becomes visible at NOTE,
    # while its preceding AT/digits continue to expose the prior position.
    current=F(0);event=0;expected=[]
    for token in decoder:
        if token==codec.note_id:
            current=points[event].beat;event+=1
        expected.append(float(current))
    expected=torch.tensor(expected)
    torch.testing.assert_close(rows['decoder_clock'][:,0]*64.,expected,rtol=0,atol=1e-5)
    assert expected.max()>512. and (expected>64.).any()
    query_expected=torch.tensor([float(beat) for beat in rows['query_beats']])
    torch.testing.assert_close(rows['query_clock'][:,0]*64.,query_expected,rtol=0,atol=1e-5)
    masked,queries=masked_clock_inputs(rows['decoder_clock'],decoder,codec,random.Random(0),
        hidden_events=[1,3,5],dropout=dropout,query_clock=rows['query_clock'],
        query_index=rows['decoder_query_index'],targets=target)
    torch.testing.assert_close(masked[:,0]*64.,expected,rtol=0,atol=1e-5)
    torch.testing.assert_close(queries[:,0]*64.,query_expected,rtol=0,atol=1e-5)
    assert masked[:,15:17].abs().sum()<rows['decoder_clock'][:,15:17].abs().sum()
