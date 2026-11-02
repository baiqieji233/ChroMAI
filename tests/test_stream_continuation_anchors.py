from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import pytest
import torch
from maichart.composer.stream_continuation import ContinuationState,continuation_rows
from maichart.composer.stream_data import StreamStep,NOTE,WAIT,EOS
from maichart.composer.stream_recovery import _risk_exits


@pytest.mark.parametrize('policy',['off','typed_runs_v1','pattern_runs_v2'])
@pytest.mark.parametrize('masked',[False,True])
def test_collected_exit_anchors_match_legacy_traversal_and_labels(policy,masked,monkeypatch):
    texts=['1']*5+['2','3','4','5']+['A1']*4+['2','3']*4+['1/2','2/3','3/4','4/5','5/6']
    streams=[]
    for song in range(2):
        steps=[StreamStep(NOTE,F(i),F(i+1),i/2,(i+1)/2,text) for i,text in enumerate(texts)]
        steps.insert(9,StreamStep(WAIT,F(9),F(10),4.5,5.,''))
        steps.append(StreamStep(EOS,F(30),F(30),15.,15.,''))
        if masked:steps=[replace(step,weight=0.) if i%7==0 else step for i,step in enumerate(steps)]
        streams.append(steps)
    unknown=[[masked and i%5==0 for i in range(len(stream))] for stream in streams]
    dataset=SimpleNamespace(continuation_policy=policy,streams=streams,unknown_steps=unknown)
    legacy=_risk_exits(dataset)
    categories={name:[] for name in ('gap','type','template','exact','period2','motion','touch','button')}
    for song,stream in enumerate(streams):
        baseline=continuation_rows(stream,policy)
        collected=continuation_rows(stream,policy,exit_anchors=categories,unknown_steps=unknown[song],song_index=song)
        for a,b in zip(baseline,collected):assert torch.equal(a,b)
    assert categories==legacy
    assert list(categories)==list(legacy)
    assert all(values==sorted(values) for values in categories.values())
    dataset._mixed_recovery_exit_anchors=((policy,len(streams)),categories)
    monkeypatch.setattr(ContinuationState,'append',lambda *_:pytest.fail('cached anchors traversed the corpus'))
    assert _risk_exits(dataset) is categories
