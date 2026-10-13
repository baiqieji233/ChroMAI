from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.form import generate
from maichart.form.onset_selection import local_peak_mask, merge_close_onsets, interval_statistics
from maichart.generation.conditions import GenerationCondition


def records(times, scores):
    return [{'second':time,'score':score,'beat':str(F(str(time))*2)} for time,score in zip(times,scores)]


def test_merging_prefers_strongest_without_chaining_a_whole_fast_passage():
    source = records([0,.025,.07,.1], [.8,.9,.7,.85])
    kept = merge_close_onsets(source,40)
    assert [r['second'] for r in kept] == [.025,.1]
    assert interval_statistics(kept)['minimum_interval_ms'] == pytest.approx(75)
    assert merge_close_onsets(source,0) == source
    assert [r['second'] for r in merge_close_onsets(records([0,.035,.07],[1,1,1]),40)] == [0,.07]
    assert [r['second'] for r in merge_close_onsets(records([0,.04],[1,1]),40)] == [0,.04]


def test_triplet_subdivisions_and_tempo_changes_use_seconds_not_fixed_beat_distance():
    tempo = [EventTempo(F(0),180),EventTempo(F(1),240)]
    from maichart.event_timing import seconds_at_beat
    beats = [F(0),F(1,6),F(1,3),F(1),F(7,6),F(4,3)]
    candidates = [{'beat':str(b),'second':seconds_at_beat(b,tempo,0),'score':.9} for b in beats]
    assert len(merge_close_onsets(candidates,40)) == len(beats)
    assert len(merge_close_onsets(candidates,60)) < len(beats)


def test_equal_peak_plateau_yields_one_candidate_and_inputs_must_be_finite():
    assert local_peak_mask([.1,.9,.9,.9,.1]).tolist() == [False,True,False,False,False]
    assert not local_peak_mask([]).any()
    with pytest.raises(ValueError,match='finite'):
        local_peak_mask([.1,float('nan')])
    for bad in (-1,float('nan'),251):
        with pytest.raises(ValueError,match='onset_merge_ms'):
            merge_close_onsets([],bad)


def mocked_plan(monkeypatch, scores, *, curve=None, merge_ms=40, first=0):
    times = np.arange(len(scores),dtype=np.float64)*.02
    song = SimpleNamespace(spans=[SimpleNamespace(start=F(0),end=F(1))],
                           tempo=[EventTempo(F(0),120)],first=first,audio=SimpleNamespace(duration_sec=1))
    monkeypatch.setattr(generate,'form_audio',lambda *_:torch.zeros(len(scores),120))
    monkeypatch.setattr(generate,'input_example',lambda *_,**__: {'form':torch.zeros(1,507)})
    monkeypatch.setattr(generate,'segment_frame_times',lambda *_:times)
    from maichart.generation import intensity
    monkeypatch.setattr(intensity,'music_activity',lambda _:SimpleNamespace(times=times,
                                               active=lambda:np.ones(len(scores),dtype=bool)))
    probabilities = torch.tensor(scores,dtype=torch.float32)
    def planner(*_):
        return {'onset_logits':torch.logit(probabilities)[None],
                'type_logits':torch.zeros(1,len(scores),4),'sustain_seconds':torch.ones(1,len(scores))}
    return generate.plan_onsets(planner,song,GenerationCondition(),threshold=.5,
                                intensity_curve=curve,onset_merge_ms=merge_ms)


def test_curve_amplification_cannot_create_peaks_on_both_sides_of_one_onset(monkeypatch):
    curve = {'points':[{'time_sec':0,'intensity':.8},{'time_sec':1,'intensity':.8}]}
    plan = mocked_plan(monkeypatch,[.1,.8,.9,.8,.1],curve=curve,merge_ms=0)
    assert len(plan['onsets']) == 1
    assert plan['onsets'][0]['source_second'] == pytest.approx(.04)
    assert plan['selection']['raw_local_peaks'] == 1
    assert plan['onsets'][0]['raw_score'] == pytest.approx(.9)


def test_curve_can_promote_a_weak_real_peak_and_rest_can_remove_it(monkeypatch):
    curve = {'points':[{'time_sec':0,'intensity':.8},{'time_sec':1,'intensity':.8}]}
    assert len(mocked_plan(monkeypatch,[.1,.4,.1],curve=curve)['onsets']) == 1
    curve['rest_regions'] = [{'start_sec':0,'end_sec':1}]
    assert not mocked_plan(monkeypatch,[.1,.9,.1],curve=curve)['onsets']


def test_plan_merges_after_rational_snapping_and_reports_removed_count(monkeypatch):
    off = mocked_plan(monkeypatch,[.1,.9,.1,.8,.1,.95,.1],merge_ms=0)
    merged = mocked_plan(monkeypatch,[.1,.9,.1,.8,.1,.95,.1],merge_ms=60)
    assert len(off['onsets']) == 3
    assert len(merged['onsets']) == 2
    assert merged['selection']['suppressed_close_onsets'] == 1
    assert merged['selection']['after_merge']['minimum_interval_ms'] >= 60-1e-6


def test_partial_note_at_token_budget_returns_the_same_points_and_count_contract():
    from maichart.form.codec import RationalCodec
    codec = RationalCodec()
    atom = '1-5[16:3]'
    class Model:
        config = SimpleNamespace(max_tokens=12)
        def eval(self): pass
        def encode(self,_): return torch.zeros(1,1,1),torch.zeros(1,1,dtype=torch.bool)
        def decode_step(self,*_):
            return torch.zeros(1,1,len(codec)),None
    example = {'action_state_padding_mask':torch.ones(1,dtype=torch.bool),'meta':{}}
    result = generate.arrange_span(Model(),codec,example,[F(0)],[atom],origin=F(0),temperature=0)
    assert isinstance(result,tuple) and len(result) == 2
    assert result == ([],0)


def test_revision_unpacks_points_and_fallback_count_before_validation(monkeypatch):
    from maichart.event_timing import EventPoint
    from maichart.form.data import organize_song
    from maichart.form.codec import RationalCodec
    from test_form_pipeline import fixture_audio
    song = organize_song(fixture_audio(3),[EventTempo(F(0),120)],0,max_seconds=3)
    original = [EventPoint(F(0),'1'),EventPoint(F(2),'3')]
    proposed = [EventPoint(F(0),'2')]
    monkeypatch.setattr(generate,'input_example',lambda *_a,**_kw:{})
    monkeypatch.setattr(generate,'arrange_span',lambda *_a,**_kw:(proposed,2))
    def accept(points,patch,*_a,**_kw):
        assert list(patch.points) == proposed
        return SimpleNamespace(points=[*patch.points,points[1]],accepted=True,reason='test',
                               checks=2,edits=1,before=None,after=None)
    monkeypatch.setattr(generate,'apply_patch_candidate',accept)
    points, report = generate.revise_chart(SimpleNamespace(config=SimpleNamespace(max_memory_tokens=64)),
        RationalCodec(),song,original,GenerationCondition(),['1','2','3'],start=F(0),end=F(1))
    assert points[0].note_text == '2' and report['decoder_fallbacks'] == 2
