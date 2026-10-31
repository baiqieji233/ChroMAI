"""Native geometry and bounded soft assignment, not hard hand rules."""
from fractions import Fraction as F
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock
from maichart.composer.stream_quality import _motion_actions, _trajectory_comfort, PhraseQualityScorer


TEMPO=[EventTempo(F(0),200.)]
CLOCK=TempoClock(TEMPO,0.)


def points(*items):return [EventPoint(F(beat),raw) for beat,raw in items]


def test_native_slide_chain_and_headless_track_are_real_continuous_jobs():
    for raw in ('1-5[4:4]','1-3-5[4:4]','1?-5[4:4]','1w5[4:4]'):
        jobs,pulses,coverage=_motion_actions(points((0,raw)),TEMPO,CLOCK)
        assert jobs and all(job['segments'] for job in jobs)
        assert all(job['end']>job['trace']>=job['start'] for job in jobs)
        assert not coverage['unknown']
        if '?' in raw:assert not pulses
    jobs,_,_=_motion_actions(points((0,'1-3-5[4:4]')),TEMPO,CLOCK)
    assert len(jobs[0]['segments'])==2


def test_center_hold_uses_real_geometry_and_reduces_remaining_capacity():
    taps=points((1,'1'),(F(5,4),'5'),(F(3,2),'1'),(F(7,4),'5'))
    free,_=_trajectory_comfort(taps,TEMPO,CLOCK,0.,1.,1.)
    occupied,_=_trajectory_comfort(points((0,'Ch[4:4]'))+taps,TEMPO,CLOCK,0.,1.,1.)
    assert occupied<free


def test_headless_slide_occupies_path_and_prior_slide_survives_phrase_boundary():
    taps=points((3,'3'),(F(13,4),'7'),(F(7,2),'3'),(F(15,4),'7'))
    free,_=_trajectory_comfort(taps,TEMPO,CLOCK,.8,1.3,1.3)
    busy,coverage=_trajectory_comfort(points((0,'1?-5[4:4]'))+taps,TEMPO,CLOCK,.8,1.3,1.3)
    assert busy<free
    assert coverage['active_jobs']>=1 and coverage['complete']


def test_handoff_only_reads_already_observed_lookahead():
    source=points((0,'Ch[4:4]'),(2,'1'),(F(21,8),'5'))
    short,a=_trajectory_comfort(source,TEMPO,CLOCK,.4,.75,.75)
    extended,b=_trajectory_comfort(source,TEMPO,CLOCK,.4,.75,.8)
    assert a['lookahead_sec']==0 and b['lookahead_sec']>0
    assert extended<short


def test_neighbor_touch_contacts_group_geometrically_without_two_note_limit():
    _,near=_trajectory_comfort(points((0,'A1/B1')),TEMPO,CLOCK,0.,.3,.3)
    _,far=_trajectory_comfort(points((0,'A1/A5')),TEMPO,CLOCK,0.,.3,.3)
    assert near['touch_contact_groups']==1 and far['touch_contact_groups']==2


def test_unknown_native_path_is_reported_instead_of_assuming_exact_motion(monkeypatch):
    from maichart.composer import stream_quality
    marker=dict(type='slide',raw='fixture',second=0.,lane=1,has_head_star=False,
        slide_trace_second=.3,end_second=1.,slide_segment_keys=['missing'],
        slide_segment_shoot_seconds=[.3],slide_segment_durations=[.7])
    monkeypatch.setattr(stream_quality,'_native_markers',lambda *args:([marker],[]))
    score,coverage=_trajectory_comfort(points((0,'1-5[4:4]')),TEMPO,CLOCK,0.,1.,1.)
    assert np.isfinite(score) and not coverage['complete']
    assert coverage['unknown_count']==1 and 'native_path_missing' in coverage['unknown'][0]['reason']


def scorer(active=True):
    seconds=torch.arange(0.,4.,.025,dtype=torch.float64)
    values=torch.zeros(len(seconds),238)
    if active:
        values[:,104+92]=1.;values[:,104+93]=1.;values[:,104+94]=1.
    bank=(values,seconds,seconds*(200./60.))
    song=SimpleNamespace(tempo=TEMPO,first=0.,audio=SimpleNamespace(duration_sec=4.))
    condition=SimpleNamespace(level=13.,radar=dict(note_density=.3,chord=.25))
    return PhraseQualityScorer(song,condition,audio_bank=bank)


def test_coverage_does_not_saturate_at_55_percent_and_sparse_is_not_free_win():
    quality=scorer()
    sparse=points((1,'1'),(6,'5'))
    adequate=points(*[(F(i,2),str(i%8+1)) for i in range(1,16)])
    empty=quality.score_phrase([],[],0,8,softcomfort_strength=0.)
    a=quality.score_phrase(sparse,[],0,8,softcomfort_strength=0.)
    b=quality.score_phrase(adequate,[],0,8,softcomfort_strength=0.)
    assert a['components']['audio_coverage']<b['components']['audio_coverage']
    assert a['components']['audio_coverage']<1.
    assert b['total']>a['total']>empty['total']
    assert b['components']['density_range']>a['components']['density_range']


def test_true_musical_silence_still_prefers_rest_over_added_noise():
    quality=scorer(active=False)
    empty=quality.score_phrase([],[],0,8,softcomfort_strength=0.)
    noisy=quality.score_phrase(points(*[(i,str(i%8+1)) for i in range(8)]),[],0,8,softcomfort_strength=0.)
    assert empty['total']>noisy['total']


def test_immutable_point_motion_cache_reuses_paths_and_edits_get_new_entries(monkeypatch):
    from maichart.composer import stream_quality
    native=stream_quality._native_markers;calls=[]
    def read(text,bpm):calls.append(text);return native(text,bpm)
    monkeypatch.setattr(stream_quality,'_native_markers',read)
    cache=OrderedDict();source=points((0,'1-5[4:4]'),(2,'Ch[4:1]'))
    first=_motion_actions(source,TEMPO,CLOCK,cache=cache)
    second=_motion_actions(source,TEMPO,CLOCK,cache=cache)
    assert len(calls)==2 and first==second
    saved=repr(cache)
    _trajectory_comfort(source,TEMPO,CLOCK,.5,1.,1.,motion_cache=cache)
    assert repr(cache)==saved
    _motion_actions(points((0,'1-5[4:2]'))+source[1:],TEMPO,CLOCK,cache=cache)
    assert len(calls)==3 and len(cache)==3
