from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.composer import stream_augmentation as aug
from maichart.composer.stream_data import StreamDataset,NOTE,WAIT,EOS,collate_stream
from maichart.composer.events import event_metadata
from maichart.event_timing import EventPoint,EventTempo
from test_stream_timeline_curriculum import TimelineRandom,capture_prefix


class PreferExit(TimelineRandom):
    def random(self):return 0.


class PositiveControl(TimelineRandom):
    def random(self):return .99


def exit_dataset(monkeypatch,change,*,sustained_previous=False):
    from test_stream_augmentation import dataset
    dataset(monkeypatch)
    count=602 if change in ('rest','end') else 700
    duration=75.5 if change=='end' else 100.
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=duration),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[]
    for i in range(count):
        beat=F(i,4)+(F(1,4) if change=='gap' and i>=602 else 0)
        text='1/5' if change=='template' and i==602 else str(i%8+1)
        if sustained_previous and 599<=i<=601:text='2h[16:1]'
        points.append(EventPoint(beat,text))
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=128,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')


def augment(ds,row,rng):
    return aug.augment_stream_row(ds,150,rng=rng,mode='timeline_loop',row=row,
        input_policy=aug.MUSICAL_AUTHORED_EXIT_INPUT_POLICY)[0]


@pytest.mark.parametrize('change',['template','gap','rest','end'])
def test_v8_selects_later_authored_exit_and_matches_the_preceding_real_template(monkeypatch,change):
    ds=exit_dataset(monkeypatch,change);row=ds[150];row['_plan_scale']=.43
    captured=capture_prefix(monkeypatch)
    assert not aug._authored_timeline_exit(ds.streams[0],600)
    assert aug._authored_timeline_exit(ds.streams[0],602)
    changed=augment(ds,row,PreferExit(0))
    _,prefix,target=captured[-1]
    assert target is ds.streams[0][602]
    previous=ds.streams[0][601]
    assert event_metadata(prefix[-1].text).atoms==event_metadata(previous.text).atoms
    assert ds.codec.encode(prefix[-1].text)['phase']==ds.codec.encode(previous.text)['phase']
    assert prefix[-1].end-prefix[-1].start==previous.end-previous.start
    assert prefix[-1].end==target.start
    assert changed['_synthetic_loop_reference_aligned']
    assert changed['_timeline_authored_exit'] and not changed['_timeline_positive_control']
    assert changed['_timeline_actual_template_exit']==(change!='gap')
    assert changed['_timeline_actual_time_exit']==(change!='template')
    assert changed['target_kind'][-1]==target.kind
    assert changed['target_second'][-1]==target.end_second
    torch.testing.assert_close(changed['sample_weight'].sum(),row['sample_weight'].sum(),rtol=0,atol=0)
    before=collate_stream([row]);after=collate_stream([changed])
    for key in ('audio_plan_weight','audio_plan_blank_weight','audio_plan_target'):
        torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_v8_first_target_control_and_v7_still_choose_first_trusted_target(monkeypatch):
    ds=exit_dataset(monkeypatch,'template');row=ds[150];captured=capture_prefix(monkeypatch)
    changed=augment(ds,row,PositiveControl(0))
    assert captured[-1][-1] is ds.streams[0][600]
    assert changed['_timeline_positive_control'] and not changed['_timeline_authored_exit']
    assert not changed['_synthetic_loop_reference_aligned']
    old=aug.augment_stream_row(ds,150,rng=PreferExit(0),mode='timeline_loop',row=row,
        input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)[0]
    assert captured[-1][-1] is ds.streams[0][600]
    assert '_timeline_authored_exit' not in old


def test_v8_without_a_true_exit_falls_back_to_the_first_trusted_target(monkeypatch):
    ds=exit_dataset(monkeypatch,'none');row=ds[150];captured=capture_prefix(monkeypatch)
    changed=augment(ds,row,PreferExit(0))
    assert captured[-1][-1] is ds.streams[0][600]
    assert not changed['_timeline_authored_exit'] and not changed['_synthetic_loop_reference_aligned']


def test_sustained_previous_template_uses_ordinary_counter_fallback(monkeypatch):
    ds=exit_dataset(monkeypatch,'template',sustained_previous=True);row=ds[150]
    captured=capture_prefix(monkeypatch)
    changed=augment(ds,row,PreferExit(0))
    assert captured[-1][-1] is ds.streams[0][602]
    assert changed['_timeline_authored_exit'] and not changed['_synthetic_loop_reference_aligned']
    assert changed['sample_weight'].sum()==row['sample_weight'].sum()


def test_aligned_loop_does_not_copy_the_future_exit_configuration_into_inputs(monkeypatch):
    ds=exit_dataset(monkeypatch,'template');row=ds[150];capture_prefix(monkeypatch)
    before=augment(ds,row,PreferExit(0))
    ds.streams[0][602]=replace(ds.streams[0][602],text='3/7')
    after=augment(ds,row,PreferExit(0))
    for key in ('history_atoms','history_shape','history_phase','step_beat','history_gap_beats','continuation_features'):
        torch.testing.assert_close(before[key],after[key])
    assert not torch.equal(before['target_phase_mask'],after['target_phase_mask'])


def test_v8_keeps_the_existing_nine_modes():
    assert aug._augmentation_modes(aug.MUSICAL_AUTHORED_EXIT_INPUT_POLICY)==aug._augmentation_modes(aug.MUSICAL_BATCH_INPUT_POLICY)
