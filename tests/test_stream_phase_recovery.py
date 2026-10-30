"""A wrong committed cursor must teach the same reachable authored onset."""
from dataclasses import replace
from fractions import Fraction as F
import random
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.composer import stream_augmentation as aug, stream_data
from maichart.composer.stream_data import NOTE, WAIT, EOS, StreamDataset, StreamStep


class _PositiveFirst(random.Random):
    def shuffle(self, values):
        values.sort(key=lambda x:(abs(x), x<0))


class _NegativeFirst(random.Random):
    def shuffle(self, values):
        values.sort(key=lambda x:(abs(x), x>0))


def _dataset(monkeypatch, *, native=False):
    from maichart.generation.sustain_state import SustainStateIndex
    if not native:
        class EmptyState:
            def __init__(self,*args): self.points=[]
            def append(self,point): self.points.append(point)
            def at(self,*args,**kwargs):return torch.zeros(1,8),torch.ones(1,dtype=torch.bool),{}
        monkeypatch.setattr(stream_data,'STATE_DIM',8)
        monkeypatch.setattr(stream_data,'SustainStateIndex',EmptyState)
        monkeypatch.setattr(stream_data,'IncrementalState',EmptyState)
    bank=(torch.zeros(320,238),torch.arange(320,dtype=torch.float64)/20,
          torch.arange(320,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=16.),
        tempo=[EventTempo(F(0),120.),EventTempo(F(4),180.)],first=0.)
    points=([EventPoint(F(0),'8h[1:5]')] if native else [EventPoint(F(0),'1')])
    points += [EventPoint(F(i),str((i-1)%6+1)) for i in range(1,24)]
    state=SustainStateIndex(points,song.tempo,song.first) if native else None
    base=SimpleNamespace(songs=[(song,points,12.,1.,state,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=16,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
        timing_output_policy='absolute_grid_v1')


@pytest.mark.parametrize('rng_type',[_PositiveFirst,_NegativeFirst])
def test_phase_recovery_moves_real_cursor_and_keeps_all_authored_target_onsets(monkeypatch,rng_type):
    ds=_dataset(monkeypatch);index=3;original=ds[index]
    song,start,end,_=ds.samples[index];source=tuple(ds.streams[song])
    modified,shift=aug._phase_shift_steps(ds,song,source,start,end,rng_type())
    changed,mode,count=aug.augment_stream_row(ds,index,rng=rng_type(),mode='phase_shift',row=original,
        input_policy=aug.PHASE_INPUT_POLICY)
    assert mode=='phase_shift' and count==1 and abs(shift)==F(1,48)
    first=max(0,start-ds.lead_in);at=start-first
    assert modified[start-1].end==source[start-1].end+shift
    assert modified[start].start==modified[start-1].end
    assert modified[start].end==source[start].end
    assert changed['step_beat'][at]==float(modified[start].start)
    assert changed['step_second'][at]==modified[start].start_second
    assert changed['history_gap_beats'][at]==float(modified[start-1].end-modified[start-1].start)
    # An OOV categorical history code must never erase the physical gap.
    assert changed['history_delta'][at]==aug._encode_gap(ds,modified[start-1].end-modified[start-1].start)
    assert changed['target_delta'][at]==aug._encode_gap(ds,modified[start].end-modified[start].start)
    assert changed['history_gap_beats'][at]>0
    mask=original['loss_mask']
    for key in ('target_kind','target_beat','target_second','target_shape','target_phase_mask','target_details','loss_mask'):
        torch.testing.assert_close(changed[key][mask],original[key][mask],rtol=0,atol=0)
    torch.testing.assert_close(changed['target_delta'][at+1:],original['target_delta'][at+1:])
    torch.testing.assert_close(changed['sample_weight'].sum(),original['sample_weight'].sum(),rtol=0,atol=0)
    assert changed['sample_weight'][at]>original['sample_weight'][at]
    assert torch.all(changed['sample_weight'][mask]>0)
    assert tuple(ds.streams[song])==source
    torch.testing.assert_close(ds[index]['step_beat'],original['step_beat'])


def test_phase_recovery_rebuilds_seconds_from_the_piecewise_tempo_clock(monkeypatch):
    ds=_dataset(monkeypatch);index=2
    song,start,end,_=ds.samples[index];source=ds.streams[song]
    # The edited NOTE follows a BPM change. Beat jitter must use TempoClock's
    # full piecewise origin and scale, not the first or last BPM alone.
    modified,shift=aug._phase_shift_steps(ds,song,source,start,end,_PositiveFirst())
    row,_,_=aug.augment_stream_row(ds,index,rng=_PositiveFirst(),mode='phase_shift',
        input_policy=aug.PHASE_INPUT_POLICY)
    at=start-max(0,start-ds.lead_in)
    expected=stream_data.second_at_beat(ds.clocks[song],modified[start].start)
    assert abs(float(row['step_second'][at])-expected)<1e-12
    assert row['target_second'][at]==source[start].end_second
    assert modified[start].end-modified[start].start>0


@pytest.mark.parametrize('rng_type',[_PositiveFirst,_NegativeFirst])
def test_phase_local_rebuild_equals_full_replay_and_keeps_long_old_hold(monkeypatch,rng_type):
    ds=_dataset(monkeypatch,native=True);original=ds[4]
    local,_,_=aug.augment_stream_row(ds,4,rng=rng_type(),mode='phase_shift',row=original,
        input_policy=aug.PHASE_INPUT_POLICY)
    assert local['_augmentation_phase_local_rebuild']
    monkeypatch.setattr(aug,'_native_prefix',lambda *args,**kwargs:None)
    full,_,_=aug.augment_stream_row(ds,4,rng=rng_type(),mode='phase_shift',row=original,
        input_policy=aug.PHASE_INPUT_POLICY)
    assert not full['_augmentation_phase_local_rebuild']
    for key in ('history_kind','history_delta','history_gap_beats','history_atoms','history_shape','history_phase',
                'history_phase_mask','step_beat','step_second','target_beat','target_delta','target_second',
                'continuation_features','transition_exit_mask','transition_exit_labels','sample_weight'):
        torch.testing.assert_close(local[key],full[key],rtol=0,atol=0)
    width=max(local['action_state'].size(1),full['action_state'].size(1))
    for key in ('action_state','action_state_mask'):
        values=[]
        for row in (local,full):
            value=row[key];pad=width-value.size(1)
            values.append(torch.nn.functional.pad(value,(0,0,0,pad)) if key=='action_state'
                else torch.nn.functional.pad(value,(0,pad),value=True))
        torch.testing.assert_close(*values,rtol=1e-6,atol=1e-6)


def test_phase_recovery_does_not_retime_wait_or_eos_as_if_it_were_a_note(monkeypatch):
    ds=_dataset(monkeypatch);song,start,end,_=ds.samples[3]
    for kind in (WAIT,EOS):
        source=list(ds.streams[song]);source[start]=replace(source[start],kind=kind,text=None)
        with pytest.raises(ValueError,match='next NOTE target'):
            aug._phase_shift_steps(ds,song,source,start,end,_PositiveFirst())


def test_old_input_recipe_cannot_silently_enable_phase_recovery(monkeypatch):
    ds=_dataset(monkeypatch)
    with pytest.raises(ValueError,match='input policy'):
        aug.augment_stream_row(ds,3,rng=_PositiveFirst(),mode='phase_shift')


def test_phase_recovery_skips_an_unknown_first_decision_before_physical_work(monkeypatch):
    ds=_dataset(monkeypatch);row=dict(ds[3]);row['sample_weight']=row['sample_weight'].clone()
    first=int(row['loss_mask'].nonzero()[0]);row['sample_weight'][first]=0
    calls=[]
    monkeypatch.setattr(aug,'_validate_window',lambda *args:calls.append(1))
    with pytest.raises(ValueError,match='known supervised first'):
        aug.augment_stream_row(ds,3,rng=_PositiveFirst(),mode='phase_shift',row=row,
            input_policy=aug.PHASE_INPUT_POLICY)
    assert calls==[]


def test_phase_recovery_selection_is_seeded_without_modifying_original_rows(monkeypatch):
    ds=_dataset(monkeypatch);items=[ds[2],ds[3],ds[4]]
    originals=[row['step_beat'].clone() for row in items]
    a,da=aug.augment_stream_items(ds,items,rate=1,seed=59,input_policy=aug.PHASE_INPUT_POLICY)
    b,db=aug.augment_stream_items(ds,items,rate=1,seed=59,input_policy=aug.PHASE_INPUT_POLICY)
    # Wall-clock timing is diagnostic only and not deterministic.
    for diagnostics in (da,db):
        diagnostics.pop('timing_full_replay_sec',None)
        diagnostics.pop('phase_shift_rebuild_sec',None)
    assert da==db
    for x,y,original,row in zip(a,b,originals,items):
        for key in x:
            if torch.is_tensor(x[key]):torch.testing.assert_close(x[key],y[key])
        torch.testing.assert_close(row['step_beat'],original)
