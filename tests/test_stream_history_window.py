from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.composer.stream_data import StreamDataset, StreamInferenceContext, WAIT, EOS, recovery_sample


def fixture_dataset(monkeypatch, *, policy='block_aligned_v2',continuation_policy='off'):
    # Dense opening, silence with WAITs, then notes crossing an eight-second
    # window. K=5 deliberately does not divide any window's event count.
    points = [EventPoint(F(i,4),str(i%8+1)) for i in range(25)]
    points += [EventPoint(F(35,2),'2'),EventPoint(F(18),'3'),EventPoint(F(43,2),'4')]
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),
                           tempo=[EventTempo(F(0),120.)],first=0.)
    base = SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    bank = (torch.zeros(24,238),torch.arange(24,dtype=torch.float64)/2,
            torch.arange(24,dtype=torch.float64))
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    return StreamDataset(base=base,lead_in=3,targets=5,history_window_policy=policy,
                         continuation_policy=continuation_policy),song,bank


def test_blocks_cover_targets_and_preserve_time_window_masses(monkeypatch):
    ds,_,_ = fixture_dataset(monkeypatch)
    stream = ds.streams[0]
    seen=[]; masses=defaultdict(float); plan_mass=defaultdict(float)
    for i,(_,start,end,_) in enumerate(ds.samples):
        assert start % ds.targets == 0
        row=ds[i]
        assert row['step_index'][0] == max(0,start-ds.lead_in)
        for absolute in range(start,end):
            seen.append(absolute)
            window=int(stream[absolute].start_second//ds.window_seconds)
            index=absolute-int(row['step_index'][0])
            masses[window]+=float(row['sample_weight'][index])/stream[absolute].weight
            plan_mass[window]+=ds.target_window_weights[0][absolute]/ds.targets
        assert ds.plan_weights[i] == pytest.approx(sum(ds.target_window_weights[0][start:end])/ds.targets)
    assert seen == list(range(len(stream)))
    assert all(v == pytest.approx(ds.targets) for v in masses.values())
    assert all(v == pytest.approx(1.) for v in plan_mass.values())
    assert sum(ds.plan_weights) == pytest.approx(len(masses))


def test_every_generated_prefix_matches_actual_teacher_inputs_across_waits(monkeypatch):
    ds,song,bank=fixture_dataset(monkeypatch,continuation_policy='typed_runs_v1')
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],
        lead_in=ds.lead_in,rollout_steps=ds.targets,history_window_policy=ds.history_window_policy,audio_bank=bank,
        continuation_policy=ds.continuation_policy)
    stream=ds.streams[0]
    assert any(s.kind == WAIT for s in stream)
    assert len(stream)>3*ds.targets
    keys=('history_kind','history_delta','history_shape','history_phase','history_phase_mask',
          'step_beat','step_second','step_index','continuation_features')
    for j,target in enumerate(stream):
        block=j//ds.targets
        teacher=ds[block]
        generated=context.current_batch()
        rebuilt=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],
            lead_in=ds.lead_in,rollout_steps=ds.targets,history_window_policy=ds.history_window_policy,
            audio_bank=bank,continuation_policy=ds.continuation_policy).batch(stream[:j],target.start)
        torch.testing.assert_close(generated['continuation_features'],rebuilt['continuation_features'])
        first=max(0,block*ds.targets-ds.lead_in)
        length=j-first+1
        assert generated['history_kind'].shape[1] == length
        for key in keys:
            torch.testing.assert_close(generated[key][0],teacher[key][:length])
        # Atom and sustain widths are padded independently; compare semantic
        # rows after removing padding, rather than only the current decision.
        for key in ('history_atoms',):
            width=max(generated[key].shape[-1],teacher[key].shape[-1])
            a=torch.nn.functional.pad(generated[key][0],(0,width-generated[key].shape[-1]))
            b=torch.nn.functional.pad(teacher[key][:length],(0,width-teacher[key].shape[-1]))
            torch.testing.assert_close(a,b)
        for k in range(length):
            a=generated['action_state'][0,k][~generated['action_state_mask'][0,k]]
            b=teacher['action_state'][k][~teacher['action_state_mask'][k]]
            torch.testing.assert_close(a,b)
        if target.kind != EOS:
            context.commit(target)


def test_continuation_statistics_use_true_types_members_and_full_prefix():
    from maichart.composer.stream_continuation import ContinuationState,typed_event,continuation_rows
    from maichart.composer.stream_data import StreamStep,NOTE
    assert typed_event('1/A1h[4:1]')[0] == frozenset(('tap','touch_hold'))
    assert typed_event('1/2')[1] == frozenset((1,2))
    assert typed_event('A1/C')[2]
    assert not typed_event('A1/1')[2]
    state=ContinuationState();stream=[]
    for j in range(30):
        step=StreamStep(NOTE,F(j),F(j+1),j*.5,(j+1)*.5,'1/2' if j%2 else '2/3')
        stream.append(step);state.append(step)
    v=state.features().expm1()
    assert v[0] == pytest.approx(30.)
    assert v[1] == pytest.approx(30.)
    assert v[2] == pytest.approx(30.)  # Both configurations are actual taps.
    assert v[6] == pytest.approx(30.)  # Button 2 survives changing chords.
    assert v[7] == pytest.approx(14.5)
    rows,_,_=continuation_rows(stream)
    assert rows[-1,0].expm1() == pytest.approx(29.)  # Current target is not an input.
    state.append(StreamStep(WAIT,F(30),F(32),15.,16.))
    assert not state.features().any()


def test_transition_exit_supervision_and_recovery_are_causal(monkeypatch):
    from maichart.composer.stream_continuation import ContinuationState
    from maichart.composer.stream_data import StreamStep,NOTE
    state=ContinuationState()
    for j in range(4):state.append(StreamStep(NOTE,F(j),F(j+1),j*.5,(j+1)*.5,'A1'))
    assert state.exit_label(StreamStep(NOTE,F(4),F(5),2.,2.5,'A2')) == (True,False)
    assert state.exit_label(StreamStep(NOTE,F(4),F(9,2),2.,2.25,'A2')) == (True,True)
    assert state.exit_label(StreamStep(NOTE,F(4),F(5),2.,2.5,'1')) == (True,True)
    ds,_,_=fixture_dataset(monkeypatch,continuation_policy='typed_runs_v1')
    j=12;prefix=ds.streams[0][:j]
    target=ds.streams[0][j]
    # A recovered target need not be found at its original stream position.
    target=replace(target,end=target.end+F(1,4),end_second=target.end_second+.125)
    row=recovery_sample(ds,0,prefix,target)
    assert row['transition_exit_mask'][-1]
    assert row['transition_exit_labels'][-1]
    assert row['continuation_features'][-1,0].expm1() == pytest.approx(11.)


def test_button_exit_labels_follow_authored_members_without_forcing_an_exit():
    from maichart.composer.stream_continuation import ContinuationState
    from maichart.composer.stream_data import StreamStep,NOTE
    def step(j,text,kind=NOTE):
        return StreamStep(kind,F(j),F(j+1),j*.5,(j+1)*.5,text)
    state=ContinuationState()
    state.append(step(0,'1'));state.append(step(1,'1'))
    assert state.exit_label(step(2,'2')) == (True,True)
    assert state.exit_label(step(2,'1')) == (True,False)
    assert state.exit_label(step(2,'1/3')) == (True,False)
    # A new chord partner does not end button 1's repeated run.
    state.append(step(2,'1/3'))
    assert state.exit_label(step(3,'1/4')) == (True,False)
    state=ContinuationState()
    state.append(step(0,'1/2'));state.append(step(1,'1/2'))
    assert state.exit_label(step(2,'1/3')) == (True,True)  # Only button 2 exits.
    assert state.exit_label(step(2,'2/3')) == (True,True)  # Only button 1 exits.
    assert state.exit_label(step(2,'1/2')) == (True,False)
    state.append(step(2,'',WAIT))
    assert state.exit_label(step(3,'3')) == (False,False)


def test_legacy_generation_keeps_rolling_history(monkeypatch):
    ds,song,bank=fixture_dataset(monkeypatch,policy='rolling_legacy_v1')
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],lead_in=3,audio_bank=bank)
    for step in ds.streams[0]:
        if step.kind == EOS: break
        context.commit(step)
        batch=context.current_batch()
        assert batch['history_kind'].shape[1] == min(len(context._committed)+1,4)


def test_block_weights_keep_chart_quality_and_unknown_masks(monkeypatch):
    ds,_,_=fixture_dataset(monkeypatch)
    ds.chart_weights[0]=.4
    stream=ds.streams[0]
    stream[2]=replace(stream[2],weight=.25)
    ds.unknown_steps[0][3]=True
    row=ds[0]
    assert row['sample_weight'][2] == pytest.approx(ds.target_window_weights[0][2]*.25*.4)
    assert row['sample_weight'][3] == 0
    assert row['sample_weight'][4] == pytest.approx(ds.target_window_weights[0][4]*stream[4].weight*.4)
    assert row['_window_weight'] == pytest.approx(ds.plan_weights[0]*.4)


def test_recovery_context_uses_same_block_origin(monkeypatch):
    ds,_,_=fixture_dataset(monkeypatch)
    j=2*ds.targets+2
    prefix=ds.streams[0][:j]
    row=recovery_sample(ds,0,prefix,ds.streams[0][j])
    assert row['step_index'].tolist() == list(range(2*ds.targets-ds.lead_in,j+1))
    assert row['loss_mask'].sum() == 1
    assert row['_plan_scale'] == 0


def test_multilayer_logits_match_each_streaming_prefix_in_every_block(monkeypatch):
    from maichart.composer.stream_data import collate_stream
    from maichart.composer.stream_model import StreamComposer,StreamComposerConfig
    torch.manual_seed(27)
    ds,song,bank=fixture_dataset(monkeypatch,continuation_policy='typed_runs_v1')
    model=StreamComposer(StreamComposerConfig(d_model=16,nhead=4,layers=2,dim_feedforward=32,
        dropout=0.,history_events=ds.lead_in,rollout_steps=ds.targets,continuation_policy=ds.continuation_policy,
        delta_beats=('0',*map(str,ds.delta_vocab.values))),ds.codec).eval()
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],audio_bank=bank,
        lead_in=ds.lead_in,rollout_steps=ds.targets,history_window_policy='block_aligned_v2',continuation_policy='typed_runs_v1')
    with torch.no_grad():
        for j,target in enumerate(ds.streams[0]):
            row=ds[j//ds.targets];teacher=collate_stream([row]);trained=model(teacher)
            streamed=context.current_batch();length=streamed['step_index'].shape[1]
            # The selected note time conditions the action stage, after timing.
            streamed['target_second']=teacher['target_second'][:,:length]
            actual=model(streamed)
            for key in ('hidden','kind_logits','delta_logits','shape_logits'):
                torch.testing.assert_close(actual[key][0,-1],trained[key][0,length-1],atol=2e-5,rtol=1e-5)
            if target.kind!=EOS:context.commit(target)


def test_recovery_selects_real_touch_exit_without_inserting_future(monkeypatch):
    import random
    from maichart.composer import stream_recovery
    ds,_,_=fixture_dataset(monkeypatch,continuation_policy='typed_runs_v1')
    ds._stream_exit_anchors=([(0,4)],[(0,8)])
    monkeypatch.setattr(stream_recovery,'_anchors',lambda dataset:([],[],[]))
    episodes=[stream_recovery._new_episode(ds,random.Random(i),0) for i in range(20)]
    assert {episode['risk'] for episode in episodes}=={'touch_exit','rhythm_exit'}
    assert all(not episode['generated'] and episode['anchor'] in (4,8) for episode in episodes)
