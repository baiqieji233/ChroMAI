from dataclasses import replace
from fractions import Fraction as F

import pytest
import torch

from maichart.composer.stream_continuation import ContinuationState,continuation_rows,continuation_dim
from maichart.composer.stream_data import StreamStep,NOTE,WAIT,EOS,StreamInferenceContext,recovery_sample


def note(j,text,end=None):
    return StreamStep(NOTE,F(j),F(j+1) if end is None else F(end),j*.5,(j+1)*.5,text)


def test_policy_dimensions_and_legacy_prefix_are_stable():
    assert continuation_dim('off') == continuation_dim('typed_runs_v1') == 8
    assert continuation_dim('pattern_runs_v2') == 16
    old=ContinuationState();new=ContinuationState('pattern_runs_v2')
    for j,text in enumerate(['1/2','2/3','A1/C','E2','E2']):
        old.append(note(j,text));new.append(note(j,text))
        torch.testing.assert_close(old.features(),new.features()[:8])
    assert ContinuationState('off').features().shape == (8,)


def test_template_ignores_duration_but_exact_signature_keeps_it():
    state=ContinuationState('pattern_runs_v2')
    for j,text in enumerate(['1h[4:1]','2h[4:2]','3h[8:1]']):state.append(note(j,text))
    values=state.features().expm1()
    assert values[8] == pytest.approx(3.)
    assert values[10] == pytest.approx(1.)
    assert state.exit_label(note(3,'4h[4:1]')) == (True,False)
    assert state.exit_label(note(3,'4/5')) == (True,True)
    # Literal fallback does not turn all Touch geometry into one template.
    fallback=ContinuationState('pattern_runs_v2')
    fallback.append(note(0,'A1/C'));fallback.append(note(1,'B2/E3'))
    assert fallback.features()[8].expm1() == pytest.approx(1.)


def test_exact_atom_order_is_a_set_and_alternation_keeps_kind_detail():
    state=ContinuationState('pattern_runs_v2')
    state.append(note(0,'1/2'));state.append(note(1,'2/1'))
    assert state.features()[10].expm1() == pytest.approx(2.)
    state=ContinuationState('pattern_runs_v2')
    for j,text in enumerate(['1','8','1','8','1','8']):state.append(note(j,text))
    assert state.features()[12].expm1() == pytest.approx(6.)
    assert state.exit_label(note(6,'1')) == (True,False)
    assert state.exit_label(note(6,'2')) == (True,True)
    # Same button at the expected turn but a different authored kind breaks ABAB.
    assert state.exit_label(note(6,'1h[4:1]')) == (True,True)
    state.append(note(6,'1h[4:1]'))
    assert state.features()[12].expm1() == pytest.approx(2.)


def test_motion_counters_preserve_rotation_direction_and_wait_reset():
    state=ContinuationState('pattern_runs_v2')
    for j,text in enumerate(['1/3','2/4','3/5','4/6']):state.append(note(j,text))
    assert state.features()[14].expm1() == pytest.approx(3.)
    assert state.exit_label(note(4,'5/7')) == (True,False)
    assert state.exit_label(note(4,'3/5')) == (True,True)
    for kind in (WAIT,EOS):
        assert state.exit_label(StreamStep(kind,F(4),F(6),2.,3.)) == (True,True)
    state.append(StreamStep(WAIT,F(4),F(6),2.,3.))
    assert not state.features().any()
    assert state.exit_label(note(6,'1')) == (False,False)
    assert ContinuationState().exit_label(StreamStep(WAIT,F(4),F(6),2.,3.)) == (False,False)
    short=ContinuationState('pattern_runs_v2');short.append(note(0,'1'))
    assert short.exit_label(StreamStep(EOS,F(1),F(1),.5,.5)) == (False,False)
    assert state.exit_label(replace(note(6,'2'),weight=0.)) == (False,False)


def test_pattern_rows_are_causal_and_do_not_saturate_at_context_length():
    stream=[note(j,'1' if j%2==0 else '8') for j in range(80)]
    features,_,_=continuation_rows(stream,policy='pattern_runs_v2')
    assert not features[0].any()
    assert features[-1,12].expm1() == pytest.approx(79.)
    altered=stream[:10]+[note(10,'C')]+stream[11:]
    changed,_,_=continuation_rows(altered,policy='pattern_runs_v2')
    torch.testing.assert_close(features[:11],changed[:11])


def test_pattern_dataset_inference_recovery_dimensions_and_terminal_labels(monkeypatch):
    from test_stream_history_window import fixture_dataset
    ds,song,bank=fixture_dataset(monkeypatch,continuation_policy='pattern_runs_v2')
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],lead_in=ds.lead_in,
        rollout_steps=ds.targets,history_window_policy=ds.history_window_policy,continuation_policy=ds.continuation_policy,audio_bank=bank)
    for j,target in enumerate(ds.streams[0]):
        row=ds[j//ds.targets];n=j-int(row['step_index'][0])+1
        torch.testing.assert_close(context.current_batch()['continuation_features'][0],row['continuation_features'][:n])
        if target.kind != EOS:context.commit(target)
    j=12
    recovered=recovery_sample(ds,0,ds.streams[0][:j],ds.streams[0][j])
    assert recovered['continuation_features'].shape[-1] == 16
    assert recovery_sample(ds,0,ds.streams[0][:j],ds.streams[0][j],continuation_policy='typed_runs_v1')['continuation_features'].shape[-1] == 8
    prefix=ds.streams[0][:j];cursor=prefix[-1].end;sec=prefix[-1].end_second
    terminal=recovery_sample(ds,0,prefix,StreamStep(EOS,cursor,cursor,sec,sec))
    assert terminal['transition_exit_mask'][-1] and terminal['transition_exit_labels'][-1]
    old=recovery_sample(ds,0,prefix,StreamStep(EOS,cursor,cursor,sec,sec),continuation_policy='typed_runs_v1')
    assert not old['transition_exit_mask'].any()
    from maichart.composer.stream_data import collate_stream
    batch=collate_stream([ds[0],ds[1]])
    assert batch['continuation_features'].shape[-1] == 16
    assert batch['_sample_indices'] == [0,1]
