from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer.acoustics import AcousticContext
from maichart.composer.event_views import EventViews
from maichart.composer.quality_inputs import (
    lifecycle_features, phrase_audio_bank, quality_context_inputs,
    quality_targets, select_history_anchors,
)
from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.grounded_inference import IncrementalState
from maichart.tempo_clock import TempoClock


TEMPO = [EventTempo(F(0), 120.)]


def point(beat, text='1'):
    return EventPoint(F(beat), text)


def audio(duration=12.):
    times = np.linspace(0., duration, int(duration * 20) + 1)
    values = np.repeat(times[:, None], 120, axis=1).astype(np.float32)
    return values, times, AcousticContext(values, times, duration)


def test_native_lifecycle_distinguishes_explicit_wait_and_exact_landing():
    delayed = SustainStateIndex([point(0, '1-5[8:1]')], TEMPO, 0.)
    immediate = SustainStateIndex([point(0, '1-5[0##1]')], TEMPO, 0.)
    delayed_marker = delayed.markers[0]
    result = lifecycle_features(delayed, [.25, delayed_marker['end_second']])
    assert result[0, 8] == pytest.approx(np.log(2.))  # Waiting at head 1.
    assert result[0, 83] == 0.  # Not tracing yet.
    assert result[1, 32 + 4] == pytest.approx(np.log(2.))  # True lane-5 landing.
    immediate_result = lifecycle_features(immediate, [.25])
    assert immediate_result[0, 8] == 0.
    assert immediate_result[0, 83] == pytest.approx(np.log(2.))
    assert immediate_result[0, 16:24].sum() > 0.


def test_future_authored_sustains_never_enter_lifecycle_and_old_holds_survive():
    prefix = [point(0, '1h[4:40]')]
    full = SustainStateIndex(prefix + [point(2, '3h[4:20]')], TEMPO, 0.)
    partial = IncrementalState(TEMPO, 0.)
    partial.append(prefix[0])
    expected = lifecycle_features(partial, [1., 2., 8.])
    actual = lifecycle_features(full, [1., 2., 8.], visible_before_beat=F(2))
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual[:, 40] > 0.)
    assert not actual[:, 42].any()


def test_lifecycle_aggregates_beyond_old_sustain_state_cap():
    # A synthetic marker population directly exercises aggregation capacity.
    state = SimpleNamespace(markers=[dict(type='hold', lane=1, second=0., end_second=8.,
                                         source_beat=F(0)) for _ in range(40)])
    result = lifecycle_features(state, [1.], visible_before_beat=F(1))
    assert result[0, 40] == pytest.approx(np.log(41.))
    assert result[0, 85] == pytest.approx(np.log(41.))


def test_segmented_native_slide_and_wifi_path_inputs_remain_finite():
    state = SustainStateIndex([point(0, '1-3-5[4:4]/2w6[4:4]')], TEMPO, 0.)
    result = lifecycle_features(state, [.25, .75, 1.25, 2.5])
    assert result.shape == (4, 88)
    assert np.isfinite(result).all()
    # Wifi terminal coverage is retained separately from the path position.
    assert np.count_nonzero(result[1, 24:32]) >= 3


def test_sparse_anchors_pair_true_audio_and_ids_at_committed_event_times():
    values, times, context = audio()
    clock = TempoClock(TEMPO, 0.)
    points = [point(i) for i in range(20)]
    ids = np.column_stack((np.arange(1, 21), np.zeros(20, dtype=int)))
    selected = select_history_anchors(np.arange(20))
    assert len(selected) == 8
    assert selected[-4:].tolist() == [16, 17, 18, 19]
    result = quality_context_inputs(points=points, atom_ids=ids, candidate_beats=[20., 21.],
        candidate_seconds=[10., 10.5], state_index=None, audio_features=values, audio_times=times,
        acoustic_context=context, clock=clock, audio_duration=12.)
    np.testing.assert_allclose(result['quality_anchor_audio'][:, 0], selected * .5)
    np.testing.assert_array_equal(result['quality_anchor_ids'][:, 0], selected + 1)
    np.testing.assert_array_equal(result['quality_anchor_ages'], 19 - selected)
    np.testing.assert_array_equal(result['quality_candidate_offsets'], [1., 2.])
    assert result['quality_anchor_mask'].all()


def test_empty_prefix_has_no_symbolic_anchors_and_audio_phrase_bank_is_cached():
    values, times, context = audio()
    clock = TempoClock(TEMPO, 0.)
    kwargs = dict(points=[], atom_ids=np.zeros((0, 3), dtype=np.int64), candidate_beats=[0., 4., 8.],
        candidate_seconds=[0., 2., 4.], state_index=None, audio_features=values, audio_times=times,
        acoustic_context=context, clock=clock, audio_duration=12.)
    result = quality_context_inputs(**kwargs)
    assert not result['quality_anchor_mask'].any()
    assert not result['quality_anchor_ids'].any()
    assert result['quality_phrase_indices'].tolist() == [0, 1, 2]
    assert result['quality_candidate_offsets'].tolist() == [0., 4., 8.]
    assert result['quality_phrase_audio'].data_ptr() == quality_context_inputs(**kwargs)['quality_phrase_audio'].data_ptr()


def test_phrase_labels_are_real_statistics_and_unknown_gap_is_not_supervised():
    points = [point(0, '1/5'), point(1, '2h[4:1]'), point(2, 'C'), point(4, '3')]
    clock = TempoClock(TEMPO, 0.)
    _, _, context = audio(6.)
    views = EventViews.build(points, TEMPO, 0., 6., excluded=[(.7, .8)])
    labels = quality_targets(points, clock, phrase_audio_bank(context, clock, 6.), views)
    first = labels['phrase_target'][0]
    assert first[0] == pytest.approx(3 / 4 / 8)
    assert first[1] == pytest.approx((4 / 3) / 4)
    assert first[2] == pytest.approx(1 / 3)
    assert first[3] == pytest.approx(1 / 3)
    assert first[5] == pytest.approx(1 / 3)
    assert labels['phrase_weight'][0] == 0.
    assert labels['gap_target'].tolist() == [3, 3, 4, 0]
    assert labels['gap_weight'].tolist() == [1., 0., 1., 0.]


def test_recovery_anchors_describe_generated_prefix_and_mask_authored_future():
    from maichart.composer.candidates import MUSIC_CANDIDATE_POLICY_V6
    from maichart.composer.recovery import RecoveryBuilder
    source = [point(i, str(i % 4 + 1)) for i in range(12)]
    views = EventViews.build(source, TEMPO, 0., 8.)
    values, times, _ = audio(8.)
    song = SimpleNamespace(tempo=TEMPO, first=0., audio=SimpleNamespace(duration_sec=8.))
    table = {'clock': TempoClock(TEMPO, 0.), 'condition': torch.zeros(16), 'audio_duration': 8.}
    dataset = SimpleNamespace(samples=[SimpleNamespace(song_index=0, event_index=1)],
        base=SimpleNamespace(songs=[(song,)]), event_views={0: views}, candidate_count=96,
        candidate_policy=MUSIC_CANDIDATE_POLICY_V6, architecture_profile='audio_grounded_v3',
        history_events=8, action_slots=3, codec_atoms=['1', '2', '3', '4'],
        atom_to_id={str(i): i for i in range(1, 5)}, quality_inputs_enabled=True,
        _song_table=lambda index: table, _song_features=lambda index: (values, times))
    item, _ = RecoveryBuilder(dataset)._inputs([point(0, '1'), point(1, '4'), point(2, '4')], 0)
    assert item['quality_anchor_ids'][:3, 0].tolist() == [1, 4, 4]
    assert not item['quality_phrase_weight'].any()
    assert item['quality_gap_weight'] == 0.
    from maichart.composer.data import collate_composer
    shorter = dict(item)
    for key in ('quality_phrase_audio', 'quality_phrase_target', 'quality_phrase_weight'):
        shorter[key] = item[key][:-1]
    shorter['quality_phrase_indices'] = item['quality_phrase_indices'].clamp_max(len(shorter['quality_phrase_audio']) - 1)
    batch = collate_composer([item, shorter])
    assert batch['quality_anchor_mask'].dtype == torch.bool
    assert batch['quality_anchor_ids'].dtype == batch['quality_phrase_indices'].dtype == torch.long
    assert batch['quality_phrase_weight'][1, -1] == 0.
    assert not batch['quality_phrase_audio'][1, -1].any()
    assert batch['quality_candidate_offsets'].shape == (2, 96)


def test_candidate_anchor_clock_preserves_far_offsets_and_fractional_phase():
    values, times, context = audio(48.)
    clock = TempoClock(TEMPO, 0.)
    result = quality_context_inputs(points=[point(1), point(2)], atom_ids=np.ones((2, 1), dtype=np.int64),
        candidate_beats=[F(17, 8), F(257, 8), F(513, 8)],
        candidate_seconds=[1.0625, 16.0625, 32.0625], state_index=None,
        audio_features=values, audio_times=times, acoustic_context=context, clock=clock, audio_duration=48.)
    np.testing.assert_array_equal(result['quality_candidate_offsets'], [.125, 30.125, 62.125])
