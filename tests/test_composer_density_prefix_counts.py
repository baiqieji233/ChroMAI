import numpy as np
import pytest

from maichart.composer.density_grid import survival_grid_inputs


class Context:
    def at(self, seconds):
        return np.zeros((len(seconds), 104), dtype=np.float32)


def test_prefix_counts_use_only_committed_events_and_past_half_open_windows():
    # At center 5.025: the 1s left edge is included, center itself excluded.
    # 1.025 is outside 1s but belongs to the 4s window.
    prefix = [1.025, 4.025, 5., 5.025, 5.5]
    values = survival_grid_inputs(Context(), 5.5, [5.8], 6., prefix_seconds=prefix)
    centers = (values['survival_edges'][:-1] + values['survival_edges'][1:]) * .5
    row = np.argmin(abs(centers - 5.025))
    center = centers[row]
    exact_prefix = [center - 4., center - 1., 5., center, 5.5]
    values = survival_grid_inputs(Context(), 5.5, [5.8], 6., prefix_seconds=exact_prefix)
    assert values['survival_prefix_counts'][row].tolist() == [2., 3.]
    # Neither future authored labels nor caller-provided post-cursor events
    # can add an input count after the true committed cursor.
    base = survival_grid_inputs(Context(), 5., [5.8], 6., prefix_seconds=[4.9],
                                point_seconds=[4.9, 5.1, 5.4])
    changed = survival_grid_inputs(Context(), 5., [5.8], 6., prefix_seconds=[4.9, 5.1, 5.4],
                                   point_seconds=[4.9, 5.2, 5.7])
    np.testing.assert_array_equal(base['survival_prefix_counts'], changed['survival_prefix_counts'])


def test_empty_prefix_has_zero_counts_and_four_second_lookback():
    values = survival_grid_inputs(Context(), 5.073, [5.2], 6., prefix_seconds=[])
    assert values['survival_edges'][0] == pytest.approx(1.05)
    assert values['survival_start_seconds'] == 5.073
    assert values['survival_candidate_seconds'].tolist() == [5.2]
    assert values['survival_prefix_counts'].shape == (len(values['survival_audio']), 2)
    assert not values['survival_prefix_counts'].any()
    assert not values['survival_count_target'].any()
    assert not values['survival_count_weight'].any()


def test_feedback_off_preserves_original_window_and_key_contract():
    values = survival_grid_inputs(Context(), 5.073, [5.2], 6.)
    assert values['survival_edges'][0] == pytest.approx(5.05)
    assert 'survival_prefix_counts' not in values
