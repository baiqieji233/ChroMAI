import numpy as np
import pytest
import torch

from maichart.composer.data import collate_composer
from maichart.composer.density_grid import survival_grid_inputs
from maichart.composer.event_views import EventViews


class Context:
    def __init__(self):
        self.calls = 0

    def at(self, seconds):
        self.calls += 1
        return np.repeat(np.asarray(seconds)[:, None], 104, axis=1)


def test_fixed_grid_reuses_audio_and_keeps_exact_cursor_and_partial_tail():
    context = Context()
    first = survival_grid_inputs(context, .073, [.11, .123], .123)
    second = survival_grid_inputs(context, .01, [.09, .12], .123)
    assert context.calls == 1
    assert first['survival_edges'] == pytest.approx([.05, .1, .123])
    assert first['survival_widths'] == pytest.approx([.05, .023])
    assert first['survival_start_seconds'] == .073
    assert first['survival_audio'][:, 0] == pytest.approx([.075, .1115])
    first['survival_audio'][:] = -1  # Returned windows cannot corrupt the cache.
    assert (second['survival_audio'] >= 0).all()


def test_counts_are_half_open_and_unknown_cells_never_become_empty_labels():
    views = EventViews((), (), np.empty(0), np.empty(0), np.empty(0),
                       ((.06, .08),), (), .123, True, 'test')
    values = survival_grid_inputs(Context(), 0., [.123], .123, views=views,
                                  point_seconds=[0., .05, .1, .123])
    assert values['survival_count_target'].tolist() == [1., 1., 1.]
    assert values['survival_count_weight'].tolist() == [1., 0., 1.]
    recovery = survival_grid_inputs(Context(), .05, [.123], .123)
    assert not recovery['survival_count_target'].any()
    assert not recovery['survival_count_weight'].any()


def _item(values):
    item = {key: torch.from_numpy(value) for key, value in values.items()}
    item.update(candidate_features=torch.zeros(1, 2), candidate_target=torch.tensor(0),
                action_target=torch.zeros(1, dtype=torch.long), action_state=torch.zeros(1),
                future_target=torch.zeros(8), condition=torch.zeros(2),
                history=torch.zeros(0, 1, dtype=torch.long), history_features=torch.zeros(0, 8),
                candidate_mask=torch.zeros(1, dtype=torch.bool), song_index=torch.tensor(0),
                event_index=torch.tensor(0), eos_target=torch.tensor(0.))
    return item


def test_collation_pads_with_zero_exposure_and_finite_repeated_edges():
    context = Context()
    short = _item(survival_grid_inputs(context, 0., [.04], .123))
    long = _item(survival_grid_inputs(context, 0., [.123], .123))
    batch = collate_composer([short, long])
    assert batch['survival_audio'].shape == (2, 3, 104)
    assert batch['survival_edges'][0].tolist() == pytest.approx([0., .05, .05, .05])
    assert batch['survival_widths'][0].tolist() == pytest.approx([.05, 0., 0.])
    assert not batch['survival_count_weight'][0, 1:].any()
    assert not batch['survival_audio'][0, 1:].any()
