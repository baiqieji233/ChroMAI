"""Semantic and memory checks for performance-only follow-up changes."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from maichart.composer.audio_lens import PlacementTargets, PLACEMENT_TOLERANCE_SEC
from maichart.composer.metric_totals import MetricTotals
from maichart.composer.train import summarize_metrics
from maichart.event_timing import EventPoint, EventTempo


def test_vector_metrics_keep_accumulation_order_weights_and_window_isolation():
    whole, recent, old = MetricTotals(), MetricTotals(), {}
    for window_index in range(3):
        window = MetricTotals()
        for batch_index in range(4):
            metrics = {'loss': torch.tensor(1.2 + batch_index / 7, requires_grad=True),
                       'action_nonstop_ce_sum': torch.tensor(float(batch_index + 2)),
                       'action_nonstop_ce_weight': torch.tensor(float(batch_index + 1))}
            window.add(metrics)
        if window_index == 1:
            window.add({'loss': torch.tensor(float('nan'))})
            continue  # discarded optimizer window must never enter reports
        whole.add(window); recent.add(window)
        for key, value in window.as_floats().items():
            old[key] = old.get(key, 0.) + value
        before = whole.as_floats()
        window.add({'loss': torch.tensor(100.)})
        assert whole.as_floats() == before
    assert whole.values.grad_fn is None
    assert summarize_metrics(whole, 8) == summarize_metrics(old, 8)
    assert whole.as_floats() == recent.as_floats()
    assert summarize_metrics(MetricTotals(), 0) == {}


def test_metric_schema_reorder_and_optional_fields_preserve_named_values():
    totals = MetricTotals()
    totals.add({'x': torch.tensor(1.), 'y': torch.tensor(2.)})
    totals.add({'y': torch.tensor(3.), 'x': torch.tensor(4.), 'z': torch.tensor(5.)})
    assert totals.as_floats() == {'x': 5., 'y': 5., 'z': 5.}


@pytest.mark.parametrize('sigma', [.001, .010, .07])
def test_typewise_nearest_gaussian_matches_bruteforce_overlapping_events(sigma):
    points = [EventPoint(Fraction(i, 1000), text) for i, text in
              [(0, '1'), (9, '2h[4:1]'), (15, 'C'), (24, '3b'), (31, '4-5[4:1]'),
               (70, 'A2/3'), (99, '6'), (105, 'Ch[4:1]')]]
    targets = PlacementTargets(points, [EventTempo(Fraction(0), 60.)], 0., .15)
    seconds = np.concatenate((np.linspace(-.1, .25, 251),
                              targets.times + PLACEMENT_TOLERANCE_SEC,
                              targets.times - PLACEMENT_TOLERANCE_SEC))
    expected = np.zeros((len(seconds), 5), dtype=np.float32)
    for row, second in enumerate(seconds):
        if not 0. <= second < .15:
            continue
        selected = ((targets.times >= second - .025 - 1e-9) &
                    (targets.times <= second + .025 + 1e-9))
        if selected.any():
            response = np.exp(-.5 * ((second - targets.times[selected]) / sigma) ** 2).astype(np.float32)
            expected[row] = (response[:, None] * targets.flags[selected]).max(axis=0)
    np.testing.assert_array_equal(targets.proximity(seconds, sigma_sec=sigma), expected)
    labels, weights, proximity = targets.query_with_proximity(seconds, sigma_sec=sigma)
    reference_labels, reference_weights = targets.query(seconds)
    np.testing.assert_array_equal(proximity, expected)
    np.testing.assert_array_equal(labels, reference_labels)
    np.testing.assert_array_equal(weights, reference_weights)


def test_device_bank_budget_includes_precomputed_storage_without_view_double_counting():
    from test_composer_device_audio import _context
    from maichart.composer.device_audio import _DeviceBank, AudioQueryMaterializer
    context = _context(np.linspace(0., 1., 101))
    bank = _DeviceBank(context, 'cpu')
    expected = {}
    for value in vars(bank).values():
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            expected[storage.data_ptr()] = storage.nbytes()
    assert bank.nbytes == sum(expected.values())
    assert bank._lens_diffs.numel() > 0 and bank._half_slopes.numel() > 0
    legacy_size = sum(getattr(bank, key).numel() * getattr(bank, key).element_size()
                      for key in ('times', 'data', 'prefix', 'slopes', 'chroma', 'global_mean', 'lens_times', 'lens_values'))
    assert bank.nbytes > legacy_size
    cache = AudioQueryMaterializer(budget_bytes=legacy_size)
    cache._get(context, 'cpu')
    assert cache.resident_bytes == 0 and not cache.cache


@pytest.mark.parametrize('lengths', [(0, 0), (2, 4)])
def test_history_pretrim_matches_postpadding_with_internal_holes(lengths, monkeypatch):
    from maichart.composer.data import collate_composer
    from torch.nn.utils.rnn import pad_sequence
    items = []
    for row, length in enumerate(lengths):
        history = torch.zeros(length, 33, dtype=torch.long)
        if length:
            history[:, 0] = 1; history[-1, 4] = 2
        item = {'candidate_features': torch.zeros(3, 8), 'candidate_target': torch.tensor(0),
                'action_target': torch.tensor([1, 0]), 'action_state': torch.zeros(50),
                'future_target': torch.zeros(8), 'condition': torch.zeros(16),
                'history': history, 'history_features': torch.zeros(length, 8),
                'candidate_mask': torch.zeros(3, dtype=torch.bool), 'song_index': torch.tensor(row),
                'event_index': torch.tensor(length), 'eos_target': torch.tensor(0.), '_audio_grounded': True}
        items.append(item)
    old = collate_composer(items)
    for item in items:
        occupied = item['history'].ne(0).any(0).nonzero()
        item['_history_atom_width'] = int(occupied[-1]) + 1 if len(occupied) else 1
    widths = []
    def record(values, *args, **kwargs):
        if values[0].ndim == 2 and values[0].dtype == torch.long:
            widths.extend(value.size(-1) for value in values)
        return pad_sequence(values, *args, **kwargs)
    monkeypatch.setattr(torch.nn.utils.rnn, 'pad_sequence', record)
    new = collate_composer(items)
    for key in old:
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    assert max(widths) == (5 if any(lengths) else 1)
