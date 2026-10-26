import torch

from maichart.composer.action_factors import CompositionalActions
from maichart.composer.capability_metrics import CapabilityMetrics, atom_kind_labels


torch.set_num_threads(1)


def _actions():
    # Includes every kind, a mixed atom, and a touch_hold atom.
    return CompositionalActions(
        ["1", "1h[4:1]", "1-4[4:1]", "C1", "Ch[4:1]", "1/Ch[4:1]"], 4)


def _logits(targets, vocab, *, scale=0.0):
    out = torch.full((*targets.shape, vocab), scale, dtype=torch.float32)
    for row in range(targets.shape[0]):
        for slot in range(targets.shape[1]):
            out[row, slot, int(targets[row, slot])] += 3.0
    return out


def test_parser_backed_kinds_include_touch_hold_and_mixed():
    labels = atom_kind_labels(_actions())
    assert labels == ("STOP", "tap", "hold", "slide", "touch", "touch", "mixed")


def test_metrics_are_batch_partition_invariant_and_report_weighted_slot0():
    actions = _actions()
    targets = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [2, 0, 0, 0]])
    logits = _logits(targets, 7)
    batch = {"action_target": targets,
             "sample_weight": torch.tensor([1., 2., 0.5]),
             "transition_weight": torch.tensor([1., 2., 1.]),
             "action_weight": torch.tensor([1., 0.5, 1.])}
    whole = CapabilityMetrics(actions)
    whole.update(logits, batch)
    split = CapabilityMetrics(actions)
    for i in range(3):
        split.update(logits[i:i + 1], {k: v[i:i + 1] for k, v in batch.items()})
    a, b = whole.report(), split.report()
    for group in ("slot0", "later_nonstop", "first_stop", "all_rows"):
        assert a[group]["row_count"] == b[group]["row_count"]
        assert a[group]["weight_sum"] == b[group]["weight_sum"]
        for metric in ("action_loss", "action_ce", "exact_atom_ce", "exact_atom_top1", "type_ce", "type_top1"):
            if a[group].get(metric) is None:
                assert b[group].get(metric) is None
            else:
                assert abs(a[group][metric] - b[group][metric]) < 1e-6
    assert a["slot0"]["weight_sum"] == 3.5
    assert a["slot0"]["type_probability_by_target"]["hold"]["row_count"] == 1


def test_zero_weight_and_absent_classes_return_null_metrics():
    actions = _actions()
    targets = torch.tensor([[1, 0, 0]])
    metric = CapabilityMetrics(actions)
    metric.update(_logits(targets, 7), {"action_target": targets, "sample_weight": torch.zeros(1)})
    report = metric.report()
    assert report["slot0"]["row_count"] == 1
    assert report["slot0"]["weight_sum"] == 0.0
    assert report["slot0"]["exact_atom_ce"] is None
    assert report["later_nonstop"]["action_ce"] is None
    assert report["slot0"]["type_probability_by_target"]["hold"] is None


def test_first_stop_and_padding_are_counted_once_and_padding_is_ignored():
    actions = _actions()
    targets = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])
    logits = _logits(targets, 7)
    # Arbitrarily bad predictions on masked tail padding must not affect metrics.
    logits[0, 2:, 1] = 1e4
    logits[1, 1:, 6] = 1e4
    metric = CapabilityMetrics(actions)
    metric.update(logits, {"action_target": targets})
    report = metric.report()
    assert report["slot0"]["row_count"] == 2
    assert report["later_nonstop"]["row_count"] == 0
    assert report["later_nonstop"]["token_count"] == 0
    assert report["first_stop"]["row_count"] == 2
    assert report["first_stop"]["token_count"] == 2
    assert [slot["row_count"] for slot in report["slots"]] == [2, 1, 0, 0]


def test_full_vocabulary_logits_are_numerically_stable():
    actions = _actions()
    targets = torch.tensor([[6, 0]])
    logits = torch.full((1, 2, 7), -10000.0)
    logits[0, 0, 6] = 10000.0
    logits[0, 1, 0] = 10000.0
    metric = CapabilityMetrics(actions)
    metric.update(logits, {"action_target": targets})
    report = metric.report()
    assert report["slot0"]["exact_atom_ce"] == 0.0
    assert report["slot0"]["type_ce"] == 0.0
    assert report["slot0"]["exact_atom_top1"] == 1.0
    assert report["slot0"]["type_probability_by_target"]["mixed"]["probabilities"]["mixed"] == 1.0


def test_nonfinite_logits_and_weights_are_rejected():
    actions = _actions()
    targets = torch.tensor([[1, 0]])
    metric = CapabilityMetrics(actions)
    bad = _logits(targets, 7)
    bad[0, 0, 0] = float("nan")
    try:
        metric.update(bad, {"action_target": targets})
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite logits must be rejected")
    try:
        metric.update(_logits(targets, 7), {"action_target": targets,
                                            "action_weight": torch.tensor([float("inf")])})
    except ValueError as exc:
        assert "action_weight" in str(exc)
    else:
        raise AssertionError("non-finite weights must be rejected")


def test_unequal_quality_weights_use_global_denominators_instead_of_batch_means():
    import pytest
    from torch.nn import functional as F

    actions = _actions()
    targets = torch.tensor([[1, 0], [2, 0], [2, 0]])
    logits = _logits(targets, 7)
    logits[1, 0, 1] = 9.  # wrong second row has three times the weight
    batch = {'action_target': targets, 'sample_weight': torch.tensor([1., 3., 0.])}
    whole, partitioned = CapabilityMetrics(actions), CapabilityMetrics(actions)
    whole.update(logits, batch)
    for index in range(3):
        partitioned.update(logits[index:index+1], {k: v[index:index+1] for k, v in batch.items()})
    a, b = whole.report(), partitioned.report()
    assert a['slot0']['exact_atom_top1'] == b['slot0']['exact_atom_top1'] == .25
    ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').reshape(3, 2)
    expected = (((ce[:, 0] + .35 * ce[:, 1]) / 1.35) * batch['sample_weight']).sum() / 4
    assert a['all_rows']['action_loss'] == pytest.approx(float(expected), abs=1e-6)
    assert a['all_rows']['action_loss'] == pytest.approx(b['all_rows']['action_loss'], abs=1e-6)
    assert a['slot0']['type_probability_by_target']['hold']['row_count'] == 2
    assert b['slot0']['type_probability_by_target']['hold']['row_count'] == 2


def test_18454_class_catalog_keeps_type_and_exact_ce_distinct():
    from types import SimpleNamespace
    import math
    import pytest

    actions = SimpleNamespace(atom_kind_mask=torch.tensor([0] + [1] * 18453),
                              factor_values={'kind_mask': ('STOP', '1')})
    metric = CapabilityMetrics(actions)
    metric.update(torch.zeros(1, 2, 18454), {'action_target': torch.tensor([[18453, 0]])})
    result = metric.report()['slot0']
    assert result['exact_atom_ce'] == pytest.approx(math.log(18454), abs=1e-5)
    assert result['type_ce'] == pytest.approx(-math.log(18453 / 18454), abs=1e-5)
