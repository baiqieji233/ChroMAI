"""Behavioral contracts for manual Composer density/chord/rest controls."""
import numpy as np
import pytest
import torch

from maichart.composer.curve import (
    INTENSITY_MAPPING_POLICY, apply_curve_to_composer,
    intensity_to_rate, rate_to_intensity,
)
from maichart.composer.generate import _sample_actions, parser
from maichart.generation.event_cli import _NoteGrammar


def profile(value=.5, *, reference=False, rests=None):
    points = [{'time_sec': 0., 'intensity': value}, {'time_sec': 20., 'intensity': value}]
    result = {'schema': 'maichart-intensity-curve-v1', 'duration_sec': 20.,
              'baseline_head_rate': 4., 'points': points, 'rest_regions': rests or []}
    if reference:
        result['reference_points'] = [dict(p) for p in points]
    return result


def predictions():
    workload = torch.tensor([[[np.log1p(1.), np.log1p(4.), .3, .4, .2],
                              [np.log1p(2.), np.log1p(3.), .2, .2, .3]]], dtype=torch.float32)
    planner = torch.cat((workload.clone(), torch.tensor([[[.15], [.7]]])), dim=-1)
    trailing = torch.tensor([[[np.log1p(1.), np.log1p(2.), np.log1p(3.), np.log1p(8.)],
                              [np.log1p(2.), np.log1p(3.), np.log1p(1.), np.log1p(4.)]]], dtype=torch.float32)
    return workload, planner, trailing


def test_rate_mapping_inverts_all_endpoints_without_clipped_peak():
    rates = np.asarray([0., .1, 1., 4., 20., 64.])
    mapped = rate_to_intensity(rates, baseline=4.)
    np.testing.assert_allclose(intensity_to_rate(mapped, baseline=4.), rates, atol=1e-10)
    assert mapped[0] == 0. and mapped[-1] == 1.
    assert np.isfinite(intensity_to_rate(1., baseline=4.))


@pytest.mark.parametrize('mode', ['soft', 'follow', 'hard', 'strict'])
def test_unedited_prediction_is_exact_identity_in_all_branches(mode):
    ctrl = apply_curve_to_composer(profile(reference=True), duration_sec=20., mode=mode)
    workload, planner, trailing = predictions()
    assert ctrl.transform_planner(planner, np.array([5., 10.])) is planner
    out = ctrl.build_window_overrides(candidate_seconds=np.array([5., 10.]),
                                     model_workload=workload, model_trailing=trailing)
    assert out['workload_override'] is None and out['trailing_override'] is None
    assert out['total_heads_budget'] is None


@pytest.mark.parametrize('mode', ['soft', 'follow', 'hard', 'strict'])
def test_zero_strength_disables_rests_and_chord_budget(mode):
    ctrl = apply_curve_to_composer(profile(0., rests=[{'start_sec': 0., 'end_sec': 20.}]),
                                  duration_sec=20., mode=mode, strength=0., chord_multiplier=.25)
    workload, planner, trailing = predictions()
    assert not ctrl.enabled and not ctrl.is_in_rest_region(5.)
    assert not ctrl.has_future_activity(0., 20.)
    assert ctrl.transform_planner(planner, np.array([5., 10.])) is planner
    out = ctrl.build_window_overrides(candidate_seconds=np.array([5., 10.]),
                                     model_workload=workload, model_trailing=trailing)
    assert out['workload_override'] is None and out['trailing_override'] is None
    assert out['total_heads_budget'] is None


def test_trim_maps_rest_and_curve_together():
    data = profile(.8, rests=[{'start_sec': 5., 'end_sec': 10.}])
    ctrl = apply_curve_to_composer(data, duration_sec=20., span_mode='trim',
                                  effective_start_sec=4., effective_end_sec=16.)
    # Profile [5, 10] maps to actual [7, 10], including its rest mask.
    assert ctrl.is_in_rest_region(8.) and ctrl.target_at(8.) == 0.
    assert not ctrl.is_in_rest_region(5.) and ctrl.target_at(5.) == pytest.approx(.8)


def test_neutral_absolute_curve_preserves_chord_ratio_and_other_channels():
    ctrl = apply_curve_to_composer(profile(), duration_sec=20., mode='follow')
    workload, planner, trailing = predictions()
    out = ctrl.transform_planner(planner[:, :1], np.array([5.]))
    torch.testing.assert_close(out, planner[:, :1])
    overrides = ctrl.build_window_overrides(candidate_seconds=np.array([5.]), model_workload=workload[:, :1])
    torch.testing.assert_close(overrides['workload_override'], workload[:, :1])


def test_chord_multiplier_redistributes_fixed_head_density():
    workload, _, _ = predictions()
    ctrl = apply_curve_to_composer(profile(reference=True), duration_sec=20., mode='strict', chord_multiplier=.5)
    out = ctrl.build_window_overrides(candidate_seconds=np.array([5., 10.]), model_workload=workload)
    rates = out['workload_override'][0, :, :2].expm1()
    torch.testing.assert_close(rates[:, 1], workload[0, :, 1].expm1())
    # Four heads at one event/s becomes four heads at two events/s, not a new
    # head density or a hard-coded 0.78 event/head ratio.
    assert rates[0, 0].item() == pytest.approx(2.)
    assert out['total_heads_budget'][0].item() == pytest.approx(2.)
    assert out['total_heads_budget'][1].item() == pytest.approx(1.)


def test_edit_only_changes_edited_region_and_keeps_types_rest_probability():
    data = profile(reference=True)
    data['points'] = [{'time_sec': 0., 'intensity': .5}, {'time_sec': 10., 'intensity': .5},
                      {'time_sec': 20., 'intensity': .8}]
    ctrl = apply_curve_to_composer(data, duration_sec=20., mode='hard')
    _, planner, _ = predictions()
    out = ctrl.transform_planner(planner, np.array([5., 20.]))
    torch.testing.assert_close(out[0, 0], planner[0, 0])
    torch.testing.assert_close(out[..., 2:], planner[..., 2:])
    assert out[0, 1, 1] > planner[0, 1, 1]


def test_total_note_budget_includes_open_touch_without_changing_button_rule():
    atoms = ['A1', 'B2', 'C', 'E4']
    grammar = _NoteGrammar.from_atoms(atoms)
    logits = torch.tensor([[-20., 4., 3., 2., 1.]]).expand(5, -1)
    unrestricted = _sample_actions(logits, atoms, grammar, 0., torch.Generator(), touch_layout_policy='open')
    limited = _sample_actions(logits, atoms, grammar, 0., torch.Generator(), touch_layout_policy='open',
                              max_total_heads=2)
    assert len(unrestricted) == 4 and len(limited) == 2


def test_parser_keeps_manual_and_native_curve_strength_independent():
    args = parser().parse_args(['--audio', 'a', '--checkpoint', 'c', '--output', 'o',
                               '--curve-strength', '1.5', '--intensity-curve-strength', '0',
                               '--intensity-curve-chord-multiplier', '.5', '--meter-map', 'm.json'])
    assert args.curve_strength == 1.5 and args.intensity_curve_strength == 0.
    assert args.intensity_curve_chord_multiplier == .5 and args.meter_map == 'm.json'


def test_default_manual_strength_does_not_follow_native_strength():
    args = parser().parse_args(['--audio', 'a', '--checkpoint', 'c', '--output', 'o', '--curve-strength', '0'])
    assert args.intensity_curve_strength == 1.
