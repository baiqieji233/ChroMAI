from fractions import Fraction

import numpy as np

from maichart.composer.workload import ExactWorkloadQuery, build_workload_targets
from maichart.event_timing import EventPoint, EventTempo


def test_exact_query_preserves_legacy_values_across_tempo_and_boundaries():
    tempo = [EventTempo(Fraction(0), 120.), EventTempo(Fraction(2), 60.)]
    points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(1), '12'),
              EventPoint(Fraction(2), 'Ch[4:1]'), EventPoint(Fraction(3), '3-6[4:1]'),
              EventPoint(Fraction(5), '4')]
    centers = [-.1, 0., .0001, .173, .5, 1.007, 2.2, 3.4999, 3.5]
    reusable = ExactWorkloadQuery(points, tempo, -.25, 3.5, parsed=False)
    expected = build_workload_targets(points, tempo, -.25, centers, 3.5)
    np.testing.assert_allclose(reusable.query(centers), expected, rtol=1e-6, atol=1e-6)


def test_exact_query_does_not_blur_an_off_grid_kernel_peak():
    tempo = [EventTempo(Fraction(0), 60.)]
    query = ExactWorkloadQuery([EventPoint(Fraction(137, 100), '12')], tempo, 0., 4.)
    result = query.query([1.12, 1.37, 1.62, 2.37])
    np.testing.assert_allclose(np.expm1(result[:, :2]),
                               [[.75, 1.5], [1., 2.], [.75, 1.5], [0., 0.]], atol=1e-6)


def test_exact_query_shared_heads_and_type_fractions_have_common_semantics():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(1), '1-4[4:1]*-6[4:1]'),
              EventPoint(Fraction(3, 2), 'Ch[4:1]')]
    result = ExactWorkloadQuery(points, tempo, 0., 4., parsed=True).query([1.25])[0]
    np.testing.assert_allclose(np.expm1(result[:2]), [1.5, 1.5], atol=1e-6)
    np.testing.assert_allclose(result[2:], [.5, .5, .5], atol=1e-6)


def test_exact_query_short_file_normalizes_actual_kernel_mass_and_reuses_prefix():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(0), '12'), EventPoint(Fraction(1, 2), '3')]
    query = ExactWorkloadQuery(points, tempo, 0., .5)
    # The event exactly at duration is excluded; triangle mass is .5 - .5**2/2.
    np.testing.assert_allclose(np.expm1(query.query([0.])[0, :2]), [1. / .375, 2. / .375], atol=1e-6)
    np.testing.assert_array_equal(query.query([-.1, .5, 1.]), np.zeros((3, 5)))
    first = query.query([.123, .249])
    np.testing.assert_array_equal(query.query([.123, .249]), first)
    assert query.query([]).shape == (0, 5)
    np.testing.assert_array_equal(ExactWorkloadQuery([], tempo, 0., 0.).query([0.]), np.zeros((1, 5)))


def test_unknown_intervals_are_removed_from_signal_and_normalization_mass():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(1, 2), '1'), EventPoint(Fraction(2), '2')]
    query = ExactWorkloadQuery(points, tempo, 0., 3., unknown_intervals=[(.4, 1.6)])
    # The center's entire kernel is unknown; nearby centers normalize only the
    # known side, and no event from inside the unknown interval contributes.
    np.testing.assert_array_equal(query.weight_at([1., 2.]), [0., 0.])
    np.testing.assert_array_equal(query.query([1.]), np.zeros((1, 5)))
    left = query.query([.1])[0]
    np.testing.assert_allclose(np.expm1(left[:2]), [0., 0.], atol=1e-6)
    right = query.query([1.9])[0]
    assert right[0] > 0. and right[1] > 0.


def test_positive_confidence_only_weights_supervision_without_changing_labels():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(1), '1'), EventPoint(Fraction(2), '2')]
    query = ExactWorkloadQuery(points, tempo, 0., 4., regions=[(0., 1.5, .25)])
    assert query.weight_at([.5])[0] == .25
    plain = ExactWorkloadQuery(points, tempo, 0., 4.).query([.5, 1., 1.5, 2.])
    np.testing.assert_array_equal(query.query([.5, 1., 1.5, 2.]), plain)
