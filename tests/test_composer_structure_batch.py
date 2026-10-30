from types import SimpleNamespace

import numpy as np
import pytest

from maichart.composer import structure


def _query_fixture(monkeypatch, anchors):
    rng = np.random.default_rng(629)
    bank = np.zeros((len(anchors), 2 + len(structure.PERIODS) * 8), np.float32)
    bank[:, 0] = anchors
    records = rng.uniform(.01, .7, (len(anchors), len(structure.PERIODS), 8)).astype(np.float32)
    bank[:, 2:] = records.reshape(len(anchors), -1)
    order = np.tile(np.array([0, 3, 9, -1]), (len(anchors), 1))
    novelty = np.array([[0., .1], [4., .6], [20., .2]], np.float32)
    local = np.array([[0., .1, .3], [1., .8, .2], [3., .2, .9], [10., .1, .2]], np.float32)
    cached = (bank, novelty, order, records[:, :4].copy(), local)
    calls = []

    def get_bank(*args, **kwargs):
        calls.append(1)
        return cached

    monkeypatch.setattr(structure, '_bank', get_bank)
    return SimpleNamespace(audio=SimpleNamespace(duration_sec=10.)), calls


@pytest.mark.parametrize('policy', sorted(structure.STRUCTURE_POLICIES))
@pytest.mark.parametrize('anchors', [[0.], [0., 8., 16.]])
def test_batch_matches_scalar_bits_with_mixed_predecessors(monkeypatch, policy, anchors):
    song, calls = _query_fixture(monkeypatch, anchors)
    beats = np.array([[-2., 0., 3.9999, 4., 8.], [8., 8.0001, 12., 16., 22.],
                      [1., 3., 5., 7., 9.]])
    # Include negative time, audio tail and nonconstant beat/second mapping.
    seconds = np.array([[-1., 0., 1., 1.1, 4.], [4., 4.1, 9.99, 10., 12.],
                        [.2, .8, 1.8, 3., 5.]])
    pb, ps = [None, 7.5, 2.], [None, 3.8, .5]
    expected = [structure.structure_features_and_transition(
        song, b, s, p, t, policy=policy, backend='native')
        for b, s, p, t in zip(beats, seconds, pb, ps)]
    calls.clear()
    actual, delta = structure.structure_features_and_transition_batch(
        song, beats, seconds, pb, ps, policy=policy, backend='native')
    assert len(calls) == (1 if policy in structure.METRICAL_POLICIES else 0)
    assert actual.dtype == delta.dtype == np.float32
    for index, (features, transition) in enumerate(expected):
        np.testing.assert_array_equal(actual[index].view(np.uint32), features.view(np.uint32))
        np.testing.assert_array_equal(delta[index].view(np.uint32), transition.view(np.uint32))


def test_batch_without_predecessors_and_empty_grids(monkeypatch):
    song, _ = _query_fixture(monkeypatch, [0., 8.])
    for shape in ((3, 2), (0, 2), (3, 0)):
        beats = np.zeros(shape)
        features, delta = structure.structure_features_and_transition_batch(song, beats, beats)
        assert features.shape == (*shape, structure.STRUCTURE_DIM)
        np.testing.assert_array_equal(features, delta)
        assert not np.shares_memory(features, delta)


@pytest.mark.parametrize('beats,seconds,pb,ps', [
    ([1.], [1.], None, None),
    ([[1.]], [[1., 2.]], None, None),
    ([[1.]], [[1.]], [1.], None),
    ([[1.]], [[1.]], [], []),
    ([[1.]], [[1.]], [None], [1.]),
    ([[float('nan')]], [[1.]], None, None),
    ([[1.]], [[1.]], [float('inf')], [1.]),
])
def test_batch_rejects_invalid_queries(beats, seconds, pb, ps):
    with pytest.raises(ValueError):
        structure.structure_features_and_transition_batch(
            None, beats, seconds, pb, ps, policy='off')
