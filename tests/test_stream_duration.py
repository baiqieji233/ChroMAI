import torch

from maichart.composer.stream_duration import _duration_coefficients, _read_intervals


def test_native_duration_fit_retains_slide_wait_and_hold_beats():
    assert _duration_coefficients('1h[4:1]') == [0., 0., 1., 0.]
    assert _duration_coefficients('1-3[4:1]') == [1., 0., 2., 0.]
    assert _duration_coefficients('1') is None


def test_interval_reads_do_not_cross_padded_song_boundaries():
    cache = dict(seconds=torch.tensor([[0., 1., 2.], [0., 1., 0.]]),
        mask=torch.tensor([[True, True, True], [True, True, False]]),
        local=torch.tensor([[[0.], [1.], [2.]], [[10.], [11.], [999.]]]))
    queries=torch.tensor([[[.5, 3.]], [[.5, 3.]], [[1., 1.5]]])
    actual=_read_intervals(cache, queries, torch.tensor([0, 1, 0]))
    expected=torch.tensor([[[[.5], [2.]]], [[[10.5], [11.]]], [[[1.], [1.5]]]])
    torch.testing.assert_close(actual, expected)
