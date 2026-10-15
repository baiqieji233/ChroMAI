from fractions import Fraction

import torch

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.rhythm_geometry import (
    CausalGeometry,
    GEOMETRY_DIM,
    causal_prefix_geometry,
    coordinate_features,
    timing_targets,
)


def test_coordinate_features_are_fraction_exact_at_the_boundary():
    values = coordinate_features([Fraction(-3, 2), Fraction(1, 4)], origin=Fraction(-1, 2))
    assert values.shape == (2, GEOMETRY_DIM)
    assert torch.isfinite(values).all()
    assert values[0, 0].item() == -1 / 64
    assert values[1, 0].item() == 3 / 4 / 64


def test_causal_prefix_does_not_see_unconsumed_position_digits():
    codec = RationalCodec()
    ids = codec.encode([EventPoint(Fraction(3, 2), "1")], origin=Fraction(0))
    prefix = causal_prefix_geometry(ids, codec)
    # BOS/AT/first digit cannot expose the future position; the complete
    # position is exposed only after <note> has been consumed.
    assert prefix[0, 0].item() == 0.0
    assert prefix[1, 0].item() == 0.0
    assert prefix[2, 0].item() == 0.0
    assert prefix[4, 0].item() == 0.0
    assert prefix[5, 0].item() == Fraction(3, 2) / 64
    assert prefix[-1, 0].item() == Fraction(3, 2) / 64


def test_timing_targets_mask_only_next_event_boundaries_and_handles_eos():
    codec = RationalCodec()
    points = [EventPoint(Fraction(1, 2), "1"), EventPoint(Fraction(5, 4), "2")]
    encoded = codec.encode(points)
    decoder_input = torch.tensor(encoded[:-1])
    target = torch.tensor(encoded[1:])
    labels = timing_targets(points, decoder_input, target, codec)
    valid = labels["mask"].nonzero().flatten().tolist()
    assert valid == [0, 7]
    assert labels["gap_log1p"][valid[0]].item() == torch.log1p(torch.tensor(.5)).item()
    assert labels["gap_log1p"][valid[1]].item() == torch.log1p(torch.tensor(.75)).item()
    assert not labels["mask"][-1]


def test_incremental_geometry_matches_full_prefix_and_can_replay_after_rewind():
    codec = RationalCodec()
    ids = codec.encode([EventPoint(Fraction(7, 6), "1"), EventPoint(Fraction(2), "2")])
    full = causal_prefix_geometry(ids, codec, coordinate_origin=Fraction(1, 2))
    state = CausalGeometry(codec, coordinate_origin=Fraction(1, 2))
    incremental = torch.tensor([state.consume(token) for token in ids])
    torch.testing.assert_close(incremental, full)
    replay = CausalGeometry(codec, coordinate_origin=Fraction(1, 2))
    for token in ids[:5]:
        replay.consume(token)
    torch.testing.assert_close(torch.tensor(replay.consume(ids[5])), full[5])
