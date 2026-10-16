from fractions import Fraction

import torch

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.rhythm_geometry import (
    CausalGeometry,
    GEOMETRY_DIM,
    GEOMETRY_DIM_V2,
    causal_prefix_geometry,
    coordinate_features,
    geometry_dim,
    timing_targets,
)
from maichart.form.timing import MeterChange, MeterPhaseIndex, meter_phase


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


# --------------------------------------------------------------------------
# exact_fraction_v2: rhythm identity is meter phase, not distance from a span
# --------------------------------------------------------------------------

def test_geometry_dim_matches_each_profile():
    assert geometry_dim('none') == 0
    assert geometry_dim('exact_fraction_v1') == GEOMETRY_DIM == 8
    assert geometry_dim('exact_fraction_v2') == GEOMETRY_DIM_V2 == 13
    try:
        geometry_dim('exact_fraction_v9')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown profile must be rejected')


def test_meter_phase_measures_from_the_bar_line_not_a_span_edge():
    meter = MeterChange(Fraction(0), 4, 4)
    # Beat 1 and beat 2 of a bar are different bar phases, which is exactly the
    # distinction the v1 offset-from-span-start basis could not express.
    assert meter_phase(meter, Fraction(1))[0] == 0.25
    assert meter_phase(meter, Fraction(2))[0] == 0.50
    # The bar line repeats, so beat 5 is beat 1 again.
    assert meter_phase(meter, Fraction(5))[0] == 0.25


def test_phase_index_repeats_across_bars_and_tracks_meter_changes():
    index = MeterPhaseIndex([MeterChange(Fraction(0), 4, 4), MeterChange(Fraction(12), 3, 4)])
    bar4 = MeterPhaseIndex([MeterChange(Fraction(0), 4, 4)])
    # Beat 1.5 in bar 1 and beat 9.5 in bar 3 share the 4/4 bar phase.
    assert index.phase(Fraction(3, 2))[0] == bar4.phase(Fraction(3, 2))[0] == 0.375
    assert index.phase(Fraction(19, 2))[0] == bar4.phase(Fraction(19, 2))[0] == 0.375
    # The 3/4 section restarts its own bar phase at its declared beat, so beat
    # 12 and beat 15 are both bar lines of a three-beat bar.
    assert index.phase(Fraction(12))[0] == 0.0
    assert index.phase(Fraction(15))[0] == 0.0
    assert index.phase(Fraction(12))[3] == 1.0
    # A beat inside that section uses the three-beat bar, not the old one.
    assert abs(index.phase(Fraction(43, 2))[0] - (1 / 6)) < 1e-9


def test_unknown_meter_uses_an_absolute_prior_and_flags_it():
    index = MeterPhaseIndex([])
    # No meter map: the prior is anchored at absolute beat zero, so it is still
    # span-independent, and the meter flag tells the model not to trust it.
    assert index.phase(Fraction(1))[0] == 0.25
    assert index.phase(Fraction(5))[0] == 0.25
    assert index.phase(Fraction(1))[3] == 0.0
    assert index.phase(Fraction(1))[1] == 0.0


def test_v2_distinguishes_a_24th_on_beat_1_from_one_on_beat_2():
    """The defect the v1 contract had: both collapsed to the same coordinate."""
    meter = [MeterChange(Fraction(0), 4, 4)]
    origin = Fraction(0)
    v1 = coordinate_features([Fraction(7, 6), Fraction(13, 6)], origin=origin)
    # v1 24th on beat 1 and 24th on beat 2 share sub-beat phase...
    for channel in (3, 4, 5):
        assert v1[0, channel].item() == v1[1, channel].item()
    # ...so the v1 basis separated them only by distance from the span start.

    v2 = coordinate_features([Fraction(7, 6), Fraction(13, 6)], origin=origin,
                             profile='exact_fraction_v2', meters=meter)
    assert v2.shape == (2, GEOMETRY_DIM_V2)
    # Bar phase (channel 5) and its sine/cosine now differ. Feature rows are
    # float32, so compare numerically rather than by exact float equality.
    assert abs(v2[0, 5].item() - 7 / 24) < 1e-6
    assert abs(v2[1, 5].item() - 13 / 24) < 1e-6
    assert v2[0, 6].item() != v2[1, 6].item()
    # Sub-beat phase stays shared, because both really are 24ths.
    assert abs(v2[0, 2].item() - 1 / 6) < 1e-6
    assert abs(v2[1, 2].item() - 1 / 6) < 1e-6


def test_v2_coordinates_are_identical_across_spans_and_bar_repetitions():
    """The same musical position must not depend on where a span began."""
    meter = [MeterChange(Fraction(0), 4, 4)]
    # Beat 33/2 is a 24th into beat 16, i.e. into beat 0 of bar 5. Starting the
    # span at 14 versus 16 must not change the coordinate.
    left = coordinate_features([Fraction(33, 2)], origin=Fraction(14),
                               profile='exact_fraction_v2', meters=meter)
    right = coordinate_features([Fraction(33, 2)], origin=Fraction(16),
                                profile='exact_fraction_v2', meters=meter)
    for channel in range(2, GEOMETRY_DIM_V2):
        assert left[0, channel].item() == right[0, channel].item(), channel
    before = coordinate_features([Fraction(1, 2)], origin=Fraction(0),
                                 profile='exact_fraction_v2', meters=meter)
    after = coordinate_features([Fraction(8 + 1, 2)], origin=Fraction(8),
                                profile='exact_fraction_v2', meters=meter)
    for channel in range(2, GEOMETRY_DIM_V2):
        assert before[0, channel].item() == after[0, channel].item(), channel


def test_v2_depends_on_the_absolute_beat_not_the_span_origin():
    """Two different origins for the same beat must produce the same row."""
    meter = [MeterChange(Fraction(0), 4, 4)]
    # Same absolute beat 5/2, but one span starts at 0 and another at 2.
    first = coordinate_features([Fraction(5, 2)], origin=Fraction(0),
                                profile='exact_fraction_v2', meters=meter)
    second = coordinate_features([Fraction(5, 2)], origin=Fraction(2),
                                 profile='exact_fraction_v2', meters=meter)
    # Channel 0 is the local offset and legitimately differs.
    assert first[0, 0].item() != second[0, 0].item()
    # Channel 2 is sub-beat phase; it must not.
    assert first[0, 2].item() == second[0, 2].item()


def test_v2_causal_geometry_is_causal_and_replays_after_rewind():
    codec = RationalCodec()
    meter = [MeterChange(Fraction(0), 4, 4)]
    ids = codec.encode([EventPoint(Fraction(7, 6), "1"), EventPoint(Fraction(13, 6), "2")])
    full = causal_prefix_geometry(ids, codec, origin=Fraction(0),
                                  profile='exact_fraction_v2', meters=meter)
    assert full.shape == (len(ids), GEOMETRY_DIM_V2)
    # Nothing before the complete position is visible.
    assert all(full[index, 0].item() == 0.0 for index in (0, 1, 2, 4))
    state = CausalGeometry(codec, origin=Fraction(0), profile='exact_fraction_v2', meters=meter)
    assert torch.allclose(torch.tensor([state.consume(token) for token in ids]), full)
    replayed = CausalGeometry(codec, origin=Fraction(0), profile='exact_fraction_v2', meters=meter)
    for token in ids[:5]:
        replayed.consume(token)
    assert torch.allclose(torch.tensor(replayed.consume(ids[5])), full[5])


def test_v2_timing_targets_use_bar_phase():
    codec = RationalCodec()
    meter = [MeterChange(Fraction(0), 4, 4)]
    points = [EventPoint(Fraction(1), "1"), EventPoint(Fraction(2), "2")]
    encoded = codec.encode(points)
    decoder_input = torch.tensor(encoded[:-1])
    target = torch.tensor(encoded[1:])
    legacy = timing_targets(points, decoder_input, target, codec, origin=Fraction(0))
    v2 = timing_targets(points, decoder_input, target, codec, origin=Fraction(0),
                        profile='exact_fraction_v2', meters=meter)
    valid = v2["mask"].nonzero().flatten().tolist()
    # Beat 1 is a quarter into the bar, beat 2 a half: sin(pi/2) and sin(pi).
    assert abs(v2["phase_sin"][valid[0]].item() - 1.0) < 1e-6
    assert abs(v2["phase_cos"][valid[1]].item() + 1.0) < 1e-6
    # v1 measured the same events from a whole-beat phase instead.
    assert abs(legacy["phase_sin"][valid[1]].item()) < 1e-6


# --------------------------------------------------------------------------
# position_event_loss: one event is one unit of supervision
# --------------------------------------------------------------------------

def _position_reference(logits, target, decoder_input, codec):
    """Independent walk of the same state machine, for cross-checking."""
    ce = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), target,
                                           reduction="none")[0]
    valid = target[0] != codec.pad_id
    state, event = 0, 0
    groups = {}
    for i in range(decoder_input.shape[1]):
        inp, tgt = int(decoder_input[0, i]), int(target[0, i])
        if inp in (codec.bos_id, codec.end_id) and tgt == codec.at_id:
            event += 1
        if inp == codec.at_id:
            state = 1
        if inp in (codec.note_id, codec.end_id):
            state = 0
        if state == 1 and bool(valid[i]):
            groups.setdefault(event, []).append(i)
    per_event = [float(ce[idx].mean()) for idx in groups.values() if idx]
    return sum(per_event) / len(per_event)


def _position_batch(points):
    codec = RationalCodec()
    ids = codec.encode(sorted(points, key=lambda p: p.beat), origin=Fraction(0))
    return codec, torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])


def test_position_event_loss_matches_an_independent_per_event_average():
    """The helper must group by event, not by token position.

    A token-averaged reduction silently weights an event by its spelling
    length, which is what this objective exists to undo.
    """
    from maichart.form.rhythm_geometry import position_event_loss

    for points in (
        [EventPoint(Fraction(0), "1")],
        [EventPoint(Fraction(0), "1"), EventPoint(Fraction(1, 6), "2")],
        [EventPoint(Fraction(0), "1"), EventPoint(Fraction(1, 6), "2"),
         EventPoint(Fraction(1), "1/2"), EventPoint(Fraction(3, 2), "1h[4:1]"),
         EventPoint(Fraction(17, 12), "3s4[8:5]")],
    ):
        codec, decoder_input, target = _position_batch(points)
        torch.manual_seed(0)
        logits = torch.randn(1, decoder_input.shape[1], len(codec))
        got = float(position_event_loss(logits, target, decoder_input, codec))
        expected = _position_reference(logits, target, decoder_input, codec)
        assert abs(got - expected) < 1e-5, points


def test_position_event_loss_batches_independently_and_ignores_padding():
    from maichart.form.rhythm_geometry import position_event_loss

    codec = RationalCodec()
    first = [EventPoint(Fraction(0), "1"), EventPoint(Fraction(1, 6), "2")]
    second = [EventPoint(Fraction(0), "7"), EventPoint(Fraction(3, 2), "1h[4:1]")]
    encoded = [codec.encode(sorted(pts, key=lambda p: p.beat), origin=Fraction(0))
               for pts in (first, second)]
    length = max(len(ids) for ids in encoded)

    def pad(values, fill):
        return torch.tensor(values + [fill] * (length - len(values)))

    decoder_input = torch.stack([pad(ids[:-1], codec.pad_id) for ids in encoded])
    target = torch.stack([pad(ids[1:], codec.pad_id) for ids in encoded])
    torch.manual_seed(0)
    logits = torch.randn(2, length, len(codec))
    rows = position_event_loss(logits, target, decoder_input, codec, reduction="none")
    assert rows.shape == (2,)
    # A longer neighbour must not change a row's value.
    for index in range(2):
        solo = float(position_event_loss(logits[index:index + 1], target[index:index + 1],
                                         decoder_input[index:index + 1], codec))
        assert abs(float(rows[index]) - solo) < 1e-5


def test_position_event_loss_supervises_only_the_position_spelling():
    """Note text is left to the token objective, so it must not enter the mean."""
    from maichart.form.rhythm_geometry import position_event_loss

    codec, decoder_input, target = _position_batch([EventPoint(Fraction(1, 6), "1h[4:1]")])
    torch.manual_seed(0)
    logits = torch.randn(1, decoder_input.shape[1], len(codec))
    got = float(position_event_loss(logits, target, decoder_input, codec))
    expected = _position_reference(logits, target, decoder_input, codec)
    ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    all_tokens = float((ce * (target != codec.pad_id)).sum() / (target != codec.pad_id).sum())
    assert abs(got - expected) < 1e-5
    # The long note string makes a whole-sequence average differ from it.
    assert abs(got - all_tokens) > 1e-6


def test_position_event_loss_is_finite_without_any_event():
    from maichart.form.rhythm_geometry import position_event_loss

    codec = RationalCodec()
    target = torch.full((1, 4), codec.pad_id, dtype=torch.long)
    decoder_input = torch.full((1, 4), codec.pad_id, dtype=torch.long)
    logits = torch.randn(1, 4, len(codec))
    result = position_event_loss(logits, target, decoder_input, codec)
    assert torch.isfinite(result).all()
    assert float(result) == 0.0
