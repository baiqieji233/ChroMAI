from fractions import Fraction

import pytest
import torch

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.layout_metrics import layout_diagnostics


def test_fraction_exact_match_is_event_level_and_detects_one_wrong_digit():
    codec = RationalCodec()
    encoded = codec.encode([
        EventPoint(Fraction(1, 2), "1"),
        EventPoint(Fraction(3, 4), "2"),
    ])
    target = torch.tensor([encoded[1:]])
    decoder_input = torch.tensor([encoded[:-1]])
    logits = torch.full((1, target.size(1), len(codec)), -8.0)
    logits.scatter_(2, target.unsqueeze(-1), 8.0)

    # Keep the complete header shape valid, but change one Fraction digit.
    digit = next(i for i, token in enumerate(codec.tokens)
                 if token == "P:1")
    wrong_digit = next(i for i, token in enumerate(codec.tokens)
                       if token == "P:2")
    first_digit = int(torch.nonzero(target[0] == digit)[0])
    logits[0, first_digit, digit] = -8.0
    logits[0, first_digit, wrong_digit] = 8.0
    token_ce = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), target, ignore_index=codec.pad_id, reduction="none")

    values, counts = layout_diagnostics(
        logits, target, decoder_input, torch.tensor([0]), token_ce, codec)

    assert counts["draft_position_fraction_exact_accuracy"].item() == 2
    assert values["draft_position_fraction_exact_accuracy"].item() == pytest.approx(.5)
    assert values["draft_position_header_accuracy"].item() > 0


def test_event_decision_metrics_catch_false_positive_eos():
    codec = RationalCodec()
    encoded = codec.encode([EventPoint(Fraction(1, 2), "1")])
    target = torch.tensor([encoded[1:]])
    decoder_input = torch.tensor([encoded[:-1]])
    logits = torch.full((1, target.size(1), len(codec)), -8.0)
    logits.scatter_(2, target.unsqueeze(-1), 8.0)

    # Predict a note character where the structural final EOS is required.
    eos_index = int(torch.nonzero(target[0] == codec.eos_id)[0])
    logits[0, eos_index, codec.eos_id] = -8.0
    logits[0, eos_index, codec.ids["N:1"]] = 8.0
    token_ce = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), target, ignore_index=codec.pad_id, reduction="none")

    values, counts = layout_diagnostics(
        logits, target, decoder_input, torch.tensor([0]), token_ce, codec)

    assert counts["draft_event_decision_accuracy"].item() == 2
    assert counts["draft_event_terminate_accuracy"].item() == 1
    assert values["draft_event_terminate_accuracy"].item() == 0
    assert values["draft_event_terminate_loss"].item() > 0
