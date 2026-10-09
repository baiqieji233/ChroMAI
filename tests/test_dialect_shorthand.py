"""Hand-made-pack shorthand forms: accepted by definition, everything else stays rejected.

The semantics come from the pack author (user decision):

* ``7h`` without a duration block is a hold that starts and ends at the same
  instant -- exactly ``7h[1:0]``;
* ``1`2`` is a "pseudo double press": the backtick means simultaneity, so it is
  rewritten to ``1/2``;
* ``Cfh[..]`` and ``Chf[..]`` are the same touch hold (the firework marker is
  order-free);
* ``EE`` is a typo for a single ``E`` touch;
* ``?``, ``!``, ``$``, a mid-cell ``{division}``, an unknown timing pair and a
  duplicated button head stay **rejected** -- they are not the training target.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from maichart.event_timing import parse_event_chart_text
from maichart.notes import parse_basic_hold_token, parse_basic_touch_token, parse_timing_point_notes
from maichart.timing import normalize_note_text, tokenize_inote_timing


def _chart(notes: str, *, division: int = 4):
    text = f"&title=T\n&wholebpm=120\n&first=0\n&lv_5=13\n&inote_5=(120){{{division}}}{notes}\n"
    return parse_event_chart_text(text, 5)


def _note_texts(notes: str, *, division: int = 4) -> list[str]:
    points = tokenize_inote_timing(f"(120){{{division}}}{notes}", 120.0)
    return [point.note_text for point in points]


# --------------------------------------------------------------------------- #
# bare holds == [1:0]
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("token", ["1h", "5h", "8h"])
def test_bare_hold_equals_the_zero_length_form(token):
    """`7h` is a hold that begins and ends together: identical to `7h[1:0]`."""

    bare = parse_basic_hold_token(token)
    explicit = parse_basic_hold_token(f"{token}[1:0]")
    assert bare is not None and explicit is not None
    assert bare.note_type == explicit.note_type == "hold"
    assert bare.position == explicit.position
    assert bare.duration_beats == explicit.duration_beats == Fraction(0)
    assert bare.duration_ticks == explicit.duration_ticks == 0
    assert bare.duration.kind == explicit.duration.kind == "grid_fraction"
    # The original spelling survives for reporting/round-trip.
    assert bare.raw == token and explicit.raw == f"{token}[1:0]"


@pytest.mark.parametrize(
    ("token", "is_break"),
    [("3xh", False), ("2hx", False), ("8bh", True), ("5xh", False), ("7hx", False)],
)
def test_bare_hold_with_modifiers_is_zero_length(token, is_break):
    """EX (`x`) and break (`b`) markers may sit on either side of `h`."""

    note = parse_basic_hold_token(token)
    assert note is not None, token
    assert note.note_type == "hold"
    assert note.duration_beats == Fraction(0)
    assert note.is_break is is_break


def test_explicit_hold_durations_are_untouched():
    """The shorthand must not loosen ordinary durations."""

    assert parse_basic_hold_token("5h[8:1]").duration_beats == Fraction(1, 2)
    assert parse_basic_hold_token("2h[4:1]").duration_beats == Fraction(1)
    assert parse_basic_hold_token("4h[16:3]").duration_beats == Fraction(3, 4)
    # An unparseable duration block is still rejected outright.
    assert parse_basic_hold_token("4h[not-a-duration]") is None


# --------------------------------------------------------------------------- #
# touch holds: firework order is free, duration optional
# --------------------------------------------------------------------------- #


def test_touch_hold_firework_order_is_free():
    """`Cfh[32:1]` and `Chf[32:1]` are the same note."""

    before = parse_basic_touch_token("Cfh[32:1]")
    after = parse_basic_touch_token("Chf[32:1]")
    assert before is not None and after is not None
    assert before.note_type == after.note_type == "touch_hold"
    assert before.position == after.position
    assert before.duration_beats == after.duration_beats == Fraction(1, 8)
    assert before.modifiers["firework"] is True
    assert after.modifiers["firework"] is True


@pytest.mark.parametrize("token", ["Cfh", "E4h", "A2h"])
def test_bare_touch_hold_is_zero_length(token):
    note = parse_basic_touch_token(token)
    assert note is not None, token
    assert note.note_type == "touch_hold"
    assert note.duration_beats == Fraction(0)


def test_plain_touch_and_positioned_touch_still_parse():
    assert parse_basic_touch_token("Cf").note_type == "touch"
    assert parse_basic_touch_token("Cf").modifiers["firework"] is True
    assert parse_basic_touch_token("E8").position == "E8"


# --------------------------------------------------------------------------- #
# pseudo double press and duplicated touch area
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1`2", "1/2"),
        ("A2`E2", "A2/E2"),
        ("6`3", "6/3"),
        ("2>4q8[4:29]`1q5<7[4:29]", "2>4q8[4:29]/1q5<7[4:29]"),
    ],
)
def test_pseudo_double_press_becomes_a_real_chord(raw, expected):
    assert normalize_note_text(raw) == expected


@pytest.mark.parametrize("text", ["1/2", "36", "E8/B8", "7h", "1<>4[4:1]", ""])
def test_normalisation_leaves_ordinary_cells_alone(text):
    assert normalize_note_text(text) == text


@pytest.mark.parametrize(("raw", "expected"), [("EE", "E"), ("EE/B8", "E/B8"), ("AAA", "A")])
def test_repeated_touch_area_collapses_to_one(raw, expected):
    """`EE` is a typo: treat it as a single `E` instead of dropping the chart."""

    assert normalize_note_text(raw) == expected


def test_pseudo_chord_yields_two_simultaneous_taps():
    """After normalisation the cell is an ordinary two-button chord."""

    points = tokenize_inote_timing("(120){4}1`2,E", 120.0)
    assert points[0].note_text == "1/2"
    assert points[0].raw.endswith("1`2"), "the raw cell keeps the original spelling"
    parsed = parse_timing_point_notes(points[0])
    assert not parsed.unknown_tokens
    assert len(parsed.notes) == 2
    assert {note.position for note in parsed.notes} == {1, 2}


def test_repeated_touch_area_yields_one_touch():
    points = tokenize_inote_timing("(120){4}EE,E", 120.0)
    assert points[0].note_text == "E"
    parsed = parse_timing_point_notes(points[0])
    assert not parsed.unknown_tokens
    assert len(parsed.notes) == 1


# --------------------------------------------------------------------------- #
# the rejected dialects stay rejected
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "token",
    [
        "7?p3[23.83#48:3]",       # `?` slide modifier
        "3!pp5<1p4[32:224]",      # `!`
        "6$b",                    # `$`
        "1{8}1bx",                # mid-cell division directive
        "8bxq1b[213.33#4:1]",     # per-note BPM timing pair on a slide
        "68[8:1]",                # duration on a compact tap chord (typo)
    ],
)
def test_unsupported_dialects_are_still_unknown_tokens(token):
    points = tokenize_inote_timing(f"(120){{4}}{token},E", 120.0)
    parsed = parse_timing_point_notes(points[0])
    assert parsed.unknown_tokens, f"{token} must stay unsupported"


def test_document_with_rejected_dialect_is_not_syntax_valid(tmp_path):
    from maichart.generation.event_quality import assess_event_chart

    chart = _chart("1,2?p3[8:1],E")
    report = assess_event_chart(chart.points, chart.tempo_changes, first=chart.first)
    assert not report["syntax_valid"]


def test_duplicate_button_head_is_still_invalid():
    """`6h/6` stays a quality rejection: two heads on one button at one beat."""

    from maichart.generation.event_quality import assess_event_chart

    chart = _chart("6h[4:1]/6,E")
    report = assess_event_chart(chart.points, chart.tempo_changes, first=chart.first)
    assert not report["syntax_valid"]


# --------------------------------------------------------------------------- #
# end to end: a shorthand chart becomes a trainable document
# --------------------------------------------------------------------------- #


def test_shorthand_chart_is_syntax_valid_and_trainable():
    """A document using every accepted shorthand passes the training QC."""

    from maichart.generation.event_quality import assess_event_chart
    from maichart.training.event_dataset import inspect_chart

    chart = _chart("7h,1`2,A2`E2,Cfh[32:1],EE,3xh,8bh,5h[8:1],E")
    report = assess_event_chart(chart.points, chart.tempo_changes, first=chart.first)
    assert report["syntax_valid"], report.get("fatal_errors")
    inspect_chart(chart)  # raises on unsupported characters or syntax

    texts = [point.note_text for point in chart.points]
    assert "7h" in texts
    assert "1/2" in texts and "A2/E2" in texts
    assert "EE" not in texts and "E" in texts
    assert "Cfh[32:1]" in texts


def test_bare_hold_keeps_its_spelling_in_the_training_text():
    """The model learns the shorthand as written; no rewriting to `[1:0]`."""

    chart = _chart("7h,1,2,3,E")
    texts = [point.note_text for point in chart.points]
    assert texts[0] == "7h"
    note = parse_basic_hold_token(texts[0])
    assert note.duration_beats == Fraction(0)
