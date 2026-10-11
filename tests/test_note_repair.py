from fractions import Fraction
from types import SimpleNamespace

from maichart.event_timing import EventTempo
from maichart.generation.conflict_detector import NoteEvent
from maichart.generation.note_repair import candidate_note_edits


TEMPO = [EventTempo(Fraction(0), 120)]


def _event(note_type, raw, button, *, beat=0, time=0.0, duration=0.0, protected=False):
    return NoteEvent(note_type, button, time, duration, raw=raw,
                     source_id=(0, 0), beat=Fraction(beat), protected=protected)


def test_tap_neighbors_preserve_break_modifier_and_wrap():
    event = _event("tap", "1b", 1)
    edits = list(candidate_note_edits(event, None, [event], TEMPO, 0))
    assert [item["raw"] for item in edits] == ["8b", "2b"]
    assert all(item["operation"] == "move_tap" for item in edits)


def test_touch_center_is_not_moved_and_ring_neighbors_preserve_modifier():
    center = _event("touch", "C", "C")
    assert list(candidate_note_edits(center, None, [center], TEMPO, 0)) == []
    event = _event("touch", "A1f", "A1")
    assert [item["raw"] for item in candidate_note_edits(event, None, [event], TEMPO, 0)] == ["A8f", "A2f"]


def test_ex_note_and_complex_slide_are_not_rewritten():
    protected = _event("tap", "1x", 1, protected=True)
    assert list(candidate_note_edits(protected, None, [protected], TEMPO, 0)) == []
    complex_slide = _event("slide", "1-3-5[4:1]", 1)
    assert list(candidate_note_edits(complex_slide, None, [complex_slide], TEMPO, 0)) == []


def test_simple_slide_moves_endpoint_only_and_preserves_duration():
    event = _event("slide", "1b>5[4:1]", 1)
    raws = [item["raw"] for item in candidate_note_edits(event, None, [event], TEMPO, 0)]
    assert raws == ["1b>4[4:1]", "1b>6[4:1]"]


def test_hold_shortening_uses_same_pad_future_head_and_grid_duration():
    hold = _event("hold", "1h[4:4]", 1, beat=0, time=0.0, duration=2.0)
    later = _event("tap", "1", 1, beat=4, time=2.0)
    edits = list(candidate_note_edits(hold, None, [hold, later], TEMPO, 0))
    assert len(edits) == 1
    assert edits[0]["operation"] == "shorten_hold"
    assert edits[0]["raw"].startswith("1h[")
    assert edits[0]["raw"] != hold.raw

