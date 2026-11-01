from fractions import Fraction

from maichart.composer.stream_data import NOTE, StreamStep
from maichart.composer.stream_target_correspondence import corresponding_shift, rotated_text


def _note(index, text):
    start, end = Fraction(index), Fraction(index + 1)
    return StreamStep(NOTE, start, end, float(start), float(end), text)


def test_exact_prefix_is_trusted_without_rotation():
    authored = [_note(i, str(i + 1)) for i in range(3)]
    assert corresponding_shift(authored, authored) == 0


def test_observed_rotation_must_cover_multiple_aligned_notes():
    authored = [_note(i, str(i + 1)) for i in range(3)]
    actual = [_note(i, rotated_text(step.text, 2)) for i, step in enumerate(authored)]
    assert corresponding_shift(authored, actual) == 2
    assert corresponding_shift(authored[:1], actual[:1]) is None
    assert corresponding_shift(authored, actual[1:]) is None


def test_one_inconsistent_note_rejects_rotation():
    authored = [_note(i, str(i + 1)) for i in range(3)]
    actual = [_note(i, rotated_text(step.text, 2)) for i, step in enumerate(authored)]
    actual[1] = _note(1, '8')
    assert corresponding_shift(authored, actual) is None
