from fractions import Fraction

from maichart.event_timing import EventTempo, parse_event_chart_text
from maichart.form.cli import _export_end_timing
from maichart.generation.event_compile import compile_event_points


def test_generated_audio_end_is_rounded_up_on_export_grid_without_moving_notes():
    tempo = [EventTempo(Fraction(0), 185.005)]
    audio_end = Fraction(123456789, 1000000)
    note_points = parse_event_chart_text(
        '&title=Test\n&wholebpm=185.005\n&first=0\n&lv_5=13\n'
        '&inote_5=(185.005){4}1,E\n', 5,
    ).points

    export_end, metadata = _export_end_timing(audio_end, tempo, 0.)
    assert export_end == Fraction((audio_end * 1920).__ceil__(), 1920)
    assert export_end >= audio_end
    assert export_end - audio_end < Fraction(1, 1920)
    assert metadata['audio_end_beat'] == str(audio_end)
    assert metadata['export_end_beat'] == str(export_end)
    assert metadata['end_only_normalized'] is True
    assert metadata['notes_unchanged'] is True

    text, _ = compile_event_points(note_points, tempo_changes=tempo, first=0., end_beat=export_end)
    reparsed = parse_event_chart_text(text, 5)
    assert reparsed.points == note_points
    assert reparsed.end_beat == export_end


def test_revise_preserves_a_later_existing_chart_end_exactly():
    tempo = [EventTempo(Fraction(0), 120.)]
    audio_end = Fraction(1000000000000000001, 100000000000000000)
    existing_end = Fraction(50, 3)
    export_end, metadata = _export_end_timing(audio_end, tempo, 0., existing_end=existing_end)

    assert export_end == existing_end
    assert metadata['export_end_beat'] == str(existing_end)
    assert metadata['end_only_normalized'] is False
    assert metadata['notes_unchanged'] is True


def test_revise_uses_normalized_audio_end_when_existing_end_is_earlier():
    tempo = [EventTempo(Fraction(0), 120.)]
    audio_end = Fraction(10001, 1000)
    export_end, metadata = _export_end_timing(audio_end, tempo, 0., existing_end=Fraction(9))

    assert export_end == Fraction((audio_end * 1920).__ceil__(), 1920)
    assert metadata['end_only_normalized'] is True
    assert metadata['end_extension_sec'] >= 0


def test_generation_checkpoint_identity_matches_the_loaded_file(tmp_path):
    import hashlib
    from test_form_api import _checkpoint
    from maichart.form.checkpoint import load
    path = tmp_path/'source.pt'
    _checkpoint(path)
    _, _, _, payload = load(path,record_identity=True)
    identity = payload['source_checkpoint_identity']
    assert identity['path'] == str(path.resolve())
    assert identity['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity['size'] == path.stat().st_size
