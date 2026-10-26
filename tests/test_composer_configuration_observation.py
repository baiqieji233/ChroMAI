from fractions import Fraction

from maichart.composer.configuration_observation import configuration_observation
from maichart.event_timing import EventPoint
from scripts.audit_composer_repetition import audit_chart


def test_tap_lane_patterns_include_modifiers_but_do_not_bridge_chords_or_other_notes():
    points = [EventPoint(Fraction(i, 2), text) for i, text in enumerate(
        ['1', '8b', '1x', '8', '1/8', '1', 'C', '1', '1h[4:1]', '1'])]
    result = configuration_observation(points)
    text, taps = (result['repetition'][key] for key in ('exact_event_text', 'consecutive_single_tap_lanes'))
    assert text['counts']['ABAB'] == 0
    assert taps['windows'] == {'2': 3, '3': 2, '4': 1}
    assert taps['counts']['lanes_1_8_ABAB'] == 1
    assert taps['counts']['lane_1_same'] == 0


def test_half_beat_patterns_break_at_rests_without_changing_all_event_denominators():
    points = [EventPoint(Fraction(beat), text) for beat, text in
              [('0', '4'), ('1/2', '5'), ('1', '4'), ('3', '5')]]
    result = configuration_observation(points)['repetition']
    all_taps = result['consecutive_single_tap_lanes']
    rapid = result['single_tap_lanes_at_most_half_beat']
    assert all_taps['counts']['lanes_4_5_ABAB'] == 1
    assert rapid['windows'] == {'2': 2, '3': 1, '4': 0}
    assert rapid['counts']['lanes_4_5_ABAB'] == 0
    assert rapid['rates']['ABAB'] is None


def test_formal_chart_audit_excludes_tempo_and_end_directives_and_keeps_sidecar_identity(tmp_path):
    import hashlib
    import json
    path = tmp_path / 'maidata.txt'
    text = '&title=Pattern\n&wholebpm=120\n&inote_5=(120){8}1,8,1,8,,(180),E\n'
    path.write_bytes(text.replace('\n', '\r\n').encode('utf-8'))
    path.with_suffix('.composer.json').write_text(json.dumps(
        {'event_count': 4, 'optimizer_steps': 20500, 'action_temperature': .75,
         'chart_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}), encoding='utf-8')
    result = audit_chart(path, 5)
    assert result['sidecar_event_count_matches'] is True
    assert result['sidecar_chart_hash_matches'] is True
    assert result['sha256'] != result['normalized_text_sha256']
    assert result['generation_provenance']['action_temperature'] == .75
    assert result['configuration_observation']['repetition']['event_count'] == 4
    assert result['configuration_observation']['repetition']['consecutive_single_tap_lanes']['counts']['ABAB'] == 1


def test_empty_report_retains_existing_fields_and_reports_undefined_ratios_as_null():
    result = configuration_observation([])
    assert result['unique_texts'] == 0
    assert result['top_texts'] == []
    assert result['max_consecutive_same_text'] == 0
    assert result['repetition']['exact_event_text']['rates']['same'] is None
