from fractions import Fraction

import pytest

from maichart.event_timing import parse_event_chart
from maichart.training.music_plan_targets import MusicPlanTargets


def _chart(tmp_path, notes):
    path = tmp_path / "maidata.txt"
    path.write_text(f"&title=Plan\n&wholebpm=120\n&first=0\n&lv_5=13\n&inote_5={notes}\n", encoding="utf-8")
    return parse_event_chart(path, 5, strict_notes=True)


def test_chart_workload_changes_with_local_notes_not_audio_volume(tmp_path):
    chart = _chart(tmp_path, "(120){4}1,,,2,{8}12,34,56,78,12,34,56,78,E")
    targets = MusicPlanTargets(chart)
    easy = targets.window(Fraction(0), Fraction(4))
    dense = targets.window(Fraction(4), Fraction(8))
    assert easy.shape == dense.shape == (9,)
    assert dense[1] > easy[1]
    assert dense[2] > easy[2]
    assert dense[7] > easy[7]
    assert dense[8] == pytest.approx(1 / 3)


def test_plan_describes_touch_hold_and_slide_separately(tmp_path):
    chart = _chart(tmp_path, "(120){4}A1,Ch[4:1],1-5[4:1],2h[4:1],E")
    plan = MusicPlanTargets(chart).window(Fraction(0), Fraction(4))
    assert 0 < plan[3] < 1
    assert 0 < plan[5] < 1
    assert 0 < plan[6] < 1
    assert plan[8] == 1


def test_single_touch_does_not_label_a_whole_plain_window_touch(tmp_path):
    chart = _chart(tmp_path, "(120){4}A1,1,2,3,4,5,6,7,E")
    plan = MusicPlanTargets(chart).window(Fraction(0), Fraction(8))
    assert 0 < plan[5] < 0.2
    assert plan[8] == 0
