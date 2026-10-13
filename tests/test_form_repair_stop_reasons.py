from fractions import Fraction
import json
from types import SimpleNamespace

from maichart.event_timing import EventPoint, EventTempo
from maichart.form import repair
from maichart.webui.form_api import update_form_progress


def test_partial_coverage_does_not_hide_exhausted_check_budget(monkeypatch):
    coverage = {'status': 'partial', 'parse_errors': [], 'unsupported': [{'raw': 'fixture'}]}
    issue = SimpleNamespace(severity='hard', action='enforce', conflict_type='fixture',
                            notes=[], repair_target=None)
    monkeypatch.setattr(repair, 'detect_chart_conflicts', lambda *_a, **_kw:
        SimpleNamespace(coverage=coverage, conflicts_detected=[issue]))
    result = repair.repair_chart([EventPoint(Fraction(0), '1')],
        [EventTempo(Fraction(0), 120.)], budget=repair.RepairBudget(1, 4, 30.))
    assert result.status == 'partial_coverage'
    assert result.to_dict()['coverage_status'] == 'partial'
    assert result.to_dict()['budget_exhausted'] is True
    task = SimpleNamespace(kind='form_generate', result={})
    update_form_progress(task, 'form_repair ' + json.dumps(result.to_dict()))
    assert task.result['repair_summary']['coverage_status'] == 'partial'
    assert task.result['repair_summary']['budget_exhausted'] is True


def test_zero_time_budget_records_the_stop_reason():
    result = repair.repair_chart([EventPoint(Fraction(0), '1')],
        [EventTempo(Fraction(0), 120.)], budget=repair.RepairBudget(24, 4, 0.))
    assert result.status == 'budget_exhausted'
    assert result.to_dict()['budget_exhausted'] is True
