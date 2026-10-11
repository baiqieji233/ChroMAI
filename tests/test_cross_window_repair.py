"""Small deterministic probes: exact source repair, bounded replay and rollback."""
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventPoint as P, EventTempo as T
from maichart.generation import event_cli, miacode_native
from maichart.generation.conditions import GenerationCondition
from maichart.generation.cross_window_repair import CrossWindowRepair, RewindPlan, rebuild_feedback
from maichart.generation.event_model import EventVocabulary, EventSequence
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts

TEMPO = [T(0, 120)]


def cfg(**kw):
    return PostProcessConfig(True, conflict_repair_mode="local", **kw)


def window(start, beats, texts):
    return event_cli._DecodedWindow(EventSequence([F(b) for b in beats], texts), F(start), F(start + 16), True)


def test_shortening_old_hold_preserves_the_new_chord_and_original():
    prior = [window(0, [15], ["1h[4:12]"])]
    trial = window(16, [1], ["1/2"])
    config = cfg(sustain_repair_mode="hold", conflict_backtrack_windows=0)
    merged = event_cli._merge_windows(prior + [trial])
    report = detect_chart_conflicts(merged, TEMPO, 0, config)
    control = CrossWindowRepair([(F(0), F(16)), (F(16), F(32))], config, TEMPO, 0, event_cli._merge_windows)
    with pytest.raises(RewindPlan) as captured:
        control.consider(report, trial, prior, 1)
    plan = captured.value
    results, restart = control.begin(plan)
    assert restart == 1
    assert results[0].sequence.note_texts != prior[0].sequence.note_texts
    assert results[0].original_sequence.note_texts == ["1h[4:12]"]
    assert prior[0].sequence.note_texts == ["1h[4:12]"]
    final = control.finish(results + [trial], 1)
    assert final[-1].sequence.note_texts == ["1/2"]
    assert control.records[-1]["status"] == "accepted"


def test_report_only_does_not_modify_or_reject_physical_overlap():
    config = cfg(conflict_kind_modes={"Overlap": "report"}, sustain_repair_mode="hold", conflict_backtrack_windows=-1)
    prior, trial = [window(0, [15], ["1h[4:12]"])], window(16, [1], ["1/2"])
    report = detect_chart_conflicts(event_cli._merge_windows(prior + [trial]), TEMPO, 0, config)
    assert report.conflicts_detected[0].severity == "hard"
    assert report.to_dict()["remaining_actionable_hard"] == 0
    control = CrossWindowRepair([(F(0), F(16)), (F(16), F(32))], config, TEMPO, 0, event_cli._merge_windows)
    control.consider(report, trial, prior, 1)
    assert control.records == []


def run_windows(monkeypatch, *, fix_source=True, **config_kw):
    calls, prefixes = [], []
    vocab = EventVocabulary.from_event_charts([], segment_beats=48)
    monkeypatch.setattr(event_cli, "audio_segment_tensor", lambda *a, **kw: torch.zeros((2, 8)))
    def prefix(_vocab, results, *a):
        prefixes.append([w.sequence.note_texts[:] for w in results])
        return []
    monkeypatch.setattr(event_cli, "_context_prefix", prefix)
    def decode(*a, **kw):
        seed = kw["seed"]
        index = (seed % 10000019) - 42
        calls.append((index, seed))
        if index == 0:
            return EventSequence([F(15)], ["1h[4:1]" if seed > 10000019 and fix_source else "1h[4:12]"])
        return EventSequence([F(1)], ["1/2"])
    monkeypatch.setattr(event_cli, "_decode", decode)
    config = cfg(conflict_backtrack_windows=-1, conflict_backtrack_budget=1, **config_kw)
    results = event_cli._decode_windows(object(), vocab, None, None, GenerationCondition(), torch.device("cpu"),
        windows=[(F(0), F(16)), (F(16), F(32))], tempo_changes=TEMPO,
        first=0, song_duration=16, bpm=120, max_tokens=512, temperature=.8, sample=True,
        seed=42, context_beats=16, context_tokens=64, post_process_config=config)
    return results, calls, prefixes


def test_scheduler_rebuilds_prefix_and_retains_chord(monkeypatch, capsys):
    results, calls, prefixes = run_windows(monkeypatch)
    assert [i for i, _ in calls] == [0, 1, 0, 1]
    assert prefixes[-1] == [["1h[4:1]"]]
    assert results[-1].sequence.note_texts == ["1/2"]
    assert results[-1].playability["cross_window"][-1]["status"] == "accepted"
    assert "playability_cross_window_done" in capsys.readouterr().out
    feedback = rebuild_feedback(results, TEMPO, 0, event_cli._merge_windows)
    assert [x["heads"] for x in feedback] == [1, 2]


def test_failed_replay_restores_original_and_respects_budget(monkeypatch):
    results, calls, _ = run_windows(monkeypatch, fix_source=False)
    assert len(calls) == 4
    assert results[0].sequence.note_texts == ["1h[4:12]"]
    assert results[-1].sequence.note_texts == ["1/2"]
    assert results[-1].playability["cross_window"][-1]["status"] == "rolled_back"


@pytest.mark.skipif(not miacode_native.library_path().is_file(), reason="native library required")
def test_native_prior_hold_and_new_double_tap_are_resolved_at_source():
    config = cfg(conflict_engine="miacode", sustain_repair_mode="hold")
    prior, trial = [window(0, [15], ["1h[4:12]"])], window(16, [1], ["3/6"])
    report = detect_chart_conflicts(event_cli._merge_windows(prior + [trial]), TEMPO, 0, config)
    assert any(i.conflict_type == "miacode_MultiTouch" for i in report.conflicts_detected)
    control = CrossWindowRepair([(F(0), F(16)), (F(16), F(32))], config, TEMPO, 0, event_cli._merge_windows)
    with pytest.raises(RewindPlan) as captured:
        control.consider(report, trial, prior, 1)
    changed, _ = control.begin(captured.value)
    final = control.finish(changed + [trial], 1)
    assert final[-1].sequence.note_texts == ["3/6"]
    assert control.records[-1]["operation"] == "shorten_hold"
    assert control.records[-1]["status"] == "accepted"


@pytest.mark.skipif(not miacode_native.library_path().is_file(), reason="native library required")
def test_two_long_holds_are_attributed_as_a_pair():
    config = cfg(conflict_engine="miacode", sustain_repair_mode="hold")
    prior, trial = [window(0, [15], ["1h[4:12]/5h[4:12]"])], window(16, [1], ["3/7"])
    report = detect_chart_conflicts(event_cli._merge_windows(prior+[trial]), TEMPO, 0, config)
    control = CrossWindowRepair([(F(0),F(16)),(F(16),F(32))],config,TEMPO,0,event_cli._merge_windows)
    with pytest.raises(RewindPlan) as captured:
        control.consider(report,trial,prior,1)
    changed,_=control.begin(captured.value)
    final=control.finish(changed+[trial],1)
    assert final[-1].sequence.note_texts == ["3/7"]
    assert control.records[-1]["status"] == "accepted"
    assert control.records[-1]["source_count"] == 2
