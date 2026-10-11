"""Bounded transactions for sustained actions inherited from older windows.

No checkpoint/vector/layout changes. Source attribution is confirmed by a
counterfactual recheck, not by duration alone. Any change to a prior window
invalidates all later prefixes and density accounting. Failed transactions
restore the previous chart rather than gradually erasing its newer notes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
import json
import re

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.generation.event_model import EventSequence
from maichart.generation.post_process import (
    actionable, actionable_hard, detect_chart_conflicts, _extract,
)


def relevant_hard(result, floor_time):
    return sum(c.severity == "hard" and actionable(c) and c.time >= floor_time - 1e-6
               for c in result.conflicts_detected)


def replace_child(points, source, text):
    """Use parsed children to preserve slash groups, breaks and compact chords."""
    return [EventPoint(p.beat, text) if i == source else p
            for i, p in enumerate(points) if i != source or text]


def active_sustains(points, tempo, first, result, floor):
    events, texts, _ = _extract(points, tempo, first)
    floor_time = float(seconds_at_beat(floor, tempo, first))
    native = {tuple(a["source_id"]): a for a in result.coverage.get("sustained_actions", [])}
    selected = []
    for event in events:
        if event.beat >= floor or event.protected:
            continue
        marker = native.get(event.source_id)
        if marker:
            end = float(marker["end_sec"])
        elif event.note_type in {"hold", "touch_hold"}:
            end = event.time + event.duration
        elif event.trajectories:
            end = max(t.start_time + t.duration for t in event.trajectories)
        else:
            continue  # unsupported shapes must not acquire an invented tail
        if end + 3 / 180 < floor_time:
            continue
        selected.append((event, end))
    # Longest remaining occupation first; cap work, do not assume it is bad.
    selected.sort(key=lambda pair: (-pair[1], pair[0].beat, pair[0].source_id))
    return selected[:8], texts


def _child_replacement(points, texts, target, new_atom):
    text = "/".join(new_atom if key == target else value
                    for key, value in texts.items() if key[0] == target[0]
                    and (key != target or new_atom))
    return replace_child(points, target[0], text)


def bounded_hold_rewrite(event, floor, tempo, first):
    """Keep the head and end at an exact 1/16-beat grid before the boundary.

    Allow the upstream release delay. Only simple, positive holds are edited;
    explicit seconds and beat durations are both rewritten losslessly to grid.
    """
    if event.note_type not in {"hold", "touch_hold"} or event.duration <= 0:
        return None
    end = (floor * 16).__floor__() * Fraction(1, 16) - Fraction(1, 16)
    boundary_sec = seconds_at_beat(floor, tempo, first)
    for _ in range(16):
        if end <= event.beat:
            return None
        if boundary_sec - seconds_at_beat(end, tempo, first) > 3 / 180 + 1e-6:
            break
        end -= Fraction(1, 16)
    else:
        return None
    duration = end - event.beat
    block = f"[{duration.denominator * 4}:{duration.numerator}]"
    if not re.search(r"\[[^\]]+\]", event.raw):
        return None
    return re.sub(r"\[[^\]]+\]", block, event.raw, count=1)


@dataclass
class RewindPlan(Exception):
    results: list
    snapshot: list
    restart: int
    goal: int
    before_hard: int
    record: dict


class CrossWindowRepair:
    def __init__(self, windows, config, tempo, first, merge):
        self.windows, self.config, self.tempo, self.first, self.merge = windows, config, tempo, first, merge
        self.pending = None
        self.used = 0
        self.records = []
        self.attempted_sources = set()

    @property
    def seed_shift(self):
        return self.used * 10000019

    def consider(self, result, trial, prior, index):
        cfg = self.config
        if (not cfg or not cfg.enable_conflict_detection or cfg.conflict_repair_mode != "local"
                or self.pending is not None or not prior or self.used >= cfg.conflict_backtrack_budget
                or cfg.conflict_backtrack_windows == 0 and cfg.sustain_repair_mode == "off"
                or result.coverage.get("parse_errors")):
            return
        floor = max(trial.start_beat, prior[-1].end_beat)
        floor_time = float(seconds_at_beat(floor, self.tempo, self.first))
        count = relevant_hard(result, floor_time)
        if not count:
            return
        points = self.merge([*prior, trial], overlap_beats=1)
        all_events = _extract(points, self.tempo, self.first)[0]
        new_events = [e for e in all_events if e.beat >= floor]
        # A diagnostic may occur late on an old Slide without involving this
        # window at all. Rewind only when a current action is near that time.
        if not any(c.severity == "hard" and actionable(c) and c.time >= floor_time - 1e-6
                   and (any(n.beat >= floor for n in c.notes)
                        or any(e.time - .15 <= c.time <= e.time + max(e.duration, .15)
                               for e in new_events)) for c in result.conflicts_detected):
            return
        candidates, texts = active_sustains(points, self.tempo, self.first, result, floor)
        baseline = actionable_hard(result)
        depth = 2 if cfg.conflict_backtrack_windows == -1 else cfg.conflict_backtrack_windows
        for event, tail in candidates:
            key = (index, str(event.beat), event.raw)
            if key in self.attempted_sources:
                continue
            source_index = next((i for i, (lo, hi) in enumerate(self.windows) if lo <= event.beat < hi), None)
            if source_index is None or source_index >= index:
                continue
            # Removing one source for the comparison is not a committed edit.
            self.attempted_sources.add(key)
            counter = _child_replacement(points, texts, event.source_id, "")
            observed = detect_chart_conflicts(counter, self.tempo, self.first, cfg)
            if (observed.coverage.get("parse_errors") or relevant_hard(observed, floor_time) >= count
                    or actionable_hard(observed) >= baseline):
                continue
            record = {"id": len(self.records) + 1, "status": "pending", "trigger_window": index,
                      "source_window": source_index, "source_beat": str(event.beat),
                      "source_raw": event.raw, "source_end_sec": tail,
                      "hard_before": baseline, "counterfactual_hard": actionable_hard(observed)}
            # Optional source edit: only confirmed sources, preserve head time.
            atom, operation = None, None
            if cfg.sustain_repair_mode in {"hold", "hold_slide"}:
                atom = bounded_hold_rewrite(event, floor, self.tempo, self.first)
                if atom:
                    operation = "shorten_hold"
            if (atom is None and cfg.sustain_repair_mode == "hold_slide" and event.note_type == "slide"
                    and re.fullmatch(r"[1-8]b?[-<>^][1-8]\[[^\]]+\]", event.raw)):
                atom = event.raw.split("[", 1)[0][:1] + ("b" if event.raw[1:2] == "b" else "")
                operation = "slide_to_head_tap"
            if atom:
                edited = _child_replacement(points, texts, event.source_id, atom)
                verification = detect_chart_conflicts(edited, self.tempo, self.first, cfg)
                if (not verification.coverage.get("parse_errors") and actionable_hard(verification) < baseline
                        and relevant_hard(verification, floor_time) < count):
                    updated = []
                    replacement = next(p.note_text for p in edited if p.beat == event.beat)
                    for window in prior:
                        if window.start_beat != self.windows[source_index][0]:
                            updated.append(window)
                            continue
                        seq = EventSequence(window.sequence.beats[:],
                            [replacement if b + window.start_beat == event.beat else t
                             for b, t in zip(window.sequence.beats, window.sequence.note_texts)])
                        updated.append(replace(window, sequence=seq,
                            original_sequence=window.original_sequence or window.sequence))
                    record.update(operation=operation, replacement=atom, proposed_hard=actionable_hard(verification))
                    restart = source_index + 1
                    kept = [w for w in updated if w.start_beat < self.windows[restart][0]]
                    raise RewindPlan(kept, [*prior, trial], restart, index, baseline, record)
            if depth > 0 and index - source_index <= depth:
                record.update(operation="regenerate_source_window", max_windows=depth)
                kept = [w for w in prior if w.start_beat < self.windows[source_index][0]]
                raise RewindPlan(kept, [*prior, trial], source_index, index, baseline, record)
            record.update(status="unresolved", operation="outside_backtrack_limit")
            self.records.append(record)
            print("playability_cross_window " + json.dumps(record, ensure_ascii=False), flush=True)
        # Two long actions can occupy both hands: removing only one still
        # leaves a MultiTouch finding for a new chord. Check a bounded pair
        # instead of concluding that neither earlier source matters.
        for pair in combinations(candidates[:4], 2):
            key = (index, tuple((str(e.beat), e.raw) for e, _ in pair))
            if key in self.attempted_sources:
                continue
            self.attempted_sources.add(key)
            sources = [e.source_id for e, _ in pair]
            source_indices = [next((i for i, (lo, hi) in enumerate(self.windows) if lo <= e.beat < hi), -1)
                              for e, _ in pair]
            if min(source_indices) < 0:
                continue
            deleted_points = _replace_children(points, texts, {s: "" for s in sources})
            result_without = detect_chart_conflicts(deleted_points, self.tempo, self.first, cfg)
            if (result_without.coverage.get("parse_errors") or actionable_hard(result_without) >= baseline
                    or relevant_hard(result_without, floor_time) >= count):
                continue
            source_index = min(source_indices)
            record = {"id": len(self.records)+1, "status": "pending", "trigger_window": index,
                      "source_window": source_index, "source_beats": [str(e.beat) for e, _ in pair],
                      "source_raw": [e.raw for e, _ in pair], "hard_before": baseline,
                      "counterfactual_hard": actionable_hard(result_without)}
            atoms = {e.source_id: bounded_hold_rewrite(e, floor, self.tempo, self.first) for e, _ in pair}
            if (cfg.sustain_repair_mode != "off" and len(set(source_indices)) == 1 and all(atoms.values())):
                edited = _replace_children(points, texts, atoms)
                checked = detect_chart_conflicts(edited, self.tempo, self.first, cfg)
                if not checked.coverage.get("parse_errors") and actionable_hard(checked) < baseline:
                    changed_texts = {p.beat: p.note_text for p in edited}
                    updated = []
                    for w in prior:
                        if w.start_beat == self.windows[source_index][0]:
                            seq = EventSequence(w.sequence.beats[:],
                                [changed_texts.get(w.start_beat + b, t)
                                 for b,t in zip(w.sequence.beats, w.sequence.note_texts)])
                            updated.append(replace(w, sequence=seq, original_sequence=w.original_sequence or w.sequence))
                        else:
                            updated.append(w)
                    restart = source_index+1
                    record.update(operation="shorten_hold", source_count=2, replacements=list(atoms.values()))
                    raise RewindPlan([w for w in updated if w.start_beat < self.windows[restart][0]],
                                     [*prior, trial], restart, index, baseline, record)
            if depth > 0 and index-source_index <= depth:
                record.update(operation="regenerate_source_window", source_count=2, max_windows=depth)
                raise RewindPlan([w for w in prior if w.start_beat < self.windows[source_index][0]],
                                 [*prior, trial], source_index, index, baseline, record)

    def begin(self, plan):
        self.used += 1
        self.pending = plan
        self.records.append(plan.record)
        print("playability_cross_window " + json.dumps(plan.record, ensure_ascii=False), flush=True)
        return plan.results, plan.restart

    def finish(self, results, index):
        if not self.pending or index != self.pending.goal:
            return results
        plan = self.pending
        result = detect_chart_conflicts(self.merge(results, overlap_beats=1), self.tempo, self.first, self.config)
        remaining = actionable_hard(result)
        # Replaying a source is not an invitation to make the affected passage
        # empty. Keep at least 80% of the onset heads in the replayed interval;
        # a shortened Hold or simplified Slide retains its head and passes this.
        before = [p for p in self.merge(plan.snapshot, overlap_beats=1)
                  if p.beat >= self.windows[plan.restart][0]]
        after = [p for p in self.merge(results, overlap_beats=1)
                 if p.beat >= self.windows[plan.restart][0]]
        before_heads = len(_extract(before, self.tempo, self.first)[0])
        after_heads = len(_extract(after, self.tempo, self.first)[0])
        retained = before_heads == 0 or after_heads >= before_heads * 0.8
        accepted = not result.coverage.get("parse_errors") and remaining < plan.before_hard and retained
        plan.record.update(status="accepted" if accepted else "rolled_back", hard_after=remaining,
                           rebuilt_through_window=index, transaction=self.used,
                           before_heads=before_heads, after_heads=after_heads)
        if not retained:
            plan.record["reason"] = "replay_lost_too_many_onsets"
        print("playability_cross_window_done " + json.dumps(plan.record, ensure_ascii=False), flush=True)
        self.pending = None
        if not accepted:
            return plan.snapshot
        previous = {w.start_beat: w for w in plan.snapshot}
        return [replace(w, original_sequence=previous[w.start_beat].original_sequence
                        or previous[w.start_beat].sequence)
                if w.start_beat in previous and w.sequence != previous[w.start_beat].sequence else w
                for w in results]


def rebuild_feedback(results, tempo, first, merge):
    from maichart.generation.event_quality import parse_event_notes
    parsed = parse_event_notes(merge(results, overlap_beats=1), tempo, first)
    return [{"time_sec": float(p.time_sec), "heads": len(p.notes)} for p in parsed.points if p.notes]


def _replace_children(points, texts, atoms):
    changed = {source[0] for source in atoms}
    rewritten = {}
    for index in changed:
        rewritten[index] = "/".join(atoms.get(key, text) for key, text in texts.items()
                                    if key[0] == index and atoms.get(key, text))
    return [EventPoint(p.beat, rewritten[i]) if i in changed else p
            for i,p in enumerate(points) if i not in changed or rewritten[i]]
