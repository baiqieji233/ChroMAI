"""Map upstream MiaCode findings back to exact generated event children."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from maichart.event_timing import EventPoint, EventTempo, tokenize_event_timing
from maichart.generation.conflict_detector import ConflictReport
from maichart.generation.event_compile import compile_event_points
from maichart.generation.miacode_native import analyze_inote, REFERENCE_COMMIT
from maichart.generation.event_quality import parse_event_notes
from maichart.maidata import parse_maidata_metadata
from maichart.notes import BASIC_COMPACT_CHORD_RE


def _native_text(note):
    # MiaCode's lenient parser can skip compact break chords (e.g. 12b).
    # Expand only this equivalent spelling for analysis, keeping child IDs,
    # modifiers, timings and the user's exported text unchanged.
    if note.note_type == "tap" and BASIC_COMPACT_CHORD_RE.fullmatch(note.raw):
        match = BASIC_COMPACT_CHORD_RE.fullmatch(note.raw)
        return str(note.button if hasattr(note, "button") else note.position) + match["modifiers"]
    return note.raw


@lru_cache(maxsize=16)
def _analyze_exact(point_keys, tempo_keys, first):
    points = [EventPoint(beat, text) for beat, text in point_keys]
    tempo = [EventTempo(beat, bpm) for beat, bpm in tempo_keys]
    parsed = parse_event_notes(points, tempo, first)
    if any(item.get("code") != "duplicate_button_head" for item in parsed.fatal_errors):
        raise ValueError("MiaCode 输入无法无损解析：" + str(parsed.fatal_errors[:3]))
    points = [EventPoint(p.beat, "/".join(_native_text(n) for n in p.notes)) for p in parsed.points]
    text, _ = compile_event_points(points, tempo_changes=tempo, first=first)
    inote = parse_maidata_metadata(text).fields["inote_5"]
    # One comma cell per line gives an unambiguous event source anchor even
    # across sparse rhythm grids and variable tempo. Durations are untouched.
    timing = tokenize_event_timing(inote, tempo[0].bpm, first=first)
    cells = ["".join(cell.raw.split()) for cell in timing]
    return analyze_inote(",\n".join(cells), first=first), timing, cells


def detect_native_conflicts(points, tempo_changes, first, events, config):
    raw, timing, cells = _analyze_exact(
        tuple((str(p.beat), p.note_text) for p in points),
        tuple((str(t.beat), float(t.bpm)) for t in tempo_changes), float(first))
    by_beat = defaultdict(list)
    for event in events:
        by_beat[event.beat].append(event)

    def locate(item):
        line, col = int(item.get("line", 0)), int(item.get("col", 0))
        if not 1 <= line <= len(timing):
            return None
        candidates = by_beat.get(timing[line - 1].beat, [])
        if len(candidates) == 1:
            return candidates[0]
        # Whole raw child spans include synthetic slide heads and all branches.
        text = cells[line - 1]
        note_text = timing[line - 1].note_text
        cursor = len(text) - len(note_text) if text.endswith(note_text) else 0
        note_start = cursor
        for event in candidates:
            atom = _native_text(event)
            start = text.find(atom, cursor)
            if start < 0:
                start = text.find(atom, note_start)
            if start >= 0 and start <= col - 1 < start + len(atom):
                return event
            if start >= 0:
                cursor = start + len(atom)
        return None

    # Diagnostic line/col can point at the cause; marker_key identifies the
    # affected note. Prefer that stable identity for repair, then the anchor.
    markers = {m["key"]: m for m in raw.get("markers", [])}
    for marker in raw.get("marker_states", []):
        key = marker["marker_key"]
        markers[key] = {**markers.get(key, {}), **marker}
    references = raw.get("static_references", [])
    for ref in references:
        for role in ("affected", "cause"):
            marker = ref.get(role) or {}
            if marker.get("marker_key"):
                markers[marker["marker_key"]] = marker
    issues, unmapped, seen = [], [], set()
    findings = [(d, "runtime") for d in raw.get("diagnostics", [])]
    findings += [({**ref, **ref.get("affected", {}),
                  "detail": f"MiaCode {ref['kind']} 判定区参考"}, "static") for ref in references]
    for finding, stage in findings:
        hard = finding.get("alert_level") == "Muri"
        if hard and not config.detect_hard_conflicts or not hard and not config.detect_soft_conflicts:
            continue
        key = finding.get("marker_key", "")
        kind = finding["kind"]
        marker = markers.get(key) or finding
        affected = locate(marker)
        anchor = locate(finding)
        cause = locate(finding.get("cause") or {})
        notes = []
        for event in (affected, anchor, cause):
            if event is not None and all(event.source_id != n.source_id for n in notes):
                notes.append(event)
        # Keep runtime and static evidence in raw; avoid counting/repairing the
        # same note+kind twice just because both stages detected it.
        identity = (kind, affected.source_id if affected else key)
        if identity in seen:
            continue
        seen.add(identity)
        if not notes:
            unmapped.append({"kind": kind, "line": finding.get("line"), "col": finding.get("col")})
        target = None
        if hard and affected is not None and not affected.protected and kind != "MultiTouch":
            target = affected.source_id
        issues.append(ConflictReport(True, "miacode_" + kind,
            float(finding.get("second", 0)), notes, "hard" if hard else "soft",
            finding.get("detail") or f"MiaCode {kind}", target,
            evidence=f"miacode_{stage}:{REFERENCE_COMMIT}"))
    represented = {event.source_id for marker in raw.get("markers", [])
                   if (event := locate(marker)) is not None}
    for event in events:
        if event.source_id not in represented:
            unmapped.append({"source": list(event.source_id), "raw": event.raw,
                             "reason": "native_parser_missing_note"})
    errors = list(raw.get("parse", {}).get("errors", []))
    if any(item.get("reason") == "native_parser_missing_note" for item in unmapped):
        errors.append({"message": "MiaCode 未完整解析输入音符，不能判为检测通过"})
    if not raw.get("ok") and not errors:
        errors.append({"message": "MiaCode parser did not complete"})
    coverage = {"status": "partial" if errors or unmapped else "engine_reference",
                "engine": "miacode_native", "reference": REFERENCE_COMMIT,
                "in_process": True, "unsupported": unmapped, "parse_errors": errors,
                "elapsed_sec": raw["elapsed_sec"], "engine_reference_only": True,
                "playable_verified": False,
                "limitations": ["使用 MiaCode 原生判定区、Slide/Wifi、Touch 联动与动作参考；仍不等同于实机玩家认证。"]}
    # Compact diagnostic evidence keeps window reports small; detailed runtime
    # marker states remain available through analyze_maidata for investigations.
    coverage["diagnostics"] = raw.get("diagnostics", [])
    coverage["static_references"] = references
    return issues, coverage
