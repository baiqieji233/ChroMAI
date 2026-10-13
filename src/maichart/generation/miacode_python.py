"""In-process Python port of the pinned MiaCode Muri judge.

No shared-library loading, executable launching or build step. Parser, static
references and 180 Hz runtime actions retain the generation report contract.
Source port: Team-MiaCode/MiaCode 6573efb (MIT). Numerical path assets retain
their upstream scope; see THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from collections import Counter
import math
import time

from .muri.data import REFERENCE_COMMIT, reference_data
from .muri.model import head_key
from .muri.parser import parse_markers
from .muri.runtime import simple_notes, judge_simple, simulate_actions
from .muri.slide import slide_states, finish_slides
from .muri.static import static_references
from .playability_policy import validate_tap_on_slide_threshold_ms

ENGINE = "miacode_python"
IMPLEMENTATION = "python"


def analyze_inote(inote: str, *, first: float = 0.0,
                  tap_on_slide_threshold_ms: float = 200.0) -> dict:
    if not isinstance(inote, str) or "\0" in inote or not math.isfinite(float(first)):
        raise ValueError("Invalid MiaCode input/offset")
    threshold = validate_tap_on_slide_threshold_ms(tap_on_slide_threshold_ms)
    started = time.perf_counter()
    reference_data()  # Fail explicitly if packaged assets are absent/corrupt.
    markers, errors, warnings = parse_markers(inote, float(first))
    notes, groups = simple_notes(markers)
    diagnostics = judge_simple(markers, notes, groups, threshold / 1000.)
    states = slide_states(markers)
    diagnostics.extend(simulate_actions(markers, notes, states))
    slide_diagnostics, state_map = finish_slides(states)
    diagnostics.extend(slide_diagnostics)
    diagnostics.sort(key=lambda d: (d["second"], d["line"], d["col"]))
    # Upstream dense Overlap diagnostics retain one finding per judge tick.
    seen, retained = set(), []
    for item in diagnostics:
        stamp = round(item["second"] * 1e6)
        if item["kind"] == "Overlap":
            if stamp in seen:
                continue
            seen.add(stamp)
        retained.append(item)
    diagnostics = retained
    references = static_references(markers, threshold / 1000.)
    marker_states = [dict(value, marker_key=m["key"], marker_type=m["type"],
                          line=m["line"], col=m["col"], second=m["second"],
                          end_second=m["end_second"])
                     for m in markers if (value := state_map.get(m["key"])) is not None]
    marker_states.sort(key=lambda m: (m["second"], m["line"], m["col"], m["marker_key"]))
    kinds = Counter("break" if (m.get("track_break") if m["type"] in {"slide", "wifi"} else m.get("is_break"))
                    else "hold" if m["type"] == "touch_hold" else m["type"] for m in markers)
    heads = {head_key(m): m for m in markers if m["type"] in {"slide", "wifi"} and m.get("has_head_star")}
    helper_break = sum(bool(m.get("head_break")) for m in heads.values())
    summary = {
        "note_count": len(markers) + len(heads), "marker_count": len(markers),
        "tap_count": kinds["tap"] + len(heads) - helper_break, "hold_count": kinds["hold"],
        "slide_count": kinds["slide"] + kinds["wifi"],
        "touch_count": kinds["touch"],
        "break_count": kinds["break"] + helper_break,
        "helper_head_tap_count": len(heads) - helper_break, "helper_head_break_count": helper_break,
        "diagnostic_count": len(diagnostics), "marker_state_count": len(marker_states),
        "early_cleared_count": sum(s["early_cleared"] for s in marker_states),
        "counts_by_kind": dict(Counter(d["kind"] for d in diagnostics)),
        "static_reference_count": len(references),
        "static_reference_counts_by_kind": dict(Counter(d["kind"] for d in references)),
    }
    visible = []
    seen = set()
    for is_static, items in ((False, diagnostics), (True, references)):
        for item in items:
            affected = item.get("affected", item)
            identity = item["kind"], affected["line"], affected["col"]
            if identity in seen:
                continue
            seen.add(identity)
            visible.append(dict(kind=item["kind"], is_static=is_static,
                                line=affected["line"], col=affected["col"],
                                second=affected["second"], occurrence_second=item.get("second", affected["second"]),
                                detail_kind=item.get("detail_kind", item["kind"]),
                                detail_args=item.get("detail_args", {}), detail=item.get("detail", "")))
    # Retain the original diagnostic JSON's public segment field names.
    for m in markers:
        for field in ("keys", "durations", "shoot_seconds", "pad_enter_times", "critical_proportions"):
            m[f"segment_{field}"] = m[f"slide_segment_{field}"]
    return {
        "ok": not errors, "source": "maichart-memory", "difficulty": "selected-inote",
        "first": float(first), "tap_on_slide_threshold_ms": threshold,
        "parse": {"ok": not errors, "error_count": len(errors), "warning_count": len(warnings),
                  "errors": errors, "warnings": warnings},
        "summary": summary, "diagnostics": diagnostics, "static_references": references,
        "visible_panel_entries": visible, "marker_states": marker_states, "markers": markers,
        "engine": ENGINE, "implementation": IMPLEMENTATION, "reference_commit": REFERENCE_COMMIT,
        "in_process": True, "dynamic_dependency": False,
        "engine_reference_only": True, "playable_verified": False,
        "elapsed_sec": time.perf_counter() - started,
    }


def analyze_maidata(text: str, *, difficulty: int = 5, first: float | None = None,
                    tap_on_slide_threshold_ms: float = 200.0) -> dict:
    from maichart.maidata import parse_maidata_metadata, resolve_chart_first
    chart = parse_maidata_metadata(text)
    inote = chart.fields.get(f"inote_{difficulty}")
    if inote is None:
        raise ValueError(f"No inote_{difficulty} in maidata")
    if chart.wholebpm:
        inote = f"({float(chart.wholebpm):.12g})" + inote
    raw_first = resolve_chart_first(chart, difficulty)
    return analyze_inote(inote, first=float(raw_first or 0) if first is None else first,
                         tap_on_slide_threshold_ms=tap_on_slide_threshold_ms)
