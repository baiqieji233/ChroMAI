"""Zero-human-labor segment-level physical, acoustic & maimai DX domain corpus audit.

Scans a training manifest on CPU and emits ``auto_quality_annotations.json`` plus
``auto_quality_audit_report.json`` without ever modifying manual
``quality_annotations.json``.

Two-tier annotation priority:
1. **Priority 1 — Problem Exclusion & Downweighting (``label="problem"`` / ``"excluded"``)**:
   * **Causal Culprit Span Physical Muri** (``MultiTouch``, ``TouchMultiAction``,
     ``Overlap``, configurable ``TapOnSlide`` & optional ``SlideTooFast``/``SlideHeadTap``):
     traces back to offending Hold/Slide launch beats so locking culprit actions are
     masked alongside collisions without over-censoring clean downstream measures.
   * **Level-Stratified Chirality Hand-Flow** (``--experimental-hand-flow-quantile``):
     calibrates ``comfort_v2`` quantiles per maimai DX level bucket (``12 / 13 / 14+``)
     so awkward low-level arm twists are caught without penalizing fast high-level streams.

2. **Priority 2 — Strict Multi-Gate Worthy Phrase Highlighting (``label="good"``)**:
   * Subordinate to all problem/warning gates: only sustain-safe 8–16 beat phrases with
     zero hard/soft conflicts, high level-relative ``chirality_comfort``,
     and varied maimai DX mechanics receive positive boost weights.
     The acoustic mode also requires verified onset support.

The report also locates short event gaps, unusual beat differences, rapid repeats,
suspected global and local acoustic offsets, and possible incomplete spans for review.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable
from fractions import Fraction
import hashlib
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from maichart.composer_v2 import chart_dialect
from maichart.composer_v2.data import _beats_at_seconds
from maichart.composer_v2.hand_flow import PrecomputedMotionStream, build_motion_stream, chirality_comfort
from maichart.composer.training_lock import TrainingOutputBusyError, _directory_lock
from maichart.event_timing import _parse_event_metadata
from maichart.generation.event_quality import parse_event_notes
from maichart.generation.muri.touch_reachability import (
    DEFAULT_COHOLD_RADIUS,
    TOUCH_REACHABILITY_POLICIES,
    validate_touch_reachability,
)
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts
from maichart.maidata import decode_text_with_encoding, parse_maidata_metadata
from maichart.tempo_clock import TempoClock
from maichart.training.dataset_catalog import _write_json
from maichart.training.event_dataset import numeric_level
from maichart.training.quality_annotations import (
    AUTO_ANNOTATION_FILENAME,
    AUTO_ANNOTATION_PREVIEW_FILENAME,
    COMPOSED_AUTO_ANNOTATION_FILENAME,
    CatalogChart,
    _auto_annotation_owner,
    annotation_path,
    annotations_fingerprint,
    audio_sha256,
    auto_annotation_path,
    auto_audit_report_path,
    iter_catalog,
    resolve_auto_audit,
    save_auto_annotations,
    split_annotation_source,
)

AUTO_QUALITY_POLICIES = ("auto", "on", "off")
AUDIT_LOCK_FILENAME = ".composer-v2-audit.lock"
TAP_ON_SLIDE_MODES = ("mask_hard", "mask_all", "allow")
INNER_MURI_MODES = ("allow", "mask_hard")
DOMAIN_PATHOLOGY_MODES = ("maimai_dx_v1", "off")
ACOUSTIC_AUDIT_MODES = ("auto", "on", "off")
HIGHLIGHT_GOOD_MODES = ("auto", "acoustic_and_flow_v1", "chart_only_v1", "off")

HARD_MULTI_TYPES = frozenset(
    {
        "miacode_MultiTouch",
        "miacode_TouchMultiAction",
        "miacode_Overlap",
    }
)
INNER_MURI_TYPES = frozenset(
    {
        "miacode_SlideTooFast",
        "miacode_SlideHeadTap",
    }
)

# Legitimate binary, ternary, and standard tuplet sub-beat denominators in maimai DX.
STANDARD_BEAT_DENOMINATORS = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 48, 64, 96}
)

MAX_CULPRIT_LOOKBACK_BEATS = 6.0
GOOD_TIMING_WEIGHT = 1.20
GOOD_COMPONENT_WEIGHT = 1.35


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Target training manifest JSON")
    p.add_argument(
        "--tap-on-slide-mode",
        choices=TAP_ON_SLIDE_MODES,
        default="mask_hard",
        help="How to treat TapOnSlide collisions in training charts (default: mask_hard)",
    )
    p.add_argument(
        "--tap-on-slide-ms",
        type=float,
        default=200.0,
        help="TapOnSlide detection window in milliseconds (0-250 ms)",
    )
    p.add_argument(
        "--inner-muri-mode",
        choices=INNER_MURI_MODES,
        default="allow",
        help="How to treat inner-screen SlideTooFast/SlideHeadTap pre-triggers (default: allow)",
    )
    p.add_argument(
        "--touch-reachability-policy",
        choices=TOUCH_REACHABILITY_POLICIES,
        default="palm_span_v1",
        help="Touch 2-hand reachability policy during audit (default: palm_span_v1)",
    )
    p.add_argument(
        "--touch-cohold-radius",
        type=float,
        default=DEFAULT_COHOLD_RADIUS,
        help="Single-hand palm/forearm co-hold span radius",
    )
    p.add_argument(
        "--domain-pathology-mode",
        choices=DOMAIN_PATHOLOGY_MODES,
        default="maimai_dx_v1",
        help="Report possible transcription and rapid-repeat issues for review (default: maimai_dx_v1)",
    )
    p.add_argument(
        "--acoustic-audit-mode",
        choices=ACOUSTIC_AUDIT_MODES,
        default="auto",
        help="Check severe global offset and report local offsets or possible incomplete spans (default: auto)",
    )
    p.add_argument(
        "--highlight-good-mode",
        choices=HIGHLIGHT_GOOD_MODES,
        default="off",
        help="Opt in to phrase reweighting (off by default; legacy auto also resolves to off)",
    )
    p.add_argument(
        "--good-quantile",
        type=float,
        default=0.82,
        help="Per-level-tier quantile cutoff in [0.50, 0.98] for highlighting worthy phrases (default: 0.82)",
    )
    p.add_argument(
        "--problem-timing-weight",
        type=float,
        default=None,
        help="Explicit timing loss weight override on hard-Muri spans (omit for smart layout/timing decoupling)",
    )
    p.add_argument(
        "--max-problem-fraction",
        type=float,
        default=0.45,
        help="Exclude whole chart when masked problem duration exceeds this fraction of chart length",
    )
    p.add_argument(
        "--experimental-hand-flow-quantile",
        type=float,
        default=0.0,
        help="Downweight configuration in the lowest q quantile of level-stratified 4-beat comfort windows (0 = off)",
    )
    p.add_argument(
        "--anchor-manifest",
        default=None,
        help="Optional reference manifest (e.g. official charts) to calibrate hand-flow & good quantiles",
    )
    p.add_argument(
        "--anchor-dataset-id",
        default=None,
        help="Optional source_dataset_id / name within --manifest to calibrate quantiles",
    )
    p.add_argument("--min-level", type=float, default=12.0)
    p.add_argument("--max-level", type=float, default=15.0)
    p.add_argument("--min-difficulty-index", type=int, default=5)
    p.add_argument("--max-difficulty-index", type=int, default=6)
    p.add_argument("--limit-charts", type=int, default=0, help="Preview at most this many charts; 0 writes the full audit")
    p.add_argument("--workers", type=int, default=0, help="Parallel CPU worker processes (0 = auto, 1 = in-process)")
    p.add_argument("--report-output", default=None, help="Optional explicit path for audit summary JSON")
    return p


def _level_tier(level_val: float) -> str:
    if level_val < 13.0:
        return "lv12"
    if level_val < 14.0:
        return "lv13"
    return "lv14+"


def _merge_intervals(intervals: list[tuple[float, float]], *, merge_gap: float = 0.05) -> list[tuple[float, float]]:
    valid = sorted((float(lo), float(hi)) for lo, hi in intervals if hi > lo + 1e-5)
    if not valid:
        return []
    merged = [valid[0]]
    for lo, hi in valid[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + merge_gap:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _subtract_intervals(
    intervals: list[tuple[float, float]],
    cutouts: list[tuple[float, float]],
    *,
    min_width: float = 0.05,
) -> list[tuple[float, float]]:
    pieces = list(intervals)
    for cut_lo, cut_hi in cutouts:
        nxt: list[tuple[float, float]] = []
        for lo, hi in pieces:
            if cut_hi <= lo + 1e-9 or cut_lo >= hi - 1e-9:
                nxt.append((lo, hi))
                continue
            if cut_lo - lo >= min_width:
                nxt.append((lo, cut_lo))
            if hi - cut_hi >= min_width:
                nxt.append((cut_hi, hi))
        pieces = nxt
        if not pieces:
            break
    return pieces


def _should_mask_issue(
    issue: Any,
    *,
    tap_on_slide_mode: str,
    inner_muri_mode: str,
) -> bool:
    kind = str(issue.conflict_type)
    sev = str(issue.severity)
    if kind in HARD_MULTI_TYPES:
        return sev == "hard"
    if kind == "miacode_TapOnSlide":
        if tap_on_slide_mode == "allow":
            return False
        if tap_on_slide_mode == "mask_all":
            return True
        return sev == "hard"
    if kind in INNER_MURI_TYPES:
        if inner_muri_mode == "mask_hard":
            return sev == "hard"
        return False
    return False


def _causal_conflict_beat_span(
    issue: Any,
    clock: TempoClock,
    total_beats: float,
) -> tuple[float, float]:
    """Bound a conflict from its causal culprit launch beat through collision resolution.

    Unlike a blind symmetric ``[b - 2, b + 2]`` window, tracing ``issue.notes`` back
    up to ``MAX_CULPRIT_LOOKBACK_BEATS`` masks the slow Slide or locking Hold launch
    that created the trap, while keeping the post-conflict margin tight (``+0.75`` beat)
    so clean downstream phrases immediately resume supervision.
    """
    b_conflict = float(_beats_at_seconds(clock, [float(issue.time)])[0])
    note_beats = [
        float(n.beat)
        for n in (getattr(issue, "notes", None) or ())
        if getattr(n, "beat", None) is not None
    ]
    if note_beats:
        earliest_culprit = max(b_conflict - MAX_CULPRIT_LOOKBACK_BEATS, min(note_beats) - 0.5)
        b_lo = max(0.0, min(b_conflict - 1.0, earliest_culprit))
        near_after = [nb for nb in note_beats if nb <= b_conflict + 1.0]
        anchor_hi = max([b_conflict, *near_after])
    else:
        b_lo = max(0.0, b_conflict - 1.5)
        anchor_hi = b_conflict
    b_hi = min(total_beats, max(b_lo + 0.5, anchor_hi + 0.75))
    return b_lo, b_hi


def _default_issue_timing_weight(issue: Any, points: list[Any], b_lo: float, b_hi: float) -> float:
    """Decouple timing vs layout supervision when ``--problem-timing-weight`` is omitted.

    ``Overlap`` (same-pad collision) or erratic subdivisions mask timing to ``0.0``;
    pure layout collisions (``TouchMultiAction``, ``TapOnSlide``) on clean musical grids
    retain ``0.5`` timing weight so valid chorus rhythms are not thrown away.
    """
    kind = str(issue.conflict_type)
    if kind == "miacode_Overlap":
        return 0.0
    span_beats = [p.beat for p in points if b_lo <= float(p.beat) <= b_hi]
    if span_beats and all(
        (right - left).denominator in STANDARD_BEAT_DENOMINATORS
        for left, right in zip(span_beats, span_beats[1:])
    ):
        return 0.5
    return 0.0


def _load_normalized_chart(
    row: CatalogChart,
    *,
    metadata_cache: dict[Path, Any] | None = None,
) -> tuple[Any, TempoClock, float] | None:
    path = Path(row.source_path)
    metadata = metadata_cache.get(path) if metadata_cache is not None else None
    if metadata is None:
        raw_bytes = path.read_bytes()
        text, _ = decode_text_with_encoding(raw_bytes)
        metadata = parse_maidata_metadata(text)
        if metadata_cache is not None:
            metadata_cache[path] = metadata
    chart = _parse_event_metadata(metadata, row.difficulty_index, strict_notes=False)
    chart, _ = chart_dialect.normalize_training_chart(chart)
    if not chart.points:
        return None
    clock = TempoClock(chart.tempo_changes, chart.audio_first)
    end_beat_val = float(Fraction(str(chart.end_beat)))
    max_point_beat = max(float(p.beat) for p in chart.points)
    total_beats = max(end_beat_val, max_point_beat + 1.0)
    duration_sec = max(float(clock.at(Fraction(math.ceil(total_beats)))), 1.0)
    return chart, clock, duration_sec


def _detect_maimai_domain_pathologies(
    points: list[Any],
    tempo_changes: list[Any],
    first: float,
    clock: TempoClock,
    duration_sec: float,
    total_beats: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Locate possible transcription and rapid-repeat issues for review."""
    if not points:
        return [], Counter()

    beats_f = np.fromiter((float(p.beat) for p in points), dtype=np.float64, count=len(points))
    secs = clock.at_floats(beats_f)
    raw_by_reason: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def mark(reason: str, lo_beat: float, hi_beat: float) -> None:
        start = max(0.0, float(clock.at(Fraction(str(round(lo_beat, 4))))))
        end = min(duration_sec, float(clock.at(Fraction(str(round(hi_beat, 4))))))
        if end > start + 0.02:
            raw_by_reason[reason].append((start, end))

    try:
        parsed = parse_event_notes(points, tempo_changes, first)
    except Exception:
        parsed = None

    def _is_unprotected_button_only(idx: int) -> bool:
        if parsed is not None and not parsed.fatal_errors and idx < len(parsed.points):
            p_notes = parsed.points[idx].notes
            if any(n.note_type in ("touch", "touch_hold") for n in p_notes):
                return False
            btn_heads = [
                n for n in p_notes
                if n.note_type in ("tap", "hold", "slide")
                and isinstance(n.position, int)
                and not (n.note_type == "slide" and getattr(n, "slide_head_hidden", False))
            ]
            if not btn_heads:
                return False
            return any(not bool(n.modifiers.get("ex")) for n in btn_heads)
        raw = points[idx].note_text
        if any(ch in raw for ch in ("A", "B", "C", "D", "E", "x", "?", "!")):
            return False
        return True

    # Close unprotected button-to-button events expose possible micro-desync pseudo-chords for review;
    # cross-zone Button<->Touch sweeps, Touch fireworks cascades, and EX-protected sweeps are valid maimai DX techniques.
    for i in range(1, len(points)):
        dt = float(secs[i] - secs[i - 1])
        if 1e-6 < dt < 0.022 and _is_unprotected_button_only(i - 1) and _is_unprotected_button_only(i):
            b0 = max(0.0, beats_f[i - 1] - 0.5)
            b1 = min(total_beats, max(b0 + 0.5, beats_f[i] + 0.5))
            mark("auto_unquantized_timing", b0, b1)

    # Local steps retain their subdivision across a phrase phase shift.
    step_fracs = [points[i].beat - points[i - 1].beat for i in range(1, len(points))]
    for i in range(len(points)):
        left_step = step_fracs[i - 1] if i >= 1 else None
        right_step = step_fracs[i] if i < len(step_fracs) else None
        steps = [step for step in (left_step, right_step) if step is not None]
        if not steps or any(step.denominator in STANDARD_BEAT_DENOMINATORS for step in steps):
            continue
        prev_step = step_fracs[i - 2] if i >= 2 else None
        next_step = step_fracs[i + 1] if i + 1 < len(step_fracs) else None
        uniform_tuplet = (
            (left_step is not None and right_step is not None and left_step == right_step and left_step.numerator == 1)
            or (left_step is not None and prev_step == left_step and left_step.numerator == 1)
            or (right_step is not None and next_step == right_step and right_step.numerator == 1)
        )
        if uniform_tuplet:
            continue
        b0 = max(0.0, beats_f[i] - 0.5)
        b1 = min(total_beats, b0 + 1.0)
        mark("auto_unquantized_timing", b0, b1)

    # A repeat interval gives a review cue. Feasibility depends on context.
    if parsed is not None and not parsed.fatal_errors:
        last_touch_by_pad: dict[str, list[tuple[float, float]]] = defaultdict(list)
        last_btn_unprotected: dict[int, list[tuple[float, float]]] = defaultdict(list)

        for pt in parsed.points:
            b_val = float(pt.beat)
            t_sec = float(pt.time_sec)
            for note in pt.notes:
                ntype = note.note_type
                if ntype in ("touch", "touch_hold"):
                    pad = str(note.position).upper()
                    hist = last_touch_by_pad[pad]
                    if pad.startswith("C"):
                        if len(hist) >= 2 and (t_sec - hist[-1][0]) < 0.048 and (hist[-1][0] - hist[-2][0]) < 0.048:
                            b0, b1 = max(0.0, hist[-2][1] - 0.5), min(total_beats, b_val + 0.5)
                            mark("auto_sensor_jackhammer", b0, b1)
                    else:
                        if hist and 1e-6 < (t_sec - hist[-1][0]) < 0.085:
                            b0, b1 = max(0.0, hist[-1][1] - 0.5), min(total_beats, b_val + 0.5)
                            mark("auto_sensor_jackhammer", b0, b1)
                    hist.append((t_sec, b_val))
                elif ntype in ("tap", "hold", "slide") and isinstance(note.position, int):
                    # Slide without a head star ('?' / '!') has no button press; EX ('x') allows hand sweep.
                    if ntype == "slide" and getattr(note, "slide_head_hidden", False):
                        continue
                    is_ex = bool(note.modifiers.get("ex"))
                    btn = int(note.position)
                    if is_ex:
                        # EX resets unprotected jackhammer chain on this button.
                        last_btn_unprotected[btn].clear()
                        continue
                    hist = last_btn_unprotected[btn]
                    if hist:
                        dt1 = t_sec - hist[-1][0]
                        if 1e-6 < dt1 < 0.036:
                            b0, b1 = max(0.0, hist[-1][1] - 0.5), min(total_beats, b_val + 0.5)
                            mark("auto_sensor_jackhammer", b0, b1)
                        elif len(hist) >= 2 and 1e-6 < dt1 < 0.052 and 1e-6 < (hist[-1][0] - hist[-2][0]) < 0.052:
                            b0, b1 = max(0.0, hist[-2][1] - 0.5), min(total_beats, b_val + 0.5)
                            mark("auto_sensor_jackhammer", b0, b1)
                    hist.append((t_sec, b_val))

    spans = [
        {"start_sec": lo, "end_sec": hi, "reason": reason}
        for reason, intervals in raw_by_reason.items()
        for lo, hi in _merge_intervals(intervals)
    ]
    return spans, Counter({reason: len(intervals) for reason, intervals in raw_by_reason.items()})


def _try_load_audio_flux(
    audio_path: str | None,
    *,
    cache_dirs: tuple[Path, ...],
    flux_cache_dir: Path,
    envelope_memo: dict[str, tuple[np.ndarray, float] | None],
) -> tuple[np.ndarray, float] | None:
    """Load 100 Hz spectral flux with zero audio decoding when AudioFeatureStore exists.

    Gracefully returns ``None`` on missing or tiny synthetic test stubs without
    calling ``audio_sha256`` or printing to stdout/stderr.
    """
    if not audio_path:
        return None
    key = str(audio_path)
    if key in envelope_memo:
        return envelope_memo[key]

    p = Path(audio_path)
    try:
        st = p.stat()
    except OSError:
        envelope_memo[key] = None
        return None
    # Skip tiny synthetic unit-test byte stubs (e.g. b"FAKE_AUDIO_...") immediately.
    if st.st_size < 2048:
        envelope_memo[key] = None
        return None

    resolved = p.resolve()
    # 1) Fast path: check existing AudioFeatureStore mel.npy (column 96 = 100 Hz flux).
    for cdir in cache_dirs:
        sqlite_file = cdir / "source_digests.sqlite3"
        audio_v1_dir = cdir / "audio_v1"
        if not sqlite_file.is_file() or not audio_v1_dir.is_dir():
            continue
        try:
            from maichart.composer_v2.audio import MODEL_ID, MODEL_REVISION, SAMPLE_RATE, SCHEMA, _json_hash
            from maichart.form.cache_support import source_digests

            audio_digest = source_digests(str(cdir)).digest(resolved)
            for backend, model_id in (("mert", {"repo": MODEL_ID, "revision": MODEL_REVISION}), ("native", None)):
                ident = {
                    "schema": SCHEMA,
                    "backend": backend,
                    "audio_sha256": audio_digest,
                    "model": model_id,
                    "sample_rate": SAMPLE_RATE,
                    "mono": "native_channel_mean_float32",
                    "resample": "torchaudio_sinc_interp_hann_default",
                    "mel": {"n_fft": 1024, "hop": 240, "bands": 96, "center": True, "power": 2.0, "log": "log1p"},
                    "semantic": {
                        "layer": 8,
                        "hop": 320,
                        "receptive_field": 400,
                        "center_samples": 199.5,
                        "chunk_frames": 225,
                        "context_frames": 75,
                        "dtype": "float16",
                        "weightnorm_compat": "legacy_gv_to_parametrizations_v1",
                    },
                }
                bank_dir = audio_v1_dir / _json_hash(ident)
                mel_file = bank_dir / "mel.npy"
                idx_file = bank_dir / "index.json"
                if mel_file.is_file() and idx_file.is_file():
                    meta = json.loads(idx_file.read_text(encoding="utf-8"))
                    dur = float(meta.get("duration_sec", 0.0))
                    mel_mmap = np.load(mel_file, mmap_mode="r")
                    if mel_mmap.ndim == 2 and mel_mmap.shape[1] >= 97 and dur >= 4.0:
                        flux = np.asarray(mel_mmap[:, 96], dtype=np.float32).copy()
                        envelope_memo[key] = (flux, dur)
                        return envelope_memo[key]
        except Exception:
            pass

    # 2) Fallback: lightweight cached 100 Hz mel-flux (persisted by stat signature).
    try:
        from maichart.training.candidate_scan import cached_flux

        _, flux, dur = cached_flux(resolved, flux_cache_dir)
        if dur >= 4.0 and len(flux) >= 100:
            envelope_memo[key] = (np.asarray(flux, dtype=np.float32), float(dur))
            return envelope_memo[key]
    except Exception:
        pass

    envelope_memo[key] = None
    return None


def _propose_event_phrases(
    points: list[Any],
    clock: TempoClock,
    motion_stream: PrecomputedMotionStream,
    duration_sec: float,
) -> list[dict[str, Any]]:
    """Cut sustain-safe 8–16 beat musical phrases directly from EventPoints & jobs."""
    if not points:
        return []
    beats = [p.beat for p in points]
    beats_f = np.fromiter((float(b) for b in beats), dtype=np.float64, count=len(beats))
    secs = clock.at_floats(beats_f)

    all_sustains = [(float(j["start"]), float(j["end"])) for j in motion_stream.jobs]
    sustain_by_onset: dict[int, float] = {}
    for i, s_val in enumerate(secs):
        ends = [end for begin, end in all_sustains if abs(begin - float(s_val)) <= 0.15]
        if ends:
            sustain_by_onset[i] = max(ends)

    spans: list[tuple[int, int]] = []
    idx = 0
    n_pts = len(points)
    while idx < n_pts:
        start_b = beats[idx]
        start_s = float(secs[idx])
        active_until = max([0.0] + [end for begin, end in all_sustains if begin < start_s])
        best: tuple[tuple[bool, bool, float, float], int] | None = None
        j = idx
        while j + 1 < n_pts:
            active_until = max(active_until, sustain_by_onset.get(j, 0.0))
            nxt_b = beats[j + 1]
            length = float(nxt_b - start_b)
            if length > 24.0:
                break
            if length >= 8.0:
                clean = active_until <= float(secs[j + 1]) + 1e-5
                gap = float(nxt_b - beats[j])
                sc = (clean, length <= 16.0, gap, -abs(length - 12.0))
                if best is None or sc > best[0]:
                    best = (sc, j + 1)
                if length >= 16.0 and clean:
                    break
            j += 1
        stop = best[1] if best is not None else max(idx + 1, min(j + 1, n_pts))
        spans.append((idx, stop))
        idx = stop

    if len(spans) > 1 and float(beats[-1] - beats[spans[-1][0]]) < 8.0:
        prev_a, _ = spans[-2]
        tail_a, tail_b = spans[-1]
        if float(beats[tail_a] - beats[tail_a - 1]) < 8.0 and float(beats[-1] - beats[prev_a]) <= 24.0:
            spans[-2:] = [(prev_a, tail_b)]

    phrases: list[dict[str, Any]] = []
    for a, b in spans:
        start_sec = max(0.0, float(secs[a]))
        last_sec = float(secs[b - 1])
        tail_end = max([last_sec] + [end for begin, end in all_sustains if begin <= last_sec + 1e-5])
        end_sec = min(
            duration_sec,
            min(float(secs[b]), max(last_sec + 0.25, tail_end + 0.20)) if b < n_pts else tail_end + 0.25,
        )
        beat_span = float((beats[b] if b < n_pts else beats[b - 1]) - beats[a])
        if end_sec > start_sec + 0.10:
            phrases.append(
                {
                    "start_idx": a,
                    "end_idx": b,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "beats": beat_span,
                    "onset_secs": secs[a:b],
                }
            )
    return phrases


def _audit_chart_acoustics(
    points: list[Any],
    clock: TempoClock,
    motion_stream: PrecomputedMotionStream,
    phrases: list[dict[str, Any]],
    flux: np.ndarray,
    audio_duration_sec: float,
    chart_duration_sec: float,
) -> dict[str, Any]:
    """Evaluate global offset alignment, local phrase tempo drift, and high-energy dead gaps."""
    from maichart.training.candidate_scan import (
        AMBIGUITY_RATIO,
        FLUX_BIAS_SEC,
        FRAME_SEC,
        _normalized_envelope,
        _peak_center,
        _score_curve,
        alignment,
        support_threshold,
    )

    beats_f = np.fromiter((float(p.beat) for p in points), dtype=np.float64, count=len(points))
    onset_secs = clock.at_floats(beats_f)
    align = alignment(onset_secs, flux)
    envelope = _normalized_envelope(flux)
    pos_env = envelope[envelope > 0]
    p60_pos = float(np.percentile(pos_env, 60)) if len(pos_env) else 0.0
    sup_thresh = max(float(support_threshold(envelope)), p60_pos * 0.12, 1e-4)
    grid = np.arange(len(envelope), dtype=np.float64) * FRAME_SEC

    align_confident = bool(align.get("confident")) or (
        bool(align.get("judgeable"))
        and not align.get("ambiguous", True)
        and float(align.get("second_peak_ratio", 1.0)) < AMBIGUITY_RATIO
        and float(align.get("confidence_z", 0.0)) >= 6.0
    )
    global_offset_suspect = bool(
        align_confident
        and abs(float(align.get("offset_ms", 0.0))) >= 45.0
    )
    global_offset_ms = float(align.get("offset_ms", 0.0)) if align.get("judgeable") else 0.0
    global_offset_sec = float(align.get("suggested_first_delta_sec", 0.0)) if align.get("judgeable") else 0.0

    drift_spans: list[dict[str, Any]] = []
    phrase_acoustics: list[dict[str, Any]] = []
    local_offsets = np.arange(-0.12, 0.12 + 1e-9, 0.002)

    for ph in phrases:
        p_onsets = np.unique(ph["onset_secs"])
        p_onsets = p_onsets[(p_onsets >= 0.0) & (p_onsets <= (len(envelope) - 1) * FRAME_SEC)]
        p_info: dict[str, Any] = {"attack_support": None}
        if len(p_onsets) >= 4 and np.any(envelope > 0):
            sup_vals = np.interp(p_onsets + global_offset_sec, grid, envelope)
            p_info["attack_support"] = float(np.mean(sup_vals >= sup_thresh))
        if len(p_onsets) >= 12 and np.any(envelope > 0):
            curve = _score_curve(p_onsets, envelope, local_offsets)
            best = _peak_center(curve)
            peak = float(curve[best])
            med = float(np.nanmedian(curve))
            far = np.abs(local_offsets - local_offsets[best]) >= 0.040
            sec_pk = float(np.nanmax(curve[far])) if np.any(far) else -np.inf
            ratio = sec_pk / peak if peak > 0 else 1.0
            contrast = peak / med if med > 1e-6 else (100.0 if peak > 1e-6 else 0.0)
            loc_ms = (float(local_offsets[best]) - FLUX_BIAS_SEC) * 1000.0
            if (
                not global_offset_suspect
                and peak > 1e-3
                and contrast >= 1.25
                and ratio < 0.86
                and abs(loc_ms) >= 42.0
                and abs(loc_ms - global_offset_ms) >= 30.0
            ):
                drift_spans.append(
                    {
                        "start_sec": ph["start_sec"],
                        "end_sec": ph["end_sec"],
                        "reason": "auto_local_tempo_drift",
                    }
                )
        phrase_acoustics.append(p_info)

    # Locate sustained high-energy gaps that may warrant a review of chart completeness.
    gap_spans: list[dict[str, Any]] = []
    if (
        not global_offset_suspect
        and audio_duration_sec >= 30.0
        and len(points) >= 48
        and np.any(envelope > 0)
    ):
        active_frame_mask = np.zeros(len(envelope), dtype=bool)
        for s_on in onset_secs:
            lo_f = max(0, int((s_on - 0.5) / FRAME_SEC))
            hi_f = min(len(envelope), int((s_on + 0.5) / FRAME_SEC) + 1)
            active_frame_mask[lo_f:hi_f] = True
        active_mean = float(np.mean(envelope[active_frame_mask])) if np.any(active_frame_mask) else 0.0
        active_sup_ratio = (
            float(np.mean(envelope[active_frame_mask] >= sup_thresh))
            if np.any(active_frame_mask)
            else 0.25
        )

        # Build sorted occupied intervals (onsets + sustains), then inspect internal & tail gaps.
        occ = [(float(s), float(s) + 0.05) for s in onset_secs]
        occ.extend((float(j["start"]), float(j["end"])) for j in motion_stream.jobs)
        merged_occ = _merge_intervals(occ, merge_gap=0.10)
        tail_limit = audio_duration_sec - 3.0
        prev_end = merged_occ[0][1] if merged_occ else 0.0
        candidate_gaps: list[tuple[float, float, bool]] = []
        for o_lo, o_hi in merged_occ[1:]:
            if o_lo > prev_end + 2.0:
                candidate_gaps.append((prev_end, o_lo, False))
            prev_end = max(prev_end, o_hi)
        if tail_limit > prev_end + 4.0:
            candidate_gaps.append((prev_end, tail_limit, True))

        for g_lo, g_hi, is_tail in candidate_gaps:
            b_gaps = _beats_at_seconds(clock, [g_lo, g_hi])
            min_beats = 24.0 if is_tail else 16.0
            if float(b_gaps[1] - b_gaps[0]) < min_beats:
                continue
            f0 = max(0, int((g_lo + 0.5) / FRAME_SEC))
            f1 = min(len(envelope), int((g_hi - 0.5) / FRAME_SEC))
            if f1 - f0 < 200:
                continue
            seg_env = envelope[f0:f1]
            mean_env = float(np.mean(seg_env))
            active_ratio = float(np.mean(seg_env >= sup_thresh))
            req_ratio = max(0.04, active_sup_ratio * (0.75 if is_tail else 0.65))
            if active_mean > 1e-4 and mean_env >= active_mean * 0.75 and active_ratio >= req_ratio:
                beat_sec = max(0.25, (g_hi - g_lo) / max(1.0, float(b_gaps[1] - b_gaps[0])))
                if is_tail:
                    m_lo = max(0.0, g_lo - 0.5 * beat_sec)
                    m_hi = max(chart_duration_sec, g_hi)
                else:
                    m_lo = g_lo + beat_sec
                    m_hi = min(chart_duration_sec, g_hi - 0.5 * beat_sec)
                if m_hi > m_lo + 1.0:
                    gap_spans.append(
                        {
                            "start_sec": m_lo,
                            "end_sec": m_hi,
                            "reason": "auto_abandoned_gap",
                        }
                    )

    return {
        "alignment": align,
        "align_confident": align_confident,
        "global_offset_suspect": global_offset_suspect,
        "drift_spans": drift_spans,
        "gap_spans": gap_spans,
        "phrase_acoustics": phrase_acoustics,
    }


def _comfort_windows_for_chart(
    points: list[Any],
    tempo_changes: list[Any],
    clock: TempoClock,
    duration_sec: float,
    total_beats: float,
    *,
    motion_stream: PrecomputedMotionStream | None = None,
) -> list[tuple[float, float, float]]:
    """Score 4-beat sliding windows with evidence of movement or coordination in the chart."""
    motion_cache: OrderedDict[Any, Any] = OrderedDict()
    stream = (
        motion_stream
        if motion_stream is not None
        else build_motion_stream(points, tempo_changes, clock, motion_cache=motion_cache)
    )
    windows: list[tuple[float, float, float]] = []
    steps = max(0, int(math.ceil(total_beats / 2.0)) - 1)
    for step in range(steps):
        b0 = float(step * 2)
        b1 = min(total_beats, b0 + 4.0)
        if b1 - b0 < 2.0:
            break
        s0 = max(0.0, float(clock.at(Fraction(str(b0)))))
        s1 = min(duration_sec, float(clock.at(Fraction(str(b1)))))
        if s1 <= s0 + 0.05:
            continue
        warm = s0 - 0.6
        p_lo = int(np.searchsorted(stream.pulse_seconds, warm, side="left"))
        p_hi = int(np.searchsorted(stream.pulse_seconds, s1, side="left"))
        nearby_pulses = {(sec, pos) for sec, pos, _, _ in stream.pulses[p_lo:p_hi]}
        if len(stream.jobs):
            j_mask = (stream.job_ends >= warm) & (stream.job_starts < s1)
            nearby_jobs = [stream.jobs[i] for i in np.flatnonzero(j_mask)]
        else:
            nearby_jobs = []
        in_window = any(sec >= s0 for sec, _ in nearby_pulses) or any(
            job["end"] >= s0 for job in nearby_jobs
        )
        moving_job = any(
            start < s1 and stop > s0 for job in nearby_jobs for start, stop, _ in job["segments"]
        )
        if not in_window or (len(nearby_pulses) + len(nearby_jobs) < 2 and not moving_job):
            continue
        score, coverage = chirality_comfort(
            points,
            tempo_changes,
            clock,
            s0,
            s1,
            duration_sec,
            motion_cache=motion_cache,
            motion_stream=stream,
        )
        if float((coverage or {}).get("known_fraction", 0.0)) > 0.0:
            spike = float((coverage or {}).get("max_step_penalty", 0.0))
            eff_score = max(0.0, float(score) - 0.035 * max(0.0, spike - 1.85))
            windows.append((s0, s1, eff_score))
    return windows


def _phrase_craftsmanship_candidates(
    points: list[Any],
    tempo_changes: list[Any],
    clock: TempoClock,
    duration_sec: float,
    motion_stream: PrecomputedMotionStream,
    phrases: list[dict[str, Any]],
    phrase_acoustics: list[dict[str, Any]] | None,
    *,
    require_acoustics: bool,
) -> list[dict[str, Any]]:
    """Score sustain-safe musical phrases on maimai DX hand-flow, variety, and acoustic lock."""
    candidates: list[dict[str, Any]] = []
    for idx, ph in enumerate(phrases):
        if ph["beats"] < 8.0:
            continue
        seg_pts = points[ph["start_idx"] : ph["end_idx"]]
        if len(seg_pts) < 6:
            continue

        # Analyze note mechanic composition & anti-monotony bigram diversity.
        has_slide = False
        has_hold = False
        has_touch = False
        has_chord = False
        has_break = False
        total_heads = 0
        normalized_shapes: list[str] = []

        for p in seg_pts:
            txt = p.note_text
            atoms = [a for a in txt.split("/") if a]
            n_atoms = len(atoms)
            if n_atoms >= 2 or (n_atoms == 1 and len(atoms[0]) == 2 and atoms[0].isdigit()):
                has_chord = True
                total_heads += max(2, n_atoms)
            else:
                total_heads += 1
            shape_tags: list[str] = []
            for atom in atoms:
                if any(ch in atom for ch in ("-", "^", "<", ">", "v", "p", "q", "s", "z", "w", "V")):
                    has_slide = True
                    shape_tags.append("S")
                elif "h" in atom:
                    has_hold = True
                    shape_tags.append("H")
                elif atom[:1] in ("A", "B", "C", "D", "E"):
                    has_touch = True
                    shape_tags.append("T:" + atom[:1])
                else:
                    shape_tags.append("B:" + atom[:1])
                if "b" in atom:
                    has_break = True
            normalized_shapes.append("+".join(shape_tags))

        if total_heads < 8:
            continue

        bigrams = list(zip(normalized_shapes, normalized_shapes[1:]))
        unique_ratio = len(set(bigrams)) / max(1, len(bigrams)) if bigrams else 0.0
        # Reject monotonous copy-paste loops (e.g. spamming the exact same 1-2 note pattern).
        if len(bigrams) >= 7 and unique_ratio < 0.28:
            continue

        flow_score, cov = chirality_comfort(
            points,
            tempo_changes,
            clock,
            ph["start_sec"],
            ph["end_sec"],
            duration_sec,
            motion_stream=motion_stream,
        )
        if float(cov.get("known_fraction", 0.0)) < 0.95:
            continue
        max_spike = float(cov.get("max_step_penalty", 0.0))
        mean_hemi = float(cov.get("mean_hemi_penalty", 0.0))
        if flow_score < 0.35 or max_spike > 1.65 or mean_hemi > 0.12:
            continue

        traj_smooth = min(1.0, max(0.0, (float(flow_score) - 0.32) / 0.48))
        hemi_clean = min(1.0, max(0.0, 1.0 - mean_hemi / 0.12))
        strain_clean = min(1.0, max(0.0, 1.0 - max(0.0, max_spike - 0.55) / 1.20))
        flow_term = 0.45 * traj_smooth + 0.30 * hemi_clean + 0.25 * strain_clean

        mechanic_count = sum((has_slide, has_hold, has_touch, has_chord, has_break))
        mechanic_synergy = min(1.0, mechanic_count / 2.5)
        diversity_term = min(1.0, unique_ratio / 0.65)
        # Pure bare single-tap scales cannot exceed moderate variety.
        variety_term = 0.55 * mechanic_synergy + 0.45 * diversity_term
        if mechanic_count == 0:
            variety_term = min(variety_term, 0.35)

        ac_info = phrase_acoustics[idx] if phrase_acoustics and idx < len(phrase_acoustics) else None
        if require_acoustics:
            if ac_info is None or ac_info.get("attack_support") is None:
                continue
            sup = float(ac_info["attack_support"])
            if sup < 0.32:
                continue
            acoustic_term = min(1.0, max(0.0, (sup - 0.25) / 0.50))
            craft_score = 0.45 * flow_term + 0.30 * variety_term + 0.25 * acoustic_term
        else:
            craft_score = 0.60 * flow_term + 0.40 * variety_term

        candidates.append(
            {
                "start_sec": ph["start_sec"],
                "end_sec": ph["end_sec"],
                "score": float(craft_score),
                "flow_score": float(flow_score),
            }
        )
    return candidates


def _check_report_destination(path: Path, protected: set[Path]) -> None:
    reserved = {"quality_annotations.json", AUTO_ANNOTATION_FILENAME,
                AUTO_ANNOTATION_PREVIEW_FILENAME, COMPOSED_AUTO_ANNOTATION_FILENAME, AUDIT_LOCK_FILENAME}
    if path.name.casefold() in reserved or path in protected:
        raise ValueError("自动质检报告不能覆盖输入文件或质量标注")
    if path.exists() and path.stat().st_nlink > 1:
        for source in protected:
            if source.exists() and path.samefile(source):
                raise ValueError("自动质检报告不能覆盖输入文件或质量标注")


def _worker_init() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _audit_single_chart_worker(args_tuple: tuple[Any, ...]) -> dict[str, Any]:
    """Process-isolated chart analyzer for large-manifest parallel execution."""
    (
        row,
        lvl,
        pp_config,
        tap_on_slide_mode,
        inner_muri_mode,
        domain_pathology_mode,
        acoustic_audit_mode,
        effective_highlight_mode,
        need_comfort_windows,
        explicit_timing_weight,
        cache_dirs,
        flux_cache_dir,
    ) = args_tuple
    return _analyze_one_chart(
        row,
        lvl=lvl,
        pp_config=pp_config,
        tap_on_slide_mode=tap_on_slide_mode,
        inner_muri_mode=inner_muri_mode,
        domain_pathology_mode=domain_pathology_mode,
        acoustic_audit_mode=acoustic_audit_mode,
        effective_highlight_mode=effective_highlight_mode,
        need_comfort_windows=need_comfort_windows,
        explicit_timing_weight=explicit_timing_weight,
        cache_dirs=cache_dirs,
        flux_cache_dir=flux_cache_dir,
        metadata_cache=None,
        envelope_memo={},
    )


def _analyze_one_chart(
    row: CatalogChart,
    *,
    lvl: float,
    pp_config: PostProcessConfig,
    tap_on_slide_mode: str,
    inner_muri_mode: str,
    domain_pathology_mode: str,
    acoustic_audit_mode: str,
    effective_highlight_mode: str,
    need_comfort_windows: bool,
    explicit_timing_weight: float | None,
    cache_dirs: tuple[Path, ...],
    flux_cache_dir: Path,
    metadata_cache: dict[Path, Any] | None,
    envelope_memo: dict[str, tuple[np.ndarray, float] | None],
) -> dict[str, Any]:
    source = {
        "chart_id": row.chart_id,
        "source_path": row.source_path,
        "difficulty_index": row.difficulty_index,
    }
    try:
        loaded = _load_normalized_chart(row, metadata_cache=metadata_cache)
        if loaded is None:
            return {"status": "unverified", "entry": dict(source, reason="no_chart_events")}
        chart, clock, duration_sec = loaded
        points = list(chart.points)
        total_beats = max(float(Fraction(str(chart.end_beat))), max(float(p.beat) for p in points) + 1.0)
        det = detect_chart_conflicts(points, chart.tempo_changes, chart.audio_first, pp_config)
    except Exception as error:
        return {
            "status": "unverified",
            "entry": dict(source, reason="audit_error", detail=f"{type(error).__name__}: {error}"),
        }

    if det.coverage.get("status") == "partial":
        return {
            "status": "unverified",
            "entry": dict(
                source,
                reason="partial_coverage",
                parse_errors=det.coverage.get("parse_errors", [])[:5],
                unsupported=det.coverage.get("unsupported", [])[:5],
            ),
        }

    detected_counts: Counter[str] = Counter()
    masked_conflicts: Counter[str] = Counter()
    problem_spans_typed: list[dict[str, Any]] = []
    warning_veto_intervals: list[tuple[float, float]] = []

    for issue in det.conflicts_detected:
        detected_counts[f"{issue.conflict_type}:{issue.severity}"] += 1
        b_lo, b_hi = _causal_conflict_beat_span(issue, clock, total_beats)
        s_lo = max(0.0, float(clock.at(Fraction(str(round(b_lo, 4))))))
        s_hi = min(duration_sec, float(clock.at(Fraction(str(round(b_hi, 4))))))
        if s_hi > s_lo + 0.02:
            # Every detected warning/inner Muri vetoes 'good' promotion even when allowed at 1.0x!
            warning_veto_intervals.append((s_lo, s_hi))
        if not _should_mask_issue(
            issue,
            tap_on_slide_mode=tap_on_slide_mode,
            inner_muri_mode=inner_muri_mode,
        ):
            continue
        if s_hi > s_lo + 0.02:
            t_w = (
                float(explicit_timing_weight)
                if explicit_timing_weight is not None
                else _default_issue_timing_weight(issue, points, b_lo, b_hi)
            )
            problem_spans_typed.append(
                {
                    "start_sec": s_lo,
                    "end_sec": s_hi,
                    "reason": "auto_hard_muri",
                    "timing_weight": t_w,
                    "component_weight": 0.0,
                }
            )
            masked_conflicts[issue.conflict_type] += 1

    pathology_counts: Counter[str] = Counter()
    review_spans: list[dict[str, Any]] = []
    if domain_pathology_mode == "maimai_dx_v1":
        dom_spans, dom_counts = _detect_maimai_domain_pathologies(
            points,
            chart.tempo_changes,
            chart.audio_first,
            clock,
            duration_sec,
            total_beats,
        )
        review_spans.extend(dom_spans)
        pathology_counts.update(dom_counts)

    need_motion = (
        need_comfort_windows
        or acoustic_audit_mode != "off"
        or effective_highlight_mode != "off"
    )
    motion_stream = (
        build_motion_stream(points, chart.tempo_changes, clock)
        if need_motion
        else None
    )

    acoustic_verified = False
    global_offset_suspect = False
    audio_stat = None
    if row.audio_path:
        try:
            stat = Path(row.audio_path).stat()
            audio_stat = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            pass
    phrase_acoustics: list[dict[str, Any]] | None = None
    phrases: list[dict[str, Any]] = []

    if motion_stream is not None and (acoustic_audit_mode != "off" or effective_highlight_mode != "off"):
        phrases = _propose_event_phrases(points, clock, motion_stream, duration_sec)

    if acoustic_audit_mode != "off" and motion_stream is not None:
        flux_loaded = _try_load_audio_flux(
            row.audio_path,
            cache_dirs=cache_dirs,
            flux_cache_dir=flux_cache_dir,
            envelope_memo=envelope_memo,
        )
        if flux_loaded is None and acoustic_audit_mode == "on":
            return {
                "status": "unverified",
                "entry": dict(source, audio_path=row.audio_path, reason="audio_unavailable"),
            }
        if flux_loaded is not None:
            flux, audio_dur = flux_loaded
            ac_res = _audit_chart_acoustics(
                points, clock, motion_stream, phrases, flux, audio_dur, duration_sec
            )
            align = ac_res["alignment"]
            global_offset_suspect = bool(ac_res["global_offset_suspect"])
            phrase_acoustics = ac_res["phrase_acoustics"]
            acoustic_verified = bool(
                ac_res.get("align_confident")
                and abs(float(align.get("offset_ms", 0.0))) <= 22.0
            )
            if global_offset_suspect:
                # A later, louder layer can produce a consistent peak even when
                # the chart follows an earlier onset. Keep the estimate for review.
                review_spans.append({
                    "start_sec": 0.0,
                    "end_sec": duration_sec,
                    "reason": "auto_global_offset_review",
                    "offset_ms": align["offset_ms"],
                    "quarter_offsets_ms": [part["offset_ms"] for part in align["quarters"]],
                })
                pathology_counts["auto_global_offset_review"] += 1
            for sp in ac_res["drift_spans"]:
                review_spans.append(sp)
                pathology_counts["auto_local_tempo_drift"] += 1
            for sp in ac_res["gap_spans"]:
                review_spans.append(sp)
                pathology_counts["auto_abandoned_gap"] += 1

    hard_intervals = _coalesce_problem_spans(problem_spans_typed)
    hard_bounds = [(sp["start_sec"], sp["end_sec"]) for sp in hard_intervals]

    comfort_windows: list[tuple[float, float, float]] = []
    if need_comfort_windows:
        # Call module-level _comfort_windows_for_chart (without kwargs when default) so test monkeypatches work!
        if _comfort_windows_for_chart is _DEFAULT_COMFORT_WINDOWS_FN:
            comfort_windows = _comfort_windows_for_chart(
                points, chart.tempo_changes, clock, duration_sec, total_beats, motion_stream=motion_stream
            )
        else:
            comfort_windows = _comfort_windows_for_chart(
                points, chart.tempo_changes, clock, duration_sec, total_beats
            )

    good_candidates: list[dict[str, Any]] = []
    if (
        effective_highlight_mode != "off"
        and not global_offset_suspect
        and motion_stream is not None
        and (effective_highlight_mode == "chart_only_v1" or acoustic_verified)
    ):
        good_candidates = _phrase_craftsmanship_candidates(
            points,
            chart.tempo_changes,
            clock,
            duration_sec,
            motion_stream,
            phrases,
            phrase_acoustics,
            require_acoustics=(effective_highlight_mode == "acoustic_and_flow_v1"),
        )

    return {
        "status": "ok",
        "row": row,
        "tier": _level_tier(lvl),
        "duration_sec": duration_sec,
        "acoustic_verified": acoustic_verified,
        "audio_stat": audio_stat,
        "detected_counts": detected_counts,
        "masked_conflicts": masked_conflicts,
        "pathology_counts": pathology_counts,
        "review_spans": sorted(review_spans, key=lambda span: (span["start_sec"], span["end_sec"])),
        "hard_spans": hard_intervals,
        "hard_bounds": hard_bounds,
        "warning_veto_bounds": _merge_intervals([*warning_veto_intervals, *hard_bounds]),
        "comfort_windows": comfort_windows,
        "good_candidates": good_candidates,
    }


def _coalesce_problem_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the lowest active weights separately over each covered interval."""
    changes: dict[float, Counter[tuple[float, float, str]]] = defaultdict(Counter)
    for span in spans:
        # Use the same precision as the published training annotations.
        lo = round(float(span["start_sec"]), 6)
        hi = round(float(span["end_sec"]), 6)
        if hi <= lo:
            continue
        key = (float(span["timing_weight"]), float(span["component_weight"]), str(span["reason"]))
        changes[lo][key] += 1
        changes[hi][key] -= 1

    bounds = sorted(changes)
    active: Counter[tuple[float, float, str]] = Counter()
    result: list[dict[str, Any]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        for key, delta in changes[lo].items():
            active[key] += delta
            if active[key] == 0:
                del active[key]
        if not active:
            continue
        attributes = {
            "timing_weight": min(key[0] for key in active),
            "component_weight": min(key[1] for key in active),
            "reason": min((key[2] for key in active), key=lambda reason: (reason != "auto_hard_muri", reason)),
        }
        if result and result[-1]["end_sec"] == lo and all(result[-1][k] == v for k, v in attributes.items()):
            result[-1]["end_sec"] = hi
        else:
            result.append(dict(start_sec=lo, end_sec=hi, **attributes))
    return result


_DEFAULT_COMFORT_WINDOWS_FN = _comfort_windows_for_chart
_DEFAULT_DETECT_CONFLICTS_FN = detect_chart_conflicts
_DEFAULT_CHIRALITY_COMFORT_FN = chirality_comfort


def audit_manifest(
    manifest_path: str | Path,
    *,
    tap_on_slide_mode: str = "mask_hard",
    tap_on_slide_ms: float = 200.0,
    inner_muri_mode: str = "allow",
    touch_reachability_policy: str = "palm_span_v1",
    touch_cohold_radius: float = DEFAULT_COHOLD_RADIUS,
    domain_pathology_mode: str = "maimai_dx_v1",
    acoustic_audit_mode: str = "auto",
    highlight_good_mode: str = "off",
    good_quantile: float = 0.82,
    problem_timing_weight: float | None = None,
    max_problem_fraction: float = 0.45,
    experimental_hand_flow_quantile: float = 0.0,
    anchor_manifest: str | Path | None = None,
    anchor_dataset_id: str | None = None,
    min_level: float = 12.0,
    max_level: float = 15.0,
    min_difficulty_index: int = 5,
    max_difficulty_index: int = 6,
    limit_charts: int = 0,
    workers: int = 0,
    report_output: str | Path | None = None,
    progress_callback: Callable[[dict[str, int | float]], None] | None = None,
) -> dict[str, Any]:
    if tap_on_slide_mode not in TAP_ON_SLIDE_MODES:
        raise ValueError(f"tap_on_slide_mode must be one of {TAP_ON_SLIDE_MODES}")
    if inner_muri_mode not in INNER_MURI_MODES:
        raise ValueError(f"inner_muri_mode must be one of {INNER_MURI_MODES}")
    if domain_pathology_mode not in DOMAIN_PATHOLOGY_MODES:
        raise ValueError(f"domain_pathology_mode must be one of {DOMAIN_PATHOLOGY_MODES}")
    if acoustic_audit_mode not in ACOUSTIC_AUDIT_MODES:
        raise ValueError(f"acoustic_audit_mode must be one of {ACOUSTIC_AUDIT_MODES}")
    if highlight_good_mode not in HIGHLIGHT_GOOD_MODES:
        raise ValueError(f"highlight_good_mode must be one of {HIGHLIGHT_GOOD_MODES}")
    if not math.isfinite(good_quantile) or not 0.50 <= good_quantile <= 0.98:
        raise ValueError("good_quantile must be between 0.50 and 0.98")
    if not math.isfinite(tap_on_slide_ms) or not 0.0 <= tap_on_slide_ms <= 250.0:
        raise ValueError("tap_on_slide_ms must be between 0 and 250")
    if problem_timing_weight is not None and (
        not math.isfinite(problem_timing_weight) or not 0.0 <= problem_timing_weight <= 4.0
    ):
        raise ValueError("problem_timing_weight must be between 0 and 4")
    if not math.isfinite(max_problem_fraction) or not 0.0 < max_problem_fraction <= 1.0:
        raise ValueError("max_problem_fraction must be in (0, 1]")
    if not math.isfinite(experimental_hand_flow_quantile) or not 0.0 <= experimental_hand_flow_quantile <= 0.25:
        raise ValueError("experimental_hand_flow_quantile must be between 0 and 0.25")
    if limit_charts < 0:
        raise ValueError("limit_charts must be nonnegative")
    if isinstance(workers, bool) or workers < 0:
        raise ValueError("workers must be a nonnegative integer")

    effective_highlight_mode = "off" if highlight_good_mode == "auto" else highlight_good_mode
    touch_policy, touch_radius = validate_touch_reachability(
        touch_reachability_policy, touch_cohold_radius
    )

    manifest_file = Path(manifest_path).resolve()
    manual_file = annotation_path(manifest_file)
    auto_file = auto_annotation_path(manifest_file)
    preview_file = auto_annotation_path(manifest_file, preview=True)
    if manifest_file in {manual_file, auto_file, preview_file}:
        raise ValueError("自动质检输入清单不能使用质量标注的保留文件名")
    preview = limit_charts > 0
    report_path = (
        Path(report_output).resolve()
        if report_output
        else auto_audit_report_path(manifest_file, preview=preview)
    )
    previous_report_path = auto_audit_report_path(manifest_file, preview=preview, previous=True)
    protected = {manifest_file, manual_file.resolve(), auto_file.resolve(), preview_file.resolve(),
                 auto_annotation_path(manifest_file, composed=True).resolve()}
    protected.update((previous_report_path, auto_audit_report_path(manifest_file, preview=not preview, previous=True)))
    manifest_bytes = manifest_file.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_payload = json.loads(manifest_bytes.decode("utf-8-sig"))
    if not preview and "auto_quality_manifests" in manifest_payload:
        raise ValueError("此组合已选定采用哪些来源的正式体检。请打开原始数据集运行正式体检。")
    origin = split_annotation_source(manifest_file, manifest_payload)
    if origin is not None:
        protected.update((origin, annotation_path(origin).resolve(), auto_annotation_path(origin).resolve(),
                          auto_annotation_path(origin, composed=True).resolve()))
    anchor_file = Path(anchor_manifest).resolve() if anchor_manifest else None
    if anchor_file is not None:
        protected.update((anchor_file, annotation_path(anchor_file).resolve(), auto_annotation_path(anchor_file).resolve(),
                          auto_annotation_path(anchor_file, composed=True).resolve()))
    _check_report_destination(report_path, protected)

    # Always enable soft conflicts internally when highlighting worthy phrases so soft TapOnSlide
    # warnings veto 'good' promotion, while _should_mask_issue still governs problem masking!
    pp_config = PostProcessConfig(
        enable_conflict_detection=True,
        detect_hard_conflicts=True,
        detect_soft_conflicts=(tap_on_slide_mode == "mask_all" or effective_highlight_mode != "off"),
        conflict_engine="miacode",
        tap_on_slide_threshold_ms=float(tap_on_slide_ms),
        touch_reachability_policy=touch_policy,
        touch_cohold_radius=float(touch_radius),
    )

    # Candidate AudioFeatureStore cache directories for zero-decode 100 Hz flux reuse.
    cache_candidates: list[Path] = []
    raw_cache = manifest_payload.get("cache_dir")
    if raw_cache:
        cp = Path(str(raw_cache))
        cache_candidates.append((cp if cp.is_absolute() else manifest_file.parent / cp).resolve())
    cache_candidates.append((manifest_file.parent / "cache").resolve())
    cache_candidates.append((Path.cwd() / "cache" / "chart_composer_v2").resolve())
    dedup_caches = tuple(dict.fromkeys(cache_candidates))
    flux_cache_dir = dedup_caches[0] / "audit_flux"

    started = time.perf_counter()
    catalog = iter_catalog(manifest_file)
    need_comfort_windows = experimental_hand_flow_quantile > 0.0
    anchor_rows = iter_catalog(anchor_file) if need_comfort_windows and anchor_file else []
    if report_path.exists():
        protected.update(
            Path(value)
            for row in (*catalog, *anchor_rows)
            for value in (row.source_path, row.audio_path)
            if value
        )
        _check_report_destination(report_path, protected)

    eligible: list[tuple[CatalogChart, float]] = []
    for row in catalog:
        if not min_difficulty_index <= row.difficulty_index <= max_difficulty_index:
            continue
        try:
            lvl = float(numeric_level(row.level))
        except Exception:
            continue
        if not min_level <= lvl <= max_level:
            continue
        eligible.append((row, lvl))
    eligible_count = len(eligible)
    if limit_charts:
        eligible = eligible[:limit_charts]
    total = len(eligible)
    last_completed = 0
    last_progress = started

    def report_progress(completed: int, *, force: bool = False) -> None:
        nonlocal last_completed, last_progress
        if progress_callback is None:
            return
        now = time.perf_counter()
        if force or completed - last_completed >= 10 or now - last_progress >= 5.0:
            progress_callback(
                {
                    "completed": completed,
                    "total": total,
                    "elapsed_sec": round(now - started, 3),
                }
            )
            last_completed, last_progress = completed, now

    report_progress(0, force=True)

    conflict_counts_detected: Counter[str] = Counter()
    conflict_counts_masked: Counter[str] = Counter()
    pathology_counts_total: Counter[str] = Counter()
    chart_entries: list[dict[str, Any]] = []
    unverified_charts: list[dict[str, Any]] = []

    anchor_comfort_by_tier: dict[str, list[float]] = defaultdict(list)
    anchor_comfort_all: list[float] = []
    clean_comfort_by_tier: dict[str, list[float]] = defaultdict(list)
    clean_comfort_all: list[float] = []
    good_scores_by_tier: dict[str, list[float]] = defaultdict(list)
    good_scores_all: list[float] = []

    metadata_cache: dict[Path, Any] = {}
    envelope_memo: dict[str, tuple[np.ndarray, float] | None] = {}

    if anchor_rows:
        for a_row in anchor_rows:
            report_progress(0)
            if not min_difficulty_index <= a_row.difficulty_index <= max_difficulty_index:
                continue
            try:
                a_lvl = float(numeric_level(a_row.level))
                loaded = _load_normalized_chart(a_row, metadata_cache=metadata_cache)
            except Exception:
                continue
            if loaded is None:
                continue
            a_chart, a_clock, a_dur = loaded
            a_pts = list(a_chart.points)
            a_total_beats = max(
                float(Fraction(str(a_chart.end_beat))),
                max(float(p.beat) for p in a_pts) + 1.0,
            )
            a_tier = _level_tier(a_lvl)
            for _, _, sc in _comfort_windows_for_chart(
                a_pts, a_chart.tempo_changes, a_clock, a_dur, a_total_beats
            ):
                anchor_comfort_all.append(sc)
                anchor_comfort_by_tier[a_tier].append(sc)

    # Use multi-process pool only when un-monkeypatched and manifest is large enough to amortize spawn.
    is_unpatched = (
        detect_chart_conflicts is _DEFAULT_DETECT_CONFLICTS_FN
        and _comfort_windows_for_chart is _DEFAULT_COMFORT_WINDOWS_FN
        and chirality_comfort is _DEFAULT_CHIRALITY_COMFORT_FN
    )
    if workers == 0:
        effective_workers = min(4, max(1, (os.cpu_count() or 2) // 2)) if (is_unpatched and total >= 24) else 1
    else:
        effective_workers = workers if is_unpatched else 1

    if effective_workers <= 1:
        raw_results: list[dict[str, Any]] = []
        for completed, (row, lvl) in enumerate(eligible):
            report_progress(completed)
            res = _analyze_one_chart(
                row,
                lvl=lvl,
                pp_config=pp_config,
                tap_on_slide_mode=tap_on_slide_mode,
                inner_muri_mode=inner_muri_mode,
                domain_pathology_mode=domain_pathology_mode,
                acoustic_audit_mode=acoustic_audit_mode,
                effective_highlight_mode=effective_highlight_mode,
                need_comfort_windows=need_comfort_windows,
                explicit_timing_weight=problem_timing_weight,
                cache_dirs=dedup_caches,
                flux_cache_dir=flux_cache_dir,
                metadata_cache=metadata_cache,
                envelope_memo=envelope_memo,
            )
            raw_results.append(res)
    else:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        ordered_res: dict[int, dict[str, Any]] = {}
        done_count = 0
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
        ) as pool:
            pending = {}
            it = iter(enumerate(eligible))

            def _submit_more() -> None:
                while len(pending) < effective_workers * 2:
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    idx_n, (r_item, r_lvl) = nxt
                    job_args = (
                        r_item,
                        r_lvl,
                        pp_config,
                        tap_on_slide_mode,
                        inner_muri_mode,
                        domain_pathology_mode,
                        acoustic_audit_mode,
                        effective_highlight_mode,
                        need_comfort_windows,
                        problem_timing_weight,
                        dedup_caches,
                        flux_cache_dir,
                    )
                    pending[pool.submit(_audit_single_chart_worker, job_args)] = idx_n

            _submit_more()
            while pending:
                finished, _ = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                if not finished:
                    report_progress(done_count)
                for fut in finished:
                    idx_n = pending.pop(fut)
                    ordered_res[idx_n] = fut.result()
                    done_count += 1
                    report_progress(done_count)
                _submit_more()
        raw_results = [ordered_res[i] for i in range(len(eligible))]

    acoustic_verified_charts = 0
    review_regions: list[dict[str, Any]] = []
    for res in raw_results:
        if res["status"] != "ok":
            unverified_charts.append(res["entry"])
            continue
        row = res["row"]
        tier = res["tier"]
        conflict_counts_detected.update(res["detected_counts"])
        pathology_counts_total.update(res["pathology_counts"])
        if res.get("review_spans"):
            review_regions.append(dict(chart_id=row.chart_id, source_path=row.source_path,
                                       difficulty_index=row.difficulty_index, regions=res["review_spans"]))
        if res["acoustic_verified"]:
            acoustic_verified_charts += 1

        hard_bounds = res["hard_bounds"]
        if need_comfort_windows:
            for s0, s1, sc in res["comfort_windows"]:
                overlaps_hard = any(not (s1 <= h0 or s0 >= h1) for h0, h1 in hard_bounds)
                if not overlaps_hard:
                    clean_comfort_all.append(sc)
                    clean_comfort_by_tier[tier].append(sc)
                    if anchor_dataset_id and anchor_dataset_id in {
                        row.source_dataset,
                        row.source_dataset_id,
                    }:
                        anchor_comfort_all.append(sc)
                        anchor_comfort_by_tier[tier].append(sc)

        for gc in res["good_candidates"]:
            overlaps_veto = any(
                not (gc["end_sec"] <= v0 - 0.15 or gc["start_sec"] >= v1 + 0.15)
                for v0, v1 in res["warning_veto_bounds"]
            )
            if not overlaps_veto:
                good_scores_all.append(gc["score"])
                good_scores_by_tier[tier].append(gc["score"])

        chart_entries.append(res)

    # Calibrate level-stratified comfort cutoffs (falling back to global pool when tier sample < 8).
    comfort_cutoff_global: float | None = None
    comfort_cutoffs_by_tier: dict[str, float] = {}
    if need_comfort_windows:
        global_ref = anchor_comfort_all if len(anchor_comfort_all) >= 8 else clean_comfort_all
        if global_ref:
            comfort_cutoff_global = float(
                np.quantile(np.asarray(global_ref, dtype=np.float64), experimental_hand_flow_quantile)
            )
            for tier_name in ("lv12", "lv13", "lv14+"):
                t_anchor = anchor_comfort_by_tier.get(tier_name, [])
                t_clean = clean_comfort_by_tier.get(tier_name, [])
                t_ref = t_anchor if len(t_anchor) >= 8 else t_clean if len(t_clean) >= 12 else global_ref
                comfort_cutoffs_by_tier[tier_name] = float(
                    np.quantile(np.asarray(t_ref, dtype=np.float64), experimental_hand_flow_quantile)
                )

    # Calibrate level-stratified worthy-phrase cutoffs (strictly >= 0.68 floor).
    good_cutoffs_by_tier: dict[str, float] = {}
    if effective_highlight_mode != "off" and good_scores_all:
        global_good_cut = max(
            0.68, float(np.quantile(np.asarray(good_scores_all, dtype=np.float64), good_quantile))
        )
        for tier_name in ("lv12", "lv13", "lv14+"):
            t_scores = good_scores_by_tier.get(tier_name, [])
            if len(t_scores) >= 8:
                good_cutoffs_by_tier[tier_name] = max(
                    0.68, float(np.quantile(np.asarray(t_scores, dtype=np.float64), good_quantile))
                )
            else:
                good_cutoffs_by_tier[tier_name] = global_good_cut

    records: list[dict[str, Any]] = []
    excluded_charts = 0
    excluded_by_reason: Counter[str] = Counter()
    problem_regions_by_reason: Counter[str] = Counter()
    partially_masked_charts = 0
    soft_weighted_charts = 0
    highlighted_charts = 0
    highlighted_regions = 0
    total_masked_sec = 0.0
    total_soft_weighted_sec = 0.0
    total_highlighted_sec = 0.0
    total_chart_sec = 0.0
    clean_charts = 0
    partially_masked_regions = 0
    soft_weighted_regions = 0
    audited_chart_ids: list[str] = []
    audited_chart_audio_stats: dict[str, list[int] | None] = {}
    audio_identities: dict[str, tuple[str | None, str | None]] = {}

    for item in chart_entries:
        report_progress(total - 1)
        row: CatalogChart = item["row"]
        tier = item["tier"]
        duration_sec = float(item["duration_sec"])
        hard_spans: list[dict[str, Any]] = item["hard_spans"]
        hard_bounds: list[tuple[float, float]] = item["hard_bounds"]

        tier_comfort_cut = comfort_cutoffs_by_tier.get(tier, comfort_cutoff_global)
        comfort_intervals: list[tuple[float, float]] = []
        if tier_comfort_cut is not None:
            raw_comfort = [
                (s0, s1)
                for s0, s1, sc in item["comfort_windows"]
                if sc < tier_comfort_cut - 1e-9
            ]
            merged_comfort = _merge_intervals(raw_comfort)
            comfort_intervals = _subtract_intervals(
                merged_comfort,
                [(lo - 1e-4, hi + 1e-4) for lo, hi in hard_bounds],
            )

        masked_sec = sum(hi - lo for lo, hi in hard_bounds)
        effective_dur = max(duration_sec, max((hi for _, hi in hard_bounds), default=0.0), 1.0)
        problem_fraction = masked_sec / effective_dur
        is_catastrophic = problem_fraction > max_problem_fraction

        # Priority 2: Secondary 'good' highlighting — strictly vetoed by ANY problem/warning/comfort span!
        good_intervals: list[tuple[float, float]] = []
        if (
            not is_catastrophic
            and problem_fraction <= 0.15
            and tier in good_cutoffs_by_tier
            and item["good_candidates"]
        ):
            g_cut = good_cutoffs_by_tier[tier]
            all_vetoes = [
                *(item["warning_veto_bounds"]),
                *comfort_intervals,
            ]
            veto_padded = [(lo - 0.15, hi + 0.15) for lo, hi in all_vetoes]
            raw_good = [
                (gc["start_sec"], gc["end_sec"])
                for gc in item["good_candidates"]
                if gc["score"] >= g_cut - 1e-9
                and not any(
                    not (gc["end_sec"] <= vlo or gc["start_sec"] >= vhi)
                    for vlo, vhi in veto_padded
                )
            ]
            good_intervals = _subtract_intervals(
                _merge_intervals(raw_good),
                veto_padded,
                min_width=1.0,
            )

        regions: list[dict[str, Any]] = []
        for sp in hard_spans:
            regions.append(
                {
                    "start_sec": round(float(sp["start_sec"]), 6),
                    "end_sec": round(float(sp["end_sec"]), 6),
                    "label": "problem",
                    "reason": str(sp["reason"]),
                    "weight": 1.0,
                    "timing_weight": float(sp["timing_weight"]),
                    "component_weight": float(sp["component_weight"]),
                }
            )
        for lo, hi in comfort_intervals:
            regions.append(
                {
                    "start_sec": round(lo, 6),
                    "end_sec": round(hi, 6),
                    "label": "problem",
                    "reason": "comfort",
                    "weight": 1.0,
                    "timing_weight": 1.0,
                    "component_weight": 0.25,
                }
            )
        for lo, hi in good_intervals:
            regions.append(
                {
                    "start_sec": round(lo, 6),
                    "end_sec": round(hi, 6),
                    "label": "good",
                    "reason": "auto_worthy_phrase",
                    "weight": 1.0,
                    "timing_weight": GOOD_TIMING_WEIGHT,
                    "component_weight": GOOD_COMPONENT_WEIGHT,
                }
            )
        regions.sort(key=lambda r: (r["start_sec"], r["end_sec"]))

        if not regions and not is_catastrophic:
            clean_charts += int(not item.get("review_spans"))
            total_chart_sec += duration_sec
            audited_chart_ids.append(row.chart_id)
            audited_chart_audio_stats[row.chart_id] = item.get("audio_stat")
            continue

        audio_path = row.audio_path
        digest, error = None, "missing audio path"
        if audio_path:
            if audio_path not in audio_identities:
                try:
                    audio_identities[audio_path] = (audio_sha256(audio_path), None)
                except OSError as exc:
                    audio_identities[audio_path] = (None, f"{type(exc).__name__}: {exc}")
            digest, error = audio_identities[audio_path]
        if digest is None:
            unverified_charts.append(
                {
                    "chart_id": row.chart_id,
                    "source_path": row.source_path,
                    "difficulty_index": row.difficulty_index,
                    "audio_path": audio_path,
                    "reason": "audio_identity_unavailable",
                    "detail": error,
                }
            )
            continue

        audited_chart_ids.append(row.chart_id)
        audited_chart_audio_stats[row.chart_id] = item.get("audio_stat")
        total_chart_sec += duration_sec
        conflict_counts_masked.update(item["masked_conflicts"])
        total_masked_sec += masked_sec
        for sp in hard_spans:
            problem_regions_by_reason[str(sp["reason"])] += 1
        if comfort_intervals:
            soft_weighted_charts += 1
            soft_weighted_regions += len(comfort_intervals)
            total_soft_weighted_sec += sum(hi - lo for lo, hi in comfort_intervals)
            problem_regions_by_reason["comfort"] += len(comfort_intervals)
        if good_intervals:
            highlighted_charts += 1
            highlighted_regions += len(good_intervals)
            total_highlighted_sec += sum(hi - lo for lo, hi in good_intervals)

        if is_catastrophic:
            excluded_charts += 1
            excluded_by_reason["auto_excessive_muri"] += 1
        elif hard_bounds:
            partially_masked_charts += 1
            partially_masked_regions += len(hard_spans)
        elif not comfort_intervals and not item.get("review_spans"):
            # Clean chart that also contributed worthy 'good' phrases!
            clean_charts += 1

        rec_reason = (
            "auto_excessive_muri"
            if is_catastrophic
            else "auto_segment_muri_mask"
            if hard_bounds
            else "auto_hand_flow_weight"
            if comfort_intervals
            else "auto_worthy_highlight"
        )
        rec: dict[str, Any] = {
            "chart_id": row.chart_id,
            "song_id": row.song_id,
            "difficulty_index": row.difficulty_index,
            "source_path": row.source_path,
            "chart_sha256": row.chart_sha256,
            "audio_path": audio_path,
            "audio_sha256": digest,
            "label": "excluded" if is_catastrophic else "normal",
            "weight": 0.0 if is_catastrophic else 1.0,
            "reason": rec_reason,
            "regions": regions,
        }
        records.append(rec)

    audit_metadata = {
        "policy": "auto_v1",
        "engine_version": "maimai_dx_two_tier_v5",
        "manifest_sha256": manifest_sha256,
        "tap_on_slide_mode": tap_on_slide_mode,
        "tap_on_slide_ms": float(tap_on_slide_ms),
        "inner_muri_mode": inner_muri_mode,
        "touch_reachability_policy": touch_policy,
        "touch_cohold_radius": float(touch_radius),
        "domain_pathology_mode": domain_pathology_mode,
        "acoustic_audit_mode": acoustic_audit_mode,
        "highlight_good_mode": effective_highlight_mode,
        "good_quantile": float(good_quantile),
        "problem_timing_weight": (
            "auto_decoupled" if problem_timing_weight is None else float(problem_timing_weight)
        ),
        "max_problem_fraction": float(max_problem_fraction),
        "experimental_hand_flow_quantile": float(experimental_hand_flow_quantile),
        "comfort_cutoff": comfort_cutoff_global,
        "comfort_cutoffs_by_tier": dict(comfort_cutoffs_by_tier),
        "good_cutoffs_by_tier": dict(good_cutoffs_by_tier),
    }
    audit_id = annotations_fingerprint({
        "settings": audit_metadata,
        "preview": preview,
        "records": records,
        "audited_chart_ids": audited_chart_ids,
        "audited_chart_audio_stats": audited_chart_audio_stats,
        "unverified_charts": unverified_charts,
    })
    audit_metadata["audit_id"] = audit_id
    report = {
        "schema": "composer-v2-auto-quality-audit-v1",
        "manifest": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "audit_id": audit_id,
        "auto_annotations_path": str(preview_file if preview else auto_file),
        "preview": preview,
        "manual_annotations_preserved": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eligible_charts": eligible_count,
        "out_of_scope_charts": len(catalog) - eligible_count,
        "catalog_charts": len(catalog),
        "requested_charts": total,
        "scope": {"min_level": min_level, "max_level": max_level,
                  "min_difficulty_index": min_difficulty_index, "max_difficulty_index": max_difficulty_index},
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "settings": audit_metadata,
        "audited_charts": len(audited_chart_ids),
        "audited_chart_ids": sorted(set(audited_chart_ids)),
        "audited_chart_audio_stats": audited_chart_audio_stats,
        "audited_chart_audio_paths": {row.chart_id: row.audio_path for row, _ in eligible
                                      if row.chart_id in audited_chart_audio_stats},
        "acoustic_verified_charts": acoustic_verified_charts,
        "review_chart_count": len(review_regions),
        "review_regions": review_regions,
        "unverified_chart_count": len(unverified_charts),
        "unverified_charts": unverified_charts,
        "unverified_by_reason": dict(Counter(item["reason"] for item in unverified_charts)),
        "clean_charts": clean_charts,
        "partially_masked_charts": partially_masked_charts,
        "partially_masked_regions": partially_masked_regions,
        "excluded_charts": excluded_charts,
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "soft_weighted_charts": soft_weighted_charts,
        "soft_weighted_regions": soft_weighted_regions,
        "highlighted_charts": highlighted_charts,
        "highlighted_regions": highlighted_regions,
        "total_chart_sec": round(total_chart_sec, 3),
        "total_masked_sec": round(total_masked_sec, 3),
        "total_soft_weighted_sec": round(total_soft_weighted_sec, 3),
        "total_highlighted_sec": round(total_highlighted_sec, 3),
        "masked_duration_fraction": round(total_masked_sec / max(total_chart_sec, 1e-6), 6),
        "detected_conflicts_by_type_severity": dict(sorted(conflict_counts_detected.items())),
        "masked_conflicts_by_type": dict(sorted(conflict_counts_masked.items())),
        "domain_pathologies_by_type": dict(sorted(pathology_counts_total.items())),
        "problem_regions_by_reason": dict(sorted(problem_regions_by_reason.items())),
    }
    # A report write failure must leave the training sidecar untouched. The
    # report and sidecar carry the same identity so readers can detect an
    # interrupted publish. Manual annotation saves run independently.
    try:
        with _directory_lock(manifest_file.parent, AUDIT_LOCK_FILENAME):
            if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != manifest_sha256:
                raise ValueError("体检期间训练清单已变化，请重新运行体检")
            if not preview and auto_file.is_file():
                try:
                    previous = json.loads(auto_file.read_text(encoding="utf-8-sig"))
                except (UnicodeError, ValueError):
                    previous = None
                owner = _auto_annotation_owner(auto_file, previous)
                if owner is not None and owner != manifest_file:
                    raise ValueError(f"自动质检标注已绑定另一训练清单：{owner}")
            # Keep the preceding report available while the training sidecar
            # still points to its audit ID, including if the process dies.
            _, _, published = resolve_auto_audit(manifest_file, manifest_payload, preview=preview)
            if published is not None and Path(published["report_path"]) != previous_report_path:
                _write_json(previous_report_path, published)
            _write_json(report_path, report)
            save_auto_annotations(manifest_file, records, audit_metadata=audit_metadata, preview=preview)
    except TrainingOutputBusyError as error:
        raise ValueError("同一目录的另一项体检正在保存结果，请稍后重试") from error
    report["report_path"] = str(report_path)
    report_progress(total, force=True)
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parser().parse_args(argv)

    def print_progress(progress: dict[str, int | float]) -> None:
        print("composer_v2_auto_quality_progress " + json.dumps(progress), flush=True)

    report = audit_manifest(
        args.manifest,
        tap_on_slide_mode=args.tap_on_slide_mode,
        tap_on_slide_ms=args.tap_on_slide_ms,
        inner_muri_mode=args.inner_muri_mode,
        touch_reachability_policy=args.touch_reachability_policy,
        touch_cohold_radius=args.touch_cohold_radius,
        domain_pathology_mode=args.domain_pathology_mode,
        acoustic_audit_mode=args.acoustic_audit_mode,
        highlight_good_mode=args.highlight_good_mode,
        good_quantile=args.good_quantile,
        problem_timing_weight=args.problem_timing_weight,
        max_problem_fraction=args.max_problem_fraction,
        experimental_hand_flow_quantile=args.experimental_hand_flow_quantile,
        anchor_manifest=args.anchor_manifest,
        anchor_dataset_id=args.anchor_dataset_id,
        min_level=args.min_level,
        max_level=args.max_level,
        min_difficulty_index=args.min_difficulty_index,
        max_difficulty_index=args.max_difficulty_index,
        limit_charts=args.limit_charts,
        workers=args.workers,
        report_output=args.report_output,
        progress_callback=print_progress,
    )
    summary = {key: value for key, value in report.items()
               if key not in {"audited_chart_ids", "audited_chart_audio_stats", "unverified_charts", "review_regions"}}
    print("composer_v2_auto_quality_audit " + json.dumps(summary, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
