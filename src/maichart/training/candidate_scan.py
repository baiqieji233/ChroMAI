"""Offline checks for candidate charts before any of them become supervision.

Every result here is evidence for a human or a later calibrated scorer. Nothing
is written back to the source chart, and nothing in this module runs inside the
training hot path.

Outcome classes
---------------
``error``       A verifiable data defect: unreadable chart/audio, missing level,
                compiler roundtrip failure or onsets outside the audio.
                Not trainable until the source is fixed.
``suspicious``  Needs a human: native hard conflicts, non-standard spelling,
                alignment that cannot be judged automatically, a small offset,
                or possible tempo drift.
``ok``          No automatic finding. This is *not* a quality judgement.

Deliberately not findings: few notes, long rests, repetition, few or uniform
Slides, unusual subdivisions, low density, low acoustic support of single
notes (a chart may follow a soft vocal line). These only enter descriptive
segment features used to order the review queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

POLICY = "candidate_scan_v2"
FLUX_SCHEMA = "candidate_flux_v2"
FRAME_SEC = 0.01
# Offset scan. Positive offset: audio attacks occur after the chart onsets,
# i.e. the chart is early and &first should increase by that amount.
OFFSET_RANGE_SEC = 0.25
OFFSET_STEP_SEC = 0.002
PEAK_RADIUS_SEC = 0.020  # two 10 ms frames; actual pooling radius
# Provisional values inherited from a v1 pilot (10 official, 12 custom charts).
# They order human review; they do not establish quality or alignment truth.
# ``FLUX_BIAS_SEC`` is the systematic lag of the spectral-flux peak relative to
# an authored onset on correctly aligned official charts; it is removed before
# thresholds are applied.
FLUX_BIAS_SEC = -0.008  # median of 10 official charts (spread -14..+6 ms); recalibrate on more
OK_OFFSET_SEC = 0.020
REVIEW_OFFSET_SEC = 0.040
MIN_CONFIDENCE_Z = 4.0  # reported only; MAD z is depressed by periodic music
MIN_CONTRAST = 1.15  # provisional, must be recalibrated (see handoff)
AMBIGUITY_RATIO = 0.92
DRIFT_SEC = 0.030
MIN_ALIGNMENT_ONSETS = 32
SUPPORT_TOLERANCE_SEC = 0.030
SEGMENT_MIN_BEATS = 8
SEGMENT_TARGET_BEATS = 12
SEGMENT_MAX_BEATS = 16
SEGMENT_HARD_MAX_BEATS = 24


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def content_key(inote: str) -> str:
    """Whitespace-insensitive content identity of one difficulty slot.

    Timing metadata (&first, wholebpm) is excluded on purpose: a copied chart
    with a corrected offset is still the same authored content.
    """
    text = re.sub(r"\s+", "", (inote or "").replace("\r", ""))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _signature_key(signature: dict[str, Any], extra: str = "") -> str:
    raw = json.dumps(signature, sort_keys=True, ensure_ascii=False) + "\0" + extra
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Audio onset envelope (same spectral flux definition as the training bank)
# ---------------------------------------------------------------------------

def compute_flux(audio_path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (times, flux, duration) at 100 Hz.

    Matches AudioFeatureStore's native mel column 96: 24 kHz mono, 1024 FFT,
    hop 240, 96 mel bands, log1p, positive first difference averaged over
    bands. Nothing is written to the training audio cache.
    """
    import torch
    import torchaudio
    from maichart.form.rational_audio import _read_mono
    signal, rate, _ = _read_mono(str(audio_path))
    if not len(signal):
        raise ValueError("音频为空")
    duration = len(signal) / float(rate)
    wave = torch.from_numpy(np.asarray(signal, dtype=np.float32).copy())
    if rate != 24000:
        wave = torchaudio.functional.resample(wave, int(rate), 24000)
    if len(wave) < 400:
        wave = torch.nn.functional.pad(wave, (0, 400 - len(wave)))
    spectrum = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=240, power=2.,
                                                 pad_mode="constant")(wave)
    bands = torchaudio.transforms.MelScale(n_mels=96, sample_rate=24000, n_stft=513)(spectrum)
    logmel = torch.log1p(bands).T
    flux = torch.zeros(len(logmel))
    flux[1:] = (logmel[1:] - logmel[:-1]).clamp_min(0).mean(dim=-1)
    values = flux.numpy().astype(np.float32)
    times = np.arange(len(values), dtype=np.float64) * FRAME_SEC
    keep = times <= duration
    return times[keep], values[keep], duration


def cached_flux(audio_path: str | Path, cache_dir: str | Path | None):
    """Small per-audio envelope cache keyed by file signature (no hashing)."""
    signature = file_signature(audio_path)
    if cache_dir is None:
        return compute_flux(audio_path)
    cache = Path(cache_dir) / "flux"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (_signature_key(signature, FLUX_SCHEMA) + ".npz")
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as data:
                flux = data["flux"].astype(np.float32)
                duration = float(data["duration"])
            return np.arange(len(flux)) * FRAME_SEC, flux, duration
        except (OSError, ValueError, KeyError):
            pass
    times, flux, duration = compute_flux(audio_path)
    # Return the same quantized values on a miss and on a hit.
    flux = flux.astype(np.float16).astype(np.float32)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, flux=flux.astype(np.float16), duration=np.float64(duration))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return times, flux, duration


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _normalized_envelope(flux: np.ndarray) -> np.ndarray:
    """Accent-emphasizing envelope, local-max pooled within PEAK_RADIUS_SEC."""
    values = np.maximum(np.asarray(flux, dtype=np.float64), 0.)
    positive = values[values > 0]
    scale = float(np.median(positive)) if len(positive) else 0.
    if scale <= 0:
        return np.zeros_like(values)
    envelope = np.log1p(values / scale) ** 2
    radius = max(1, int(round(PEAK_RADIUS_SEC / FRAME_SEC)))
    padded = np.pad(envelope, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    return windows.max(axis=1)


def support_threshold(envelope: np.ndarray) -> float:
    """Accent level counted as acoustic support: 70th percentile of all frames."""
    return float(np.percentile(envelope, 70)) if len(envelope) else 0.


def _score_curve(onsets: np.ndarray, envelope: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    grid = np.arange(len(envelope)) * FRAME_SEC
    queries = onsets[None, :] + offsets[:, None]
    values = np.interp(queries.ravel(), grid, envelope, left=np.nan, right=np.nan).reshape(queries.shape)
    return np.nanmean(values, axis=1)


def _peak_center(curve: np.ndarray) -> int:
    """Centre of the contiguous maximum plateau containing the first maximum."""
    best = int(np.nanargmax(curve))
    equal = np.isclose(curve, curve[best], rtol=1e-7, atol=1e-12)
    left = right = best
    while left > 0 and equal[left - 1]:
        left -= 1
    while right + 1 < len(curve) and equal[right + 1]:
        right += 1
    return (left + right) // 2


def alignment(onsets: np.ndarray, flux: np.ndarray) -> dict[str, Any]:
    """Global offset with confidence, ambiguity and drift; never a quality score."""
    onsets = np.unique(np.asarray(onsets, dtype=np.float64))
    onsets = onsets[(onsets >= 0) & (onsets <= (len(flux) - 1) * FRAME_SEC)]
    result: dict[str, Any] = dict(onsets=int(len(onsets)), policy=POLICY,
                                  flux_bias_ms=FLUX_BIAS_SEC * 1000)
    if len(onsets) < MIN_ALIGNMENT_ONSETS:
        result.update(judgeable=False, reason="too_few_onsets")
        return result
    envelope = _normalized_envelope(flux)
    if not np.any(envelope > 0):
        result.update(judgeable=False, reason="silent_audio")
        return result
    offsets = np.arange(-OFFSET_RANGE_SEC, OFFSET_RANGE_SEC + 1e-9, OFFSET_STEP_SEC)
    curve = _score_curve(onsets, envelope, offsets)
    # Max pooling creates flat peaks. Use the centre of the winning plateau,
    # rather than introducing a systematic earliest-bin preference.
    best = _peak_center(curve)
    peak = float(curve[best])
    median = float(np.nanmedian(curve))
    mad = float(np.nanmedian(np.abs(curve - median))) * 1.4826 + 1e-9
    z = (peak - median) / mad
    # A second peak a beat-subdivision away (e.g. dense hi-hats) makes the
    # estimate ambiguous; report it instead of guessing.
    far = np.abs(offsets - offsets[best]) >= 0.040
    second = float(np.nanmax(curve[far])) if np.any(far) else -np.inf
    ratio = second / peak if peak > 0 else 1.
    contrast = peak / median if median > 0 else 0.
    raw_offset = float(offsets[best])
    offset = raw_offset - FLUX_BIAS_SEC
    # Per-quarter offsets near the global optimum reveal tempo drift.
    quarters = []
    for part in np.array_split(onsets, 4):
        if len(part) < 12:
            continue
        local = np.arange(offsets[best] - .06, offsets[best] + .06 + 1e-9, OFFSET_STEP_SEC)
        local_curve = _score_curve(part, envelope, local)
        quarters.append(dict(start_sec=float(part[0]), end_sec=float(part[-1]),
                             offset_ms=round((float(local[_peak_center(local_curve)]) - FLUX_BIAS_SEC) * 1000, 1)))
    drift = (max(q["offset_ms"] for q in quarters) - min(q["offset_ms"] for q in quarters)) if len(quarters) >= 2 else None
    # Fraction of onsets with a local accent at the corrected position.
    grid = np.arange(len(envelope)) * FRAME_SEC
    support_values = np.interp(onsets + offsets[best], grid, envelope)
    threshold = support_threshold(envelope)
    result.update(judgeable=True, raw_offset_ms=round(raw_offset * 1000, 1), offset_ms=round(offset * 1000, 1), confidence_z=round(z, 2),
                  contrast=round(contrast, 3),
                  second_peak_ratio=round(ratio, 3), ambiguous=bool(ratio >= AMBIGUITY_RATIO),
                  confident=bool(contrast >= MIN_CONTRAST and ratio < AMBIGUITY_RATIO),
                  quarters=quarters, drift_ms=None if drift is None else round(drift, 1),
                  supported_fraction=round(float(np.mean(support_values >= threshold)), 3),
                  suggested_first_delta_sec=round(offset, 4))
    return result


# ---------------------------------------------------------------------------
# Native conflicts
# ---------------------------------------------------------------------------

INNER_SLIDE_CONFLICTS = frozenset({"miacode_SlideTooFast", "miacode_SlideHeadTap"})


def _inspect_native_conflicts(
    chart, audio_end: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one MiaCode pass, separating acceptable inner-screen slide early triggers from hard Muri."""
    from maichart.composer.guard import EventGuard
    from maichart.generation.post_process import detect_chart_conflicts
    from maichart.event_timing import EventTempo
    changes = {change.beat: change.bpm for change in chart.tempo_changes}
    tempo = [EventTempo(beat, bpm) for beat, bpm in sorted(changes.items())]
    guard = EventGuard(tempo, chart.audio_first, max(audio_end, 1.), min_same_head_sec=0.,
                       touch_layout_policy="open", physical_policy="runtime_hard_v2")
    result = detect_chart_conflicts(list(chart.points), tempo, chart.audio_first, guard.config)
    parse_errors = list(result.coverage.get("parse_errors") or []) if result.coverage.get("status") == "partial" else []
    hard_issues: list[dict[str, Any]] = []
    inner_issues: list[dict[str, Any]] = []
    for item in result.conflicts_detected:
        entry = dict(kind=item.conflict_type, severity=str(item.severity), time_sec=round(float(item.time), 4),
                     notes=[dict(beat=str(note.beat), raw=note.raw) for note in item.notes][:6])
        if item.conflict_type in INNER_SLIDE_CONFLICTS:
            if str(item.severity) == "hard":
                inner_issues.append(entry)
            continue
        if not guard.enforced(item):
            continue
        hard_issues.append(entry)
    return hard_issues, inner_issues, parse_errors


def native_hard_conflicts(chart, audio_end: float) -> list[dict[str, Any]]:
    """One whole-chart MiaCode pass returning actionable hard conflicts (excluding inner-slide pre-triggers)."""
    hard_issues, _, _ = _inspect_native_conflicts(chart, audio_end)
    return hard_issues


# ---------------------------------------------------------------------------
# Review segments
# ---------------------------------------------------------------------------

def _phase_flags(beats) -> dict[str, bool]:
    from fractions import Fraction
    quarter = any((4 * b).denominator == 1 and (2 * b).denominator != 1 for b in beats)
    sixth = any((6 * b).denominator == 1 and (2 * b).denominator != 1 for b in beats)
    other = any(Fraction(b).denominator not in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48) for b in beats)
    return dict(binary=quarter, ternary=sixth, mixed_16_24=quarter and sixth, unusual_grid=other)


def propose_segments(points, clock, notes, *, envelope=None, offset_sec=0.,
                     conflicts=()) -> list[dict[str, Any]]:
    """Musically movable 8–16 beat review spans, cut at onsets and never inside
    a sustained action when avoidable. Spans are review/supervision ranges in
    seconds [start, end): end is the next span's first onset, so an onset
    belongs to exactly one span, matching the UI's selection semantics.
    """
    from fractions import Fraction
    beats = sorted({p.beat for p in points})
    if not beats:
        return []
    seconds = {b: float(clock.at(b)) for b in beats}
    # Latest sustained end (seconds) of actions starting at each onset.
    sustain_end: dict[float, float] = {}
    for note in notes:
        end = note.get("end_sec")
        if end is not None:
            key = round(float(note["start_sec"]), 6)
            sustain_end[key] = max(sustain_end.get(key, 0.), float(end))
    # Include sustains entering later spans, including unavoidable crossings.
    all_sustains = [(float(n["start_sec"]), float(n["end_sec"]))
                    for n in notes if n.get("end_sec") is not None]
    spans = []
    index = 0
    while index < len(beats):
        start = beats[index]
        active_until = max([0.] + [end for begin, end in all_sustains if begin < seconds[start]])
        best = None
        j = index
        while j + 1 < len(beats):
            current = beats[j]
            active_until = max(active_until, sustain_end.get(round(seconds[current], 6), 0.))
            nxt = beats[j + 1]
            length = nxt - start
            if length > SEGMENT_HARD_MAX_BEATS:
                break
            if length >= SEGMENT_MIN_BEATS:
                clean = active_until <= seconds[nxt] + 1e-6
                gap = float(nxt - current)
                score = (clean, length <= SEGMENT_MAX_BEATS, gap, -abs(float(length) - SEGMENT_TARGET_BEATS))
                if best is None or score > best[0]:
                    best = (score, j + 1)
                if length >= SEGMENT_MAX_BEATS and clean:
                    break
            j += 1
        # No admissible cut: a long rest follows; end the span at that rest.
        stop = best[1] if best is not None else max(index + 1, min(j + 1, len(beats)))
        spans.append((index, stop))
        index = stop
    # Merge a short tail into its predecessor. An isolated short passage remains
    # visible in diagnostics but is explicitly excluded from the review queue.
    if len(spans) > 1 and beats[-1] - beats[spans[-1][0]] < SEGMENT_MIN_BEATS:
        previous, tail_span = spans[-2], spans[-1]
        gap = beats[tail_span[0]] - beats[tail_span[0] - 1]
        if gap < SEGMENT_MIN_BEATS and beats[-1] - beats[previous[0]] <= SEGMENT_HARD_MAX_BEATS:
            spans[-2:] = [(previous[0], tail_span[1])]
    result = []
    conflict_times = [float(c["time_sec"]) for c in conflicts]
    for number, (a, b) in enumerate(spans):
        span_beats = beats[a:b]
        start_sec = seconds[span_beats[0]]
        last = span_beats[-1]
        tail = max([seconds[last]] + [end for begin, end in all_sustains if begin <= seconds[last]])
        end_sec = seconds[beats[b]] if b < len(beats) else tail + 0.25
        selected_notes = [n for n in notes if start_sec <= float(n["start_sec"]) <= seconds[last] + 1e-7]
        heads = sum(bool(n.get("has_head", True)) for n in selected_notes)
        support = None
        if envelope is not None and len(span_beats):
            grid = np.arange(len(envelope)) * FRAME_SEC
            values = np.interp(np.asarray([seconds[x] for x in span_beats]) + offset_sec, grid, envelope)
            threshold = support_threshold(envelope)
            support = round(float(np.mean(values >= threshold)), 3)
        duration = max(1e-6, end_sec - start_sec)
        result.append(dict(
            index=number, start_sec=round(start_sec, 4), end_sec=round(end_sec, 4),
            start_beat=str(span_beats[0]), end_beat=str(beats[b]) if b < len(beats) else None,
            beats=float((beats[b] if b < len(beats) else last) - span_beats[0]),
            queue_eligible=bool((beats[b] if b < len(beats) else last) - span_beats[0] >= SEGMENT_MIN_BEATS),
            onsets=len(span_beats), heads=heads, onset_rate=round(len(span_beats) / duration, 3),
            slides=sum(n.get("kind") == "slide" for n in selected_notes),
            holds=sum(n.get("kind") in {"hold", "touch_hold"} for n in selected_notes),
            sustained_crossing=(b < len(beats) and tail > end_sec + 1e-6),
            conflicts=sum(1 for t in conflict_times if start_sec <= t < end_sec),
            heads_complete=all(not n.get("unsupported") for n in selected_notes),
            attack_support=support, **_phase_flags(span_beats)))
    return result


# ---------------------------------------------------------------------------
# One chart
# ---------------------------------------------------------------------------

def scan_chart(maidata_path: str | Path, difficulty_index: int, audio_path: str | Path | None,
               *, cache_dir: str | Path | None = None) -> dict[str, Any]:
    from maichart.maidata import parse_maidata_file
    from maichart.event_timing import parse_event_chart, EventTempo
    from maichart.tempo_clock import TempoClock
    began = time.perf_counter()
    maidata_path = Path(maidata_path).resolve()
    issues: list[dict[str, Any]] = []
    def issue(severity, code, message, **extra):
        issues.append(dict(severity=severity, code=code, message=message, **extra))
    result: dict[str, Any] = dict(policy=POLICY, source_path=str(maidata_path),
                                  difficulty_index=int(difficulty_index),
                                  audio_path=str(Path(audio_path).resolve()) if audio_path else None)
    try:
        raw = parse_maidata_file(maidata_path)
        slot = next((d for d in raw.difficulties if d.index == int(difficulty_index)), None)
        if slot is None or not (slot.inote or "").strip():
            raise ValueError("缺少该难度槽位的谱面")
    except Exception as exc:  # noqa: BLE001 - every failure becomes a finding
        issue("error", "parse_failed", f"谱面无法读取：{exc}")
        return _finish(result, issues, began)
    result.update(title=raw.title, level=slot.level, designer=slot.designer,
                  content_key=content_key(slot.inote))
    if not (slot.level or "").strip():
        issue("error", "missing_level", "缺少等级，训练无法使用")
    try:
        raw_chart = parse_event_chart(maidata_path, int(difficulty_index), strict_notes=False)
    except Exception as exc:  # noqa: BLE001
        issue("error", "event_parse_failed", f"事件解析失败：{exc}")
        return _finish(result, issues, began)
    if not raw_chart.points:
        issue("error", "empty_chart", "没有任何音符")
        return _finish(result, issues, began)
    from maichart.composer_v2 import chart_dialect
    chart, dialect_edits = chart_dialect.normalize_training_chart(raw_chart)
    if any(dialect_edits.values()):
        result["dialect_edits"] = dialect_edits
    if not chart.points:
        issue("error", "empty_chart", "方言规范化后没有任何可训练音符")
        return _finish(result, issues, began)
    if dialect_edits.get("unsupported", 0) > 0:
        issue("suspicious", "unsupported_dialect_atoms",
              f"含 {dialect_edits['unsupported']} 处无法编码的异常音符（训练时会自动剥离）")
    try:
        from maichart.generation.event_compile import compile_event_points
        compile_event_points(chart.points, tempo_changes=chart.tempo_changes,
                             bpm=chart.tempo_changes[0].bpm if chart.tempo_changes else 120.,
                             first=chart.first, end_beat=chart.end_beat, build_ir=False, strict_notes=False)
    except Exception as exc:  # noqa: BLE001
        issue("error", "roundtrip_failed", f"解析→写出→再解析不一致：{exc}")
    changes = {change.beat: change.bpm for change in chart.tempo_changes}
    clock = TempoClock([EventTempo(b, v) for b, v in sorted(changes.items())], chart.audio_first)
    onset_seconds = np.asarray([float(clock.at(p.beat)) for p in chart.points])
    from maichart.webui.chart_timeline import timeline
    try:
        view = timeline(maidata_path, int(difficulty_index))
        notes = view["notes"]
        result["timeline_complete"] = not bool(view.get("warnings"))
        result["timeline_warnings"] = view.get("warnings", [])
        unsupported = view.get("warnings", [])
        if unsupported:
            issue("suspicious", "unsupported_notes", f"{len(unsupported)} 处写法无法完整还原时值/轨迹",
                  examples=unsupported[:5])
    except Exception as exc:  # noqa: BLE001
        notes = []
        result["timeline_complete"] = False
        issue("suspicious", "timeline_failed", f"时间轴无法生成：{exc}")
    action_end = max([float(n.get("end_sec", n["start_sec"])) for n in notes] + [float(onset_seconds.max())])
    stats = dict(events=len(chart.points), heads=sum(bool(n.get("has_head", True)) for n in notes),
                 heads_complete=result.get("timeline_complete", False),
                 first_onset_sec=round(float(onset_seconds.min()), 4), last_onset_sec=round(float(onset_seconds.max()), 4),
                 last_action_end_sec=round(action_end, 4), tempo_changes=len(changes),
                 bpm=round(float(chart.tempo_changes[0].bpm), 4) if chart.tempo_changes else None)
    result["stats"] = stats
    envelope = None
    align = None
    duration = None
    if not audio_path or not Path(audio_path).is_file():
        issue("error", "audio_missing", "音频文件不存在")
    else:
        try:
            _, flux, duration = cached_flux(audio_path, cache_dir)
            result["audio_duration_sec"] = round(duration, 3)
            if onset_seconds.min() < -1e-3:
                issue("error", "onset_before_audio", f"有音符早于音频开头（{onset_seconds.min():.3f} 秒）")
            if onset_seconds.max() > duration + 0.5:
                issue("error", "onset_after_audio", f"有音符晚于音频结尾（{onset_seconds.max():.2f} > {duration:.2f} 秒）")
            elif action_end > duration + 0.5:
                issue("suspicious", "action_after_audio", f"持续动作结束晚于音频（{action_end:.2f} > {duration:.2f} 秒）")
            align = alignment(onset_seconds, flux)
            result["alignment"] = align
            envelope = _normalized_envelope(flux)
            if not align.get("judgeable"):
                issue("suspicious", "alignment_unjudgeable", "起音不足，无法自动核对对齐")
            elif not align["confident"]:
                issue("suspicious", "alignment_uncertain",
                      f"自动对齐不确定（估计 {align['offset_ms']:+.0f} ms，峰值对比 {align['contrast']:.2f}，次峰比 {align['second_peak_ratio']:.2f}）")
            else:
                offset = abs(align["offset_ms"]) / 1000
                if offset > REVIEW_OFFSET_SEC:
                    issue("suspicious", "global_offset",
                          f"声学估计偏移约 {align['offset_ms']:+.0f} ms，需人工核对（候选 &first {align['suggested_first_delta_sec']:+.3f} 秒，仅为建议）",
                          suggested_first_delta_sec=align["suggested_first_delta_sec"])
                elif offset > OK_OFFSET_SEC:
                    issue("suspicious", "small_offset", f"可能有 {align['offset_ms']:+.0f} ms 偏移")
                if align.get("drift_ms") is not None and align["drift_ms"] > DRIFT_SEC * 1000 and len(changes) <= 1:
                    issue("suspicious", "tempo_drift",
                          f"前后段对齐相差 {align['drift_ms']:.0f} ms，可能 BPM 不准或变速未写")
        except Exception as exc:  # noqa: BLE001
            if duration is None:
                issue("error", "audio_unreadable", f"音频无法解码：{exc}")
            else:
                issue("suspicious", "alignment_failed", f"声学对齐检查失败：{exc}")
    conflicts = []
    inner_conflicts = []
    try:
        conflicts, inner_conflicts, parse_errors = _inspect_native_conflicts(chart, duration or action_end + 1.)
        if parse_errors:
            examples = [str(err.get("raw") or err.get("message") or "") for err in parse_errors[:3]]
            issue("error", "invalid_sustain_syntax",
                  f"含训练引擎无法解析的修饰符或时值语法（{', '.join(x for x in examples if x)}）",
                  parse_errors=parse_errors[:5])
        elif conflicts:
            issue("suspicious", "native_hard_conflicts", f"{len(conflicts)} 处原生硬冲突（多押/撞尾/不可达，可按区间屏蔽）",
                  times=[c["time_sec"] for c in conflicts][:32])
    except Exception as exc:  # noqa: BLE001
        issue("suspicious", "conflict_check_failed", f"原生冲突检查失败：{exc}")
    result["conflicts"] = conflicts[:64]
    if inner_conflicts:
        result["inner_slide_conflicts"] = inner_conflicts[:64]
    offset_sec = (align["offset_ms"] / 1000 + FLUX_BIAS_SEC) if align and align.get("judgeable") else 0.
    result["segments"] = propose_segments(chart.points, clock, notes, envelope=envelope,
                                          offset_sec=offset_sec, conflicts=conflicts)
    return _finish(result, issues, began)


def _finish(result, issues, began):
    severities = {i["severity"] for i in issues}
    result["status"] = "error" if "error" in severities else "suspicious" if "suspicious" in severities else "ok"
    result["issues"] = issues
    result["elapsed_sec"] = round(time.perf_counter() - began, 3)
    return result


# ---------------------------------------------------------------------------
# Manifest-level CLI (used for calibration and by the pool builder)
# ---------------------------------------------------------------------------

def iter_manifest_charts(manifest_path: str | Path):
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = Path(str(data.get("source_root") or "."))
    root = root if root.is_absolute() else (manifest_path.parent / root).resolve()
    def resolve(value):
        if not value:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else (root / path).resolve()
    for song in data.get("songs", []):
        for difficulty in song.get("difficulties", []):
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is None:
                continue
            yield dict(song_id=song.get("song_id"), maidata=resolve(song.get("maidata_path")),
                       audio=resolve(song.get("audio_path")), difficulty_index=int(index),
                       level=difficulty.get("level_raw", difficulty.get("level")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="webui_workspace/candidate_pools/_scan_cache")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Take every Nth chart (spread sampling)")
    parser.add_argument("--min-difficulty", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, args.threads))
    charts = [c for c in iter_manifest_charts(args.manifest) if c["difficulty_index"] >= args.min_difficulty]
    charts = charts[::max(1, args.stride)]
    if args.limit:
        charts = charts[:args.limit]
    results = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for number, chart in enumerate(charts, 1):
        row = scan_chart(chart["maidata"], chart["difficulty_index"], chart["audio"], cache_dir=args.cache_dir)
        row["song_id"] = chart["song_id"]
        results.append(row)
        print("candidate_scan " + json.dumps(dict(completed=number, total=len(charts), status=row["status"],
                                                   title=row.get("title")), ensure_ascii=False), flush=True)
    summary = dict(total=len(results), **{k: sum(r["status"] == k for r in results) for k in ("error", "suspicious", "ok")})
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(schema=POLICY, manifest=str(Path(args.manifest).resolve()),
                                         summary=summary, charts=results), ensure_ascii=False, indent=1),
                         encoding="utf-8")
    os.replace(temporary, output)
    print("candidate_scan_done " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
