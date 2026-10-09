"""Acceptance metrics: does a generated chart look like the gold set?

Three groups of measurements, all computed from a chart plus (optionally) the
song's cached audio features:

* **structure** -- how much of the chart is empty and how long its longest blank
  run is.  This is the failure mode the project hit with long training windows:
  a window that exhausts its token budget leaves the rest of its beats blank, so
  the chart shows large holes at regular intervals.
* **distribution** -- note density, chord/slide/touch/jack ratios and button
  entropy, taken from :func:`maichart.generation.event_quality.assess_event_chart`.
* **audio follow** -- the correlation between the song's own loudness envelope
  and the note density the chart writes over it.  A chart that ignores the music
  scores near zero here.

:func:`gold_reference` turns a list of per-chart metric dicts into percentiles;
:func:`evaluate` compares one chart against that reference and reports
PASS / WARN / FAIL per criterion with plain-language messages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from maichart.event_timing import parse_event_chart_text, seconds_at_beat
from maichart.generation.event_audio import (
    RELATIVE_RMS_COLUMN,
    EventAudio,
    audio_cache_dir,
    extract_event_audio,
)

# One "measure" of 4/4.  Blank detection works on this grid because a hole that
# does not even span one measure is not what players perceive as a blank.
MEASURE_BEATS = 4.0
FOLLOW_WINDOW_SEC = 2.0

# Metric key -> (Chinese label, one-line explanation for the WebUI).
# Everything here is compared against the gold percentiles.
GOLD_METRICS: tuple[tuple[str, str, str], ...] = (
    ("note_density_nps", "平均音符密度", "每秒平均音符数；太低谱面空洞，太高不现实"),
    ("peak_nps", "峰值密度", "最密一秒的音符数；反映爆发段强度"),
    ("chord_ratio", "和弦比例", "同时按多个键的比例；好谱面通常有稳定的和声使用量"),
    ("slide_ratio", "滑条比例", "滑条占比；风格指纹之一"),
    ("touch_ratio", "触摸音符比例", "触摸占比；风格指纹之一"),
    ("jack_ratio", "Jack 连打比例", "短时间内重复敲同一个键的比例"),
    ("entropy_buttons", "按键分布熵", "8 个键使用是否均衡；越高越均衡"),
    ("blank_fraction", "空白小节比例", "完全没有任何音符的小节占比（本次结构性问题的核心指标）"),
    ("max_blank_run_sec", "最长连续空白（秒）", "连续空小节的最长时长；超过 5 秒玩家会明显感到断档"),
    ("coverage_ratio", "谱面覆盖率", "最后一个音符位置占全曲时长比例；太低等于后半段没写"),
    ("rms_density_correlation", "音频贴合度", "音量包络与音符密度的相关性；越接近好谱面越贴合音乐"),
    ("longest_config_run_bars", "最长同型连段（小节）",
     "连续多少小节使用同一种配置（单键/和弦 × 有无滑条/触摸 × 疏密）。越大说明越像"
     "「一大段只有一类配置」，人写的谱面通常几小节就换一次"),
    ("configuration_switch_rate", "配置切换频率",
     "每小节发生配置切换的比例；过低就是块状同质化，过高则破碎无结构"),
    ("section_variety", "段落多样性",
     "把全曲切成多段后，各段配置分布的熵；越高说明各段差异越丰富"),
)


def configuration_metrics(
    points: Iterable[Any],
    tempo_changes: Sequence[Any],
    first: float = 0.0,
    *,
    bar_beats: float = MEASURE_BEATS,
    section_bars: int = 8,
) -> dict[str, Any]:
    """Detect "one long stretch of one style, then one long stretch of another".

    Every bar is labelled with a coarse configuration (single vs chord, plus
    slide / touch, plus dense vs sparse).  Real charts switch configuration
    every few bars; a decoder that picks a style and keeps it produces very long
    runs of one label, which is exactly the failure this measures.

    ``section_variety`` looks at it from the other side: it splits the song into
    sections of ``section_bars`` bars, describes each section by its label
    histogram, and reports the entropy of those histograms.  All sections
    identical -> low entropy.
    """

    import math as _math

    notes = _notes(points)
    if not notes:
        return {
            "longest_config_run_bars": None,
            "configuration_switch_rate": None,
            "section_variety": None,
            "configuration_labels": {},
        }
    bars: dict[int, list[Any]] = {}
    for note in notes:
        bars.setdefault(int(Fraction(note.beat) // int(bar_beats)), []).append(note)
    last_bar = max(bars)
    labels = [_bar_label(bars.get(index, [])) for index in range(last_bar + 1)]
    runs: list[int] = []
    length = 1
    for index in range(1, len(labels)):
        if labels[index] == labels[index - 1]:
            length += 1
        else:
            runs.append(length)
            length = 1
    runs.append(length)
    switches = sum(1 for index in range(1, len(labels)) if labels[index] != labels[index - 1])
    histogram = Counter(labels)
    # How varied the sections are: entropy of each section's label histogram,
    # averaged, normalised by the entropy of the whole song.
    sections: list[float] = []
    for start in range(0, len(labels), max(1, section_bars)):
        chunk = labels[start:start + section_bars]
        counts = Counter(chunk)
        total = len(chunk)
        sections.append(-sum((count / total) * _math.log(count / total) for count in counts.values()))
    overall = -sum((count / len(labels)) * _math.log(count / len(labels))
                   for count in histogram.values())
    return {
        "longest_config_run_bars": max(runs),
        "configuration_switch_rate": switches / max(1, len(labels) - 1),
        "section_variety": (sum(sections) / len(sections) / overall) if overall > 1e-9 else 0.0,
        "configuration_labels": dict(histogram.most_common()),
    }


def _bar_label(points: Sequence[Any]) -> str:
    """Coarse "what does this bar look like" label."""

    if not points:
        return "blank"
    texts = [str(getattr(point, "note_text", "")) for point in points]
    joined = "|".join("".join(text.split()) for text in texts)
    chord = "chord" if "/" in joined else "single"
    if any(character in joined for character in "<>vVpqszw"):
        kind = "slide"
    elif "E" in joined or "!" in joined:
        kind = "touch"
    else:
        kind = "plain"
    density = "dense" if len(points) >= 3 else "sparse"
    return f"{kind}-{chord}-{density}"


def _notes(points: Iterable[Any]) -> list[Any]:
    return [point for point in points if getattr(point, "note_text", "")]


def _measure_seconds(tempo_changes: Sequence[Any], first: float, beat: Any) -> float:
    """Beat -> seconds, converting to an exact rational first.

    ``seconds_at_beat`` deliberately rejects float beats (a float beat cannot be
    integrated exactly), so computed boundaries are converted through their
    decimal text, which keeps values such as 415.5 exact.
    """

    if isinstance(beat, Fraction):
        exact = beat
    elif isinstance(beat, int):
        exact = Fraction(beat)
    else:
        exact = Fraction(str(float(beat))).limit_denominator(1_000_000)
    return float(seconds_at_beat(exact, tempo_changes, first))


def structural_metrics(
    points: Iterable[Any],
    tempo_changes: Sequence[Any],
    first: float = 0.0,
    *,
    measure_beats: float = MEASURE_BEATS,
    audio_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Blank structure of a chart, measured on a 4/4 measure grid."""

    notes = _notes(points)
    if not notes:
        return {
            "notes": 0, "measure_beats": measure_beats, "measures": 0,
            "blank_measures": 0, "blank_fraction": None, "max_blank_run_measures": 0,
            "max_blank_run_sec": 0.0, "first_note_sec": None, "last_note_sec": None,
            "coverage_ratio": None, "note_density_nps": 0.0,
        }
    beats = [float(point.beat) for point in notes]
    last_beat = max(beats)
    # Measure 0 covers beats [0, 4); the last measure is the one holding the
    # final note, so every occupied index is < measures by construction.
    measures = int(math.floor(max(last_beat, 0.0) / measure_beats)) + 1
    occupied = {int(beat // measure_beats) for beat in beats}
    # Longest run of consecutive measures without a single note.
    longest, current, run_start, best_start = 0, 0, None, 0
    for index in range(measures):
        if index in occupied:
            current, run_start = 0, None
            continue
        if run_start is None:
            run_start = index
        current += 1
        if current > longest:
            longest, best_start = current, run_start
    first_sec = _measure_seconds(tempo_changes, first, min(beats))
    last_sec = _measure_seconds(tempo_changes, first, last_beat)
    run_end_beat = min((best_start + longest) * measure_beats, last_beat)
    run_sec = max(
        _measure_seconds(tempo_changes, first, run_end_beat)
        - _measure_seconds(tempo_changes, first, best_start * measure_beats),
        0.0,
    )
    duration = float(audio_duration_sec) if audio_duration_sec else last_sec
    span = max(last_sec - first_sec, 1e-6)
    return {
        "notes": len(notes),
        "measure_beats": measure_beats,
        "measures": measures,
        "blank_measures": measures - len(occupied),
        "blank_fraction": (measures - len(occupied)) / measures,
        "max_blank_run_measures": longest,
        "max_blank_run_sec": run_sec,
        "first_note_sec": first_sec,
        "last_note_sec": last_sec,
        "coverage_ratio": last_sec / duration if duration > 0 else None,
        "note_density_nps": len(notes) / span,
    }


def _feature_index(times: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.zeros(len(seconds), dtype=np.int64)
    index = np.searchsorted(times, seconds, side="left")
    return np.clip(index, 0, len(times) - 1)


def audio_follow_metrics(
    points: Iterable[Any],
    audio: EventAudio,
    tempo_changes: Sequence[Any],
    first: float = 0.0,
    *,
    duration_sec: float | None = None,
    window_sec: float = FOLLOW_WINDOW_SEC,
) -> dict[str, Any]:
    """Correlation between the song's loudness envelope and note density."""

    notes = _notes(points)
    duration = float(duration_sec if duration_sec is not None else audio.duration_sec)
    if not notes or duration <= 0 or window_sec <= 0:
        return {"rms_density_correlation": None, "windows": 0, "window_sec": window_sec}
    note_seconds = np.asarray(
        [_measure_seconds(tempo_changes, first, float(point.beat)) for point in notes], dtype=np.float64
    )
    edges = np.arange(0.0, duration + window_sec, window_sec)
    if len(edges) < 3:
        return {"rms_density_correlation": None, "windows": 0, "window_sec": window_sec}
    counts, _ = np.histogram(note_seconds, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2.0
    intensity = audio.features[:, RELATIVE_RMS_COLUMN] if audio.features.size else np.zeros(1, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float64)
    loudness = np.zeros(len(centres), dtype=np.float64)
    for index, centre in enumerate(centres):
        low = max(0.0, centre - window_sec / 2.0)
        high = min(float(audio.times[-1]) if len(audio.times) else duration, centre + window_sec / 2.0)
        selected = (audio.times >= low) & (audio.times < max(high, low + 1e-6))
        loudness[index] = float(intensity[selected].mean()) if selected.any() else 0.0
    if counts.std() < 1e-9 or loudness.std() < 1e-9:
        correlation = None
    else:
        correlation = float(np.corrcoef(counts.astype(np.float64), loudness)[0, 1])
    return {
        "rms_density_correlation": correlation,
        "windows": int(len(counts)),
        "window_sec": float(window_sec),
    }


def timeline_series(
    points: Iterable[Any],
    audio: EventAudio | None,
    tempo_changes: Sequence[Any],
    first: float = 0.0,
    *,
    duration_sec: float | None = None,
    resolution: int = 240,
    measure_beats: float = MEASURE_BEATS,
) -> dict[str, Any]:
    """Compact series for the WebUI canvas: intensity, density, blank spans."""

    notes = _notes(points)
    duration = float(duration_sec if duration_sec is not None else (audio.duration_sec if audio else 0.0))
    if duration <= 0 or resolution < 2:
        return {"points": [], "blank_spans": [], "duration_sec": duration}
    note_seconds = np.asarray(
        [_measure_seconds(tempo_changes, first, float(point.beat)) for point in notes], dtype=np.float64
    )
    sample_times = np.linspace(0.0, duration, int(resolution))
    if len(note_seconds):
        counts, _ = np.histogram(note_seconds, bins=np.append(sample_times, duration + 1e-6))
    else:
        counts = np.zeros(len(sample_times), dtype=np.int64)
    loudness = np.zeros(len(sample_times), dtype=np.float64)
    if audio is not None and audio.features.size:
        intensity = np.asarray(audio.features[:, RELATIVE_RMS_COLUMN], dtype=np.float64)
        loudness = intensity[_feature_index(np.asarray(audio.times, dtype=np.float64), sample_times)]
    # Blank measures, expressed in seconds so the UI can shade them directly.
    spans: list[list[float]] = []
    if notes:
        beats = [float(point.beat) for point in notes]
        measures = int(math.floor(max(beats) / measure_beats)) + 1
        occupied = {int(beat // measure_beats) for beat in beats}
        run_start = None
        for index in range(measures + 1):
            blank = index not in occupied and index < measures
            if blank and run_start is None:
                run_start = index
            elif not blank and run_start is not None:
                spans.append([
                    _measure_seconds(tempo_changes, first, run_start * measure_beats),
                    _measure_seconds(tempo_changes, first, index * measure_beats),
                ])
                run_start = None
    return {
        "points": [
            {"t": float(sample_times[index]), "intensity": float(loudness[index]),
             "density": float(counts[index])}
            for index in range(len(sample_times))
        ],
        "blank_spans": spans,
        "duration_sec": duration,
    }


def metric_bundle(
    points: Iterable[Any],
    audio: EventAudio | None,
    tempo_changes: Sequence[Any],
    first: float = 0.0,
    *,
    duration_sec: float | None = None,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every acceptance metric for one chart, as a flat dict."""

    points = list(points)
    duration = float(duration_sec if duration_sec is not None else (audio.duration_sec if audio else 0.0))
    metrics: dict[str, Any] = {}
    structure = structural_metrics(
        points, tempo_changes, first, audio_duration_sec=duration or None
    )
    metrics.update({key: structure[key] for key in
                    ("blank_fraction", "max_blank_run_sec", "coverage_ratio", "note_density_nps")})
    metrics["measures"] = structure["measures"]
    metrics["blank_measures"] = structure["blank_measures"]
    metrics["notes"] = structure["notes"]
    if audio is not None:
        metrics.update(audio_follow_metrics(points, audio, tempo_changes, first, duration_sec=duration))
    else:
        metrics["rms_density_correlation"] = None
    if quality is None:
        from maichart.generation.event_quality import assess_event_chart
        quality = assess_event_chart(points, tempo_changes, first=first,
                                     duration_sec=duration if duration > 0 else None)
    metrics["peak_nps"] = quality.get("peak_nps")
    metrics["chord_ratio"] = quality.get("chord_ratio")
    jack = quality.get("jack") or {}
    metrics["jack_ratio"] = jack.get("ratio")
    entropy = quality.get("entropy") or {}
    metrics["entropy_buttons"] = entropy.get("buttons")
    ratios = quality.get("type_ratios") or {}
    metrics["slide_ratio"] = ratios.get("slide")
    metrics["touch_ratio"] = (ratios.get("touch") or 0.0) + (ratios.get("touch_hold") or 0.0)
    metrics["syntax_valid"] = bool(quality.get("syntax_valid"))
    metrics["duplicate_head_count"] = len([
        issue for issue in quality.get("fatal_errors", [])
        if issue.get("code") == "duplicate_button_head"
    ])
    # Block-homogeneity diagnostics: the "one long stretch of one style" failure.
    configuration = configuration_metrics(points, tempo_changes, first)
    metrics.update({key: configuration[key] for key in
                    ("longest_config_run_bars", "configuration_switch_rate", "section_variety")})
    metrics["configuration_labels"] = configuration["configuration_labels"]
    return metrics


def _percentiles(values: Sequence[float]) -> dict[str, float] | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if len(clean) < 3:
        return None

    def pick(fraction: float) -> float:
        index = min(len(clean) - 1, max(0, int(round(fraction * (len(clean) - 1)))))
        return clean[index]

    return {
        "p05": pick(0.05), "p10": pick(0.10), "p50": pick(0.50),
        "p90": pick(0.90), "p95": pick(0.95), "n": len(clean),
    }


def gold_reference(metric_sets: Sequence[dict[str, Any]], *, label: str = "gold") -> dict[str, Any]:
    """Percentile bands per metric, computed from the gold charts."""

    reference: dict[str, Any] = {"label": label, "charts": len(metric_sets), "metrics": {}}
    for key, _label, _help in GOLD_METRICS:
        band = _percentiles([item.get(key) for item in metric_sets])
        if band is not None:
            reference["metrics"][key] = band
    return reference


def _verdict_for(value: float, band: dict[str, float]) -> str:
    if band["p10"] <= value <= band["p90"]:
        return "pass"
    if band["p05"] <= value <= band["p95"]:
        return "warn"
    return "fail"


def evaluate(metrics: dict[str, Any], reference: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare one chart's metrics against the gold reference (or absolutes only)."""

    criteria: list[dict[str, Any]] = []

    def add(key: str, label: str, help_text: str, value: Any, low: Any, high: Any, verdict: str,
            message: str) -> None:
        criteria.append({
            "key": key, "label": label, "help": help_text, "value": value,
            "low": low, "high": high, "verdict": verdict, "message": message,
        })

    # Absolute criteria: these are not "how much like the gold set", they are
    # defects that must not exist at all.
    syntax_ok = bool(metrics.get("syntax_valid"))
    add("syntax_valid", "语法合法", "谱面必须能被解析器完整解析",
        1 if syntax_ok else 0, 1, 1, "pass" if syntax_ok else "fail",
        "语法通过" if syntax_ok else "谱面存在无法解析的音符，需要修谱")
    duplicates = int(metrics.get("duplicate_head_count") or 0)
    duplicate_verdict = "pass" if duplicates == 0 else ("warn" if duplicates <= 2 else "fail")
    add("duplicate_head_count", "重复按键", "同一格内同一个键被敲两次属于非法单元",
        duplicates, 0, 0, duplicate_verdict,
        "无重复按键" if duplicates == 0 else f"有 {duplicates} 处同格重复按键")

    bands = (reference or {}).get("metrics") or {}
    for key, label, help_text in GOLD_METRICS:
        value = metrics.get(key)
        band = bands.get(key)
        if value is None:
            add(key, label, help_text, None, None, None, "unknown", "本次没有计算出该指标")
            continue
        value = float(value)
        if band is None:
            add(key, label, help_text, value, None, None, "unknown", "金标集数据不足，无法给出参考区间")
            continue
        verdict = _verdict_for(value, band)
        if verdict == "pass":
            message = "落在金标区间内"
        elif verdict == "warn":
            message = ("高于金标 p90" if value > band["p90"] else "低于金标 p10")
        else:
            message = ("明显高于金标 p95" if value > band["p95"] else "明显低于金标 p05")
        add(key, label, help_text, value, band["p10"], band["p90"], verdict, message)

    order = {"fail": 0, "warn": 1, "unknown": 2, "pass": 3}
    failures = sum(1 for item in criteria if item["verdict"] == "fail")
    warnings = sum(1 for item in criteria if item["verdict"] == "warn")
    unknowns = sum(1 for item in criteria if item["verdict"] == "unknown")
    overall = "fail" if failures else ("warn" if warnings else ("pass" if not unknowns else "warn"))
    criteria.sort(key=lambda item: order[item["verdict"]])
    return {
        "schema": "maichart-acceptance-v1",
        "verdict": overall,
        "failures": failures,
        "warnings": warnings,
        "unknowns": unknowns,
        "reference": (reference or {}).get("label"),
        "reference_charts": (reference or {}).get("charts"),
        "criteria": criteria,
    }


def load_reference(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_maidata(
    maidata_path: str | Path,
    *,
    audio_path: str | Path | None = None,
    difficulty: int = 5,
    cache_dir: str | Path | None = None,
    reference: dict[str, Any] | None = None,
    bpm: float | None = None,
    include_timeline: bool = True,
    resolution: int = 240,
) -> dict[str, Any]:
    """Parse a maidata file, measure it, and evaluate it against ``reference``."""

    source = Path(maidata_path)
    text = source.read_text(encoding="utf-8")
    chart = parse_event_chart_text(text, int(difficulty), strict_notes=False)
    audio = None
    if audio_path is not None:
        audio = extract_event_audio(audio_path, cache_dir=audio_cache_dir(cache_dir))
    duration = float(audio.duration_sec) if audio is not None else float(chart.end_beat) * 0.5
    metrics = metric_bundle(
        chart.points, audio, chart.tempo_changes, first=chart.first, duration_sec=duration
    )
    report = evaluate(metrics, reference)
    report["source"] = str(source)
    report["audio"] = str(audio_path) if audio_path else None
    report["difficulty"] = int(difficulty)
    report["metrics"] = metrics
    if bpm:
        report["bpm"] = float(bpm)
    if include_timeline:
        report["timeline"] = timeline_series(
            chart.points, audio, chart.tempo_changes, first=chart.first,
            duration_sec=duration, resolution=resolution,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acceptance metrics for a generated chart.")
    parser.add_argument("maidata", help="generated maidata.txt")
    parser.add_argument("--audio", help="song audio used for generation (enables audio-follow metrics)")
    parser.add_argument("--gold-stats", help="gold_stats.json produced by maichart.training.gold_set")
    parser.add_argument("--difficulty", type=int, default=5)
    parser.add_argument("--cache-dir", help="audio feature cache directory")
    parser.add_argument("--bpm", type=float, help="BPM to record in the report")
    parser.add_argument("--resolution", type=int, default=240, help="timeline points for the WebUI")
    parser.add_argument("-o", "--output", help="write the report JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reference = load_reference(args.gold_stats) if args.gold_stats else None
    report = evaluate_maidata(
        args.maidata, audio_path=args.audio, difficulty=args.difficulty,
        cache_dir=args.cache_dir, reference=reference, bpm=args.bpm,
        resolution=args.resolution,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(f"acceptance_verdict={report['verdict']} failures={report['failures']} "
          f"warnings={report['warnings']} unknowns={report['unknowns']}", flush=True)
    for item in report["criteria"]:
        value = item["value"]
        value_text = "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        band = "" if item["low"] is None else f" [gold {item['low']:.4f}..{item['high']:.4f}]"
        print(f"  {item['verdict']:<7} {item['label']:<14} {value_text:>12}{band}  {item['message']}", flush=True)
    if args.output:
        print(f"report={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
