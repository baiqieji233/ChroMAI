"""Corpus-level condition profile: the defaults generation should actually use.

Why this exists
---------------
``condition_from_events`` derives the 16 control values from the chart being
predicted.  That is fine for training, but it means the model can read the
answer's summary off the condition instead of following the audio, and it means
generation must supply numbers in the same range the model saw.  The UI used to
send only ``level`` and let every other control default to 0.5 -- while the real
corpus medians are 0.00-0.30 for most controls.

Measured on this project's corpus (60 charts): 14 of 16 controls differ from
0.5 by more than 0.15, the worst being ``melody_following`` (0.000),
``pattern_reuse`` (0.010), ``touch_bias`` (0.066) and ``chord`` (0.267).  Feeding
0.5 asks the model for a far more chordy, far more touchy chart than anything it
was trained on, and the decoder then holds that one style for hundreds of beats.

So: measure the corpus, store the medians, and use them as the generation
default.  The profile is a small JSON file written next to the training manifest.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from maichart.generation.conditions import CONDITION_KEYS, GenerationCondition, RADAR_KEYS, STYLE_KEYS

PROFILE_SCHEMA = "maichart-condition-profile-v1"
PROFILE_NAME = "condition_profile.json"


def _chart_duration(chart: Any) -> float:
    from maichart.event_timing import seconds_at_beat

    try:
        return float(seconds_at_beat(chart.end_beat, chart.tempo_changes, chart.first))
    except Exception:  # noqa: BLE001 - an unmeasurable chart is skipped
        return 0.0


def _level_of(difficulty: dict[str, Any]) -> float:
    raw = str(difficulty.get("level_raw") or difficulty.get("level") or "")
    try:
        return float(raw[:-1]) + 0.7 if raw.endswith("+") else float(raw)
    except ValueError:
        return 0.0


def measure_condition_profile(
    manifest_path: str | Path,
    *,
    sample: int | None = None,
    min_level: float = 0.0,
) -> dict[str, Any]:
    """Median/p10/p90 of every control, measured over the corpus."""

    from maichart.event_timing import parse_event_chart
    from maichart.generation.event_quality import condition_from_events

    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    base = source.parent
    vectors: list[list[float]] = []
    levels: list[float] = []
    skipped: list[dict[str, str]] = []
    for song in manifest.get("songs", []):
        if sample is not None and len(vectors) >= sample:
            break
        for difficulty in song.get("difficulties", []):
            if sample is not None and len(vectors) >= sample:
                break
            if not (difficulty.get("event_usable_for_training") or difficulty.get("usable_for_training")):
                continue
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is None or int(index) < 4:
                continue
            level = _level_of(difficulty)
            if level < min_level:
                continue
            try:
                chart = parse_event_chart(base / song["maidata_path"], int(index))
                duration = _chart_duration(chart)
                if duration <= 0:
                    raise ValueError("non-positive chart duration")
                condition = condition_from_events(
                    chart.points, chart.tempo_changes, level=level, duration_sec=duration, audio=None,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                skipped.append({
                    "song_id": str(song.get("song_id", "?")),
                    "difficulty_index": str(index),
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                })
                continue
            vectors.append(condition.to_vector().tolist())
            levels.append(level)

    if not vectors:
        raise ValueError(
            f"没有可测量的谱面：{source}。请确认 manifest 里有 event_usable_for_training 的 Master/Re:Master 难度。"
        )

    controls: dict[str, Any] = {}
    for position, key in enumerate(CONDITION_KEYS):
        column = sorted(row[position] for row in vectors)
        if key == "level":
            continue

        # Rounded indices, matching acceptance._percentiles: plain truncation
        # inverts the band on tiny samples (with two charts p10 and p90 both land
        # on the minimum, below the median).
        def pick(fraction: float, _column: list[float] = column) -> float:
            index = min(len(_column) - 1, max(0, int(round(fraction * (len(_column) - 1)))))
            return _column[index]

        controls[key] = {
            "p10": pick(0.10),
            "median": pick(0.50),
            "p90": pick(0.90),
            "mean": statistics.mean(column),
            "min": column[0],
            "max": column[-1],
        }
    return {
        "schema": PROFILE_SCHEMA,
        "source_manifest": str(source),
        "charts": len(vectors),
        "skipped": len(skipped),
        "skipped_examples": skipped[:10],
        "level": {
            "min": min(levels), "median": statistics.median(levels), "max": max(levels),
        },
        "controls": controls,
    }


def profile_default_vector(profile: dict[str, Any]) -> list[float]:
    """The condition vector a generator should assume when the user gives none."""

    controls = profile.get("controls", {})
    values = [0.5]  # level is replaced by the caller
    for key in RADAR_KEYS + STYLE_KEYS:
        entry = controls.get(key)
        values.append(float(entry["median"]) if entry else 0.5)
    return values


def condition_from_profile(profile: dict[str, Any], *, level: float | None = None) -> GenerationCondition:
    vector = profile_default_vector(profile)
    resolved = float(level) if level is not None else float(profile.get("level", {}).get("median") or 13.0)
    radar = {key: vector[1 + index] for index, key in enumerate(RADAR_KEYS)}
    style = {key: vector[1 + len(RADAR_KEYS) + index] for index, key in enumerate(STYLE_KEYS)}
    return GenerationCondition(level=resolved, radar=radar, style=style)


def load_profile(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"不是条件基线档案：{path}")
    return data


def default_profile_path(manifest_path: str | Path) -> Path:
    return Path(manifest_path).resolve().parent / PROFILE_NAME


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the corpus condition profile used as generation defaults.",
    )
    parser.add_argument("manifest")
    parser.add_argument("-o", "--out", help=f"output path (default: <manifest dir>/{PROFILE_NAME})")
    parser.add_argument("--sample", type=int, default=0, help="measure only the first N charts (0 = all)")
    parser.add_argument("--min-level", type=float, default=0.0)
    args = parser.parse_args(argv)

    profile = measure_condition_profile(
        args.manifest, sample=args.sample or None, min_level=args.min_level,
    )
    target = Path(args.out) if args.out else default_profile_path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"condition_profile charts={profile['charts']} skipped={profile['skipped']} -> {target}", flush=True)
    for item in profile.get("skipped_examples", [])[:5]:
        print(f"  skipped {item['song_id']}:{item['difficulty_index']} — {item['error']}", flush=True)
    print(f"levels: {profile['level']['min']:.1f} .. {profile['level']['max']:.1f} "
          f"(median {profile['level']['median']:.1f})", flush=True)
    for key, entry in profile["controls"].items():
        gap = abs(entry["median"] - 0.5)
        mark = "  <-- was 0.5, far off" if gap > 0.15 else ""
        print(f"  {key:<18} median={entry['median']:.3f} "
              f"p10={entry['p10']:.3f} p90={entry['p90']:.3f}{mark}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
