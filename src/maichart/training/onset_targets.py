"""Frame-aligned chart onset supervision for event audio windows."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import torch

from maichart.event_timing import seconds_at_beat
from maichart.generation.event_quality import parse_event_notes

ONSET_PROFILES = frozenset({"none", "chart_onsets_v1"})
ONSET_SIGMA_SEC = 0.02


def validate_onset_profile(profile: str) -> str:
    if profile not in ONSET_PROFILES:
        raise ValueError(f"unknown onset_profile {profile!r}; expected none or chart_onsets_v1")
    return profile


def onset_times(chart: Any) -> list[float]:
    """Unique playable onset seconds, mapped through the chart tempo clock."""
    parsed = parse_event_notes(chart.points, chart.tempo_changes, first=chart.first)
    if not parsed.syntax_valid:
        raise ValueError("onset supervision requires a syntactically valid event chart")
    beats = sorted({Fraction(note.beat) for note in parsed.notes})
    return [float(seconds_at_beat(beat, chart.tempo_changes, chart.audio_first)) for beat in beats]


def _region_at(timestamp: float, regions: Sequence[Mapping[str, Any]]) -> tuple[int, float, float, float]:
    for index, region in enumerate(regions, start=1):
        start = float(region["start_sec"])
        end = float(region["end_sec"])
        if start <= timestamp < end:
            factor = float(region.get("weight", 1.0))
            return (index if factor <= 0.0 else 0), start, end, factor
    return 0, -math.inf, math.inf, 1.0


def build_onset_targets(
    chart: Any,
    sample_times: Sequence[float] | torch.Tensor,
    *,
    core_start_sec: float,
    core_end_sec: float,
    audio_duration_sec: float,
    quality_base_weight: float = 1.0,
    quality_regions: Sequence[Mapping[str, Any]] = (),
    sigma_sec: float = ONSET_SIGMA_SEC,
    onset_seconds: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build raw onset target, quality weight, and core/audio validity vectors.

    Validity is independent of quality and covers only timestamps in the
    half-open core and real audio interval. Quality weights follow annotation
    base and region multipliers. Gaussian support is constrained to its source
    annotation region so zero-weight regions cannot create positive spillover.
    """
    times = torch.as_tensor(sample_times, dtype=torch.float64).flatten()
    core_start, core_end = float(core_start_sec), float(core_end_sec)
    duration = max(0.0, float(audio_duration_sec))
    base = float(quality_base_weight)
    sigma = float(sigma_sec)
    if not math.isfinite(base) or not 0.0 <= base <= 4.0:
        raise ValueError("quality_base_weight must be finite and within [0, 4].")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_sec must be finite and positive.")
    if not math.isfinite(core_start) or not math.isfinite(core_end) or core_end < core_start:
        raise ValueError("core interval must be finite and nondecreasing.")

    valid = ((times >= core_start) & (times < core_end)
             & (times >= 0.0) & (times < duration)).to(torch.float32)
    weights = torch.zeros_like(times, dtype=torch.float32)
    zones: list[int] = []
    for index, timestamp in enumerate(times.tolist()):
        zone, _, _, factor = _region_at(timestamp, quality_regions)
        zones.append(zone)
        weights[index] = max(0.0, min(4.0, base * factor))

    target = torch.zeros_like(times, dtype=torch.float32)
    events = [float(value) for value in (onset_seconds if onset_seconds is not None else onset_times(chart))
              if core_start <= float(value) < core_end and 0.0 <= float(value) < duration]
    active_events = []
    active_zones = []
    for event_sec in events:
        event_zone, _, _, event_factor = _region_at(event_sec, quality_regions)
        if base > 0.0 and event_factor > 0.0:
            active_events.append(event_sec)
            active_zones.append(event_zone)
    if active_events and times.numel():
        event_tensor = torch.tensor(active_events, dtype=torch.float64)
        event_zone_tensor = torch.tensor(active_zones, dtype=torch.int64).unsqueeze(1)
        frame_zone_tensor = torch.tensor(zones, dtype=torch.int64).unsqueeze(0)
        allowed = ((event_zone_tensor == frame_zone_tensor)
                   & (valid.unsqueeze(0) > 0)
                   & (weights.unsqueeze(0) > 0))
        distances = (times.unsqueeze(0) - event_tensor.unsqueeze(1)) / sigma
        gaussians = torch.exp(-0.5 * distances.square()).float()
        target = torch.where(allowed, gaussians, torch.zeros_like(gaussians)).amax(dim=0)
    return target, weights, valid


def quality_onset_params(record: Mapping[str, Any] | None) -> tuple[float, Sequence[Mapping[str, Any]]]:
    """Return chart-level and local weights matching EventDataset semantics."""
    if record is None:
        return 1.0, ()
    label = str(record.get("label", "normal"))
    regions = record.get("regions", ())
    if label == "excluded" or (label == "negative" and not regions):
        return 0.0, regions
    base = 1.0 if label == "negative" else float(record.get("weight", 1.0))
    return max(0.0, min(4.0, base)), regions
