"""Boundary snapshots of active MiaCode-style sustained chart actions.

This is an input feature over authored notes and the pinned reference paths. It
does not simulate a player's body or certify that a chart is playable.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Any, Iterable

import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.generation.action_state_spec import MAX_STATES, PROFILE, STATE_DIM
from maichart.generation.event_compile import compile_event_points
from maichart.generation.muri.data import reference_data
from maichart.generation.muri.model import CENTERS, PADS, RADII, RELEASE
from maichart.generation.muri.parser import parse_markers
from maichart.maidata import parse_maidata_metadata


HAND_RADIUS = 15.0
MAX_SECONDS_SCALE = 60.0
_TYPE_OFFSETS = {"hold": 0, "touch_hold": 1, "slide": 2, "wifi": 3}
_PAD_OFFSET = 4
_LANE_OFFSET = _PAD_OFFSET + len(PADS)
_WAITING_OFFSET = _LANE_OFFSET + 8
_REMAINING_OFFSET = _WAITING_OFFSET + 1
_UNTIL_TRACE_OFFSET = _REMAINING_OFFSET + 1
_PROGRESS_OFFSET = _UNTIL_TRACE_OFFSET + 1
_EX_OFFSET = _PROGRESS_OFFSET + 1
_EPS = 1e-9


def action_state_kwargs(batch: dict, *, batched: bool = True) -> dict:
    if "action_state" not in batch and "action_state_padding_mask" not in batch:
        return {}
    if "action_state" not in batch or "action_state_padding_mask" not in batch:
        raise ValueError("Incomplete sustained-action batch")
    return {key: batch[key] if batched else batch[key].unsqueeze(0)
            for key in ("action_state", "action_state_padding_mask")}


def neutral_action_state(model, *, batch_size: int = 1, device=None) -> dict:
    """Audio-only curve prediction has no authored/generated prior chart."""
    if getattr(getattr(model, "config", None), "action_state_profile", "none") == "none":
        return {}
    return {"action_state": torch.zeros(batch_size, 1, STATE_DIM, device=device),
            "action_state_padding_mask": torch.ones(batch_size, 1, dtype=torch.bool, device=device)}


def _as_beat(value: Fraction | int | str | float) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("start_beat must be an exact beat value")
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _unit_log_seconds(seconds: float) -> float:
    value = max(0.0, min(MAX_SECONDS_SCALE, float(seconds)))
    return math.log1p(value) / math.log1p(MAX_SECONDS_SCALE)


def _sample_path(path: list[dict[str, Any]], progress: float) -> tuple[float, float]:
    if not path:
        raise ValueError("MiaCode path reference is empty")
    if len(path) == 1:
        item = path[0]
        return float(item["x"]), float(item["y"])
    position = max(0.0, min(1.0, progress)) * (len(path) - 1)
    left = min(int(position), len(path) - 1)
    right = min(left + 1, len(path) - 1)
    ratio = position - left
    a, b = path[left], path[right]
    return (
        float(a["x"]) + (float(b["x"]) - float(a["x"])) * ratio,
        float(a["y"]) + (float(b["y"]) - float(a["y"])) * ratio,
    )


def _path_pads(position: tuple[float, float], radius: float = HAND_RADIUS) -> set[str]:
    return {
        pad for pad in PADS
        if math.dist(position, CENTERS[pad]) <= radius + RADII[pad[0]] + _EPS
    }


def _wifi_end_lanes(end_lane: int) -> tuple[int, int, int]:
    # MiaCode's Wifi end-lane checks cover the central endpoint and its two
    # neighboring A pads, with wraparound at lanes 1 and 8.
    return ((end_lane + 6) % 8 + 1, end_lane, end_lane % 8 + 1)


def _wifi_paths(entry: dict[str, Any]) -> list[list[dict[str, Any]]]:
    paths = entry.get("di_real_path_samples") or entry.get("lane_samples") or []
    if len(paths) < 2:
        raise ValueError("MiaCode Wifi reference requires its two outer action paths")
    return [paths[0], paths[-1]]


class SustainStateIndex:
    """Parse a chart once and return bounded exact-clock state at any boundary."""

    def __init__(self, points: Iterable[EventPoint], tempo_changes: Iterable[EventTempo], first: float):
        self.points = tuple(EventPoint(point.beat, point.note_text) for point in points)
        self.tempo_changes = tuple(EventTempo(Fraction(t.beat), float(t.bpm)) for t in tempo_changes)
        self.first = float(first)
        bpm = float(self.tempo_changes[0].bpm) if self.tempo_changes else 120.0
        text, _ = compile_event_points(
            self.points,
            tempo_changes=self.tempo_changes,
            bpm=bpm,
            first=self.first,
        )
        parsed_chart = parse_maidata_metadata(text)
        inote = parsed_chart.fields.get("inote_5")
        if inote is None:
            raise ValueError("Compiled event chart has no inote_5 field")
        self.markers, errors, _warnings = parse_markers(inote, first=self.first)
        if errors:
            detail = "; ".join(str(item.get("message", item)) for item in errors[:3])
            raise ValueError(f"MiaCode sustain input contains unsupported notes: {detail}")
        self.markers = [m for m in self.markers if m["type"] in _TYPE_OFFSETS]
        self._references: dict[str, Any] | None = None

    def at(self, start_beat: Fraction | int | str | float) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        beat = _as_beat(start_beat)
        boundary = float(seconds_at_beat(beat, self.tempo_changes, self.first))
        selected: list[tuple[float, int, dict[str, Any], float, bool]] = []
        for marker in self.markers:
            kind = marker.get("type")
            if kind not in _TYPE_OFFSETS or marker.get("is_mine") or marker.get("track_mine"):
                continue
            onset = float(marker.get("second", 0.0))
            if onset >= boundary - _EPS:
                continue
            release = self._release_for(marker, boundary)
            end = float(marker.get("end_second", -1.0))
            if not math.isfinite(end):
                raise ValueError(f"MiaCode marker has no end time: {marker.get('raw', '')}")
            deadline = end + release
            trace = float(marker.get("slide_trace_second", -1.0))
            waiting = kind in {"slide", "wifi"} and trace > boundary + _EPS
            if deadline <= boundary + _EPS:
                continue
            if kind in {"slide", "wifi"} and (not math.isfinite(trace) or trace < onset - _EPS):
                raise ValueError(f"MiaCode {kind} marker has no trace time: {marker.get('raw', '')}")
            selected.append((deadline, int(marker.get("parse_order", -1)), marker, release, waiting))

        selected.sort(key=lambda item: (item[0], item[1], int(item[2].get("line", 0)),
                                        int(item[2].get("col", 0))))
        total = len(selected)
        retained = selected[:MAX_STATES]
        if not retained:
            states = torch.zeros((1, STATE_DIM), dtype=torch.float32)
            mask = torch.ones((1,), dtype=torch.bool)
        else:
            states = torch.zeros((len(retained), STATE_DIM), dtype=torch.float32)
            mask = torch.zeros((len(retained),), dtype=torch.bool)
            for row, (deadline, _order, marker, release, waiting) in enumerate(retained):
                self._encode_state(states[row], marker, boundary, deadline, waiting)

        metadata = {
            "profile": PROFILE,
            "state_dim": STATE_DIM,
            "max_states": MAX_STATES,
            "feature_layout": {
                "types": {name: index for name, index in _TYPE_OFFSETS.items()},
                "pads_offset": _PAD_OFFSET,
                "pads": list(PADS),
                "terminal_lanes_offset": _LANE_OFFSET,
                "waiting_offset": _WAITING_OFFSET,
                "remaining_seconds_offset": _REMAINING_OFFSET,
                "until_trace_offset": _UNTIL_TRACE_OFFSET,
                "progress_offset": _PROGRESS_OFFSET,
                "ex_offset": _EX_OFFSET,
                "seconds_normalization": "log1p(clamp(seconds,0,60))/log1p(60)",
            },
            "start_beat": str(beat),
            "boundary_second": boundary,
            "state_count": len(retained),
            "selected_state_count": total,
            "truncated_count": max(0, total - len(retained)),
            "states": [self._state_metadata(item[2], boundary, item[0], item[4])
                       for item in retained],
            "limitations": [
                "Sustained geometry follows the pinned MiaCode reference paths.",
                "This is not evidence of real-player body mechanics or playable certification.",
            ],
        }
        return states, mask, metadata

    def _release_for(self, marker: dict[str, Any], boundary: float) -> float:
        # Postparse before-slide/tail flags may depend on *future* target heads.
        # Only already-authored heads can affect this causal boundary snapshot.
        lane = marker.get("lane") if marker["type"] in {"hold", "touch_hold"} else marker.get("end_lane")
        end = marker["end_second"]
        for following in self.markers:
            if (following["key"] != marker["key"] and following["type"] in {"slide", "wifi"}
                    and following["second"] < boundary - _EPS and following["lane"] == lane
                    and abs(following["slide_trace_second"] - end) < 1/180):
                return 0.0
        return RELEASE

    def _encode_state(self, row: torch.Tensor, marker: dict[str, Any], boundary: float,
                      deadline: float, waiting: bool) -> None:
        kind = marker["type"]
        row[_TYPE_OFFSETS[kind]] = 1.0
        remaining = max(0.0, deadline - boundary)
        row[_REMAINING_OFFSET] = _unit_log_seconds(remaining)

        start = float(marker.get("second", 0.0))
        end = float(marker.get("end_second", start))
        trace = float(marker.get("slide_trace_second", -1.0))
        if kind in {"slide", "wifi"}:
            until_trace = max(0.0, trace - boundary)
            row[_UNTIL_TRACE_OFFSET] = _unit_log_seconds(until_trace)
            duration = max(0.0, end - trace)
            row[_PROGRESS_OFFSET] = 0.0 if waiting or duration <= _EPS else max(
                0.0, min(1.0, (boundary - trace) / duration)
            )
            if waiting:
                pads = {f"A{int(marker['lane'])}"}
            elif boundary >= end:
                pads = self._terminal_pads(marker)
            else:
                pads = self._slide_pads(marker, boundary)
        else:
            duration = max(0.0, end - start)
            row[_PROGRESS_OFFSET] = 0.0 if duration <= _EPS else max(
                0.0, min(1.0, (boundary - start) / duration)
            )
            pad = (f"A{int(marker['lane'])}" if kind == "hold"
                   else str(marker.get("touch_pad", "")))
            pads = {pad} if pad in PADS else set()

        for pad in pads:
            index = PADS.index(pad)
            row[_PAD_OFFSET + index] = 1.0
        for lane in self._terminal_lanes(marker):
            if 1 <= lane <= 8:
                row[_LANE_OFFSET + lane - 1] = 1.0
        row[_WAITING_OFFSET] = float(waiting)
        row[_EX_OFFSET] = float(bool(marker.get("is_ex") or marker.get("head_ex")))

    def _terminal_lanes(self, marker: dict[str, Any]) -> tuple[int, ...]:
        try:
            end_lane = int(marker.get("end_lane", 0))
        except (TypeError, ValueError):
            return ()
        if not 1 <= end_lane <= 8:
            return ()
        if marker["type"] == "wifi":
            return _wifi_end_lanes(end_lane)
        return (end_lane,)

    def _terminal_pads(self, marker: dict[str, Any]) -> set[str]:
        return {f"A{lane}" for lane in self._terminal_lanes(marker)}

    def _slide_pads(self, marker: dict[str, Any], boundary: float) -> set[str]:
        if self._references is None:
            self._references = reference_data()
        data = self._references
        kind = marker["type"]
        trace = float(marker["slide_trace_second"])
        if kind == "wifi":
            entry = data.get("wifi", {}).get(marker.get("slide_track_key"), {})
            paths = _wifi_paths(entry)
            duration = max(0.0, float(marker["end_second"]) - trace)
            progress = 1.0 if duration <= _EPS else (boundary - trace) / duration
            return set().union(*(_path_pads(_sample_path(path, progress), 50.0) for path in paths))

        keys = marker.get("slide_segment_keys", [])
        shoots = marker.get("slide_segment_shoot_seconds", [])
        durations = marker.get("slide_segment_durations", [])
        if not keys or not (len(keys) == len(shoots) == len(durations)):
            raise ValueError(f"MiaCode Slide marker has incomplete segment timing: {marker.get('raw', '')}")
        for key, shoot, duration_value in zip(keys, shoots, durations):
            shoot = float(shoot)
            duration = max(0.0, float(duration_value))
            end = shoot + duration
            if shoot - _EPS <= boundary <= end + _EPS:
                entry = data.get("slides", {}).get(key, {})
                path = entry.get("real_path_samples") or entry.get("samples") or []
                progress = 1.0 if duration <= _EPS else (boundary - shoot) / duration
                return _path_pads(_sample_path(path, progress))
        # At exact segment boundaries the next segment owns the current sample.
        for key, shoot, duration_value in zip(keys, shoots, durations):
            shoot = float(shoot)
            duration = max(0.0, float(duration_value))
            if abs(boundary - shoot) <= _EPS:
                entry = data.get("slides", {}).get(key, {})
                path = entry.get("real_path_samples") or entry.get("samples") or []
                return _path_pads(_sample_path(path, 0.0))
        if boundary < trace:
            return {f"A{int(marker['lane'])}"}
        raise ValueError(f"Boundary fell outside MiaCode Slide segment timing: {marker.get('raw', '')}")

    @staticmethod
    def _state_metadata(marker: dict[str, Any], boundary: float, deadline: float,
                        waiting: bool) -> dict[str, Any]:
        return {
            "marker_key": marker.get("key"),
            "type": marker["type"],
            "line": marker.get("line"),
            "col": marker.get("col"),
            "source_second": float(marker.get("second", 0.0)),
            "end_second": float(marker.get("end_second", 0.0)),
            "deadline_second": float(deadline),
            "remaining_seconds": max(0.0, float(deadline) - boundary),
            "waiting_for_trace": bool(waiting),
            "slide_track_key": marker.get("slide_track_key", ""),
        }
