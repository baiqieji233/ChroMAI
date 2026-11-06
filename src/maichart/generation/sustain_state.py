"""Boundary snapshots of active MiaCode-style sustained chart actions.

This is an input feature over authored notes and the pinned reference paths. It
does not simulate a player's body or certify that a chart is playable.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from fractions import Fraction
import math
import threading
from typing import Any, Iterable

import numpy as np
import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat, tokenize_event_timing
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
_PAD_INDEX = {pad: idx for idx, pad in enumerate(PADS)}
_PAD_GEOMETRY = tuple((pad, float(CENTERS[pad][0]), float(CENTERS[pad][1]), float(RADII[pad[0]]) + _EPS) for pad in PADS)
_PAD_X = np.asarray([item[1] for item in _PAD_GEOMETRY], dtype=np.float64)
_PAD_Y = np.asarray([item[2] for item in _PAD_GEOMETRY], dtype=np.float64)
_PAD_RADII = np.asarray([item[3] for item in _PAD_GEOMETRY], dtype=np.float64)
_SLIDE_PAD_CACHE_LIMIT = 1024
_LOG1P_MAX_SECONDS = math.log1p(MAX_SECONDS_SCALE)


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
    return math.log1p(value) / _LOG1P_MAX_SECONDS


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
    px, py = position
    return {
        pad for pad, cx, cy, base_r in _PAD_GEOMETRY
        if math.hypot(px - cx, py - cy) <= radius + base_r
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
            build_ir=False,
        )
        parsed_chart = parse_maidata_metadata(text)
        inote = parsed_chart.fields.get("inote_5")
        if inote is None:
            raise ValueError("Compiled event chart has no inote_5 field")
        self.markers, errors, _warnings = parse_markers(inote, first=self.first)
        if errors:
            detail = "; ".join(str(item.get("message", item)) for item in errors[:3])
            raise ValueError(f"MiaCode sustain input contains unsupported notes: {detail}")
        # Bind parser locations to exact source beats, avoiding float onset
        # comparisons for visibility. Compiler commas delimit timing groups.
        timings = tokenize_event_timing(inote, first=self.first)
        separators = [i for i, char in enumerate(inote) if char == ","]
        line_offsets = [0]
        line_offsets.extend(i + 1 for i, char in enumerate(inote) if char == "\n")
        for marker in self.markers:
            offset = line_offsets[int(marker["line"]) - 1] + int(marker["col"]) - 1
            marker["source_beat"] = timings[bisect_left(separators, offset)].beat
        self.markers = [m for m in self.markers if m["type"] in _TYPE_OFFSETS]
        self._references: dict[str, Any] | None = None
        self._release_index = None

    @classmethod
    def from_markers(cls, points, tempo_changes, first, markers):
        """Restore the parsed markers while rebuilding transient indexes on demand."""
        index = cls.__new__(cls)
        index.points = tuple(points)
        index.tempo_changes = tuple(tempo_changes)
        index.first = float(first)
        index.markers = markers
        index._references = None
        index._release_index = None
        return index

    def at(self, start_beat: Fraction | int | str | float, *, include_current=False, visible_before_beat=None, visibility_version="legacy_v1", max_states=MAX_STATES, include_metadata: bool = True) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if visibility_version not in {"legacy_v1", "causal_v2"}:
            raise ValueError("Unknown sustain visibility version")
        if max_states is not None and max_states < 1:
            raise ValueError("max_states must be positive or None")
        cutoff = None if visible_before_beat is None else _as_beat(visible_before_beat)
        beat = _as_beat(start_beat)
        boundary = float(seconds_at_beat(beat, self.tempo_changes, self.first))
        selected: list[tuple[float, int, dict[str, Any], float, bool]] = []
        for marker in self._candidate_markers(boundary):
            kind = marker.get("type")
            if kind not in _TYPE_OFFSETS or marker.get("is_mine") or marker.get("track_mine"):
                continue
            if cutoff is not None and not self._visible_before(marker, cutoff):
                continue
            onset = float(marker.get("second", 0.0))
            # v1 preserves old checkpoint inputs; v2 tolerates parser clock
            # roundoff at include_current boundaries.
            upper = boundary if visibility_version == "legacy_v1" else boundary + _EPS
            if (onset > upper if include_current else onset >= boundary - _EPS):
                continue
            end = float(marker.get("end_second", -1.0))
            if not math.isfinite(end):
                raise ValueError(f"MiaCode marker has no end time: {marker.get('raw', '')}")
            # A causal following head can only shorten the release to zero.
            # Once even its maximum release expired, scanning possible heads
            # cannot change the selected state. Most authored actions are in
            # this category at later training boundaries.
            if end + RELEASE <= boundary + _EPS:
                continue
            release = self._release_for(marker, boundary, visible_before_beat=cutoff)
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
        retained = selected if max_states is None else selected[:max_states]
        if not retained:
            states = torch.zeros((1, STATE_DIM), dtype=torch.float32)
            mask = torch.ones((1,), dtype=torch.bool)
        else:
            states = torch.zeros((len(retained), STATE_DIM), dtype=torch.float32)
            mask = torch.zeros((len(retained),), dtype=torch.bool)
            for row, (deadline, _order, marker, release, waiting) in enumerate(retained):
                self._encode_state(states[row], marker, boundary, deadline, waiting)

        if not include_metadata:
            return states, mask, {}

        metadata = {
            "visibility_version": visibility_version,
            "visible_before_beat": None if cutoff is None else str(cutoff),
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

    def _candidate_markers(self,boundary):
        # Coarse buckets only prune impossible intervals. Exact onset,
        # deadline, causal release and ordering checks remain in at().
        cached=getattr(self,'_active_marker_index',None)
        if cached is None or cached[0] is not self.markers or cached[1]!=len(self.markers):
            # Keep the actual list identity: integer ids can be recycled after
            # replacement. Drop the previous cache even on the fallback path.
            self._active_marker_index=None
            buckets={}
            for marker in self.markers:
                start=float(marker.get('second',0.))
                end=float(marker.get('end_second',-1.))+RELEASE
                if not math.isfinite(start) or not math.isfinite(end) or end-start>4096:
                    # Preserve the original validation path for malformed or
                    # extremely long intervals rather than allocating bins.
                    return self.markers
                # causal_v2 accepts onsets up to EPS after the query clock.
                # Include the preceding bin when roundoff straddles an integer
                # second; at() still applies the exact visibility policy.
                for second in range(math.floor(start-_EPS),math.floor(end)+1):
                    buckets.setdefault(second,[]).append(marker)
            cached=self._active_marker_index=(self.markers,len(self.markers),buckets)
        return cached[2].get(math.floor(boundary),())

    @staticmethod
    def _visible_before(marker, cutoff):
        if "source_beat" not in marker:
            raise ValueError("Prefix queries require a rebuilt index with exact source beats")
        return marker["source_beat"] < cutoff

    def _release_for(self, marker: dict[str, Any], boundary: float, *, visible_before_beat=None) -> float:
        # Postparse before-slide/tail flags may depend on *future* target heads.
        # Only already-authored heads can affect this causal boundary snapshot.
        lane = marker.get("lane") if marker["type"] in {"hold", "touch_hold"} else marker.get("end_lane")
        end = marker["end_second"]
        # Preparation caches restore markers with __new__, so build lazily
        # rather than requiring a new serialized cache/checkpoint contract.
        # The marker list is immutable during normal use. Detect replacement
        # or append as well, for callers rebuilding the parsed chart.
        cached = getattr(self, '_release_index', None)
        if cached is None or cached[0] is not self.markers or cached[1] != len(self.markers):
            by_lane: dict[Any, list[dict[str, Any]]] = {}
            for following in self.markers:
                if (following["type"] in {"slide", "wifi"}
                        and math.isfinite(following["slide_trace_second"])):
                    by_lane.setdefault(following["lane"], []).append(following)
            index = {}
            for source_lane, values in by_lane.items():
                values.sort(key=lambda item: item['slide_trace_second'])
                index[source_lane] = ([item['slide_trace_second'] for item in values], values)
            cached = self._release_index = (self.markers, len(self.markers), index)
        traces, following_markers = cached[2].get(lane, ((), ()))
        radius = 1/180
        lo, hi = bisect_left(traces, end-radius), bisect_right(traces, end+radius)
        for following in following_markers[lo:hi]:
            if visible_before_beat is not None and not self._visible_before(following, visible_before_beat):
                continue
            if (following["key"] != marker["key"] and following["type"] in {"slide", "wifi"}
                    and following["second"] < boundary - _EPS and following["lane"] == lane
                    and abs(following["slide_trace_second"] - end) < radius):
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
            index = _PAD_INDEX.get(pad)
            if index is not None:
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

    def _slide_pad_matrix(self, marker: dict[str, Any], boundaries) -> np.ndarray:
        """Return native pad occupancy for many clocks of one visible marker.

        Keep segment selection and interpolation identical to the scalar path;
        only the independent pad-distance calculations are batched. A bounded
        per-index cache retains primitive occupancy for exact clocks.
        """
        if self._references is None:
            self._references = reference_data()
        clocks = np.asarray(boundaries, dtype=np.float64)
        # Cache only primitive per-marker geometry, never aggregate symbolic
        # features. Visibility is checked by the caller before this method.
        # A list replacement or append (including incremental rollback) resets
        # the cache; marker identity and geometry fields guard reused entries.
        cache_state = self.__dict__.get('_slide_pad_cache')
        if cache_state is None:
            cache_state = self.__dict__.setdefault('_slide_pad_cache', (threading.RLock(), [None, -1, OrderedDict()]))
        lock, state = cache_state
        signature = (marker['type'], marker.get('slide_trace_second'), marker.get('end_second'),
                     marker.get('lane'), marker.get('slide_track_key'),
                     tuple(marker.get('slide_segment_keys', ())),
                     tuple(marker.get('slide_segment_shoot_seconds', ())),
                     tuple(marker.get('slide_segment_durations', ())), id(self._references))
        result = np.zeros((len(clocks), len(PADS)), dtype=bool)
        missing = []
        with lock:
            if state[0] is not self.markers or state[1] != len(self.markers):
                state[:] = [self.markers, len(self.markers), OrderedDict()]
            cache = state[2]
            for row, clock in enumerate(clocks):
                key = (id(marker), signature, float(clock))
                entry = cache.get(key)
                if entry is not None and entry[0] is marker:
                    result[row] = entry[1]
                    cache.move_to_end(key)
                else:
                    missing.append((row, key))
        if missing:
            calculated = self._slide_pad_matrix_uncached(marker, clocks[[row for row, _ in missing]])
            with lock:
                # An append/replacement during calculation invalidates the
                # entire old generation and must not repopulate its cache.
                current = state[0] is self.markers and state[1] == len(self.markers)
                for (row, key), occupancy in zip(missing, calculated):
                    result[row] = occupancy
                    if current:
                        cache[key] = (marker, occupancy.copy())
                        cache.move_to_end(key)
                while len(cache) > _SLIDE_PAD_CACHE_LIMIT:
                    cache.popitem(last=False)
        return result

    def _slide_pad_matrix_uncached(self, marker: dict[str, Any], boundaries) -> np.ndarray:
        data = self._references
        clocks = np.asarray(boundaries, dtype=np.float64)
        result = np.zeros((len(clocks), len(PADS)), dtype=bool)
        positions = []
        rows = []
        trace = float(marker['slide_trace_second'])
        if marker['type'] == 'wifi':
            paths = _wifi_paths(data.get('wifi', {}).get(marker.get('slide_track_key'), {}))
            duration = max(0.0, float(marker['end_second']) - trace)
            for row, boundary in enumerate(clocks):
                progress = 1.0 if duration <= _EPS else (float(boundary) - trace) / duration
                for path in paths:
                    positions.append(_sample_path(path, progress))
                    rows.append(row)
            radius = 50.0
        else:
            keys = marker.get('slide_segment_keys', [])
            shoots = marker.get('slide_segment_shoot_seconds', [])
            durations = marker.get('slide_segment_durations', [])
            if not keys or not (len(keys) == len(shoots) == len(durations)):
                raise ValueError(f"MiaCode Slide marker has incomplete segment timing: {marker.get('raw', '')}")
            segments = [(key, float(shoot), max(0.0, float(duration)))
                        for key, shoot, duration in zip(keys, shoots, durations)]
            for row, clock in enumerate(clocks):
                boundary = float(clock)
                segment = next(((key, shoot, duration) for key, shoot, duration in segments
                                if shoot - _EPS <= boundary <= shoot + duration + _EPS), None)
                if segment is None:
                    segment = next(((key, shoot, duration) for key, shoot, duration in segments
                                    if abs(boundary - shoot) <= _EPS), None)
                if segment is None:
                    if boundary < trace:
                        pad_index = _PAD_INDEX.get(f"A{int(marker['lane'])}")
                        if pad_index is not None:
                            result[row, pad_index] = True
                        continue
                    raise ValueError(f"Boundary fell outside MiaCode Slide segment timing: {marker.get('raw', '')}")
                key, shoot, duration = segment
                entry = data.get('slides', {}).get(key, {})
                path = entry.get('real_path_samples') or entry.get('samples') or []
                progress = 1.0 if duration <= _EPS else (boundary - shoot) / duration
                positions.append(_sample_path(path, progress))
                rows.append(row)
            radius = HAND_RADIUS
        if positions:
            xy = np.asarray(positions, dtype=np.float64)
            rows = np.asarray(rows, dtype=np.intp)
            thresholds = radius + _PAD_RADII
            distances = np.hypot(xy[:, 0, None] - _PAD_X, xy[:, 1, None] - _PAD_Y)
            inside = distances <= thresholds
            # numpy and math hypot can round differently exactly at a pad
            # edge; use the original scalar predicate in that narrow band.
            for index, column in np.argwhere(np.abs(distances - thresholds) <= 1e-12):
                inside[index, column] = math.hypot(float(xy[index, 0]) - _PAD_X[column],
                                                   float(xy[index, 1]) - _PAD_Y[column]) <= thresholds[column]
            np.logical_or.at(result, rows, inside)
        return result

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
