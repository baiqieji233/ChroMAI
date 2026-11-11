"""Filter Composer v2 token choices using known time and occupancy constraints.

Filter time bounds, cooldowns, occupied pads and configured Slide collisions.
Grid durations use the time available in the audio. Explicit Slide waits and
connected Touch groups are evaluated on the complete event by ``EventGuard``.
``balanced_v1`` also limits grid Hold durations that would occupy a pad at a
waiting Slide's arrival or keep both hands occupied across a salient onset.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from fractions import Fraction
import math
from typing import Any

import numpy as np

from maichart.event_timing import seconds_at_beat
from maichart.generation.muri.data import reference_data
from maichart.generation.muri.model import (
    CENTERS,
    EPS,
    RELEASE,
    STATIC_EPS,
    TPS,
    end_tick,
    has_head,
    pad as marker_pad,
    start_tick,
    usable_slide,
)
from maichart.generation.muri.touch_reachability import (
    DEFAULT_COHOLD_RADIUS,
    MIN_SUSTAIN_LOCK_SEC,
    TAIL_RELEASE_GRACE_SEC,
    effective_cohold_radius,
    validate_touch_reachability,
)

FORWARD_POLICIES = ("off", "hard_only_v1", "balanced_v1")
MAX_TWO_HAND_LOCK_BEATS = Fraction(4, 1)


def validate_forward_feasibility(
    policy: str = "balanced_v1",
    forward_tap_on_slide_ms: float | None = None,
) -> tuple[str, float | None]:
    normalized = str(policy or "off").strip()
    if normalized not in FORWARD_POLICIES:
        raise ValueError(
            f"forward_feasibility must be one of {', '.join(FORWARD_POLICIES)}"
        )
    if forward_tap_on_slide_ms is None or forward_tap_on_slide_ms == "":
        return normalized, None
    try:
        ms = float(forward_tap_on_slide_ms)
    except (TypeError, ValueError) as exc:
        raise ValueError("forward_tap_on_slide_ms must be a finite number") from exc
    if not math.isfinite(ms) or not 0.0 <= ms <= 250.0:
        raise ValueError("forward_tap_on_slide_ms must be between 0 and 250 ms")
    return normalized, ms


def _salient_audio_onsets(song: Any) -> np.ndarray:
    """Extract salient spectral-flux onset peaks (seconds) once per song."""
    if song is None or getattr(song, "audio", None) is None:
        return np.empty(0, dtype=np.float64)
    cached = getattr(song, "_forward_salient_onsets", None)
    if cached is not None:
        return cached
    audio = song.audio
    mel = getattr(audio, "mel", None)
    times = getattr(audio, "mel_times_sec", None)
    if mel is None or times is None or len(times) < 4 or mel.shape[1] <= 96:
        out = np.empty(0, dtype=np.float64)
        song._forward_salient_onsets = out
        return out
    flux = np.maximum(np.asarray(mel[:, 96], dtype=np.float64), 0.0)
    positive = flux[flux > 1e-6]
    if len(positive) < 4:
        out = np.empty(0, dtype=np.float64)
        song._forward_salient_onsets = out
        return out
    p85 = float(np.percentile(positive, 85))
    median = float(np.median(positive))
    floor = max(1.35 * median, 0.55 * p85)
    peaks: list[float] = []
    last_peak = 0.0
    for i in range(1, len(flux) - 1):
        val = flux[i]
        if val < floor or val < flux[i - 1] or val < flux[i + 1]:
            continue
        sec = float(times[i])
        if peaks and sec - peaks[-1] < 0.18:
            if val > last_peak:
                peaks[-1] = sec
                last_peak = val
            continue
        peaks.append(sec)
        last_peak = val
    out = np.asarray(peaks, dtype=np.float64)
    song._forward_salient_onsets = out
    return out


class ForwardFeasibilityGuard:
    """Fast incremental physical & lookahead constraint engine for EventSampler."""

    def __init__(
        self,
        guard: Any,
        codec: Any,
        *,
        song: Any = None,
        policy: str = "balanced_v1",
        forward_tap_on_slide_ms: float | None = None,
    ):
        self.policy, self.forward_tap_on_slide_ms = validate_forward_feasibility(
            policy, forward_tap_on_slide_ms
        )
        self.codec = codec
        self.guard = guard
        self.song = song
        self.enabled = bool(
            self.policy != "off"
            and guard is not None
            and hasattr(guard, "tempo")
            and hasattr(guard, "prefix")
            and hasattr(guard, "_markers")
        )
        if not self.enabled:
            return

        self.tempo = guard.tempo
        self._slide_has_optional_head = any(
            pattern[0] == "N:slide" and any(part in {"S:?", "S:!", "S:m"} for part in pattern[:4])
            for pattern in codec.templates if pattern
        )
        self.first = float(guard.first)
        self.audio_end = float(guard.audio_end)
        self.min_same_head_sec = float(getattr(guard, "min_same_head_sec", 0.0))
        self.last_head_seconds = dict(getattr(guard, "last_head_seconds", {}))
        self.touch_layout_policy = getattr(guard, "touch_layout_policy", "open")
        self.intercept_tap_on_slide = bool(getattr(guard, "intercept_tap_on_slide", False))
        self.intercept_slide_head_tap = bool(getattr(guard, "intercept_slide_head_tap", False))
        cfg = getattr(guard, "config", None)
        self.touch_policy, raw_radius = validate_touch_reachability(
            getattr(cfg, "touch_reachability_policy", "off") if cfg is not None else "off",
            getattr(cfg, "touch_cohold_radius", DEFAULT_COHOLD_RADIUS) if cfg is not None else DEFAULT_COHOLD_RADIUS,
        )
        self.touch_radius = effective_cohold_radius(self.touch_policy, raw_radius)
        native_tos_ms = float(getattr(cfg, "tap_on_slide_threshold_ms", 200.0)) if cfg is not None else 200.0
        if not self.intercept_tap_on_slide:
            self.tos_hard_sec = 0.0
            self.tos_threshold_sec = 0.0
        else:
            eff_ms = 150.0 if self.forward_tap_on_slide_ms is None else float(self.forward_tap_on_slide_ms)
            self.tos_hard_sec = eff_ms / 1000.0
            self.tos_threshold_sec = max(native_tos_ms / 1000.0, self.tos_hard_sec)

        self._salient_onsets = _salient_audio_onsets(song) if self.policy == "balanced_v1" else np.empty(0)
        self._beat_open_cache: dict[Fraction, bool] = {}
        self._state_cache: dict[tuple[Fraction, tuple[str, ...]], dict[str, Any]] = {}
        self._build_prefix_index()

    def _local_bpm(self, beat: Fraction) -> float:
        for change in reversed(self.tempo):
            if change.beat <= beat:
                return float(change.bpm)
        return float(self.tempo[0].bpm)

    def _second_at(self, beat: Fraction) -> float:
        if self.song is not None and getattr(self.song, "clock", None) is not None:
            return float(self.song.clock.at(beat))
        return float(seconds_at_beat(beat, self.tempo, self.first))

    def _build_prefix_index(self) -> None:
        self.held_pads: dict[str, list[tuple[int, int, float, float, int]]] = defaultdict(list)
        self.tos_windows: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.sht_windows: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self.slide_launches: dict[int, list[float]] = defaultdict(list)
        self.non_touch_holds: list[tuple[float, float, tuple[float, float], int]] = []
        self.sustain_locks: list[tuple[str, float, float, bool]] = []
        self.slide_actions: list[tuple[float, float, float, tuple[tuple[float, float], ...], str]] = []
        self.future_pad_arrivals: dict[str, list[tuple[float, float]]] = defaultdict(list)

        shifted_markers: list[dict[str, Any]] = []
        refs = reference_data()
        slides_ref = refs.get("slides", {})
        wifi_ref = refs.get("wifi", {})

        earliest = self.guard.prefix[-1][1] if self.guard.prefix else 0.0
        for point, onset_sec, finish, sustained in self.guard.prefix:
            if not sustained or finish < earliest - 0.3:
                continue
            bpm = self._local_bpm(point.beat)
            markers, errors = self.guard._markers(point.note_text, bpm)
            if errors:
                continue
            for m in markers:
                if m.get("is_mine") or m.get("track_mine"):
                    continue
                sm = dict(m)
                sm["second"] = onset_sec + float(m.get("second", 0.0))
                if m.get("end_second", -1.0) >= 0.0:
                    sm["end_second"] = onset_sec + float(m["end_second"])
                if m.get("slide_trace_second", -1.0) >= 0.0:
                    sm["slide_trace_second"] = onset_sec + float(m["slide_trace_second"])
                if m.get("slide_segment_shoot_seconds"):
                    sm["slide_segment_shoot_seconds"] = [
                        onset_sec + float(s) for s in m["slide_segment_shoot_seconds"]
                    ]
                shifted_markers.append(sm)

        # Index the physical pad entries of committed Slides.
        for sm in shifted_markers:
            if not usable_slide(sm):
                continue
            lane = int(sm.get("lane") or 0)
            trace = float(sm["slide_trace_second"])
            end = float(sm["end_second"])
            self.slide_launches[lane].append(trace)
            self.future_pad_arrivals[f"A{lane}"].append((trace, trace))
            if sm["type"] == "slide":
                for shoot, dur, pad_set in zip(
                    sm.get("slide_segment_shoot_seconds", ()),
                    sm.get("slide_segment_durations", ()),
                    sm.get("slide_segment_pad_enter_times", ()),
                ):
                    for item in pad_set:
                        p = str(item.get("pad", "")).upper()
                        enter = float(shoot) + float(item.get("t", 0.0)) * float(dur)
                        self.future_pad_arrivals[p].append((trace, enter))
            elif sm["type"] == "wifi":
                dur = end - trace
                for item in sm.get("wifi_pad_enter_times", ()):
                    p = str(item.get("pad", "")).upper()
                    enter = trace + float(item.get("t", 0.0)) * dur
                    self.future_pad_arrivals[p].append((trace, enter))

        for sm in shifted_markers:
            kind = sm["type"]
            sec = float(sm["second"])
            if kind in {"hold", "touch_hold"}:
                p = marker_pad(sm)
                end_sec = max(sec, float(sm.get("end_second", sec)))
                if end_sec <= sec + 1e-9 or not p:
                    continue
                lane = int(sm.get("lane") or 0) if kind == "hold" else 0
                tail_on_head = kind == "hold" and any(
                    abs(end_sec - tr) <= 1.0 / TPS + EPS for tr in self.slide_launches.get(lane, ())
                )
                press_end = end_sec + (0.0 if tail_on_head else RELEASE)
                self.held_pads[p].append((start_tick(sec), end_tick(press_end), sec, end_sec, lane))
                if p in CENTERS:
                    if kind == "hold":
                        self.non_touch_holds.append((sec, press_end, CENTERS[p], lane))
                    if end_sec - sec > MIN_SUSTAIN_LOCK_SEC:
                        self.sustain_locks.append((p, sec, end_sec - TAIL_RELEASE_GRACE_SEC, kind == "hold"))
            elif usable_slide(sm):
                trace = float(sm["slide_trace_second"])
                end_sec = float(sm["end_second"])
                lane = int(sm.get("lane") or 0)
                if sm["type"] == "slide":
                    segs = list(
                        zip(
                            sm.get("slide_segment_keys", ()),
                            sm.get("slide_segment_shoot_seconds", ()),
                            sm.get("slide_segment_durations", ()),
                        )
                    )
                    for i, (key, begin, dur) in enumerate(segs):
                        entry = slides_ref.get(key)
                        if not entry:
                            continue
                        pts = tuple(
                            (float(pt["x"]), float(pt["y"]))
                            for pt in entry.get("real_path_samples", entry.get("samples", ()))
                        )
                        seg_end = float(begin) + float(dur) + (RELEASE if i == len(segs) - 1 else 0.0)
                        if pts:
                            self.slide_actions.append((float(begin), seg_end, float(dur), pts, "normal"))
                elif sm["type"] == "wifi":
                    entry = wifi_ref.get(sm.get("slide_track_key"))
                    if entry:
                        paths = entry.get("di_real_path_samples", entry.get("lane_samples", ()))
                        if paths:
                            for path in (paths[0], paths[-1]):
                                pts = tuple((float(pt["x"]), float(pt["y"])) for pt in path)
                                if pts:
                                    self.slide_actions.append((trace, end_sec, end_sec - trace, pts, "wifi"))

                if self.tos_hard_sec > 0.0:
                    self._index_slide_collisions(sm, trace, end_sec, lane)
                if self.intercept_slide_head_tap:
                    self.sht_windows[f"A{lane}"].append((trace + 1.0 / TPS, trace + 0.150))

    def _index_slide_collisions(self, sm: dict[str, Any], trace: float, end_sec: float, lane: int) -> None:
        extra_delta = self.tos_threshold_sec - (27.0 / TPS)
        hard_after = self.tos_hard_sec
        hard_before = min(max(0.0, extra_delta), hard_after)
        if sm["type"] == "wifi":
            prop = float(sm.get("wifi_critical_proportion", 0.0))
            critical = trace + (end_sec - trace) * prop
            end_lane = int(sm.get("end_lane") or 0)
            end_lanes = {end_lane, ((end_lane + 6) % 8) + 1, (end_lane % 8) + 1}
            lo = max(critical - hard_before, trace + 1.0 / TPS)
            hi = critical + hard_after
            for el in end_lanes:
                if 1 <= el <= 8:
                    self.tos_windows[f"A{el}"].append((lo, hi))
        else:
            for shoot, dur, pad_set in zip(
                sm.get("slide_segment_shoot_seconds", ()),
                sm.get("slide_segment_durations", ()),
                sm.get("slide_segment_pad_enter_times", ()),
            ):
                for item in pad_set:
                    p = str(item.get("pad", "")).upper()
                    if not p.startswith("A"):
                        continue
                    enter = float(shoot) + float(item.get("t", 0.0)) * float(dur)
                    lo = max(enter - hard_before, trace + 1.0 / TPS)
                    hi = enter + hard_after
                    self.tos_windows[p].append((lo, hi))

    def _sample_slide_clusters(self, second: float) -> list[tuple[float, float]]:
        """Active merged Slide/Wifi action coordinates at ``second`` (MiaCode MultiTouch)."""
        active_pts: list[tuple[tuple[float, float], tuple[float, float] | None, str]] = []
        for begin, end, dur, pts, merge_kind in self.slide_actions:
            if second + EPS < begin or second >= end - EPS:
                continue
            prog = min(1.0, max(0.0, (second - begin) / dur)) if dur > EPS else 1.0
            val = prog * (len(pts) - 1)
            idx = min(int(val), len(pts) - 1)
            nxt = min(idx + 1, len(pts) - 1)
            t = val - idx
            dx, dy = pts[nxt][0] - pts[idx][0], pts[nxt][1] - pts[idx][1]
            length = math.hypot(dx, dy)
            tan = (dx / length, dy / length) if length > EPS else None
            pos = (pts[idx][0] + dx * t, pts[idx][1] + dy * t)
            merged = False
            for ep, et, ek in active_pts:
                if ek == merge_kind and tan is not None and et is not None:
                    if math.dist(ep, pos) < 10.0 and math.dist(et, tan) < 0.05235389661574631:
                        merged = True
                        break
            if not merged:
                active_pts.append((pos, tan, merge_kind))
        return [item[0] for item in active_pts]

    def _pad_in_cooldown(self, pad_name: str, second: float) -> bool:
        if self.min_same_head_sec <= 0.0:
            return False
        key = (
            f"A{pad_name}"
            if self.touch_layout_policy == "legacy_two_heads" and pad_name.isdigit()
            else pad_name
        )
        prior = self.last_head_seconds.get(key)
        return prior is not None and (second - prior < self.min_same_head_sec - 1e-9)

    def _pad_held(self, elec_pad: str, second: float, *, is_slide_launch: bool = False) -> bool:
        if is_slide_launch:
            return False
        t_tick = start_tick(second)
        for b_tick, e_tick, start_s, end_s, _ in self.held_pads.get(elec_pad, ()):
            if b_tick <= t_tick <= e_tick or (start_s - 1e-9 <= second < end_s - 1e-9):
                return True
        return False

    def _pad_in_slide_collision(self, elec_pad: str, second: float) -> bool:
        for lo, hi in self.tos_windows.get(elec_pad, ()):
            if lo - STATIC_EPS <= second <= hi + STATIC_EPS:
                return True
        for lo, hi in self.sht_windows.get(elec_pad, ()):
            if lo - STATIC_EPS <= second <= hi + STATIC_EPS:
                return True
        return False

    def _is_slide_launch(self, lane: int, second: float) -> bool:
        return any(abs(second - tr) <= 1.0 / TPS + EPS for tr in self.slide_launches.get(lane, ()))

    def _beat_state(self, beat: Fraction, completed_atoms: tuple[str, ...] = ()) -> dict[str, Any]:
        cache_key = (beat, completed_atoms)
        cached = self._state_cache.get(cache_key)
        if cached is not None:
            return cached

        sec = self._second_at(beat)
        bpm = self._local_bpm(beat)
        beat_sec = 60.0 / bpm

        non_touch_points: list[tuple[float, float]] = []
        for start_s, press_end, xy, _ in self.non_touch_holds:
            if start_s - EPS <= sec < press_end - EPS and xy not in non_touch_points:
                non_touch_points.append(xy)
        non_touch_points.extend(self._sample_slide_clusters(sec))
        sustain_locks = [
            (pad, end, is_button) for pad, start, end, is_button in self.sustain_locks
            if start - EPS <= sec < end
        ]
        used_heads: set[str] = set()
        if completed_atoms:
            markers, errors = self.guard._markers("/".join(completed_atoms), bpm)
            if not errors:
                for m in markers:
                    kind = m["type"]
                    if m.get("is_mine") or m.get("track_mine"):
                        continue
                    if kind in {"tap", "hold", "slide", "wifi"}:
                        lane = int(m["lane"])
                        used_heads.add(str(lane))
                        has_action = kind == "hold" or (
                            not self._is_slide_launch(lane, sec)
                            and (kind == "tap" or has_head(m))
                        )
                        xy = CENTERS[f"A{lane}"]
                        if has_action and xy not in non_touch_points:
                            non_touch_points.append(xy)
                    elif kind in {"touch", "touch_hold"}:
                        used_heads.add(marker_pad(m))
                    if kind in {"hold", "touch_hold"}:
                        duration = float(m.get("end_second", 0.0))
                        if duration > MIN_SUSTAIN_LOCK_SEC:
                            sustain_locks.append((marker_pad(m), sec + duration - TAIL_RELEASE_GRACE_SEC, kind == "hold"))

        state = {
            "sec": sec,
            "beat_sec": beat_sec,
            "non_touch_points": non_touch_points,
            "used_heads": used_heads,
            "sustain_locks": sustain_locks,
        }
        self._state_cache[cache_key] = state
        return state

    def _max_sustain_end_sec(self, state: dict[str, Any], elec_pad: str, *, is_button: bool) -> float:
        sec = state["sec"]
        limit = self.audio_end
        if self.policy != "balanced_v1":
            return limit

        lane = int(elec_pad[1:]) if is_button else 0
        for trace, enter in self.future_pad_arrivals.get(elec_pad, ()):
            if trace > sec + 0.04 and enter > sec + 0.04:
                # Preserve exact Hold release onto a Slide launch.
                is_launch = lane > 0 and abs(enter - trace) <= 1.0 / TPS + EPS and any(
                    abs(trace - launch) <= EPS for launch in self.slide_launches.get(lane, ())
                )
                limit = min(limit, enter if is_launch else enter - RELEASE)

        two_hand_cap = sec + float(MAX_TWO_HAND_LOCK_BEATS) * state["beat_sec"]
        if len(self._salient_onsets):
            idx = int(np.searchsorted(self._salient_onsets, sec + 0.18, side="left"))
            if idx < len(self._salient_onsets):
                two_hand_cap = min(two_hand_cap, float(self._salient_onsets[idx]) + 0.06)

        for other_pad, other_end, other_is_button in state["sustain_locks"]:
            if other_end <= two_hand_cap + 0.02 or other_pad == elec_pad:
                continue
            if self.touch_policy == "off":
                two_hands = is_button and other_is_button
            else:
                two_hands = math.dist(CENTERS[elec_pad], CENTERS[other_pad]) > self.touch_radius + EPS
            if two_hands:
                limit = min(limit, two_hand_cap)
                break
        return limit

    def _button_viable_for_tap(
        self,
        state: dict[str, Any],
        btn: int,
        extra_buttons: tuple[int, ...] = (),
        *,
        allow_launch_exemption: bool = True,
    ) -> bool:
        sec = state["sec"]
        if sec > self.audio_end + 1e-8 or str(btn) in state["used_heads"] or btn in extra_buttons:
            return False
        elec = f"A{btn}"
        launch = allow_launch_exemption and self._is_slide_launch(btn, sec)
        if self._pad_in_cooldown(str(btn), sec) or self._pad_held(elec, sec, is_slide_launch=launch):
            return False
        if not launch and self._pad_in_slide_collision(elec, sec):
            return False

        extra_pts: list[tuple[float, float]] = []
        for b in (*extra_buttons, btn):
            if not (allow_launch_exemption and self._is_slide_launch(b, sec)):
                xy = CENTERS[f"A{b}"]
                if not any(math.dist(xy, p) <= 0.001 for p in state["non_touch_points"]) and not any(
                    math.dist(xy, p) <= 0.001 for p in extra_pts
                ):
                    extra_pts.append(xy)

        if len(state["non_touch_points"]) + len(extra_pts) > 2:
            return False
        return True

    def _hold_max_beats(self, state: dict[str, Any], btn: int) -> Fraction:
        if not self._button_viable_for_tap(state, btn, allow_launch_exemption=False):
            return Fraction(-1)
        max_end = self._max_sustain_end_sec(state, f"A{btn}", is_button=True)
        rem_sec = max_end - state["sec"]
        return Fraction(str(max(0.0, rem_sec) / state["beat_sec"]))

    def _touch_viable(self, state: dict[str, Any], pad_name: str) -> bool:
        sec = state["sec"]
        if sec > self.audio_end + 1e-8 or pad_name in state["used_heads"] or pad_name not in CENTERS:
            return False
        # A later Touch can connect this pad to a group that judges together,
        # including a pad already held down. Evaluate the completed event.
        return not self._pad_in_cooldown(pad_name, sec)

    def _touch_hold_max_beats(self, state: dict[str, Any], pad_name: str) -> Fraction:
        sec = state["sec"]
        if sec > self.audio_end + 1e-8 or pad_name in state["used_heads"] or pad_name not in CENTERS:
            return Fraction(-1)
        if self._pad_in_cooldown(pad_name, sec) or self._pad_held(pad_name, sec):
            return Fraction(-1)
        max_end = self._max_sustain_end_sec(state, pad_name, is_button=False)
        rem_sec = max_end - sec
        return Fraction(str(max(0.0, rem_sec) / state["beat_sec"]))

    def _grid_slide_max_beats(self, state: dict[str, Any]) -> Fraction:
        # D:B implies the parser's standard one-beat wait. D:R can encode a
        # shorter explicit wait and remains available until native validation.
        seconds = self.audio_end - state["sec"]
        return Fraction(str(seconds / state["beat_sec"])) - 1

    def _slide_head_viable(self, state: dict[str, Any], button: int) -> bool:
        if str(button) in state["used_heads"]:
            return False
        return self._slide_has_optional_head or self._button_viable_for_tap(state, button)

    def viable_kind_ids(self, beat: Fraction, completed_atoms: tuple[str, ...] = ()) -> tuple[int, ...]:
        if not self.enabled:
            return tuple(sorted(self.codec.kind_ids))
        state = self._beat_state(beat, completed_atoms)
        if state["sec"] > self.audio_end + 1e-8:
            return ()
        out: list[int] = []
        ids = self.codec.ids

        if "N:tap" in ids and ids["N:tap"] in self.codec.kind_ids:
            if any(
                "P:" + str(b) in ids and self._button_viable_for_tap(state, b)
                for b in range(1, 9)
            ):
                out.append(ids["N:tap"])

        if "N:touch" in ids and ids["N:touch"] in self.codec.kind_ids:
            touch_pads = [
                self.codec.tokens[tid][2:] for tid in self.codec.domains.get("<touch>", ())
            ]
            if any(self._touch_viable(state, p) for p in touch_pads):
                out.append(ids["N:touch"])

        if "N:hold" in ids and ids["N:hold"] in self.codec.kind_ids:
            if any(
                "P:" + str(b) in ids
                and self._has_grid_duration(self._hold_max_beats(state, b), allow_zero=True)
                for b in range(1, 9)
            ):
                out.append(ids["N:hold"])

        if "N:touch_hold" in ids and ids["N:touch_hold"] in self.codec.kind_ids:
            touch_pads = [
                self.codec.tokens[tid][2:] for tid in self.codec.domains.get("<touch>", ())
            ]
            if any(
                self._has_grid_duration(self._touch_hold_max_beats(state, p), allow_zero=True)
                for p in touch_pads
            ):
                out.append(ids["N:touch_hold"])

        if "N:slide" in ids and ids["N:slide"] in self.codec.kind_ids:
            if state["sec"] < self.audio_end and any(
                "P:" + str(b) in ids and self._slide_head_viable(state, b) for b in range(1, 9)
            ):
                out.append(ids["N:slide"])

        return tuple(sorted(out))

    def is_beat_open(self, beat: Fraction) -> bool:
        if not self.enabled:
            return True
        cached = self._beat_open_cache.get(beat)
        if cached is not None:
            return cached
        sec = self._second_at(beat)
        if sec < 0.0 or sec > self.audio_end + 1e-8:
            self._beat_open_cache[beat] = False
            return False
        result = bool(self.viable_kind_ids(beat))
        self._beat_open_cache[beat] = result
        return result

    def _integer_has_open_phase(self, whole: int, after: Fraction, end: Fraction) -> bool:
        phases = self.codec._phase_order
        lo = bisect_right(phases, after - whole)
        hi = bisect_left(phases, end - whole)
        return any(self.is_beat_open(whole + phase) for phase in phases[lo:hi])

    def filter_time_choices(
        self,
        allowed: tuple[int, ...],
        prefix: list[int],
        after: Fraction,
        end: Fraction,
    ) -> tuple[int, ...]:
        """Filter ``StreamCodec.time_choices`` so dead beats before audio_end are skipped."""
        if not self.enabled or not allowed:
            return allowed
        lo, hi = self.codec.integer_bounds(after, end)
        if lo > hi:
            return allowed

        first_phase = self.codec._phase_order[0]
        # The tempo clock is monotonic. Find the last possible integer without
        # stepping through a long chart tail outside the audio bounds.
        left, right = lo, hi + 1
        while left < right:
            middle = (left + right) // 2
            if self._second_at(middle + first_phase) <= self.audio_end + 1e-8:
                left = middle + 1
            else:
                right = middle
        hi = left - 1

        if not prefix:
            has_any = any(self._integer_has_open_phase(w, after, end) for w in range(lo, hi + 1))
            if has_any:
                return allowed
            return tuple(t for t in allowed if t != self.codec.at_id)

        if self.codec.phase_id in prefix:
            split = prefix.index(self.codec.phase_id)
            whole = int("".join(self.codec.digits[i] for i in prefix[1:split]))
            filtered = tuple(
                tid
                for tid in allowed
                if tid in self.codec.phases and self.is_beat_open(Fraction(whole) + self.codec.phases[tid])
            )
            return filtered

        digits = "".join(self.codec.digits[i] for i in prefix[1:])
        if hi < lo:
            return ()
        max_digits = len(str(hi))

        def has_open_integer(decimal_prefix: str) -> bool:
            value = int(decimal_prefix)
            sizes = range(max_digits - len(decimal_prefix) + 1) if value else range(1)
            for extra in sizes:
                scale = 10 ** extra
                lower = max(lo, value * scale)
                upper = min(hi, (value + 1) * scale - 1)
                if any(self._integer_has_open_phase(w, after, end) for w in range(lower, upper + 1)):
                    return True
            return False

        out: list[int] = []
        for tid in allowed:
            if tid == self.codec.phase_id:
                if digits and lo <= int(digits) <= hi and self._integer_has_open_phase(int(digits), after, end):
                    out.append(tid)
            elif tid in self.codec.digits:
                if has_open_integer(digits + self.codec.digits[tid]):
                    out.append(tid)
        return tuple(out)

    def filter_atom_start_choices(
        self,
        allowed: tuple[int, ...],
        beat: Fraction,
        completed_atoms: tuple[str, ...],
    ) -> tuple[int, ...]:
        if not self.enabled or not allowed:
            return allowed
        viable = set(self.viable_kind_ids(beat, completed_atoms))
        out = tuple(
            tid for tid in allowed if tid == self.codec.event_end_id or tid in viable
        )
        return out

    def filter_atom_next_choices(
        self,
        allowed: tuple[int, ...],
        atom: list[int],
        beat: Fraction,
        completed_atoms: tuple[str, ...],
        *,
        rejected_closures: list[int] | None = None,
    ) -> tuple[int, ...]:
        if not self.enabled or not allowed or not atom:
            return allowed

        tokens = [self.codec.tokens[i] for i in atom]
        kind = tokens[0]
        state = self._beat_state(beat, completed_atoms)
        p_tokens = [t[2:] for t in tokens if t.startswith("P:")]

        # Parse active duration state if currently inside D:B / D:R.
        active_dur_kind: str | None = None
        dur_body = ""
        for tok in tokens[1:]:
            if tok in ("D:B", "D:R"):
                active_dur_kind = tok
                dur_body = ""
            elif active_dur_kind is not None and tok.startswith("V:"):
                dur_body += tok[2:]
            elif tok == "[END_DURATION]" and active_dur_kind is not None:
                active_dur_kind = None

        if kind == "N:tap":
            extra_btns = tuple(int(p) for p in p_tokens if p.isdigit())
            out = [
                tid
                for tid in allowed
                if not (
                    self.codec.tokens[tid].startswith("P:")
                    and self.codec.tokens[tid][2:].isdigit()
                    and not self._button_viable_for_tap(state, int(self.codec.tokens[tid][2:]), extra_btns)
                )
            ]
            return tuple(out) if out else allowed

        if kind == "N:touch":
            if not p_tokens:
                out = [
                    tid
                    for tid in allowed
                    if not (
                        self.codec.tokens[tid].startswith("P:")
                        and not self._touch_viable(state, self.codec.tokens[tid][2:])
                    )
                ]
                return tuple(out) if out else allowed
            return allowed

        if kind == "N:hold":
            if not p_tokens:
                out = [
                    tid
                    for tid in allowed
                    if not (
                        self.codec.tokens[tid].startswith("P:")
                        and self.codec.tokens[tid][2:].isdigit()
                        and not self._has_grid_duration(
                            self._hold_max_beats(state, int(self.codec.tokens[tid][2:])),
                            allow_zero=True,
                        )
                    )
                ]
                return tuple(out) if out else allowed
            head_btn = int(p_tokens[0])
            d_max = self._hold_max_beats(state, head_btn)
            return self._filter_sustain_duration(
                allowed, active_dur_kind, dur_body, d_max,
                allow_zero=True, rejected_closures=rejected_closures,
            )

        if kind == "N:touch_hold":
            if not p_tokens:
                out = [
                    tid
                    for tid in allowed
                    if not (
                        self.codec.tokens[tid].startswith("P:")
                        and not self._has_grid_duration(
                            self._touch_hold_max_beats(state, self.codec.tokens[tid][2:]),
                            allow_zero=True,
                        )
                    )
                ]
                return tuple(out) if out else allowed
            pad_name = p_tokens[0]
            d_max = self._touch_hold_max_beats(state, pad_name)
            return self._filter_sustain_duration(
                allowed, active_dur_kind, dur_body, d_max,
                allow_zero=True, rejected_closures=rejected_closures,
            )

        if kind == "N:slide":
            if not p_tokens:
                return tuple(tid for tid in allowed if not self.codec.tokens[tid].startswith("P:")
                             or self._slide_head_viable(state, int(self.codec.tokens[tid][2:])))
            if "[END_DURATION]" not in tokens:
                total_max = self._grid_slide_max_beats(state)
                return self._filter_sustain_duration(
                    allowed, active_dur_kind, dur_body, total_max,
                    rejected_closures=rejected_closures,
                )

        return allowed

    def _has_grid_duration(self, d_max: Fraction, *, allow_zero: bool) -> bool:
        # The codec always provides every decimal digit and the fraction bar.
        # For any positive bound, 1 over a sufficiently large denominator fits.
        return "D:B" in self.codec.ids and (d_max > 0 or allow_zero and d_max == 0)

    def _filter_sustain_duration(
        self,
        allowed: tuple[int, ...],
        active_dur_kind: str | None,
        dur_body: str,
        d_max: Fraction,
        *,
        allow_zero: bool = False,
        rejected_closures: list[int] | None = None,
    ) -> tuple[int, ...]:
        if active_dur_kind is None:
            if not self._has_grid_duration(d_max, allow_zero=allow_zero):
                return tuple(tid for tid in allowed if tid != self.codec.ids.get("D:B"))
        elif active_dur_kind == "D:B":
            end_id = self.codec.ids["[END_DURATION]"]
            out = []
            for tid in allowed:
                if tid == end_id:
                    value = Fraction(dur_body)
                    if 0 <= value <= d_max and (value or allow_zero):
                        out.append(tid)
                    elif rejected_closures is not None:
                        rejected_closures.append(tid)
                    continue
                char = self.codec.tokens[tid][2:]
                if not dur_body and char == "0" and allow_zero and d_max >= 0:
                    out.append(tid)
                elif d_max > 0 and dur_body != "0" and (
                    char in "123456789" or char in "0/" and bool(dur_body)
                ):
                    # Each positive numerator or denominator prefix can reach
                    # the bound by choosing a sufficiently large denominator.
                    out.append(tid)
            return tuple(out)
        return allowed
