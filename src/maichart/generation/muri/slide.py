"""Faithful 180 Hz runtime judgment for slide and wifi tracks.

Ported from MiaCode MuriSlideWifiJudge.cpp at commit 6573efb611056bccd046dd17bc2f4cceda78bf88.
Upstream implementation is MIT licensed; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from .data import reference_data
from .model import EPS, TPS, diagnostic, head_key, has_head


def _bad_judge(judge_second: float, critical_second: float, delta: float, *, wifi: bool = False) -> bool:
    distance = abs(judge_second - critical_second)
    if distance <= delta + EPS:
        return False
    if not wifi and abs(judge_second - critical_second + 9.0 / TPS) <= 42.0 / TPS + EPS:
        return False
    return True


def _cause_is_foreign(cause: dict | None, state: dict) -> bool:
    if not cause:
        return False
    source = cause.get("marker_key", "")
    return bool(source and source != state["key"] and source != state.get("head_star_key"))


def _hit() -> dict:
    return {"judged": False, "skipped": False, "second": -1.0, "cause": None}


def _pads(area) -> list[str]:
    return [str(p) for p in area if p]


def _build_slide_sequence(marker: dict, slides: dict):
    sequence, partition, starts, ends = [], [], [], []
    is_l = is_special_l = False
    keys = marker.get("slide_segment_keys", [])
    if not keys:
        return None
    for index, key in enumerate(keys):
        entry = slides.get(key, {})
        segment = [_pads(area) for area in entry.get("judge_sequence", [])]
        segment = [area for area in segment if area]
        if not segment:
            return None
        start = 0
        if index == 0:
            sequence = segment[:]
            partition = [False] * len(sequence)
        else:
            shared = bool(sequence and segment and sequence[-1] == segment[0])
            if shared and partition:
                start = len(sequence) - 1
                partition[start] = True
            else:
                start = len(sequence)
            sequence.extend(segment[1:] if shared else segment)
            partition.extend([False] * (len(segment) - (1 if shared else 0)))
        starts.append(start)
        ends.append(len(sequence) - 1)
        if index == 0 and len(keys) == 1:
            is_l = bool(entry.get("is_l", False))
            is_special_l = bool(entry.get("is_special_l", False))
    partition.append(False)
    return sequence, partition, starts, ends, is_l, is_special_l


def slide_states(markers: list[dict]) -> list[dict]:
    data = reference_data()
    slides, wifi = data.get("slides", {}), data.get("wifi", {})
    config = data.get("config", {})
    states = []
    for marker in markers:
        kind = marker.get("type")
        if kind not in ("slide", "wifi") or marker.get("is_mine") or marker.get("track_mine"):
            continue
        seq_data = _build_slide_sequence(marker, slides) if kind == "slide" else None
        wifi_entry = wifi.get(marker.get("slide_track_key"), {}) if kind == "wifi" else {}
        lane_sequence = [[_pads(area) for area in lane] for lane in wifi_entry.get("tri_judge_sequence", [])]
        if kind == "slide" and seq_data is None:
            continue
        if kind == "wifi" and (len(lane_sequence) != 3 or not lane_sequence[1] or any(len(lane) != len(lane_sequence[1]) for lane in lane_sequence)):
            continue
        available = float(marker.get("second", 0.0)) - 15.0 / TPS
        duration = max(0.0, float(marker.get("end_second", 0.0)) - float(marker.get("slide_trace_second", 0.0)))
        if kind == "slide":
            sequence, partition, starts, ends, is_l, is_special_l = seq_data
            last = min(len(marker.get("slide_segment_durations", [])), len(marker.get("slide_segment_critical_proportions", []))) - 1
            if last >= 0:
                last_area_duration = max(0.0, float(marker["slide_segment_durations"][last])) * (1.0 - float(marker["slide_segment_critical_proportions"][last]))
            else:
                last_area_duration = 0.0
            critical = float(marker.get("end_second", 0.0)) - last_area_duration
            total = len(sequence)
            segment_expected = []
            segment_critical = []
            shoots = marker.get("slide_segment_shoot_seconds", [])
            durations = marker.get("slide_segment_durations", [])
            proportions = marker.get("slide_segment_critical_proportions", [])
            pad_times = marker.get("slide_segment_pad_enter_times", [])
            for index in range(len(starts)):
                shoot = float(shoots[index]) if index < len(shoots) else float(marker.get("slide_trace_second", 0.0))
                segment_duration = max(0.0, float(durations[index])) if index < len(durations) else max(0.0, float(marker.get("end_second", 0.0)) - shoot)
                expected = shoot
                if index < len(pad_times):
                    for entry in pad_times[index]:
                        expected = max(expected, shoot + float(entry.get("proportion", 0.0)) * segment_duration)
                segment_expected.append(expected)
                segment_critical.append(
                    shoot + float(proportions[index]) * segment_duration
                    if index < len(proportions) else expected
                )
        else:
            sequence, partition, starts, ends = [], [], [], []
            is_l = is_special_l = False
            total = len(lane_sequence[1])
            critical_proportion = float(marker.get("wifi_critical_proportion", 1.0))
            last_area_duration = (1.0 - critical_proportion) * duration
            critical = float(marker.get("end_second", 0.0)) - last_area_duration
            wifi_expected = float(marker.get("slide_trace_second", 0.0))
            for entry in marker.get("wifi_pad_enter_times", []):
                wifi_expected = max(wifi_expected, float(marker.get("slide_trace_second", 0.0)) + float(entry.get("proportion", 0.0)) * duration)
            wifi_critical = float(marker.get("slide_trace_second", 0.0)) + critical_proportion * duration
        critical_delta = min(24.0 * 60.0 * 60.0, 42.0 / TPS + max(0.0, last_area_duration) / 4.0)
        state = {
            "marker": marker, "key": marker.get("key") or head_key(marker),
            "head_star_key": head_key(marker) if has_head(marker) else "", "type": kind,
            "line": marker.get("line", 1), "col": marker.get("col", 1),
            "available_second": available, "shoot_second": marker.get("slide_trace_second", 0.0),
            "end_second": marker.get("end_second", 0.0), "critical_second": critical,
            "critical_delta": critical_delta, "sequence": sequence, "partition": partition,
            "segment_expected": segment_expected if kind == "slide" else [],
            "segment_critical": segment_critical if kind == "slide" else [],
            "wifi_expected": wifi_expected if kind == "wifi" else -1.0,
            "wifi_critical": wifi_critical if kind == "wifi" else -1.0,
            "segment_starts": starts, "segment_ends": ends, "is_l": is_l,
            "is_special_l": is_special_l, "total": total, "area_hits": [_hit() for _ in range(total)],
            "cur": 0, "segment": 0, "pressing": "", "judged": False,
            "judge_bad": False, "judge_second": -1.0, "judge_action": None,
            "lane_sequence": lane_sequence, "lane_cur": [0, 0, 0], "lane_pressing": ["", "", ""],
            "lane_finished": [False, False, False],
            "lane_hits": [[_hit() for _ in range(total)] for _ in range(3)],
            "lane_progress": [[-1.0] * (total + 1) for _ in range(3)],
            "pad_c_passed": not bool(config.get("wifi_need_c", config.get("wifiNeedC", False))),
            "pad_c_second": -1.0,
        }
        states.append(state)
    return states


def _slide_can_skip(state: dict) -> bool:
    cur, total = state["cur"], state["total"]
    if cur >= total - 1:
        return False
    if state["pressing"]:
        return True
    if len(state["segment_ends"]) == 1:
        if state["is_special_l"]:
            return cur not in (1, 3)
        if state["is_l"]:
            return cur != 1
    if total >= 4:
        return True
    return cur != total - 2


def _progress_slide(state: dict, now: float, down: dict, up: dict) -> bool:
    cur, total = state["cur"], state["total"]
    if cur >= total:
        return False
    if not state["pressing"]:
        for pad in state["sequence"][cur]:
            cause = down.get(pad)
            if cause is None:
                continue
            state["pressing"] = pad
            state["area_hits"][cur].update(judged=True, second=now, cause=cause)
            if cur >= total - 1:
                state["cur"] += 1
                state["judge_action"] = cause
            if state["partition"][state["cur"]]:
                state["segment"] += 1
            return True
    elif state["pressing"] not in down:
        state["pressing"] = ""
        state["cur"] += 1
        return True
    if _slide_can_skip(state):
        skipped = cur
        for pad in state["sequence"][cur + 1]:
            down_cause, up_cause = down.get(pad), up.get(pad)
            if down_cause is None and up_cause is None:
                continue
            if not state["area_hits"][skipped]["judged"]:
                state["area_hits"][skipped].update(judged=True, skipped=True, second=now)
            state["pressing"] = pad
            state["cur"] += 1
            state["area_hits"][state["cur"]].update(judged=True, second=now, cause=down_cause if down_cause is not None else up_cause)
            if state["cur"] >= total - 1:
                state["cur"] += 1
                state["judge_action"] = down_cause
            if state["partition"][state["cur"]]:
                state["segment"] += 1
            return True
    return False


def _progress_wifi_lane(state: dict, now: float, lane: int, down: dict, up: dict) -> bool:
    cur, total = state["lane_cur"][lane], state["total"]
    if cur >= total:
        return False
    if not state["lane_pressing"][lane]:
        for pad in state["lane_sequence"][lane][cur]:
            cause = down.get(pad)
            if cause is None:
                continue
            state["lane_pressing"][lane] = pad
            state["lane_hits"][lane][cur].update(judged=True, second=now, cause=cause)
            if cur >= total - 1:
                state["lane_cur"][lane] += 1
                state["lane_progress"][lane][state["lane_cur"][lane]] = now
                state["lane_finished"][lane] = True
                state["judge_action"] = cause
            return True
    elif state["lane_pressing"][lane] not in down:
        state["lane_pressing"][lane] = ""
        state["lane_cur"][lane] += 1
        state["lane_progress"][lane][state["lane_cur"][lane]] = now
        return True
    if cur < total - 1:
        for pad in state["lane_sequence"][lane][cur + 1]:
            down_cause, up_cause = down.get(pad), up.get(pad)
            if down_cause is None and up_cause is None:
                continue
            state["lane_pressing"][lane] = pad
            state["lane_cur"][lane] += 1
            state["lane_progress"][lane][state["lane_cur"][lane]] = now
            state["lane_hits"][lane][state["lane_cur"][lane]].update(judged=True, second=now, cause=down_cause if down_cause is not None else up_cause)
            if state["lane_cur"][lane] >= total - 1:
                state["lane_cur"][lane] += 1
                state["lane_progress"][lane][state["lane_cur"][lane]] = now
                state["lane_finished"][lane] = True
                state["judge_action"] = down_cause
            return True
    return False


def step_slides(states: list[dict], now: float, pad_states: dict, pad_up: dict) -> None:
    for state in states:
        if state["judged"] or now + EPS < state["available_second"]:
            continue
        if state["type"] == "slide":
            while state["cur"] < state["total"] and _progress_slide(state, now, pad_states, pad_up):
                pass
            done = state["cur"] >= state["total"]
            if done:
                state["judged"] = True
                state["judge_second"] = now
                final = state["area_hits"][-1] if state["area_hits"] else None
                state["judge_bad"] = _bad_judge(now, state["critical_second"], state["critical_delta"]) and bool(final and _cause_is_foreign(final["cause"], state))
            elif now > state["end_second"] + 24.0 * 60.0 * 60.0 + EPS:
                state.update(judged=True, judge_bad=True, judge_second=now)
        else:
            for lane in range(3):
                while state["lane_cur"][lane] < state["total"] and _progress_wifi_lane(state, now, lane, pad_states, pad_up):
                    pass
            if not state["pad_c_passed"] and state["lane_cur"][1] > 0 and "C" in pad_up:
                state["pad_c_passed"] = True
                state["pad_c_second"] = now
                if len(state["lane_hits"][1]) > 2:
                    state["lane_hits"][1][2].update(judged=True, second=now, cause=pad_up["C"])
            done = all(state["lane_finished"]) and state["pad_c_passed"]
            if done:
                state["judged"] = True
                state["judge_second"] = now
                state["judge_bad"] = _bad_judge(now, state["critical_second"], state["critical_delta"], wifi=True) and _cause_is_foreign(state["judge_action"], state)
            elif now > state["end_second"] + 24.0 * 60.0 * 60.0 + EPS:
                state.update(judged=True, judge_bad=True, judge_second=now)


def _latest_foreign(state: dict) -> dict | None:
    hits = state["area_hits"] if state["type"] == "slide" else [hit for lane in state["lane_hits"] for hit in lane]
    candidates = [hit for hit in hits if hit["second"] >= 0 and _cause_is_foreign(hit["cause"], state)]
    return max(candidates, key=lambda hit: hit["second"])["cause"] if candidates else None


def finish_slides(states: list[dict]) -> tuple[list[dict], dict]:
    diagnostics = []
    marker_states = {}
    for state in states:
        if not state["judged"]:
            state.update(judged=True, judge_bad=False, judge_second=max(state["end_second"], state["critical_second"]))
        marker = state["marker"]
        cause = _latest_foreign(state)
        early = bool(state["judge_bad"])
        if early:
            diagnostics.append(diagnostic("SlideTooFast", "Muri", marker, state["judge_second"], cause=cause,
                                          detail="slide judge completed outside its critical timing window"))
        if state["type"] == "slide":
            completed = []
            for end_idx in state["segment_ends"]:
                second = state["area_hits"][end_idx]["second"] if 0 <= end_idx < len(state["area_hits"]) else -1.0
                if second < 0 and end_idx == state["segment_ends"][-1]:
                    second = state["judge_second"]
                completed.append(second)
            segments = []
            for i, (start, end) in enumerate(zip(state["segment_starts"], state["segment_ends"])):
                areas = []
                for index in range(start, end + 1):
                    hit = state["area_hits"][index]
                    areas.append([{
                        "second": hit["second"], "pads": state["sequence"][index],
                        "skipped": hit["skipped"],
                        "cause_marker_key": (hit["cause"] or {}).get("marker_key", ""),
                        "cause_type": "skipped" if hit["skipped"] else (hit["cause"] or {}).get("source_type", ""),
                        "cause": "unknown" if not (hit["cause"] or {}).get("marker_key") else ("self" if (hit["cause"] or {}).get("marker_key") == state["key"] else "other"),
                    }] if hit["judged"] else [])
                segments.append({"completed_second": completed[i], "area_checkpoints": areas})
                judge_areas = []
                for index in range(start, end + 1):
                    hit = state["area_hits"][index]
                    cause_marker = (hit["cause"] or {}).get("marker_key", "")
                    cause_kind = "unknown" if not cause_marker else ("self" if cause_marker == state["key"] else "other")
                    judge_areas.append({
                        "pads": state["sequence"][index] if hit["judged"] else [],
                        "time": hit["second"] if hit["judged"] else -1.0,
                        "skipped": hit["skipped"] if hit["judged"] else False,
                        "cause": cause_kind if hit["judged"] else "pending",
                        "cause_marker_key": cause_marker,
                        "cause_type": "skipped" if hit["skipped"] else (hit["cause"] or {}).get("source_type", ""),
                    })
                segments[-1]["judge_areas"] = judge_areas
                segments[-1]["expected_completed_second"] = state["segment_expected"][i]
                segments[-1]["critical_second"] = state["segment_critical"][i]
            marker_states[state["key"]] = {"early_cleared": early, "flash_second": state["judge_second"] if early else -1.0,
                                           "critical_second": state["critical_second"], "slide_segments": segments}
        else:
            lanes = []
            lane_judge_areas = []
            for lane in range(3):
                lanes.append([[{
                    "second": hit["second"], "pads": state["lane_sequence"][lane][index],
                    "skipped": hit["skipped"],
                    "cause_marker_key": (hit["cause"] or {}).get("marker_key", ""),
                    "cause_type": "skipped" if hit["skipped"] else (hit["cause"] or {}).get("source_type", ""),
                    "cause": "unknown" if not (hit["cause"] or {}).get("marker_key") else ("self" if (hit["cause"] or {}).get("marker_key") == state["key"] else "other"),
                }] if hit["judged"] else [] for index, hit in enumerate(state["lane_hits"][lane])])
                judge_areas = []
                for index, hit in enumerate(state["lane_hits"][lane]):
                    cause_marker = (hit["cause"] or {}).get("marker_key", "")
                    judge_areas.append({
                        "pads": state["lane_sequence"][lane][index] if hit["judged"] else [],
                        "time": hit["second"] if hit["judged"] else -1.0,
                        "skipped": hit["skipped"] if hit["judged"] else False,
                        "cause": ("unknown" if not cause_marker else ("self" if cause_marker == state["key"] else "other")) if hit["judged"] else "pending",
                        "cause_marker_key": cause_marker,
                        "cause_type": "skipped" if hit["skipped"] else (hit["cause"] or {}).get("source_type", ""),
                    })
                lane_judge_areas.append(judge_areas)
            marker_states[state["key"]] = {"early_cleared": early, "flash_second": state["judge_second"] if early else -1.0,
                                           "wifi_completed_second": state["judge_second"], "wifi_lane_areas": lanes,
                                           "wifi_lane_judge_areas": lane_judge_areas,
                                           "wifi_lane_progress_seconds": state["lane_progress"],
                                           "wifi_pad_c_second": state["pad_c_second"],
                                           "wifi_expected_completed_second": state["wifi_expected"],
                                           "wifi_critical_second": state["wifi_critical"]}
    return diagnostics, marker_states
