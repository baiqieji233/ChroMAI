"""Conservative static playability references with explicit coverage.

A geometric star overlap alone is NOT proof of an impossible Perfect. Slide
reference footprints do not reproduce the cabinet's ordered-area state machine.
Only same-pad press conflicts supported by the static rules are repairable.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

from maichart.generation.judgment_system import (
    PADS, PAD_RADII, OVERLAP_SECONDS, TAP_AVAILABLE_SECONDS, RELEASE_SECONDS,
    pad_center, pad_name, pads_at, touch_groups, touch_group_threshold,
)
from maichart.generation.maimai_physics import ButtonLayout


@dataclass
class SlideTrajectory:
    start_button: int
    pattern: str
    end_button: int
    start_time: float  # trace start, AFTER head wait
    duration: float
    source_id: tuple[int, int] = (-1, -1)

    @property
    def supported(self) -> bool:
        return self.pattern in {"-", "<", ">", "^"} and self.duration > 0

    def angular_span(self) -> float:
        if self.pattern == "-":
            return 0.0  # a chord through the board, never a ring arc
        cw = (self.end_button - self.start_button) % 8 * 45.0
        if self.pattern == "^":
            return -cw if cw <= 180 else 360 - cw
        clockwise = (self.pattern == ">") == (self.start_button in {1, 2, 7, 8})
        return -(cw or 360) if clockwise else ((360 - cw) or 360)

    def position_at(self, time: float) -> tuple[float, float] | None:
        if not self.supported or not self.start_time <= time <= self.start_time + self.duration:
            return None
        progress = (time - self.start_time) / self.duration
        a, b = pad_center(self.start_button), pad_center(self.end_button)
        if self.pattern == "-":
            return a[0] + progress * (b[0] - a[0]), a[1] + progress * (b[1] - a[1])
        angle = math.radians(ButtonLayout.button_angle(self.start_button) + progress * self.angular_span())
        return math.cos(angle), math.sin(angle)

    def star_position_at(self, time: float) -> int | None:
        """Legacy visual helper, not an input-collision rule."""
        position = self.position_at(time)
        if position is None:
            return None
        return next((int(p[1]) for p in pads_at(*position) if p.startswith("A")), None)

    def pad_intervals(self) -> list[tuple[str, float, float]]:
        return [(pad, self.start_time + low * self.duration, self.start_time + high * self.duration)
                for pad, low, high in _reference_intervals(self.start_button, self.pattern, self.end_button)] if self.supported else []

    def _raw_pad_intervals(self) -> list[tuple[str, float, float]]:
        """Reference sensor coverage along the trace.

        Lines use exact circle/segment intersections. Arcs are sampled at
        1/720 of the duration; their results are warning-only references.
        """
        if not self.supported:
            return []
        result = []
        if self.pattern == "-":
            a, b = pad_center(self.start_button), pad_center(self.end_button)
            dx, dy = b[0] - a[0], b[1] - a[1]
            aa = dx * dx + dy * dy
            if aa < 1e-12:
                return []
            for pad in PADS:
                cx, cy = pad_center(pad)
                x, y = a[0] - cx, a[1] - cy
                bb, cc = 2 * (x * dx + y * dy), x*x + y*y - PAD_RADII[pad[0]]**2
                discriminant = bb*bb - 4*aa*cc
                if discriminant < 0:
                    continue
                low = max(0.0, (-bb - math.sqrt(discriminant)) / (2*aa))
                high = min(1.0, (-bb + math.sqrt(discriminant)) / (2*aa))
                if low <= high:
                    result.append((pad, self.start_time + low*self.duration,
                                   self.start_time + high*self.duration))
            return sorted(result, key=lambda x: x[1])
        active = {}
        for step in range(721):
            time = self.start_time + step*self.duration/720
            covered = set(pads_at(*self.position_at(time)))
            for pad in list(active):
                if pad not in covered:
                    result.append((pad, active.pop(pad), time))
            for pad in covered:
                active.setdefault(pad, time)
        result.extend((pad, low, self.start_time + self.duration) for pad, low in active.items())
        return sorted(result, key=lambda x: x[1])


@lru_cache(maxsize=2048)
def _reference_intervals(start, pattern, end):
    return tuple(SlideTrajectory(start, pattern, end, 0.0, 1.0)._raw_pad_intervals())


@dataclass
class NoteEvent:
    note_type: str
    button: int | str
    time: float
    duration: float = 0.0
    slide_trajectory: SlideTrajectory | None = None
    raw: str = ""
    source_id: tuple[int, int] = (-1, -1)
    beat: Fraction = Fraction(0)
    protected: bool = False
    trajectories: list[SlideTrajectory] = field(default_factory=list)

    @property
    def pad(self) -> str:
        return pad_name(self.button)


@dataclass
class ConflictReport:
    has_conflict: bool
    conflict_type: str
    time: float
    notes: list[NoteEvent]
    severity: str
    message: str
    repair_target: tuple[int, int] | None = None
    evidence: str = "static_reference"
    action: str = "enforce"

    def to_dict(self) -> dict:
        return {"type": self.conflict_type, "severity": self.severity,
                "time": self.time, "message": self.message, "evidence": self.evidence,
                "action": self.action,
                "sources": [{"point_index": n.source_id[0], "note_index": n.source_id[1],
                             "beat": str(n.beat), "time_sec": n.time, "raw": n.raw,
                             "pad": n.pad} for n in self.notes],
                "repair_target": list(self.repair_target) if self.repair_target is not None else None}


class ConflictDetector:
    def __init__(self, enable_hard_conflicts=True, enable_soft_conflicts=False,
                 conflict_kind_modes=None, **_legacy):
        self.enable_hard = bool(enable_hard_conflicts)
        self.enable_soft = bool(enable_soft_conflicts)
        self.conflict_kind_modes = conflict_kind_modes
        self.ignored: list[ConflictReport] = []

    def detect_conflicts(self, notes: list[NoteEvent]) -> list[ConflictReport]:
        result = []
        if self.enable_hard:
            result.extend(self._overlaps(notes))
            # These warnings are part of judgement checks, not hand ergonomics.
            result.extend(self._slide_references(notes))
        if self.enable_soft:
            groups = defaultdict(list)
            for note in notes:
                if isinstance(note.button, int):
                    groups[note.time].append(note)
            for stamp, group in groups.items():
                if len({n.pad for n in group}) > 2:
                    result.append(ConflictReport(
                        True, "hand_assignment_review", stamp, group, "soft",
                        "同时有三个以上独立按键，需要结合持续动作检查双手分配；此项不自动拒绝。",
                        evidence="hand_assignment_unverified"))
        return self._apply_policy(result)

    @staticmethod
    def _policy_kind(issue: ConflictReport) -> str:
        kind = issue.conflict_type
        if kind.startswith("miacode_"):
            return kind[len("miacode_"):]
        if kind in {"same_pad_overlap", "hold_tap_overlap"}:
            return "Overlap"
        if kind in {"slide_pad_early_input", "touch_covers_slide_reference"}:
            return "TapOnSlide"
        if kind == "hand_assignment_review":
            return "MultiTouch"
        return kind

    def _apply_policy(self, issues):
        from maichart.generation.playability_policy import kind_mode

        kept = []
        self.ignored = []
        for issue in issues:
            action = kind_mode(self.conflict_kind_modes, self._policy_kind(issue))
            issue.action = action
            if action == "off":
                issue.repair_target = None
                self.ignored.append(issue)
                continue
            if action == "report":
                issue.repair_target = None
            kept.append(issue)
        return kept

    def _overlaps(self, notes):
        by_pad = defaultdict(list)
        for note in notes:
            by_pad[note.pad].append(note)
        result = []
        for pad, group in by_pad.items():
            group.sort(key=lambda n: n.time)
            for i, a in enumerate(group):
                a_end = a.time + (max(0, a.duration) if a.note_type in {"hold", "touch_hold"} else 0)
                for b in group[i+1:]:
                    if b.time > a_end + OVERLAP_SECONDS:
                        break
                    gap = b.time - a.time
                    hold = a.note_type in {"hold", "touch_hold"} and a.duration > 0
                    inside = hold and a.time + OVERLAP_SECONDS < b.time < a_end - OVERLAP_SECONDS
                    exact = abs(gap) <= 1e-9
                    head_pair_types = {"tap", "touch", "hold", "touch_hold"}
                    near_hold_head = (hold and 0 < gap <= OVERLAP_SECONDS
                                      and b.note_type in head_pair_types)
                    near_hold_start = (b.note_type in {"hold", "touch_hold"} and b.duration > 0
                                       and 0 < gap <= OVERLAP_SECONDS and a.note_type in head_pair_types)
                    if near_hold_head or near_hold_start:
                        if self.enable_soft:
                            kind = ("hold_tap_overlap"
                                    if {a.note_type, b.note_type} & {"tap", "touch"}
                                    else "same_pad_overlap")
                            result.append(ConflictReport(
                                True, kind, b.time, [a, b], "soft",
                                f"{pad} 上的 {a.raw or a.note_type} 与 {b.raw or b.note_type} "
                                "处于 Hold 起点的判定容差内，需人工检查。",
                                evidence="static_hold_head_tolerance"))
                        continue
                    if not (inside or exact or abs(b.time - a_end) <= OVERLAP_SECONDS):
                        continue
                    # Release at the intended hold tail is a legal re-articulation.
                    if hold and abs(b.time - a_end) <= 1e-9 and not exact:
                        continue
                    kind = ("hold_tap_overlap"
                            if inside and b.note_type in {"tap", "touch"}
                            else "same_pad_overlap")
                    hard = (inside or exact) and not (a.protected or b.protected)
                    target = b
                    if b.note_type == "slide":
                        target = a if a.note_type == "tap" else b
                    repairable = hard and target.note_type in {"tap", "touch"}
                    result.append(ConflictReport(
                        True, kind, b.time, [a, b], "hard" if hard else "soft",
                        f"{pad} 上的 {a.raw or a.note_type} 与 {b.raw or b.note_type} 存在按下区间重叠。",
                        target.source_id if repairable else None))
        # Connected simultaneous Touch groups can satisfy remaining children.
        simultaneous = defaultdict(list)
        for n in notes:
            if n.note_type == "touch":
                simultaneous[n.time].append(n)
        for group in simultaneous.values():
            for indexes in touch_groups([n.pad for n in group]):
                if len(indexes) <= 1:
                    continue
                children = [group[i] for i in indexes]
                affected = {id(n) for issue in result if issue.severity == "hard"
                            for n in issue.notes if any(n is c for c in children)}
                if len(children) - len(affected) >= touch_group_threshold(len(children)):
                    for issue in result:
                        if any(any(n is c for c in children) for n in issue.notes):
                            issue.severity, issue.repair_target = "soft", None
                            issue.message += " 同刻相邻 Touch 组可能联动判定，保留人工检查。"
        return result

    def _slide_references(self, notes):
        result = []
        for slide in notes:
            traces = slide.trajectories or ([slide.slide_trajectory] if slide.slide_trajectory else [])
            for trace in traces:
                if not trace.supported:
                    continue
                intervals = trace.pad_intervals()
                seen = set()
                for other in notes:
                    if other is slide or other.source_id == slide.source_id and other.source_id != (-1, -1):
                        continue
                    if not isinstance(other.button, int):
                        continue
                    for pad, enter, leave in intervals:
                        # Earlier trace input may auto-trigger a later Tap. Exact
                        # simultaneous contact can be intentional and is not hard.
                        delta = other.time - enter
                        if pad != other.pad or not 1/180 < delta <= TAP_AVAILABLE_SECONDS:
                            continue
                        key = id(other)
                        if key in seen:
                            continue
                        seen.add(key)
                        result.append(ConflictReport(
                            True, "slide_pad_early_input", other.time, [slide, other], "soft",
                            f"Slide 参考轨迹进入 {pad} 比该音符早 {delta*1000:.1f}ms；需检查提前判定。",
                            evidence="reference_footprint_not_runtime_judgement"))
                touches = [n for n in notes if n.note_type in {"touch", "touch_hold"}
                           and n.time <= trace.start_time + trace.duration
                           and n.time + max(n.duration, RELEASE_SECONDS) >= trace.start_time]
                if touches and intervals:
                    covered = {n.pad for n in touches}
                    required = {pad for pad, _, _ in intervals}
                    if required and required <= covered:
                        result.append(ConflictReport(
                            True, "touch_covers_slide_reference", trace.start_time,
                            [slide, *touches], "soft",
                            "Touch 同时覆盖了此 Slide 的全部参考判定区，需核对各区触发顺序与提前完成。",
                            evidence="ordered_area_runtime_not_simulated"))
        return result
