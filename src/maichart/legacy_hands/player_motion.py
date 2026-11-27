"""Plan and preview a realistic two-handed maimai DX performance and 3D chart view."""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
from pathlib import Path

from maichart.event_timing import parse_event_chart
from maichart.generation.judgment_system import touch_group_threshold, touch_groups
from maichart.notes import BASIC_COMPACT_CHORD_RE, _split_top_level_slashes
from maichart.tempo_clock import ScalarTempoClock

from maichart.generation.muri.data import reference_data
from maichart.generation.muri.model import RADII, PADS, RELEASE, TPS, has_head
from maichart.generation.muri.parser import _parse_one
from . import SWEEP_STYLES
from .player_geometry import _plane_xyz, button_xyz, machine_payload, machine_profile, slide_xyz, touch_xyz
from .player_kinematics import (
    ARM_OVERLAP_TOLERANCE_MM,
    JOINT_LIMITS,
    ArmFrame,
    PlayerProfile,
    _add,
    _dot,
    _scale,
    _smooth_fraction,
    _sub,
    arm_pair_clearance,
    arm_pose,
    free_motion,
    interpolate_arm_joints,
    profile_for,
    repair_elbow_clearance,
    resolve_bimanual_arms,
    segment_distance_3d,
    torso_joints,
)
from .player_stance import NEUTRAL_TORSO, StancePlanner, _posture_choices, step_torso

Vec3 = tuple[float, float, float]
MAX_CONTACTS = 8000
MAX_FRAMES = 18000
MAX_DURATION_SEC = 1800
BEAM_WIDTH = 16
CONTACT_PRIORITY = ('idle', 'travel', 'tap', 'touch', 'hold', 'touch_hold', 'slide', 'wifi')
SUSTAIN_KINDS = frozenset(('hold', 'touch_hold', 'slide', 'wifi'))
STATIC_HOLD_KINDS = frozenset(('hold', 'touch_hold'))
BUTTON_CONTACT_OFFSET_MM = 45.
TOUCH_CONTACT_OFFSET_MM = 15.
# Native sensor areas have extent. Use the smallest area here, scaled into
# cabinet coordinates, so a hand need not follow the visual star's exact center.
SLIDE_CONTACT_OFFSET_MM = min(RADII.values()) * machine_profile().screen_radius_mm / 240.
BALLISTIC_NET_DIST_MM = 290.
BALLISTIC_SPEED_MM_S = 5200.
SLIDE_MEETING_MM = 78.
MIN_TRAVEL_SEC = .030
COHOLD_TRANSITION_SEC = .12
COHOLD_PLACEMENT_MM_S = 2200.
TORSO_ASSIST_MARGIN_MM = 85.

_TILT_RAD = math.radians(machine_profile().tilt_deg_from_vertical)
_PANEL_NORMAL: Vec3 = (0., math.sin(_TILT_RAD), math.cos(_TILT_RAD))

# Keep legacy helper alias for tests importing _forearm_distance.
_forearm_distance = segment_distance_3d


def _ergonomic_button_xyz(lane: int | str, *, star_head: bool = False) -> Vec3:
    """Real players strike the inner/middle rim of outer buttons (~288-304mm),
    especially on Slide star heads so the palm flows straight onto the glass."""
    raw = button_xyz(lane)
    shrink = .90 if star_head else .95
    # Shrink in the panel plane, retaining the button's height above the glass.
    normal_offset = sum(a * b for a, b in zip(raw, _PANEL_NORMAL))
    return tuple(value * shrink + normal * normal_offset * (1. - shrink)
                 for value, normal in zip(raw, _PANEL_NORMAL))


def _lane_number(pad: str) -> int:
    return int(pad) if len(pad) == 1 and pad in '12345678' else 0


def _reachable_button_target(point, shoulder, reach):
    """Use a reachable part of the same button, keeping its surface height."""
    toward = _sub(shoulder, point)
    if math.hypot(*toward) <= reach:
        return point
    normal = sum(a*b for a,b in zip(toward, _PANEL_NORMAL))
    along = _sub(toward, _scale(_PANEL_NORMAL, normal))
    distance = math.hypot(*along)
    if distance < 1e-6:
        return point
    radius = math.sqrt(max(0., reach*reach-normal*normal))
    shift = min(BUTTON_CONTACT_OFFSET_MM, distance, max(0., distance-radius))
    return _add(point, _scale(along, shift/distance))


def _ring_step(prev_lane: int, next_lane: int) -> int:
    """Signed shortest button-ring step in [-4, 4]; +/-1 means adjacent buttons."""
    if not (1 <= prev_lane <= 8 and 1 <= next_lane <= 8) or prev_lane == next_lane:
        return 0
    diff = (next_lane - prev_lane) % 8
    return diff - 8 if diff > 4 else diff


def _arc_button_motion(start: Vec3, end: Vec3, fraction: float, lift_mm: float, *, smooth=True) -> Vec3:
    """Sweep smoothly along the circular button rim instead of cutting across screen center."""
    prof = machine_profile()
    cos_t, sin_t = math.cos(_TILT_RAD), math.sin(_TILT_RAD)
    up1 = start[1] * cos_t - start[2] * sin_t
    up2 = end[1] * cos_t - end[2] * sin_t
    r1 = math.hypot(start[0], up1)
    r2 = math.hypot(end[0], up2)
    if r1 < 120. or r2 < 120.:
        return free_motion(start, end, fraction, lift_mm, panel_normal=_PANEL_NORMAL)
    a1 = math.atan2(up1, start[0])
    a2 = math.atan2(up2, end[0])
    da = math.remainder(a2 - a1, math.tau)
    u = max(0., min(1., fraction))
    p = _smooth_fraction(u) if smooth else u
    ang = a1 + da * p
    rad = r1 + (r2 - r1) * p
    lift = max(0., lift_mm) * (4. * u * (1. - u)) ** 2
    return _plane_xyz(rad * math.cos(ang), rad * math.sin(ang), prof, prof.button_offset_mm + lift)


def _downsample_glass_path(samples: tuple[tuple[float, float], ...], max_points: int = 20) -> list[list[float]]:
    if not samples:
        return []
    n = len(samples)
    if n <= max_points:
        chosen = samples
    else:
        chosen = tuple(samples[round(i * (n - 1) / (max_points - 1))] for i in range(max_points))
    return [[round(v, 1) for v in slide_xyz(px, py)] for px, py in chosen]


@dataclass(frozen=True, slots=True)
class SlideSegment:
    start: float
    end: float
    samples: tuple[tuple[float, float], ...]
    sample_start: float | None = None
    sample_end: float | None = None

    def point(self, second: float) -> Vec3:
        return slide_xyz(*self.raw_xy(second))

    def raw_xy(self, second: float) -> tuple[float, float]:
        start = self.start if self.sample_start is None else self.sample_start
        end = self.end if self.sample_end is None else self.sample_end
        if end <= start + 1e-9:
            return self.samples[-1]
        progress = min(1., max(0., (second - start) / (end - start)))
        index = progress * (len(self.samples) - 1)
        low = int(index)
        high = min(low + 1, len(self.samples) - 1)
        a, b = self.samples[low], self.samples[high]
        f = index - low
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def _slice_segments(segments: tuple[SlideSegment, ...], t0: float, t1: float) -> tuple[SlideSegment, ...]:
    """Extract an exact continuous sub-interval [t0, t1] of a Slide's trajectory."""
    out: list[SlideSegment] = []
    for seg in segments:
        lo = max(t0, seg.start)
        hi = min(t1, seg.end)
        if hi <= lo + 1e-5:
            continue
        if abs(lo - seg.start) <= 1e-5 and abs(hi - seg.end) <= 1e-5:
            out.append(seg)
            continue
        out.append(SlideSegment(lo, hi, seg.samples,
                                seg.start if seg.sample_start is None else seg.sample_start,
                                seg.end if seg.sample_end is None else seg.sample_end))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Contact:
    id: int
    start: float
    end: float
    kind: str
    pad: str
    point: Vec3
    segments: tuple[SlideSegment, ...] = ()
    touch_group: int | None = None
    group_required: int = 1
    is_ex: bool = False
    is_star_head: bool = False
    in_spin_run: bool = False
    slide_chain: int | None = None
    judge_marker: dict | None = field(default=None, compare=False, repr=False)
    sweep_predecessor: int | None = None
    brush: bool = False
    fingertip_trace: bool = False

    @property
    def source_id(self) -> int:
        return self.id if self.slide_chain is None else self.slide_chain

    def at(self, second: float) -> Vec3:
        if not self.segments:
            return self.point
        segment = next((part for part in self.segments if second <= part.end), self.segments[-1])
        return segment.point(second)

    @property
    def is_button(self) -> bool:
        return self.kind in ('tap', 'hold')

    @property
    def representative_x(self) -> float:
        """Hemisphere bias: for a Slide segment, weight where it goes (65%) as well as where it starts (35%)."""
        if not self.segments:
            return self.point[0]
        return .35 * self.at(self.start)[0] + .65 * self.at(self.end)[0]


def _issue(kind: str, second: float, message: str, **details) -> dict:
    return dict(time_sec=round(max(0., second), 6), kind=kind, message=message, **details)


def _slide_segments(marker, refs, paths) -> tuple[SlideSegment, ...]:
    parts = []
    kind = marker['type']
    keys = marker['slide_segment_keys']
    starts = marker['slide_segment_shoot_seconds']
    durations = marker['slide_segment_durations']
    if not keys or not len(keys) == len(starts) == len(durations):
        raise ValueError('滑条分段时间不完整')
    for key, start, duration in zip(keys, starts, durations):
        cache_key = (kind, key)
        if cache_key not in paths:
            if kind == 'wifi':
                lanes = refs['wifi'][key].get('lane_samples', ())
                raw = lanes[1] if len(lanes) == 3 else ()
            else:
                entry = refs['slides'][key]
                raw = entry.get('real_path_samples') or entry.get('samples') or ()
            paths[cache_key] = tuple((float(p['x']), float(p['y'])) for p in raw)
        points = paths[cache_key]
        if len(points) < 2 or duration <= 0 or not math.isfinite(start + duration):
            raise ValueError('缺少可用的滑条轨迹')
        parts.append(SlideSegment(float(start), float(start + duration), points))
    return tuple(parts)


def _wifi_visual_lanes(marker, refs, wifi_cache: dict) -> list[list[list[float]]]:
    key = marker.get('slide_track_key') or (marker.get('slide_segment_keys') or ('',))[0]
    if key not in wifi_cache:
        lanes = refs.get('wifi', {}).get(key, {}).get('lane_samples', ())
        wifi_cache[key] = [
            _downsample_glass_path(tuple((float(p['x']), float(p['y'])) for p in lane), max_points=14)
            for lane in lanes
        ] if len(lanes) == 3 else []
    return wifi_cache[key]


def _split_intersecting_slides(contacts: list[Contact]) -> list[Contact]:
    """Offer handoffs at half-screen crossings, joined segments and nearby tracks."""
    slide_indices = sorted(
        (i for i, c in enumerate(contacts) if c.kind == 'slide' and c.segments and c.end - c.start >= .08),
        key=lambda i: (contacts[i].start, contacts[i].end, contacts[i].pad),
    )
    # When an already visible return uses the very same line, covering two
    # sensors at once on the outward leg can finish the return prematurely.
    recent_lines = {}
    narrow = set()
    for idx in slide_indices:
        c = contacts[idx]
        marker = c.judge_marker
        if not marker or len(marker['slide_segment_keys']) != 1 or marker['slide_track_key'][1:2] != '-':
            continue
        key = c.pad, str(marker['end_lane'])
        back = recent_lines.get((key[1], key[0]))
        if back is not None:
            out = contacts[back]
            if (marker['second'] <= out.start and
                    -1e-6 <= c.start-out.end <= out.end-out.start+RELEASE):
                narrow.add(back)
                narrow.add(idx)
        recent_lines[key] = idx
    if narrow:
        contacts = [replace(c, fingertip_trace=True) if i in narrow else c for i, c in enumerate(contacts)]
    split_times: dict[int, set[float]] = defaultdict(set)
    for index in slide_indices:
        contact = contacts[index]
        for segment in contact.segments:
            if contact.start + .025 < segment.end < contact.end - .025:
                split_times[index].add(segment.end)
            for sample, (a, b) in enumerate(zip(segment.samples, segment.samples[1:])):
                if a[0] == b[0] or a[0] * b[0] > 0.:
                    continue
                fraction = (sample + a[0] / (a[0] - b[0])) / (len(segment.samples) - 1)
                second = segment.start + fraction * (segment.end - segment.start)
                if contact.start + .025 < second < contact.end - .025:
                    split_times[index].add(round(second, 6))
    for pos_a in range(len(slide_indices)):
        ia = slide_indices[pos_a]
        ca = contacts[ia]
        for pos_b in range(pos_a + 1, len(slide_indices)):
            ib = slide_indices[pos_b]
            cb = contacts[ib]
            if cb.start >= ca.end - .03:
                break
            t0 = max(ca.start, cb.start)
            t1 = min(ca.end, cb.end)
            if t1 - t0 < .06:
                continue
            # Approaching tracks can share a hand even without an X-shaped crossing.
            best_t = None
            best_d = math.inf
            for step in range(2, 23):
                t = t0 + (t1 - t0) * (step / 24.)
                d = math.dist(ca.at(t), cb.at(t))
                if d < best_d:
                    best_d = d
                    best_t = t
            if best_t is not None and best_d <= SLIDE_MEETING_MM:
                if ca.start + .025 < best_t < ca.end - .025:
                    split_times[ia].add(round(best_t, 6))
                if cb.start + .025 < best_t < cb.end - .025:
                    split_times[ib].add(round(best_t, 6))

    if not split_times:
        return contacts

    rebuilt: list[Contact] = []
    next_id = max(c.id for c in contacts) + 1
    for idx, c in enumerate(contacts):
        if idx not in split_times:
            rebuilt.append(c)
            continue
        cuts = [c.start]
        for second in sorted(split_times[idx]):
            if second - cuts[-1] >= .025 and c.end - second >= .025:
                cuts.append(second)
        cuts.append(c.end)
        if len(cuts) <= 2:
            rebuilt.append(c)
            continue
        chain_id = c.id
        for part_i, (a_t, b_t) in enumerate(zip(cuts, cuts[1:])):
            sub_segs = _slice_segments(c.segments, a_t, b_t)
            if not sub_segs:
                continue
            cid = c.id if part_i == 0 else next_id
            if part_i > 0:
                next_id += 1
            rebuilt.append(replace(c, id=cid, start=sub_segs[0].start, end=sub_segs[-1].end,
                                   point=sub_segs[0].point(sub_segs[0].start), segments=sub_segs,
                                   slide_chain=chain_id))
    return rebuilt


def _mark_unidirectional_spin_runs(contacts: list[Contact], beats_by_second) -> list[Contact]:
    """Find directed runs on the button ring, including a sweep landing on a Hold or EACH."""
    by_time: dict[float, list[int]] = defaultdict(list)
    for idx, c in enumerate(contacts):
        if not c.segments:
            by_time[c.start].append(idx)
    onset_groups = [by_time[t] for t in sorted(by_time)]
    solo_seq: list[int] = []
    for group in onset_groups:
        if len(group) == 1 and contacts[group[0]].kind == 'tap' and _lane_number(contacts[group[0]].pad):
            solo_seq.append(group[0])
        else:
            solo_seq.append(-1)

    spin_indices: set[int] = set()
    predecessors: dict[int, int] = {}
    longer_run_steps: set[tuple[int, int]] = set()
    preceding = {after: before for before, after in zip(solo_seq, solo_seq[1:]) if before >= 0 and after >= 0}
    solo_offsets = {index: pos for pos, index in enumerate(solo_seq) if index >= 0}
    hold_releases = sorted(c.end for c in contacts if c.kind == 'hold')
    holds = [c for c in contacts if c.kind == 'hold']
    hold_starts = [c.start for c in holds]
    hold_until = []
    for c in holds:
        hold_until.append(max(c.end, hold_until[-1] if hold_until else -math.inf))

    def paired_sweep_at_release(run):
        # One hand can prepare the next sweep during the first. Look for a
        # discontinuity in position at unchanged fast pulse, then another run.
        start = contacts[run[0]].start
        released = bisect_right(hold_releases, start + 1e-6) - 1
        if released < 0 or start - hold_releases[released] > .055:
            return False
        pos = solo_offsets[run[-1]] + 1
        following = solo_seq[pos:pos+3]
        if len(following) < 3 or min(following) < 0:
            return False
        steps = [contacts[i] for i in [*run, *following]]
        if max(b.start-a.start for a,b in zip(steps, steps[1:])) > .055:
            return False
        end, pickup, second, third = steps[-4:]
        jump = _ring_step(int(end.pad), int(pickup.pad))
        step = _ring_step(int(pickup.pad), int(second.pad))
        return abs(jump) >= 2 and abs(step) == 1 and step == _ring_step(int(second.pad),int(third.pad))

    def chord_pickup(run):
        # A rim sweep can start from one side of a prepared pair, and land
        # straight into a second pair with the other hand already available.
        lo, hi = solo_offsets[run[0]], solo_offsets[run[-1]]
        if len(run) < 4 or lo == 0 or hi+1 >= len(onset_groups):
            return False
        before, after = onset_groups[lo-1], onset_groups[hi+1]
        if len(before) != 2 or len(after) != 2 or any(not contacts[i].is_button for i in (*before,*after)):
            return False
        first, last = contacts[run[0]], contacts[run[-1]]
        return (first.pad in {contacts[i].pad for i in before}
                and first.start-contacts[before[0]].start <= .35
                and contacts[after[0]].start-last.start <= .11)

    def finish_run(run: list[int], direction: int, *, sixteenths: bool):
        if len(run) < 3:
            return
        if len(run) > 3:
            longer_run_steps.update(zip(run, run[1:]))
        first, last = contacts[run[0]], contacts[run[-1]]
        # A fast sweep needs a prepared striking hand while the other hand
        # holds, releases or approaches a sustained button.
        previous_hold = bisect_right(hold_starts, first.start) - 1
        beside_hold = (len(run) >= 5 and
                       max(contacts[b].start-contacts[a].start for a,b in zip(run,run[1:])) <= .060 and
                       (bisect_left(hold_releases, first.start-.06) < bisect_right(hold_releases, last.start+.001)
                        or bisect_left(hold_starts, first.start) < bisect_right(hold_starts, last.start+.15)
                        or (previous_hold >= 0 and hold_until[previous_hold] >= first.start)))
        if not sixteenths:
            spin_indices.update(run)
            if len(run) < 6 and not (beside_hold or paired_sweep_at_release(run) or chord_pickup(run)):
                return
        # A directional passage covering the ring can carry its
        # velocity across hands' usual sides. Admit one longer pickup interval
        # when its direction and spacing agree with the established passage.
        pickup = preceding.get(run[0])
        if pickup is not None and not sixteenths:
            before, first, second = (contacts[index] for index in (pickup, run[0], run[1]))
            gap = first.start - before.start
            step = _ring_step(_lane_number(before.pad), _lane_number(first.pad))
            if step == direction and .078 < gap <= min(.13, 2 * (second.start - first.start) + 1e-6):
                run = [pickup, *run]
        # Keep the normal assignment below the movement model's time floor.
        # At those spacings, the short tap contacts can occupy most of the arc.
        if any(contacts[b].start - contacts[a].start < MIN_TRAVEL_SEC for a, b in zip(run, run[1:])):
            return
        # A trace under the sweep occupies *one* hand. Keep the trajectory
        # evidence for the free hand; _plan checks each hand's actual contacts.
        spin_indices.update(run)
        for before, after in zip(run, run[1:]):
            predecessors[after] = contacts[before].id

    for sixteenths in (False, True):
        run: list[int] = []
        direction = 0
        for index in solo_seq:
            if index < 0 or not run:
                finish_run(run, direction, sixteenths=sixteenths)
                run, direction = ([index] if index >= 0 else []), 0
                continue
            previous, current = contacts[run[-1]], contacts[index]
            step = _ring_step(_lane_number(previous.pad), _lane_number(current.pad))
            if sixteenths:
                a, b = beats_by_second.get(previous.start), beats_by_second.get(current.start)
                timed = a is not None and b is not None and 4 * (b - a) == 1
            else:
                timed = 1e-4 < current.start - previous.start <= .078
            if abs(step) == 1 and timed:
                if direction not in (0, step):
                    finish_run(run, direction, sixteenths=sixteenths)
                    run = [run[-1]]
                run.append(index)
                direction = step
            else:
                finish_run(run, direction, sixteenths=sixteenths)
                run, direction = [index], 0
        finish_run(run, direction, sixteenths=sixteenths)

    # An independent three-note gesture can span a triplet, a Hold onset,
    # or one side of an EACH. The longer-run planner owns its own exits: a Hold
    # after a long sweep may be taken by the spare hand to prepare the next run.
    for i, j, landing in zip(solo_seq, solo_seq[1:], onset_groups[2:]):
        if i < 0 or j < 0 or (i, j) in longer_run_steps or len(landing) > 2:
            continue
        if not all(contacts[k].is_button for k in landing):
            continue
        a, b = contacts[i], contacts[j]
        step = _ring_step(int(a.pad), int(b.pad))
        if abs(step) != 1:
            continue
        gaps = b.start - a.start, contacts[landing[0]].start - b.start
        if min(gaps) < MIN_TRAVEL_SEC or max(gaps) > min(.13, 1.20 * min(gaps)):
            continue
        target = (int(b.pad) - 1 + step) % 8 + 1
        ends = [k for k in landing if int(contacts[k].pad) == target]
        if len(ends) == 1:
            end = ends[0]
            spin_indices.update((i, j, end))
            predecessors.setdefault(j, a.id)
            predecessors.setdefault(end, b.id)

    if not spin_indices:
        return contacts
    return [replace(c, in_spin_run=True, sweep_predecessor=predecessors.get(i))
            if i in spin_indices else c for i, c in enumerate(contacts)]


def _extract_chart_contacts_and_visuals(chart, *, include_event_ids=False) -> tuple[list[Contact], list[dict], list[dict], float]:
    changes = {t.beat: t for t in chart.tempo_changes}
    clock = ScalarTempoClock([changes[beat] for beat in sorted(changes)], chart.audio_first)
    chart_end = float(clock.at(chart.end_beat))
    if len(chart.points) > MAX_CONTACTS or chart_end > MAX_DURATION_SEC:
        raise ValueError('谱面超出动作预览的长度上限')
    refs = reference_data()
    contacts: list[Contact] = []
    visual_notes: list[dict] = []
    issues: list[dict] = []
    paths: dict = {}
    wifi_cache: dict = {}
    heads: dict[tuple[float, str], dict] = {}
    touch_by_each: dict[tuple[int, int], list[int]] = defaultdict(list)
    visual_heads_by_group: dict[int, list[dict]] = defaultdict(list)
    visual_slides_by_group: dict[int, list[dict]] = defaultdict(list)
    beats_by_second = {}

    def add(start, end, kind, pad, xyz, segments=(), *, is_ex=False, is_star_head=False, judge_marker=None):
        if len(contacts) >= MAX_CONTACTS:
            raise ValueError('谱面超出动作预览的音符数上限')
        if not math.isfinite(start + end) or abs(start) > MAX_DURATION_SEC or abs(end) > MAX_DURATION_SEC:
            raise ValueError('动作时间超出预览范围')
        start, end = round(start, 6), round(end, 6)
        if end <= start:
            end = round(start + RELEASE, 6)
            kind = 'touch' if kind == 'touch_hold' else 'tap'
        contacts.append(Contact(len(contacts), start, end, kind, pad, xyz, segments,
                                 is_ex=is_ex, is_star_head=is_star_head, judge_marker=judge_marker))

    group_counter = 0
    for index, point in enumerate(chart.points, 1):
        first_visual = len(visual_notes)
        second = float(clock.at(point.beat))
        beats_by_second[round(second, 6)] = point.beat
        atoms_with_group = [(each, raw) for each, text in enumerate(point.note_text.split('`'))
                            for raw in _split_top_level_slashes(text)]
        group_ids: dict[int, int] = {}
        for child, (each, raw) in enumerate(atoms_with_group, 1):
            if each not in group_ids:
                group_counter += 1
                group_ids[each] = group_counter
            gid = group_ids[each]
            compact = BASIC_COMPACT_CHORD_RE.fullmatch(raw)
            atoms = [lane + compact['modifiers'] for lane in compact['positions']] if compact else [raw]
            for atom in atoms:
                try:
                    markers = _parse_one(atom, index, child, second, 0, clock.bpm(point.beat), refs)
                    if not markers:
                        raise ValueError('音符格式未覆盖')
                except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                    issues.append(_issue('unknown_note', second, '这个音符尚无可用的动作表示', raw=atom, detail=str(exc)))
                    continue
                for marker in markers:
                    kind = marker['type']
                    if kind in ('slide', 'wifi'):
                        pad = str(marker['lane'])
                        head = (second, pad)
                        if has_head(marker):
                            if head not in heads:
                                head_ex = bool(marker.get('head_ex') or (marker.get('is_ex') and has_head(marker)))
                                add(second, second + RELEASE, 'tap', pad,
                                    _ergonomic_button_xyz(pad, star_head=True),
                                    is_ex=head_ex, is_star_head=True)
                                head_brk = bool(marker.get('head_break') or (marker.get('is_break') and not marker.get('track_break')))
                                v_star = dict(
                                    contact_id=contacts[-1].id,
                                    kind='star', lane=int(marker['lane']), start=round(second, 5), end=round(second, 5),
                                    brk=head_brk, ex=head_ex, each=False, double=False, group=gid,
                                )
                                heads[head] = v_star
                                visual_notes.append(v_star)
                                visual_heads_by_group[gid].append(v_star)
                            else:
                                heads[head]['double'] = True
                        if marker.get('track_mine'):
                            issues.append(_issue('unknown_note', second, '地雷滑条需要独立的避让模型', raw=atom))
                            continue
                        try:
                            parts = _slide_segments(marker, refs, paths)
                        except (ValueError, KeyError, TypeError, IndexError) as exc:
                            issues.append(_issue('unknown_note', second, '这个滑条的轨迹不完整', raw=atom, detail=str(exc)))
                            continue
                        add(parts[0].start, parts[-1].end, kind, pad, parts[0].point(parts[0].start), parts,
                            judge_marker=marker)
                        if kind == 'wifi':
                            issues.append(_issue('wifi_approximation', parts[0].start, 'WiFi 用扇形中心示意掌部位置，分轨覆盖待核对', raw=atom))
                            v_slide = dict(
                                contact_id=contacts[-1].id,
                                kind='wifi', lane=int(marker['lane']), end_lane=int(marker.get('end_lane') or marker['lane']),
                                onset=round(second, 5), trace=round(parts[0].start, 5), end=round(parts[-1].end, 5),
                                brk=bool(marker.get('track_break')), each=False, group=gid,
                                lanes=_wifi_visual_lanes(marker, refs, wifi_cache),
                            )
                        else:
                            v_slide = dict(
                                contact_id=contacts[-1].id,
                                kind='slide', lane=int(marker['lane']), end_lane=int(marker.get('end_lane') or marker['lane']),
                                onset=round(second, 5), trace=round(parts[0].start, 5), end=round(parts[-1].end, 5),
                                brk=bool(marker.get('track_break')), each=False, group=gid,
                                segments=[[round(seg.start, 5), round(seg.end, 5), _downsample_glass_path(seg.samples)] for seg in parts],
                            )
                        visual_notes.append(v_slide)
                        visual_slides_by_group[gid].append(v_slide)
                    elif marker.get('is_mine'):
                        issues.append(_issue('unknown_note', second, '地雷音符需要独立的避让模型', raw=atom))
                    elif kind in ('tap', 'hold', 'touch', 'touch_hold'):
                        pad = str(marker['touch_pad'] or marker['lane'])
                        is_ex = bool(marker.get('is_ex'))
                        xyz = touch_xyz(pad) if kind.startswith('touch') else button_xyz(pad)
                        end = float(marker['end_second']) if kind in ('hold', 'touch_hold') else second + RELEASE
                        add(second, end, kind, pad, xyz, is_ex=is_ex)
                        if kind == 'touch':
                            touch_by_each[index, each].append(len(contacts) - 1)
                        if kind in ('tap', 'hold'):
                            n_end = float(marker['end_second']) if kind == 'hold' and float(marker['end_second']) > second else second
                            v_btn = dict(
                                contact_id=contacts[-1].id,
                                kind='hold' if n_end > second + 1e-4 else 'tap',
                                lane=int(marker['lane']), start=round(second, 5), end=round(max(second, n_end), 5),
                                brk=bool(marker.get('is_break')), ex=is_ex, each=False, group=gid,
                            )
                            visual_notes.append(v_btn)
                            visual_heads_by_group[gid].append(v_btn)
                        else:
                            n_end = float(marker['end_second']) if kind == 'touch_hold' and float(marker['end_second']) > second else second
                            v_tch = dict(
                                contact_id=contacts[-1].id,
                                kind='touch_hold' if n_end > second + 1e-4 else 'touch',
                                pad=pad, xyz=[round(v, 1) for v in xyz],
                                start=round(second, 5), end=round(max(second, n_end), 5),
                                fw=bool(marker.get('is_firework')), each=False, group=gid,
                            )
                            visual_notes.append(v_tch)
                            visual_heads_by_group[gid].append(v_tch)
                    else:
                        issues.append(_issue('unknown_note', second, '这个音符尚无可用的动作表示', raw=atom))

        if include_event_ids:
            for note in visual_notes[first_visual:]:
                note['source_event'] = index-1

    for gid, h_list in visual_heads_by_group.items():
        if len(h_list) >= 2:
            for item in h_list:
                item['each'] = True
    for gid, s_list in visual_slides_by_group.items():
        if len(s_list) >= 2:
            for item in s_list:
                item['each'] = True

    for ids in touch_by_each.values():
        for group in touch_groups([contacts[i].pad for i in ids]):
            required = touch_group_threshold(len(group))
            if required == len(group):
                continue
            group_id = ids[group[0]]
            for member in group:
                i = ids[member]
                contacts[i] = replace(contacts[i], touch_group=group_id, group_required=required)

    # Enable hemisphere relay on X-crossing slides and pre-tag true unidirectional spin runs!
    contacts = _split_intersecting_slides(contacts)
    sorted_contacts = sorted(contacts, key=lambda c: (c.start, c.touch_group is not None, c.touch_group or 0,
                                                     -c.end, c.kind, c.pad, c.at(c.end), c.id))
    sorted_contacts = _mark_unidirectional_spin_runs(sorted_contacts, beats_by_second)
    sorted_visuals = sorted(visual_notes, key=lambda n: (n.get('onset', n.get('start', 0.)), n['end']))
    return sorted_contacts, sorted_visuals, issues, chart_end


def _read_contacts(chart) -> tuple[list[Contact], list[dict], float]:
    contacts, _, issues, chart_end = _extract_chart_contacts_and_visuals(chart)
    return contacts, issues, chart_end


def _center(contacts: tuple[Contact, ...], second: float) -> Vec3:
    points = [c.at(second) for c in contacts]
    return tuple(sum(axis) / len(points) for axis in zip(*points))


def _contact_span(contact: Contact, profile: PlayerProfile) -> float:
    allowance = (BUTTON_CONTACT_OFFSET_MM if contact.is_button else
                 SLIDE_CONTACT_OFFSET_MM if contact.segments else TOUCH_CONTACT_OFFSET_MM)
    return profile.hand_span_mm / 2 + allowance


def _fits_palm(contacts: tuple[Contact, ...], second: float, profile: PlayerProfile) -> bool:
    if len(contacts) < 2:
        return True
    pads = [c.pad for c in contacts if not c.segments]
    buttons = [c.pad for c in contacts if c.is_button]
    if len(set(pads)) != len(pads) or len(buttons) > 2:
        return False
    center = _center(contacts, second)
    return all(math.dist(center, contact.at(second)) <= _contact_span(contact, profile) for contact in contacts)


def _shared_motion_fits(active: tuple[Contact, ...], contact: Contact, profile: PlayerProfile) -> bool:
    together = (*active, contact)
    if not _fits_palm(together, contact.start, profile):
        return False
    if not any(c.segments for c in together):
        return True
    boundaries = sorted({min(c.end, contact.end) for c in active})
    previous = contact.start
    for end in boundaries:
        for fraction in (0.25, 0.5, 0.75, 0.999):
            second = previous + (end - previous) * fraction
            if not _fits_palm(tuple(c for c in together if c.end > second), second, profile):
                return False
        previous = end
    return True


def _shared_launch(first: Contact, second: Contact) -> bool:
    tap, trace = (first, second) if first.kind == 'tap' else (second, first)
    return (tap.kind == 'tap' and trace.kind == 'slide' and trace.judge_marker is not None
            and tap.pad == trace.pad and abs(tap.start-trace.start) < 1e-6
            and abs(trace.judge_marker['slide_trace_second']-tap.start) < 1e-6)


@dataclass(frozen=True, slots=True)
class HandState:
    last: Vec3
    ready: float
    active: tuple[Contact, ...] = ()
    last_pad: str = ''
    sweep_dir: int = 0
    last_chain: int | None = None
    last_contact: Contact | None = None

    def advance(self, second: float) -> HandState:
        active = tuple(c for c in self.active if c.end > second)
        released = tuple(c for c in self.active if c.end <= second)
        if not released:
            return self
        ready = max(c.end for c in released)
        ended_now = tuple(c for c in released if c.end == ready)
        last_pad = ended_now[-1].pad if len(ended_now) == 1 else self.last_pad
        last_chain = ended_now[-1].slide_chain if len(ended_now) == 1 else None
        return HandState(
            _center(ended_now, ready), ready, active, last_pad,
            self.sweep_dir, last_chain,
            ended_now[-1] if len(ended_now) == 1 else None,
        )


@dataclass(frozen=True, slots=True)
class Plan:
    hands: tuple[HandState, HandState]
    misses: int = 0
    effort: float = 0.
    parent: Plan | None = None
    choice: int = -1
    group_hits: int = 0
    last_strike_hand: int = -1
    last_strike_sec: float = -10.
    same_hand_streak: int = 0
    chord_compromises: float = 0.
    rapid_jumps: int = 0


def _requires_rapid_jump(distance: float, interval: float) -> bool:
    return (-1e-7 <= interval < .15 and distance - 80. >= BALLISTIC_NET_DIST_MM
            and (distance - 80.) / max(.025, interval) >= BALLISTIC_SPEED_MM_S)


def _reposition_interval(previous: Contact, onset: float) -> float:
    # A tap can leave the button once it triggers. A Hold or Slide has a
    # real release deadline, even when the following button is EX.
    return onset - (previous.start if previous.kind == 'tap' else previous.end)


def _is_sweep_continuation(current: HandState, contact: Contact) -> tuple[bool, int]:
    """Use run direction and EX/Touch evidence to distinguish a sweep from isolated strikes."""
    if current.active or contact.segments:
        return False, 0
    dt = contact.start - current.ready
    linked = (contact.sweep_predecessor is not None and current.last_contact is not None
              and current.last_contact.id == contact.sweep_predecessor)
    if dt < -.015 or (dt > .115 and not linked):
        return False, 0
    prev_lane = _lane_number(current.last_pad)
    next_lane = _lane_number(contact.pad)
    if prev_lane and next_lane:
        if current.last_contact is None or current.last_contact.kind != 'tap':
            return False, 0
        step = _ring_step(prev_lane, next_lane)
        if abs(step) == 1 and (contact.is_ex or contact.in_spin_run):
            if (contact.is_star_head and not contact.in_spin_run and not linked
                    and current.sweep_dir != step):
                # An isolated EX head after a jack is a new strike.
                # Only carry the sweep discount when movement is established.
                return False, 0
            if contact.sweep_predecessor is not None and not linked:
                return False, 0
            if current.sweep_dir == 0 or step == current.sweep_dir or contact.is_ex:
                return True, step
        return False, 0
    if not contact.is_button and current.last_pad and current.last_pad[0] in 'ABCDE':
        if contact.pad != current.last_pad and math.dist(current.last, contact.point) <= 158. and dt <= .11:
            return True, current.sweep_dir
    return False, 0


def _is_relay(previous: Contact | None, following: Contact) -> bool:
    return (previous is not None and previous.slide_chain is not None
            and following.slide_chain is not None and previous.source_id != following.source_id
            and abs(previous.end - following.start) < 1e-6)


@dataclass(frozen=True, slots=True)
class RelayMotion:
    previous: Contact
    following: Contact
    start: float
    end: float
    side_offset: float

    def point(self, second: float) -> Vec3:
        progress = _smooth_fraction((second - self.start) / (self.end - self.start))
        # Original samples retain their time mapping on both sides of the cut.
        # A small lateral arc lets the two palms exchange paths without sharing
        # the same center. Each palm must still cover its assigned contact.
        before = self.previous.at(second)
        after = self.following.at(second)
        center = tuple(a + progress * (b - a) for a, b in zip(before, after))
        return (center[0] + self.side_offset * 16. * progress**2 * (1. - progress)**2,
                center[1], center[2])


def _relay_motion(previous: Contact, following: Contact, profile: PlayerProfile, hand: int) -> RelayMotion | None:
    half_window = min(.06, (previous.end - previous.start) * .45,
                      (following.end - following.start) * .45)
    if half_window < .005:
        return None
    cut = following.start
    motion = RelayMotion(previous, following, cut - half_window, cut + half_window,
                         (-1. if hand == 0 else 1.) * min(30., profile.hand_span_mm * .2))
    for index in range(17):
        second = motion.start + (motion.end - motion.start) * index / 16
        target = previous if second < cut else following
        if math.dist(motion.point(second), target.at(second)) > _contact_span(target, profile):
            return None
    return motion


@lru_cache(maxsize=8)
def _sweep_stances(profile: PlayerProfile) -> tuple[tuple[float, Vec3], ...]:
    """Estimate one supported shoulder position for each hand's whole sweep."""
    rim = tuple(_ergonomic_button_xyz(lane) for lane in range(1, 9))
    choices = tuple(_posture_choices(profile))
    result = []
    for hand in (0, 1):
        pose, shoulders, _ = min(
            choices,
            key=lambda choice: (max(math.dist(choice[1][hand], point) for point in rim), choice[2]),
        )
        result.append((pose.lateral_mm, shoulders[hand]))
    return tuple(result)


def _stacked_chord_exits(groups):
    """Two hands on one flank, followed by one outward step at each end."""
    eligible = set()
    for pair, a, b in zip(groups, groups[1:], groups[2:]):
        if len(pair) != 2 or len(a) != 1 or len(b) != 1:
            continue
        first, second = a[0], b[0]
        if any(c.kind != 'tap' for c in (*pair, first, second)) or second.start - pair[0].start > .40:
            continue
        top, bottom = sorted(pair, key=lambda c: -c.point[1])
        if (top.point[0] * bottom.point[0] <= 0 or
                abs(top.point[0]-bottom.point[0]) >= .5*(top.point[1]-bottom.point[1])):
            continue
        high, low = sorted((first, second), key=lambda c: -c.point[1])
        if (high.point[1] > top.point[1] and low.point[1] < bottom.point[1]
                and abs(_ring_step(int(top.pad), int(high.pad))) == 1
                and abs(_ring_step(int(bottom.pad), int(low.pad))) == 1):
            eligible.update(c.id for c in pair)
    return eligible


def _translating_pairs(groups):
    """For each tap, identify its preceding tap in a translating two-hand phrase."""
    links = {}
    pending = []
    direction = 0
    previous = {}
    for group in groups:
        current = {int(c.pad): c for c in group} if len(group)==2 and all(
            c.kind == 'tap' and not c.is_star_head for c in group) else {}
        step = 0
        if len(current)==len(previous)==2:
            gap = group[0].start - next(iter(previous.values())).start
            if RELEASE <= gap <= .32:
                for shift in (1,-1):
                    if {(lane-1+shift)%8+1 for lane in previous} == current.keys():
                        step = shift
                        break
        if not step or step != direction:
            if len(pending)>=2:
                for pair in pending:
                    links.update(pair)
            pending = []
        if step:
            pending.append({current[(k-1+step)%8+1].id: old.id for k,old in previous.items()})
        direction, previous = step,current
    if len(pending)>=2:
        for pair in pending:
            links.update(pair)
    return links


def _hold_pair_entries(contacts):
    """Starting a staggered pair after a rest allows both hands to approach in order."""
    entries = {}
    last_end = -math.inf
    for first, following in zip(contacts, contacts[1:]):
        if (first.kind == following.kind == 'hold' and first.start-last_end >= .30
                and first.start+RELEASE < following.start < first.end-RELEASE
                and first.point[0]*following.point[0] > 0):
            entries[first.id] = following.point[0] - first.point[0]
        last_end = max(last_end, first.end)
    return entries


def _repeated_hold_phrases(contacts, groups, profile):
    """A, B, Ah followed by B at release leaves both prepared hands in place."""
    sustains = [c for c in contacts if c.kind in SUSTAIN_KINDS]
    starts = [c.start for c in sustains]
    ends = []
    for c in sustains:
        ends.append(max(ends[-1] if ends else -math.inf, c.end))
    for a,b,h,answer in zip(groups, groups[1:], groups[2:], groups[3:]):
        if any(len(g)!=1 for g in (a,b,h,answer)):
            continue
        first, second, hold, last = a[0], b[0], h[0], answer[0]
        if (first.kind == second.kind == last.kind == 'tap' and hold.kind == 'hold'
                and first.pad == hold.pad and second.pad == last.pad and first.pad != second.pad
                and abs(last.start-hold.end) <= RELEASE
                and 0. < second.start-first.start <= .16 and 0. < hold.start-second.start <= .16):
            offset = hold.point[0]-second.point[0]
            lo, hi = bisect_left(starts, first.start-1e-6), bisect_right(starts, last.start+1e-6)
            if ((lo and ends[lo-1]>first.start+1e-6) or hi-lo != 1 or abs(offset)<=profile.hand_span_mm/2):
                continue
            side = int(offset>0.)
            yield ((first.id,side), (second.id,1-side), (hold.id,side), (last.id,1-side))


def _spaced_strike_links(groups, *, follow_into_chords=False):
    """Recognize directional, separately struck keys, including a stride onto a Hold."""
    links = {}
    for a, b, landing in zip(groups, groups[1:], groups[2:]):
        if len(a) != 1 or len(b) != 1:
            continue
        first, middle = a[0], b[0]
        if first.kind != 'tap' or middle.kind != 'tap' or not all(c.is_button for c in landing):
            continue
        step = _ring_step(int(first.pad), int(middle.pad))
        gaps = middle.start-first.start, landing[0].start-middle.start
        if not (.12 <= min(gaps) and max(gaps) <= min(.25, 1.25*min(gaps))):
            continue
        if len(landing) != 1:
            # Allow the moving hand to take the near end of an upcoming EACH.
            # This only waives the alternation cost; either hand can still play it.
            if (follow_into_chords and len(landing) == 2 and abs(step) == 1
                    and first.point[0]*middle.point[0] > 0.
                    and {_ring_step(int(middle.pad), int(c.pad)) for c in landing} == {step, 2*step}):
                links[middle.id] = first.id
            continue
        last = landing[0]
        exit_step = _ring_step(int(middle.pad), int(last.pad))
        if abs(step) == 1 and exit_step == step and last.kind == 'tap':
            # At a moderate pulse each hand can walk adjacent buttons on its
            # own flank. Leave the seam free for the other hand to receive.
            for a, b in ((first, middle), (middle, last)):
                if a.point[0]*b.point[0] > 0.:
                    links[b.id] = a.id
            continue
        if not (abs(step) == 2 and exit_step == step) and not (
                last.kind == 'hold' and step * exit_step == 2):
            continue
        links[middle.id] = first.id
        links[last.id] = middle.id
    return links


def _prepared_slide_heads(contacts):
    """Link an ordinary launch to its own star; a branched head has no unique route."""
    heads = {(c.pad, round(c.start, 6)): c.id for c in contacts if c.is_star_head}
    launches = defaultdict(list)
    for c in contacts:
        marker = c.judge_marker
        if (c.kind == 'slide' and marker and not c.brush and c.source_id == c.id
                and abs(c.start-marker['slide_trace_second']) < 1e-6):
            launches[c.pad, round(marker['second'], 6)].append(c.id)
    return {ids[0]: heads[key] for key, ids in launches.items() if len(ids) == 1 and key in heads}


def _slide_transfer_cost(current, waiting, contact):
    approach = _sub(contact.point, current.last)
    extra = math.hypot(*approach) - math.dist(waiting.last, contact.point)
    if extra <= 0.:
        return 0.
    # How much of the incoming movement must be reversed at the start
    # of the trace? A tangential approach needs less braking than a U-turn.
    departure = _sub(contact.at(min(contact.end, contact.start+MIN_TRAVEL_SEC)), contact.point)
    reverse = max(0., -_dot(approach, departure)/max(1e-6, math.hypot(*departure)))
    available = max(MIN_TRAVEL_SEC, contact.start-current.ready)
    return (extra+reverse)/(1650.*available)


def _overlapping_star_launches(contacts, launch_heads):
    # The incoming star head lands while the preceding trace is moving. Keep
    # both routes as possible participants, including the last in the phrase.
    starts = [(round(c.judge_marker['second'], 6), c) for c in contacts if c.id in launch_heads]
    starts.sort(key=lambda item: item[0])
    times = [t for t,_ in starts]
    linked = set()
    for head_time, c in starts:
        lo = bisect_left(times, max(head_time+1e-6, round(c.start, 6)))
        hi = bisect_left(times, round(c.judge_marker['end_second'], 6))
        for _, following in starts[lo:hi]:
            if following.pad != c.pad:
                linked.update((c.id, following.id))
    return linked


def _ready_to_receive(current, contact, hand):
    """A spare hand on its own side can intercept the next rim button at normal travel speed."""
    if current.active or contact.point[0] * (1 if hand == 1 else -1) <= 0.:
        return False
    available = contact.start - current.ready
    return available >= .12 and math.dist(current.last, contact.point) <= 1650. * available


def _touch_arc_links(groups):
    """Recognize angular motion, including A and D alternating on the outer rim."""
    links = {}
    for a, b, landing in zip(groups, groups[1:], groups[2:]):
        if len(a) != 1 or len(b) != 1:
            continue
        first, middle = a[0], b[0]
        if (first.kind != 'touch' or middle.kind != 'touch' or len(first.pad) != 2
                or len(middle.pad) != 2):
            continue
        gaps = middle.start-first.start, landing[0].start-middle.start
        if min(gaps) < MIN_TRAVEL_SEC or max(gaps) > min(.20, min(gaps)*1.25):
            continue
        if first.pad[0] == middle.pad[0]:
            step = _ring_step(int(first.pad[1]), int(middle.pad[1]))
            if abs(step) != 1:
                continue
            target = first.pad[0] + str((int(middle.pad[1])-1+step) % 8 + 1)
        elif {first.pad[0], middle.pad[0]} == {'A', 'D'}:
            # D1 lies between A8 and A1. Two half steps return to the first ring.
            p = 2*(int(first.pad[1])-1) + (first.pad[0]=='A')
            q = 2*(int(middle.pad[1])-1) + (middle.pad[0]=='A')
            step = (q-p+8) % 16 - 8
            if abs(step) != 1:
                continue
            target = first.pad[0] + str(((q+step) % 16)//2+1)
        else:
            continue
        matching = [c for c in landing if c.kind=='touch' and c.pad == target]
        if len(matching) == 1:
            links[middle.id] = first.id
            links[matching[0].id] = middle.id
    return links


@dataclass(frozen=True, slots=True)
class TouchRoll:
    first: Contact
    last: Contact
    entry: Vec3
    exit: Vec3

    def point(self, second):
        u = min(1., max(0., (second-self.first.start)/(self.last.start-self.first.start)))
        return _add(self.entry, _scale(_sub(self.exit, self.entry), u))


def _touch_rolls(contacts, profile):
    """Two isolated, closely staggered Touches that fit under one open palm."""
    rolls = {}
    occupied_until = -math.inf
    window = 2*RELEASE
    for i, (a,b) in enumerate(zip(contacts, contacts[1:])):
        previous_end = occupied_until
        occupied_until = max(occupied_until, a.end)
        if (a.kind != 'touch' or b.kind != 'touch' or a.pad == b.pad or previous_end > a.start
                or not a.end <= b.start <= a.start+window
                or (i and a.start-contacts[i-1].start <= window)
                or (i+2 < len(contacts) and contacts[i+2].start-b.start <= window)):
            continue
        distance = math.dist(a.point,b.point)
        if not 0. < distance <= profile.hand_span_mm:
            continue
        # Use the same palm spread as a compact simultaneous Touch pair.
        # Only the residual wrist travel has to fit between the two onsets.
        offset = min(.5, (.28*profile.hand_span_mm+TOUCH_CONTACT_OFFSET_MM)/distance)
        delta = _scale(_sub(b.point,a.point),offset)
        roll = TouchRoll(a,b,_add(a.point,delta),_sub(b.point,delta))
        if math.dist(roll.point(a.end),a.point) <= _contact_span(a,profile):
            rolls[b.id] = roll
    return rolls


def _link_trace_exits(contacts, groups, links):
    tails = sorted((c for c in contacts if c.kind=='slide' and c.judge_marker and
                    abs(c.end-c.judge_marker['end_second']) < 1e-6), key=lambda c:c.end)
    ends = [c.end for c in tails]
    connected = set()
    for i, (a,b) in enumerate(zip(groups, groups[1:])):
        if len(a)!=1 or len(b)!=1:
            continue
        first, following = a[0], b[0]
        if first.id in links or links.get(following.id) != first.id or first.pad[0] not in 'AD':
            continue
        pulse = following.start-first.start
        earliest = max(first.start-min(.12, 2*pulse), groups[i-1][0].start if i else -math.inf)
        matches = []
        for tail in tails[bisect_left(ends,earliest):bisect_right(ends,first.start+1e-6)]:
            tip = tail.at(tail.end)
            exit_vector = _sub(tip, tail.at(max(tail.start, tail.end-pulse)))
            bridge = _sub(first.point, tip)
            onward = _sub(following.point, first.point)
            if (0. < math.hypot(*bridge) <= machine_profile().button_radius_mm
                    and _dot(exit_vector, bridge) > .65*math.hypot(*exit_vector)*math.hypot(*bridge)
                    and _dot(bridge, onward) > .65*math.hypot(*bridge)*math.hypot(*onward)
                    and math.hypot(*bridge) <= 2.5*math.hypot(*onward)):
                matches.append(tail.id)
        if len(matches)==1:
            links[first.id] = matches[0]
            connected.add(first.id)
    for group in groups:
        connected.update(c.id for c in group if links.get(c.id) in connected)
    return connected


def _crossed_hold_slide_cost(trace, hold, slide_hand, profile):
    # Both arms reaching across the center while one stays anchored. Integrate
    # the actual route so an internal cut or a distant endpoint cannot hide it.
    side = 1 if slide_hand == 0 else -1
    if side * hold.point[0] >= 0.:
        return 0.
    start, end = max(trace.start, hold.start), min(trace.end, hold.end)
    inverted = 0.
    for fraction in (.125, .375, .625, .875):
        x = trace.at(start+(end-start)*fraction)[0]
        if side*x > 0.:
            inverted += side*(x-hold.point[0])
    return .25 * inverted/profile.hand_span_mm * (end-start)/COHOLD_TRANSITION_SEC


@dataclass(frozen=True, slots=True)
class RepeatedSweepEdge:
    previous: int
    reference: tuple[int, int]
    finish: tuple[tuple[int, ...], tuple[int, ...]] | None = None


def _repeated_sweep_edges(contacts, groups):
    """Match a sweep from one EACH to the next with its translated repeat."""
    positions = {c.id: i for i, c in enumerate(contacts)}
    edges = {}
    previous = None
    sustain_starts, sustain_ends = [], []
    for c in contacts:
        if c.kind in SUSTAIN_KINDS:
            sustain_starts.append(c.start)
            sustain_ends.append(max(c.end, sustain_ends[-1] if sustain_ends else -math.inf))
    boundaries = [i for i, g in enumerate(groups) if len(g) != 1 or g[0].kind != 'tap']
    for lo, hi in zip(boundaries, boundaries[1:]):
        opening, closing = groups[lo], groups[hi]
        # Repeated short strokes. Multi-revolution spins retain the ordinary run model.
        if (not 3 <= hi-lo <= 8 or len(opening) != 2 or len(closing) != 2
                or not all(c.kind == 'tap' for c in opening) or not all(c.is_button for c in closing)):
            previous = None
            continue
        middle = [g[0] for g in groups[lo+1:hi]]
        direction = _ring_step(int(middle[0].pad), int(middle[1].pad))
        start_lane = str((int(middle[0].pad)-1-direction) % 8+1)
        end_lane = str((int(middle[-1].pad)-1+direction) % 8+1)
        first = next((c for c in opening if c.pad == start_lane), None)
        last = next((c for c in closing if c.pad == end_lane), None)
        if (not first or not last or abs(direction) != 1 or
                any(_ring_step(int(a.pad), int(b.pad)) != direction for a,b in zip(middle, middle[1:]))):
            previous = None
            continue
        run = [first, *middle, last]
        gaps = [b.start-a.start for a,b in zip(run, run[1:])]
        pulse = sum(gaps)/len(gaps)
        active = bisect_left(sustain_starts, last.start-1e-6) - 1
        if (min(gaps) < MIN_TRAVEL_SEC or max(gaps) > min(.13, min(gaps)*1.20)
                or (active >= 0 and sustain_ends[active] > first.start+1e-6)):
            previous = None
            continue
        if previous is not None:
            old, step, rate = previous
            if (len(run) == len(old) and step == direction and old[-1].id != first.id
                    and abs(rate-pulse) <= .15*pulse):
                for a,b,x,y in zip(old, old[1:], run, run[1:]):
                    # Distances into the search history. The two endpoints of
                    # an EACH may be visited in either order.
                    offset = positions[y.id]-1
                    finish = None
                    if y is last:
                        finish = (tuple(offset-positions[c.id] for c in old),
                                  tuple(offset-positions[c.id] for c in run[:-1]))
                    edges[y.id] = RepeatedSweepEdge(offset-positions[x.id],
                                                    (offset-positions[a.id], offset-positions[b.id]), finish)
        previous = run, direction, pulse
    return edges


def _past_hand(state, depth):
    for _ in range(depth):
        state = state.parent
    return state.choice


def _repeat_costs(state, edge):
    before = _past_hand(state, edge.previous)
    costs = [0., 0.]
    if before < 0:
        return costs
    a, b = (_past_hand(state, depth) for depth in edge.reference)
    if min(a,b) >= 0:
        costs[1-before if a == b else before] += 4.
    if edge.finish is not None:
        old, current = ({_past_hand(state, depth) for depth in depths} for depths in edge.finish)
        if -1 not in old and -1 not in current:
            for hand in (0, 1):
                if (len(old) == 1) != (len(current | {hand}) == 1):
                    # A different handoff point preserves a split approach;
                    # abandoning it for just the last group changes the whole gesture.
                    costs[hand] += 8.
    return costs


def _free_hand_can_reach(spare, contact, profile):
    if spare.active:
        return False
    last = spare.last_contact
    allowance = _contact_span(contact, profile)
    speed = COHOLD_PLACEMENT_MM_S
    available = max(0., contact.start-spare.ready)
    if last is not None:
        # Both contacts have extent; travelling from center to center is not required.
        allowance += _contact_span(last, profile)
        if contact.is_button and last.kind == 'tap' and last.in_spin_run and contact.start-last.start <= .13:
            step = _ring_step(_lane_number(last.pad), _lane_number(contact.pad))
            if 0 < step * spare.sweep_dir <= 2:
                speed = 3200.
                available = contact.start-last.start
    return math.dist(spare.last, contact.point) <= allowance + speed * available


def _independent_trace_pairs(contacts):
    """Identify two distinct authored routes with a common launch and no third duty."""
    launches = defaultdict(dict)
    cuts = defaultdict(dict)
    others = []
    for c in contacts:
        if c.kind == 'slide' and c.judge_marker and not c.brush:
            marker = c.judge_marker
            launches[round(marker['slide_trace_second'], 6)][c.source_id] = marker
            if c.id != c.source_id:
                cuts[c.source_id][round(c.start, 6)] = c.point
        else:
            others.append(c)
    starts = [c.start for c in others]
    ends = []
    for c in others:
        ends.append(max(ends[-1] if ends else -math.inf, c.end))
    times = sorted(launches)
    pairs = {}
    prior_end = -math.inf
    for i, start in enumerate(times):
        routes = launches[start]
        end = min(m['end_second'] for m in routes.values())
        clear = prior_end <= start+1e-6 and (i+1 == len(times) or times[i+1] >= end-1e-6)
        prior_end = max(prior_end, *(m['end_second'] for m in routes.values()))
        if not clear or len(routes) != 2:
            continue
        (a, ma), (b, mb) = routes.items()
        if ma['slide_segment_keys'] == mb['slide_segment_keys']:
            continue
        if ma['lane'] != mb['lane'] and ma['end_lane'] != mb['end_lane']:
            entry = _sub(button_xyz(ma['lane']), button_xyz(mb['lane']))
            exit_delta = _sub(button_xyz(ma['end_lane']), button_xyz(mb['end_lane']))
            # Opposing endpoints alone do not make a handoff. At a usable
            # interior cut both routes must also be nearby at the same time.
            if _dot(entry, exit_delta) < 0. and any(
                    math.dist(cuts[a][t], cuts[b][t]) <= SLIDE_MEETING_MM
                    for t in cuts[a].keys() & cuts[b].keys()):
                continue
        lo = bisect_left(starts, start-1e-6)
        hi = bisect_left(starts, end-1e-6)
        if lo and ends[lo-1] > start+1e-6:
            continue
        if any(c.kind != 'tap' or abs(c.start-start) > 1e-6 or int(c.pad) not in (ma['lane'], mb['lane'])
               for c in others[lo:hi]):
            continue
        duration = end - start
        if duration > 1e-6:
            pairs[a], pairs[b] = (b, duration), (a, duration)
    return pairs


def _unoccupied_sweep_steps(contacts, starts_by_id):
    """Only offer an elective relay if the phrase leaves both hands available."""
    sustain_starts, sustain_until = [], []
    roots, ends = {}, {}
    for c in contacts:
        if c.kind in SUSTAIN_KINDS:
            sustain_starts.append(c.start)
            sustain_until.append(max(c.end, sustain_until[-1] if sustain_until else -math.inf))
        if c.sweep_predecessor in starts_by_id:
            root = roots.get(c.sweep_predecessor, c.sweep_predecessor)
            roots[c.id] = root
            ends[root] = c.end
    available = set()
    for root, end in ends.items():
        i = bisect_right(sustain_starts, end + COHOLD_TRANSITION_SEC) - 1
        if i < 0 or sustain_until[i] < starts_by_id[root] - COHOLD_TRANSITION_SEC:
            available.add(root)
    return {cid for cid, root in roots.items() if root in available}


def _plan(contacts: list[Contact], profile: PlayerProfile, *, stance: str = 'adaptive',
          sweep_style: str = 'continuous', observe=None) -> list[int]:
    if sweep_style not in SWEEP_STYLES:
        raise ValueError('未知的扫键手顺偏好')
    if not contacts:
        return []
    first = min(0., contacts[0].start)
    beam = [Plan((HandState(profile.homes[0], first - .5), HandState(profile.homes[1], first - .5)))]
    sweep_predecessors = {c.sweep_predecessor for c in contacts if c.sweep_predecessor is not None}
    seconds_by_id = {c.id: c.start for c in contacts}
    mates = _independent_trace_pairs(contacts)
    launch_heads = _prepared_slide_heads(contacts)
    trace_reach = {}
    for c in contacts:
        if c.segments:
            points = [part.point(part.start+(part.end-part.start)*fraction/8)
                      for part in c.segments for fraction in range(9)]
            trace_reach[c.id] = tuple(max(math.dist(shoulder, p) for p in points)
                                      for shoulder in profile.shoulders)
    relay_steps = _unoccupied_sweep_steps(contacts, seconds_by_id) if sweep_style == 'relay' else set()
    fast_pickups = {c.sweep_predecessor for c in contacts if c.sweep_predecessor in seconds_by_id
                    and c.start-seconds_by_id[c.sweep_predecessor] <= .078}
    hold_entries = _hold_pair_entries(contacts)
    onsets = defaultdict(list)
    for c in contacts:
        if not c.segments:
            onsets[c.start].append(c)
    groups = list(onsets.values())
    hold_anchors = dict(pair for phrase in _repeated_hold_phrases(contacts, groups, profile) for pair in phrase)
    linked_launches = _overlapping_star_launches(contacts, launch_heads) if sweep_style == 'continuous' else set()
    touch_links = _touch_arc_links(groups)
    exit_sweeps = _link_trace_exits(contacts, groups, touch_links)
    touch_rolls = _touch_rolls(contacts, profile)
    repeated_edges = _repeated_sweep_edges(contacts, groups)
    stacked_exits = _stacked_chord_exits(groups)
    pair_predecessors = _translating_pairs(groups)
    spaced_links = _spaced_strike_links(groups, follow_into_chords=sweep_style == 'continuous')
    # Alternation is evidence for individual strikes. It does not depend on
    # how many hands will eventually cover a simultaneous set of Touch zones.
    chord_onsets = {t for t, group in onsets.items() if sum(c.is_button for c in group) >= 2}
    compact_touches = {}
    for t, group in onsets.items():
        touches = [c for c in group if c.kind=='touch']
        if len(touches)<2:
            continue
        for cluster in touch_groups([c.pad for c in touches]):
            members = tuple(touches[i] for i in cluster)
            if len(members)>1 and _fits_palm(members,t,profile):
                compact_touches.update((c.id, members[0].id) for c in members)
                chord_onsets.add(t)
        if len(touches)==len(group)==2 and not _fits_palm(tuple(touches),t,profile):
            chord_onsets.add(t)
    group_remaining = Counter(c.touch_group for c in contacts if c.touch_group is not None)
    static_max = profile.upper_arm_mm + profile.forearm_mm
    assisted_max = static_max + TORSO_ASSIST_MARGIN_MM
    sweep_stances = _sweep_stances(profile) if stance == 'adaptive' and any(
        contact.sweep_predecessor is not None for contact in contacts
    ) else None

    for contact in contacts:
        candidates: list[Plan] = []
        hold_slide_costs = {}
        touch_previous = touch_links.get(contact.id)
        roll = touch_rolls.get(contact.id)
        repeat = repeated_edges.get(contact.id)
        group = contact.touch_group
        if group is not None:
            group_remaining[group] -= 1
        remaining = group_remaining[group]

        def extend(parent: Plan, hands, hand: int = -1, effort: float = 0.,
                   is_sweep: bool = False, rapid_jump: bool = False) -> Plan:
            compromises = parent.chord_compromises
            if hand >= 0 and contact.is_button:
                prior = hands[hand].last_contact
                prepared = hold_anchors.get(contact.id)
                spare = hands[1-hand]
                if (prepared is not None and prepared != hand and not spare.active
                        and _free_hand_can_reach(spare, contact, profile)):
                    # Compare the two roles from their entry into the phrase:
                    # tap and hold the anchor, answer on the other button.
                    compromises += 1
                if (contact.kind == 'hold' and len(hands[hand].active)==1 and prior is not None and prior.kind == 'hold'
                        and prior.pad != contact.pad and prior.end <= contact.start+1e-6):
                    # Ordinary Hold placement lasts through release. If there is
                    # no time to move from one key area to the next, reaching
                    # both with spread fingers is a fallback, as for overlapping Holds.
                    travel = math.dist(prior.point, contact.point) - 2*BUTTON_CONTACT_OFFSET_MM
                    gap = max(0., contact.start-prior.end)
                    if travel > COHOLD_PLACEMENT_MM_S*gap and _free_hand_can_reach(hands[1-hand], contact, profile):
                        compromises += 1
                for companion in hands[hand].active:
                    if companion.id == contact.id or not companion.is_button:
                        continue
                    if abs(companion.start-contact.start) < 1e-6:
                        compromises += 1
                        break
                    if companion.kind == 'hold' and (contact.kind == 'tap'
                            or min(companion.end, contact.end)-contact.start > RELEASE):
                        spare = hands[1-hand]
                        # A newly released hand across the ring cannot already
                        # be at this key. Sharing a palm is legitimate there.
                        if _free_hand_can_reach(spare, contact, profile):
                            compromises += 1
                            break
            if hand >= 0 and contact.source_id in mates:
                mate, duration = mates[contact.source_id]
                for prior in hands[hand].active:
                    if prior.source_id == mate and _free_hand_can_reach(hands[1-hand], contact, profile):
                        # Handoff cuts are internal. Count the shared fraction of
                        # the authored pair, not the number of geometry fragments.
                        compromises += (min(prior.end, contact.end)-contact.start) / duration
                        break
            if hand >= 0 and not contact.segments:
                same_onset = abs(contact.start - parent.last_strike_sec) <= 1e-4
                if same_onset:
                    strike_h = parent.last_strike_hand
                    streak = 0
                elif hand == parent.last_strike_hand and not is_sweep and parent.last_strike_sec not in chord_onsets:
                    strike_h = hand
                    streak = parent.same_hand_streak + 1
                else:
                    strike_h = hand
                    streak = 0
                strike_t = contact.start
            else:
                strike_h = parent.last_strike_hand
                strike_t = parent.last_strike_sec
                streak = parent.same_hand_streak

            jumps = parent.rapid_jumps + rapid_jump
            if group is None:
                return Plan(hands, parent.misses + int(hand < 0), parent.effort + effort,
                            parent, hand, 0, strike_h, strike_t, streak, compromises, jumps)
            hits = parent.group_hits + int(hand >= 0)
            if remaining:
                return Plan(hands, parent.misses, parent.effort + effort,
                            parent, hand, hits, strike_h, strike_t, streak, compromises, jumps)
            return Plan(hands, parent.misses + max(0, contact.group_required - hits), parent.effort + effort,
                        parent, hand, 0, strike_h, strike_t, streak, compromises, jumps)

        # Different beam histories can share the same immutable hand state and
        # active contacts. Both lookups are valid only for this contact time.
        advanced = {}
        geometry = {}
        for state in beam:
            repeat_cost = _repeat_costs(state, repeat) if repeat else (0., 0.)
            for prior in state.hands:
                if id(prior) not in advanced:
                    advanced[id(prior)] = prior.advance(contact.start) if prior.active else prior
            hands = (advanced[id(state.hands[0])], advanced[id(state.hands[1])])
            candidates.append(extend(state, hands))
            for hand, current in enumerate(hands):
                other = hands[1 - hand]
                if not contact.segments and any(not c.segments and c.pad == contact.pad for c in other.active):
                    continue
                occupancy = tuple(c.id for c in current.active)
                if occupancy not in geometry:
                    if _shared_motion_fits(current.active, contact, profile):
                        active = (*current.active, contact)
                        geometry[occupancy] = (active, _center(active, contact.start))
                    else:
                        geometry[occupancy] = None
                if geometry[occupancy] is None:
                    continue
                active, target = geometry[occupancy]
                reach = math.dist(profile.shoulders[hand], target)
                # For Slide segments, ensure the destination of the segment is also reachable!
                if contact.segments:
                    end_reach = math.dist(profile.shoulders[hand], contact.at(contact.end))
                    reach = max(reach, end_reach)
                if reach > assisted_max:
                    continue
                if contact.segments:
                    # The body may assist a long reach. Account for the actual
                    # arc between the endpoints when comparing the two hands.
                    reach = max(reach, trace_reach[contact.id][hand])

                is_sweep, next_sweep_dir = _is_sweep_continuation(current, contact)
                rolling = (roll is not None and not current.active and current.last_contact is not None
                           and current.last_contact.id == roll.first.id)
                if rolling:
                    is_sweep, next_sweep_dir = True, 0
                if touch_previous is not None and not current.active:
                    is_sweep = current.last_contact is not None and current.last_contact.id == touch_previous
                    next_sweep_dir = 0
                sweep_body_x = 0.
                enters_sweep = contact.id in fast_pickups and contact.sweep_predecessor is None
                prepared_entry = enters_sweep and not current.active and contact.start-current.ready >= .20
                if sweep_stances is not None and not other.active and (prepared_entry or (
                        is_sweep and contact.sweep_predecessor is not None)):
                    # With the other hand free, the same stance can support
                    # a full circle. The frame solver still checks joint limits
                    # and collisions along the actual moving trajectory.
                    sweep_body_x, sweep_shoulder = sweep_stances[hand]
                    reach = min(reach, math.dist(sweep_shoulder, target))
                entry_span = hold_entries.get(contact.id, 0.)
                clearance_cost = max(0., entry_span if hand == 1 else -entry_span) / (.20 * profile.hand_span_mm)
                for partner in other.active:
                    if (contact.kind, partner.kind) in (('hold','slide'), ('slide','hold')):
                        key = (partner.id, hand)
                        if key not in hold_slide_costs:
                            args = (contact,partner,hand) if contact.segments else (partner,contact,1-hand)
                            hold_slide_costs[key] = _crossed_hold_slide_cost(*args, profile)
                        clearance_cost += hold_slide_costs[key]
                expected = pair_predecessors.get(contact.id)
                if expected is not None and other.last_contact is not None and other.last_contact.id == expected:
                    # Swapping roles makes one hand stop on the shared button
                    # and the other skip a button, although both could move one step.
                    clearance_cost += 2.5
                travel = 0.
                alt_penalty = 0.
                sweep_bonus = 0.
                sweep_handoff = 0.
                if (roll and not current.active and not other.active and other.last_contact is not None
                        and roll.first.id == other.last_contact.id):
                    # Deploying a second palm is cheap when it was already
                    # in position; otherwise the first can reach both zones.
                    sweep_handoff = min(1., math.dist(current.last,target)/profile.hand_span_mm)
                if (contact.id in exit_sweeps and not other.active and other.last_contact
                        and other.last_contact.id == touch_previous):
                    prior = other.last_contact
                    elapsed = contact.start-(prior.end if prior.segments else prior.start)
                    speed = math.dist(other.last, contact.point)/max(MIN_TRAVEL_SEC, elapsed)
                    sweep_handoff = 2.*speed/3200.
                rapid_jump = False
                if not current.active:
                    since_strike = contact.start - state.last_strike_sec
                    seconds = max(MIN_TRAVEL_SEC, contact.start - current.ready)
                    dist = math.dist(current.last, target)
                    rapid_jump = (current.last_contact is not None and not contact.segments
                                  and not is_sweep and _requires_rapid_jump(
                                      dist, _reposition_interval(current.last_contact, contact.start)))
                    if current.last_contact and contact.id in sweep_predecessors and current.last_contact.kind == 'hold':
                        # The holding hand has no approach time before release.
                        remaining_travel = max(0., dist - profile.hand_span_mm * .6)
                        clearance_cost += max(0., remaining_travel / 1800. - max(0., contact.start-current.ready)) / .06
                    if current.last_contact and other.last_contact and contact.is_button:
                        previous, neighbor = current.last_contact, other.last_contact
                        if (contact.kind == neighbor.kind == 'tap' and contact.is_star_head
                                and neighbor.start == state.last_strike_sec and not other.active
                                and abs(_ring_step(int(neighbor.pad), int(contact.pad))) == 1
                                and contact.point[0]*neighbor.point[0] < 0.
                                and not is_sweep and not contact.in_spin_run):
                            # Rapid independent strikes straddling a seam can
                            # still cross both arms even though the earlier key
                            # has just released. Let this cost fade with time.
                            inverted = (contact.point[0]-neighbor.point[0]) * (1 if hand==0 else -1)
                            clearance_cost += (max(0., inverted)/profile.hand_span_mm
                                               * max(0., 1.-since_strike/.28))
                        if (previous.id in stacked_exits and neighbor.id in stacked_exits and
                                abs(previous.start-neighbor.start) < 1e-5 and not other.active):
                            move = _ring_step(_lane_number(previous.pad), _lane_number(contact.pad))
                            between = _ring_step(_lane_number(previous.pad), _lane_number(neighbor.pad))
                            if 0 < abs(between) < abs(move) < 4 and between * move > 0:
                                clearance_cost += 4. * max(0., 1.-(contact.start-previous.end)/.35)
                    if _is_relay(current.last_contact, contact):
                        relay = _relay_motion(current.last_contact, contact, profile, hand)
                        if relay is None:
                            continue
                        seconds = relay.end - relay.start
                    if is_sweep:
                        if rolling:
                            dist = math.dist(roll.entry, roll.exit)
                            seconds = contact.start-roll.first.start
                        if contact.sweep_predecessor is not None:
                            # The palm can already move along the rim while the
                            # previous button remains inside its contact area.
                            seconds = contact.start - current.last_contact.start
                        travel = dist / (3200. * min(seconds, .55))
                        sweep_bonus = -.24 if current.sweep_dir == next_sweep_dir and next_sweep_dir != 0 else -.14
                    else:
                        travel = dist / (1650. * min(seconds, .55))
                    # Prefer alternation on rapid separate strikes; continuous sweeps
                    # already have their own travel cost and direction evidence.
                    directional_strike = (current.last_contact is not None
                                          and spaced_links.get(contact.id) == current.last_contact.id)
                    if (directional_strike and current.last_contact.id in spaced_links
                            and state.last_strike_hand == hand and state.same_hand_streak >= 1):
                        # The previous two separate strikes already established
                        # this velocity. The palm still lifts over the skipped key.
                        travel = dist / (2450. * min(seconds, .55))
                    # Alternation compares successive individual strikes. All
                    # buttons of an EACH share one onset, in either text order.
                    if (not is_sweep and not directional_strike and not contact.segments
                            and contact.start not in chord_onsets and state.last_strike_sec not in chord_onsets
                            and state.last_strike_hand == hand
                            and 1e-4 < since_strike < .28 and not other.active):
                        same_pad = (contact.pad == current.last_pad)
                        # Detect zig-zag pattern: high-speed, non-spin, large step distance
                        dist = math.dist(current.last, target)
                        # Thresholds for 'high-speed zig-zag' that should NOT be single-handed
                        is_high_speed_zigzag = (since_strike < 0.16 and dist > 180.)

                        if same_pad:
                            # Slow 8th jacks (>=0.18s) stay comfortably on one hand;
                            # fast jacks (<0.14s) prefer alternating two-hand split!
                            if since_strike < .145:
                                alt_penalty = 1.15 * (1. - since_strike / .145) * (1. + .6 * state.same_hand_streak)
                        elif is_high_speed_zigzag:
                            alt_penalty = 2.5
                        else:
                            urgency = (1. - since_strike / .28)
                            jump_factor = 1. + min(1.2, dist / 280.)
                            streak_mult = 1. + .85 * state.same_hand_streak
                            alt_penalty = 1.45 * urgency * jump_factor * streak_mult

                if contact.sweep_predecessor is not None and _is_sweep_continuation(other, contact)[0]:
                    # A handoff asks one arm to brake and the other to start
                    # an already moving sweep. Charge both changes in speed.
                    speed = math.dist(other.last_contact.point, contact.point) / (
                        contact.start - other.last_contact.start
                    )
                    sweep_handoff = 2. * speed / 3200.
                    if contact.id in relay_steps and _ready_to_receive(current, contact, hand):
                        sweep_handoff *= .15
                if (contact.id in relay_steps and is_sweep
                        and _ready_to_receive(other, contact, 1-hand)):
                    # Retaining the sweep is still legal. Favour a prepared
                    # reception as the moving hand travels deeper across the body.
                    clearance_cost += abs(contact.point[0]) / profile.hand_span_mm

                # Slide launch & mid-slide chain continuity:
                slide_bonus = 0.
                if contact.segments:
                    # During alternating stars, the hand still waiting at its own
                    # launch needs no new approach. Transferring the launch to
                    # the hand coming off another trace requires a reposition.
                    if (other.last_contact is not None and other.last_contact.id == launch_heads.get(contact.id)
                            and current.last_contact is not None
                            and (current.last_contact.segments or contact.id in linked_launches)
                            and not current.active and all(c.start >= contact.start-1e-6 for c in other.active)):
                        transfer = _slide_transfer_cost(current, other, contact)
                        # Continuity style prefers keeping the prepared hand
                        # when a transfer would also exchange roles in the phrase.
                        slide_bonus += transfer
                        if transfer > 0. and contact.id in linked_launches:
                            extra = math.dist(current.last, contact.point) - math.dist(other.last, contact.point)
                            slide_bonus += min(2., extra/profile.hand_span_mm)
                    if (contact.slide_chain is not None and current.last_chain != contact.slide_chain
                            and (other.last_chain == contact.slide_chain
                                 or any(c.source_id == contact.slide_chain for c in other.active))):
                        slide_bonus += .25
                        if contact.id != contact.source_id and contact.source_id in linked_launches:
                            dt = min(MIN_TRAVEL_SEC, contact.end-contact.start)
                            speed = math.dist(contact.point, contact.at(contact.start+dt))/dt
                            slide_bonus += 2.*speed/3200.
                    if contact.slide_chain is not None and current.last_chain == contact.slide_chain:
                        # Slight continuity preference on a single slide, easily overridden when
                        # swapping at an X-intersection avoids crossing into the opposite hemisphere!
                        slide_bonus -= .08

                shared_slide = bool(current.active) and contact.kind == 'slide' and all(
                    c.kind == 'slide' for c in current.active)
                cohold_cost = (
                    .18 * len(current.active) if shared_slide else
                    (1.65 if not other.active else .65) * len(current.active)
                )
                cluster = compact_touches.get(contact.id)
                if cluster is not None and current.active and all(compact_touches.get(c.id)==cluster for c in current.active):
                    # The same placement can reach several distinct sensors.
                    # Its effort grows with the required spread, irrespective
                    # of whether the game's group-clear threshold is discounted.
                    radius = max(math.dist(target, c.point) for c in active)
                    cohold_cost = (radius / _contact_span(contact, profile)) ** 2
                elif contact.touch_group is not None and any(c.touch_group==contact.touch_group for c in current.active):
                    cohold_cost = .35 * len(current.active)
                if current.active and contact.is_button and any(c.is_button for c in current.active):
                    cohold_cost = 2.5 * len(current.active)
                if len(current.active) == 1 and _shared_launch(current.active[0], contact):
                    cohold_cost = 0.

                # Evaluate hemisphere bias using representative_x (includes Slide destination!).
                # Mild crossing within the center corridor (|x| <= 130mm, e.g. borrowing hand to hit 4/5/1/8)
                # is effortless in maimai DX; deep outer crossing (e.g. L->2/3 or R->6/7) costs much more!
                rep_x = contact.representative_x if len(active) == 1 else target[0]
                over_mid = max(0., (rep_x - sweep_body_x) * (1 if hand == 0 else -1))
                crossing = (
                    .14 * (min(over_mid, 135.) / 135.)
                    + .62 * (max(0., over_mid - 135.) / 200.) ** 1.5
                )

                # Topological uncrossed order ONLY against genuinely concurrent actions (chords / active holds),
                # never against an already-released alternating hand from the previous 16th beat!
                topo_cross = 0.
                other_pos = None
                if other.active:
                    other_pos = (
                        other.active[0].representative_x
                        if len(other.active) == 1 and other.active[0].segments
                        else _center(other.active, contact.start)[0]
                    )
                    if contact.is_button and len(other.active) == 1 and other.active[0].is_button:
                        neighbor = other.active[0]
                        if abs(neighbor.start-contact.start) < 1e-5:
                            p, q = contact.point, neighbor.point
                            # A vertically stacked pair on one flank leaves the near
                            # shoulder above the reaching arm. Prefer this orientation
                            # softly; preceding motion and sustained notes can outweigh it.
                            if p[0]*q[0] > 0 and abs(p[0]-q[0]) < .5*abs(p[1]-q[1]):
                                near_hand = int(p[0] > 0)
                                inverted = (q[1]-p[1]) if hand == near_hand else (p[1]-q[1])
                                # At a prepared chord one hand can already be at
                                # its button while the other approaches its own.
                                # Apply the neutral orientation only to the extent
                                # that the incoming positions leave it undecided.
                                advantage = 0.
                                if (current.last_contact and other.last_contact and not current.active
                                        and contact.start-min(current.ready, other.ready) <= .35):
                                    advantage = max(0., min(math.dist(current.last,q)-math.dist(current.last,p),
                                                            math.dist(other.last,p)-math.dist(other.last,q)))
                                weight = max(0., 1.-advantage/profile.hand_span_mm)
                                clearance_cost += weight * max(0., inverted) / 100. * min(1., abs(p[0])/160.)
                elif abs(contact.start - state.last_strike_sec) <= 1e-4 and state.last_strike_hand == 1 - hand:
                    other_pos = other.last[0]
                if other_pos is not None:
                    left_x = rep_x if hand == 0 else other_pos
                    right_x = other_pos if hand == 0 else rep_x
                    if left_x > right_x + 25.:
                        topo_cross = .75 * (left_x - right_x - 25.) / 300.
                        if contact.kind == 'hold':
                            # A crossed pair held for a whole interval costs
                            # more than a momentary crossing in a rotating EACH.
                            remaining = max((min(c.end, contact.end)-contact.start
                                             for c in other.active if c.kind == 'hold'), default=0.)
                            topo_cross += (left_x-right_x)/profile.hand_span_mm * remaining/COHOLD_TRANSITION_SEC

                torso_assist_pen = 6.0 * ((reach - static_max) / TORSO_ASSIST_MARGIN_MM) ** 2 if reach > static_max else 0.
                effort = (
                    travel
                    + clearance_cost + repeat_cost[hand]
                    + alt_penalty
                    + sweep_bonus
                    + sweep_handoff
                    + slide_bonus
                    + crossing
                    + topo_cross
                    + cohold_cost
                    + .18 * (min(reach, static_max) / static_max) ** 4
                    + torso_assist_pen
                )
                next_hands = list(hands)
                next_hands[hand] = HandState(
                    current.last, current.ready, active,
                    contact.pad, next_sweep_dir, contact.slide_chain, current.last_contact,
                )
                candidates.append(extend(state, tuple(next_hands), hand, effort,
                                         is_sweep=is_sweep, rapid_jump=rapid_jump))

        candidates.sort(key=lambda state: (
            state.misses + (max(0, contact.group_required - state.group_hits - remaining) if remaining else 0),
            state.chord_compromises,
            state.rapid_jumps,
            state.effort,
        ))
        beam = []
        seen = set()
        for state in candidates:
            signature = (
                state.group_hits,
                state.last_strike_hand,
                state.last_strike_sec,
                min(state.same_hand_streak, 2),
                tuple((h.last, h.ready, h.last_pad, h.sweep_dir,
                       h.last_chain, h.last_contact.id if h.last_contact else None,
                       tuple(c.id for c in h.active)) for h in state.hands),
            )
            if signature in seen:
                continue
            seen.add(signature)
            beam.append(state)
            if len(beam) == BEAM_WIDTH:
                break
        if observe is not None:
            observe(contact, candidates, beam, remaining)
    result = []
    current = beam[0]
    while current.parent is not None:
        result.append(current.choice)
        current = current.parent
    return list(reversed(result))


def _review_assignments(contacts, hands, issues) -> tuple[int, int]:
    groups: dict[int, list[tuple[Contact, int]]] = defaultdict(list)
    required = unassigned = 0
    prev_by_hand: list[Contact | None] = [None, None]
    sources: dict[int, bool] = {}

    for contact, hand in zip(contacts, hands):
        if hand in (0, 1):
            prev = prev_by_hand[hand]
            # Ergonomic check: ballistic single-hand jump overload (excluding sweeps & relay continuations)
            if prev is not None and not contact.segments and contact.slide_chain is None:
                dt = _reposition_interval(prev, contact.start)
                if contact.start >= prev.end-1e-7 and dt < .15:
                    p0 = prev.at(prev.end)
                    p1 = contact.at(contact.start)
                    raw_dist = math.dist(p0, p1)
                    if _requires_rapid_jump(raw_dist, dt):
                        side_name = '左手' if hand == 0 else '右手'
                        issues.append(_issue(
                            'ballistic_overload', contact.start,
                            f'{side_name}需在 {round(dt * 1000)} ms 内跨区急跳 {round(raw_dist / 10)} cm，可复核分边或拆解',
                            hand=side_name, pad=contact.pad,
                        ))
            prev_by_hand[hand] = contact

        if contact.touch_group is not None:
            groups[contact.touch_group].append((contact, hand))
            continue
        sources[contact.source_id] = sources.get(contact.source_id, False) or hand < 0
        if hand < 0:
            issues.append(_issue('unassigned', contact.start, '此处未找到双手分配',
                                 pad=contact.pad, action=contact.kind, end_sec=contact.end,
                                 contact_id=contact.source_id))
    required += len(sources)
    unassigned += sum(sources.values())
    for group in groups.values():
        first = group[0][0]
        hit = sum(hand >= 0 for _, hand in group)
        required += first.group_required
        if hit < first.group_required:
            unassigned += first.group_required - hit
            issues.append(_issue('touch_group_shortfall', first.start,
                                 f'这组触屏需覆盖 {first.group_required} 个触区，模拟找到了 {hit} 个',
                                 pads=[contact.pad for contact, _ in group]))
    return required, unassigned


@dataclass(frozen=True, slots=True)
class CoholdShift:
    start: float
    end: float
    contact: Contact
    anchors: frozenset[int]

    def weight(self, second: float) -> float:
        if second < self.contact.start:
            return _smooth_fraction((second - self.start) / (self.contact.start - self.start))
        if second >= self.contact.end and self.end > self.contact.end:
            return _smooth_fraction((self.end - second) / (self.end - self.contact.end))
        return 0.


def _cohold_shifts(contacts: list[Contact]) -> list[CoholdShift]:
    """Prepare additions and releases during both nested and partial overlap."""
    sustained: list[Contact] = []
    prepare = defaultdict(list)
    recover = defaultdict(list)
    by_id = {c.id: c for c in contacts}
    for c in contacts:
        sustained = [anchor for anchor in sustained if anchor.end > c.start]
        for anchor in sustained:
            if anchor.start < c.start:
                prepare[c.id].append(anchor)
            if anchor.end > c.end:
                recover[c.id].append(anchor)
            elif anchor.end < c.end and c.kind in SUSTAIN_KINDS:
                recover[anchor.id].append(c)
        if c.kind in SUSTAIN_KINDS:
            sustained.append(c)
    shifts = []
    for cid, anchors in prepare.items():
        c = by_id[cid]
        start = max(c.start - COHOLD_TRANSITION_SEC, min(a.start for a in anchors))
        shifts.append(CoholdShift(start, c.end, c, frozenset(a.id for a in anchors)))
    for cid, anchors in recover.items():
        c = by_id[cid]
        end = min(c.end + COHOLD_TRANSITION_SEC, max(a.end for a in anchors))
        shifts.append(CoholdShift(c.start, end, c, frozenset(a.id for a in anchors)))
    return sorted(shifts, key=lambda shift: shift.start)


def _cohold_placement(active, nearby_shifts, second, profile):
    points = [contact.at(second) for contact in active]
    center = tuple(sum(axis) / len(points) for axis in zip(*points))
    if not nearby_shifts:
        return center
    active_ids = {contact.id for contact in active}
    adjustments = [(shift.contact.at(second), shift.weight(second)) for shift in nearby_shifts
                   if shift.contact.id not in active_ids and active_ids & shift.anchors]
    weight = sum(value for _, value in adjustments)
    if weight <= 0.:
        return center
    desired = tuple((center[i] * len(points) + sum(point[i] * value for point, value in adjustments))
                    / (len(points) + weight) for i in range(3))
    return _covered_center(center, desired, active, second, profile)


def _covered_center(center, desired, active, second, profile):
    """Move toward a preferred palm position within all current contact areas."""
    points = [contact.at(second) for contact in active]
    delta = tuple(b - a for a, b in zip(center, desired))
    length_sq = sum(d * d for d in delta)
    if length_sq <= 1e-10:
        return center
    fraction = 1.
    for contact, point in zip(active, points):
        radius = _contact_span(contact, profile)
        relative = tuple(a - b for a, b in zip(center, point))
        slack = radius * radius - sum(d * d for d in relative)
        if slack < -1e-6:
            return center
        along = sum(a * b for a, b in zip(relative, delta))
        fraction = min(fraction, max(0., (math.sqrt(along * along + length_sq * max(0., slack)) - along) / length_sq))
    return tuple(p + fraction * d for p, d in zip(center, delta))


def _merge_assigned_tracks(contacts: list[Contact]) -> list[Contact]:
    """Join same-hand pieces so optional handoff points do not create extra gestures."""
    result = []
    latest = {}
    for contact in contacts:
        index = latest.get(contact.source_id)
        previous = result[index] if index is not None else None
        if (contact.segments and previous is not None and previous.segments
                and abs(previous.end - contact.start) < 1e-6):
            result[index] = replace(previous, end=contact.end,
                                    segments=previous.segments + contact.segments)
        else:
            latest[contact.source_id] = len(result)
            result.append(contact)
    return sorted(result, key=lambda c: (c.start, -c.end, c.id))


class HandPlayback:
    """Synthesize continuous human-like striking, sweeping, and early-hover trajectories."""

    def __init__(self, contacts: list[Contact], home: Vec3, profile: PlayerProfile, hand: int = 0, *, rolls=None):
        self.rolls = rolls or {}
        contacts = _merge_assigned_tracks(contacts)
        self.contacts = contacts
        self.index = 0
        self.state = HandState(home, min(-.5, contacts[0].start - .5) if contacts else -.5)
        self.home = home
        self.profile = profile
        self.shifts = _cohold_shifts(contacts)
        self.shift_index = 0
        self.nearby_shifts: list[CoholdShift] = []
        self.last_hold_ids: tuple[int, ...] = ()
        self.last_placement = home
        self.last_sample = 0.
        self.last_ended_kind = 'idle'
        self.last_ended_pad = ''
        self.relays = [motion for before, after in zip(contacts, contacts[1:])
                       if _is_relay(before, after)
                       if (motion := _relay_motion(before, after, profile, hand)) is not None]
        self.relay_index = 0

    def sample(self, second: float) -> tuple[Vec3, str, tuple[Contact, ...], float]:
        prev_active = self.state.active
        state = self.state.advance(second)
        if len(state.active) < len(prev_active):
            released = [c for c in prev_active if c not in state.active]
            if released:
                latest = max(released, key=lambda c: c.end)
                self.last_ended_kind = latest.kind
                self.last_ended_pad = latest.pad
                if latest.id in self.rolls:
                    state = replace(state, last=self.rolls[latest.id].point(latest.end))
        while self.index < len(self.contacts) and self.contacts[self.index].start <= second:
            contact = self.contacts[self.index]
            self.index += 1
            if contact.end > second:
                state = HandState(state.last, state.ready, (*state.active, contact), contact.pad)
            elif contact.end > state.ready:
                self.last_ended_kind = contact.kind
                self.last_ended_pad = contact.pad
                state = HandState(contact.at(contact.end), contact.end, state.active, contact.pad)
        self.state = state
        while self.shift_index < len(self.shifts) and self.shifts[self.shift_index].start <= second:
            self.nearby_shifts.append(self.shifts[self.shift_index])
            self.shift_index += 1
        if self.nearby_shifts:
            self.nearby_shifts = [shift for shift in self.nearby_shifts if shift.end > second]
        if state.active:
            mode = max((c.kind for c in state.active), key=CONTACT_PRIORITY.index)
            point = _cohold_placement(state.active, self.nearby_shifts, second, self.profile)
            if len(state.active)==1 and state.active[0].id in self.rolls:
                point = self.rolls[state.active[0].id].point(second)
            while self.relay_index < len(self.relays) and self.relays[self.relay_index].end <= second:
                self.relay_index += 1
            if self.relay_index < len(self.relays):
                relay = self.relays[self.relay_index]
                if relay.start <= second < relay.end and len(state.active) == 1:
                    point = relay.point(second)
            holds = tuple(c.id for c in state.active if c.kind in STATIC_HOLD_KINDS) if mode not in ('slide', 'wifi') else ()
            if holds and holds == self.last_hold_ids:
                distance = math.dist(self.last_placement, point)
                limit = COHOLD_PLACEMENT_MM_S * max(0., second - self.last_sample)
                if distance > limit:
                    fraction = limit / distance
                    point = tuple(a + fraction * (b - a) for a, b in zip(self.last_placement, point))
            self.last_hold_ids, self.last_placement, self.last_sample = holds, point, second
            return point, mode, state.active, 0.

        self.last_hold_ids = ()
        upcoming = self.contacts[self.index] if self.index < len(self.contacts) else None
        if upcoming is None:
            f = (second - state.ready) / .38
            return free_motion(state.last, self.home, f, 18., panel_normal=_PANEL_NORMAL), 'idle' if f >= 1 else 'travel', (), _smooth_fraction(f)

        last = self.index + 1
        while last < len(self.contacts) and self.contacts[last].start == upcoming.start:
            last += 1
        end = _center(tuple(self.contacts[self.index:last]), upcoming.start)
        if upcoming.id in self.rolls:
            roll = self.rolls[upcoming.id]
            if second >= roll.first.start:
                return roll.point(second), 'travel', (), 0.
            end = roll.entry
        gap = upcoming.start - state.ready

        if gap > 1.35:
            if second < state.ready + .38:
                f = (second - state.ready) / .38
                return free_motion(state.last, self.home, f, 18., panel_normal=_PANEL_NORMAL), 'travel', (), _smooth_fraction(f)
            if second < upcoming.start - .42:
                return self.home, 'idle', (), 1.
            prep_start = upcoming.start - .42
            hover = _add(end, _scale(_PANEL_NORMAL, 26.))
            strike_start = upcoming.start - .085
            if second < strike_start:
                f = (second - prep_start) / max(1e-6, strike_start - prep_start)
                return free_motion(self.home, hover, f, 14., panel_normal=_PANEL_NORMAL), 'travel', (), 1. - _smooth_fraction(f)
            f = (second - strike_start) / .085
            p = _smooth_fraction(f)
            return _add(hover, _scale(_sub(end, hover), p)), 'travel', (), 0.

        dist = math.dist(state.last, end)
        if gap <= .045 and dist <= 75. and (self.last_ended_kind in ('slide', 'wifi') or upcoming.kind in ('slide', 'wifi')):
            f = (second - state.ready) / max(1e-6, gap)
            p = _smooth_fraction(f)
            return _add(state.last, _scale(_sub(end, state.last), p)), 'travel', (), 0.

        prev_lane = _lane_number(self.last_ended_pad)
        next_lane = _lane_number(upcoming.pad) if last == self.index + 1 else 0
        if gap <= .115 and prev_lane and next_lane and abs(_ring_step(prev_lane, next_lane)) == 1 and (upcoming.is_ex or upcoming.in_spin_run):
            f = (second - state.ready) / max(1e-6, gap)
            return _arc_button_motion(state.last, end, f, 4.), 'travel', (), 0.

        if gap > .28:
            strike_window = .085
            move_dur = min(.30, (gap - strike_window) * .72)
            hover = _add(end, _scale(_PANEL_NORMAL, 26.))
            if second < state.ready + move_dur:
                f = (second - state.ready) / max(1e-6, move_dur)
                return free_motion(state.last, hover, f, 18., panel_normal=_PANEL_NORMAL), 'travel', (), 0.
            if second < upcoming.start - strike_window:
                return hover, 'travel', (), 0.
            f = (second - (upcoming.start - strike_window)) / strike_window
            p = _smooth_fraction(f)
            return _add(hover, _scale(_sub(end, hover), p)), 'travel', (), 0.

        f = (second - state.ready) / max(1e-6, gap)
        lift = (6. if not upcoming.is_button and dist <= 160. else 34.) * min(1., max(0., gap) / .13)
        return free_motion(state.last, end, f, lift, panel_normal=_PANEL_NORMAL), 'travel', (), 0.


def _relax_slide_samples(times, samples, profile, *, share_sensors=False, protected=frozenset()):
    """Smooth small path loops into palm gestures within their contact coverage."""
    result = [list(row) for row in samples]
    if share_sensors:
        for index, row in enumerate(samples):
            active = [contact for sample in row for contact in sample[2]]
            if len(active) < 2 or any(c.kind != 'slide' or c.brush or c.source_id in protected for c in active):
                continue
            for hand in (0, 1):
                assigned = tuple(c for c in active if (c.at(times[index])[0] < 0.) == (hand == 0))
                point = _center(assigned, times[index]) if assigned else row[hand][0]
                result[index][hand] = (point, 'slide' if assigned else 'travel', assigned, 0.)
        samples = [tuple(row) for row in result]
    for hand in (0, 1):
        runs = []
        start = None
        for index, row in enumerate(samples):
            active = row[hand][2]
            all_active = [contact for sample in row for contact in sample[2]]
            sliding = bool(all_active) and all(c.kind == 'slide' for c in all_active) and (share_sensors or bool(active))
            if sliding and start is None:
                start = index
            if start is not None and (not sliding or index == len(samples) - 1):
                runs.append((start, index if sliding else index - 1))
                start = None
        for start, end in runs:
            if end <= start + 1:
                continue
            filtered = [samples[i][hand][0] for i in range(start, end + 1)]
            for offset in range(1, len(filtered)):
                dt = times[start + offset] - times[start + offset - 1]
                blend = -math.expm1(-dt / .04)
                filtered[offset] = tuple(a + blend * (b - a) for a, b in zip(filtered[offset-1], filtered[offset]))
            for offset in range(len(filtered) - 2, -1, -1):
                dt = times[start + offset + 1] - times[start + offset]
                blend = -math.expm1(-dt / .04)
                filtered[offset] = tuple(a + blend * (b - a) for a, b in zip(filtered[offset+1], filtered[offset]))
            for index in range(start + 1, end):
                point, mode, active, home = samples[index][hand]
                envelope = _smooth_fraction((times[index] - times[start]) / .06) * _smooth_fraction((times[end] - times[index]) / .06)
                desired = filtered[index-start]
                # Keep nearby central work on each hand's side when the sensor
                # area permits it. This applies to any path shape.
                side = (-1. if hand == 0 else 1.) * 30. * max(0., 1. - abs(desired[0]) / 180.)
                desired = (desired[0] + side, desired[1], desired[2])
                desired = tuple(a + envelope * (b-a) for a,b in zip(point,desired))
                if active and (not share_sensors or any(c.source_id in protected for c in active)):
                    desired = _covered_center(point, desired, active, times[index], profile)
                result[index][hand] = (desired, mode, active, home)
    return [tuple(row) for row in result]


def _carry_gesture_positions(times, original, candidate, profile):
    """Connect changed gestures to their actual entry and exit positions."""
    result = [list(row) for row in candidate]
    for hand in (0, 1):
        index = 0
        while index < len(times):
            if result[index][hand][2]:
                index += 1
                continue
            end = index
            while end < len(times) and not result[end][hand][2]:
                end += 1
            previous = max(0, index-1)
            start_point = result[previous][hand][0] if index else profile.homes[hand]
            end_point = result[end][hand][0] if end < len(times) else profile.homes[hand]
            changed = (math.dist(start_point, original[previous][hand][0]) > 1e-6
                       or end < len(times) and math.dist(end_point, original[end][hand][0]) > 1e-6
                       or any(original[i][hand][2] for i in range(index, end)))
            if changed:
                begin = times[previous]
                finish = times[end] if end < len(times) else min(times[-1], begin+.38)
                gap = finish-begin
                for frame in range(index, end):
                    t = times[frame]
                    if gap > 1.35 and begin+.38 <= t < finish-.42:
                        result[frame][hand] = (profile.homes[hand], 'idle', (), 1.)
                        continue
                    a, b, lo, hi = start_point, end_point, begin, finish
                    if gap > 1.35:
                        if t < begin+.38:
                            b, hi = profile.homes[hand], begin+.38
                        else:
                            a, lo = profile.homes[hand], finish-.42
                    point = free_motion(a, b, (t-lo)/max(1e-6, hi-lo), 24., panel_normal=_PANEL_NORMAL)
                    idle = end == len(times) and t >= finish
                    result[frame][hand] = (point, 'idle' if idle else 'travel', (), float(idle))
            index = end
    return [tuple(row) for row in result]


def _palm_radius(point, active, mode, profile, second):
    spread = .15 if mode in ('tap', 'hold') else .21 if mode in ('slide', 'touch', 'touch_hold') else .17
    if mode == 'wifi' or len(active) > 1:
        spread = .28
    elif len(active) == 1 and active[0].fingertip_trace:
        spread = .12
    if any(c.brush for c in active):
        spread = .45
    radius = profile.hand_span_mm * spread
    for contact in active:
        allowance = _contact_span(contact, profile) - profile.hand_span_mm / 2
        radius = max(radius, min(profile.hand_span_mm / 2, math.dist(point, contact.at(second)) - allowance))
    return radius


def _judge_slide_samples(times, samples, contacts, profile, joint_frames=None):
    """Replay both palms against the existing ordered MiaCode sensor judge."""
    from maichart.generation.muri.slide import slide_states, step_slides

    markers = {c.source_id: c.judge_marker for c in contacts if c.kind == 'slide' and c.judge_marker}
    if not markers:
        return {}
    states = slide_states(list(markers.values()))
    if not states:
        return {}
    source_ids = {marker['key']: source for source, marker in markers.items()}
    pending = sorted(states, key=lambda state: state['available_second'])
    screen_radius = machine_profile().screen_radius_mm
    sensors = sorted(
        ((pad, touch_xyz(pad), RADII[pad[0]] * screen_radius / 240.) for pad in PADS),
        key=lambda sensor: sensor[1][0],
    )
    sensor_x = [center[0] for _, center, _ in sensors]
    maximum_sensor_radius = max(radius for _, _, radius in sensors)
    causes = tuple({'marker_key': f'player_hand_{hand}', 'source_type': 'player'} for hand in (0, 1))
    start = max(0., min(s['available_second'] for s in states))
    end = min(times[-1], max(s['end_second'] for s in states) + .3)
    pointer = frame = 0
    active_states = []
    previous = {}
    tick = math.floor(start * TPS)
    last_tick = math.ceil(end * TPS)
    while tick <= last_tick:
        second = tick / TPS
        tick += 1
        while pointer < len(pending) and pending[pointer]['available_second'] <= second:
            active_states.append(pending[pointer])
            pointer += 1
        active_states = [s for s in active_states if not s['judged'] and second <= s['end_second'] + .3]
        if not active_states:
            if pointer == len(pending):
                break
            previous = {}
            tick = max(tick, math.floor(pending[pointer]['available_second'] * TPS))
            continue
        frame = max(0, bisect_right(times, second, lo=frame) - 1)
        following = min(frame+1, len(times)-1)
        fraction = min(1., max(0., (second-times[frame]) / max(1e-9, times[following]-times[frame])))
        pressed = {}
        if joint_frames is not None:
            before, after = joint_frames[frame], joint_frames[following]
            yaw = before.yaw_rad + fraction*(after.yaw_rad-before.yaw_rad)
        for hand in (0,1):
            point, mode, active, _ = samples[frame][hand]
            if joint_frames is None:
                point = tuple(a + fraction*(b-a) for a,b in zip(point,samples[following][hand][0]))
            else:
                shoulder = tuple(a + fraction*(b-a) for a,b in zip(before.shoulders[hand], after.shoulders[hand]))
                _, point = interpolate_arm_joints(profile, hand, shoulder, yaw,
                                                 before.arms[hand].joint_angles, after.arms[hand].joint_angles, fraction)
            cause = causes[hand]
            for contact in active:
                if contact.is_button and math.dist(point, contact.at(second)) <= _contact_span(contact, profile):
                    pressed['A'+contact.pad] = cause
            if abs(sum(v*n for v,n in zip(point,_PANEL_NORMAL))) > 8.:
                continue
            radius = _palm_radius(point, active, mode, profile, second)
            # A sensor farther away horizontally cannot pass the 3D distance
            # check. Leave margin for rounding when forming the search bounds.
            extent = radius + maximum_sensor_radius + 1e-9
            lo = bisect_left(sensor_x, point[0] - extent)
            hi = bisect_right(sensor_x, point[0] + extent, lo=lo)
            for pad, center, sensor_radius in sensors[lo:hi]:
                if math.dist(point, center) <= radius + sensor_radius:
                    pressed[pad] = cause
        released = {pad:cause for pad,cause in previous.items() if pad not in pressed}
        step_slides(active_states, second, pressed, released)
        previous = pressed
    return {source_ids[s['key']]: {'complete': bool(s['judged'] and not s['judge_bad']),
                                    'timing_error': bool(s['judge_bad']),
                                    'areas': s['cur'], 'total_areas': s['total']} for s in states}


@lru_cache(maxsize=4096)
def _estimated_arm_pose(profile, hand, point, shoulder, yaw, previous_elbow, dt):
    return arm_pose(profile, hand, point, previous_elbow, shoulder=shoulder,
                    yaw_rad=yaw, dt=dt, panel_normal=_PANEL_NORMAL)


def _estimated_arm_gap(profile, points, cache, context=None):
    key = points, context
    if key not in cache:
        shoulders, yaw, previous, dt = context or (profile.shoulders, 0., (None, None), 1./30.)
        # Detours vary one hand's target at a time. Reuse the other arm's solve.
        poses = tuple(_estimated_arm_pose(profile, hand, points[hand], shoulders[hand], yaw,
                                          previous[hand], dt) for hand in (0, 1))
        cache[key] = (arm_pair_clearance(profile, shoulders, poses),
                      sum(math.dist(pose.hand, point) for pose,point in zip(poses,points)))
    return cache[key]


def _detour_free_motion(times, samples, profile, cache, contexts=None):
    """Lift a free transition early when its swept arm would cross the other arm."""
    result = [list(row) for row in samples]
    runs = []
    owners = [[None] * len(times) for _ in range(2)]
    for hand in (0, 1):
        index = 0
        while index < len(times):
            if samples[index][hand][1] != 'travel' or samples[index][hand][2]:
                index += 1
                continue
            end = index
            while end < len(times) and samples[end][hand][1] == 'travel' and not samples[end][hand][2]:
                end += 1
            start = max(0, index-1)
            prior = samples[start][hand][2]
            if len(prior) == 1 and prior[0].kind == 'tap':
                while start > 0 and samples[start-1][hand][2] == prior:
                    start -= 1
            run = (hand, start, min(end, len(times)-1))
            for frame in range(start+1, end):
                owners[hand][frame] = len(runs)
            runs.append(run)
            index = end
    offsets = [(0., 0., 0.)] * len(runs)
    up = (0., math.cos(_TILT_RAD), -math.sin(_TILT_RAD))
    directions = (_PANEL_NORMAL, _add(up, _scale(_PANEL_NORMAL, .3)),
                  _add(_scale(up, -1.), _scale(_PANEL_NORMAL, .3)))
    for index in range(1, len(times), 2):
        if all(owners[hand][index] is None for hand in (0, 1)):
            continue
        points = tuple(result[index][hand][0] for hand in (0, 1))
        context = contexts[index] if contexts is not None else None
        gap, error = _estimated_arm_gap(profile, points, cache, context)
        if gap >= 0.:
            continue
        best = (max(0., -gap), 0.)
        choice = None
        for hand in (0, 1):
            owner = owners[hand][index]
            if owner is None:
                continue
            _, start, end = runs[owner]
            duration = times[end] - times[start]
            u = (times[index] - times[start]) / max(1e-6, duration)
            envelope = 16*u*u*(1-u)*(1-u)
            for direction in directions:
                for amplitude in (60., 100., 140., 180., 220.):
                    offset = _scale(direction, amplitude)
                    if offset == offsets[owner]:
                        continue
                    movement_cost = (math.hypot(*offset)/max(.05,duration))**2
                    if best[0] == 0. and movement_cost >= best[1]:
                        continue
                    proposed = list(points)
                    proposed[hand] = _add(samples[index][hand][0], _scale(offset, envelope))
                    required = samples[index][hand][2]
                    if any(math.dist(proposed[hand], c.at(times[index])) > _contact_span(c, profile) for c in required):
                        continue
                    new_gap, new_error = _estimated_arm_gap(profile, tuple(proposed), cache, context)
                    rank = (max(0., -new_gap), movement_cost)
                    if new_error <= error + .01 and rank < best:
                        best, choice = rank, (owner, offset)
        if choice is None:
            continue
        owner, offset = choice
        hand, start, end = runs[owner]
        for frame in range(start+1, end):
            if owners[hand][frame] != owner:
                continue
            u = (times[frame]-times[start])/(times[end]-times[start])
            point, mode, active, home = samples[frame][hand]
            desired = _add(point, _scale(offset, 16*u*u*(1-u)*(1-u)))
            if active:
                desired = _covered_center(point, desired, active, times[frame], profile)
            result[frame][hand] = (desired, mode, active, home)
        offsets[owner] = offset
    return [tuple(row) for row in result]


def _continue_sweep_motion(times, samples, contacts, hands, profile):
    """Carry angular velocity through a sweep, accelerating and braking at its ends."""
    result = [list(row) for row in samples]
    for hand in (0, 1):
        assigned = [c for c,h in zip(contacts,hands) if h == hand]
        runs = []
        run = []
        direction = 0
        for contact in assigned:
            previous = run[-1] if run else None
            step = _ring_step(_lane_number(previous.pad), _lane_number(contact.pad)) if previous else 0
            linked = previous is not None and contact.sweep_predecessor == previous.id
            continuous = (previous is not None and previous.kind == 'tap'
                          and (contact.kind == 'tap' or (linked and contact.kind == 'hold'))
                          and (contact.in_spin_run or contact.is_ex) and abs(step) == 1
                          and contact.sweep_predecessor in (None, previous.id)
                          and 0. < contact.start-previous.start
                          and (linked or contact.start-previous.start <= .13) and direction in (0, step))
            if continuous:
                run.append(contact)
                direction = step
            else:
                if len(run) >= 3:
                    runs.append(run)
                run = [contact] if contact.kind == 'tap' else []
                direction = 0
        if len(run) >= 3:
            runs.append(run)
        for run in runs:
            updates = []
            valid = True
            speeds = [1./(b.start-a.start) for a,b in zip(run,run[1:])]
            tangents = [0.] + [2*a*b/(a+b) for a,b in zip(speeds,speeds[1:])] + [0.]
            for pair, (a,b) in enumerate(zip(run,run[1:])):
                dt = b.start-a.start
                begin = max(0, bisect_right(times,a.start)-1)
                # Land on the Hold and leave its anchored frames to the contact solver.
                end = bisect_left(times, b.start) if b.kind == 'hold' else bisect_right(times, b.start)
                for index in range(begin,end):
                    if not a.start <= times[index] <= b.start:
                        continue
                    point, mode, active, home = samples[index][hand]
                    if any(c.kind != 'tap' for c in active):
                        valid = False
                        break
                    u = (times[index]-a.start)/dt
                    progress = (-2*u**3+3*u*u) + (u**3-2*u*u+u)*tangents[pair]*dt + (u**3-u*u)*tangents[pair+1]*dt
                    target = _arc_button_motion(a.point,b.point,progress,0.,smooth=False)
                    if any(math.dist(target,c.at(times[index])) > _contact_span(c,profile) for c in active):
                        valid = False
                        break
                    updates.append((index,(target,mode,active,home)))
                if not valid:
                    break
            if valid:
                for index, sample in updates:
                    result[index][hand] = sample
    return [tuple(row) for row in result]


def _gesture_cost(times, samples, profile=None, cache=None):
    cost = 0.
    collision = 0.
    previous_velocity = [None, None]
    cache = {} if cache is None else cache
    for index in range(1, len(times)):
        dt = times[index] - times[index-1]
        if dt <= 1e-6:
            continue
        for hand in (0, 1):
            point = samples[index][hand][0]
            before = samples[index-1][hand][0]
            velocity = tuple((b-a)/dt for a,b in zip(before,point))
            cost += math.dist(point, before)
            if previous_velocity[hand] is not None:
                cost += .006 * math.dist(velocity, previous_velocity[hand])
            previous_velocity[hand] = velocity
        left, right = samples[index]
        if left[2] and right[2]:
            cost += dt * 2. * max(0., left[0][0] - right[0][0])
        if profile is not None and (left[1] != 'idle' or right[1] != 'idle'):
            gap, _ = _estimated_arm_gap(profile, (left[0], right[0]), cache)
            collision += dt * max(0., -gap)
    return collision, cost


def _playback_samples(contacts, hands, profile, duration):
    owners = {c.id: h for c,h in zip(contacts,hands)}
    rolls = {key: roll for roll in _touch_rolls(contacts, profile).values()
             if owners[roll.first.id] == owners[roll.last.id] >= 0 for key in (roll.first.id, roll.last.id)}
    playback = [HandPlayback([c for c, h in zip(contacts, hands) if h == i], profile.homes[i], profile, i, rolls=rolls)
                for i in range(2)]
    boundaries = {0., duration}
    for c in contacts:
        boundaries.update(t for t in (c.start, c.end, *(round(part.end, 6) for part in c.segments)) if 0. <= t <= duration)
    for player in playback:
        for relay in player.relays:
            boundaries.update((relay.start, relay.end))
    hz = min(30., (MAX_FRAMES - len(boundaries) - 1) / max(duration, .001))
    if hz < 4.:
        raise ValueError('谱面过密，动作预览超出采样上限')
    # Prepare before the first visible frame, including charts that start on a
    # held or crossed contact. The preview exports only nonnegative times.
    preroll = min(0., min((c.start for c in contacts), default=0.)) - .5
    times = sorted(boundaries | {round(i / hz, 6) for i in range(math.floor(preroll * hz), math.ceil(duration * hz))})
    samples = [tuple(player.sample(t) for player in playback) for t in times]
    return times, samples, hz


def _can_reassign_slides(times, samples):
    # Only the shared-sensor candidate in _relax_slide_samples changes which
    # contact belongs to which hand. Its side test is in cabinet coordinates.
    for second, pair in zip(times, samples):
        left, right = pair[0][2], pair[1][2]
        if len(left) + len(right) < 2 or any(c.kind != 'slide' or c.brush for c in (*left, *right)):
            continue
        if any(c.at(second)[0] >= 0. for c in left) or any(c.at(second)[0] < 0. for c in right):
            return True
    return False


def _restore_trace_intervals(times, reference, candidate, markers):
    """Restore proven gestures, joining only where both hands already agree for two frames."""
    from maichart.generation.muri.slide import slide_states
    cuts = [i for i in range(1, len(times))
            if reference[i] == candidate[i] and reference[i-1] == candidate[i-1]]
    intervals = []
    for state in sorted(slide_states(markers), key=lambda s: s['available_second']):
        lo = bisect_left(times, state['available_second']-1/TPS)
        hi = bisect_right(times, state['end_second']+.3)
        before, after = bisect_right(cuts, lo), bisect_left(cuts, hi)
        start = cuts[before-1] if before else 0
        stop = cuts[after] if after < len(cuts) else len(times)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = intervals[-1][0], max(stop, intervals[-1][1])
        else:
            intervals.append((start, stop))
    result = list(candidate)
    for start, stop in intervals:
        result[start:stop] = reference[start:stop]
    return result


def _select_gesture(times, samples, contacts, hands, profile, posture_cache, *, hands_only=False):
    if hands_only and not _can_reassign_slides(times, samples):
        return samples, {}
    samples = _continue_sweep_motion(times, samples, contacts, hands, profile)
    if not any(c.kind == 'slide' and c.judge_marker for c in contacts):
        return _detour_free_motion(times, samples, profile, posture_cache), {}
    pairs = _independent_trace_pairs(contacts)
    protected = set()
    markers = {c.source_id: c.judge_marker for c in contacts if c.judge_marker}
    finalists = []
    fewest_misses = math.inf
    for mode in ('original', 'smooth', 'shared'):
        candidate = samples
        if mode != 'original':
            candidate = _relax_slide_samples(times, samples, profile, share_sensors=mode == 'shared',
                                             protected=protected)
            candidate = _carry_gesture_positions(times, samples, candidate, profile)
        candidate = _detour_free_motion(times, candidate, profile, posture_cache)
        judged = _judge_slide_samples(times, candidate, contacts, profile)
        if mode == 'original':
            reference, reference_judged = candidate, judged
        else:
            lost = [markers[k] for k,v in reference_judged.items() if v['complete'] and not judged[k]['complete']]
            if lost:
                repaired = _restore_trace_intervals(times, reference, candidate, lost)
                checked = _judge_slide_samples(times, repaired, contacts, profile)
                if all(not v['complete'] or checked[k]['complete'] for k,v in judged.items()):
                    candidate, judged = repaired, checked
        if mode != 'shared':
            # Never turn two already successful paths into one palm simply
            # because both stars are on the same half of the screen.
            protected.update(a for a,(b, _) in pairs.items()
                             if judged.get(a, {}).get('complete') and judged.get(b, {}).get('complete'))
        misses = sum(not state['complete'] for state in judged.values())
        if misses < fewest_misses:
            finalists.clear()
            fewest_misses = misses
        if misses == fewest_misses:
            finalists.append((candidate, judged))
    if len(finalists) == 1:
        return finalists[0]
    return min(finalists, key=lambda item: _gesture_cost(times, item[0], profile, posture_cache))


def _sample_frames(contacts, hands, profile, duration, issues, *, stance='adaptive'):
    times, initial_samples, hz = _playback_samples(contacts, hands, profile, duration)
    posture_cache = {}
    samples, judged = _select_gesture(times, initial_samples, contacts, hands, profile, posture_cache)
    # A repair changes only a few travel paths. Reuse a frame solve when its
    # full inputs, including the carried elbows and look-ahead, still match.
    pose_solutions = [None] * len(times)

    def build_pose_frames(samples, body_override=None):
        planner = StancePlanner(profile, times, samples) if stance == 'adaptive' and body_override is None else None
        issues = []
        body_states = []
        contexts = []
        actual_samples = []
        joint_frames = []
        clearances = []
        torso = NEUTRAL_TORSO
        neutral_joints = torso_joints(profile, torso)
        previous_elbows: list[Vec3 | None] = [None, None]
        warnings: dict[tuple[int, str], float] = {}
        frames = []

        def solve(torso_pose, targets):
            forecast = body_override[future_index] if body_override is not None else (step_torso(torso_pose, planner.desired, .3) if planner is not None else torso_pose)
            future_samples = samples[future_index]
            elbow_history = tuple(previous_elbows)
            key = torso_pose, forecast, elbow_history, targets, future_samples
            cached = pose_solutions[index]
            if cached is not None and cached[0] == key:
                return cached[1]
            joints = neutral_joints if torso_pose == NEUTRAL_TORSO else torso_joints(profile, torso_pose)
            future_joints = torso_joints(profile, forecast)
            adj_targets: list[Vec3] = []
            modes: list[str] = []
            for hand, (target, mode, active, home_weight) in enumerate(targets):
                if home_weight:
                    target = tuple(p + home_weight * (current - initial)
                                   for p, current, initial in zip(target, joints.homes[hand], profile.homes[hand]))
                if (mode == 'tap' and len(active) == 1 and not active[0].is_star_head
                        and target == active[0].point):
                    target = _reachable_button_target(target, joints.shoulders[hand],
                                                      profile.upper_arm_mm + profile.forearm_mm - 1.)
                adj_targets.append(target)
                modes.append(mode)
            poses = resolve_bimanual_arms(
                profile,
                (adj_targets[0], adj_targets[1]),
                (modes[0], modes[1]),
                elbow_history,
                joints.shoulders,
                torso_pose.yaw_rad,
                _PANEL_NORMAL,
                dt=dt,
                future_targets=tuple(
                    tuple(p + sample[3] * (current-initial)
                          for p,current,initial in zip(sample[0], future_joints.homes[h], profile.homes[h]))
                    for h,sample in enumerate(future_samples)),
                future_shoulders=future_joints.shoulders,
                future_yaw_rad=forecast.yaw_rad,
            )
            errors = [
                (math.dist(poses[h].hand, adj_targets[h]) if targets[h][2] else 0.,
                 max(0., poses[h].panel_penetration_mm - 3.))
                for h in range(2)
            ]
            result = joints, poses, errors
            pose_solutions[index] = key, result
            return result

        for index, (second, targets) in enumerate(zip(times, samples)):
            dt = second - times[index - 1] if index else 0.
            future_index = min(len(times)-1, bisect_right(times, second + .30))
            proposal = body_override[index] if body_override is not None else (planner.advance(torso, index, dt) if planner is not None else torso)
            joints, poses, errors = solve(proposal, targets)
            if proposal != torso and any(reach > .01 or depth > 0. for reach, depth in errors):
                previous = solve(torso, targets)
                if any(new_reach > old_reach + .01 or new_depth > old_depth + .01
                       for (new_reach, new_depth), (old_reach, old_depth) in zip(errors, previous[2])):
                    proposal = torso
                    joints, poses, errors = previous
            torso = proposal
            body_states.append(torso)
            contexts.append((joints.shoulders, torso.yaw_rad, tuple(previous_elbows), dt))
            actual_samples.append(tuple((pose.hand, target[1], target[2], target[3])
                                        for pose,target in zip(poses,targets)))
            joint_frames.append(ArmFrame(joints.shoulders, torso.yaw_rad, poses))
            clearances.append(arm_pair_clearance(profile, joints.shoulders, poses))
            row = [second]
            hand_markers = []
            for hand, (pose, (_, mode, active, _)) in enumerate(zip(poses, targets)):
                previous_elbows[hand] = pose.elbow
                marker = _hand_marker(pose, active, mode, profile, second, judged)
                hand_markers.append(marker)
                status = {
                    'reach_limit': bool(active) and pose.limited and not pose.joint_limited,
                    'joint_limit': bool(active) and pose.joint_limited,
                    'palm_coverage': bool(active) and not pose.limited and not errors[hand][1] and any(not site[6] for site in marker[2]),
                    'cabinet_clearance': errors[hand][1] > 0.,
                }
                for kind, present in status.items():
                    key = hand, kind
                    if present:
                        warnings.setdefault(key, second)
                    elif key in warnings:
                        _pose_warning(issues, kind, hand, warnings.pop(key), second)
                row.append([*(round(x, 2) for p in (pose.elbow, pose.hand) for x in p), mode])
            row.append([*(round(x, 2) for p in (joints.pelvis, *joints.shoulders) for x in p), round(torso.yaw_rad, 5)])
            row.append(hand_markers)
            row.append([[round(value, 8) for value in pose.joint_angles] for pose in poses])

            if second >= 0.:
                frames.append(row)

        for (hand, kind), start in warnings.items():
            _pose_warning(issues, kind, hand, start, duration)
        issues.extend(_arm_collision_issues(times, clearances))
        return frames, issues, body_states, contexts, actual_samples, joint_frames, clearances

    frames, pose_issues, body_states, contexts, actual_samples, joint_frames, clearances = build_pose_frames(samples)
    for _ in range(4):
        if not any(item['kind'] == 'arm_collision' for item in pose_issues):
            break
        repaired = _detour_free_motion(times, samples, profile, posture_cache, contexts)
        if repaired == samples:
            break
        verdicts = _judge_slide_samples(times, repaired, contacts, profile)
        if all(not state['complete'] or verdicts.get(source, {}).get('complete') for source, state in judged.items()):
            changed_frames, changed_issues, changed_body, changed_contexts, changed_samples, changed_joints, changed_clearances = build_pose_frames(repaired, body_states)

            def rank(items):
                duration_by_kind = Counter()
                for item in items:
                    duration_by_kind[item['kind']] += max(0., item.get('end_sec', item['time_sec'])-item['time_sec'])
                return (duration_by_kind['reach_limit']+duration_by_kind['joint_limit']+duration_by_kind['palm_coverage'],
                        duration_by_kind['cabinet_clearance'], duration_by_kind['arm_collision'])

            if rank(changed_issues) < rank(pose_issues):
                frames, pose_issues, judged = changed_frames, changed_issues, verdicts
                samples, body_states, contexts = repaired, changed_body, changed_contexts
                actual_samples = changed_samples
                joint_frames = changed_joints
                clearances = changed_clearances
                continue
        break
    # Joint limits may move a requested palm away from its intended path. Judge
    # the reachable motion that is drawn, not only the proposed gesture.
    judged = _judge_slide_samples(times, actual_samples, contacts, profile, joint_frames)
    adjusted, adjusted_clearances = repair_elbow_clearance(profile, times, joint_frames, clearances, _PANEL_NORMAL)
    if adjusted is not joint_frames:
        adjusted_judgment = _judge_slide_samples(times, actual_samples, contacts, profile, adjusted)
        if all(not state['complete'] or adjusted_judgment.get(source, {}).get('complete') for source, state in judged.items()):
            first_visible = bisect_left(times, 0.)
            for row, before, after in zip(frames, joint_frames[first_visible:], adjusted[first_visible:]):
                if after is before:
                    continue
                for side, arm in enumerate(after.arms):
                    row[1+side][:3] = [round(x, 2) for x in arm.elbow]
                    row[5][side] = [round(x, 8) for x in arm.joint_angles]
            pose_issues = [item for item in pose_issues if item['kind'] != 'arm_collision']
            pose_issues.extend(_arm_collision_issues(times, adjusted_clearances))
            judged = adjusted_judgment
    issues.extend(pose_issues)
    for source, state in judged.items():
        if not state['complete']:
            contact = next(c for c in contacts if c.source_id == source)
            issues.append(_issue('slide_sensor_review', contact.start,
                                 '本次动作的滑条触区覆盖或尾部时序需要复核', contact_id=source,
                                 sensor_result=state, end_sec=max(c.end for c in contacts if c.source_id == source)))
    return frames, hz


def _hand_marker(pose, active, mode, profile, second, sensor_results=None):
    center = [round(x, 2) for x in pose.hand]
    sites = []
    for contact in active:
        point = contact.at(second)
        covered = (not pose.limited and pose.panel_penetration_mm <= 3.
                   and math.dist(pose.hand, point) <= _contact_span(contact, profile) + 1e-5)
        if contact.kind == 'slide' and sensor_results and sensor_results.get(contact.source_id, {}).get('complete'):
            point = pose.hand
            covered = not pose.limited and pose.panel_penetration_mm <= 3.
        sites.append([contact.id, contact.kind, *(round(x, 2) for x in point), contact.pad, covered, contact.source_id])
    # Show the footprint actually needed to cover the currently assigned sites.
    # Button/sensor area is accounted for separately from the palm's extent.
    radius = _palm_radius(pose.hand, active, mode, profile, second)
    return [center, round(radius, 2), sites]


def _pose_warning(issues, kind, hand, start, end):
    side = '左手' if hand == 0 else '右手'
    detail = {
        'reach_limit': '超出当前站姿的可达范围',
        'joint_limit': '到达该触点会超出肩肘活动范围，需复核换手或站位',
        'palm_coverage': '的掌幅不足以覆盖这一组接触点',
        'cabinet_clearance': '的前臂或肘在这个姿态中越过盘面',
    }[kind]
    issues.append(_issue(kind, start, side + detail, end_sec=round(end, 6), hand=side))


def _arm_collision_issues(times, clearances):
    issues = []
    start = None
    for index, (second, clearance) in enumerate(zip(times, clearances)):
        if second < 0.:
            continue
        if clearance < -ARM_OVERLAP_TOLERANCE_MM and start is None:
            start = second
        if start is not None and (clearance >= -ARM_OVERLAP_TOLERANCE_MM or index == len(times) - 1):
            issues.append(_issue('arm_collision', start, '当前模拟的双臂路径仍有重叠，可复核换手或站位', end_sec=second))
            start = None
    return issues


def _overview(frames, issues, duration, neutral_pelvis=(0., 0., 0.)):
    """Time-weighted motion activity and a fixed-size whole-chart index."""
    count = max(1, min(240, math.ceil(duration / .5)))
    width = duration / count
    bins = [[i * width, (i + 1) * width, 0., 0., 0., 0, 0.] for i in range(count)]
    if duration > 0.:
        for frame, after in zip(frames, frames[1:]):
            start, end = frame[0], after[0]
            index = min(count - 1, int(start / width))
            active = (int(frame[1][-1] != 'idle') + int(frame[2][-1] != 'idle')) / 2.
            lateral = abs(frame[3][0] - neutral_pelvis[0])
            yaw = abs(math.degrees(frame[3][9]))
            shift = math.dist(frame[3][:3], neutral_pelvis)
            while index < count and start < end:
                cell = bins[index]
                overlap_end = min(end, cell[1])
                if overlap_end > start:
                    cell[2] += active * (overlap_end - start) / width
                    cell[3] = max(cell[3], lateral)
                    cell[4] = max(cell[4], yaw)
                    cell[6] = max(cell[6], shift)
                start = max(start, overlap_end)
                index += 1
    for issue in issues:
        index = min(count - 1, int(issue['time_sec'] / width)) if width else 0
        bins[index][5] += 1
    return dict(duration_sec=duration, bins=[
        [round(start, 6), round(end, 6), round(min(activity, 1.), 4), round(lateral, 2), round(yaw, 2), count, round(shift, 2)]
        for start, end, activity, lateral, yaw, count, shift in bins
    ])


def _annotate_playback(notes, contacts, hands, frames, issues=()):
    """Attach contact coverage to visual notes, including deliberate Touch group clears."""
    gaps = defaultdict(list)
    group_hits = Counter(c.touch_group for c, hand in zip(contacts, hands)
                         if hand >= 0 and c.touch_group is not None)
    groups = defaultdict(list)
    by_id = {c.id: c for c in contacts}
    for contact in contacts:
        if contact.touch_group is not None:
            groups[contact.touch_group].append(contact)
    group_cleared = set()
    for contact, hand in zip(contacts, hands):
        if hand >= 0:
            continue
        if contact.touch_group is not None and group_hits[contact.touch_group] >= contact.group_required:
            group_cleared.add(contact.source_id)
        else:
            gaps[contact.source_id].append((contact.start, contact.end))
    for issue in issues:
        if issue['kind'] == 'slide_sensor_review':
            gaps[issue['contact_id']].append((issue['time_sec'], issue['end_sec']))
    for frame, after in zip(frames, frames[1:]):
        covered_groups = Counter()
        active_groups = set()
        for _, _, sites in frame[4]:
            for site in sites:
                group = by_id[site[0]].touch_group
                if group is not None:
                    active_groups.add(group)
                    covered_groups[group] += int(site[6])
                elif not site[6]:
                    gaps[site[7]].append((frame[0], after[0]))
        for group in active_groups:
            if covered_groups[group] < groups[group][0].group_required:
                for contact in groups[group]:
                    gaps[contact.source_id].append((frame[0], after[0]))
    for note in notes:
        source = note['contact_id']
        merged = []
        for start, end in sorted(gaps[source]):
            if merged and start <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        status = 'incomplete' if merged else 'group' if source in group_cleared else 'covered'
        note['simulation'] = dict(status=status, gaps=merged)


def simulate_parsed_chart(chart, *, profile_id='standard', stance='adaptive', sweep_style='continuous'):
    profile = profile_for(profile_id)
    contacts, visual_notes, issues, chart_end = _extract_chart_contacts_and_visuals(chart)
    duration = max(0., chart_end, max((c.end for c in contacts), default=0.))
    from .rim_stroke import plan_playback
    contacts, hands = plan_playback(contacts, profile, stance, sweep_style)
    required, unassigned = _review_assignments(contacts, hands, issues)
    frames, hz = _sample_frames(contacts, hands, profile, duration, issues, stance=stance)
    _annotate_playback(visual_notes, contacts, hands, frames, issues)
    neutral_pelvis = torso_joints(profile, NEUTRAL_TORSO).pelvis
    return dict(schema='maichart-player-motion-v2', machine=machine_payload(),
                profile=dict(id=profile.id, label=profile.label, shoulders=profile.shoulders,
                             joint_limits=JOINT_LIMITS,
                             stature_mm=profile.stature_mm,
                             upper_arm_radius_mm=profile.upper_arm_radius_mm, forearm_radius_mm=profile.forearm_radius_mm,
                             neutral_pelvis=neutral_pelvis, upper_arm_mm=profile.upper_arm_mm,
                             forearm_mm=profile.forearm_mm, hand_span_mm=profile.hand_span_mm),
                stance=stance, sweep_style=sweep_style, overview=_overview(frames, issues, duration, neutral_pelvis),
                frames=frames, notes=visual_notes,
                issues=sorted(issues, key=lambda issue: issue['time_sec']),
                summary=dict(contacts=required, unassigned=unassigned,
                             authored_contacts=len({c.source_id for c in contacts}),
                             visual_notes=len(visual_notes),
                             unknown=sum(issue['kind'] == 'unknown_note' for issue in issues), sample_hz=round(hz, 2)))


@lru_cache(maxsize=4)
def _cached_simulation(path, modified_ns, size, difficulty_index, profile_id, stance, sweep_style):
    chart = parse_event_chart(path, difficulty_index, strict_notes=False, preserve_each_groups=True)
    result = simulate_parsed_chart(chart, profile_id=profile_id, stance=stance, sweep_style=sweep_style)
    stamp = Path(path).stat()
    if (stamp.st_mtime_ns, stamp.st_size) != (modified_ns, size):
        raise ValueError('谱面已改变，请重新载入动作预览')
    return result


def simulate_chart(path: str | Path, difficulty_index: int, *, profile: str = 'standard',
                   stance: str = 'adaptive', sweep_style: str = 'continuous') -> dict:
    """Simulate on demand using immutable cached poses for the selected chart."""
    path = Path(path).resolve()
    stamp = path.stat()
    if stance not in ('adaptive', 'fixed'):
        raise ValueError('请选择可动或固定站位')
    return _cached_simulation(str(path), stamp.st_mtime_ns, stamp.st_size, int(difficulty_index),
                              profile, stance, sweep_style)
