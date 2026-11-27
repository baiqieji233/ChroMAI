"""Shared torso planning for natural maimai DX body weight shifts, crouch, and shoulder turn."""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import math

from .player_geometry import machine_profile
from .player_kinematics import PlayerProfile, TorsoPose, arm_pair_clearance, arm_pose, arm_posture_cost, torso_joints

# Preview response limits, to be calibrated against recorded player movement.
TORSO_TRANSLATION_MM_S = 420.
TORSO_YAW_RAD_S = math.radians(120.)
CONTROL_PERIOD_SEC = .10
LOOKAHEAD_SEC = .40
RESPONSE_SEC = .18
NEUTRAL_TORSO = TorsoPose()


def _posture_choices(profile: PlayerProfile):
    for x, y, z, yaw in product(
        (-90., 0., 90.),
        (-80., -35., 0., 40.),
        (-45., 0., 70.),
        (-26., 0., 26.),
    ):
        pose = TorsoPose(x, y, z, math.radians(yaw))
        y_scale = 85. if y <= 0. else 55.
        regularization = .016 * ((x / 80.) ** 2 + (y / y_scale) ** 2 + (z / 45.) ** 2 + (yaw / 24.) ** 2)
        yield pose, torso_joints(profile, pose).shoulders, regularization


def _target_geometry(choices, maximum_reach: float, upper_arm: float):
    @lru_cache(maxsize=256)
    def lookup(hand, target):
        # Adjacent lookahead windows share targets. Their weights change, but
        # each target's geometry relative to the candidate shoulders is fixed.
        result = []
        for _, shoulders, _ in choices:
            shoulder = shoulders[hand]
            extension = math.dist(shoulder, target) / maximum_reach
            jam = max(0., .38 - extension)
            stretch = max(0., extension - .72)
            extreme = max(0., extension - .90)
            low_contact = max(0., (shoulder[1] - target[1]) / upper_arm - .9)
            effort = 2.6 * stretch ** 2 + 4.0 * jam ** 2 + 160. * extreme ** 2 + .8 * low_contact ** 2
            result.append((extension, effort))
        return tuple(result)

    return lookup


def step_torso(current: TorsoPose, desired: TorsoPose, dt: float) -> TorsoPose:
    """Apply a damped and speed-bounded update using actual frame spacing."""
    if dt <= 0. or current == desired:
        return current
    dx = desired.lateral_mm - current.lateral_mm
    dy = desired.vertical_mm - current.vertical_mm
    dz = desired.depth_mm - current.depth_mm
    da = desired.yaw_rad - current.yaw_rad
    fraction = min(-math.expm1(-dt / RESPONSE_SEC), TORSO_TRANSLATION_MM_S * dt / max(1e-9, math.hypot(dx, dy, dz)),
                   TORSO_YAW_RAD_S * dt / max(1e-9, abs(da)))
    return TorsoPose(current.lateral_mm + fraction * dx, current.vertical_mm + fraction * dy,
                     current.depth_mm + fraction * dz, current.yaw_rad + fraction * da)


class StancePlanner:
    """Choose one shared posture for both hands from present and near-future targets."""

    def __init__(self, profile: PlayerProfile, times, samples):
        self.profile = profile
        self.maximum_reach = profile.upper_arm_mm + profile.forearm_mm
        self.upper_arm = profile.upper_arm_mm
        tilt = math.radians(machine_profile().tilt_deg_from_vertical)
        self.panel_normal = (0., math.sin(tilt), math.cos(tilt))
        self.choices = tuple(_posture_choices(profile))
        self._target_geometry = _target_geometry(self.choices, self.maximum_reach, self.upper_arm)
        self.times = times
        self.samples = samples
        self.next_control = times[0] if times else 0.
        self.desired = NEUTRAL_TORSO

    def advance(self, current: TorsoPose, index: int, dt: float) -> TorsoPose:
        second = self.times[index]
        if second >= self.next_control:
            self.next_control = second + CONTROL_PERIOD_SEC
            self.desired = self._choose(index, current)
        return step_torso(current, self.desired, dt)

    def _choose(self, index: int, current: TorsoPose = NEUTRAL_TORSO) -> TorsoPose:
        second = self.times[index]
        limit = second + LOOKAHEAD_SEC
        intents = {}
        for frame in range(index, len(self.times) - 1):
            start = self.times[frame]
            if start >= limit:
                break
            end = min(self.times[frame + 1], limit)
            midpoint = .5 * (start + end)
            lookahead_weight = (end - start) * 10. * (1. - .5 * (midpoint - second) / LOOKAHEAD_SEC)
            for hand, (target, mode, active, home_weight) in enumerate(self.samples[frame]):
                if mode == 'idle':
                    continue
                weight = lookahead_weight if active else lookahead_weight * .35 * (1. - home_weight)
                if weight > 0.:
                    key = (hand, target)
                    intents[key] = intents.get(key, 0.) + weight
        if not intents:
            return NEUTRAL_TORSO

        target_costs = [(self._target_geometry(hand, target), weight) for (hand, target), weight in intents.items()]
        candidates = []
        for choice_index, (pose, shoulders, value) in enumerate(self.choices):
            peak_extension = 0.
            # The torso has inertia. A small hand movement should not repeatedly
            # switch the preferred stance between opposing discrete candidates.
            value += .025 * (
                ((pose.lateral_mm - current.lateral_mm) / 90.) ** 2
                + ((pose.vertical_mm - current.vertical_mm) / 80.) ** 2
                + ((pose.depth_mm - current.depth_mm) / 45.) ** 2
                + ((pose.yaw_rad - current.yaw_rad) / math.radians(26.)) ** 2
            )
            for geometry, weight in target_costs:
                extension, effort = geometry[choice_index]
                if extension > peak_extension:
                    peak_extension = extension
                value += weight * effort
            # A brief tap still requires reaching its endpoint. Keep bending
            # reserve even when that contact occupies only a few sampled frames.
            value += 25. * max(0., peak_extension - .9) ** 2
            candidates.append((value, pose, shoulders))

        # Reach is a cheap first pass. Inspect actual arm posture only for a few
        # candidates, including the best stance at each available depth.
        candidates.sort(key=lambda item: item[0])
        shortlist = candidates[:3]
        depths = set()
        turns = set()
        for candidate in candidates:
            depth = candidate[1].depth_mm
            yaw = candidate[1].yaw_rad
            if depth not in depths or yaw not in turns:
                depths.add(depth)
                turns.add(yaw)
                if candidate not in shortlist:
                    shortlist.append(candidate)
        mean_targets = {}
        for hand in (0, 1):
            targets = [(target, weight) for (side, target), weight in intents.items() if side == hand]
            total = sum(weight for _, weight in targets)
            if total:
                mean = tuple(sum(target[axis] * weight for target, weight in targets) / total for axis in range(3))
                mean_targets[hand] = (mean, total)
        best = math.inf
        chosen = NEUTRAL_TORSO
        for value, pose, shoulders in shortlist:
            arms = {}
            for hand, (target, weight) in mean_targets.items():
                arm = arm_pose(self.profile, hand, target, shoulder=shoulders[hand],
                               yaw_rad=pose.yaw_rad, panel_normal=self.panel_normal)
                arms[hand] = arm
                value += .65 * weight * arm_posture_cost(self.profile, hand, arm, shoulders[hand], pose.yaw_rad)
                value += 30. * weight * (math.dist(arm.hand, target) / self.profile.hand_span_mm) ** 2
            if len(arms) == 2:
                clearance = arm_pair_clearance(self.profile, shoulders, (arms[0], arms[1]))
                weight = min(mean_targets[0][1], mean_targets[1][1])
                value += 12. * weight * (max(0., -clearance) / 60.) ** 2
            if value < best:
                chosen, best = pose, value
        return chosen
