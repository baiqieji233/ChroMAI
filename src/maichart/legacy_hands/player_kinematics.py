"""Two-segment 3D arm inverse kinematics (shoulder → elbow → hand) for maimai DX.

Coordinates are millimetres, with +x to the player's right, +y up, and +z
toward the player. Wrist articulation is intentionally omitted: the forearm
terminates directly at the active hand contact center so motion modeling
focuses cleanly on upper-limb, crossing-clearance, and torso mechanics.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
import math

from .player_geometry import machine_profile

Vec3 = tuple[float, float, float]
_EPS = 1e-9
MAX_ELBOW_SWIVEL_RAD_S = math.radians(300.)
# Collision depth tolerance used for the preview's arm capsules.
ARM_OVERLAP_TOLERANCE_MM = 1.
# Working ranges of this preview rig, not individual medical measurements.
JOINT_LIMITS = dict(elevation_max=math.radians(170.), plane_min=math.radians(-70.),
                    plane_max=math.radians(150.), extension_max=math.radians(50.),
                    rotation_max=math.pi/2, flexion_max=math.radians(150.))


@dataclass(frozen=True)
class PlayerProfile:
    id: str
    label: str
    shoulders: tuple[Vec3, Vec3]
    homes: tuple[Vec3, Vec3]
    upper_arm_mm: float
    forearm_mm: float
    hand_span_mm: float
    torso_height_mm: float = 480.
    stature_mm: float = 1750.
    upper_arm_radius_mm: float = 45.
    forearm_radius_mm: float = 35.


@dataclass(frozen=True, slots=True)
class TorsoPose:
    """Rigid torso motion around the hip center in the cabinet coordinate frame."""

    lateral_mm: float = 0.
    vertical_mm: float = 0.
    depth_mm: float = 0.
    yaw_rad: float = 0.


@dataclass(frozen=True, slots=True)
class TorsoJoints:
    pelvis: Vec3
    shoulders: tuple[Vec3, Vec3]
    homes: tuple[Vec3, Vec3]


@dataclass(frozen=True, slots=True)
class ArmPose:
    elbow: Vec3
    hand: Vec3
    reach_ratio: float
    limited: bool
    panel_penetration_mm: float = 0.
    joint_angles: tuple[float, float, float, float] = ()
    joint_limited: bool = False


@dataclass(frozen=True, slots=True)
class ArmFrame:
    shoulders: tuple[Vec3, Vec3]
    yaw_rad: float
    arms: tuple[ArmPose, ArmPose]


# ANSUR II conditional medians for stature within +/-10 mm of 165/175/185 cm.
# Acromion-radiale and forearm-to-grip-center approximate joint-center lengths.
# The standing depth is a preview assumption, independent of these measurements.
_SCREEN_HEIGHT = machine_profile().screen_center_height_mm
_PROFILES = {
    "standard": PlayerProfile(
        "standard", "约 175 cm", ((-206.5, 1435.-_SCREEN_HEIGHT, 500.), (206.5, 1435.-_SCREEN_HEIGHT, 500.)),
        ((-185., 1435.-_SCREEN_HEIGHT-100., 220.), (185., 1435.-_SCREEN_HEIGHT-100., 220.)),
        333., 347., 165., 539., 1750.,
    ),
    "compact": PlayerProfile(
        "compact", "约 165 cm", ((-190., 1352.-_SCREEN_HEIGHT, 470.), (190., 1352.-_SCREEN_HEIGHT, 470.)),
        ((-170., 1352.-_SCREEN_HEIGHT-95., 210.), (170., 1352.-_SCREEN_HEIGHT-95., 210.)),
        313., 323., 145., 503., 1650.,
    ),
    "long_reach": PlayerProfile(
        "long_reach", "约 185 cm", ((-214., 1522.5-_SCREEN_HEIGHT, 530.), (214., 1522.5-_SCREEN_HEIGHT, 530.)),
        ((-190., 1522.5-_SCREEN_HEIGHT-106., 238.), (190., 1522.5-_SCREEN_HEIGHT-106., 238.)),
        353., 365., 185., 564.5, 1850.,
    ),
}


def profile_for(name: str) -> PlayerProfile:
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown player profile: {name}") from None


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, scale: float) -> Vec3:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _yaw(vector: Vec3, cosine: float, sine: float) -> Vec3:
    x, y, z = vector
    return (x * cosine - z * sine, y, x * sine + z * cosine)


def torso_joints(profile: PlayerProfile, torso: TorsoPose) -> TorsoJoints:
    """Transform a fixed-width shoulder line and resting hands with the hips."""
    left, right = profile.shoulders
    center = _scale(_add(left, right), .5)
    base_pelvis = (center[0], center[1] - profile.torso_height_mm, center[2])
    pelvis = _add(base_pelvis, (torso.lateral_mm, torso.vertical_mm, torso.depth_mm))
    if torso.yaw_rad == 0.:
        offset = (torso.lateral_mm, torso.vertical_mm, torso.depth_mm)
        return TorsoJoints(pelvis, (_add(left, offset), _add(right, offset)),
                           (_add(profile.homes[0], offset), _add(profile.homes[1], offset)))

    cosine, sine = math.cos(torso.yaw_rad), math.sin(torso.yaw_rad)

    def transform(point: Vec3) -> Vec3:
        return _add(pelvis, _yaw(_sub(point, base_pelvis), cosine, sine))

    return TorsoJoints(pelvis, (transform(left), transform(right)),
                       (transform(profile.homes[0]), transform(profile.homes[1])))


def _smooth_fraction(fraction: float) -> float:
    u = max(0., min(1., fraction))
    return u ** 3 * (10. + u * (-15. + 6. * u))


def interpolate_torso(start: TorsoPose, end: TorsoPose, fraction: float) -> TorsoPose:
    """Smooth position and yaw together, with zero velocity at each endpoint."""
    if fraction <= 0. or start == end:
        return start
    if fraction >= 1.:
        return end
    progress = _smooth_fraction(fraction)
    return TorsoPose(
        start.lateral_mm + (end.lateral_mm - start.lateral_mm) * progress,
        start.vertical_mm + (end.vertical_mm - start.vertical_mm) * progress,
        start.depth_mm + (end.depth_mm - start.depth_mm) * progress,
        start.yaw_rad + math.remainder(end.yaw_rad - start.yaw_rad, math.tau) * progress,
    )


def _unit(a: Vec3, fallback: Vec3) -> Vec3:
    length = math.hypot(*a)
    return _scale(a, 1. / length) if length > _EPS else fallback


def _perpendicular(vector: Vec3, axis: Vec3) -> Vec3:
    return _sub(vector, _scale(axis, _dot(vector, axis)))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _clamp_dot(value: float) -> float:
    return max(-1., min(1., value))


def _circle_point(axis: Vec3, sideways: Vec3, projection: float) -> Vec3:
    return _add(_scale(axis, projection), _scale(sideways, math.sqrt(max(0., 1. - projection * projection))))


def _panel_elbow_pole(pole: Vec3, normal: Vec3, anchor: Vec3, radius: float, axis: Vec3) -> Vec3:
    """Choose the nearest elbow along its fixed-length circle in front of the panel."""
    if _dot(anchor, normal) + radius * _dot(pole, normal) >= -_EPS:
        return pole
    cross_axis = _perpendicular(normal, axis)
    cross_length = math.hypot(*cross_axis)
    if radius * cross_length <= _EPS:
        return pole
    in_plane = _scale(cross_axis, 1. / cross_length)
    required = -_dot(anchor, normal) / (radius * cross_length)
    if required >= 1. - _EPS:
        return in_plane
    projection = _clamp_dot(required)
    sideways = _unit(_perpendicular(pole, in_plane), _cross(axis, in_plane))
    return _circle_point(in_plane, sideways, projection)


def _pole_direction(axis: Vec3, outward: float, cosine: float = 1., sine: float = 0.,
                    swivel_bias_y: float = -1.) -> Vec3:
    # Start from a low elbow near the flank. Outward/rearward poles make even
    # resting arms spread sideways and retract behind the shoulders.
    pole = _perpendicular(_yaw((outward * .22, swivel_bias_y, .05), cosine, sine), axis)
    length = math.hypot(*pole)
    if length > _EPS:
        return _scale(pole, 1. / length)
    backup = _perpendicular(_yaw((0., 0., -1.), cosine, sine), axis)
    return _unit(backup, _yaw((outward, 0., 0.), cosine, sine))


def _bend_reference(upper: Vec3) -> Vec3:
    """Transport the forward-facing elbow hinge from the arm-at-side pose."""
    x, y, z = upper
    cosine = -y
    if cosine < -1. + 1e-6:
        return (0., 0., -1.)
    # Rodrigues' rotation specialized to the fixed down and forward axes.
    scale = 1. / (1. + cosine)
    return _unit(((-x * z) * scale, z, 1. + (-z * z) * scale), (0., 0., 1.))


def arm_joint_angles(shoulder: Vec3, elbow: Vec3, palm: Vec3,
                     hand: int, yaw_rad: float) -> tuple[float, float, float, float]:
    """Shoulder elevation/plane/axial rotation and one-way elbow flexion.

    Local axes are outward from the torso, up, and forward. Forearm pronation
    changes palm orientation and does not provide another elbow bending axis.
    """
    sign = -1. if hand == 0 else 1.
    return _angles_from_segments(_sub(elbow, shoulder), _sub(palm, elbow), sign,
                                 math.cos(yaw_rad), -math.sin(yaw_rad))


def _angles_from_segments(upper: Vec3, fore: Vec3, sign: float, cosine: float, sine: float):
    x, y, z = _yaw(_unit(upper, (0., -1., 0.)), cosine, sine)
    upper = (sign*x, y, -z)
    x, y, z = _yaw(_unit(fore, (0., -1., 0.)), cosine, sine)
    fore = (sign*x, y, -z)
    elevation = math.acos(_clamp_dot(-upper[1]))
    plane = math.atan2(upper[0], upper[2]) if math.sin(elevation) > 1e-6 else 0.
    flexion = math.acos(_clamp_dot(_dot(upper, fore)))
    rotation = 0.
    if math.sin(flexion) > 1e-6:
        reference = _bend_reference(upper)
        bend = _unit(_perpendicular(fore, upper), reference)
        rotation = math.atan2(_dot(bend, _cross(reference, upper)), _dot(bend, reference))
    return elevation, plane, rotation, flexion


def _limited_joint_angles(angles):
    elevation, plane, rotation, flexion = angles
    elevation = min(JOINT_LIMITS['elevation_max'], max(0., elevation))
    maximum_plane = JOINT_LIMITS['plane_max']
    extension = math.sin(JOINT_LIMITS['extension_max'])
    elevation_sine = math.sin(elevation)
    if elevation_sine > extension:
        maximum_plane = min(maximum_plane, math.acos(-extension / elevation_sine))
    plane = min(maximum_plane, max(JOINT_LIMITS['plane_min'], plane))
    rotation = min(JOINT_LIMITS['rotation_max'], max(-JOINT_LIMITS['rotation_max'], rotation))
    flexion = min(JOINT_LIMITS['flexion_max'], max(0., flexion))
    return elevation, plane, rotation, flexion


def _joint_violation(angles):
    bounded = _limited_joint_angles(angles)
    if angles == bounded:
        return 0.
    # The elevation plane and humeral rotation become indeterminate at their
    # respective straight-axis singularities.
    elevation, plane, rotation, flexion = angles
    return sum((abs(math.remainder(elevation-bounded[0], math.tau)),
                math.sin(elevation) * abs(math.remainder(plane-bounded[1], math.tau)),
                math.sin(flexion) * abs(math.remainder(rotation-bounded[2], math.tau)),
                abs(math.remainder(flexion-bounded[3], math.tau))))


def _joint_pose(profile, hand, shoulder, yaw_rad, angles):
    elevation, plane, rotation, flexion = angles
    upper = (math.sin(elevation)*math.sin(plane), -math.cos(elevation), math.sin(elevation)*math.cos(plane))
    reference = _bend_reference(upper)
    bend = _add(_scale(reference, math.cos(rotation)), _scale(_cross(reference, upper), math.sin(rotation)))
    fore = _add(_scale(upper, math.cos(flexion)), _scale(bend, math.sin(flexion)))
    sign = -1. if hand == 0 else 1.

    def world(vector):
        return _yaw((sign*vector[0], vector[1], -vector[2]), math.cos(yaw_rad), math.sin(yaw_rad))

    elbow = _add(shoulder, _scale(world(upper), profile.upper_arm_mm))
    palm = _add(elbow, _scale(world(fore), profile.forearm_mm))
    return elbow, palm


def interpolate_arm_joints(profile, hand, shoulder, yaw_rad, first, second, fraction):
    """Interpolate bounded joint motion without stretching the two arm segments."""
    angles = tuple(a + fraction * (b-a) for a,b in zip(first,second))
    return _joint_pose(profile, hand, shoulder, yaw_rad, _limited_joint_angles(angles))


_ELBOW_SWIVELS = tuple((angle, math.cos(angle), math.sin(angle))
                      for angle in (0., -.5, .5, -1., 1., -1.5, 1.5, -2., 2., math.pi))


@lru_cache(maxsize=2048)
def _elbow_candidates(profile, hand, actual_hand, shoulder, yaw_rad,
                       panel_normal, reach_axis, anchor, panel_anchor, bend, pole):
    """Share fixed-target IK geometry across the bimanual rotation alternatives."""
    tangent = _cross(reach_axis, pole)
    sign = -1. if hand == 0 else 1.
    cosine, sine = math.cos(yaw_rad), -math.sin(yaw_rad)
    inverse_upper = 1. / profile.upper_arm_mm
    result = []
    for angle, swivel_cosine, swivel_sine in _ELBOW_SWIVELS:
        candidate = _add(_scale(pole, swivel_cosine), _scale(tangent, swivel_sine))
        if panel_normal is not None:
            candidate = _panel_elbow_pole(candidate, panel_normal, panel_anchor, bend, reach_axis)
        elbow = _add(anchor, _scale(candidate, bend))
        upper = _sub(elbow, shoulder)
        cost = _posture_cost(_yaw(_scale(upper, inverse_upper), cosine, sine), sign)
        result.append((candidate, elbow, cost, angle))
    return tuple(result)


@lru_cache(maxsize=16384)
def _candidate_limits(elbow, shoulder, palm, hand, yaw_rad):
    angles = arm_joint_angles(shoulder, elbow, palm, hand, yaw_rad)
    return angles[2], _joint_violation(angles)


def arm_pose(profile: PlayerProfile, hand: int, contact: Vec3,
             previous_elbow: Vec3 | None = None, *, shoulder: Vec3 | None = None,
             yaw_rad: float = 0., panel_normal: Vec3 | None = None,
             elbow_lift_bias: float = 0., dt: float = 1. / 30.,
             shoulder_rotation_target: float | None = None) -> ArmPose:
    """Fit a bounded shoulder and flexion-only elbow, preserving bone lengths."""
    if hand not in (0, 1):
        raise ValueError("hand must be 0 (left) or 1 (right)")
    if not all(math.isfinite(value) for value in contact):
        raise ValueError("contact must be finite")
    contact = tuple(contact)
    shoulder = profile.shoulders[hand] if shoulder is None else shoulder
    shoulder = tuple(shoulder)
    previous_elbow = tuple(previous_elbow) if previous_elbow is not None else None
    panel_normal = tuple(panel_normal) if panel_normal is not None else None
    upper = profile.upper_arm_mm
    lower = profile.forearm_mm
    outer = upper + lower
    inner = math.sqrt(upper*upper + lower*lower + 2*upper*lower*math.cos(JOINT_LIMITS['flexion_max']))
    outward = -1. if hand == 0 else 1.
    cosine, sine = (1., 0.) if yaw_rad == 0. else (math.cos(yaw_rad), math.sin(yaw_rad))

    reach = _sub(contact, shoulder)
    length = math.hypot(*reach)
    reach_axis = _scale(reach, 1. / length) if length > _EPS else (0., -1., 0.)
    clamped = max(inner, min(outer, length))
    limited = abs(length - clamped) > _EPS
    actual_hand = _add(shoulder, _scale(reach_axis, clamped)) if limited else contact

    if clamped > _EPS:
        along = (upper * upper - lower * lower + clamped * clamped) / (2. * clamped)
    else:
        along = 0.
    anchor = _add(shoulder, _scale(reach_axis, along))
    bend = math.sqrt(max(0., upper * upper - along * along))
    panel_anchor = _sub(anchor, _scale(panel_normal, profile.upper_arm_radius_mm)) if panel_normal is not None else anchor

    pole = _pole_direction(reach_axis, outward, cosine, sine, -1. + elbow_lift_bias)
    if bend > _EPS:
        # The lowest geometric elbow can retract behind the torso for close,
        # low contacts. Compare neighbouring swivel solutions before smoothing.
        preferred = pole
        best_cost = (math.inf, math.inf)
        upper_squared = upper**2
        preference = 1.5 if abs(elbow_lift_bias) > .05 else .12
        candidates = _elbow_candidates(profile, hand, actual_hand, shoulder, yaw_rad,
                                       panel_normal, reach_axis, anchor, panel_anchor, bend, pole)
        for candidate, candidate_elbow, cost, angle in candidates:
            if previous_elbow is not None:
                cost += .8 * math.dist(candidate_elbow, previous_elbow)**2 / upper_squared
            # Joint violation and the rotation penalty are nonnegative. A feasible
            # incumbent already wins any tie because candidates are visited in order.
            if best_cost[0] == 0. and cost + preference*angle*angle >= best_cost[1]:
                continue
            rotation, violation = _candidate_limits(candidate_elbow, shoulder, actual_hand, hand, yaw_rad)
            if shoulder_rotation_target is not None:
                cost += 3. * math.remainder(rotation-shoulder_rotation_target, math.tau)**2
            ranked = (violation, cost + preference*angle*angle)
            if ranked < best_cost:
                preferred, best_cost = candidate, ranked
        pole = preferred
    if previous_elbow is not None:
        previous_pole = _unit(_perpendicular(_sub(previous_elbow, anchor), reach_axis), pole)
        angle = math.acos(_clamp_dot(_dot(previous_pole, pole)))
        # Smooth swivel on the IK circle using elapsed time. Note boundaries can
        # be only a few milliseconds apart and must not count as full frames.
        step = min(angle * -math.expm1(-max(0., dt) / .12), MAX_ELBOW_SWIVEL_RAD_S * max(0., dt))
        tangent = _unit(_perpendicular(pole, previous_pole), _cross(reach_axis, previous_pole))
        pole = _add(_scale(previous_pole, math.cos(step)), _scale(tangent, math.sin(step)))
    if panel_normal is not None:
        pole = _panel_elbow_pole(pole, panel_normal, panel_anchor, bend, reach_axis)

    elbow = _add(anchor, _scale(pole, bend))
    angles = arm_joint_angles(shoulder, elbow, actual_hand, hand, yaw_rad)
    joint_limited = _joint_violation(angles) > 1e-6
    if joint_limited and bend > _EPS:
        # A moving reach axis can put the smoothed swivel just beyond a shoulder
        # limit even when this contact has a valid solution. Move to the nearest
        # feasible boundary on the same fixed-length circle before giving up the
        # contact. Clipping joint angles immediately would move the held palm.
        feasible = [candidate for candidate, candidate_elbow, _, _ in candidates
                    if _candidate_limits(candidate_elbow, shoulder, actual_hand, hand, yaw_rad)[1] <= 1e-6]
        if feasible:
            destination = max(feasible, key=lambda candidate: _dot(candidate, pole))
            angle = math.acos(_clamp_dot(_dot(pole, destination)))
            tangent = _unit(_perpendicular(destination, pole), _cross(reach_axis, pole))
            lower_fraction, upper_fraction = 0., 1.
            for _ in range(10):
                fraction = (lower_fraction + upper_fraction) / 2
                attempt = _add(_scale(pole, math.cos(angle*fraction)), _scale(tangent, math.sin(angle*fraction)))
                if panel_normal is not None:
                    attempt = _panel_elbow_pole(attempt, panel_normal, panel_anchor, bend, reach_axis)
                attempt_elbow = _add(anchor, _scale(attempt, bend))
                attempt_angles = arm_joint_angles(shoulder, attempt_elbow, actual_hand, hand, yaw_rad)
                if _joint_violation(attempt_angles) <= 1e-6:
                    upper_fraction = fraction
                    destination = attempt
                else:
                    lower_fraction = fraction
            pole = destination
            elbow = _add(anchor, _scale(pole, bend))
            angles = arm_joint_angles(shoulder, elbow, actual_hand, hand, yaw_rad)
            joint_limited = _joint_violation(angles) > 1e-6
    if joint_limited:
        angles = _limited_joint_angles(angles)
        elbow, actual_hand = _joint_pose(profile, hand, shoulder, yaw_rad, angles)
        limited = True
    penetration = 0. if panel_normal is None else max(0., profile.upper_arm_radius_mm - min(_dot(shoulder, panel_normal), _dot(elbow, panel_normal)), -_dot(actual_hand, panel_normal))
    return ArmPose(elbow, actual_hand, length / outer, limited, penetration, angles, joint_limited)


def segment_distance_3d(first: tuple[Vec3, Vec3], second: tuple[Vec3, Vec3],
                        *, s_bounds: tuple[float, float] = (0., 1.),
                        t_bounds: tuple[float, float] = (0., 1.)) -> float:
    """Closest centerline distance of two 3D segments within optional parameter sub-ranges."""
    a, b = first
    c, d = second
    s_lo, s_hi = s_bounds
    t_lo, t_hi = t_bounds
    ux, uy, uz = _sub(b, a)
    vx, vy, vz = _sub(d, c)
    wx, wy, wz = _sub(a, c)
    uu = math.fsum((ux*ux, uy*uy, uz*uz))
    vv = math.fsum((vx*vx, vy*vy, vz*vz))
    uv = math.fsum((ux*vx, uy*vy, uz*vz))
    uw = math.fsum((ux*wx, uy*wy, uz*wz))
    vw = math.fsum((vx*wx, vy*wy, vz*wz))

    def distance(s: float, t: float) -> float:
        return math.hypot(wx + s*ux - t*vx, wy + s*uy - t*vy, wz + s*uz - t*vz)

    closest = math.inf
    for s in s_bounds:
        t = max(t_lo, min(t_hi, (vw + s * uv) / vv)) if vv > _EPS else t_lo
        closest = min(closest, distance(s, t))
    for t in t_bounds:
        s = max(s_lo, min(s_hi, (t * uv - uw) / uu)) if uu > _EPS else s_lo
        closest = min(closest, distance(s, t))
    determinant = uu*vv - uv*uv
    if determinant > 1e-8:
        s, t = (uv*vw - vv*uw) / determinant, (uu*vw - uv*uw) / determinant
        if s_lo <= s <= s_hi and t_lo <= t <= t_hi:
            closest = min(closest, distance(s, t))
    return closest


def arm_posture_cost(profile: PlayerProfile, hand: int, pose: ArmPose,
                     shoulder: Vec3, yaw_rad: float) -> float:
    """Soft preference for low-effort postures, expressed in the torso frame.

    These dimensionless weights guide the animation. Raised and crossed arms
    remain available when the contact or the other arm requires them.
    """
    upper = _scale(_sub(pose.elbow, shoulder), 1. / profile.upper_arm_mm)
    local = _yaw(upper, math.cos(yaw_rad), -math.sin(yaw_rad))
    return _posture_cost(local, -1. if hand == 0 else 1.)


def _posture_cost(local: Vec3, outward_sign: float) -> float:
    outward = local[0] * outward_sign
    elevation = 1. + local[1]
    retraction = max(0., local[2] - .05)
    across_torso = max(0., -outward) * max(0., local[2] + .25)
    return .55 * elevation + .25 * outward**2 + 2.5 * retraction**2 + 8. * across_torso**2


def arm_pair_clearance(profile: PlayerProfile, shoulders: tuple[Vec3, Vec3],
                       pair: tuple[ArmPose, ArmPose]) -> float:
    """Signed capsule clearance, omitting wrist endpoints where palms may meet."""
    a, b = pair
    return _capsule_clearance(shoulders, a.elbow, a.hand, b.elbow, b.hand,
                              profile.upper_arm_radius_mm, profile.forearm_radius_mm)


@lru_cache(maxsize=4096)
def _capsule_clearance(shoulders, left_elbow, left_palm, right_elbow, right_palm, upper, forearm):
    # The chosen pair is checked again when a frame is emitted. Clearance
    # depends on the geometry only and is shared across candidate scores.
    return min(
        segment_distance_3d((left_elbow, left_palm), (right_elbow, right_palm),
                            s_bounds=(.08, .82), t_bounds=(.08, .82)) - 2. * forearm,
        segment_distance_3d((shoulders[0], left_elbow), (right_elbow, right_palm),
                            t_bounds=(.08, .82)) - upper - forearm,
        segment_distance_3d((left_elbow, left_palm), (shoulders[1], right_elbow),
                            s_bounds=(.08, .82)) - upper - forearm,
    )


def _elbow_orbit(shoulder: Vec3, arm: ArmPose) -> tuple[Vec3, Vec3, Vec3]:
    """Center and two equal-radius vectors on the shoulder–palm orbit."""
    axis = _unit(_sub(arm.hand, shoulder), (0., 0., -1.))
    center = _add(shoulder, _scale(axis, _dot(_sub(arm.elbow, shoulder), axis)))
    radial = _sub(arm.elbow, center)
    return center, radial, _cross(axis, radial)


def _swivel_arm(profile, hand, frame, orbit, angle, panel_normal):
    arm = frame.arms[hand]
    if not angle:
        return arm
    if arm.limited or arm.panel_penetration_mm > 3.:
        return None
    center, radial, tangent = orbit
    elbow = _add(center, _add(_scale(radial, math.cos(angle)), _scale(tangent, math.sin(angle))))
    shoulder = frame.shoulders[hand]
    angles = arm_joint_angles(shoulder, elbow, arm.hand, hand, frame.yaw_rad)
    if _joint_violation(angles) > 1e-6:
        return None
    penetration = 0.
    if panel_normal is not None:
        penetration = max(0., profile.upper_arm_radius_mm - min(_dot(shoulder, panel_normal), _dot(elbow, panel_normal)),
                          -_dot(arm.hand, panel_normal))
        if penetration > 3.:
            return None
    return ArmPose(elbow, arm.hand, arm.reach_ratio, False, penetration, angles)


def _elbow_angle(orbit, elbow):
    center, radial, tangent = orbit
    offset = _sub(elbow, center)
    x, y = _dot(offset, radial), _dot(offset, tangent)
    return math.atan2(y, x) if abs(x) + abs(y) > _EPS else 0.


def _swivel_track(profile, hand, window, orbits, limits, weights, target, panel_normal):
    if target == 0.:
        return [frame.arms[hand] for frame in window]
    track = [window[0].arms[hand]]
    for index, frame in enumerate(window[1:], 1):
        orbit = orbits[index][hand]
        previous = _elbow_angle(orbit, track[-1].elbow)
        limit = limits[index - 1][hand] + 1e-7
        if index == len(window) - 1:
            if abs(previous) > limit:
                return None
            track.append(frame.arms[hand])
            continue
        desired = math.remainder(target * weights[index] - previous, math.tau)
        move = max(-limit, min(limit, desired))
        angle = previous + move
        arm = _swivel_arm(profile, hand, frame, orbit, angle, panel_normal)
        if arm is None:
            # Stay on the reachable arc when the desired position meets a
            # shoulder limit. Try the baseline and intermediate movements.
            alternatives = [previous, previous + .5 * move, previous - .5 * move]
            if abs(previous) <= limit:
                alternatives.append(0.)
            for angle in sorted(alternatives, key=lambda angle: abs(angle - previous - desired)):
                arm = _swivel_arm(profile, hand, frame, orbit, angle, panel_normal)
                if arm is not None:
                    break
        if arm is None:
            return None
        track.append(arm)
    return track


def _midpoint_clearance(profile, before, after, first_arms, last_arms):
    shoulders = tuple(_scale(_add(a, b), .5) for a, b in zip(before.shoulders, after.shoulders))
    yaw = (before.yaw_rad + after.yaw_rad) * .5
    arms = tuple(ArmPose(*interpolate_arm_joints(profile, hand, shoulders[hand], yaw,
                                                a.joint_angles, b.joint_angles, .5), 0., False)
                 for hand, (a, b) in enumerate(zip(first_arms, last_arms)))
    return arm_pair_clearance(profile, shoulders, arms)


def _preserves_interpolated_clearance(profile, window, left, right, tolerance):
    pairs = list(zip(left, right))
    for i, (before, after) in enumerate(zip(window, window[1:])):
        a, b = pairs[i], pairs[i + 1]
        if a == before.arms and b == after.arms:
            continue
        gap = _midpoint_clearance(profile, before, after, a, b)
        if gap < tolerance:
            original = _midpoint_clearance(profile, before, after, before.arms, after.arms)
            if gap < min(tolerance, original) - _EPS:
                return False
    return True


def repair_elbow_clearance(profile: PlayerProfile, times: list[float], frames: list[ArmFrame],
                          clearances: list[float], panel_normal: Vec3 | None):
    """Try smooth elbow adjustments near overlaps while preserving each palm path."""
    overlap = -ARM_OVERLAP_TOLERANCE_MM
    if not any(gap < overlap for gap in clearances):
        return frames, clearances
    adjusted, gaps = list(frames), list(clearances)
    changed = False
    # A smooth lead-in lets the elbow move before the conflicting contact.
    lead = .50
    offsets = tuple(math.radians(a) for a in (0, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90))
    index = 0
    while index < len(times):
        if gaps[index] >= overlap:
            index += 1
            continue
        first = index
        while index < len(times) and gaps[index] < overlap:
            index += 1
        last = index - 1
        start, end = times[first], times[last]
        lo = max(0, bisect_left(times, start - lead) - 1)
        hi = min(len(times), bisect_right(times, end + lead) + 1)
        window = adjusted[lo:hi]
        original_gaps = gaps[lo:hi]
        weights = [_smooth_fraction(min((t - start + lead) / lead, (end + lead - t) / lead))
                   for t in times[lo:hi]]
        orbits = [tuple(_elbow_orbit(s, arm) for s, arm in zip(frame.shoulders, frame.arms)) for frame in window]
        widths = [b - a for a, b in zip(times[lo:hi], times[lo+1:hi+1])]
        if len(widths) < len(window):
            widths.append(0.)
        step_limits = [tuple(max(MAX_ELBOW_SWIVEL_RAD_S * (times[lo + i] - times[lo + i - 1]),
                                 abs(_elbow_angle(orbits[i][h], window[i - 1].arms[h].elbow)))
                             for h in (0, 1)) for i in range(1, len(window))]
        # Each arm's reachable paths depend only on its own history. Reuse them
        # when comparing different left/right combinations.
        tracks = []
        for side in (0, 1):
            options = []
            for angle in offsets:
                track = _swivel_track(profile, side, window, orbits, step_limits, weights, angle, panel_normal)
                if track is not None:
                    options.append((angle, track))
            tracks.append(options)

        peak = min(range(first - lo, last - lo + 1), key=original_gaps.__getitem__)
        anchor = window[peak]
        ranked = []
        for a, left in tracks[0]:
            for b, right in tracks[1]:
                if not (a or b):
                    continue
                gap = arm_pair_clearance(profile, anchor.shoulders, (left[peak], right[peak]))
                if gap > original_gaps[peak]:
                    ranked.append((max(0., overlap - gap), a*a + b*b, left, right))

        def risk(values):
            return (sum(width for width, gap in zip(widths, values) if gap < overlap),
                    sum(width * max(0., overlap - gap) for width, gap in zip(widths, values)))

        best_risk = risk(original_gaps)
        minimum_gap = min(original_gaps)
        best = None
        for _, _, left, right in sorted(ranked, key=lambda row: row[:2])[:20]:
            candidate_gaps = []
            for i, (frame, a, b) in enumerate(zip(window, left, right)):
                gap = arm_pair_clearance(profile, frame.shoulders, (a, b))
                if gap < overlap <= original_gaps[i]:
                    break
                candidate_gaps.append(gap)
            else:
                candidate_risk = risk(candidate_gaps)
                if (candidate_risk < best_risk and candidate_risk[1] <= best_risk[1]
                        and min(candidate_gaps) >= minimum_gap - _EPS
                        and _preserves_interpolated_clearance(profile, window, left, right, overlap)):
                    best_risk = candidate_risk
                    best = left, right, candidate_gaps
                    if candidate_risk[0] == 0.:
                        break
        if best is not None:
            left, right, candidate_gaps = best
            adjusted[lo:hi] = [ArmFrame(frame.shoulders, frame.yaw_rad, (a, b))
                               for frame, a, b in zip(window, left, right)]
            gaps[lo:hi] = candidate_gaps
            changed = True
    return (adjusted, gaps) if changed else (frames, clearances)


@lru_cache(maxsize=16384)
def _future_arm_pose(profile, side, target, shoulder, yaw_rad, panel_normal, rotation):
    # Look-ahead poses have no elbow history. Repair passes can reuse them even
    # when a preceding edit changes the current frame's elbow solution.
    return arm_pose(profile, side, target, shoulder=shoulder, yaw_rad=yaw_rad,
                    panel_normal=panel_normal, shoulder_rotation_target=rotation)


def resolve_bimanual_arms(
    profile: PlayerProfile,
    targets: tuple[Vec3, Vec3],
    modes: tuple[str, str],
    previous_elbows: tuple[Vec3 | None, Vec3 | None],
    shoulders: tuple[Vec3, Vec3],
    yaw_rad: float,
    panel_normal: Vec3 | None,
    *,
    desired_clearance_mm: float = 68.0,
    dt: float = 1. / 30.,
    future_targets: tuple[Vec3, Vec3] | None = None,
    future_shoulders: tuple[Vec3, Vec3] | None = None,
    future_yaw_rad: float | None = None,
) -> tuple[ArmPose, ArmPose]:
    """Try both elbow orders while preserving contacts, bone lengths and continuity."""
    left = arm_pose(profile, 0, targets[0], previous_elbows[0], shoulder=shoulders[0],
                    yaw_rad=yaw_rad, panel_normal=panel_normal, dt=dt)
    right = arm_pose(profile, 1, targets[1], previous_elbows[1], shoulder=shoulders[1],
                     yaw_rad=yaw_rad, panel_normal=panel_normal, dt=dt)
    if modes[0] == modes[1] == "idle":
        return left, right

    def clearance(pair):
        return arm_pair_clearance(profile, shoulders, pair)

    future_pairs = {}
    future_shoulders = shoulders if future_shoulders is None else future_shoulders
    future_yaw_rad = yaw_rad if future_yaw_rad is None else future_yaw_rad
    normal = tuple(panel_normal) if panel_normal is not None else None

    def future_pair(preferences):
        if future_targets is None:
            return None
        pair = []
        for side in (0, 1):
            key = side, preferences[side]
            if key not in future_pairs:
                future_pairs[key] = _future_arm_pose(
                    profile, side, tuple(future_targets[side]), tuple(future_shoulders[side]),
                    future_yaw_rad, normal, preferences[side],
                )
            pair.append(future_pairs[key])
        return tuple(pair)

    def score(pair, preferences=(None, None)):
        distance = clearance(pair)
        overlap = max(0., -distance)
        effort = sum(arm_posture_cost(profile, side, pose, shoulders[side], yaw_rad)
                     for side, pose in enumerate(pair))
        motion = sum(math.dist(pose.elbow, old)**2 / profile.upper_arm_mm**2
                     for pose, old in zip(pair, previous_elbows) if old is not None)
        error = sum(math.dist(pose.hand, target) for pose, target in zip(pair, targets))
        future = future_pair(preferences)
        risk = overlap + (.65 * max(0., -arm_pair_clearance(profile, future_shoulders, future)) if future is not None else 0.)
        return (round(error, 4), risk, effort + .35 * motion)

    if clearance((left, right)) >= max(0., desired_clearance_mm - 2. * profile.forearm_radius_mm):
        future = future_pair((None, None))
        if future is None or arm_pair_clearance(profile, future_shoulders, future) >= 0.:
            return left, right

    best_pair = (left, right)
    best_score = score(best_pair)
    alternatives = []
    for side, base in enumerate((left, right)):
        choices = [(None, base)]
        for rotation in (-1.4, -.7, .7, 1.4):
            pose = arm_pose(profile, side, targets[side], previous_elbows[side], shoulder=shoulders[side],
                            yaw_rad=yaw_rad, panel_normal=panel_normal, dt=dt,
                            shoulder_rotation_target=rotation)
            if pose.panel_penetration_mm <= base.panel_penetration_mm + .5:
                choices.append((rotation, pose))
        alternatives.append(choices)
    for left_rotation, cand_l in alternatives[0]:
        for right_rotation, cand_r in alternatives[1]:
            candidate_score = score((cand_l, cand_r), (left_rotation, right_rotation))
            if candidate_score < best_score:
                best_score = candidate_score
                best_pair = (cand_l, cand_r)
    return best_pair


def free_motion(start: Vec3, end: Vec3, fraction: float, lift_mm: float,
                *, panel_normal: Vec3 = (0., 0., 1.)) -> Vec3:
    """Smooth point-to-point travel lifted along the cabinet's outward normal."""
    u = max(0., min(1., fraction))
    progress = _smooth_fraction(u)
    point = _add(start, _scale(_sub(end, start), progress))
    lift = max(0., lift_mm) * (4. * u * (1. - u)) ** 3
    return _add(point, _scale(panel_normal, lift))
