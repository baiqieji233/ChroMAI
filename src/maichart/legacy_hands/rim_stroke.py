"""Single-palm Slide gestures checked against the ordered sensor sequences."""
from dataclasses import replace
from functools import partial
from itertools import groupby
import math

from maichart.generation.muri.model import CENTERS, RADII, TPS
from maichart.generation.muri.slide import slide_states, step_slides
from . import player_motion as motion


def _rim_sequence(marker, slides):
    keys = marker.get('slide_segment_keys', ())
    if len(keys) != 1:
        return ()
    seq = []
    for area in slides.get(keys[0], {}).get('judge_sequence', ()):
        lanes = [int(p[1:]) for p in area if p.startswith('A')]
        if len(lanes) != 1:
            return ()
        seq.append(lanes[0])
    steps = {motion._ring_step(a, b) for a, b in zip(seq, seq[1:])}
    if len(seq) < 3 or steps not in ({1}, {-1}):
        return ()
    return tuple(seq)


def _judged_path(segments, states, palm, timing_tolerance=None):
    pads = {p for state in states for area in state['sequence'] for p in area}
    previous = {}
    cause = {'marker_key': 'single_palm', 'source_type': 'player'}
    part = 0
    for tick in range(math.ceil(segments[0].start*TPS), math.floor(segments[-1].end*TPS)+1):
        second = tick/TPS
        while part+1 < len(segments) and second > segments[part].end:
            part += 1
        xy = segments[part].raw_xy(second)
        down = {p:cause for p in pads if math.dist(xy,CENTERS[p]) <= palm+RADII[p[0]]}
        step_slides(states,second,down,{p:cause for p in previous if p not in down})
        previous = down
    return all(s['judged'] and not s['judge_bad'] and (timing_tolerance is None or
               abs(s['judge_second']-s['critical_second']) <= timing_tolerance) for s in states)


def _sweep_candidate(run, profile):
    contacts = [c for c, _ in run]
    markers = [c.judge_marker for c in contacts]
    states = slide_states(markers)
    if len(states) != len(markers):
        return None
    step_time = (states[-1]['critical_second'] - states[0]['critical_second']) / (len(run)-1)
    if not .045 <= step_time <= .30:
        return None
    first = run[0][1]
    direction = motion._ring_step(first[0], first[1])
    ax, ay = CENTERS['A'+str(first[0])]
    sensor_ring = math.hypot(ax, ay)
    # The fingers can follow the A/B overlap inside the rim. Hand span stays
    # bounded by the selected player profile in both planning and replay.
    radius = .88 * sensor_ring
    palm = .45 * profile.hand_span_mm * 240 / motion.machine_profile().screen_radius_mm
    reach = palm + RADII['A']
    contact_angle = math.acos(max(-1., min(1., (sensor_ring**2 + radius**2 - reach**2)/(2*radius*sensor_ring))))
    start = round(states[0]['critical_second'] - step_time*(len(first)-1-contact_angle/(math.pi/4)), 6)
    stop = round(markers[-1]['end_second'],6)
    if start < markers[0]['second'] or start >= markers[0]['slide_trace_second'] or stop <= start:
        return None
    initial = math.atan2(ay, ax)
    omega = direction * math.pi / (4*step_time)
    count = max(2, math.ceil((stop-start)*TPS))
    points = tuple((radius*math.cos(initial+omega*(stop-start)*i/count),
                    radius*math.sin(initial+omega*(stop-start)*i/count)) for i in range(count+1))
    route = motion.SlideSegment(start,stop,points)
    if not _judged_path((route,), states, palm):
        return None
    common = replace(contacts[0], id=contacts[0].source_id, start=start, end=stop,
                     point=route.point(start), segments=(route,), slide_chain=None, brush=True)
    members=[]
    for i,c in enumerate(contacts):
        enter = round(start+i*step_time,6)
        end = round(c.judge_marker['end_second'],6)
        if not start <= enter < end <= stop:
            return None
        members.append(replace(c,id=c.source_id,start=enter,end=end,slide_chain=None,
                               point=route.point(enter),segments=(replace(route,start=enter,end=end,
                                      sample_start=start,sample_end=stop),),brush=True))
    return common,members


def _staggered_fan(first, second, profile):
    # Two staggered straight traces can accept the same early sensor entries.
    # Finish at the first endpoint, then move to the adjacent second endpoint.
    a, b = first.judge_marker, second.judge_marker
    if first.pad != second.pad or any(len(m['slide_segment_keys']) != 1 or m['slide_track_key'][1:2] != '-'
                                      for m in (a, b)):
        return None
    start, turn, launch, end = (round(t, 6) for t in
                                (a['slide_trace_second'], a['end_second'], b['slide_trace_second'], b['end_second']))
    if not (0 <= launch-turn <= min(.12, turn-start) and start < turn < end
            and abs(motion._ring_step(int(a['end_lane']), int(b['end_lane']))) == 1):
        return None
    states = slide_states([a, b])
    if len(states) != 2 or any(len(s['sequence']) < 3 for s in states):
        return None
    if not set(states[0]['sequence'][1]).intersection(states[1]['sequence'][1]):
        return None
    resume = (turn + states[1]['critical_second']) / 2
    if not turn < resume < end:
        return None
    origin = CENTERS['A'+first.pad]
    return_distance = math.dist(motion.slide_xyz(*CENTERS['A'+str(a['end_lane'])]),
                                motion.slide_xyz(*origin))
    if return_distance <= profile.hand_span_mm + motion.COHOLD_PLACEMENT_MM_S*(launch-turn):
        return None
    near, far = [tuple(v*.88 for v in CENTERS['A'+str(m['end_lane'])]) for m in (a, b)]
    segments = (motion.SlideSegment(start, turn, (origin, near)),
                motion.SlideSegment(turn, resume, (near, near)),
                motion.SlideSegment(resume, end, (near, far)))
    palm = .45 * profile.hand_span_mm * 240 / motion.machine_profile().screen_radius_mm
    if any(math.dist(s.point(s.start), s.point(s.end)) > motion.BALLISTIC_SPEED_MM_S*(s.end-s.start)
           for s in segments) or not _judged_path(segments, states, palm, timing_tolerance=.06):
        return None
    common = replace(first, id=first.source_id, slide_chain=None, start=start, end=end,
                     point=segments[0].point(start), segments=segments, brush=True)
    members = [replace(common, end=turn, segments=segments[:1]),
               replace(common, id=second.source_id, judge_marker=b)]
    return common, members


def _overlapping_straights(first, second, profile):
    """A later stroke may clear two independently judged, identical tracks."""
    a, b = first.judge_marker, second.judge_marker
    keys = a['slide_segment_keys']
    if (len(keys) != 1 or keys != b['slide_segment_keys'] or a['slide_track_key'][1:2] != '-'
            or not a['slide_trace_second'] < b['slide_trace_second'] < b['end_second'] <= a['end_second']):
        return None
    start = round(b['slide_trace_second'], 6)
    end = round(b['end_second'], 6)
    route = motion.SlideSegment(start, end, second.segments[0].samples)
    if math.dist(route.point(start), route.point(end)) > motion.BALLISTIC_SPEED_MM_S*(end-start):
        return None
    palm = .45 * profile.hand_span_mm * 240 / motion.machine_profile().screen_radius_mm
    states = slide_states([a, b])
    if len(states) != 2 or not _judged_path((route,), states, palm, timing_tolerance=.06):
        return None
    common = replace(first, id=first.source_id, slide_chain=None, start=start, end=end,
                     point=route.point(start), segments=(route,), brush=True)
    return common, [common, replace(common, id=second.source_id, judge_marker=b)]


def candidates(contacts, profile):
    sources = {}
    for contact in contacts:
        if contact.kind == 'slide' and contact.judge_marker:
            sources.setdefault(contact.source_id,contact)
    slides = motion.reference_data()['slides']
    run = []
    ordered = sorted(sources.values(),key=lambda c:c.judge_marker['slide_trace_second'])
    used = set()
    for contact in ordered:
        marker = contact.judge_marker
        seq = _rim_sequence(marker,slides)
        if run:
            prev, before = run[-1]
            prior = prev.judge_marker
            if not (seq and before[1:]==seq[:-1] and prior['slide_trace_second'] < marker['slide_trace_second'] < prior['end_second']
                    and prior['end_second'] < marker['end_second']):
                if len(run)>=3:
                    choice = _sweep_candidate(run,profile)
                    if choice:
                        used.update(c.id for c in choice[1])
                        yield choice
                run=[]
        if seq:
            run.append((contact,seq))
    if len(run)>=3:
        choice = _sweep_candidate(run,profile)
        if choice:
            used.update(c.id for c in choice[1])
            yield choice
    # Offer paired fans only in a two-sided phrase. Ordinary simultaneous
    # branches remain in the two-hand solver, even if their sensors are close.
    for a, b, c, d in zip(ordered, ordered[1:], ordered[2:], ordered[3:]):
        if a.pad != b.pad or c.pad != d.pad or abs(motion._ring_step(int(a.pad), int(c.pad))) < 3:
            continue
        if (not 0 <= c.start-b.judge_marker['end_second'] <= .15
                or any(x.source_id in used for x in (a, b, c, d))):
            continue
        first, second = _staggered_fan(a, b, profile), _staggered_fan(c, d, profile)
        if first and second:
            used.update(x.source_id for x in (a, b, c, d))
            yield first
            yield second
    for first, second in zip(ordered, ordered[1:]):
        if first.source_id in used or second.source_id in used:
            continue
        shared = _overlapping_straights(first, second, profile)
        if shared:
            used.update((first.source_id, second.source_id))
            yield shared


def plan_playback(contacts, profile, stance, sweep_style='continuous', *, raw_hands=None):
    from .tail_touch import attach_pickups
    original = contacts
    def assign(rows):
        if rows is original and raw_hands is not None:
            return list(raw_hands)
        return motion._plan(rows, profile, stance=stance, sweep_style=sweep_style)
    contacts, hands = attach_pickups(contacts, profile, stance, partial(_plan_strokes, assign=assign))
    if any(c.touch_group is not None and h < 0 for c,h in zip(contacts,hands)):
        _cover_incidental_touches(contacts, hands, profile)
    return contacts, hands


def _cover_incidental_touches(contacts, hands, profile):
    """Recognize optional children inside an already placed palm at the same onset."""
    active = [[], []]
    scale = motion.machine_profile().screen_radius_mm / 240.
    for t, rows in groupby(enumerate(contacts), key=lambda item: item[1].start):
        groups = {}
        active = [[c for c in hand if c.end > t] for hand in active]
        for i,c in rows:
            if hands[i] >= 0:
                active[hands[i]].append(c)
            if c.touch_group is not None:
                groups.setdefault(c.touch_group, []).append(i)
        next_start = contacts[i+1].start if i+1<len(contacts) else math.inf
        for group, indices in groups.items():
            missing = [i for i in indices if hands[i] < 0]
            sides = {hands[i] for i in indices if hands[i] >= 0}
            if not missing or len(sides)!=1 or len(indices)-len(missing)<contacts[indices[0]].group_required:
                continue
            side = sides.pop()
            sites = active[side]
            if any(c.touch_group != group or c.end > next_start+1e-6 for c in sites):
                continue
            center = motion._center(sites,t)
            radius = motion._palm_radius(center, sites, 'touch', profile, t)
            if any(math.dist(center,contacts[i].point) > radius+RADII[contacts[i].pad[0]]*scale for i in missing):
                continue
            whole = tuple(contacts[i] for i in indices)
            if motion._fits_palm(whole, t, profile):
                for i in missing:
                    hands[i] = side
                    sites.append(contacts[i])


def _plan_strokes(contacts, profile, stance, *, assign):
    groups = list(candidates(contacts,profile))
    if not groups:
        return contacts,assign(contacts)
    subsumed = {c.source_id for _,members in groups for c in members}
    macros = {macro.id:members for macro,members in groups}
    trial = [c for c in contacts if c.source_id not in subsumed]
    def order(c):
        return c.start,c.touch_group is not None,c.touch_group or 0,-c.end,c.kind,c.pad,c.at(c.end),c.id
    trial.extend(macro for macro,_ in groups)
    trial.sort(key=order)
    hands = assign(trial)
    original = assign(contacts)
    # Keep the ordinary solution when the single-palm route loses a contact.
    was_played = {c.id for c,h in zip(contacts,original) if h >= 0}
    if any(h < 0 and (c.id in macros or c.id in was_played) for c,h in zip(trial,hands)):
        return contacts,original
    expanded = [(member,hand) for c,hand in zip(trial,hands) for member in macros.get(c.id,(c,))]
    expanded.sort(key=lambda pair:order(pair[0]))
    return [c for c,_ in expanded],[h for _,h in expanded]
