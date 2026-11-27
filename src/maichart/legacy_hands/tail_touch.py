"""Match isolated Touch notes to native sensor entries at ordinary Slide tails."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import replace
import math

from maichart.generation.muri.model import AVAILABLE, EPS, RADII, RELEASE, TPS, start_tick
from .player_geometry import machine_profile


def tail_pickups(contacts):
    """Return unambiguous Touch IDs mapped to an emitting Slide and a judgment tick."""
    by_pad = defaultdict(list)
    buttons = defaultdict(list)
    for contact in contacts:
        if contact.kind in ('touch', 'touch_hold'):
            by_pad[contact.pad].append(contact)
        elif contact.is_button:
            buttons['A' + contact.pad].append(contact)
    if not by_pad:
        return {}
    starts = {pad: [c.start for c in items] for pad, items in by_pad.items()}
    button_starts = {pad: [c.start for c in items] for pad, items in buttons.items()}
    button_ends = {}
    for pad, items in buttons.items():
        ends = []
        for c in items:
            ends.append(max(c.end, ends[-1] if ends else -math.inf))
        button_ends[pad] = ends
    # A source may already be split into handoff fragments by the planner.
    sources = {c.source_id: c.judge_marker for c in contacts if c.kind == 'slide' and c.judge_marker}
    proposals = defaultdict(list)
    for source, marker in sources.items():
        pad = 'A' + str(marker['end_lane'])
        if pad not in by_pad:
            continue
        intervals = marker['slide_segment_pad_enter_times']
        if not intervals:
            continue
        entries = [e['t'] for e in intervals[-1] if e['pad'] == pad]
        if not entries:
            continue
        shoot = marker['slide_segment_shoot_seconds'][-1]
        duration = marker['slide_segment_durations'][-1]
        t = start_tick(shoot + duration * max(entries)) / TPS
        times = starts[pad]
        lo = bisect_left(times, t - AVAILABLE - EPS)
        hi = bisect_right(times, t + AVAILABLE + EPS)
        if hi-lo != 1:
            continue
        touch = by_pad[pad][lo]
        if touch.kind != 'touch' or touch.touch_group is not None:
            continue
        if lo and by_pad[pad][lo-1].end > t + EPS:
            continue
        # A held or simultaneous button on the same A sensor can consume
        # the edge. Keep the independent assignment in that case.
        i = bisect_right(button_starts.get(pad, ()), t + 1/TPS) - 1
        if i >= 0 and button_ends[pad][i] >= t - 1/TPS:
            continue
        proposals[touch.id].append((source,t))
    return {cid: entries[0] for cid, entries in proposals.items() if len(entries)==1}


def attach_pickups(contacts, profile, stance, planner):
    claims = tail_pickups(contacts)
    # Replan once if only some of the candidate touches can be absorbed.
    for _ in range(2):
        if not claims:
            break
        free, owners = planner([c for c in contacts if c.id not in claims], profile, stance)
        slides = defaultdict(list)
        for c, hand in zip(free, owners):
            if c.kind == 'slide' and hand >= 0:
                slides[c.source_id].append((c, hand))
        attached = []
        rejected = []
        for touch in contacts:
            if touch.id not in claims:
                continue
            source,t = claims[touch.id]
            spans = slides[source]
            # Check the actual planned hand and geometry, including modified
            # single-stroke routes. A candidate on paper may be unreachable.
            hit = next(((c,h) for c,h in spans if c.start <= t < c.end),None)
            margin = profile.hand_span_mm*.21 + RADII[touch.pad[0]]*machine_profile().screen_radius_mm/240.
            if hit is None or math.dist(hit[0].at(t),touch.point) > margin:
                rejected.append(touch.id)
                continue
            stop = min(t+RELEASE, hit[0].end)
            attached.append((replace(touch,start=t,end=stop),hit[1]))
        if not rejected:
            pairs = sorted([*zip(free,owners),*attached],key=lambda pair:pair[0].start)
            return [c for c,_ in pairs],[h for _,h in pairs]
        for cid in rejected:
            del claims[cid]
    return planner(contacts,profile,stance)
