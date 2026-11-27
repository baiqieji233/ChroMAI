"""Ordinal labels for actual heads and their joint paths. No video or pose frames."""
from bisect import bisect_right
from collections import defaultdict
from itertools import groupby

import numpy as np

from . import player_motion as motion
from . import HAND_FIELDS, LINK_SECONDS, MISSING, NO_RIVAL
from .annotation import _margin
from .rim_stroke import plan_playback

def _grade(choices, margins, anchor=False):
    if any(side < 0 for side in choices):
        return MISSING
    if len(set(choices)) != 1:
        return 0
    side = choices[0]
    sign = 1 if side == 0 else -1
    if anchor:
        return 2 * sign
    seen = [m for m in margins if m is not None]
    if not seen:
        return NO_RIVAL
    if any(s != side + 1 or (gap is not None and gap < 1.) for s, gap in seen):
        return 0
    strong = len(seen) == 2 and all(gap is not None and gap >= 2.5 for _, gap in seen)
    return sign * (2 if strong else 1)


def calculate(chart):
    contacts, notes, issues, _ = motion._extract_chart_contacts_and_visuals(chart,include_event_ids=True)
    result = np.full((len(chart.points), len(HAND_FIELDS)), MISSING, np.int8)
    # Partial parsing could remove an obligation that changes both hands.
    if any(issue['kind'] == 'unknown_note' for issue in issues) or not contacts:
        return result
    times = [c.start for c in contacts]
    by_id = {c.id: i for i, c in enumerate(contacts) if not c.segments and not c.is_star_head}
    groups = defaultdict(list)
    for note in notes:
        groups[note['source_event']].append(note)
    tasks = defaultdict(list)
    singles = {}
    for event, rows in sorted(groups.items()):
        if len(rows) > 2 or any(r['contact_id'] not in by_id for r in rows):
            continue
        keys = [str(r.get('pad') or r.get('lane')) for r in rows]
        if len(set(keys)) != len(keys):
            continue
        ordered = [by_id[r['contact_id']] for _, r in sorted(zip(keys, rows), key=lambda x: x[0])]
        for slot, i in enumerate(ordered):
            tasks[i].append((event, slot, i))
        if len(ordered) == 2:
            a, b = sorted(ordered)
            tasks[b].append((event, 3, a))
            continue
        i = ordered[0]
        if event-1 in singles and times[i]-times[singles[event-1]] <= LINK_SECONDS:
            tasks[i].append((event, 2, singles[event-1]))
            if event-2 in singles:
                a, mid = singles[event-2], singles[event-1]
                if (times[mid]-times[a] <= LINK_SECONDS and contacts[a].pad == contacts[i].pad
                        and contacts[mid].pad != contacts[i].pad):
                    tasks[i].append((event, 4, a))
        singles[event] = i
    due = defaultdict(list)
    margins = {}
    # Bounds apply to observation cost, not to the length of a candidate path.
    for i, entries in tasks.items():
        stop = min(i+6, bisect_right(times, times[i]+.35)-1)
        for event, slot, a in entries:
            if i-a > 16:
                continue
            key = (event, slot)
            margins[key] = [None, None]
            due[i].append((key, 0, 0, i-a))
            if stop > i:
                due[stop].append((key, 1, stop-i, i-a))
    def observe(c, candidates, beam, remaining):
        for key, late, depth, earlier in due.get(indices[c.id], ()):
            margins[key][late] = _margin(beam if late else candidates, depth, c, remaining, earlier=earlier)

    indices = {c.id: i for i,c in enumerate(contacts)}
    profile = motion.profile_for('standard')
    raw = motion._plan(contacts, profile, observe=observe)
    paths = [raw]
    for style in ('continuous', 'relay'):
        parts, hands = plan_playback(contacts, profile, 'adaptive', style,
                                     raw_hands=raw if style == 'continuous' else None)
        owners = {c.id: h for c,h in zip(parts,hands) if not c.segments}
        paths.append([owners.get(c.id,-1) for c in contacts])
    groups = [list(g) for _,g in groupby((c for c in contacts if not c.segments), key=lambda c:c.start)]
    anchors = {}
    for phrase in motion._repeated_hold_phrases(contacts, groups, profile):
        if all(all(path[indices[key]] == hand for key,hand in phrase) for path in paths):
            anchors.update(phrase)
    for i, entries in tasks.items():
        for event, slot, a in entries:
            key = event,slot
            if key not in margins:
                continue
            choices = [path[i] for path in paths]
            if i!=a:
                choices = [int(path[i]!=path[a]) if min(path[i],path[a])>=0 else -1 for path in paths]
            result[key] = _grade(choices, margins[key], i==a and contacts[i].id in anchors)
    return result
