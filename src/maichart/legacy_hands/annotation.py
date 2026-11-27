"""Conservative left/right annotations, separate from the playable candidate."""
from bisect import bisect_right
from collections import defaultdict
from itertools import groupby

from . import player_motion as motion
from .prediction import predict_hands


def _margin(states, depth, contact, remaining, *, earlier=0):
    """Best observed left versus right paths under the solver's actual ordering."""
    ranks = {}
    for state in states:
        node = state
        for _ in range(depth):
            node = node.parent
        choice = node.choice
        if choice < 0:
            continue
        if earlier:
            for _ in range(earlier):
                node = node.parent
            if node.choice < 0:
                continue
            # 0 means same, 1 means different, on this joint trajectory.
            choice = int(choice != node.choice)
        if choice in ranks:
            continue
        pending = max(0, contact.group_required-state.group_hits-remaining) if remaining else 0
        ranks[choice] = (state.misses+pending, state.chord_compromises, state.rapid_jumps, state.effort)
        if len(ranks)==2:
            break
    if len(ranks)<2:
        return None
    left, right = ranks[0], ranks[1]
    if left[:3] != right[:3]:
        # A difference in heuristic feasibility is directional evidence, but
        # does not estimate how likely a human is to choose either hand.
        return (1 if left<right else 2), None
    return (1 if left<right else 2), abs(left[3]-right[3])


def annotate(chart, continuous=None, relay=None):
    """Ordinal support only. An absent or pruned alternative is not certainty."""
    if continuous is None:
        continuous = predict_hands(chart, sweep_style='continuous')['noteHands']
    if relay is None:
        relay = predict_hands(chart, sweep_style='relay')['noteHands']
    models = [{r['contact_id']:r for r in source} for source in (continuous, relay)]
    contacts, notes, _, _ = motion._extract_chart_contacts_and_visuals(chart)
    profile = motion.profile_for('standard')
    groups = [list(g) for _,g in groupby((c for c in contacts if not c.segments),key=lambda c:c.start)]
    anchors = {}
    for phrase in motion._repeated_hold_phrases(contacts, groups, profile):
        if all(model.get(key, {}).get('head') == hand+1 for model in models for key,hand in phrase):
            anchors.update(phrase)
    # Inspect actual rival paths at the decision and again after nearby events.
    # Keep this additional cheap assignment pass out of normal pose generation.
    times = [c.start for c in contacts]
    ends = defaultdict(list)
    pieces = defaultdict(list)
    for i,c in enumerate(contacts):
        pieces[c.source_id].append(c.id)
        stop = min(i+6, bisect_right(times, c.start+.35)-1)
        if stop>i:
            ends[stop].append(i)
    immediate, later = {}, {}

    def observe(contact, candidates, beam, remaining):
        i = len(immediate)
        immediate[contact.id] = _margin(candidates, 0, contact, remaining)
        for j in ends.get(i, ()):
            later[contacts[j].id] = _margin(beam, i-j, contact, remaining)

    selected = motion._plan(contacts, profile, observe=observe)
    raw_sides = defaultdict(set)
    for c, hand in zip(contacts,selected):
        raw_sides[c.source_id].add(hand+1)
    labels = []
    for note in notes:
        key = note['contact_id']
        rows = [model.get(key) for model in models]
        head = note['kind'] not in ('slide','wifi')
        def owners(row):
            if not row:
                return set()
            return {row['head']} if head else {h for _,h in row['track']}
        a,b = map(owners, rows)
        base = dict(contact_id=key, kind=note['kind'], pad=note.get('pad') or 'A'+str(note.get('lane')),
                    start=note.get('onset',note.get('start')), end=note['end'])
        if not head:
            base.update(trace=note['trace'],endPad='A'+str(note['end_lane']))
        owner, grade, reason = None, 'uncertain', 'missing'
        if a and b:
            if a!=b:
                reason = 'styles_differ'
            elif len(a)!=1 or note['kind']=='wifi':
                reason = 'partial_route'
            elif raw_sides[key] != a:
                reason = 'assignment_changed'
            else:
                side = next(iter(a))
                if key in anchors and side==anchors[key]+1:
                    owner, grade, reason = side,'supported','hold_anchor'
                else:
                    levels = []
                    for part in pieces[key]:
                        first, look = immediate.get(part), later.get(part)
                        seen = [r for r in (first,look) if r is not None]
                        if not seen or any(h!=side or (gap is not None and gap<1.) for h,gap in seen):
                            levels.append(0)
                        elif len(seen)==2 and all(gap is not None and gap>=2.5 for _,gap in seen):
                            levels.append(2)
                        else:
                            levels.append(1)
                    level = min(levels, default=0)
                    if level:
                        owner, grade, reason = side, ('supported' if level==2 else 'lean'), 'rival_paths'
                    else:
                        reason = 'insufficient_margin'
        labels.append(dict(base, hand=owner, confidence=grade, reason=reason))
    return dict(schema='chromai-hand-labels-v1', scale='ordinal_uncalibrated', labels=labels)
