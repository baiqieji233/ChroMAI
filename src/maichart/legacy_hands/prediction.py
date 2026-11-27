"""Extract the assignments visible in full legacy playback without exporting poses."""
from collections import defaultdict

from . import player_motion as motion
from .rim_stroke import plan_playback


def predict_hands(chart, *, profile_id='standard', stance='adaptive', sweep_style='continuous'):
    """Use the legacy beam and the same full-song Slide candidate selection."""
    profile = motion.profile_for(profile_id)
    contacts, notes, issues, end = motion._extract_chart_contacts_and_visuals(chart)
    duration = max(0., end, max((c.end for c in contacts), default=0.))
    contacts, hands = plan_playback(contacts, profile, stance, sweep_style)
    times, samples, _ = motion._playback_samples(contacts, hands, profile, duration)
    samples, _ = motion._select_gesture(times, samples, contacts, hands, profile, {}, hands_only=True)
    owners = defaultdict(list)
    for second, pair in zip(times, samples):
        if second < 0.:
            continue
        sides = {}
        for side, sample in enumerate(pair, 1):
            for contact in sample[2]:
                key = contact.source_id
                sides[key] = sides.get(key, 0) | side
        for key, side in sides.items():
            if side == 3:  # Concurrent attribution is ambiguous in the full playback too.
                continue
            changes = owners[key]
            if not changes or changes[-1][1] != side:
                changes.append([second, side])
    records = note_hand_records(notes, owners)
    return dict(schema='legacy-player-hands-v1', sweep_style=sweep_style, noteHands=records, authored=len(notes),
                unknown_notes=sum(issue['kind']=='unknown_note' for issue in issues))


def note_hand_records(notes, owners):
    """Serialize the owner of each head, and timestamped ownership of each route."""
    records = []
    for note in notes:
        changes = owners[note['contact_id']]
        if not changes:
            continue
        kind = note['kind']
        pad = note.get('pad') or 'A' + str(note['lane'])
        if kind in ('slide', 'wifi'):
            records.append(dict(contact_id=note['contact_id'], kind=kind, pad=pad,
                                start=note['onset'], trace=note['trace'], end=note['end'],
                                endPad='A'+str(note['end_lane']), track=changes, centerOnly=kind=='wifi'))
        elif len(changes) == 1:
            records.append(dict(contact_id=note['contact_id'], kind=kind, pad=pad,
                                start=note['start'], end=note['end'], head=changes[0][1]))
    return records
