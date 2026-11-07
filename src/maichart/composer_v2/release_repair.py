"""Transactional repair of Hold occupancy against real native failures.

This is a fallback after normal complete-event proposals have failed. It does
not move onsets, invent continuations, or cap otherwise valid long sustains.
Every edit must make an actually proposed continuation feasible, together with
the entire retained suffix. Training and the native judge are unchanged.
"""
from fractions import Fraction
import math
import re

from maichart.event_timing import EventPoint
from maichart.composer.event_set_codec import _atoms
from maichart.generation.muri.model import RELEASE, TPS
from .codec import atom_components
from .generate import _new_guard, _check_window, _atom_heads


# A same-pad rearticulation needs one inactive native frame between presses.
# Merely ending the old press on the new press's tick produces no rising edge.
_CLEARANCE = Fraction(str(RELEASE)).limit_denominator(1000000) + 1/Fraction(str(TPS))


def _elapsed_at_head_bpm(clock, start, end):
    """Integrate the tempo map exactly, then express seconds in the head BPM.

    Native Hold [n:k] durations use the BPM at the head even across a later
    tempo change. Subtracting chart beats alone would change that meaning.
    """
    cuts = [start, *(b for b in clock.beats if start < b < end), end]
    seconds = sum(((b-a)*60/Fraction(str(clock.bpm(a))) for a,b in zip(cuts,cuts[1:])), Fraction(0))
    return seconds * Fraction(str(clock.bpm(start))) / 60


def _grid_hold_beats(atom):
    parts = atom_components(atom)
    if parts[0] != 'N:hold' or parts.count('D:B') != 1 or 'D:R' in parts:
        return None
    begin = parts.index('D:B') + 1
    end = parts.index('[END_DURATION]', begin)
    return Fraction(''.join(p[2:] for p in parts[begin:end]))


def _hold_tails(points, clock):
    """Index once per dead end; native issue anchors are not all participants."""
    tails = []
    for index, source in enumerate(points):
        for atom_index, atom in enumerate(_atoms(source.note_text)):
            beats = _grid_hold_beats(atom)
            if beats is not None and beats > 0:
                scale = 60 / clock.bpm(source.beat)
                tails.append((index, atom_index, beats, clock.at(source.beat)+float(beats)*scale, scale))
    return tails


def _tail_variants(points, tails, witness, clock, ticks_per_beat, reported_sources):
    quantum = Fraction(1, ticks_per_beat)
    second = clock.at(witness.beat)
    eligible = {}
    steps = 0
    # The cheap seconds check avoids exact tempo integration for distant tails.
    for index, atom_index, beats, finish, scale in tails:
        if not -float(_CLEARANCE)-1e-9 < finish-second <= float(quantum)*scale+1e-9:
            continue
        source = points[index]
        elapsed = _elapsed_at_head_bpm(clock, source.beat, witness.beat)
        release = _CLEARANCE * Fraction(str(clock.bpm(source.beat))) / 60
        # Only the release-boundary neighborhood; genuinely ongoing long
        # Holds still use ordinary source-event withdrawal and regeneration.
        if -release < beats-elapsed <= quantum:
            eligible[(index, atom_index)] = beats
            steps = max(steps, math.ceil((beats-elapsed+release)/quantum))
    if not eligible:
        return
    # Preserve the reported-source ordering when it is useful, but include
    # other active tails. A MultiTouch anchor can be the *long* Hold while a
    # different, just-ending Hold is what needs a release margin.
    order = sorted(eligible, key=lambda key:(points[key[0]].beat not in reported_sources, -key[0], key[1]))
    for step in range(1, steps+1):
        changed = {}
        for key in order:
            beats = eligible[key]
            shorter = beats-step*quantum
            if shorter > 0:
                block = f'[{4*shorter.denominator}:{shorter.numerator}]'
                index, atom_index = key
                atom = _atoms(points[index].note_text)[atom_index]
                changed[key] = re.sub(r'\[[^\]]*\]', block, atom, count=1)
        # Prefer an individual tail edit; a double Hold may require both.
        # Joint edits may also involve Holds that began at different events.
        groups = [(key,) for key in changed]
        if len(changed)>1:
            groups.append(tuple(changed))
        for group in groups:
            edited = {}
            for index, atom_index in group:
                edited.setdefault(index, list(_atoms(points[index].note_text)))
                edited[index][atom_index] = changed[(index, atom_index)]
            yield {i:EventPoint(points[i].beat, '/'.join(atoms)) for i,atoms in edited.items()}, step, group


def _rearticulation_variants(points, tails, witness, details, clock, ticks_per_beat):
    """Revise an ongoing Hold only for an actually overlapping new same-pad head.

    Jump directly to the latest plausible end before that head. A 55-beat Hold
    does not cause thousands of one-tick probes. Native checking decides how
    much release clearance is really required.
    """
    heads=set(_atom_heads(witness.note_text))
    implicated=set()
    for issue in details:
        if issue.get('kind')!='miacode_Overlap':continue
        for note in issue.get('notes',[]):
            if note.get('raw'):implicated.update(_atom_heads(note['raw']))
    heads &= implicated
    if not heads:return
    quantum=Fraction(1,ticks_per_beat);second=clock.at(witness.beat)
    eligible={}
    for index,atom_index,beats,finish,scale in tails:
        # Near-end cases were already searched by the release-tail pass.
        if finish-second<=float(quantum)*scale+1e-9:continue
        atom=_atoms(points[index].note_text)[atom_index]
        if not heads.intersection(_atom_heads(atom)):continue
        elapsed=_elapsed_at_head_bpm(clock,points[index].beat,witness.beat)
        clearance=_CLEARANCE*Fraction(str(clock.bpm(points[index].beat)))/60
        start=max(1,math.ceil((beats-elapsed)/quantum))
        stop=math.ceil((beats-elapsed+clearance)/quantum)
        eligible[(index,atom_index)]=(beats,start,stop)
    if not eligible:return
    for offset in range(max(stop-start for _,start,stop in eligible.values())+1):
        changed={};reductions={}
        for key,(beats,start,stop) in eligible.items():
            steps=min(start+offset,stop);shorter=beats-steps*quantum
            if shorter<=0:continue
            index,atom_index=key;atom=_atoms(points[index].note_text)[atom_index]
            changed[key]=re.sub(r'\[[^\]]*\]',f'[{4*shorter.denominator}:{shorter.numerator}]',atom,count=1)
            reductions[key]=steps*quantum
        groups=[(key,) for key in changed]
        if len(changed)>1:groups.append(tuple(changed))
        for group in groups:
            edited={}
            for index,atom_index in group:
                edited.setdefault(index,list(_atoms(points[index].note_text)))
                edited[index][atom_index]=changed[(index,atom_index)]
            yield ({i:EventPoint(points[i].beat,'/'.join(atoms)) for i,atoms in edited.items()},
                {key:reductions[key] for key in group},group,'same_pad_rearticulation')


def _hold_variants(points,tails,witness,details,clock,codec,sources):
    for replacements,steps,keys in _tail_variants(points,tails,witness,clock,codec.ticks_per_beat,sources):
        yield replacements,{key:Fraction(steps,codec.ticks_per_beat) for key in keys},keys,'release_tail'
    yield from _rearticulation_variants(points,tails,witness,details,clock,codec.ticks_per_beat)


def repair_hold_conflict(points, rejected, tempo, clock, duration, codec, *, attempted=None, max_checks=32, audit=None,guard_options=None):
    """Return edited prefix, unchanged witness, and its committed guard, or None.

    `rejected` contains (point, native conflict details). Trial identity includes
    the whole causal prefix and witness, so deduplication cannot ban an action
    in a different context. At most max_checks native suffix checks are made.
    Nothing in the caller's points/guard/cache is modified on failure.
    """
    attempted = set() if attempted is None else attempted
    audit = {} if audit is None else audit
    audit.update(native_checks=0, duplicate_trials=0)
    prefix_key = tuple((p.beat, p.note_text) for p in points)
    bases = {}
    tails = _hold_tails(points, clock)
    seen_witnesses = set()
    for witness, details in rejected:
        if not details or not points or witness.beat <= points[-1].beat:
            continue
        sources = {Fraction(b) for issue in details for b in issue['note_beats']}
        witness_key = (witness.beat, witness.note_text, tuple(sorted(sources)))
        if witness_key in seen_witnesses:
            continue
        seen_witnesses.add(witness_key)
        for replacements, reductions, atom_keys, mode in _hold_variants(points,tails,witness,details,clock,codec,sources):
            index = min(replacements)
            key = (prefix_key, witness.beat, witness.note_text,
                   tuple((i,p.note_text) for i,p in sorted(replacements.items())))
            if key in attempted:
                audit['duplicate_trials'] += 1
                continue
            if audit['native_checks'] >= max_checks:
                return None
            attempted.add(key)
            # Never commit a rewrite the checkpoint cannot read next time.
            codec.encode_points(list(replacements.values()))
            if index not in bases:
                base = _new_guard(tempo, clock.seconds[0], duration,**({'guard_options':guard_options} if guard_options is not None else {}))
                # The unchanged prefix was already committed, possibly under
                # continue-and-report. Validate the edited suffix below.
                for p in points[:index]:
                    base.commit(p)
                bases[index] = base
            audit['native_checks'] += 1
            suffix = [replacements.get(i,points[i]) for i in range(index,len(points))]
            trial, _ = _check_window(bases[index], [*suffix, witness])
            if trial is None:
                continue
            changes=[]
            for i,p in sorted(replacements.items()):
                deltas={a:reductions[(j,a)] for j,a in atom_keys if i==j}
                changes.append(dict(source_beat=str(points[i].beat),original_note_text=points[i].note_text,
                    replacement_note_text=p.note_text,shortened_atom_indices=list(deltas),
                    duration_reduction_beats=str(max(deltas.values())),
                    duration_reduction_seconds=float(max(deltas.values()))*60/clock.bpm(points[i].beat),
                    atom_duration_reductions={str(a):str(delta) for a,delta in deltas.items()}))
            report = dict(policy='native_hold_conflict_repair_v3', mode=mode, **changes[0], changes=changes,
                reported_source_beats=[str(b) for b in sorted(sources)],
                source_selection='active_release_boundary_counterfactual',
                witness=dict(beat=str(witness.beat),note_text=witness.note_text),
                retained_suffix_events=len(points)-index-len(replacements), **audit)
            return [*points[:index], *suffix], witness, trial, report
    return None


# Existing callers share the same transaction and native-checking contract.
repair_hold_release=repair_hold_conflict
