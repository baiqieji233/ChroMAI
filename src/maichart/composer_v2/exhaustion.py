"""Explicit continuation policy after physical recovery cannot succeed."""
from copy import deepcopy

from maichart.composer.guard import _event_syntax_error


def resolve_policy(requested,sequence_policy):
    if requested not in ('auto','report','stop','downgrade_or_skip'):raise ValueError('Unknown exhaustion policy')
    if requested=='auto':return 'report' if sequence_policy=='continuous_events_v1' else 'stop'
    if requested in ('report','downgrade_or_skip') and sequence_policy!='continuous_events_v1':
        raise ValueError('Continue-and-report or downgrade-or-skip requires a continuous-event checkpoint; use stop for old window checkpoints')
    return requested


def downgrade_or_skip_conflict(guard,rejected,after):
    """Try a zero-conflict reduced note at the failed beat before skipping it.

    1. Drop conflicting chord children while keeping legal single atoms.
    2. Downgrade a sustained Hold / TouchHold / Slide to an instantaneous Tap / Touch.
    3. If all hands/pads are completely locked at that beat, return a beat skip
       so no hard Muri is ever admitted into the exported chart.
    """
    from maichart.event_timing import EventPoint
    from maichart.composer.event_set_codec import _atoms
    import re
    candidates_seen=set()
    latest_beat=None
    for point,_ in rejected:
        if point.beat<=after:continue
        if latest_beat is None or point.beat>latest_beat:latest_beat=point.beat
        atoms=list(_atoms(point.note_text))
        # 1) Proper single-atom subsets of a chord.
        if len(atoms)>1:
            for atom in atoms:
                key=(point.beat,atom)
                if key in candidates_seen or _event_syntax_error(atom):continue
                candidates_seen.add(key)
                sub=EventPoint(point.beat,atom)
                trial=deepcopy(guard)
                if trial.accepts(sub):
                    trial.commit(sub)
                    return dict(kind='downgraded',point=sub,guard=trial,
                                original=point.note_text,replacement=atom,
                                operation='drop_chord_children')
        # 2) Downgrade sustained note to instantaneous Tap / Touch on its head pad.
        for atom in atoms:
            m_btn=re.match(r'^([1-8])(b?)(x?)(?:h|[-^v<>Vpqszw])',atom)
            m_tch=re.match(r'^([ABDE][1-8]|C)(f?)h',atom)
            simple=None
            if m_btn:simple=m_btn.group(1)+m_btn.group(2)+m_btn.group(3)
            elif m_tch:simple=m_tch.group(1)+m_tch.group(2)
            if simple:
                key=(point.beat,simple)
                if key in candidates_seen or _event_syntax_error(simple):continue
                candidates_seen.add(key)
                sub=EventPoint(point.beat,simple)
                trial=deepcopy(guard)
                if trial.accepts(sub):
                    trial.commit(sub)
                    return dict(kind='downgraded',point=sub,guard=trial,
                                original=point.note_text,replacement=simple,
                                operation='downgrade_sustain_to_pulse')
    if latest_beat is not None:
        return dict(kind='skipped_beat',skip_until_beat=latest_beat)
    return None


def recover_grid_duration(guard, codec, failures, after, end):
    """Try a legal reduction at the first rejected time on this prefix."""
    from maichart.event_timing import EventPoint
    from .generate import _check_window

    failures = [failure for failure in failures if after < failure.beat < end]
    if not failures:
        return None
    beat = min(failure.beat for failure in failures)
    seen = set()
    for failure in failures:
        if failure.beat != beat:
            continue
        completed = failure.completed_atoms
        candidates = [(completed, 'drop_unfinished_sustain')]
        if failure.pulse is not None:
            candidates.insert(0, ((*completed, failure.pulse), 'downgrade_unfinished_sustain'))
        for atoms, operation in candidates:
            text = '/'.join(atoms)
            if not text or text in seen:
                continue
            seen.add(text)
            point = EventPoint(beat, text)
            try:
                encoded = codec.encode_points([point])
            except ValueError:
                continue
            trial, _ = _check_window(guard, [point])
            if trial is not None:
                return dict(kind='downgraded', point=point, guard=trial, encoded=encoded,
                            original=None, replacement=text, operation=operation,
                            duration_failure=failure.diagnostic())
    return dict(kind='skipped_beat', skip_until_beat=beat, reason='exhaustion_skipped_invalid_duration')


def admit_physical_conflict(guard,rejected,after):
    """Keep the first model proposal that can be exported and advances time.

    Native physical errors may be bypassed, while syntax, unsupported native
    parsing, negative/expired time and action-past-audio errors remain fatal.
    The guard still receives the actual event; occupied pads never disappear.
    """
    for point,details in rejected:
        if not details or point.beat<=after or _event_syntax_error(point.note_text):continue
        second,finish,_,errors=guard._lifetime(point)
        if errors or second<0 or finish>guard.audio_end+1e-8:continue
        trial=deepcopy(guard);trial.commit(point)
        return point,trial,details
    return None


def conflict_assessment(points,tempo,first,guard):
    from maichart.generation.post_process import detect_chart_conflicts
    try:
        result=detect_chart_conflicts(points,tempo,first,guard.config)
        issues=[issue.to_dict() for issue in result.conflicts_detected]
        enforced=[issue.to_dict() for issue in result.conflicts_detected if guard.enforced(issue)]
        return dict(status='complete',enforced_conflict_count=len(enforced),
            enforced_conflicts=enforced,all_conflicts=issues,coverage=result.coverage)
    except Exception as error:
        # This optional final assessment must not discard an otherwise
        # exportable chart explicitly requested in report mode.
        return dict(status='failed',error=repr(error),enforced_conflict_count=None)
