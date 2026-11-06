"""Explicit continuation policy after physical recovery cannot succeed."""
from copy import deepcopy

from maichart.composer.guard import _event_syntax_error


def resolve_policy(requested,sequence_policy):
    if requested not in ('auto','report','stop'):raise ValueError('Unknown exhaustion policy')
    if requested=='auto':return 'report' if sequence_policy=='continuous_events_v1' else 'stop'
    if requested=='report' and sequence_policy!='continuous_events_v1':
        raise ValueError('Continue-and-report requires a continuous-event checkpoint; use stop for old window checkpoints')
    return requested


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
