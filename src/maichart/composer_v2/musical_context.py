"""Training views that reduce dependence on older symbolic history."""

from fractions import Fraction

import numpy as np

from .defaults import MUSICAL_CONTEXT_CHOICES as POLICIES, MUSICAL_CONTEXT_POLICY as POLICY

RECENT_BEATS = (2, 4)


def recent_history_beats(*, seed, epoch, index, probability, history_beats):
    """Choose a reproducible history span for an existing authored window."""
    if probability <= 0 or history_beats <= RECENT_BEATS[0]:
        return None
    rng = np.random.default_rng((seed % (1 << 64), epoch, index, 19))
    if rng.random() >= probability:
        return None
    return min(history_beats, RECENT_BEATS[(index + epoch) % len(RECENT_BEATS)])


def shorten_history(history, cursor, recent_beats, active_beats=()):
    """Keep the most recent phrase context and sources of ongoing sustains."""
    if not history:
        return history
    boundary = min(history[-1].beat, cursor - max(Fraction(recent_beats), RECENT_BEATS[0]))
    return [point for point in history if point.beat >= boundary or point.beat in active_beats]
