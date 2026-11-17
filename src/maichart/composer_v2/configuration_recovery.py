"""Compare continuations when generated taps dominate the musical context."""

from fractions import Fraction
import re

import numpy as np


_SINGLE_TAP = re.compile(r'([1-8])[bx]*$')
_MIN_TAPS = 16
_MIN_BEATS = 8
_RETRY_BEATS = 4


def tap_key(point):
    match = _SINGLE_TAP.fullmatch(point.note_text)
    return match[1] if match else None


def _music_change(dataset, song, beat):
    """Compare neighboring two-beat passages against this song's variation."""
    novelty = getattr(song, '_configuration_novelty', None)
    if novelty is None:
        profile = dataset._profile(song)
        semantic = profile.semantic
        if len(semantic) < 5:
            return None
        pair = semantic[:-1] + semantic[1:]
        norm = np.linalg.norm(pair, axis=1)
        available = (norm[:-2] > 1e-6) & (norm[2:] > 1e-6)
        unit = pair / np.maximum(norm[:, None], 1e-6)
        distance = np.clip(1. - (unit[:-2] * unit[2:]).sum(1), 0., 2.)
        typical = float(np.quantile(distance[available], .7)) if available.any() else 2.
        novelty = (distance, available, max(.18, typical))
        song._configuration_novelty = novelty
    distance, available, threshold = novelty
    i = int(beat) - 2
    if 0 <= i < len(distance) and available[i] and distance[i] > threshold:
        return float(distance[i])
    return None


class ConfigurationRecovery:
    def __init__(self):
        self.last_trial = Fraction(-1)
        self.attempts = 0
        self.accepted = 0
        self.recent = []

    def opportunity(self, points, proposal, dataset, song):
        beat = proposal.beat
        key = tap_key(proposal)
        if key is None or len(points) < _MIN_TAPS or beat < self.last_trial + _RETRY_BEATS:
            return None
        last = beat
        repeated = []
        for point in reversed(points):
            if last - point.beat > 1 or tap_key(point) != key:
                break
            repeated.append(point.beat)
            last = point.beat
            if len(repeated) >= _MIN_TAPS and beat - last >= _MIN_BEATS:
                break
        if len(repeated) < _MIN_TAPS or beat - last < _MIN_BEATS:
            return None
        change = _music_change(dataset, song, beat)
        if change is None:
            return None
        # Retain the latest strike, unrelated history, and all true sustains.
        # These omissions apply only to the extra candidate's symbolic context.
        for point in points[-len(repeated)-1::-1]:
            if last - point.beat > 1 or tap_key(point) != key:
                break
            repeated.append(point.beat)
            last = point.beat
        self.last_trial = beat
        self.attempts += 1
        return key, frozenset(repeated[1:]), change

    def record(self, evidence):
        self.accepted += evidence['used_alternative']
        self.recent = [*self.recent[-15:], evidence]

    def report(self):
        return dict(attempts=self.attempts, accepted=self.accepted, recent=self.recent)
