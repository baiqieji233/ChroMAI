"""Bounded physical previews; cached model contexts are proposals, not posteriors.

This module never commits to the real chart and never consumes its RNG.  It
compares at most two current events at one proposed future time.  A replacement
must keep the original event's type counts and number of heads.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math

import torch

from maichart.event_timing import EventPoint
from .events import event_metadata

LOOKAHEAD_POLICY = 'current_supported_retained_mass_v3'
MIN_CURRENT_RELATIVE_SUPPORT = .25
MIN_JOINT_PREVIEW_GAIN = math.log(1.10)
GAP_BEATS = (.125, .25, .5, 1., 2., 4., 8.)


def clone_generator(generator):
    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state())
    return clone


def short_gap_candidate(candidates, choice, gap_logits, *, allowed=None):
    """One optional proposal from a confident short-gap prediction.

    This time is never forced into the chart.  The next ordinary pointer draw
    still decides whether and where an event will actually be written.
    """
    if gap_logits is None:
        return None
    probabilities = gap_logits.detach().float().softmax(-1).cpu()
    if probabilities.shape != (len(GAP_BEATS),) or not bool(torch.isfinite(probabilities).all()):
        return None
    if float(probabilities[:2].sum()) < .6:
        return None
    gap = GAP_BEATS[int(probabilities[:2].argmax())]
    origin = candidates[choice]
    future = [i for i, beat in enumerate(candidates)
              if Fraction(0) < beat-origin <= Fraction(1, 4)
              and (allowed is None or bool(allowed[i]))]
    return min(future, key=lambda i: abs(math.log(float(candidates[i]-origin)/gap))) if future else None


def event_intent(event):
    meta = event_metadata(event.note_text)
    return meta.type_counts, len(meta.heads)


@dataclass(frozen=True)
class FutureAssessment:
    event: EventPoint | None
    # Fraction of ONE shared pre-commit first-slot distribution that survives
    # this branch's updated physical mask. Not a full future-chart likelihood.
    retained_probability: float


@dataclass(frozen=True)
class LookaheadResult:
    event: EventPoint
    reason: str
    future_trials: int = 0
    alternate_draws: int = 0
    switched: bool = False
    baseline_retained_probability: float | None = None
    alternative_retained_probability: float | None = None
    current_log_probability_ratio: float | None = None


def bounded_physical_lookahead(baseline, *, future_beat, guard, generator,
                              alternative_factory, future_factory,
                              strength=1., protect_reported_hard=False,current_probability_factory=None):
    """Use identical private RNG states for the two future proposals.

    Factories receive private generators. ``future_factory`` receives a guard
    that has already committed the proposed current event, so it can reconstruct
    occupancy and cooldowns before sampling. Only the returned current event may
    later be committed by the caller. Ties always retain the original sample.
    """
    strength = float(strength)
    if not math.isfinite(strength) or not 0. <= strength <= 1.:
        raise ValueError('quality_lookahead_strength must be in [0,1]')
    if strength == 0. or future_beat is None or future_beat <= baseline.beat:
        return LookaheadResult(baseline, 'disabled_or_no_future')
    local_rng = clone_generator(generator)
    if strength < 1. and float(torch.rand((), device=generator.device, generator=local_rng)) >= strength:
        return LookaheadResult(baseline, 'strength_skipped')
    future_rng = clone_generator(local_rng)

    def trial(current):
        branch = deepcopy(guard)
        if not branch.accepts(current, protect_reported_hard=protect_reported_hard):
            return False, False, None
        branch.commit(current)
        future = future_factory(branch, current, future_beat, clone_generator(future_rng))
        mass=None
        if isinstance(future,FutureAssessment):
            mass=float(future.retained_probability)
            future=future.event
        if future is None:
            return True, False, mass
        # One position only: callbacks cannot silently broaden the search.
        if future.beat != future_beat:
            raise ValueError('Lookahead future proposal changed the requested position')
        return True, branch.accepts(future, protect_reported_hard=protect_reported_hard), mass

    baseline_valid, baseline_future, baseline_mass = trial(baseline)
    if not baseline_valid:
        return LookaheadResult(baseline, 'baseline_not_accepted')
    metrics={'baseline_retained_probability':baseline_mass}
    if baseline_future and (baseline_mass is None or baseline_mass>=1.-1e-7):
        return LookaheadResult(baseline, 'baseline_future_legal', future_trials=1,**metrics)
    alternative = alternative_factory(local_rng)
    if alternative is None or alternative.note_text == baseline.note_text:
        return LookaheadResult(baseline, 'no_alternative', future_trials=1, alternate_draws=1,**metrics)
    if alternative.beat != baseline.beat or event_intent(alternative) != event_intent(baseline):
        return LookaheadResult(baseline, 'different_event_intent', future_trials=1, alternate_draws=1,**metrics)
    current_ratio=None
    if current_probability_factory is not None:
        baseline_logp=float(current_probability_factory(baseline))
        alternative_logp=float(current_probability_factory(alternative))
        current_ratio=alternative_logp-baseline_logp
        metrics['current_log_probability_ratio']=current_ratio if math.isfinite(current_ratio) else None
        if not math.isfinite(current_ratio) or current_ratio<math.log(MIN_CURRENT_RELATIVE_SUPPORT):
            return LookaheadResult(baseline,'insufficient_current_support',future_trials=1,alternate_draws=1,**metrics)
    alternative_valid, alternative_future, alternative_mass = trial(alternative)
    metrics['alternative_retained_probability']=alternative_mass
    better_mass=(baseline_mass is not None and alternative_mass is not None
                 and alternative_mass>baseline_mass+1e-7)
    if baseline_future and current_ratio is not None and better_mass:
        joint_gain=current_ratio+math.log(max(alternative_mass,1e-12)/max(baseline_mass,1e-12))
        better_mass=joint_gain>=MIN_JOINT_PREVIEW_GAIN
    if alternative_valid and alternative_future and (not baseline_future or better_mass):
        return LookaheadResult(alternative, 'alternative_preserves_more_mass' if baseline_future else 'alternative_future_legal',
                               future_trials=2,alternate_draws=1,switched=True,**metrics)
    return LookaheadResult(baseline, 'alternative_not_better',
                           future_trials=1+int(alternative_valid), alternate_draws=1,**metrics)
