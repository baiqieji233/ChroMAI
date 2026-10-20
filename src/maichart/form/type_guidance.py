"""Bounded inference-time type guidance over the existing note grammar."""
from __future__ import annotations

from collections import Counter, OrderedDict
from functools import lru_cache
from fractions import Fraction
import math

import numpy as np

from maichart.event_timing import seconds_at_beat
from maichart.generation.conditions import GENERATION_DEFAULTS
from maichart.generation.event_cli import _NoteGrammar
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint


HEAD_TYPES = ('tap', 'slide', 'hold', 'touch', 'touch_hold')
PLANNER_TYPES = ('tap', 'hold', 'slide', 'touch')
MAX_POTENTIAL = 4.0
PSEUDO_HEADS = 8.0
CACHE_BYTES = 8*1024**2
CACHE_ENTRIES = 512
LANES = 8


def validate_strength(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 3:
        raise ValueError(f'{name} must be finite and between 0 and 3')
    return value


def exclusive_preferences(style):
    # Displayed Hold and Touch shares both include TouchHold heads.
    raw = np.asarray([style[f'{name}_bias'] for name in HEAD_TYPES[:4]], dtype=np.float64)
    overlap = min(max(float(raw.sum()) - 1.0, 0.0), raw[2], raw[3])
    values = np.asarray([raw[0], raw[1], raw[2]-overlap, raw[3]-overlap, overlap])
    return values / values.sum() if values.sum() else np.full(5, 0.2)


@lru_cache(maxsize=65536)
def _counts(text):
    parsed = parse_timing_point_notes(RawTimingPoint(
        index=0, raw=text, note_text=text, beat=Fraction(0), tick=0,
        time_sec=0.0, division=4, bpm=120.0))
    if parsed.unknown_tokens or not parsed.notes:
        return None
    counts = Counter(note.note_type for note in parsed.notes)
    return np.asarray([counts[name] for name in HEAD_TYPES], dtype=np.float64)


@lru_cache(maxsize=65536)
def _break_count(text):
    parsed = parse_timing_point_notes(RawTimingPoint(
        index=0, raw=text, note_text=text, beat=Fraction(0), tick=0,
        time_sec=0.0, division=4, bpm=120.0))
    return sum(bool(note.is_break) for note in parsed.notes) if not parsed.unknown_tokens else 0


@lru_cache(maxsize=65536)
def _ex_count(text):
    """EX modifiers outside duration brackets ('x' is not used by any shape)."""
    depth, count = 0, 0
    for char in str(text):
        if char == '[':
            depth += 1
        elif char == ']':
            depth = max(0, depth - 1)
        elif char == 'x' and not depth:
            count += 1
    return count


def _primary(counts):
    # Match blueprint supervision: Slide > Hold > Touch/TouchHold > Tap.
    return np.where(counts[..., 1] > 0, 2, np.where(
        counts[..., 2] > 0, 1, np.where(counts[..., 3:5].sum(-1) > 0, 3, 0)))


def _probabilities(cue):
    values = cue.get('type_probabilities') if cue else None
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError('Blueprint type_probabilities must contain four finite nonnegative values')
    scaled = values / values.max()
    return scaled / scaled.sum()


class TypeGuidance:
    """Keep head-share feedback across resource passes and musical regions."""

    def __init__(self, atoms, condition, cues, *, planner_strength=1.0, style_strength=1.0,
                 local_chord_strength=0.0, physical_strength=0.0, section_profile=None, section_strength=0.0,
                 plan_intent_strength=0.0, max_star_slides=None):
        self.planner_strength = validate_strength(planner_strength, 'planner_type_strength')
        self.style_strength = validate_strength(style_strength, 'style_balance_strength')
        self.local_chord_strength = validate_strength(local_chord_strength, 'local_chord_strength')
        self.section_strength = validate_strength(section_strength, 'section_type_strength')
        # The decoder's own next-event plan (chord/slide/hold/touch/density
        # [/break/ex]) gates how hard style feedback may push. It can only
        # weaken a push or a suppression, never add a new force: share
        # feedback acts where the model already finds a type plausible.
        self.plan_intent_strength = validate_strength(plan_intent_strength, 'plan_intent_strength')
        if self.plan_intent_strength > 1:
            raise ValueError('plan_intent_strength must be between 0 and 1')
        self.intent_sum = None
        self.intent_events = 0
        self._pending_intent = None
        self.gate_sum = np.zeros(7, dtype=np.float64)
        self.gate_events = 0
        # Callable second -> five positive head-type multipliers (mean ~1).
        self.section_profile = section_profile if self.section_strength else None
        self.section_events = 0
        self.section_multiplier_sum = np.zeros(5, dtype=np.float64)
        self._condition = condition
        from maichart.form.physical_guard import validate_physical_strength
        self.physical_strength = validate_physical_strength(physical_strength)
        self.grammar = _NoteGrammar.from_atoms(atoms, max_star_slides=max_star_slides)
        if self.grammar is None:
            raise ValueError('Type guidance requires a nonempty checkpoint atom catalog')
        self.cues = {Fraction(cue['beat']): cue for cue in cues}
        self.local_chords = None
        if self.local_chord_strength:
            from maichart.form.local_load import LocalChordGuard
            self.local_chords = LocalChordGuard(list(self.cues.values()),
                target_chord_ratio=condition.style['chord_bias'], strength=self.local_chord_strength)
        # Causal physical feasibility: holds/slides occupy their buttons, and a
        # hand must be able to travel between successive events. Both facts are
        # already known from the committed prefix alone.
        self.physical = None
        if self.physical_strength:
            from maichart.form.physical_guard import PhysicalFeasibilityGuard
            self.physical = PhysicalFeasibilityGuard(self.physical_strength)
        probabilities = [_probabilities(cue) for cue in self.cues.values()] if self.planner_strength else []
        probabilities = [value for value in probabilities if value is not None]
        # Use event-class probabilities, not head fractions, as the reference.
        # A uniform component keeps a constant confident cue from cancelling.
        self.planner_reference = (0.8*np.mean(probabilities, axis=0)+0.2/4
                                  if probabilities else np.full(4, 0.25))
        self.target = exclusive_preferences(condition.style)
        self.requested = {name: condition.style[f'{name}_bias'] for name in HEAD_TYPES[:4]}
        self.counts = np.zeros(5, dtype=np.float64)
        self.events = 0
        self.chord_events = 0
        self.break_heads = 0
        self.recent_chord_run = 0
        self.recent_break_run = 0
        self.bar_phase_history = OrderedDict()
        self.prior = exclusive_preferences(GENERATION_DEFAULTS)
        self.atom_counts = {}
        self.prefix_indices = {'': []}
        self.atoms = []
        for atom in sorted(self.grammar.atoms):
            counts = _counts(atom)
            if counts is None:
                continue
            index = len(self.atoms)
            self.atoms.append(atom)
            self.atom_counts[atom] = counts
            self.prefix_indices[''].append(index)
            for length in range(1, len(atom)+1):
                self.prefix_indices.setdefault(atom[:length], []).append(index)
        self.matrix = np.asarray([self.atom_counts[atom] for atom in self.atoms]).reshape(-1, 5)
        # Hand lanes per atom, for the causal physical guard. Positions are the
        # strings "1".."8" for taps and slide heads and "A4"/"C" for touch, so
        # only the former become lanes; a Touch is not a hand on a button.
        from maichart.form.physical_guard import lane_mask
        self.atom_index = {atom: index for index, atom in enumerate(self.atoms)}
        self.lane_matrix = np.asarray(
            [lane_mask(self.grammar.heads.get(atom, ())) for atom in self.atoms], dtype=bool
        ).reshape(-1, LANES)
        self._compatible_indices = OrderedDict()
        self._cache_bytes = 0
        self.cue_events = 0
        self.argmax_counts = Counter()
        self.argmax_matches = 0
        self.guided_tokens = 0
        self.max_abs_bias = 0.0

    def _intent_gates(self, intent):
        """Seven attenuation factors in [0, 1]: five head types, chord, Break.

        Each channel is compared with the model's own running mean for that
        channel, so a model that rarely predicts Slides is not silenced:
        ``up`` scales a push toward a type by how much more plausible the
        model finds it here than usual, ``down`` scales a suppression by how
        much less plausible. Missing channels (8-dim legacy plans) stay 1.
        """
        ones = np.ones(7, dtype=np.float64)
        if intent is None or not self.plan_intent_strength or len(intent) < 5:
            return ones, ones
        values = np.clip(np.asarray(intent, dtype=np.float64), 0., 1.)
        reference = (self.intent_sum / self.intent_events if self.intent_events and self.intent_sum is not None
                     and len(self.intent_sum) == len(values) else values)
        # plan[3:] = 0 chord, 1 slide, 2 hold(+TouchHold), 3 touch(+TouchHold),
        # 4 density, [5 break, 6 ex]. Gate order: tap, slide, hold, touch,
        # touch_hold, chord, break. Tap is the neutral remainder (ungated).
        def pair(index):
            return (values[index], reference[index]) if index < len(values) else (None, None)
        pairs = [(None, None), pair(1), pair(2), pair(3),
                 (min(values[2], values[3]), min(reference[2], reference[3])),
                 pair(0), pair(5)]
        up, down = ones.copy(), ones.copy()
        for index, (p, ref) in enumerate(pairs):
            if p is None:
                continue
            up[index] = min(1., p / max(ref, .02))
            down[index] = min(1., (1. - p) / max(1. - ref, .02))
        s = self.plan_intent_strength
        return 1. - s*(1. - up), 1. - s*(1. - down)

    def _phase_runs(self, beat):
        b = Fraction(beat)
        bar = int(math.floor(b / 4))
        phase = b - 4 * bar
        rows = self.bar_phase_history.get(phase, ())
        by_bar = dict(rows)
        runs = [0, 0, 0, 0]
        for idx in range(4):
            k = 1
            while (bar - k) in by_bar and by_bar[bar - k][idx]:
                runs[idx] += 1
                k += 1
        return tuple(runs)

    def event(self, beat, second=None, intent=None):
        probabilities = _probabilities(self.cues.get(Fraction(beat))) if self.planner_strength else None
        observed = (self.counts + PSEUDO_HEADS*self.prior)/(self.counts.sum()+PSEUDO_HEADS)
        target = self.target
        if self.section_profile is not None and second is not None:
            multiplier = np.clip(np.asarray(self.section_profile(float(second)), dtype=np.float64), .5, 2.)
            multiplier = multiplier**self.section_strength
            self.section_events += 1
            self.section_multiplier_sum += multiplier
            weighted = self.target*multiplier
            target = weighted/weighted.sum() if weighted.sum() > 0 else self.target
        style_log = self.style_strength*np.log((target+1e-4)/(observed+1e-4))
        up, down = self._intent_gates(intent)
        self._pending_intent = None if intent is None else np.asarray(intent, dtype=np.float64)
        if intent is not None and self.plan_intent_strength:
            style_log = style_log*np.where(style_log > 0, up[:5], down[:5])
            self.gate_sum += np.where(np.r_[style_log, 0., 0.] > 0, up, down)
            self.gate_events += 1
        chord_phase_run, slide_phase_run, hold_phase_run, break_phase_run = self._phase_runs(beat)
        if self.style_strength:
            if slide_phase_run >= 2:
                style_log[1] -= self.style_strength * min(1.2, .35 * (slide_phase_run - 1)) * float(down[1])
            if hold_phase_run >= 2:
                style_log[2] -= self.style_strength * min(1.2, .35 * (hold_phase_run - 1)) * float(down[2])
        planner_log = np.zeros(4) if probabilities is None else self.planner_strength*np.log(
            np.maximum(probabilities, 1e-4)/self.planner_reference)
        local = self.local_chords.event(beat) if self.local_chords is not None else None
        physical = self.physical.event(beat) if self.physical is not None else None
        bias = _EventBias(self, planner_log, style_log, local, physical)
        # Both remaining token feedbacks are suppressions (chord runs, Break).
        bias.chord_gate, bias.break_gate = float(down[5]), float(down[6])
        bias.phase_chord_run, bias.phase_break_run = chord_phase_run, break_phase_run
        return bias

    def compatible(self, prefix, used):
        key = (prefix, used)
        if key in self._compatible_indices:
            self._compatible_indices.move_to_end(key)
            return self._compatible_indices[key]
        indices = np.asarray([i for i in self.prefix_indices.get(prefix, ())
                              if not (self.grammar.heads[self.atoms[i]] & used)], dtype=np.int64)
        if indices.nbytes <= CACHE_BYTES:
            while self._compatible_indices and (self._cache_bytes+indices.nbytes > CACHE_BYTES
                                                 or len(self._compatible_indices) >= CACHE_ENTRIES):
                _, previous = self._compatible_indices.popitem(last=False)
                self._cache_bytes -= previous.nbytes
            self._compatible_indices[key] = indices
            self._cache_bytes += indices.nbytes
        return indices

    def commit(self, text, beat):
        counts = _counts(text)
        if counts is None:
            return
        if self._pending_intent is not None:
            intent = np.clip(self._pending_intent, 0., 1.)
            if self.intent_sum is None or len(self.intent_sum) != len(intent):
                self.intent_sum, self.intent_events = np.zeros(len(intent)), 0
            self.intent_sum += intent
            self.intent_events += 1
            self._pending_intent = None
        self.counts += counts
        self.events += 1
        is_chord = bool(counts.sum() > 1)
        self.chord_events += int(is_chord)
        self.recent_chord_run = self.recent_chord_run + 1 if is_chord else 0
        breaks = _break_count(text)
        self.break_heads += breaks
        self.recent_break_run = self.recent_break_run + 1 if breaks else 0
        b = Fraction(beat)
        bar = int(math.floor(b / 4))
        phase = b - 4 * bar
        rows = self.bar_phase_history.setdefault(phase, [])
        rows.append((bar, (is_chord, bool(counts[1] > 0), bool(counts[2] + counts[4] > 0), bool(breaks > 0))))
        del rows[:-8]
        self.bar_phase_history.move_to_end(phase)
        while len(self.bar_phase_history) > 128:
            self.bar_phase_history.popitem(last=False)
        if self.local_chords is not None:
            self.local_chords.commit(counts, beat)
        if self.physical is not None:
            self.physical.commit(text, beat)
        probabilities = _probabilities(self.cues.get(Fraction(beat))) if self.planner_strength else None
        if probabilities is not None:
            index = int(probabilities.argmax())
            self.argmax_counts[PLANNER_TYPES[index]] += 1
            self.argmax_matches += int(index == int(_primary(counts)))
            self.cue_events += 1

    def replay(self, points, tempo, first, curve_factor=None):
        """Rebuild causal clocks after accepted timing edits, retaining decode diagnostics."""
        if self.local_chords is not None:
            from maichart.form.local_load import LocalChordGuard
            self.local_chords = LocalChordGuard([], target_chord_ratio=self._condition.style['chord_bias'],
                strength=self.local_chord_strength, window_seconds=self.local_chords.window_seconds)
        if self.physical is not None:
            from maichart.form.physical_guard import PhysicalFeasibilityGuard
            self.physical = PhysicalFeasibilityGuard(self.physical_strength,
                max_hand_speed=self.physical.max_hand_speed)
        self.counts = np.zeros(5, dtype=np.float64)
        self.events = self.chord_events = self.break_heads = 0
        self.recent_chord_run = self.recent_break_run = 0
        self.bar_phase_history.clear()
        self.cue_events = self.argmax_matches = 0
        self.argmax_counts.clear()
        # Intent statistics are kept (they describe the model, not the chart);
        # a rejected speculative event's intent must not be attributed here.
        self._pending_intent = None
        for point in points:
            second = float(seconds_at_beat(point.beat, tempo, first))
            if self.local_chords is not None:
                self.local_chords.cues[Fraction(point.beat)] = (
                    second, float(curve_factor(second)) if curve_factor else 1.0)
            if self.physical is not None:
                self.physical.register(point.beat, second, tempo, first)
            self.commit(point.note_text, point.beat)

    def report(self):
        from maichart.form.physical_guard import MAX_PHYSICAL_PENALTY
        available = self.matrix.sum(0) > 0
        style_available = available[:4].copy()
        style_available[2:4] |= available[4]
        ratios = self.counts/max(1.0, float(self.counts.sum()))
        actual = dict(zip(HEAD_TYPES[:4], map(float, ratios[:4])))
        actual['hold'] += float(ratios[4])
        actual['touch'] += float(ratios[4])
        return {
            'schema': 'song_form_type_guidance_v1', 'applied': bool(self.planner_strength or self.style_strength
                                                                     or self.local_chord_strength or self.physical_strength),
            'planner_type_strength': self.planner_strength, 'style_balance_strength': self.style_strength,
            'local_chord_guidance': (self.local_chords.report() if self.local_chords is not None else
                                     {'enabled':False, 'strength':0.0}),
            'physical_feasibility': (self.physical.report() if self.physical is not None else
                                     {'enabled':False, 'strength':0.0}),
            'section_type_guidance': {'enabled': self.section_profile is not None, 'strength': self.section_strength,
                                      'events': self.section_events,
                                      'mean_multiplier': dict(zip(HEAD_TYPES, map(float,
                                          self.section_multiplier_sum/max(1, self.section_events))))},
            'planner_reference': dict(zip(PLANNER_TYPES, map(float, self.planner_reference))),
            'plan_intent_gating': {'strength': self.plan_intent_strength, 'gated_events': self.gate_events,
                                   'intent_events': self.intent_events,
                                   'mean_intent': ([] if self.intent_sum is None or not self.intent_events
                                                   else (self.intent_sum/self.intent_events).tolist()),
                                   'mean_gate': dict(zip(HEAD_TYPES+('chord', 'break'),
                                                         map(float, self.gate_sum/max(1, self.gate_events)))),
                                   'attenuation_only': True},
            'hard_quota': False, 'counter_unit': 'note_heads', 'counter_scope': 'whole_song',
            'measurement_stage': 'draft_before_optional_repair',
            'catalog_types': [name for name, present in zip(HEAD_TYPES, available) if present],
            'catalog_style_types': [name for name, present in zip(HEAD_TYPES[:4], style_available) if present],
            'unreachable_target_types': [name for name, present in zip(HEAD_TYPES[:4], style_available)
                                         if not present and self.requested[name] > 0],
            'unreachable_sampling_types': [name for name, present, target in zip(HEAD_TYPES, available, self.target)
                                           if not present and target > 0],
            'unclassified_atoms': len(self.grammar.atoms)-len(self.atoms),
            'requested_style': self.requested, 'sampling_target_exclusive': dict(zip(HEAD_TYPES, map(float, self.target))),
            'committed_events': self.events, 'committed_chord_events': self.chord_events,
            'committed_break_heads': self.break_heads,
            'actual_style': actual, 'actual_minus_requested': {name: actual[name]-value for name, value in self.requested.items()},
            'heads': int(self.counts.sum()), 'planner_cues_used': self.cue_events,
            'planner_argmax_counts': dict(self.argmax_counts), 'planner_argmax_matches': self.argmax_matches,
            'guided_tokens': self.guided_tokens, 'max_abs_token_bias': self.max_abs_bias,
            'token_bias_bound': (2*MAX_POTENTIAL+(4.0 if self.local_chords is not None else 0.0)
                                 +(MAX_PHYSICAL_PENALTY*self.physical_strength if self.physical is not None else 0.0)),
            'prefix_cache': {'entries': len(self._compatible_indices), 'bytes': self._cache_bytes,
                             'budget_bytes': CACHE_BYTES, 'max_entries': CACHE_ENTRIES},
            'limitations': ['Soft guidance does not guarantee target shares or playability.',
                            'Planner type labels describe event priority, not individual head fractions.',
                            'Duration predictions are not forced into note durations.']}


class _EventBias:
    """Potential differences bias complete cells without a per-character reward."""
    def __init__(self, owner, planner_log, style_log, local_chords=None, physical=None):
        self.owner = owner
        self.planner_log = planner_log
        self.style_log = style_log
        self._potential_cache = {}
        self.local_chords = local_chords
        self._local_cache = {}
        self.physical = physical
        self._physical_cache = {}
        self.chord_gate = 1.0
        self.break_gate = 1.0
        self.phase_chord_run = 0
        self.phase_break_run = 0

    def _score(self, counts):
        style = (counts*self.style_log).sum(-1)/np.maximum(counts.sum(-1), 1.0)
        return np.clip(self.planner_log[_primary(counts)]+style, -MAX_POTENTIAL, MAX_POTENTIAL)

    def _completed(self, parts):
        if any(part not in self.owner.atom_counts for part in parts):
            return None
        return sum((self.owner.atom_counts[part] for part in parts), np.zeros(5))

    def complete(self, text):
        counts = self._completed(self.owner.grammar._split_slashes(text))
        return float(self._score(counts)) if counts is not None else 0.0

    def terminal_score(self, text):
        """One preference for a fully written cell, independent of its spelling.

        Stream sampling uses this instead of prefix differences: local softmax
        normalizers prevent those differences from telescoping into a true
        sequence preference. Legacy form decoding retains its existing API.
        """
        return (self.complete(text) + self._local_potential(text, complete=True)
                + self._physical_potential(text, complete=True))

    def token_bias(self, text, continuations):
        """Bounded causal feedback for Break and chord runs."""
        owner = self.owner
        if not owner.style_strength:
            return [0.0] * len(continuations)
        observed_chord = (owner.chord_events + 8.0 * owner._condition.style['chord_bias']) / max(8.0 + owner.events, 1.0)
        target_chord = owner._condition.style['chord_bias']
        break_target = min(.08, max(.02, owner.requested['tap'] * .1))
        observed_break = (owner.break_heads + 8.0 * break_target) / max(8.0 + owner.counts.sum(), 1.0)
        values=[]
        for char in continuations:
            value=0.0
            if char == '/':
                value -= owner.style_strength * min(2.5, max(0.0, observed_chord-target_chord) * 12.0)
                if owner.recent_chord_run >= 4:
                    value -= owner.style_strength * min(2.0, .35 * (owner.recent_chord_run-3))
                if self.phase_chord_run >= 2:
                    value -= owner.style_strength * min(1.5, .35 * (self.phase_chord_run-1))
                value *= self.chord_gate
            if char == 'b':
                value -= owner.style_strength * min(2.0, max(0.0, observed_break-break_target) * 10.0)
                if owner.recent_break_run >= 4:
                    value -= owner.style_strength * min(2.0, .35 * (owner.recent_break_run-3))
                if self.phase_break_run >= 2:
                    value -= owner.style_strength * min(1.5, .35 * (self.phase_break_run-1))
                value *= self.break_gate
            values.append(float(max(-3.0, min(0.0, value))))
        maximum=max(map(abs,values),default=0.0)
        owner.max_abs_bias=max(owner.max_abs_bias,maximum)
        owner.guided_tokens += int(maximum > 1e-9)
        return values

    def potential(self, text):
        if text in self._potential_cache:
            return self._potential_cache[text]
        parts = self.owner.grammar._split_slashes(text)
        fixed = self._completed(parts[:-1])
        if text.endswith('/'):
            value = float(self._score(fixed)) if fixed is not None else 0.0
        else:
            used = frozenset().union(*(self.owner.grammar.heads[part] for part in parts[:-1]))
            indices = self.owner.compatible(parts[-1], used)
            if fixed is None or not len(indices):
                value = 0.0
            else:
                scores = self._score(self.owner.matrix[indices]+fixed)
                maximum = float(scores.max())
                value = maximum+float(np.log(np.exp(scores-maximum).mean()))
        self._potential_cache[text] = value
        return value

    def _local_potential(self, text, *, complete=False):
        """Price the eventual complete cell, including a head required by '/'.

        Waiting until after the slash would subtract the same penalty from
        every next-head choice, which cancels in softmax and cannot stop chords.
        """
        if self.local_chords is None:
            return 0.0
        key = (text, complete)
        if key in self._local_cache:
            return self._local_cache[key]
        parts = self.owner.grammar._split_slashes(text)
        if complete:
            counts = self._completed(parts)
            value = -float(self.local_chords.penalty(counts[:3].sum())) if counts is not None else 0.0
        else:
            fixed = self._completed(parts[:-1])
            used = frozenset().union(*(self.owner.grammar.heads[part] for part in parts[:-1]))
            indices = self.owner.compatible(parts[-1], used)
            if fixed is None or not len(indices):
                value = 0.0
            else:
                heads = self.owner.matrix[indices, :3].sum(-1)+fixed[:3].sum()
                scores = -self.local_chords.penalty(heads)
                maximum = float(scores.max())
                value = maximum+float(np.log(np.exp(scores-maximum).mean()))
        self._local_cache[key] = value
        return value

    def _lane_set(self, parts):
        """Union of the hand lanes a list of completed atoms strikes."""
        lanes = np.zeros(LANES, dtype=bool)
        for part in parts:
            index = self.owner.atom_index.get(part)
            if index is None:
                return None
            lanes |= self.owner.lane_matrix[index]
        return lanes

    def _physical_potential(self, text, *, complete=False):
        """Price an occupied or unreachable button set for the eventual cell.

        Same telescoping shape as the local-chord term: the delta against the
        parent prefix is what reaches the softmax, so a penalty that applies to
        every continuation cancels and cannot steer the choice.
        """
        if self.physical is None:
            return 0.0
        key = (text, complete)
        if key in self._physical_cache:
            return self._physical_cache[key]
        parts = self.owner.grammar._split_slashes(text)
        if complete:
            lanes = self._lane_set(parts)
            # Pass the boolean lane mask to the matrix API; the scalar entry
            # point takes lane numbers, and an array would iterate element-wise.
            value = (-float(self.physical.vector_penalty(lanes[None])[0])
                     if lanes is not None else 0.0)
        else:
            fixed = self._lane_set(parts[:-1])
            used = frozenset().union(*(self.owner.grammar.heads[part] for part in parts[:-1]))
            indices = self.owner.compatible(parts[-1], used)
            if fixed is None or not len(indices):
                value = 0.0
            else:
                # Score the eventual complete cell: every atom that can finish
                # the current part, combined with the already fixed lanes.
                candidates = self.owner.lane_matrix[indices] | fixed
                scores = -self.physical.vector_penalty(candidates)
                maximum = float(scores.max())
                value = maximum+float(np.log(np.exp(scores-maximum).mean()))
        self._physical_cache[key] = value
        return value

    def biases(self, text, continuations):
        parent = self.potential(text)
        values = [self.complete(text)-parent if char is None else self.potential(text+char)-parent
                  for char in continuations]
        if self.local_chords is not None:
            local_parent = self._local_potential(text)
            values = [value+self._local_potential(text if char is None else text+char,
                                                  complete=char is None)-local_parent
                      for value, char in zip(values, continuations)]
        if self.physical is not None:
            physical_parent = self._physical_potential(text)
            values = [value+self._physical_potential(text if char is None else text+char,
                                                     complete=char is None)-physical_parent
                      for value, char in zip(values, continuations)]
        maximum = max(map(abs, values), default=0.0)
        self.owner.max_abs_bias = max(self.owner.max_abs_bias, maximum)
        self.owner.guided_tokens += int(maximum > 1e-9)
        return values
