"""Bounded inference-time type guidance over the existing note grammar."""
from __future__ import annotations

from collections import Counter, OrderedDict
from fractions import Fraction
import math

import numpy as np

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


def _counts(text):
    parsed = parse_timing_point_notes(RawTimingPoint(
        index=0, raw=text, note_text=text, beat=Fraction(0), tick=0,
        time_sec=0.0, division=4, bpm=120.0))
    if parsed.unknown_tokens or not parsed.notes:
        return None
    counts = Counter(note.note_type for note in parsed.notes)
    return np.asarray([counts[name] for name in HEAD_TYPES], dtype=np.float64)


def _primary(counts):
    # Match blueprint supervision: Slide > Hold > Touch/TouchHold > Tap.
    return np.where(counts[..., 1] > 0, 2, np.where(
        counts[..., 2] > 0, 1, np.where(counts[..., 3:5].sum(-1) > 0, 3, 0)))


def _probabilities(cue):
    values = cue.get('type_probabilities') if cue else None
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or not values.sum():
        raise ValueError('Blueprint type_probabilities must contain four finite nonnegative values')
    return values / values.sum()


class TypeGuidance:
    """Keep head-share feedback across resource passes and musical regions."""

    def __init__(self, atoms, condition, cues, *, planner_strength=1.0, style_strength=1.0):
        self.planner_strength = validate_strength(planner_strength, 'planner_type_strength')
        self.style_strength = validate_strength(style_strength, 'style_balance_strength')
        self.grammar = _NoteGrammar.from_atoms(atoms)
        if self.grammar is None:
            raise ValueError('Type guidance requires a nonempty checkpoint atom catalog')
        self.cues = {Fraction(cue['beat']): cue for cue in cues}
        probabilities = [_probabilities(cue) for cue in self.cues.values()] if self.planner_strength else []
        probabilities = [value for value in probabilities if value is not None]
        # Use event-class probabilities, not head fractions, as the reference.
        # A uniform component keeps a constant confident cue from cancelling.
        self.planner_reference = (0.8*np.mean(probabilities, axis=0)+0.2/4
                                  if probabilities else np.full(4, 0.25))
        self.target = exclusive_preferences(condition.style)
        self.requested = {name: condition.style[f'{name}_bias'] for name in HEAD_TYPES[:4]}
        self.counts = np.zeros(5, dtype=np.float64)
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
        self._compatible_indices = OrderedDict()
        self._cache_bytes = 0
        self.cue_events = 0
        self.argmax_counts = Counter()
        self.argmax_matches = 0
        self.guided_tokens = 0
        self.max_abs_bias = 0.0

    def event(self, beat):
        probabilities = _probabilities(self.cues.get(Fraction(beat))) if self.planner_strength else None
        observed = (self.counts + PSEUDO_HEADS*self.prior)/(self.counts.sum()+PSEUDO_HEADS)
        style_log = self.style_strength*np.log((self.target+1e-4)/(observed+1e-4))
        planner_log = np.zeros(4) if probabilities is None else self.planner_strength*np.log(
            np.maximum(probabilities, 1e-4)/self.planner_reference)
        return _EventBias(self, planner_log, style_log)

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
        self.counts += counts
        probabilities = _probabilities(self.cues.get(Fraction(beat))) if self.planner_strength else None
        if probabilities is not None:
            index = int(probabilities.argmax())
            self.argmax_counts[PLANNER_TYPES[index]] += 1
            self.argmax_matches += int(index == int(_primary(counts)))
            self.cue_events += 1

    def report(self):
        available = self.matrix.sum(0) > 0
        style_available = available[:4].copy()
        style_available[2:4] |= available[4]
        ratios = self.counts/max(1.0, float(self.counts.sum()))
        actual = dict(zip(HEAD_TYPES[:4], map(float, ratios[:4])))
        actual['hold'] += float(ratios[4])
        actual['touch'] += float(ratios[4])
        return {
            'schema': 'song_form_type_guidance_v1', 'applied': bool(self.planner_strength or self.style_strength),
            'planner_type_strength': self.planner_strength, 'style_balance_strength': self.style_strength,
            'planner_reference': dict(zip(PLANNER_TYPES, map(float, self.planner_reference))),
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
            'actual_style': actual, 'actual_minus_requested': {name: actual[name]-value for name, value in self.requested.items()},
            'heads': int(self.counts.sum()), 'planner_cues_used': self.cue_events,
            'planner_argmax_counts': dict(self.argmax_counts), 'planner_argmax_matches': self.argmax_matches,
            'guided_tokens': self.guided_tokens, 'max_abs_token_bias': self.max_abs_bias,
            'token_bias_bound': 2*MAX_POTENTIAL,
            'prefix_cache': {'entries': len(self._compatible_indices), 'bytes': self._cache_bytes,
                             'budget_bytes': CACHE_BYTES, 'max_entries': CACHE_ENTRIES},
            'limitations': ['Soft guidance does not guarantee target shares or playability.',
                            'Planner type labels describe event priority, not individual head fractions.',
                            'Duration predictions are not forced into note durations.']}


class _EventBias:
    """Potential differences bias complete cells without a per-character reward."""
    def __init__(self, owner, planner_log, style_log):
        self.owner = owner
        self.planner_log = planner_log
        self.style_log = style_log
        self._potential_cache = {}

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

    def biases(self, text, continuations):
        parent = self.potential(text)
        values = [self.complete(text)-parent if char is None else self.potential(text+char)-parent
                  for char in continuations]
        maximum = max(map(abs, values), default=0.0)
        self.owner.max_abs_bias = max(self.owner.max_abs_bias, maximum)
        self.owner.guided_tokens += int(maximum > 1e-9)
        return values
