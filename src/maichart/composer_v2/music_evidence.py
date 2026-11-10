"""Small audio-only inputs for matched passages and future entry decisions.

The existing beat profile already contains music representations and twelve
within-beat onset cells. These functions reuse it: no waveform reads, encoder
calls, chart labels, meter assumptions or compulsory notes are introduced.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import weakref

import numpy as np
import torch
from torch import nn


POLICY = 'song_paired_v2'
SEMANTIC_FEATURES = 16
ONSET_BINS = 12
PAIR_FEATURES = 48
ENTRY_SPANS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16))
ENTRY_FEATURES = len(ENTRY_SPANS) * (SEMANTIC_FEATURES + ONSET_BINS + 1)

_PROJECTIONS = {}


def _projection(width):
    """Fixed shared compression; source and current music use the same map."""
    matrix = _PROJECTIONS.get(width)
    if matrix is None:
        rng = np.random.default_rng(20261110 + width)
        matrix = (rng.standard_normal((width, SEMANTIC_FEATURES)) /
                  math.sqrt(SEMANTIC_FEATURES)).astype(np.float32)
        # Normally only native (<=96) and MERT-derived (256) widths occur.
        if len(_PROJECTIONS) >= 8:
            _PROJECTIONS.clear()
        _PROJECTIONS[width] = matrix
    return matrix


@dataclass
class _Prepared:
    semantic: np.ndarray
    onset: np.ndarray
    onset_raw: np.ndarray
    prefix: np.ndarray

    @property
    def nbytes(self):
        return sum(value.nbytes for value in
                   (self.semantic, self.onset, self.onset_raw, self.prefix))


class MusicEvidenceCache:
    """Bounded derived profiles, independent of model weights and chart notes.

    Profiles are immutable during their lifetime. Weak references prevent an id
    reused for a different profile from selecting stale evidence. Arrays, not
    the source profile or audio mappings, are retained here.
    """
    def __init__(self, max_bytes=16 * 1024 * 1024, max_profiles=64):
        self.max_bytes = max(0, int(max_bytes))
        self.max_profiles = max(0, int(max_profiles))
        self.items = OrderedDict()
        self.bytes = 0
        self.builds = 0
        self.hits = 0

    def get(self, profile):
        key = id(profile)
        cached = self.items.get(key)
        if cached is not None:
            reference, value = cached
            if reference() is profile:
                self.items.move_to_end(key)
                self.hits += 1
                return value
            self.items.pop(key)
            self.bytes -= value.nbytes
        value = _prepare(profile)
        self.builds += 1
        if self.max_profiles and value.nbytes <= self.max_bytes:
            try:
                reference = weakref.ref(profile)
            except TypeError:
                # Duck-typed lightweight profiles can still be queried safely.
                return value
            self.items[key] = (reference, value)
            self.bytes += value.nbytes
            while (len(self.items) > self.max_profiles or self.bytes > self.max_bytes):
                _, (_, evicted) = self.items.popitem(last=False)
                self.bytes -= evicted.nbytes
        return value


_CACHE = MusicEvidenceCache()


def _prepare(profile):
    semantic = np.asarray(profile.semantic, dtype=np.float32)
    onset = np.asarray(profile.onset, dtype=np.float32)
    raw = np.asarray(profile.onset_raw, dtype=np.float32)
    if (semantic.ndim != 2 or onset.shape != (len(semantic), ONSET_BINS)
            or raw.shape != onset.shape):
        raise ValueError('Paired music needs the shared twelve-cell beat profile')
    semantic = semantic @ _projection(semantic.shape[1])
    raw = np.maximum(raw, 0.)
    # One prefix-sum pass per immutable profile; interval pooling is O(1).
    combined = np.concatenate((semantic, raw), axis=1)
    prefix = np.concatenate((np.zeros((1, combined.shape[1]), np.float64),
                             np.cumsum(combined, axis=0, dtype=np.float64)), axis=0)
    return _Prepared(semantic, onset, raw, prefix)


def paired_music_features(profile, beats, shift, *, start, end, cache=None):
    """[queries,48] source/current music at the same aligned integer beat.

    Fields are: coverage; source semantic[16]; current semantic[16];
    current-minus-source onset pattern[12]; source/current log mean flux;
    compressed semantic cosine. No reference or unchecked positions are zero.
    The scalar similarities complement, rather than replace, the existing
    detailed source/current vectors.
    """
    beats = np.asarray(beats, dtype=np.float64).reshape(-1)
    result = np.zeros((len(beats), PAIR_FEATURES), dtype=np.float32)
    if shift is None or not len(beats):
        return result
    value = (cache if cache is not None else _CACHE).get(profile)
    finite = np.isfinite(beats)
    target = np.floor(np.where(finite, beats, 0.)).astype(np.int64)
    source = target - int(shift)
    valid = (finite & (beats >= float(start)) & (beats < float(end)) &
             (target >= 0) & (target < len(value.semantic)) &
             (source >= 0) & (source < len(value.semantic)))
    rows = np.flatnonzero(valid)
    if not len(rows):
        return result
    t, s = target[rows], source[rows]
    current, previous = value.semantic[t], value.semantic[s]
    result[rows, 0] = 1.
    result[rows, 1:17] = previous
    result[rows, 17:33] = current
    result[rows, 33:45] = value.onset[t] - value.onset[s]
    result[rows, 45] = np.log1p(value.onset_raw[s].mean(1))
    result[rows, 46] = np.log1p(value.onset_raw[t].mean(1))
    denominator = np.linalg.norm(current, axis=1) * np.linalg.norm(previous, axis=1)
    result[rows, 47] = (current * previous).sum(1) / np.maximum(denominator, 1e-8)
    return result


def entry_preview(profile, anchors, *, cache=None):
    """[anchors,145] bounded multi-scale future music, independent of reference.

    Each of five relative beat spans contains a mean semantic vector[16],
    log1p(mean raw onset cells)[12], and covered fraction[1]. Missing music is
    zero; a partially covered end interval pools only the available beats.
    Anchors are integer audio tile anchors used by both training and generation.
    """
    anchors = np.asarray(anchors, dtype=np.int64).reshape(-1)
    result = np.zeros((len(anchors), ENTRY_FEATURES), dtype=np.float32)
    if not len(anchors):
        return result
    value = (cache if cache is not None else _CACHE).get(profile)
    length = len(value.semantic)
    for index, (left_offset, right_offset) in enumerate(ENTRY_SPANS):
        left = np.clip(anchors + left_offset, 0, length)
        right = np.clip(anchors + right_offset, 0, length)
        count = np.maximum(0, right-left)
        valid = count > 0
        pooled = ((value.prefix[right] - value.prefix[left]) /
                  np.maximum(count, 1)[:, None])
        pooled[:, SEMANTIC_FEATURES:] = np.log1p(np.maximum(0., pooled[:, SEMANTIC_FEATURES:]))
        width = SEMANTIC_FEATURES + ONSET_BINS + 1
        result[:, index*width:(index+1)*width-1] = pooled.astype(np.float32) * valid[:, None]
        result[:, (index+1)*width-1] = count / (right_offset-left_offset)
    return result


class ZeroMusicProjection(nn.Linear):
    """Residual input that preserves its parent's function before finetune."""
    def __init__(self, features, width):
        # Adding the module does not change initialization RNG of old modules.
        with torch.random.fork_rng(devices=[]):
            super().__init__(features, width, bias=False)
        nn.init.zeros_(self.weight)
