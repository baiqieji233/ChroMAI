"""Music-clock phrases and committed-prefix descriptors shared by all streams.

No chart labels participate in boundaries or reference selection. Descriptors
are additive ordered statistics, not a codebook or instructions to copy notes.
"""
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
import weakref

import numpy as np
import torch

from .stream_continuation import typed_event

PHRASE_FEATURE_DIM = 96
PHRASE_MEMORY_SLOTS = 4
MUSIC_LAYOUT_SCHEMA = 'acoustic32_interval_novelty_v2'
_LAYOUTS = OrderedDict()
_LAYOUT_LOCK = threading.RLock()
_LAYOUT_BUDGET = 4*1024**2
PHRASE_POLICIES = ('off', 'causal_phrase_v1')
PHRASE_ROW_FIELDS = (
    'phrase_start_beat', 'phrase_end_beat', 'phrase_progress',
    'phrase_memory_features', 'phrase_memory_start_beat',
    'phrase_memory_end_beat', 'phrase_memory_mask', 'target_phrase_features',
    'target_phrase_mask', 'target_phrase_weight')


def acoustic_descriptors(bank):
    """32 existing acoustic channels, normalized on the song's own scale.

    Production bank = 104 macro + 120 local + 14 contrasts. Tiny fixtures
    are padded finitely, without assigning missing channels invented values.
    """
    values = np.nan_to_num(bank[0].detach().cpu().numpy(),nan=0.,posinf=0.,neginf=0.)
    padded = np.zeros((len(values),238),dtype=np.float32)
    width = min(values.shape[-1],238)
    padded[:,:width] = values[:,:width]
    local = padded[:,104:224]
    raw = np.concatenate((local[:,:80].reshape(-1,8,10).mean(-1),
                          local[:,80:92],local[:,92:98],padded[:,232:238]),axis=-1)
    center = raw.mean(0)
    scale = raw.std(0)
    return np.clip((raw-center)/np.maximum(scale,1e-5),-8.,8.).astype(np.float32)


def _phrase_boundaries(values, beats, end):
    """Choose bounded, variable beat spans using existing acoustic changes.

    The beat unit is a clock unit, never an assumed time signature. Future
    music is allowed; future authored events are absent from this function.
    """
    start = min(0., float(beats[0])) if len(beats) else 0.
    edges = [start]
    columns = values.shape[-1]
    while edges[-1] < end:
        anchor = edges[-1]
        candidates = np.asarray([anchor + x for x in (2., 3., 4., 6., 8.)])
        changes = []
        for candidate in candidates:
            # Half-beat evidence on both sides, not a single bank-frame probe.
            left, split, right = np.searchsorted(beats,(candidate-.5,candidate,candidate+.5))
            if split>left and right>split:
                before = values[left:split,:columns].mean(0)
                after = values[split:right,:columns].mean(0)
                changes.append(float(np.abs(after-before).mean()))
            else:changes.append(0.)
        changes = np.asarray(changes)
        # Prefer a longer span when there is no meaningful acoustic edge.
        choice = int(np.argmax(changes)) if changes.max() > 1e-4 else 4
        following = min(end, float(candidates[choice]))
        if following <= anchor: break
        edges.append(following)
    if len(edges) == 1: edges.append(start + 2.)
    return np.asarray(edges, dtype=np.float64)


@dataclass(frozen=True)
class MusicPhraseLayout:
    boundaries: np.ndarray
    audio: np.ndarray
    references: tuple
    schema: str = MUSIC_LAYOUT_SCHEMA

    @property
    def nbytes(self):
        return self.boundaries.nbytes+self.audio.nbytes+len(self.references)*256


def music_phrase_layout(bank,end_beat=None):
    """Small immutable music-only cache; weak identities never own audio banks."""
    end = float(bank[2][-1]) if end_beat is None else float(end_beat)
    tensors = tuple(bank[:3])
    key = (MUSIC_LAYOUT_SCHEMA,tuple(id(tensor) for tensor in tensors),end)
    with _LAYOUT_LOCK:
        entry = _LAYOUTS.get(key)
        if entry is not None and all(reference() is tensor for reference,tensor in zip(entry[0],tensors)):
            _LAYOUTS.move_to_end(key)
            return entry[1]
        if entry is not None:del _LAYOUTS[key]
        values = acoustic_descriptors(bank)
        beats = bank[2].detach().cpu().numpy()
        boundaries = _phrase_boundaries(values,beats,end)
        audio = []
        for start,stop in zip(boundaries[:-1],boundaries[1:]):
            lo,hi = np.searchsorted(beats,(start,stop))
            rows = values[lo:max(lo+1,hi)]
            audio.append(rows.mean(0) if len(rows) else np.zeros(32))
        audio = np.asarray(audio,dtype=np.float32)
        references = []
        for segment in range(len(boundaries)-1):
            near = list(range(max(0,segment-2),segment))[::-1]
            older = np.arange(max(0,segment-2))
            if len(older):
                previous,query = audio[older],audio[segment]
                score = (previous@query)/(np.linalg.norm(previous,axis=-1)*np.linalg.norm(query)+1e-8)
                far = older[np.argsort(-score,kind='stable')[:2]].tolist()
            else:far = []
            references.append(tuple(near+far))
        boundaries.setflags(write=False);audio.setflags(write=False)
        layout = MusicPhraseLayout(boundaries,audio,tuple(references))
        if layout.nbytes<=_LAYOUT_BUDGET:
            while _LAYOUTS and (len(_LAYOUTS)>=8 or sum(value[1].nbytes for value in _LAYOUTS.values())+layout.nbytes>_LAYOUT_BUDGET):
                _LAYOUTS.popitem(last=False)
            _LAYOUTS[key] = (tuple(weakref.ref(tensor) for tensor in tensors),layout)
        return layout


def phrase_boundaries(bank,end_beat=None):
    return music_phrase_layout(bank,end_beat).boundaries


def _event_vector(step, start, end, previous_lane=None):
    result = np.zeros(PHRASE_FEATURE_DIM, dtype=np.float32)
    span = max(end-start, 1e-8)
    slot = min(3, max(0, int(4*(float(step.end)-start)/span)))
    part = result[slot*24:(slot+1)*24]
    gap = max(0., float(step.end-step.start))
    part[16 + min(3, int(np.searchsorted((.25, .5, 1.), gap)))] = 1.
    part[20] = 1.
    part[21] = min(gap/span, 1.)
    lane = None
    if step.kind == 2:
        types, buttons, _ = typed_event(step.text)
        for i, kind in enumerate(('tap', 'hold', 'slide', 'touch', 'touch_hold')):
            part[i] = float(kind in types)
        for button in buttons: part[5+(button-1)%8] = 1.
        if buttons:
            angles = np.asarray([(button-1)*math.pi/4 for button in buttons])
            sine,cosine = float(np.sin(angles).mean()),float(np.cos(angles).mean())
            if math.hypot(sine,cosine)>1e-6:lane = math.atan2(sine,cosine)
            if lane is not None and previous_lane is not None:
                movement = math.atan2(math.sin(lane-previous_lane), math.cos(lane-previous_lane))
                part[13:16] = (math.sin(movement), math.cos(movement), abs(movement)/math.pi)
        part[22] = len(buttons)/8.
        part[23] = 1.
    return result, lane


def describe_phrase(steps, start_beat, end_beat):
    """Describe only supplied committed steps; ordered four-bin geometry."""
    total = np.zeros(PHRASE_FEATURE_DIM, dtype=np.float32)
    lane = None
    for step in steps:
        if start_beat <= float(step.end) < end_beat:
            value, lane = _event_vector(step, start_beat, end_beat, lane)
            total += value
    return torch.from_numpy(np.sign(total)*np.log1p(np.abs(total)))


class PhraseIndex:
    """One compact prefix index per chart, also supports incremental commit.

    Each event stores one cumulative feature vector, never R memory tensors.
    Query reads are logarithmic and only expose indices strictly before q.
    """
    def __init__(self, bank, end_beat, steps=()):
        self.layout = music_phrase_layout(bank,end_beat)
        self.boundaries,self.audio,self.references = self.layout.boundaries,self.layout.audio,self.layout.references
        count = len(self.boundaries)-1
        self.indices = [[] for _ in range(count)]
        self.prefix = [[] for _ in range(count)]
        self.lanes = [None]*count
        self.lane_history = [[] for _ in range(count)]
        self.count = 0
        self._feature_cache = {}
        for step in steps: self.append(step)

    @property
    def nbytes(self):
        # Include Python/list overhead conservatively in the cache budget.
        return self.count*(PHRASE_FEATURE_DIM*8+336)+len(self.boundaries)*256

    def segment(self, beat):
        return min(len(self.indices)-1, max(0, bisect_right(self.boundaries, float(beat))-1))

    def append(self, step):
        segment = self.segment(step.end)
        self._feature_cache.pop(segment,None)
        # EOS has no event and is only a supervision query.
        if step.kind in (2, 3):
            value, lane = _event_vector(step, self.boundaries[segment], self.boundaries[segment+1], self.lanes[segment])
            self.lanes[segment] = lane
            self.lane_history[segment].append(lane)
            self.indices[segment].append(self.count)
            previous = self.prefix[segment][-1] if self.prefix[segment] else 0.
            self.prefix[segment].append(value+previous)
        self.count += 1

    def edited(self, steps, begin, end):
        """Reuse immutable prefix vectors; recompute only the edited suffix."""
        result = object.__new__(PhraseIndex)
        result.layout = self.layout
        result.boundaries, result.audio, result.references = self.boundaries, self.audio, self.references
        result.indices, result.prefix, result.lane_history, result.lanes = [], [], [], []
        for indices, values, lanes in zip(self.indices, self.prefix, self.lane_history):
            keep = bisect_left(indices, begin)
            result.indices.append(indices[:keep]); result.prefix.append(values[:keep])
            result.lane_history.append(lanes[:keep]); result.lanes.append(lanes[keep-1] if keep else None)
        result.count = begin
        result._feature_cache = {}
        for step in steps[begin:end]: result.append(step)
        return result

    def _features(self, segment, query):
        pos = bisect_left(self.indices[segment], query)-1
        raw = self.prefix[segment][pos] if pos >= 0 else np.zeros(PHRASE_FEATURE_DIM, dtype=np.float32)
        return torch.from_numpy(np.sign(raw)*np.log1p(np.abs(raw)))

    def row(self, query, cursor, *, supervise=False):
        segment = self.segment(cursor)
        starts = torch.zeros(PHRASE_MEMORY_SLOTS, dtype=torch.float64)
        ends = starts.clone()
        memory = torch.zeros(PHRASE_MEMORY_SLOTS, PHRASE_FEATURE_DIM)
        valid = torch.zeros(PHRASE_MEMORY_SLOTS, dtype=torch.bool)
        for slot, reference in enumerate(self.references[segment]):
            starts[slot], ends[slot] = float(self.boundaries[reference]), float(self.boundaries[reference+1])
            memory[slot] = self._features(reference, query)
            valid[slot] = True
        return dict(phrase_start_beat=torch.tensor(self.boundaries[segment], dtype=torch.float64),
            phrase_end_beat=torch.tensor(self.boundaries[segment+1], dtype=torch.float64),
            phrase_progress=self._features(segment, query), phrase_memory_features=memory,
            phrase_memory_start_beat=starts, phrase_memory_end_beat=ends, phrase_memory_mask=valid,
            target_phrase_features=self._features(segment, self.count) if supervise else torch.zeros(PHRASE_FEATURE_DIM),
            target_phrase_mask=torch.tensor(bool(supervise)), target_phrase_weight=torch.tensor(1. if supervise else 0.))

    def rows(self, steps, first, end, *, supervise=False):
        queries = np.arange(first,end,dtype=np.int64)
        cursors = np.fromiter((float(step.start) for step in steps[first:end]),dtype=np.float64,count=end-first)
        return self.rows_at(queries,cursors,supervise=supervise)

    def _packed_features(self,segment):
        packed = self._feature_cache.get(segment)
        if packed is None:
            indices = np.asarray(self.indices[segment],dtype=np.int64)
            if len(indices):
                raw = np.stack(self.prefix[segment])
                features = np.sign(raw)*np.log1p(np.abs(raw))
            else:features = np.empty((0,PHRASE_FEATURE_DIM),dtype=np.float32)
            packed = indices,features
            self._feature_cache[segment] = packed
        return packed

    def rows_at(self,queries,cursors,*,supervise=False):
        """Bulk CPU extraction: ten NumPy allocations, no per-query Torch ops.

        Prefix selection remains strictly index < query, including references.
        Cached descriptors only depend on committed/raw events; visibility is
        applied on every gather, so a packed full training phrase cannot leak.
        """
        queries = np.asarray(queries,dtype=np.int64)
        cursors = np.asarray(cursors,dtype=np.float64)
        segments = np.clip(np.searchsorted(self.boundaries,cursors,side='right')-1,0,len(self.indices)-1)
        length = len(queries)
        progress = np.zeros((length,PHRASE_FEATURE_DIM),dtype=np.float32)
        memory = np.zeros((length,PHRASE_MEMORY_SLOTS,PHRASE_FEATURE_DIM),dtype=np.float32)
        memory_start = np.zeros((length,PHRASE_MEMORY_SLOTS),dtype=np.float64)
        memory_end = np.zeros_like(memory_start)
        memory_mask = np.zeros((length,PHRASE_MEMORY_SLOTS),dtype=np.bool_)
        target = np.zeros_like(progress)
        def gather(segment,selected):
            indices,features = self._packed_features(segment)
            positions = np.searchsorted(indices,selected,side='left')-1
            result = np.zeros((len(selected),PHRASE_FEATURE_DIM),dtype=np.float32)
            valid = positions>=0
            if valid.any():result[valid] = features[positions[valid]]
            return result
        for segment in np.unique(segments):
            group = np.flatnonzero(segments==segment)
            selected = queries[group]
            progress[group] = gather(segment,selected)
            if supervise:
                indices,features = self._packed_features(segment)
                if len(indices):target[group] = features[-1]
            for slot,reference in enumerate(self.references[segment]):
                memory_start[group,slot] = self.boundaries[reference]
                memory_end[group,slot] = self.boundaries[reference+1]
                memory_mask[group,slot] = True
                indices,features = self._packed_features(reference)
                if len(indices) and np.all(selected>indices[-1]):
                    memory[group,slot] = features[-1]
                elif len(indices):memory[group,slot] = gather(reference,selected)
        arrays = dict(phrase_start_beat=self.boundaries[segments].copy(),
            phrase_end_beat=self.boundaries[segments+1].copy(),phrase_progress=progress,
            phrase_memory_features=memory,phrase_memory_start_beat=memory_start,
            phrase_memory_end_beat=memory_end,phrase_memory_mask=memory_mask,
            target_phrase_features=target,target_phrase_mask=np.full(length,supervise,dtype=np.bool_),
            target_phrase_weight=np.full(length,1. if supervise else 0.,dtype=np.float32))
        return {key:torch.from_numpy(value) for key,value in arrays.items()}
