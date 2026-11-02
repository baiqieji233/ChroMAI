"""Music-clock phrases and committed-prefix descriptors shared by all streams.

No chart labels participate in boundaries or reference selection. Descriptors
are additive ordered statistics, not a codebook or instructions to copy notes.
"""
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import math
import hashlib
import threading
import weakref
from fractions import Fraction

import numpy as np
import torch

from .stream_continuation import typed_event

PHRASE_FEATURE_DIM = 96
PHRASE_MEMORY_SLOTS = 4
MUSIC_LAYOUT_SCHEMA = 'acoustic32_interval_novelty_v2'
_LAYOUTS = OrderedDict()
_LAYOUT_LOCK = threading.RLock()
_LAYOUT_BUDGET = 4*1024**2
PHRASE_POLICIES = ('off', 'causal_phrase_v1', 'ordered_phrase_v2', 'aligned_phrase_v3')
PHRASE_SEQUENCE_LENGTH = 32
PHRASE_SEQUENCE_FIELDS = ('kind','shape','phase_mask','atoms','beats','seconds','gap_beats','gap_seconds','mask')
_ENCODE_UNSET = object()


def collate_phrase_music(batch):
    """Exact CPU interval table shared by all ordered-phrase audio queries.

    Song identity participates in deduplication: equal beat ranges from two
    different audio banks are never merged. No tensor is copied back from GPU.
    """
    names=('phrase_start_beat','phrase_end_beat','phrase_memory_start_beat','phrase_memory_end_beat','song_inverse')
    if any(batch[name].device.type!='cpu' for name in names):
        raise ValueError('Phrase music intervals must be deduplicated during CPU collation')
    start=batch['phrase_start_beat'].double().numpy()
    end=batch['phrase_end_beat'].double().numpy()
    reference_start=batch['phrase_memory_start_beat'].double().numpy()
    reference_end=batch['phrase_memory_end_beat'].double().numpy()
    songs=batch['song_inverse'].numpy()
    anchor_songs=np.broadcast_to(songs[:,None],start.shape)
    reference_songs=np.broadcast_to(songs[:,None,None],reference_start.shape)
    anchor=np.stack((anchor_songs,start,end),-1).reshape(-1,3)
    reference=np.stack((reference_songs,reference_start,reference_end),-1).reshape(-1,3)
    unique,inverse=np.unique(np.concatenate((anchor,reference),0),axis=0,return_inverse=True)
    split=start.size
    return dict(phrase_music_song_inverse=torch.from_numpy(unique[:,0].astype(np.int64)),
        phrase_music_start_beat=torch.from_numpy(unique[:,1].copy()),
        phrase_music_end_beat=torch.from_numpy(unique[:,2].copy()),
        phrase_music_anchor_inverse=torch.from_numpy(inverse[:split].reshape(start.shape).copy()),
        phrase_music_reference_inverse=torch.from_numpy(inverse[split:].reshape(reference_start.shape).copy()))


def _sequence_arrays(count,atom_width=1):
    length = PHRASE_SEQUENCE_LENGTH
    return dict(kind=np.zeros((count,length),dtype=np.int64),shape=np.zeros((count,length),dtype=np.int64),
        phase_mask=np.zeros((count,length,8),dtype=np.bool_),atoms=np.zeros((count,length,max(1,atom_width)),dtype=np.int64),
        beats=np.zeros((count,length),dtype=np.float64),seconds=np.zeros((count,length),dtype=np.float64),
        gap_beats=np.zeros((count,length),dtype=np.float64),gap_seconds=np.zeros((count,length),dtype=np.float64),
        mask=np.zeros((count,length),dtype=np.bool_))


def _pack_sequences(sequences,prefix):
    width = max((sequence['atoms'].shape[-1] for sequence in sequences),default=1)
    arrays = _sequence_arrays(len(sequences),width)
    for i,sequence in enumerate(sequences):
        for field in PHRASE_SEQUENCE_FIELDS:
            if field=='atoms':arrays[field][i,:,:sequence[field].shape[-1]] = sequence[field]
            else:arrays[field][i] = sequence[field]
    return {prefix+field:torch.from_numpy(value) for field,value in arrays.items()}


def _positive_ids(fingerprints):
    identities = {}
    result = np.empty(len(fingerprints),dtype=np.int64)
    for i,value in enumerate(fingerprints):
        key = tuple(int(x) for x in value)
        result[i] = identities.setdefault(key,len(identities))
    return torch.from_numpy(result)


def _visible_sequence_fingerprint(sequence):
    """Positive identity follows only the configuration actually encoded.

    Unselected events remain in the full provenance fingerprint, but cannot
    distinguish contrastive labels whose 32-event encoder inputs are equal.
    Padding widths and absolute song positions are deliberately excluded.
    """
    digest = hashlib.blake2b(digest_size=16,person=b'phrasevisiblev2')
    indices = np.flatnonzero(sequence['mask'])
    if len(indices):
        first = indices[0]
        origin_beat,origin_second = sequence['beats'][first],sequence['seconds'][first]
        for i in indices:
            atoms = tuple(int(atom) for atom in sequence['atoms'][i] if atom!=0)
            record = (int(sequence['kind'][i]),int(sequence['shape'][i]),
                tuple(bool(value) for value in sequence['phase_mask'][i]),atoms,
                round(float(sequence['beats'][i]-origin_beat),9),
                round(float(sequence['seconds'][i]-origin_second),9),
                round(float(sequence['gap_beats'][i]),9),round(float(sequence['gap_seconds'][i]),9))
            digest.update(repr(record).encode('utf-8'));digest.update(b'\0')
    return np.frombuffer(digest.digest(),dtype='<i8').copy()


def collate_phrase_sequences(items, *, pack_references=True):
    """Deduplicate by archive identity and segment, never by original song ID.

    An augmentation owns a distinct archive, so its edited events cannot be
    accidentally replaced by the unedited chart's sequence during collation.
    """
    if not all('_phrase_sequence_owner' in item for item in items):
        raise ValueError('Cannot mix ordered and legacy phrase rows')
    length = max(len(item['history_kind']) for item in items)
    output = {}
    for target in (False,True):
        prefix = 'target_phrase_sequence_' if target else 'phrase_sequence_'
        segment_key = '_target_phrase_sequence_segments' if target else '_phrase_sequence_segments'
        shape = (len(items),length) if target else (len(items),length,PHRASE_MEMORY_SLOTS)
        inverse = np.full(shape,-1,dtype=np.int64)
        identities,unique = {},[]
        for batch,item in enumerate(items):
            owner = item['_phrase_sequence_owner']
            segments = item[segment_key]
            source = item[prefix+'inverse'].numpy()
            used = set(range(len(segments)))
            if target and 'target_phrase_mask' in item and 'target_phrase_weight' in item:
                active = (item['target_phrase_mask'].bool() &
                          item['target_phrase_weight'].gt(0)).numpy()
                used = {int(value) for value in source[active] if value >= 0}
            mapping = np.full(len(segments),-1,dtype=np.int64)
            for local in sorted(used):
                segment = segments[local]
                identity = id(owner),segment
                if identity not in identities:
                    identities[identity] = len(unique)
                    unique.append(owner.sequence(segment))
                mapping[local] = identities[identity]
            destination = inverse[batch,:len(source)]
            valid = source>=0
            destination[valid] = mapping[source[valid]]
        if target or pack_references:output.update(_pack_sequences(unique,prefix))
        output[prefix+'inverse'] = torch.from_numpy(inverse)
        if target:
            fingerprints = np.stack([sequence['_fingerprint'] for sequence in unique]) if unique else np.empty((0,2),dtype=np.int64)
            visible = np.stack([sequence['_visible_fingerprint'] for sequence in unique]) if unique else np.empty((0,2),dtype=np.int64)
            output[prefix+'fingerprint'] = torch.from_numpy(fingerprints)
            output[prefix+'visible_fingerprint'] = torch.from_numpy(visible)
            output[prefix+'positive_ids'] = _positive_ids(visible)
    return output


def prepare_candidate_phrase_inputs(index, queries, candidate_beats, *, supervise=False, pack=True):
    """Read the landing segment from a committed prefix at each query index.

    Candidate beats are proposed clock positions, never target configurations.
    Reference visibility is strictly before the query, including a segment
    that was the departure segment but is complete at the candidate landing.
    """
    result = index.rows_at(queries, candidate_beats, supervise=supervise)
    result.update(index.sequence_rows(queries, candidate_beats,pack=pack))
    return {'candidate_'+key:value for key,value in result.items()}


def collate_candidate_phrase_sequences(items, *, pack_references=True):
    """Share the ordered segment deduplication path for landing references."""
    converted = []
    for item in items:
        selected = {key[len('candidate_'):]:value for key,value in item.items()
                    if key.startswith('candidate_')}
        selected['history_kind'] = item['history_kind']
        converted.append(selected)
    result = collate_phrase_sequences(converted,pack_references=pack_references)
    return {'candidate_'+key:value for key,value in result.items()}


def share_aligned_phrase_sequences(items, output):
    """One reference bank for departure and landing in the same batch.

    Archive owner and segment form the identity. Both inverse tables index the
    same packed tensors, so a single forward-local GRU encoding serves both.
    """
    identities,unique = {},[]
    replacements = {}
    for source_prefix in ('','candidate_'):
        shape = tuple(output[source_prefix+'phrase_sequence_inverse'].shape)
        inverse = np.full(shape,-1,dtype=np.int64)
        for batch,item in enumerate(items):
            owner = item[source_prefix+'_phrase_sequence_owner']
            segments = item[source_prefix+'_phrase_sequence_segments']
            mapping = np.empty(len(segments),dtype=np.int64)
            for local,segment in enumerate(segments):
                identity = id(owner),segment
                if identity not in identities:
                    identities[identity] = len(unique)
                    unique.append(owner.sequence(segment))
                mapping[local] = identities[identity]
            source = item[source_prefix+'phrase_sequence_inverse'].numpy()
            valid = source>=0
            inverse[batch,:len(source)][valid] = mapping[source[valid]]
        replacements[source_prefix+'phrase_sequence_inverse'] = torch.from_numpy(inverse)
    packed = _pack_sequences(unique,'phrase_sequence_')
    output.update(replacements)
    output.update(packed)
    for key,value in packed.items():
        output['candidate_'+key] = value
    output['candidate_phrase_sequence_shared'] = True
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
    def __init__(self, bank, end_beat, steps=(), *, sequence_codec=None, encoded=None):
        self.layout = music_phrase_layout(bank,end_beat)
        self.boundaries,self.audio,self.references = self.layout.boundaries,self.layout.audio,self.layout.references
        count = len(self.boundaries)-1
        self.indices = [[] for _ in range(count)]
        self.prefix = [[] for _ in range(count)]
        self.lanes = [None]*count
        self.lane_history = [[] for _ in range(count)]
        self.count = 0
        self._feature_cache = {}
        self.sequence_codec = sequence_codec
        self.sequence_events = [[] for _ in range(count)] if sequence_codec is not None else None
        self._sequence_cache = {}
        self._sequence_atom_count = 0
        self._max_sequence_atoms = 1
        for i,step in enumerate(steps):self.append(step,encoded=encoded[i][1] if encoded is not None else _ENCODE_UNSET)

    @property
    def nbytes(self):
        # Include Python/list overhead conservatively in the cache budget.
        sequences = (self.count*512+self._sequence_atom_count*16
                     +len(self.indices)*PHRASE_SEQUENCE_LENGTH*(128+self._max_sequence_atoms*8)
                     if self.sequence_events is not None else 0)
        return self.count*(PHRASE_FEATURE_DIM*8+336)+len(self.boundaries)*256+sequences

    def segment(self, beat):
        return min(len(self.indices)-1, max(0, bisect_right(self.boundaries, float(beat))-1))

    def append(self, step, *, encoded=_ENCODE_UNSET):
        segment = self.segment(step.end)
        self._feature_cache.pop(segment,None)
        self._sequence_cache.pop(segment,None)
        # EOS has no event and is only a supervision query.
        if step.kind in (2, 3):
            value, lane = _event_vector(step, self.boundaries[segment], self.boundaries[segment+1], self.lanes[segment])
            self.lanes[segment] = lane
            self.lane_history[segment].append(lane)
            self.indices[segment].append(self.count)
            previous = self.prefix[segment][-1] if self.prefix[segment] else 0.
            self.prefix[segment].append(value+previous)
            if self.sequence_events is not None:
                event = (encoded if encoded is not _ENCODE_UNSET else self.sequence_codec.encode(step.text)) if step.kind==2 else None
                atoms = tuple(event['atoms']) if event is not None else ()
                phase_mask = tuple(bool(x) for x in event['phase_mask']) if event is not None else (False,)*8
                # Preserve original event gaps, including clocks crossing a boundary.
                record = (step.kind,int(event['shape']) if event is not None else 0,phase_mask,atoms,
                          float(step.end),float(step.end_second),float(step.end-step.start),
                          float(step.end_second-step.start_second),str(step.end),str(step.end-step.start))
                self.sequence_events[segment].append(record)
                self._sequence_atom_count += len(atoms)
                self._max_sequence_atoms = max(self._max_sequence_atoms,len(atoms))
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
        result.sequence_codec = self.sequence_codec
        result._sequence_cache = {}
        result._sequence_atom_count = 0
        result._max_sequence_atoms = 1
        result.sequence_events = [] if self.sequence_events is not None else None
        if result.sequence_events is not None:
            for indices,events in zip(self.indices,self.sequence_events):
                keep = bisect_left(indices,begin)
                kept = events[:keep]
                result.sequence_events.append(kept)
                result._sequence_atom_count += sum(len(event[3]) for event in kept)
                result._max_sequence_atoms = max(result._max_sequence_atoms,max((len(event[3]) for event in kept),default=1))
        for step in steps[begin:end]: result.append(step)
        return result

    def _features(self, segment, query):
        pos = bisect_left(self.indices[segment], query)-1
        raw = self.prefix[segment][pos] if pos >= 0 else np.zeros(PHRASE_FEATURE_DIM, dtype=np.float32)
        return torch.from_numpy(np.sign(raw)*np.log1p(np.abs(raw)))

    def sequence(self,segment):
        """A bounded ordered archive; main stream events are never truncated."""
        if self.sequence_events is None:raise ValueError('Ordered sequence codec is not enabled')
        cached = self._sequence_cache.get(segment)
        if cached is not None:return cached
        events = self.sequence_events[segment]
        count = len(events)
        if count>PHRASE_SEQUENCE_LENGTH:
            # Nearest real event to equal-time anchors, with strict progression
            # and explicit first/last retention even for simultaneous events.
            times = np.asarray([event[4] for event in events])
            anchors = np.linspace(times[0],times[-1],PHRASE_SEQUENCE_LENGTH)
            candidates = np.searchsorted(times,anchors).clip(0,count-1)
            left = np.maximum(0,candidates-1)
            candidates = np.where(np.abs(times[left]-anchors)<np.abs(times[candidates]-anchors),left,candidates)
            chosen = [0]
            for i in range(1,PHRASE_SEQUENCE_LENGTH-1):
                chosen.append(min(count-(PHRASE_SEQUENCE_LENGTH-i),max(chosen[-1]+1,int(candidates[i]))))
            chosen.append(count-1)
        else:chosen = list(range(count))
        width = max((len(event[3]) for event in events),default=1)
        arrays = {key:value[0] for key,value in _sequence_arrays(1,width).items()}
        for i,position in enumerate(chosen):
            event = events[position]
            arrays['kind'][i],arrays['shape'][i] = event[:2]
            arrays['phase_mask'][i] = event[2]
            arrays['atoms'][i,:len(event[3])] = event[3]
            for field,value in zip(('beats','seconds','gap_beats','gap_seconds'),event[4:8]):arrays[field][i] = value
            arrays['mask'][i] = True
        # Full unselected events remain provenance diagnostics. They cannot
        # define negatives for an encoder that never observes those events.
        digest = hashlib.blake2b(digest_size=16,person=b'orderedphrasev2')
        if events:
            beat_origin = Fraction(events[0][8])-Fraction(events[0][9])
            second_origin = events[0][5]-events[0][7]
            for event in events:
                digest.update(repr((event[0],event[1],event[2],event[3],
                    str(Fraction(event[8])-beat_origin),event[9],
                    round(event[5]-second_origin,9),round(event[7],9))).encode('utf-8'))
                digest.update(b'\0')
        arrays['_fingerprint'] = np.frombuffer(digest.digest(),dtype='<i8').copy()
        arrays['_visible_fingerprint'] = _visible_sequence_fingerprint(arrays)
        self._sequence_cache[segment] = arrays
        return arrays

    def sequence_rows(self,queries,cursors,*,target_mask=None,pack=True):
        """Only inverse IDs scale with queries; actual sequence rows are unique.

        Training may defer tensor packing until archive owners are deduplicated
        by the collator. Default eager packing retains the inference API.
        """
        queries = np.asarray(queries,dtype=np.int64)
        cursors = np.asarray(cursors,dtype=np.float64)
        segments = np.clip(np.searchsorted(self.boundaries,cursors,side='right')-1,0,len(self.indices)-1)
        refs = np.full((len(queries),PHRASE_MEMORY_SLOTS),-1,dtype=np.int64)
        for segment in np.unique(segments):
            group = np.flatnonzero(segments==segment)
            for slot,reference in enumerate(self.references[segment]):
                indices = self.indices[reference]
                if not indices:continue
                visible = (queries[group]>indices[-1]) & (self.boundaries[reference+1]<=self.boundaries[segment])
                refs[group[visible],slot] = reference
        reference_segments = np.unique(refs[refs>=0])
        inverse = np.full_like(refs,-1)
        valid = refs>=0
        inverse[valid] = np.searchsorted(reference_segments,refs[valid])
        target_inverse = np.full(len(queries),-1,dtype=np.int64)
        eligible = np.zeros(len(queries),dtype=np.bool_) if target_mask is None else np.asarray(target_mask,dtype=np.bool_)
        target_segments = np.unique(segments[eligible])
        if len(target_segments):target_inverse[eligible] = np.searchsorted(target_segments,segments[eligible])
        references = [self.sequence(int(segment)) for segment in reference_segments] if pack else []
        targets = [self.sequence(int(segment)) for segment in target_segments]
        result = _pack_sequences(references,'phrase_sequence_') if pack else {}
        if pack:result.update(_pack_sequences(targets,'target_phrase_sequence_'))
        fingerprints = np.stack([target['_fingerprint'] for target in targets]) if targets else np.empty((0,2),dtype=np.int64)
        visible = np.stack([target['_visible_fingerprint'] for target in targets]) if targets else np.empty((0,2),dtype=np.int64)
        result.update(phrase_sequence_inverse=torch.from_numpy(inverse),
            target_phrase_sequence_inverse=torch.from_numpy(target_inverse),
            target_phrase_sequence_fingerprint=torch.from_numpy(fingerprints),
            target_phrase_sequence_visible_fingerprint=torch.from_numpy(visible),
            target_phrase_sequence_positive_ids=_positive_ids(visible),
            _phrase_sequence_owner=self,_phrase_sequence_segments=tuple(int(segment) for segment in reference_segments),
            _target_phrase_sequence_segments=tuple(int(segment) for segment in target_segments))
        return result

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
