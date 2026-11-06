"""Event-aligned computational slices. No synthetic end-of-window targets."""
from collections import OrderedDict
from fractions import Fraction
import bisect
import hashlib
import json
import math
import numpy as np
import torch
from .data import ComposerV2Dataset,collate_composer_v2
from .stream_codec import POLICY,StreamCodec
from . import timing_recovery


_SELECT_REFERENCE = object()
_REFERENCE_CACHE_SONGS = 128
_REFERENCE_QUERY_ROWS = 1024


def _song_cache_key(song):
    clock = song.clock
    content = (str(song.audio_path) + str(clock.beats) + str(clock.bpms) + str(clock.seconds)
               + json.dumps(song.audio.identity, sort_keys=True))
    return hashlib.sha256(content.encode()).hexdigest()


class StreamDataset(ComposerV2Dataset):
    def __init__(self,*args,refresh_events=16,timing_recovery_policy='off',augmentation_seed=0,**kwargs):
        super().__init__(*args,sequence_policy=POLICY,**kwargs)
        if not isinstance(self.codec,StreamCodec):raise ValueError('Continuous training requires a new vocabulary and fresh checkpoint')
        self.refresh_events=int(refresh_events)
        if self.refresh_events<1:raise ValueError('refresh_events must be positive')
        self.tile_cache=OrderedDict();self.windows=[];self.ranges=[];self.lengths=[]
        self._remaining_references = []
        for song_index,song in enumerate(self.songs):
            first_window = len(self.windows)
            token_lengths = song.event_token_lengths
            song.event_token_lengths = []
            begin=0
            while begin<len(song.points):
                end=begin;tokens=0
                while end<len(song.points) and end-begin<self.refresh_events:
                    length=token_lengths[end]
                    if tokens+length+1>self.max_tokens//2:break
                    tokens+=length;end+=1
                if end==begin:raise ValueError(f'One whole event exceeds target budget: {song.source_path}, {song.points[begin].beat}')
                self.windows.append((song_index,begin));self.ranges.append(end);self.lengths.append(tokens+1)
                begin=end
            # Empty charts and actual song endings still have one real EOS target.
            if not song.points:self.windows.append((song_index,0));self.ranges.append(0);self.lengths.append(1)
            self._remaining_references.append(len(self.windows) - first_window)
        self._reference_starts = [None] * len(self.windows)
        self._reference_ready = bytearray(len(self.windows))
        self._reference_descriptors = OrderedDict()
        if timing_recovery_policy not in ('off',timing_recovery.POLICY):
            raise ValueError('Unknown timing recovery policy')
        self.recovery_sources = {}
        self.training_epoch = 0
        self.augmentation_seed = int(augmentation_seed)
        self._recovery_span = min(timing_recovery.SPAN, Fraction(self.history_beats))
        self._recovery_phases = frozenset(self.codec.phases.values())
        self._timing_ids = timing_recovery.timing_ids(self.codec)
        if timing_recovery_policy != 'off':
            for index in range(7,len(self.windows),8):
                si,begin=self.windows[index]
                if begin<2:continue
                song=self.songs[si]
                first=bisect.bisect_left(song.point_beats,song.point_beats[begin-1]-self._recovery_span,hi=begin)
                if any(timing_recovery.is_tap(p) for p in song.points[first:begin-1]):
                    self.recovery_sources[len(self.windows)] = index
                    self.windows.append(self.windows[index])
                    self.ranges.append(self.ranges[index])
                    self.lengths.append(self.lengths[index])
        self.audit.update(windows=len(self.windows),sequence_policy=POLICY,
            segmentation='whole_events_compute_only',refresh_events=self.refresh_events,eos='true_chart_end_only')
        self.audit['timing_recovery'] = dict(policy=timing_recovery_policy, views=len(self.recovery_sources),
            original_windows=len(self.windows)-len(self.recovery_sources),target='authored_future_time_only')

    def set_epoch(self, epoch):
        self.training_epoch = int(epoch)
    def _reference_stream(self, song, points, cursor, *, prefix_end=None):
        stop = cursor - self.history_beats - self.window_beats
        if prefix_end is None:
            beats = [point.beat for point in points if point.beat <= stop]
            count = len(beats)
        else:
            beats = song.point_beats
            count = bisect.bisect_right(beats, stop, hi=prefix_end)
        if count == 0:
            return None
        indices = np.linspace(0, count - 1, 64, dtype=int) if count > 64 else range(count)
        candidates = [beats[index] for index in indices]
        starts = [cursor, *candidates]
        cache = song.descriptor_cache
        descriptors = {start: cache[start] for start in starts if start in cache}
        missing = list(dict.fromkeys(start for start in starts if start not in descriptors))
        if missing:
            offsets = np.arange(self.window_beats * 4) / 4
            starts_per_read = max(1, _REFERENCE_QUERY_ROWS // len(offsets))
            for index in range(0, len(missing), starts_per_read):
                chunk = missing[index:index + starts_per_read]
                beat_queries = np.asarray(chunk, dtype=np.float64)[:, None] + offsets
                unique, inverse = np.unique(beat_queries.ravel(), return_inverse=True)
                mel, sem, valid = song.audio.sample(song.clock.at_floats(unique))
                for start, rows in zip(chunk, inverse.reshape(beat_queries.shape)):
                    # Keep the original reduction order and validity per window.
                    values = sem if np.any(valid[rows]) else mel
                    vec = values[rows].mean(0)
                    descriptors[start] = vec / max(float(np.linalg.norm(vec)), 1e-8)
        for start in starts:
            if start not in cache:
                if len(cache) >= 128:
                    cache.pop(next(iter(cache)))
                cache[start] = descriptors[start]
        query = descriptors[cursor]
        return max(candidates, key=lambda beat: float(np.dot(query, descriptors[beat])))
    def tile(self,song,anchor):
        if not hasattr(self,'tile_cache'):self.tile_cache=OrderedDict()
        key=(str(song.audio_path),str(song.clock.beats),str(song.clock.bpms),str(song.clock.seconds),int(anchor),self.future_beats)
        if key in self.tile_cache:self.tile_cache.move_to_end(key);return self.tile_cache[key]
        ticks=np.arange(-48,(self.future_beats+2)*24,dtype=np.int64)
        beats=(int(anchor)*24+ticks)/24.;seconds=song.clock.at_floats(beats)
        mel,sem,valid=song.audio.sample(seconds)
        result=dict(local_audio=torch.as_tensor(mel,dtype=torch.float32),
            local_semantic=torch.as_tensor(sem,dtype=torch.float32),local_semantic_mask=torch.as_tensor(valid,dtype=torch.bool),
            local_mask=torch.as_tensor((seconds>=0)&(seconds<=song.audio.duration_sec),dtype=torch.bool),
            local_beats=torch.tensor(beats,dtype=torch.float32))
        self.tile_cache[key]=result
        while len(self.tile_cache)>64:self.tile_cache.popitem(last=False)
        return result
    def build_stream(self,song,points,cursor,*,target=(),ended=False,training=False,
                     reference_start=_SELECT_REFERENCE,_prefix_end=None):
        cursor=Fraction(cursor);boundary=max(Fraction(0),cursor-self.history_beats)
        if _prefix_end is None:
            known = list(points)
            if any(p.beat > cursor for p in known):
                raise ValueError('Uncommitted future chart in continuous prefix')
            history = [p for p in known if p.beat >= boundary]
        else:
            # Dataset windows use the sorted source with an exclusive prefix index.
            known = points
            history_index = bisect.bisect_left(song.point_beats, boundary, hi=_prefix_end)
            history = known[history_index:_prefix_end]
        # Training can reuse the immutable native index, with strict causal filtering.
        state=song.sustain if training else None
        if not training and any('[' in p.note_text for p in known):
            from maichart.generation.sustain_state import SustainStateIndex
            from maichart.event_timing import EventTempo
            state=SustainStateIndex(known,[EventTempo(b,t) for b,t in zip(song.clock.beats,song.clock.bpms)],song.clock.seconds[0])
        def active_sources(at):
            if state is None:return set()
            second=song.clock.at(at)
            return {Fraction(m['source_beat']) for m in state._candidate_markers(second)
                if m['source_beat']<=cursor and float(m['end_second'])>second}
        older=active_sources(boundary)
        active=active_sources(cursor)
        if reference_start is _SELECT_REFERENCE:
            reference_start = self._reference_stream(song, known, cursor, prefix_end=_prefix_end)
        reference = []
        if _prefix_end is None:
            carry = [p for p in known if p.beat < boundary and p.beat in older]
            if reference_start is not None:
                stop = min(reference_start + self.window_beats, boundary)
                reference = [p for p in known if reference_start <= p.beat < stop]
        else:
            beats = song.point_beats
            carry = []
            for beat in sorted(older):
                first = bisect.bisect_left(beats, beat, hi=history_index)
                end = bisect.bisect_right(beats, beat, lo=first, hi=history_index)
                carry.extend(known[first:end])
            if reference_start is not None:
                first = bisect.bisect_left(beats, reference_start, hi=history_index)
                end = bisect.bisect_left(beats, reference_start + self.window_beats, lo=first, hi=history_index)
                reference = known[first:end]
        codec = self.codec
        carry_ids = codec.encode_points(carry)
        history_ids = codec.encode_points(history)
        reference_ids = codec.encode_points(reference)
        prefix_length = 5 + len(carry_ids) + len(history_ids) + len(reference_ids)
        budget = self.max_tokens // 2
        if prefix_length > budget:
            prefix_length -= len(reference_ids)
            reference_ids = []
            kept_start = 0
            carried_beats = {point.beat for point in carry}
            while prefix_length > budget and kept_start < len(history):
                removed = history[kept_start]
                kept_start += 1
                if removed.beat in active and removed.beat not in carried_beats:
                    carry.append(removed)
                    carried_beats.add(removed.beat)
                else:
                    prefix_length -= len(codec.encode_points([removed]))
            if prefix_length > budget:
                raise ValueError('Mandatory sustained context exceeds token budget; increase max_tokens')
            if kept_start:
                carry_ids = codec.encode_points(carry)
                history_ids = codec.encode_points(history[kept_start:])
        prefix = ([codec.bos_id, codec.carry_id] + carry_ids + [codec.ctx_id] + history_ids
                  + [codec.ref_id] + reference_ids + [codec.tgt_id])
        suffix=self.codec.encode_points(target)
        if ended:suffix.append(self.codec.eos_id)
        full=prefix+suffix
        if training and len(full)-1>self.max_tokens:raise ValueError('Complete event slice exceeds token budget')
        ids=full[:-1] if training else full
        full_clocks = self.codec.token_clocks(full, cursor=cursor)
        clocks = full_clocks[:-1] if training else full_clocks
        labels=np.full(len(ids),-100,dtype=np.int64)
        if training:labels[len(prefix)-1:]=full[len(prefix):]
        weights=(labels!=-100).astype(np.float32)
        regions = (song.quality or {}).get('regions', [])
        if training and regions:
            seconds = song.clock.at_floats(np.asarray(full_clocks[1:]))
            for region in regions:
                weights[(seconds>=float(region['start_sec']))&(seconds<float(region['end_sec']))]*=float(region.get('weight',1.))
        anchors=sorted({math.floor(t) for t in clocks});lookup={a:i for i,a in enumerate(anchors)}
        # Adjacent query tiles overlap heavily. Read/project each source frame
        # once; gather the small learned vectors into independent encoder tiles.
        relative=np.arange(-48,(self.future_beats+2)*24,dtype=np.int64)
        all_ticks=np.asarray(anchors)[:,None]*24+relative[None,:]
        ticks=np.unique(all_ticks);tile_frames=np.searchsorted(ticks,all_ticks)
        seconds=song.clock.at_floats(ticks/24.)
        sample_frames = getattr(song.audio, 'sample_for_training', None) if training else None
        if sample_frames is not None:
            mel, semantic, semantic_right, semantic_weight, valid = sample_frames(seconds)
        else:
            mel,semantic,valid=song.audio.sample(seconds)
        summary=song.summary
        out=dict(input_ids=torch.tensor(ids),labels=torch.tensor(labels),token_weights=torch.tensor(weights),
            token_beats=torch.tensor(clocks,dtype=torch.float32),token_mask=torch.ones(len(ids),dtype=torch.bool),
            tile_ids=[lookup[math.floor(t)] for t in clocks],anchors=anchors,
            bank_audio=torch.as_tensor(mel,dtype=torch.float32),bank_semantic=torch.as_tensor(semantic),
            bank_semantic_mask=torch.as_tensor(valid,dtype=torch.bool),
            tile_frames=torch.tensor(tile_frames,dtype=torch.long),
            local_beats=torch.tensor(all_ticks/24.,dtype=torch.float32),
            local_mask=torch.as_tensor(((seconds>=0)&(seconds<=song.audio.duration_sec))[tile_frames],dtype=torch.bool),
            condition=torch.as_tensor(song.condition),condition_known=torch.as_tensor(song.condition_known if song.condition_known is not None else [True]+[False]*15),
            sample_weight=torch.tensor(float((song.quality or {}).get('weight',1.))),
            song_audio=torch.as_tensor(summary['mel']),song_semantic=torch.as_tensor(summary['semantic']),
            song_semantic_mask=torch.as_tensor(summary['semantic_valid']),song_beats=torch.as_tensor(summary['beats'],dtype=torch.float32),
            song_mask=torch.ones(len(summary['times_sec']),dtype=torch.bool),start_beat=Fraction(0),cursor=cursor,
            song_id=song.song_id,reference_start=reference_start,pad_id=self.codec.pad_id,
            song_cache_key=song.stream_cache_key or _song_cache_key(song))
        if sample_frames is not None:
            out['bank_semantic_right'] = torch.as_tensor(semantic_right)
            out['bank_semantic_weight'] = torch.as_tensor(semantic_weight)
        return out
    def __getitem__(self,index):
        source = self.recovery_sources.get(index)
        if source is not None:
            return self._recovery_item(source)
        si,begin=self.windows[index];song=self.songs[si];end=self.ranges[index]
        cursor=song.points[begin-1].beat if begin else Fraction(0)
        reference_start=self._prepare_reference(index)
        if song.stream_cache_key is None:
            song.stream_cache_key = _song_cache_key(song)
        return self.build_stream(song, song.points, cursor, target=song.points[begin:end],
                                 ended=end==len(song.points), training=True,
                                 reference_start=reference_start,_prefix_end=begin)

    def _prepare_reference(self,index):
        si,begin=self.windows[index];song=self.songs[si]
        cursor=song.points[begin-1].beat if begin else Fraction(0)
        if not self._reference_ready[index]:
            self._reference_starts[index] = self._reference_stream(song, song.points, cursor, prefix_end=begin)
            self._reference_ready[index] = 1
            self._remaining_references[si] -= 1
            if self._remaining_references[si] and song.descriptor_cache:
                self._reference_descriptors[si] = song.descriptor_cache
                self._reference_descriptors.move_to_end(si)
                if len(self._reference_descriptors) > _REFERENCE_CACHE_SONGS:
                    _, evicted = self._reference_descriptors.popitem(last=False)
                    evicted.clear()
            else:
                self._reference_descriptors.pop(si, None)
                song.descriptor_cache.clear()
        return self._reference_starts[index]

    def _recovery_item(self, source):
        si,begin=self.windows[source];song=self.songs[si];end=self.ranges[source]
        cursor=song.points[begin-1].beat
        seed=self.augmentation_seed+source*1009+self.training_epoch*1000003
        changed=timing_recovery.perturb_prefix(song.points[:begin],self._recovery_phases,
            seed,dense=(source//8+self.training_epoch)%2==0,span=self._recovery_span)
        if changed is None:
            raise RuntimeError('Timing recovery view lost its eligible committed prefix')
        prefix,_=changed
        item=self.build_stream(song,prefix,cursor,target=song.points[begin:end],
            ended=end==len(song.points),training=True,reference_start=self._prepare_reference(source))
        keep=torch.tensor([int(token) in self._timing_ids for token in item['labels']],dtype=torch.bool)
        item['labels']=item['labels'].masked_fill(~keep,-100)
        item['token_weights']=item['token_weights']*keep*2.
        return item


def collate_stream(items):
    # The existing collator handles sequence fields; tiles have a distinct batch axis.
    temporary=[{**item,'local_audio':item['bank_audio'][:1],
        'local_semantic':item['bank_semantic'][:1],'local_semantic_mask':item['bank_semantic_mask'][:1],
        'local_mask':item['local_mask'][0,:1],'local_beats':item['local_beats'][0,:1]} for item in items]
    out=collate_composer_v2(temporary)
    for k in ('bank_audio','bank_semantic','bank_semantic_mask','local_mask','local_beats'):out[k]=torch.cat([item[k] for item in items],0)
    if any('bank_semantic_right' in item for item in items):
        out['bank_semantic_right'] = torch.cat([item.get('bank_semantic_right', item['bank_semantic']) for item in items])
        out['bank_semantic_weight'] = torch.cat([item['bank_semantic_weight']
            if 'bank_semantic_weight' in item else torch.zeros(len(item['bank_semantic']), dtype=torch.float64)
            for item in items])
    owners=[];ids=[];offset=0;bank_owners=[];frames=[];bank_offset=0
    for row,item in enumerate(items):
        owners.extend([row]*len(item['anchors']));ids.append([i+offset for i in item['tile_ids']]+[-1]*(out['input_ids'].size(1)-len(item['tile_ids'])))
        offset+=len(item['anchors'])
        bank_owners.extend([row]*len(item['bank_audio']));frames.append(item['tile_frames']+bank_offset);bank_offset+=len(item['bank_audio'])
    out['tile_owner']=torch.tensor(owners);out['tile_ids_cpu']=ids
    out['bank_owner']=torch.tensor(bank_owners);out['tile_frames']=torch.cat(frames)
    out['cursor']=[item['cursor'] for item in items]
    return out
