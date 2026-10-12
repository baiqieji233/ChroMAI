"""Song-level organization and causal training inputs for the new family."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.event_timing import EventPoint, parse_event_chart, seconds_at_beat
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.event_audio import (
    audio_segment_tensor, extract_event_audio, _normalize_tempo_map,
    _beats_at_audio_offsets, segment_frame_times,
)
from maichart.generation.sustain_state import SustainStateIndex
from maichart.form.codec import RationalCodec
from maichart.form.recurrence import analyze_recurrence, CausalMotifMemory
from maichart.form.timing import MetricSpan, normalize_meter_map
from maichart.form.revision import DraftPoint
from maichart.training.chart_sanitization import sanitize_chart
from maichart.training.dataset_catalog import _source_asset
from maichart.training.event_dataset import inspect_chart, numeric_level


class _AudioStore:
    def __init__(self, loader, cache_dir, budget=256*1024*1024):
        self.loader,self.cache_dir,self.budget=loader,cache_dir,budget
        self.entries=OrderedDict();self.bytes=0

    def get(self,path):
        key=str(path)
        if key in self.entries:
            self.entries.move_to_end(key);return self.entries[key]
        audio=self.loader(path,cache_dir=self.cache_dir)
        self.entries[key]=audio;self.bytes+=audio.features.nbytes+audio.times.nbytes
        while len(self.entries)>1 and self.bytes>self.budget:
            _,old=self.entries.popitem(last=False);self.bytes-=old.features.nbytes+old.times.nbytes
        return audio


class _LazyAudio:
    def __init__(self,store,path): self.store,self.path=store,path
    def __getattr__(self,key): return getattr(self.store.get(self.path),key)


@dataclass
class FormSong:
    audio: object
    tempo: list
    first: float
    meters: list
    spans: list
    relations: object
    audit: dict
    regions: list[MetricSpan] | None = None
    span_region_indices: list[int] | None = None

    def __post_init__(self):
        # Old direct constructors treat every supplied span as its own region.
        if self.regions is None:
            self.regions = list(self.spans)
        if self.span_region_indices is None:
            self.span_region_indices = list(range(len(self.spans)))
        if len(self.span_region_indices) != len(self.spans):
            raise ValueError("span_region_indices must map every computational span")
        if any(index < 0 or index >= len(self.regions) for index in self.span_region_indices):
            raise ValueError("span_region_indices contains an unknown musical region")


def organize_song(audio, tempo, first, *, meters=None, spans=None, min_seconds=2.0, max_seconds=14.0):
    meters = normalize_meter_map(meters)
    if spans is None:
        from maichart.form.segmentation import segment_music
        segmented = segment_music(audio, tempo, first, meters=meters,
                                  min_seconds=min_seconds, max_seconds=max_seconds)
        spans, regions = segmented.spans, segmented.region_spans
        span_region_indices = list(segmented.span_region_indices)
        audit = dict(segmented.audit)
        audit['musical_regions'] = [{"start":str(r.start),"end":str(r.end),"reason":r.end_reason} for r in segmented.regions]
    else:
        spans, audit = list(spans), {"source": "explicit_spans"}
        regions = list(spans)
        span_region_indices = list(range(len(spans)))
    relations = analyze_recurrence(audio, regions, tempo, first)
    audit['region_count'] = len(regions)
    audit['computational_span_count'] = len(spans)
    return FormSong(audio, list(tempo), float(first), meters, spans, relations, audit,
                    regions, span_region_indices)


def form_audio(song, start, end):
    from maichart.form.timing import metrical_features
    lo = float(seconds_at_beat(start, song.tempo, song.first))
    hi = float(seconds_at_beat(end, song.tempo, song.first))
    values = audio_segment_tensor(song.audio, lo, hi, start_beat=start, first=song.first,
        tempo_changes=song.tempo, bpm=song.tempo[0].bpm)
    # Legacy bar/phrase phase was hard-coded to 4/16 beats. Do not carry that
    # assumption into the new branch; explicit meter features replace it.
    values[:, -4:] = 0
    stamps = segment_frame_times(lo, hi)
    beats = _beats_at_audio_offsets(np.asarray(stamps) - song.first,
                                   _normalize_tempo_map(song.tempo, song.tempo[0].bpm))
    rational = [Fraction(str(float(b))).limit_denominator(96000) for b in beats]
    metric = torch.from_numpy(metrical_features(rational, song.meters))
    if len(metric) != len(values):
        raise ValueError("Metrical feature clock differs from acoustic clock")
    return torch.cat((values, metric), -1)


def input_example(song, index, history, *, condition=None, codec=None, task=0,
                  draft=(), right=(), max_memory_tokens=512, memory_strength=1.0,
                  state_index=None):
    codec = codec or RationalCodec()
    span = song.spans[index]
    region_index = song.span_region_indices[index]
    region = song.regions[region_index]
    history = tuple(history)
    if any(p.beat >= span.start for p in history):
        raise ValueError("History includes current/future chart targets")
    if task == 0 and (len(draft) or len(right)):
        raise ValueError("Fresh generation cannot read draft/right targets")
    if any(p.beat < span.end for p in right):
        raise ValueError("Right context overlaps edit region")
    memory = CausalMotifMemory(song.regions)
    for prior in song.regions:
        if prior.end <= region.start:
            memory.commit(prior.index, [p for p in history if prior.start <= p.beat < prior.end])
    ref = memory.reference(region.index, song.relations)
    # A recurrence is expressed across the full musical region. A compute cap
    # receives only its overlapping portion and never becomes a motif boundary.
    points = [p for p in ref.points if span.start <= p.beat < span.end]
    motif = codec.bounded_memory(points, origin=span.start, max_tokens=max_memory_tokens, newest=False)
    relation = ref.relation
    relative = []
    extent = max(float(song.regions[-1].end), 1.)
    for s in song.regions:
        relative.append([(float(s.start)-float(region.start))/extent,
                         (float(s.end)-float(region.start))/extent, float(s.index == region_index)])
    form = np.concatenate((song.relations.descriptors, np.asarray(relative,dtype=np.float32)),axis=1)
    # Bound the audio-structure memory independently of the decoder; always
    # retain the current and retrieved source segment among sampled rows.
    if len(form) > 128:
        keep = set(np.linspace(0,len(form)-1,126,dtype=int).tolist()) | {region_index}
        if relation is not None: keep.add(relation.source)
        form = form[sorted(keep)]
    hist_origin = history[0].beat if history else span.start
    states, padding, state_meta = (state_index or SustainStateIndex(history,song.tempo,song.first)).at(span.start)
    # A reused index must still be causal. Its implementation filters all heads
    # at or after the boundary, including future-dependent release flags.
    condition = condition or GenerationCondition(radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
    return {
        "audio": form_audio(song,span.start,span.end), "form": torch.from_numpy(form),
        "history": torch.tensor(codec.bounded_memory(history,origin=hist_origin,max_tokens=max_memory_tokens)),
        "motif": torch.tensor(motif),
        "draft": torch.tensor(codec.bounded_memory(draft,origin=span.start,max_tokens=max_memory_tokens,newest=False)),
        "right": torch.tensor(codec.bounded_memory(right,origin=span.end,max_tokens=max_memory_tokens,newest=False)),
        "condition": condition.to_vector(), "task": torch.tensor(task),
        "action_state": states, "action_state_padding_mask": padding,
        "relation": torch.tensor([relation.similarity,relation.confidence,relation.duration_ratio,
            (region_index-relation.source)/max(len(song.regions),1)] if relation else [0.,0.,1.,0.]),
        "memory_strength": torch.tensor(float(memory_strength)),
        "skeleton": torch.tensor(codec.bounded_memory((), origin=span.start,
            max_tokens=max_memory_tokens, newest=False)),
        "meta": {"span": span.to_dict(), "reference_source": ref.source,
                 "region": region.to_dict(), "region_index": region_index,
                 "state_count": state_meta["state_count"], "task": "draft" if task==0 else "reconstruction"},
    }


def relation_target(target, reference, span):
    # Soft similarity evidence, not a prescribed chorus/variation class.
    def signature(points):
        rhythm = np.zeros(48,dtype=np.float32)
        types = np.zeros(4,dtype=np.float32)
        for p in points:
            pos = min(47,max(0,int(float((p.beat-span.start)/(span.end-span.start))*48)))
            rhythm[pos] += 1
            types[3 if any(c in p.note_text for c in 'ABCDE') else 2 if '[' in p.note_text and 'h' not in p.note_text else 1 if 'h' in p.note_text else 0] += 1
        return rhythm,types
    a,b=signature(target),signature(reference)
    return torch.tensor([float(np.dot(x,y)/(max(np.linalg.norm(x)*np.linalg.norm(y),1e-8))) for x,y in zip(a,b)])


class FormDataset(Dataset):
    """Separate target schema; old checkpoints never consume these examples."""
    def __init__(self, manifest, *, cache_dir, max_tokens=2048, max_memory_tokens=512,
                 meters=None, seed=0, limit_charts=0, reconstruction=True,
                 min_level=12.,max_level=15.,min_difficulty_index=4,max_difficulty_index=6,
                 max_region_seconds=14., audio_loader=extract_event_audio):
        self.codec=RationalCodec(); self.items=[]; self.songs=[]; self.audit={"skipped":[],"filtered":[],"charts":[],"supervision":"draft_and_masked_reconstruction"}
        self.max_memory_tokens=max_memory_tokens; self.seed=seed
        self.audio_store=_AudioStore(audio_loader,cache_dir)
        path=Path(manifest).resolve(); data=json.loads(path.read_text(encoding='utf-8-sig'))
        annotations=None
        annotation_path=path.with_name('quality_annotations.json')
        if annotation_path.exists():
            from maichart.training.quality_annotations import load_quality_annotations
            annotations=load_quality_annotations(annotation_path)
        for entry in data.get('songs',[]):
            for difficulty in entry.get('difficulties',[]):
                slot=int(difficulty.get('difficulty_index',difficulty.get('index',0)))
                if not min_difficulty_index<=slot<=max_difficulty_index: continue
                if limit_charts and len(self.songs)>=limit_charts: break
                try:
                    level=numeric_level(difficulty.get('level_raw',difficulty.get('level')))
                    if not min_level<=level<=max_level: continue
                    source=_source_asset(entry['maidata_path'],path,data)
                    audio_path=_source_asset(entry['audio_path'],path,data)
                    original=parse_event_chart(source,slot)
                    filtered=sanitize_chart(original)
                    chart=filtered.chart; inspect_chart(chart)
                    if filtered.removed:
                        self.audit['filtered'].append({'source':str(source),'difficulty':slot,**filtered.to_dict()})
                        print(f'form_chart_filter source={source} removed={len(filtered.removed)}',flush=True)
                    record=None
                    if annotations:
                        from maichart.training.quality_annotations import resolve_annotation
                        for candidate in annotations.get('records',[]):
                            if int(candidate.get('difficulty_index',-1))==slot:
                                resolved=resolve_annotation(candidate,source,chart,slot,audio_path)
                                if resolved: record=resolved; break
                    if record and (record.get('label') in ('excluded','negative') or float(record.get('weight',1))<=0):
                        self.audit['skipped'].append({'source':str(source),'reason':'quality_excluded_or_negative'}); continue
                    weight=float((record or {}).get('weight',1.))
                    audio=self.audio_store.get(audio_path)
                    structure_path=source.parent/'music_structure.json'
                    structure=json.loads(structure_path.read_text(encoding='utf-8')) if structure_path.exists() else {}
                    meter_data=structure.get('meter_map',meters)
                    song=organize_song(audio,chart.tempo_changes,chart.audio_first,meters=meter_data,max_seconds=max_region_seconds)
                    song.audio=_LazyAudio(self.audio_store,audio_path)
                    # Local quality exclusions must not reappear as history.
                    excluded=[]
                    for region in (record or {}).get('regions',[]):
                        if float(region.get('weight',1))<=0: excluded.append((float(region['start_sec']),float(region['end_sec'])))
                    points=[p for p in chart.points if not any(lo<=seconds_at_beat(p.beat,chart.tempo_changes,chart.audio_first)<hi for lo,hi in excluded)]
                    idx=len(self.songs); self.songs.append((song,points,level,weight,SustainStateIndex(points,chart.tempo_changes,chart.audio_first)))
                    self.audit['charts'].append({'source':str(source),'difficulty':slot,'spans':len(song.spans),'links':len(song.relations.links),'unknown_meter':not bool(song.meters),
                        'chart_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                        'audio_sha256':(audio.metadata or {}).get('audio_sha256'),
                        'meter_map':[m.to_dict() for m in song.meters],
                        'quality':record,'weight':weight})
                    print('form_prepare '+json.dumps({'stage':'chart','charts':len(self.songs),
                        'source':str(source),'difficulty':slot,'spans':len(song.spans)},ensure_ascii=False),flush=True)
                    for span in song.spans:
                        lo=seconds_at_beat(span.start,song.tempo,song.first);hi=seconds_at_beat(span.end,song.tempo,song.first)
                        if any(lo<b and hi>a for a,b in excluded): continue
                        target=[p for p in points if span.start<=p.beat<span.end]
                        tokens=self.codec.encode(target,origin=span.start)
                        if len(tokens)>max_tokens:
                            self.audit['skipped'].append({'source':str(source),'span':span.index,'reason':'target_token_budget','tokens':len(tokens)});continue
                        self.items.append((idx,span.index,tokens,0))
                        if reconstruction and target: self.items.append((idx,span.index,tokens,1))
                except (OSError,ValueError,KeyError) as error:
                    self.audit['skipped'].append({'song':entry.get('song_id'),'difficulty':slot,'reason':str(error)})
        if not self.items: raise ValueError(f'No usable song-form examples: {self.audit["skipped"][:4]}')

    def __len__(self): return len(self.items)

    def __getitem__(self,index):
        song_index,span_index,tokens,task=self.items[index]
        song,points,level,weight,state_index=self.songs[song_index]
        span=song.spans[span_index]
        history=[p for p in points if p.beat<span.start]
        target=[p for p in points if span.start<=p.beat<span.end]
        draft=[];right=[]
        if task:
            rng=np.random.default_rng(self.seed+index)
            clean=rng.random()<.1
            selected=set() if clean else set(rng.choice(len(target),size=max(1,int(len(target)*.3)),replace=False).tolist())
            draft=[DraftPoint(p.beat,None if i in selected else p.note_text,i in selected) for i,p in enumerate(target)]
            right=[p for p in points if p.beat>=span.end]
        condition=GenerationCondition(level=level,radar=GENERATION_DEFAULTS,style=GENERATION_DEFAULTS)
        result=input_example(song,span_index,history,condition=condition,codec=self.codec,task=task,
            draft=draft,right=right,max_memory_tokens=self.max_memory_tokens,state_index=state_index)
        ref_index=result['meta']['reference_source']
        reference=[]
        if ref_index is not None and result['motif'].numel()>1:
            # The auxiliary target must see the same phase-aligned, clipped
            # reference as the encoder, not a separate zero-offset projection.
            reference=self.codec.decode(result['motif'].tolist(),origin=span.start)
        # This blueprint contains positions only. Every chart configuration is
        # masked, so training cannot reveal future note types or note text.
        skeleton=[DraftPoint(p.beat,None,True) for p in points if p.beat>=span.start]
        result['skeleton']=torch.tensor(self.codec.bounded_memory(
            skeleton,origin=span.start,max_tokens=self.max_memory_tokens,newest=False))
        result.update(decoder_input=torch.tensor(tokens[:-1]),target=torch.tensor(tokens[1:]),
                      relation_target=relation_target(target,reference,span),
                      relation_valid=torch.tensor(ref_index is not None),weight=torch.tensor(weight))
        return result


def collate_form(items):
    result={}
    padded=('audio','form','history','motif','draft','right','skeleton','action_state','decoder_input','target')
    for key in padded:
        if key not in items[0]: continue
        result[key]=torch.nn.utils.rnn.pad_sequence([i[key] for i in items],batch_first=True)
        if key in ('audio','form'):
            result[key+'_padding_mask']=torch.arange(result[key].size(1))[None,:]>=torch.tensor([len(i[key]) for i in items])[:,None]
    result['action_state_padding_mask']=torch.nn.utils.rnn.pad_sequence([i['action_state_padding_mask'] for i in items],batch_first=True,padding_value=True)
    for key in ('condition','task','relation','memory_strength','relation_target','relation_valid','weight'):
        if key in items[0]: result[key]=torch.stack([i[key] for i in items])
    result['meta']=[i['meta'] for i in items]
    return result
