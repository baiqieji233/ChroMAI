"""Event-level composer dataset built on the existing FormDataset/audio cache."""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
import json
import hashlib
from pathlib import Path
from collections import OrderedDict
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.form.data import FormDataset
from maichart.stream.grounding import detail_queries
from maichart.event_timing import seconds_at_beat
from maichart.generation.conditions import GenerationCondition
from maichart.stream.context import prefix_action_state
from maichart.stream.audio_bounds import generation_bounds
from .candidates import candidate_beats, candidate_features, candidate_index
from .events import atom_catalog,structure_signature
from maichart.generation.event_cli import _NoteGrammar
from maichart.form.type_guidance import _counts


@dataclass(frozen=True, slots=True)
class ComposerSample:
    song_index: int
    event_index: int
    previous: Fraction | None
    target: Fraction
    note_text: str
    candidates: tuple[Fraction, ...]
    history: tuple[tuple[int, ...], ...]
    history_features: tuple[tuple[float, ...], ...]
    action_state: tuple[float, ...]
    condition: tuple[float, ...]
    future_target: tuple[float, ...]
    eos_target: float


class ComposerDataset(Dataset):
    def __init__(self, manifest, *, cache_dir='cache/chart_composer_v1',
                 candidate_count=96, history_events=128, action_slots=8,
                 preparation_workers=1, audio_cache_gb=4., limit_charts=0,
                 quality_enabled=True, seed=0):
        self.codec_atoms=[]; self.atom_to_id={}; self.samples=[]; self.songs=[]
        self.candidate_count=int(candidate_count);self.history_events=int(history_events)
        if self.candidate_count < 16:
            raise ValueError('Composer candidate_count must be at least 16 for common onset and rest proposals')
        self.action_slots=int(action_slots);self.seed=int(seed);self.epoch=0
        self._feature_cache_budget_bytes=int(max(64*1024**2,min(float(audio_cache_gb)*1024**3*.25,1024*1024**2)))
        self.base=FormDataset(manifest,cache_dir=cache_dir,max_tokens=1_000_000,
            max_memory_tokens=max(256,history_events),max_region_seconds=999.,
            preparation_workers=preparation_workers,audio_cache_gb=audio_cache_gb,
            limit_charts=limit_charts,quality_enabled=quality_enabled,
            preserve_excluded_context=True,quality_profile='musical_evidence_v1',
            rhythm_geometry_profile='none',audio_profile='rational_audio_queries_v1')
        records=[]
        for record in self.base.songs:
            song,points,level,weight,state,condition,excluded,annotation=record
            records.append(record)
            for point in points:self.codec_atoms.extend(atom_catalog([point]))
        self.codec_atoms=sorted(set(self.codec_atoms))
        self.atom_to_id={atom:index+1 for index,atom in enumerate(self.codec_atoms)}
        self.candidate_total=0; self.candidate_misses=0; self.candidate_miss_examples=[]
        self.action_grammar=_NoteGrammar.from_atoms(self.codec_atoms)
        for song_index,record in enumerate(records):
            song,points,level,weight,state,condition,excluded,annotation=record
            if not points or weight<=0:continue
            song_sample_start=len(self.samples)
            source_song=self._source_song(song_index)
            audio_end_sec=generation_bounds(source_song.audio,trim_exact_silence=True)['effective_end_sec']
            previous=None
            future_targets=[]
            for future_index in range(len(points)):
                window=points[future_index:min(len(points),future_index+4)]
                if window:
                    before_second=float(seconds_at_beat(points[future_index-1].beat,song.tempo,song.first)) if future_index else 0.
                    gaps=[max(0.,float(seconds_at_beat(p.beat,song.tempo,song.first))-before_second) for p in window]
                    texts=[str(p.note_text) for p in window]
                    counts=[int((parsed.sum() if (parsed:=_counts(text)) is not None else 0.)) for text in texts]
                    head_rate=float(sum(counts)/max(sum(gaps),1e-3))
                    chord=sum(int('/' in t) for t in texts)/len(texts)
                    slide=sum(int(any(c in t for c in '-<>^pqszvwV')) for t in texts)/len(texts)
                    hold=sum(int('h' in t) for t in texts)/len(texts)
                    touch=sum(int(any(c in t for c in 'ABCDE')) for t in texts)/len(texts)
                    future_targets.append((min(1.,float(np.log1p(gaps[0])/np.log1p(8.))),
                        min(1.,head_rate/16.),float(max(gaps)>1.),chord,slide,hold,touch,
                        float(future_index>0 and structure_signature(str(points[future_index].note_text))!=
                             structure_signature(str(points[future_index-1].note_text)))))
                else:future_targets.append((0.,)*8)
            for event_index,point in enumerate(points):
                if float(seconds_at_beat(point.beat,song.tempo,song.first)) >= float(audio_end_sec):
                    previous=Fraction(point.beat)
                    continue
                candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,
                    audio_end_sec=float(audio_end_sec),candidate_count=self.candidate_count)
                self.candidate_total += 1
                if not candidates:
                    previous=Fraction(point.beat)
                    continue
                target_index=candidate_index(point.beat,list(candidates))
                if target_index < 0:
                    self.candidate_misses += 1
                    if len(self.candidate_miss_examples) < 8:
                        self.candidate_miss_examples.append({'song_index':song_index,
                            'event_index':event_index,'target':str(point.beat),'previous':str(previous)})
                    previous=Fraction(point.beat)
                    continue
                history=[]; history_features=[]
                start=max(0,event_index-self.history_events)
                prior_beat=points[start-1].beat if start else Fraction(0)
                prior_second=float(seconds_at_beat(prior_beat,song.tempo,song.first))
                for prior in points[start:event_index]:
                    atoms=_NoteGrammar._split_slashes(str(prior.note_text))
                    ids=tuple(self.atom_to_id.get(atom,0) for atom in atoms[:self.action_slots])
                    history.append(ids+(0,)*(self.action_slots-len(ids)))
                    current_second=float(seconds_at_beat(prior.beat,song.tempo,song.first))
                    text=str(prior.note_text)
                    history_features.append((min(1.,max(0.,float(prior.beat-prior_beat)/4.)),
                        min(1.,max(0.,(current_second-prior_second)/2.)),
                        min(1.,len(atoms)/4.),float('/' in text),float('h' in text),
                        float(any(char in text for char in '-<>^pqszvwV')),
                        float(any(char in text for char in 'ABCDE')),float(any(char in text for char in 'bx'))))
                    prior_beat=prior.beat; prior_second=current_second
                action,mask,_=prefix_action_state(state,points,event_index)
                active=action[~mask]
                action_state=active.mean(0).tolist() if active.numel() else [0.]*50
                self.samples.append(ComposerSample(song_index,event_index,previous,point.beat,
                    str(point.note_text),tuple(candidates),tuple(history),tuple(history_features),tuple(action_state),
                    tuple(condition.to_vector().tolist()),future_targets[event_index],
                    float(event_index==len(points)-1)))
                previous=Fraction(point.beat)
            if len(self.samples)>song_sample_start:
                last_index=len(self.samples)-1
                self.samples[last_index]=replace(self.samples[last_index],eos_target=1.)
            self.songs.append(record)
        self._feature_cache=OrderedDict()
        manifest_path=Path(manifest)
        manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
        self.fingerprint=json.dumps({'manifest':str(manifest_path.resolve()),'manifest_sha256':manifest_hash,'candidate_count':self.candidate_count,
            'history_events':self.history_events,'action_slots':self.action_slots,'samples':len(self.samples),
            'candidate_total':self.candidate_total,'candidate_misses':self.candidate_misses},sort_keys=True)
        if self.candidate_misses:
            examples='; '.join(f"{item['target']} after {item['previous']}" for item in self.candidate_miss_examples)
            raise ValueError(
                f'Composer candidate coverage failed for {self.candidate_misses}/{self.candidate_total} '
                f'authored events; increase candidate_count or extend the canonical grid. Examples: {examples}'
            )

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __len__(self):return len(self.samples)

    def _song_features(self,song_index):
        if song_index not in self._feature_cache:
            song=self._source_song(song_index)
            values,times,_=detail_queries(song,8192)
            arrays=(values.detach().cpu().numpy().astype(np.float32,copy=False),np.asarray(times,dtype=np.float64))
            self._feature_cache[song_index]=arrays
            self._feature_cache.move_to_end(song_index)
            while sum(int(value[0].nbytes+value[1].nbytes) for value in self._feature_cache.values())>self._feature_cache_budget_bytes:
                self._feature_cache.popitem(last=False)
        else:
            self._feature_cache.move_to_end(song_index)
        return self._feature_cache[song_index]

    def _source_song(self,index):
        song=self.base.songs[index][0]
        audio=getattr(song,'audio',None)
        if hasattr(audio,'store') and hasattr(audio,'path'):
            return replace(song,audio=audio.store.get(audio.path))
        return song

    def __getitem__(self,index):
        sample=self.samples[index];song=self.base.songs[sample.song_index][0]
        values,times=self._song_features(sample.song_index)
        candidates=list(sample.candidates)
        candidate_mask=np.zeros(self.candidate_count,dtype=bool)
        if len(candidates)<self.candidate_count:
            if not candidates:raise ValueError('Composer sample has no candidate beats')
            candidate_mask[len(candidates):]=True
            candidates.extend([candidates[-1]]*(self.candidate_count-len(candidates)))
        else:candidates=candidates[:self.candidate_count]
        feature=candidate_features(candidates,sample.previous,song.tempo,song.first)
        seconds=np.asarray([float(seconds_at_beat(beat,song.tempo,song.first)) for beat in candidates])
        audio=np.column_stack([np.interp(seconds,times,values[:,column],left=0.,right=0.)
                               for column in range(values.shape[1])]).astype(np.float32)
        features=np.concatenate((audio,feature),axis=1)
        target=candidate_index(sample.target,list(candidates))
        if target<0:raise ValueError('Composer candidate generator missed an authored target')
        action=np.zeros(self.action_slots,dtype=np.int64)
        for slot,atom in enumerate(_NoteGrammar._split_slashes(str(sample.note_text))[:self.action_slots]):
            action[slot]=self.atom_to_id.get(atom,0)
        return {'candidate_features':torch.from_numpy(features),
                'candidate_target':torch.tensor(target,dtype=torch.long),
                'candidate_mask':torch.from_numpy(candidate_mask),
                'action_target':torch.from_numpy(action),
                'history':torch.tensor(sample.history,dtype=torch.long).reshape(-1,self.action_slots),
                'history_features':torch.tensor(sample.history_features,dtype=torch.float32).reshape(-1,8),
                'action_state':torch.tensor(sample.action_state,dtype=torch.float32),
                'condition':torch.tensor(sample.condition,dtype=torch.float32),
                'future_target':torch.tensor(sample.future_target,dtype=torch.float32),
                'eos_target':torch.tensor(sample.eos_target,dtype=torch.float32),
                'song_index':torch.tensor(sample.song_index),
                'event_index':torch.tensor(sample.event_index)}


def collate_composer(items):
    from torch.nn.utils.rnn import pad_sequence
    result={}
    result['candidate_features']=torch.stack([item['candidate_features'] for item in items])
    result['candidate_target']=torch.stack([item['candidate_target'] for item in items])
    result['action_target']=torch.stack([item['action_target'] for item in items])
    result['action_state']=torch.stack([item['action_state'] for item in items])
    result['future_target']=torch.stack([item['future_target'] for item in items])
    result['condition']=torch.stack([item['condition'] for item in items])
    result['history']=pad_sequence([item['history'] for item in items],batch_first=True,padding_value=0)
    result['history_features']=pad_sequence([item['history_features'] for item in items],batch_first=True,padding_value=0.)
    result['history_padding_mask']=torch.arange(result['history'].size(1))[None]>=torch.tensor(
        [len(item['history']) for item in items])[:,None]
    result['candidate_padding_mask']=torch.stack([item['candidate_mask'] for item in items])
    result['song_index']=torch.stack([item['song_index'] for item in items])
    result['event_index']=torch.stack([item['event_index'] for item in items])
    result['eos_target']=torch.stack([item['eos_target'] for item in items])
    return result
