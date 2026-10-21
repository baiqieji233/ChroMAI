"""Event-level composer dataset built on the existing FormDataset/audio cache."""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.form.data import FormDataset
from maichart.form.rational_audio import planner_audio_queries
from maichart.form.curve import resample_load_curve
from maichart.event_timing import seconds_at_beat
from maichart.generation.conditions import GenerationCondition
from maichart.stream.context import prefix_action_state
from .candidates import candidate_beats, candidate_features, candidate_index
from .events import atom_catalog


@dataclass(frozen=True, slots=True)
class ComposerSample:
    song_index: int
    event_index: int
    previous: Fraction
    target: Fraction
    note_text: str
    candidates: tuple[Fraction, ...]
    history: tuple[int, ...]
    action_state: tuple[float, ...]
    condition: tuple[float, ...]


class ComposerDataset(Dataset):
    def __init__(self, manifest, *, cache_dir='cache/chart_composer_v1',
                 candidate_count=64, history_events=128, action_slots=8,
                 preparation_workers=1, audio_cache_gb=4., limit_charts=0,
                 quality_enabled=True, seed=0):
        self.codec_atoms=[]; self.atom_to_id={}; self.samples=[]; self.songs=[]
        self.candidate_count=int(candidate_count);self.history_events=int(history_events)
        self.action_slots=int(action_slots);self.seed=int(seed);self.epoch=0
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
        for song_index,record in enumerate(records):
            song,points,level,weight,state,condition,excluded,annotation=record
            if not points or weight<=0:continue
            values,times,_=planner_audio_queries(self._source_song(song_index))
            values=np.asarray(values,dtype=np.float32);times=np.asarray(times,dtype=np.float64)
            previous=Fraction(0)
            for event_index,point in enumerate(points):
                candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,
                    audio_end_sec=float(song.audio.duration_sec),candidate_count=self.candidate_count,
                    target=point.beat)
                history=[]
                for prior in points[max(0,event_index-self.history_events):event_index]:
                    atoms=str(prior.note_text).split('/')
                    history.append(self.atom_to_id.get(atoms[0],0))
                action,mask,_=prefix_action_state(state,points,event_index)
                active=action[~mask]
                action_state=active.mean(0).tolist() if active.numel() else [0.]*50
                self.samples.append(ComposerSample(song_index,event_index,previous,point.beat,
                    str(point.note_text),tuple(candidates),tuple(history),tuple(action_state),
                    tuple(condition.to_vector().tolist())))
                previous=Fraction(point.beat)
            self.songs.append(record)
        self._feature_cache={}
        self.fingerprint=json.dumps({'manifest':str(manifest),'candidate_count':self.candidate_count,
            'history_events':self.history_events,'action_slots':self.action_slots,'samples':len(self.samples)},sort_keys=True)

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __len__(self):return len(self.samples)

    def _song_features(self,song_index):
        if song_index not in self._feature_cache:
            song=self._source_song(song_index)
            values,times,_=planner_audio_queries(song)
            self._feature_cache[song_index]=(np.asarray(values,dtype=np.float32),np.asarray(times,dtype=np.float64))
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
        feature=candidate_features(sample.candidates,sample.previous,song.tempo,song.first)
        seconds=np.asarray([float(seconds_at_beat(beat,song.tempo,song.first)) for beat in sample.candidates])
        audio=np.column_stack([np.interp(seconds,times,values[:,column],left=0.,right=0.)
                               for column in range(values.shape[1])]).astype(np.float32)
        curve=np.asarray([resample_load_curve(song.load_curve_times_sec,song.load_curve,
            np.asarray([second-float(song.first)]))[0] for second in seconds],dtype=np.float32)
        features=np.concatenate((audio,feature,curve),axis=1)
        target=candidate_index(sample.target,list(sample.candidates))
        if target<0:raise ValueError('Composer candidate generator missed an authored target')
        action=np.zeros(self.action_slots,dtype=np.int64)
        for slot,atom in enumerate(str(sample.note_text).split('/')[:self.action_slots]):
            action[slot]=self.atom_to_id.get(atom,0)
        return {'candidate_features':torch.from_numpy(features),
                'candidate_target':torch.tensor(target,dtype=torch.long),
                'action_target':torch.from_numpy(action),
                'history':torch.tensor(sample.history,dtype=torch.long),
                'action_state':torch.tensor(sample.action_state,dtype=torch.float32),
                'condition':torch.tensor(sample.condition,dtype=torch.float32),
                'song_index':torch.tensor(sample.song_index),
                'event_index':torch.tensor(sample.event_index)}


def collate_composer(items):
    from torch.nn.utils.rnn import pad_sequence
    result={}
    result['candidate_features']=torch.stack([item['candidate_features'] for item in items])
    result['candidate_target']=torch.stack([item['candidate_target'] for item in items])
    result['action_target']=torch.stack([item['action_target'] for item in items])
    result['action_state']=torch.stack([item['action_state'] for item in items])
    result['condition']=torch.stack([item['condition'] for item in items])
    result['history']=pad_sequence([item['history'] for item in items],batch_first=True,padding_value=0)
    result['history_padding_mask']=torch.arange(result['history'].size(1))[None]>=torch.tensor(
        [len(item['history']) for item in items])[:,None]
    result['candidate_padding_mask']=torch.zeros(result['candidate_features'].shape[:2],dtype=torch.bool)
    result['song_index']=torch.stack([item['song_index'] for item in items])
    result['event_index']=torch.stack([item['event_index'] for item in items])
    return result
