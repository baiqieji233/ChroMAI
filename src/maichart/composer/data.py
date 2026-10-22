"""Event-level composer dataset built on the existing FormDataset/audio cache."""
from __future__ import annotations

from dataclasses import dataclass,replace
from fractions import Fraction
import json
import hashlib
from pathlib import Path
from collections import OrderedDict
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from maichart.form.data import FormDataset
from maichart.stream.context import prefix_action_state
from maichart.stream.audio_bounds import generation_bounds
from maichart.tempo_clock import TempoClock
from .candidates import (
    _ALL_CANONICAL_OFFSETS,
    _select_candidate_offsets,
    candidate_beats,
    candidate_features,
    candidate_index,
    candidate_spec,
    composer_audio_nodes,
    interp_candidate_audio,
)
from .events import structure_signature
from maichart.generation.event_cli import _NoteGrammar
from maichart.form.type_guidance import _counts


@dataclass(frozen=True, slots=True)
class ComposerSample:
    song_index: int
    event_index: int
    previous: Fraction | None
    target: Fraction
    note_text: str
    target_index: int
    valid_candidates: int
    eos_target: float


class ComposerDataset(Dataset):
    def __init__(self, manifest, *, cache_dir='cache/chart_composer_v1',
                 candidate_count=96, history_events=128, action_slots=8,
                 preparation_workers=1, audio_cache_gb=4., limit_charts=0,
                 quality_enabled=True, seed=0):
        started = time.monotonic()
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.codec_atoms=[]; self.atom_to_id={}; self.samples=[]; self.songs=[]
        self.candidate_count=int(candidate_count);self.history_events=int(history_events)
        if self.candidate_count < 16:
            raise ValueError('Composer candidate_count must be at least 16 for common onset and rest proposals')
        self.action_slots=int(action_slots);self.seed=int(seed);self.epoch=0
        # The exposed budget covers both stores rather than granting each a
        # full independent budget. Parsed labels/history are separate CPU data.
        self._feature_cache_budget_bytes=int(max(0.,float(audio_cache_gb))*(1024**3)*.75)
        self._feature_cache_hits=0;self._feature_cache_misses=0
        self.base=FormDataset(manifest,cache_dir=cache_dir,max_tokens=1_000_000,
            max_memory_tokens=max(256,history_events),max_region_seconds=999.,
            preparation_workers=preparation_workers,audio_cache_gb=max(0.,float(audio_cache_gb))*.25,
            limit_charts=limit_charts,quality_enabled=quality_enabled,
            preserve_excluded_context=True,quality_profile='musical_evidence_v1',
            rhythm_geometry_profile='none',audio_profile='rational_audio_queries_v1')
        records = list(self.base.songs)
        self._split_cache: dict[str, tuple[str, ...]] = {}
        self._text_meta_cache: dict[str, tuple] = {}
        atom_set: set[str] = set()
        for record in records:
            for point in record[1]:
                text = str(point.note_text)
                atoms = self._split_cache.get(text)
                if atoms is None:
                    atoms = tuple(_NoteGrammar._split_slashes(text))
                    self._split_cache[text] = atoms
                    atom_set.update(atoms)
        self.codec_atoms = sorted(atom_set)
        self.atom_to_id = {atom: index + 1 for index, atom in enumerate(self.codec_atoms)}
        self.candidate_total = 0
        self.candidate_misses = 0
        self.candidate_miss_examples = []
        self.action_grammar = _NoteGrammar.from_atoms(self.codec_atoms)
        self._song_tables: dict[int, dict] = {}
        self._feature_cache = OrderedDict()
        self._feature_cache_bytes = 0

        bounds_cache, bounds_path = self._load_bounds_cache()
        bounds_dirty = False
        total_charts = len(records)
        full_hi = len(_ALL_CANONICAL_OFFSETS)
        full_offsets, full_lookup, _ = _select_candidate_offsets(False, full_hi, self.candidate_count)
        max_off = _ALL_CANONICAL_OFFSETS[-1]
        for song_index, record in enumerate(records):
            song, points, level, weight, state, condition, excluded, annotation = record
            if not points or weight <= 0:
                continue
            song_sample_start = len(self.samples)
            audio_end_sec, updated = self._audio_end_sec(song_index, bounds_cache)
            bounds_dirty = bounds_dirty or updated
            clock = TempoClock(song.tempo, song.first)
            previous = None
            prev_sec = 0.0
            for event_index, point in enumerate(points):
                beat = point.beat if isinstance(point.beat, Fraction) else Fraction(point.beat)
                beat_sec = float(clock.at(beat))
                if beat_sec >= audio_end_sec:
                    previous = beat
                    prev_sec = beat_sec
                    continue
                if (
                    previous is not None
                    and previous >= 0
                    and (sec_plus_max := float(clock.at(previous + max_off))) < audio_end_sec
                    and sec_plus_max - prev_sec <= 16.0 - 1e-12
                ):
                    valid_hi, offsets, lookup = full_hi, full_offsets, full_lookup
                else:
                    _, _, valid_hi, offsets, lookup, _ = candidate_spec(
                        previous,
                        tempo=song.tempo,
                        first=song.first,
                        audio_end_sec=audio_end_sec,
                        candidate_count=self.candidate_count,
                        clock=clock,
                    )
                self.candidate_total += 1
                if not offsets:
                    previous = beat
                    prev_sec = beat_sec
                    continue
                offset_needed = beat if previous is None else beat - previous
                target_index = lookup.get(offset_needed, -1)
                if target_index < 0:
                    self.candidate_misses += 1
                    if len(self.candidate_miss_examples) < 8:
                        self.candidate_miss_examples.append({
                            'song_index': song_index,
                            'event_index': event_index,
                            'target': str(beat),
                            'previous': str(previous),
                        })
                    previous = beat
                    prev_sec = beat_sec
                    continue
                self.samples.append(ComposerSample(
                    song_index,
                    event_index,
                    previous,
                    beat,
                    str(point.note_text),
                    target_index,
                    valid_hi,
                    float(event_index == len(points) - 1),
                ))
                previous = beat
                prev_sec = beat_sec
            if len(self.samples) > song_sample_start:
                last_index = len(self.samples) - 1
                self.samples[last_index] = replace(self.samples[last_index], eos_target=1.)
            self.songs.append(record)
            if (song_index + 1) % 250 == 0:
                print('composer_prepare ' + json.dumps({
                    'stage': 'scanning',
                    'completed': song_index + 1,
                    'total': total_charts,
                    'samples': len(self.samples),
                    'elapsed_sec': round(time.monotonic() - started, 2),
                }), flush=True)
        if bounds_dirty and bounds_path is not None:
            self._save_bounds_cache(bounds_path, bounds_cache)
        manifest_path = Path(manifest)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
        self.fingerprint = json.dumps({
            'manifest': str(manifest_path.resolve()),
            'manifest_sha256': manifest_hash,
            'candidate_count': self.candidate_count,
            'history_events': self.history_events,
            'action_slots': self.action_slots,
            'samples': len(self.samples),
            'candidate_total': self.candidate_total,
            'candidate_misses': self.candidate_misses,
        }, sort_keys=True)
        if not self.samples:
            raise ValueError('Composer has no eligible events after candidate filtering; check manifest and candidate recall')
        print('composer_prepare ' + json.dumps({
            'stage': 'ready',
            'charts': len(self.songs),
            'samples': len(self.samples),
            'candidate_total': self.candidate_total,
            'candidate_misses': self.candidate_misses,
            'elapsed_sec': round(time.monotonic() - started, 2),
        }), flush=True)
        miss_rate = self.candidate_misses / max(1, self.candidate_total)
        if self.candidate_misses and miss_rate > 0.05:
            examples = '; '.join(f"{item['target']} after {item['previous']}" for item in self.candidate_miss_examples)
            raise ValueError(
                f'Composer candidate coverage failed for {self.candidate_misses}/{self.candidate_total} '
                f'authored events; increase candidate_count or extend the canonical grid. Examples: {examples}'
            )

    def _load_bounds_cache(self) -> tuple[dict[str, float], Path | None]:
        if not self.cache_dir:
            return {}, None
        path = Path(self.cache_dir) / 'composer_prepare' / 'audio_bounds_v1.json'
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    return {str(k): float(v) for k, v in data.items()}, path
            except Exception:
                pass
        return {}, path

    @staticmethod
    def _save_bounds_cache(path: Path, cache: dict[str, float]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp', encoding='utf-8', delete=False) as f:
                tmp = Path(f.name)
                json.dump(cache, f)
            tmp.replace(path)
        except Exception:
            pass

    def _audio_end_sec(self, song_index: int, bounds_cache: dict[str, float]) -> tuple[float, bool]:
        song = self.base.songs[song_index][0]
        audio = getattr(song, 'audio', None)
        if hasattr(audio, 'path') and hasattr(audio, 'store'):
            try:
                path = Path(audio.path).resolve()
                stat = path.stat()
                key = f'{path}:{stat.st_size}:{stat.st_mtime_ns}'
                if key in bounds_cache:
                    return bounds_cache[key], False
                if self.cache_dir:
                    from maichart.form.rational_audio import _source_digest
                    from maichart.form.waveform_cache import mapped_waveform
                    from maichart.stream.audio_bounds import exact_signal_end
                    digest = _source_digest(path, self.cache_dir)
                    signal, sr = mapped_waveform(path, digest, self.cache_dir)
                    duration = float(len(signal) / sr)
                    measured = exact_signal_end(signal, sr)
                    end_sec = float(measured if duration - measured >= 0.05 else duration)
                else:
                    end_sec = float(generation_bounds(self._source_song(song_index).audio, trim_exact_silence=True)['effective_end_sec'])
                bounds_cache[key] = end_sec
                return end_sec, True
            except Exception:
                pass
        return float(generation_bounds(self._source_song(song_index).audio, trim_exact_silence=True)['effective_end_sec']), False

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.samples)

    def _song_table(self, song_index: int) -> dict:
        table = self._song_tables.get(song_index)
        if table is not None:
            return table
        song, points, _level, _weight, _state, condition, _excluded, _annotation = self.base.songs[song_index]
        n = len(points)
        clock = TempoClock(song.tempo, song.first)
        point_seconds = [float(clock.at(p.beat)) for p in points]
        atom_ids = np.zeros((n, self.action_slots), dtype=np.int64)
        hist_feats = np.zeros((n, 8), dtype=np.float32)
        counts = [0] * n
        sigs = [None] * n
        chords = [0.0] * n
        holds = [0.0] * n
        slides = [0.0] * n
        touches = [0.0] * n
        prior_beat = Fraction(0)
        prior_second = float(clock.at(prior_beat))
        for i, p in enumerate(points):
            text = str(p.note_text)
            meta = self._text_meta_cache.get(text)
            if meta is None:
                atoms = self._split_cache.get(text) or tuple(_NoteGrammar._split_slashes(text))
                parsed = _counts(text)
                meta = (
                    atoms,
                    int(parsed.sum() if parsed is not None else 0.),
                    structure_signature(text),
                    float('/' in text),
                    float('h' in text),
                    float(any(c in text for c in '-<>^pqszvwV')),
                    float(any(c in text for c in 'ABCDE')),
                    float(any(c in text for c in 'bx')),
                )
                self._text_meta_cache[text] = meta
            atoms, count, sig, is_chord, is_hold, is_slide, is_touch, is_break_ex = meta
            for slot, atom in enumerate(atoms[:self.action_slots]):
                atom_ids[i, slot] = self.atom_to_id.get(atom, 0)
            cur_sec = point_seconds[i]
            hist_feats[i] = (
                min(1., max(0., float(p.beat - prior_beat) / 4.)),
                min(1., max(0., (cur_sec - prior_second) / 2.)),
                min(1., len(atoms) / 4.),
                is_chord,
                is_hold,
                is_slide,
                is_touch,
                is_break_ex,
            )
            prior_beat = p.beat
            prior_second = cur_sec
            counts[i] = count
            sigs[i] = sig
            chords[i] = is_chord
            holds[i] = is_hold
            slides[i] = is_slide
            touches[i] = is_touch

        future_targets = np.zeros((n, 8), dtype=np.float32)
        log1p_8 = float(np.log1p(8.0))
        for fi in range(n):
            end_w = min(n, fi + 4)
            w_len = end_w - fi
            if w_len > 0:
                before_sec = point_seconds[fi - 1] if fi else 0.0
                gaps = [max(0.0, point_seconds[j] - before_sec) for j in range(fi, end_w)]
                head_rate = float(sum(counts[fi:end_w]) / max(sum(gaps), 1e-3))
                future_targets[fi] = (
                    min(1., float(np.log1p(gaps[0]) / log1p_8)),
                    min(1., head_rate / 16.),
                    float(max(gaps) > 1.),
                    sum(chords[fi:end_w]) / w_len,
                    sum(slides[fi:end_w]) / w_len,
                    sum(holds[fi:end_w]) / w_len,
                    sum(touches[fi:end_w]) / w_len,
                    float(fi > 0 and sigs[fi] != sigs[fi - 1]),
                )
        action_states = torch.zeros((n, 50), dtype=torch.float32)
        action_ready = bytearray(n)
        if n > 0:
            action_ready[0] = 1
        table = {
            'clock': clock,
            'atom_ids': torch.from_numpy(atom_ids),
            'history_features': torch.from_numpy(hist_feats),
            'future_targets': torch.from_numpy(future_targets),
            'condition': condition.to_vector().detach().cpu().to(torch.float32),
            'action_states': action_states,
            'action_ready': action_ready,
        }
        self._song_tables[song_index] = table
        return table

    def _action_state(self, song_index: int, event_index: int, table: dict) -> torch.Tensor:
        if not table['action_ready'][event_index]:
            record = self.base.songs[song_index]
            action, mask, _ = prefix_action_state(record[4], record[1], event_index)
            active = action[~mask]
            if active.numel():
                table['action_states'][event_index] = active.mean(0)
            table['action_ready'][event_index] = 1
        return table['action_states'][event_index]

    def _song_feature_key(self, song_index: int):
        song = self.base.songs[song_index][0]
        audio = getattr(song, 'audio', None)
        audio_key = str(getattr(audio, 'path', None) or getattr(audio, 'source', None) or song_index)
        tempo_key = tuple((str(t.beat), float(t.bpm)) for t in song.tempo)
        meter_key = tuple(
            (str(m.beat), int(m.numerator), int(m.denominator), tuple(getattr(m, 'grouping', None) or ()))
            for m in (getattr(song, 'meters', None) or ())
        )
        regions = getattr(song, 'regions', None)
        spans = getattr(song, 'spans', None)
        extent_key = str(regions[-1].end if regions else (spans[-1].end if spans else 0))
        return (audio_key, float(song.first), tempo_key, meter_key, extent_key)

    def _song_features(self, song_index: int):
        key = self._song_feature_key(song_index)
        arrays = self._feature_cache.get(key)
        if arrays is None:
            self._feature_cache_misses+=1
            song = self._source_song(song_index, cache_in_base=False)
            arrays = composer_audio_nodes(song)
            item_bytes = int(arrays[0].nbytes + arrays[1].nbytes)
            if item_bytes>self._feature_cache_budget_bytes:
                return arrays
            self._feature_cache[key] = arrays
            self._feature_cache.move_to_end(key)
            self._feature_cache_bytes += item_bytes
            while self._feature_cache_bytes > self._feature_cache_budget_bytes and len(self._feature_cache) > 1:
                _, evicted = self._feature_cache.popitem(last=False)
                self._feature_cache_bytes -= int(evicted[0].nbytes + evicted[1].nbytes)
        else:
            self._feature_cache_hits+=1
            self._feature_cache.move_to_end(key)
        return arrays

    def cache_stats(self):
        return {'hits':self._feature_cache_hits,'misses':self._feature_cache_misses,
                'bytes':self._feature_cache_bytes,'budget_bytes':self._feature_cache_budget_bytes,
                'entries':len(self._feature_cache),'scope':'composer_audio_bank'}

    def _source_song(self, index, *, cache_in_base=True):
        song = self.base.songs[index][0]
        audio = getattr(song, 'audio', None)
        if hasattr(audio, 'store') and hasattr(audio, 'path'):
            if not cache_in_base:
                try:
                    from maichart.form.rational_audio import load_rational_audio
                    cache_path = audio.store.disk_cache_path(audio.path)
                    if cache_path is not None and cache_path.is_file():
                        return replace(song, audio=load_rational_audio(cache_path, source=audio.path, cache_dir=audio.store.cache_dir))
                except Exception:
                    pass
            return replace(song, audio=audio.store.get(audio.path))
        return song

    def __getitem__(self, index):
        sample = self.samples[index]
        song = self.base.songs[sample.song_index][0]
        table = self._song_table(sample.song_index)
        clock = table['clock']
        values, times = self._song_features(sample.song_index)
        offsets, _, valid_floats = _select_candidate_offsets(
            sample.previous is None, sample.valid_candidates, self.candidate_count
        )
        actual_len = len(offsets)
        if actual_len <= 0:
            raise ValueError('Composer sample has no candidate beats')
        candidate_mask = np.zeros(self.candidate_count, dtype=bool)
        base_float = 0.0 if sample.previous is None else float(sample.previous)
        if actual_len < self.candidate_count:
            candidate_mask[actual_len:] = True
            beat_floats = np.empty(self.candidate_count, dtype=np.float64)
            beat_floats[:actual_len] = base_float + valid_floats
            beat_floats[actual_len:] = beat_floats[actual_len - 1]
        else:
            beat_floats = base_float + valid_floats
        if len(clock.beats) == 1:
            seconds = clock.seconds[0] + beat_floats * (60.0 / clock.bpms[0])
            candidates = None
        else:
            base_frac = Fraction(0) if sample.previous is None else sample.previous
            candidates = [base_frac + off for off in offsets]
            if len(candidates) < self.candidate_count:
                candidates.extend([candidates[-1]] * (self.candidate_count - len(candidates)))
            seconds = np.asarray([clock.at(b) for b in candidates], dtype=np.float64)
        feature = candidate_features(
            candidates,
            sample.previous,
            song.tempo,
            song.first,
            clock=clock,
            candidate_seconds=seconds,
            beat_floats=beat_floats,
        )
        audio = interp_candidate_audio(values, times, seconds)
        features = np.concatenate((audio, feature), axis=1)
        target = sample.target_index
        if target < 0 or target >= self.candidate_count:
            raise ValueError('Composer candidate generator missed an authored target')
        ev = sample.event_index
        start = max(0, ev - self.history_events)
        return {
            'candidate_features': torch.from_numpy(features),
            'candidate_target': torch.tensor(target, dtype=torch.long),
            'candidate_mask': torch.from_numpy(candidate_mask),
            'action_target': table['atom_ids'][ev],
            'history': table['atom_ids'][start:ev],
            'history_features': table['history_features'][start:ev],
            'action_state': self._action_state(sample.song_index, ev, table),
            'condition': table['condition'],
            'future_target': table['future_targets'][ev],
            'eos_target': torch.tensor(sample.eos_target, dtype=torch.float32),
            'song_index': torch.tensor(sample.song_index),
            'event_index': torch.tensor(ev),
        }


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
