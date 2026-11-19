"""Audio-predicted local writing tendencies shared by training and generation."""
from bisect import bisect_left

import numpy as np
import torch
from torch import nn
from . import motion_writing
from .defaults import AUDIO_WRITING_POLICIES, BOUNDARY_WRITING, GRID_WRITING_POLICIES, MOTION_WRITING_POLICIES
from . import phrase_decisions

from .ordered_configuration import (
    BEAT_INTERVALS, FEATURES, KINDS, _event_atoms,
)

POLICY = 'audio_writing_v2'
SOURCE_RANGES = ((-4, 0), (0, 2), (2, 4), (4, 8))
BASE_FEATURES = tuple(name for name in FEATURES if not name.startswith('position_'))
FIELDS = (*BASE_FEATURES, *(f'direction_{i}' for i in range(8)), 'chord')
WIDTH = len(FIELDS)
MOTION_POLICY = 'audio_writing_v3'
POLICIES = AUDIO_WRITING_POLICIES
RELEASE_FIELDS = ('release_to_rim','release_to_slide','release_to_touch')
MOTION_FIELDS = (*FIELDS, 'no_new_head', *motion_writing.ORDER_FIELDS, *RELEASE_FIELDS)
TIME_COLUMNS = tuple(MOTION_FIELDS.index(name) for name in
                     ('no_new_head', 'rest', 'hold_occupied', 'slide_occupied', 'touch_hold_occupied',
                      'hold_release', 'slide_release', 'touch_hold_release'))
HORIZONS = len(BEAT_INTERVALS)
_BASE_COLUMNS = tuple(FEATURES.index(name) for name in BASE_FEATURES)
_KIND_END = 1 + len(KINDS)
_STEP_END = _KIND_END + 5
GROUPS = ((0, 1), (1, _KIND_END), (_KIND_END, _STEP_END),
          (_STEP_END, len(BASE_FEATURES)), (len(BASE_FEATURES), WIDTH - 1), (WIDTH - 1, WIDTH))
# Native profile uses 256 semantic values and 12 onset bins.
INPUT_WIDTH = (256 + 12 + 12 + 1) * len(SOURCE_RANGES) + 1


def audio_rows(profile, clock):
    """One compact row per integer beat, computed from the existing audio cache."""
    semantic = np.pad(profile.semantic, ((0, 0), (0, max(0, 256 - profile.semantic.shape[1]))))
    block = np.concatenate((semantic, profile.onset, np.log1p(profile.onset_raw)), axis=1)
    if block.shape[1] != 280:
        raise ValueError('Local writing requires the 256-dimensional beat music profile')
    count = len(block)
    sums = np.concatenate((np.zeros((1, block.shape[1])), block.cumsum(0, dtype=np.float64)))
    anchors = np.arange(count + 1)
    parts = []
    for left, right in SOURCE_RANGES:
        a = np.clip(anchors + left, 0, count)
        b = np.clip(anchors + right, 0, count)
        parts.append((sums[b] - sums[a]) / np.maximum(1, b - a)[:, None])
        parts.append(((b - a) / (right - left))[:, None])
    seconds = clock.at_floats(anchors.astype(np.float64))
    beat_seconds = clock.at_floats(anchors.astype(np.float64) + 1) - seconds
    parts.append(np.log(np.maximum(beat_seconds, 1e-6))[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


def inputs(rows, song, anchors):
    anchors = np.asarray(anchors, dtype=np.int64)
    indices = np.clip(anchors, 0, len(rows) - 1)
    values = rows[indices].copy()
    values[(anchors < 0) | (anchors >= len(rows))] = 0
    memory = np.asarray(song.summary['beats'], dtype=np.float64)
    right = np.searchsorted(memory, anchors).clip(0, len(memory) - 1)
    left = np.maximum(0, right - 1)
    mix = np.clip((anchors - memory[left]) / np.maximum(memory[right] - memory[left], 1e-6), 0, 1)
    return dict(writing_audio=torch.from_numpy(values), writing_left=torch.from_numpy(left),
                writing_right=torch.from_numpy(right), writing_mix=torch.as_tensor(mix, dtype=torch.float32))


def _motion_rows(configuration, song):
    cached = getattr(configuration, '_writing_motion', None)
    if cached is not None:
        return cached
    movement = np.zeros((len(song.points), 9), dtype=np.float32)
    available = np.zeros(len(song.points), dtype=np.bool_)
    previous = ()
    from .generate import _atom_heads
    for index, point in enumerate(song.points):
        _, _, buttons = _event_atoms(point.note_text)
        movement[index, -1] = len(set(_atom_heads(point.note_text))) > 1
        if len(previous) == len(buttons) == 1 and index and configuration.event_weights[index - 1, 1] > 0:
            movement[index, (buttons[0] - previous[0]) % 8] = 1
            available[index] = True
        previous = buttons
    configuration._writing_motion = movement, available
    return movement, available


def add_targets(item, configuration, song, policy=POLICY, *, dataset=None, offset=.5):
    """Future notes are read here only and never enter planner inputs."""
    if policy in MOTION_WRITING_POLICIES:
        return _add_motion_targets(item, configuration, song, policy, dataset, offset)
    mass = float(item['token_weights'].sum() * item['sample_weight'])
    anchors = item['anchors']
    targets = np.zeros((len(anchors), HORIZONS, WIDTH), dtype=np.float32)
    weights = np.zeros_like(targets)
    supervised = {int(i) for i, weight in zip(item['tile_ids'], item['token_weights']) if weight > 0}
    movement, available = _motion_rows(configuration, song)
    cached = getattr(configuration, '_writing_targets', None)
    if cached is None:
        cached = configuration._writing_targets = {}
    for index in supervised:
        beat = anchors[index]
        if beat not in cached:
            first = bisect_left(configuration.beats, beat)
            base, trust = configuration.target_at(first, beat)
            values = np.zeros((HORIZONS, WIDTH), dtype=np.float32)
            valid = np.zeros_like(values)
            values[:, :len(BASE_FEATURES)] = base[2:][:, _BASE_COLUMNS]
            valid[:, :len(BASE_FEATURES)] = trust[2:][:, _BASE_COLUMNS]
            for row, (left, right) in enumerate(BEAT_INTERVALS):
                a = bisect_left(configuration.beats, beat + left)
                b = bisect_left(configuration.beats, beat + right)
                quality = configuration.quality.covered(*song.clock.at_floats(np.array([beat + left, beat + right])))
                if b > a and quality > 0:
                    use = available[a:b]
                    if use.any():
                        values[row, len(BASE_FEATURES):-1] = movement[a:b][use, :-1].mean(0)
                        valid[row, len(BASE_FEATURES):-1] = quality
                    values[row, -1] = movement[a:b, -1].mean()
                    valid[row, -1] = quality
            for left, right in GROUPS:
                valid[:, left:right] /= (right - left) * len(GROUPS)
            cached[beat] = values, valid
        targets[index], weights[index] = cached[beat]
    # Preserve quality attenuation and exact accumulation across unequal batches.
    weights *= mass / max(1, len(supervised) * HORIZONS)
    item['writing_targets'] = torch.from_numpy(targets)
    item['writing_weights'] = torch.from_numpy(weights)
    item['curve_teacher_allowed'] = False


def _motion_intervals(configuration, song, intervals):
    """Calculate only new intervals. Adjacent queries reuse the same two-beat span."""
    from .ordered_configuration import TARGET_DIM
    c = configuration
    if not hasattr(c, '_motion_intervals'):
        c._motion_intervals = {}
        c._ordered_motion = motion_writing.sums(song.points, c.event_weights[:,1] > 0)
        directions, known = _motion_rows(c,song)
        events = np.column_stack((c.events[:,_BASE_COLUMNS[1:_STEP_END]], directions, known))
        c._motion_totals = np.vstack((np.zeros(events.shape[1],dtype=np.float32), events.cumsum(0,dtype=np.float32)))
        c._motion_beats = np.asarray(c.beats,dtype=np.float64)
        releases = np.zeros((len(c.beats),4),dtype=np.float32)
        if len(c.sustains) and len(c.beats):
            indices = np.searchsorted(c.seconds,c.sustains[:,2])
            found = indices < len(c.beats)
            indices, sources = indices[found], c.sustains[found,0]
            # The held action, its release and the following head must all
            # belong to trusted material. Several tails can lead to one head.
            if len(indices):
                cover = c.quality.coverage_pairs(sources,np.nextafter(c.seconds[indices],np.inf))
                trusted = np.unique(indices[cover.min(1)>0])
                kinds = c.events[trusted,1:1+len(KINDS)]
                releases[trusted,:3] = np.column_stack((kinds[:,0] | kinds[:,1],kinds[:,2],kinds[:,3] | kinds[:,4]))
                releases[trusted,3] = 1
        c._release_totals = np.vstack((np.zeros(4,dtype=np.float32),releases.cumsum(0)))
    missing = sorted(set(intervals) - c._motion_intervals.keys())
    if not missing:
        return c._motion_intervals
    spans = np.asarray(missing)
    ends = spans[:,0] + spans[:,1]
    starts_s = song.clock.at_floats(spans[:,0].astype(np.float64))
    stops_s = song.clock.at_floats(ends.astype(np.float64))
    coverage = c.quality.coverage_pairs(starts_s,stops_s)
    first, last = (np.searchsorted(c._motion_beats,t) for t in (spans[:,0],ends))
    totals = c._motion_totals[last]-c._motion_totals[first]
    sizes = last-first
    values = np.zeros((len(missing),len(MOTION_FIELDS)),dtype=np.float32)
    weight = np.zeros_like(values)
    values[:,WIDTH] = sizes == 0
    weight[:,WIDTH] = coverage[:,0]
    values[:,1:_STEP_END] = totals[:,:_STEP_END-1]/np.maximum(1,sizes[:,None])
    weight[:,1:_STEP_END] = (coverage[:,1]*(sizes>0))[:,None]
    if len(c.event_weights):
        weight[(first>0) & (c.event_weights[np.maximum(0,first-1),1]<=0), _KIND_END:_STEP_END] = 0
    values[:,len(BASE_FEATURES):WIDTH-1] = totals[:,-10:-2]/np.maximum(1,totals[:,-1:])
    weight[:,len(BASE_FEATURES):WIDTH-1] = (coverage[:,1]*(totals[:,-1]>0))[:,None]
    values[:,WIDTH-1] = totals[:,-2]/np.maximum(1,sizes)
    weight[:,WIDTH-1] = coverage[:,1]*(sizes>0)
    mid=(first+last+1)//2
    counts, observed=(np.column_stack((x[mid]-x[first],x[last]-x[mid])) for x in c._ordered_motion)
    values[:,WIDTH+1:-len(RELEASE_FIELDS)] = counts/np.maximum(1,observed)
    weight[:,WIDTH+1:-len(RELEASE_FIELDS)] = (observed>0)*coverage[:,1:2]
    releases=c._release_totals[last]-c._release_totals[first]
    values[:,-3:]=releases[:,:3]/np.maximum(1,releases[:,3:])
    weight[:,-3:]=(releases[:,3:]>0)*coverage.min(1)[:,None]*(.05/3)
    for i in np.flatnonzero(coverage.min(1)>0):
        v=np.zeros(TARGET_DIM,dtype=np.float32)
        w=np.full(TARGET_DIM,coverage[i].min(),dtype=np.float32)
        v[0] = sizes[i] == 0
        c.sustain_features(v,w,starts_s[i],stops_s[i])
        values[i,0],weight[i,0] = v[0],w[0]
        cols=_BASE_COLUMNS[_STEP_END:]
        values[i,_STEP_END:len(BASE_FEATURES)] = v[list(cols)]
        weight[i,_STEP_END:len(BASE_FEATURES)] = w[list(cols)]
    for left,right in GROUPS:
        weight[:,left:right] *= .55/((right-left)*len(GROUPS))
    weight[:,WIDTH] *= .15
    offset=WIDTH+1
    for _ in range(2):
        for size in map(len,(motion_writing.CONTACT,motion_writing.MOVEMENT,motion_writing.PATHS)):
            weight[:,offset:offset+size] *= .25/(6*size)
            offset += size
    c._motion_intervals.update(zip(missing,zip(values,weight)))
    return c._motion_intervals


def _add_motion_targets(item, configuration, song, policy=MOTION_POLICY, dataset=None, offset=.5):
    anchors=item['anchors']
    shape=(len(anchors),HORIZONS,len(MOTION_FIELDS))
    values, weights=np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32)
    tiles=np.unique(np.asarray(item['tile_ids'])[item['token_weights'].numpy()>0])
    spans=[(anchors[i]+left,right-left) for i in tiles for left,right in BEAT_INTERVALS]
    if spans:
        cache=_motion_intervals(configuration,song,spans)
        v,w=zip(*(cache[span] for span in spans))
        values[tiles]=np.reshape(v,(-1,HORIZONS,shape[-1]))
        weights[tiles]=np.reshape(w,(-1,HORIZONS,shape[-1]))
    if policy in GRID_WRITING_POLICIES:
        values,weights=phrase_decisions.grid_targets(item,configuration,song,dataset,values,weights,tiles,spans,offset)
    elif spans:
        mass=float(item['token_weights'].sum()*item['sample_weight'])
        weights*=mass/(len(tiles)*HORIZONS)
    item['writing_targets']=torch.from_numpy(values)
    item['writing_weights']=torch.from_numpy(weights)


def collate(items, out):
    if 'writing_audio' not in items[0]:
        return
    extras=[(index,item['writing_extra']) for index,item in enumerate(items) if 'writing_extra' in item]
    blocks=[*enumerate(items),*extras]
    for name in ('writing_audio', 'writing_left', 'writing_right', 'writing_mix'):
        out[name] = torch.cat([part[name] for _,part in blocks])
    out['writing_owner'] = torch.cat([torch.full((len(part['writing_audio']),), index, dtype=torch.long)
                                     for index,part in blocks])
    if 'writing_targets' in items[0]:
        for source,target in (('writing_targets','future_configuration_targets'),('writing_weights','future_configuration_weights')):
            out[target]=torch.cat([it[source][:len(it['anchors'])] for it in items]+
                                  [items[i][source][len(items[i]['anchors']):] for i,_ in extras])
    phrase_decisions.collate(items,out)


class LocalWriting(nn.Module):
    def __init__(self, dimension, policy=POLICY):
        super().__init__()
        width = min(128, dimension)
        self.audio = nn.Sequential(nn.LayerNorm(INPUT_WIDTH), nn.Linear(INPUT_WIDTH, width), nn.SiLU())
        self.song = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width, bias=False))
        self.level = nn.Linear(1, width, bias=False)
        self.fields = MOTION_FIELDS if policy in MOTION_WRITING_POLICIES else FIELDS
        if policy in GRID_WRITING_POLICIES:
            self.fields = (*self.fields, *phrase_decisions.GRID_FIELDS)
        if policy == BOUNDARY_WRITING:
            self.successor = nn.Linear(dimension, len(phrase_decisions.DECISION_FIELDS))
            self.execute_successor = nn.Linear(len(phrase_decisions.DECISION_FIELDS),dimension,bias=False)
            nn.init.zeros_(self.execute_successor.weight)
        self.predictor = nn.Sequential(nn.SiLU(), nn.Linear(width, HORIZONS * len(self.fields)))
        self.projection = nn.Linear(HORIZONS * len(self.fields), dimension, bias=False)
        nn.init.zeros_(self.projection.weight)
        if policy in MOTION_WRITING_POLICIES:
            self.timing = nn.Linear(len(TIME_COLUMNS)*HORIZONS, dimension, bias=False)
            nn.init.zeros_(self.timing.weight)

    def plan(self, memory, batch):
        owner = batch['writing_owner']
        left = memory[owner, batch['writing_left']]
        right = memory[owner, batch['writing_right']]
        mix = batch['writing_mix'][:, None]
        song = left * (1 - mix) + right * mix
        level = batch['condition'][owner, :1].float()
        hidden = self.audio(batch['writing_audio'].float()) + self.song(song) + self.level(level)
        logits = self.predictor(hidden).reshape(-1, HORIZONS, len(self.fields))
        # The same predicted distribution enters both training and generation.
        return dict(writing_logits=logits, **self.project(logits.float().sigmoid()))

    def project(self, probabilities):
        context = self.projection(probabilities.flatten(1))
        if hasattr(self, 'timing'):
            timing = self.timing(probabilities[:, :, TIME_COLUMNS].flatten(1))
            context = torch.stack((context,timing),dim=1)
        return dict(writing_context=context)

    def read(self, context, batch, start, value, role_mask, *, time_mask=None, indices=None):
        if indices is None:
            indices = torch.as_tensor([row[start:] for row in batch['tile_ids_cpu']],
                                      dtype=torch.long, device=context.device)
        tiles = indices.clamp_min(0)
        if context.ndim==2:
            return context[tiles] * (role_mask[value] & indices.ge(0))[..., None]
        # The beat-sized table is packed once, including in incremental decoding.
        time=time_mask[value]
        return context.flatten(0,1)[tiles*2+time.long()] * ((role_mask[value] | time) & indices.ge(0))[..., None]


def migrate_state(model, state):
    """Retain the same feature inside each horizon when widening the writing head."""
    template = model.state_dict()
    prefixes = ('future_configuration_head.', 'local_writing.')
    result = {k:v for k,v in state.items() if not k.startswith(prefixes)}
    for name, target in template.items():
        if not name.startswith('local_writing.') or name not in state:
            continue
        old = state[name]
        if old.shape == target.shape:
            result[name] = old
        elif name in ('local_writing.predictor.1.weight','local_writing.predictor.1.bias','local_writing.projection.weight'):
            axis = 1 if name.endswith('projection.weight') else 0
            new = target.clone()
            dest = new.movedim(axis, 0).unflatten(0, (HORIZONS, -1))
            src = old.movedim(axis, 0).unflatten(0, (HORIZONS, -1))
            width = min(dest.size(1), src.size(1))
            dest[:, :width].copy_(src[:, :width])
            result[name] = new
        else:
            raise ValueError(f'Incompatible local writing parameter {name}: {old.shape} -> {target.shape}')
    return result, {k for k in template if k.startswith(prefixes) and k not in result}
