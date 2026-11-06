"""Beat-owned ordered drafts shared by planning, execution and real references.

The causal predictor never reads target tensors. Targets enter only
``draft_mode_nll``; every supervised NOTE, including dense middle events,
is read at its actual beat rather than reduced to a phrase histogram.
"""
from __future__ import annotations

import math
from fractions import Fraction

import torch
from torch import nn
from torch.nn import functional as F

from .events import event_metadata, head_pad_indices_for_policy

DRAFT_WIDTH = 96
DRAFT_ANCHORS = 16
DRAFT_FEATURES = 59


def _catalog_features(codec):
    """Compile exact parser-backed metadata once, outside the training loop."""
    from maichart.notes import parse_timing_point_notes
    from maichart.timing import RawTimingPoint
    features = torch.zeros(codec.num_atoms, DRAFT_FEATURES)
    for index, atom in enumerate(codec.atom_catalog, 1):
        meta = event_metadata(atom)
        features[index, :5] = torch.tensor([float(x > 0) for x in meta.type_counts])
        for pad in head_pad_indices_for_policy(meta, 'separate_pads_v2'):
            features[index, 5+pad] = 1.
        parsed = parse_timing_point_notes(RawTimingPoint(index=0, raw=atom,
            note_text=atom, beat=Fraction(0), tick=0, time_sec=0., division=4, bpm=120.))
        for note in parsed.notes:
            terminal = note.end_position
            if note.slide_segments:
                terminal = note.slide_segments[-1].get('end_position', terminal)
            if note.note_type == 'hold':
                terminal = note.position
            if terminal is not None and str(terminal).isdigit():
                value = int(str(terminal)[-1])
                if 1 <= value <= 8:
                    features[index, 46+value-1] = 1.
            duration = note.duration
            if duration is not None:
                # Keep beat and second semantics separate; do not infer a
                # fictional acoustic release for an authored short hold.
                if duration.kind == 'grid_fraction' and duration.beats is not None:
                    features[index, 54] = max(features[index, 54], math.log1p(max(0., float(duration.beats))))
                elif duration.seconds is not None:
                    features[index, 55] = max(features[index, 55], math.log1p(max(0., float(duration.seconds))))
            features[index, 56] = max(features[index, 56], float(note.note_type in ('hold', 'touch_hold')))
            features[index, 57] = max(features[index, 57], float(note.note_type == 'slide'))
        features[index, 58] = math.log1p(len(meta.heads))
    maximum = max(1, max(codec.shape_sizes))
    offsets = torch.full((codec.num_shapes, maximum), -1, dtype=torch.long)
    atoms = [0]
    for shape, slots in enumerate(codec.shape_catalog[1:], 1):
        for slot, (_, head) in enumerate(slots):
            offsets[shape, slot] = len(atoms)
            atoms.extend(codec._atoms[str(head+1)+suffix] for suffix in codec.detail_catalog[shape][slot])
    return features, offsets, torch.tensor(atoms, dtype=torch.long)


def _event_features(atoms, table):
    """Reduce an unordered event set while retaining every physical sensor."""
    active = atoms.gt(0)
    values = table[atoms.clamp_min(0)] * active[..., None]
    binary = values[..., :54].amax(-2)
    duration = values[..., 54:58].amax(-2)
    heads = binary[..., 5:46].sum(-1).log1p()[..., None]
    return torch.cat((binary, duration, heads), -1)


def _query_layout(batch):
    if 'draft_query_rows' in batch:
        return batch['draft_query_rows'], batch['draft_query_positions'], batch['draft_query_inverse']
    # Eager fixtures and callers without collation remain correct. Production
    # CPU collation supplies deduplicated rows to avoid this repeated work.
    shape = batch['phrase_start_beat'].shape
    rows = torch.arange(shape[0], device=batch['phrase_start_beat'].device)[:, None].expand(shape).flatten()
    positions = torch.arange(shape[1], device=rows.device)[None].expand(shape).flatten()
    return rows, positions, torch.arange(rows.numel(), device=rows.device).reshape(shape)


def _read_bank(bank, inverse, start, end, query_beats):
    """Gather two local micro anchors, never broadcast the full plan per step."""
    extra = query_beats.ndim-inverse.ndim
    if extra < 0:
        raise ValueError('Draft query must preserve the batch/time axes')
    ids = inverse.reshape(*inverse.shape, *([1]*extra)).expand(query_beats.shape)
    begin, stop = start[ids], end[ids]
    slots = bank.size(-3)*bank.size(-2)
    # Two ordered micro anchors per half beat; short final intervals retain
    # their actual beat coordinate and validity mask.
    position = ((query_beats-begin).float()*4.-.5).clamp(0, slots-1)
    left = position.floor().long();right = (left+1).clamp_max(slots-1)
    values = bank.flatten(-3,-2)  # [unique, mode, micro-time, feature]
    mode = torch.arange(values.size(1), device=values.device)
    low = values[ids[..., None],mode,left[..., None]]
    high = values[ids[..., None],mode,right[..., None]]
    mix = (position-left).float()[..., None, None]
    result = low+(high-low)*mix
    valid = (query_beats>=begin) & (query_beats<stop)
    return result*valid[..., None, None]


def read_draft(output, batch, query_beats, projection=None):
    """Read [B,T,(candidate),mode,width] at an actual absolute beat.

    ``projection`` is evaluated on the unique bank before candidate expansion;
    scalar/low-rank rate heads need not materialize width-96 candidate tensors.
    Beyond the planned phrase, return zero rather than extending its closing.
    """
    bank = output['_draft_plan_bank']
    if projection is not None and not isinstance(projection,nn.Linear):
        return projection(read_draft(output,batch,query_beats))
    if projection is not None:bank = projection(bank)
    inverse = output['_draft_query_inverse']
    extra = query_beats.ndim-inverse.ndim
    ids = inverse.reshape(*inverse.shape,*([1]*extra)).expand(query_beats.shape)
    start,end = output['_draft_query_start'][ids],output['_draft_query_end'][ids]
    result = _read_bank(bank,inverse,output['_draft_query_start'],output['_draft_query_end'],query_beats)
    relative = (query_beats-start).float()
    periods = relative.new_tensor([1/24,1/12,1/8,1/6,.25,.5,1.,2.])
    angle = 2*math.pi*relative[...,None]/periods
    clock = torch.cat((relative.remainder(.5)[...,None]*2,angle.sin(),angle.cos()),-1)
    fine = F.linear(clock,output['_draft_query_weight'])
    if projection is not None:fine = F.linear(fine,projection.weight,None)
    result = result+fine[...,None,:]
    return result*((query_beats>=start)&(query_beats<end))[...,None,None]


class OrderedPhraseDraft(nn.Module):
    """Small parallel plans, with a learned null/original reference choice."""
    def __init__(self, phrase_width, condition_dim, codec, *, width=96, modes=4, max_anchors=16):
        super().__init__()
        self.width,self.modes,self.max_anchors = int(width),int(modes),int(max_anchors)
        if width%4 or max_anchors<1:
            raise ValueError('Draft width must divide four heads and anchors must be positive')
        self.music = nn.Linear(64, width)
        self.controls = nn.Linear(condition_dim, width, bias=False)
        self.entry = nn.Linear(50, width, bias=False)
        self.clock = nn.Linear(18, width, bias=False)
        self.reference_geometry = nn.Linear(DRAFT_FEATURES, width, bias=False)
        self.reference_sequence = nn.Linear(25, width, bias=False)
        self.reference_music = nn.Linear(64, width, bias=False)
        self.reference_key = nn.Linear(width, width, bias=False)
        self.reference_query = nn.Linear(width, width, bias=False)
        self.null = nn.Parameter(torch.zeros(width))
        self.choices = nn.Parameter(torch.randn(modes, width)*.03)
        layer = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True)
        self.organize = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        # Encoder clones otherwise start with identical attention/FF weights.
        # Independent layers preserve the intended two-step organization.
        for block in self.organize.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:nn.init.zeros_(block.self_attn.in_proj_bias)
            for part in block.modules():
                if isinstance(part,nn.Linear):
                    nn.init.xavier_uniform_(part.weight)
                    if part.bias is not None:nn.init.zeros_(part.bias)
        self.mode = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, modes))
        self.query_clock = nn.Linear(17,width,bias=False)
        nn.init.zeros_(self.query_clock.weight)
        self.reconstruct = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,DRAFT_FEATURES))
        self.order = nn.Linear(width, 8)
        features, offsets, detail_atoms = _catalog_features(codec)
        self.register_buffer('atom_features', features, persistent=False)
        self.register_buffer('detail_offsets', offsets, persistent=False)
        self.register_buffer('detail_atoms', detail_atoms, persistent=False)
        rotation = torch.arange(DRAFT_FEATURES).expand(8,-1).clone()
        for shift in range(8):
            rotation[shift,5:13] = 5+(torch.arange(8)-shift)%8
            rotation[shift,46:54] = 46+(torch.arange(8)-shift)%8
            for group in range(4):
                rotation[shift,13+group*8:21+group*8] = 13+group*8+(torch.arange(8)-shift)%8
        self.register_buffer('reference_rotations', rotation, persistent=False)

    @staticmethod
    def _clock_cache(cache):
        from .stream_duration import prepare_interval_clock
        beats = cache['beats'].double()
        origin = beats.masked_fill(~cache['mask'],torch.inf).amin().clamp_max(0)
        clock_cache = {key:value for key,value in cache.items() if key!='_duration_clock'}
        clock_cache['seconds'] = beats-origin
        clock_cache['_duration_clock'] = prepare_interval_clock(clock_cache)
        return clock_cache,origin

    def forward(self, model, cache, inverse, batch, sequence_states, music_state=None):
        from .stream_duration import _read_intervals
        rows, positions, mapping = _query_layout(batch)
        start = batch['phrase_start_beat'][rows,positions]
        end = torch.minimum(batch['phrase_end_beat'][rows,positions],start+self.max_anchors*.5)
        count = rows.numel()
        runtime = batch.get('_draft_runtime_cache')
        runtime_scope = batch.get('_draft_runtime_key')
        cpu_keys = batch.get('draft_query_keys')
        # This state belongs to one forward and is shared with its selected
        # landing branch. Reuse the live graph, never a detached training plan
        # or a graph from an earlier optimizer update. A future landing may
        # have a different visible entry snapshot from a departure query;
        # CPU-packed entry content, not just phrase coordinates, guards reuse.
        forward_cache=None;forward_keys=None;missing=None
        entry_keys=batch.get('draft_query_entry_keys')
        if (music_state is not None and cpu_keys is not None and len(cpu_keys)==count
                and entry_keys is not None and len(entry_keys)==count):
            source=tuple(id(batch.get('phrase_sequence_'+name)) for name in ('atoms','mask','beats'))
            forward_keys=[(source,tuple(key),tuple(entry)) for key,entry in zip(cpu_keys,entry_keys)]
            forward_cache=music_state.setdefault('draft_forward_plans',{})
            missing=[index for index,key in enumerate(forward_keys) if key not in forward_cache]
            if not missing:
                records=[forward_cache[key] for key in forward_keys]
                return self._output(torch.stack([record[0] for record in records]),
                    torch.stack([record[1] for record in records]),torch.stack([record[2] for record in records]),
                    mapping,start,end,batch,cached=False)
            if len(missing)<count:
                original_start,original_end=start,end
                selected=torch.tensor(missing,device=rows.device,dtype=torch.long)
                rows,positions=rows[selected],positions[selected]
                start,end=start[selected],end[selected]
                count=len(missing)
        cache_keys = None
        if (not self.training and not torch.is_grad_enabled() and isinstance(runtime,dict)
                and runtime_scope is not None and cpu_keys is not None and len(cpu_keys)==count):
            cache_keys = [('ordered_draft',runtime_scope,tuple(key)) for key in cpu_keys]
            if all(key in runtime for key in cache_keys):
                records = [runtime[key] for key in cache_keys]
                return self._output(torch.stack([x[0] for x in records]),
                    torch.stack([x[1] for x in records]),torch.stack([x[2] for x in records]),
                    mapping,start,end,batch,cached=True)
        # Caller-owned lifetime is one forward/immutable inference encoding.
        if music_state is not None and 'draft_clock' in music_state:
            clock_cache,origin = music_state['draft_clock']
        else:
            clock_cache,origin = self._clock_cache(cache)
            if music_state is not None:music_state['draft_clock'] = (clock_cache,origin)
        offsets = (torch.arange(self.max_anchors*2,device=start.device,dtype=start.dtype)+.5)*.25
        beats = start[:,None]+offsets
        valid = beats<end[:,None]
        valid[:,0] |= end.gt(start)
        clipped = torch.minimum(beats,end[:,None])
        songs = inverse[rows]
        audio = _read_intervals(clock_cache,(clipped-origin)[:,None,:],songs,
            field='ordered_configuration_keys')[:,0]
        span = (end-start).clamp_min(1e-8)
        normalized = (beats-start[:,None])/span[:,None]
        angle = offsets[None,:,None]*offsets.new_tensor([.25,.5,1.,2.,3.,4.,6.,8.])[None,None,:].reciprocal()*2*math.pi
        clock = torch.cat((normalized[...,None],span.log1p()[:,None,None].expand(-1,len(offsets),1),
            angle.sin().expand(count,-1,-1),angle.cos().expand(count,-1,-1)),-1).float()
        entry = batch.get('draft_entry_state')
        entry = audio.new_zeros(count,50) if entry is None else entry[rows,positions].float()
        controls = batch['condition'].float()[rows]
        base = self.music(audio)+self.clock(clock)+self.controls(controls)[:,None]+self.entry(entry)[:,None]
        plans = base[:,None]+self.choices[None,:,None,:]
        reference_null = audio.new_ones(count,self.modes,len(offsets))
        if batch['phrase_sequence_kind'].size(0):
            indices = batch['phrase_sequence_inverse'][rows,positions]
            rs = batch['phrase_memory_start_beat'][rows,positions]
            re = batch['phrase_memory_end_beat'][rows,positions]
            has_notes = (batch['phrase_sequence_mask'].bool() & batch['phrase_sequence_kind'].eq(2)).any(-1)
            note_bank = torch.cat((has_notes.new_zeros(1),has_notes))
            visible = (batch['phrase_memory_mask'][rows,positions].bool() & indices.ge(0)
                & re.le(start[:,None]) & re.gt(rs) & note_bank[(indices+1).clamp_min(0)])
            idx = (indices+1).clamp_min(0)
            event_features = _event_features(batch['phrase_sequence_atoms'],self.atom_features)
            feat_bank = torch.cat((event_features.new_zeros(1,*event_features.shape[1:]),event_features))
            time_bank = torch.cat((batch['phrase_sequence_beats'].new_zeros(1,batch['phrase_sequence_beats'].size(1)),batch['phrase_sequence_beats']))
            note_mask = batch['phrase_sequence_mask'].bool() & batch['phrase_sequence_kind'].eq(2)
            mask_bank = torch.cat((note_mask.new_zeros(1,note_mask.size(1)),note_mask))
            event_pos = ((time_bank[idx]-rs[...,None])/(re-rs).clamp_min(1e-8)[...,None]).float()
            # Float64 clocks stay available for audio interpolation, while
            # learned geometry and its kernel use the model's FP32 contract.
            weights = -(event_pos[:,:,None,:]-normalized.float()[:,None,:,None]).square()*64.
            mask = mask_bank[idx].bool() & visible[...,None]
            weights = weights.masked_fill(~mask[:,:,None,:],-1e4).softmax(-1)*mask.any(-1)[:,:,None,None]
            geometry = weights@feat_bank[idx]
            # Only actual NOTE clocks and NOTE-to-NOTE geometry participate.
            # Inserting extra virtual WAIT tokens cannot change this path.
            ep = torch.arange(note_mask.size(1),device=note_mask.device)[None].expand_as(note_mask)
            last = ep.masked_fill(~note_mask,-1).cummax(-1).values
            previous = torch.cat((last.new_full((len(last),1),-1),last[:,:-1]),-1)
            old_heads = event_features[...,5:13].gather(1,previous.clamp_min(0)[...,None].expand(-1,-1,8))
            old_heads = old_heads*previous.ge(0)[...,None]
            motion = torch.stack([(old_heads.roll(s,-1)*event_features[...,5:13]).sum(-1) for s in range(8)],-1)
            motion = motion/motion.sum(-1,keepdim=True).clamp_min(1)
            motion_bank = torch.cat((motion.new_zeros(1,*motion.shape[1:]),motion))
            old_beats = batch['phrase_sequence_beats'].gather(1,previous.clamp_min(0))
            gaps = ((batch['phrase_sequence_beats']-old_beats).clamp_min(0)*previous.ge(0)).float().log1p()
            gap_bank = torch.cat((gaps.new_zeros(1,gaps.size(1)),gaps))
            rel_beat = (time_bank[idx]-rs[...,None]).float()
            periods = rel_beat.new_tensor([1/24,1/12,1/8,1/6,.25,.5,1.,2.])
            clocks = 2*math.pi*rel_beat[...,None]/periods
            ordered_features = torch.cat((clocks.sin(),clocks.cos(),gap_bank[idx][...,None],motion_bank[idx]),-1)
            ordered = self.reference_sequence(weights@ordered_features)
            rbeats = rs[...,None]+(re-rs).clamp_min(0)[...,None]*normalized[:,None]
            ref_music = _read_intervals(clock_cache,(rbeats-origin).reshape(count*indices.size(-1),1,-1),
                songs[:,None].expand_as(indices).reshape(-1),field='ordered_configuration_keys')[:,0]
            ref_music = ref_music.reshape(count,indices.size(-1),len(offsets),-1)
            difference = self.reference_music(audio[:,None]-ref_music)
            rotated = geometry[...,self.reference_rotations]  # [unique,ref,time,rotation,feature]
            references = self.reference_geometry(rotated)+ordered[:,:,:,None]+difference[:,:,:,None]
            references = references.permute(0,2,1,3,4).flatten(2,3)
            available = visible[:,:,None].expand(-1,-1,8).flatten(1)
            null = self.null.expand(count,len(offsets),1,self.width)
            references = torch.cat((null,references),2)
            scores = torch.einsum('umkd,ukrd->umkr',self.reference_query(plans),self.reference_key(references))/math.sqrt(self.width)
            allow = torch.cat((available.new_ones(count,1),available),-1)
            scores = scores.masked_fill(~allow[:,None,None,:],-torch.inf)
            reference_weights = scores.float().softmax(-1)
            plans = plans+torch.einsum('umkr,ukrd->umkd',reference_weights.to(references),references)
            reference_null = reference_weights[...,0]
        safe_valid = valid.clone();safe_valid[:,0] = True
        plans = self.organize(plans.reshape(count*self.modes,len(offsets),self.width),
            src_key_padding_mask=(~safe_valid)[:,None].expand(-1,self.modes,-1).reshape(count*self.modes,-1))
        plans = plans.reshape(count,self.modes,self.max_anchors,2,self.width)
        plans = plans*valid.reshape(count,1,self.max_anchors,2,1)
        organized = plans.mean(1).flatten(1,2)
        mode_logits = self.mode((organized*valid[...,None]).sum(1)/valid.sum(-1).clamp_min(1)[:,None])
        null_mean = reference_null.mean((-1,-2))
        if forward_cache is not None:
            for index,original in enumerate(missing):
                forward_cache[forward_keys[original]]=(plans[index],mode_logits[index],null_mean[index])
            if len(missing)<len(forward_keys):
                records=[forward_cache[key] for key in forward_keys]
                plans=torch.stack([record[0] for record in records])
                mode_logits=torch.stack([record[1] for record in records])
                null_mean=torch.stack([record[2] for record in records])
                start,end=original_start,original_end
        if cache_keys is not None:
            for index,key in enumerate(cache_keys):
                runtime[key] = (plans[index].detach().clone(),mode_logits[index].detach().clone(),null_mean[index].detach().clone())
            # Resource eviction changes work reuse, never musical eligibility.
            while len(runtime)>256:
                runtime.pop(next(iter(runtime)))
        return self._output(plans,mode_logits,null_mean,mapping,start,end,batch,cached=False)

    def _output(self,plans,mode_logits,null_mean,mapping,start,end,batch,*,cached):
        output = dict(_draft_plan_bank=plans,_draft_query_inverse=mapping,
            _draft_query_start=start,_draft_query_end=end,
            _draft_query_weight=self.query_clock.weight,
            draft_mode_logits=mode_logits[mapping],
            draft_reference_null=null_mean[mapping],draft_runtime_cache_hit=cached)
        output['draft_context'] = read_draft(output,batch,batch['step_beat'])
        return output

    def target_features(self, model, batch):
        shapes = batch['target_shape'].clamp_min(0)
        details = batch['target_details']
        slots = torch.arange(details.size(-1),device=details.device)
        safe_slots = slots.clamp_max(self.detail_offsets.size(-1)-1)
        offsets = self.detail_offsets[shapes[...,None],safe_slots]
        regular_ids = self.detail_atoms[(offsets.clamp_min(0)+details.clamp_min(0)).clamp_max(self.detail_atoms.numel()-1)]
        regular_ids = regular_ids*(offsets.ge(0)&details.ge(0)&slots.lt(self.detail_offsets.size(-1)))
        atoms = torch.where(shapes[...,None].eq(0),details.clamp_min(0),regular_ids)
        features = _event_features(atoms,self.atom_features)
        # All regular canonical features rotate together into authored space.
        phase_mask = batch['target_phase_mask'].float()
        rotated = features[...,self.reference_rotations]
        physical = (rotated*phase_mask[...,None]).sum(-2)/phase_mask.sum(-1).clamp_min(1)[...,None]
        return torch.where(shapes[...,None].gt(0),physical,features)


def draft_mode_nll(model, output, batch):
    """Mode-wise ordered geometry supervision at every actual target event.

    Time/REST are charged by the final arrival law, never by a second draft
    classifier. The per-event result is incorporated into the same phrase
    responsibility as execution instead of training an unused side head.
    """
    draft = model.phrase_context.draft
    beat = batch.get('target_beat',batch['step_beat'])
    # A NOTE may land in the next music phrase. Its geometry belongs to the
    # actual landing draft, not the departure draft's last/zero anchor.
    landing = output.get('draft_action_context')
    context = landing if landing is not None else read_draft(output,batch,beat)
    prediction = draft.reconstruct(context).float()
    target = draft.target_features(model,batch).detach()
    binary = F.binary_cross_entropy_with_logits(prediction[...,:54],target[...,None,:54].expand_as(prediction[...,:54]),reduction='none')
    # Give button, Touch, endpoint and semantic groups equal diagnostic units;
    # sparse Touch sensors do not dilute the button target or erase C.
    geometry = (binary[...,:5].mean(-1)+binary[...,5:13].mean(-1)
        +binary[...,13:46].mean(-1)+binary[...,46:54].mean(-1))*.25
    continuous = F.smooth_l1_loss(prediction[...,54:],target[...,None,54:].expand_as(prediction[...,54:]),reduction='none').mean(-1)
    # Ordered transition sees the true immediately preceding NOTE geometry.
    positions = torch.arange(batch['history_kind'].size(1),device=context.device)[None]
    previous = positions.expand_as(batch['history_kind']).masked_fill(batch['history_kind'].ne(2),-1).cummax(-1).values
    atoms = batch['history_atoms'].gather(1,previous.clamp_min(0)[...,None].expand_as(batch['history_atoms']))
    prior = _event_features(atoms,draft.atom_features)[...,5:13]*previous.ge(0)[...,None]
    rotations = torch.stack([prior.roll(shift,-1) for shift in range(8)],-2)
    relation = (rotations*target[...,None,5:13]).sum(-1)
    relation = relation/relation.sum(-1,keepdim=True).clamp_min(1)
    relation_loss = -(draft.order(context).float().log_softmax(-1)*relation[...,None,:]).sum(-1)
    note = batch['target_kind'].eq(2) & batch.get('configuration_mask',torch.ones_like(batch['target_kind'],dtype=torch.bool))
    if landing is not None and 'candidate_phrase_start_beat' in batch:
        start,end = batch['candidate_phrase_start_beat'],batch['candidate_phrase_end_beat']
    else:
        start = output['_draft_query_start'][output['_draft_query_inverse']]
        end = output['_draft_query_end'][output['_draft_query_inverse']]
    note = note & beat.ge(start) & beat.lt(end)
    return (geometry+continuous+.25*relation_loss)*note[...,None]
