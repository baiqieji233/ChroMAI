"""Audio-grounded Composer: ordered listening, hierarchical time, shared actions."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .model import ChartComposer
from .audio_lens import AudioLens
from .action_factors import CompositionalActions
from .planning import AudioPlanner,read_plan
from .stability import bounded_history_memory, history_attention_context
from .events import merge_separate_head_timing

GROUNDED_PROFILE='audio_grounded_v3'
BIN_EDGES=(.5,1.,2.,4.,8.)


def hierarchical_log_probs(fine,coarse,bins,valid):
    """Choose a time band first; its mass does not grow with candidate count."""
    membership=F.one_hot(bins,6).bool()&valid[...,None]
    present=membership.any(1)
    safe_coarse=coarse.float().masked_fill(~present,-1e4)
    coarse_log=safe_coarse.log_softmax(-1)
    energies=fine.float()[...,None].masked_fill(~membership,-1e4)
    normalizer=energies.logsumexp(1)
    within=fine.float()-normalizer.gather(1,bins)
    return (within+coarse_log.gather(1,bins)).masked_fill(~valid,-1e4)


def bin_pool(hidden,bins,valid):
    membership=F.one_hot(bins,6).to(hidden)*valid[...,None]
    return torch.einsum('bcg,bcd->bgd',membership,hidden)/membership.sum(1).clamp_min(1)[...,None]


class AudioGroundedComposer(ChartComposer):
    def __init__(self,config,atom_catalog):
        # Retain shared parameter names for explicit, partial finetune transfer.
        super().__init__(replace(config,architecture_profile='music_context_v2'))
        self.config=config
        width=config.d_model
        self.lens=AudioLens(config.lens_width,config.lens_chunk_size)
        self.lens_projection=nn.Linear(config.lens_width,width)
        self.placement_head=nn.Linear(config.lens_width+config.condition_dim,5)
        self.placement_projection=nn.Linear(5,width)
        self.planner=AudioPlanner(config.lens_width,config.condition_dim)
        self.plan_projection=nn.Linear(6,width)
        self.audio_fusion=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
        self.bin_embedding=nn.Embedding(6,width)
        self.bin_head=nn.Linear(width,1)
        self.history_pointer=nn.Linear(width,1)
        self.history_bin=nn.Linear(width,1)
        nn.init.zeros_(self.history_pointer.weight);nn.init.zeros_(self.history_pointer.bias)
        nn.init.zeros_(self.history_bin.weight);nn.init.zeros_(self.history_bin.bias)
        self.action_factors=CompositionalActions(list(atom_catalog),width,config.history_anchor_policy,config.slide_factor_policy)
        self.action_relation=nn.Linear(8,width,bias=False)
        nn.init.zeros_(self.action_relation.weight)
        if config.structure_policy != 'off':
            from .structure import structure_dim_for_policy
            structure_dim=structure_dim_for_policy(config.structure_policy)
            self.structure_projection=nn.Linear(structure_dim,width,bias=False)
            self.structure_action_projection=nn.Linear(structure_dim,width,bias=False)
            self.structure_transition_projection=nn.Linear(structure_dim,width,bias=False)
            # New feature branches have their own versioned input contract.
            # Neutral initialization also supports explicit transfer studies.
            nn.init.zeros_(self.structure_projection.weight)
            nn.init.zeros_(self.structure_action_projection.weight)
            nn.init.zeros_(self.structure_transition_projection.weight)
        if config.head_timing_policy=='separate_pads_v2':
            self.head_timing_detail_projection=nn.Linear(123,width,bias=False)
            nn.init.zeros_(self.head_timing_detail_projection.weight)
        if config.cross_event_relation_policy=='relative_heads_v1':
            self.cross_event_relative_factors=nn.Embedding(8,width)
            nn.init.zeros_(self.cross_event_relative_factors.weight)
            self.register_buffer('cross_event_geometry',self.action_factors._normalized_head_lanes.clone(),persistent=False)
            kinds=[None]+[(None if atom[0] not in '12345678' else
                2 if any(c in atom for c in '-<>^pqszvwV') else 1 if 'h' in atom else 0) for atom in atom_catalog]
            for kind in range(3):
                self.register_buffer(f'cross_event_kind_{kind}',torch.tensor(
                    [i for i,value in enumerate(kinds) if value==kind],dtype=torch.long),persistent=False)
        if config.motif_memory_policy != 'off':
            from .motif_memory import MotifMemoryAdapter
            self.motif_memory=MotifMemoryAdapter(width)
        if getattr(config,'future_horizon_policy','off')=='candidate_horizon_v1':
            from .future_horizon import FutureAudioHorizon
            # A valid base model may have e.g. width=30 and nhead=6. Its
            # width need not divide four, so choose a compatible small head
            # count instead of rejecting otherwise supported configurations.
            horizon_heads=min(4,config.nhead)
            while width % horizon_heads:
                horizon_heads-=1
            self.future_horizon=FutureAudioHorizon(width,nhead=horizon_heads)
        if getattr(config,'action_head_policy','flat_v1')=='conserved_grouped_v2':
            from .conserved_actions import ConservedGroupedActionHead
            self.conserved_action_head=ConservedGroupedActionHead(self.action_factors,width)
        # A full ledger still describes valid causal state. Its direct and
        # discrepancy routes cannot write an unbounded pointer logit anymore.

    def _embedding(self,ids,*,factor_weight=None):
        return (self.event_embedding(ids)+self.action_factors.embed(ids,composed_weight=factor_weight))*ids.ne(0)[...,None]

    def forward(self,candidate_features,history,condition,action_state=None,history_features=None,
                candidate_mask=None,history_padding_mask=None,candidate_indices=None,
                decode_actions=True,workload_audio=None,ledger=None,action_targets=None,
                curve_strength=1.,interval_audio=None,trailing_audio=None,workload_override=None,
                symbolic_keep=None,audio_auxiliary=True,head_timing=None,audio_lens=None,
                audio_lens_mask=None,planner_audio=None,planner_mask=None,planner_indices=None,
                planner_mix=None,planner_override=None,lens_embeddings=None,
                motif_ids=None,motif_relative_beats=None,motif_mask=None,motif_strength=1.,
                selected_action_state=None,structure_features=None,structure_transition=None,**unused):
        if (audio_lens is None and lens_embeddings is None) or planner_audio is None or head_timing is None:
            raise ValueError('audio_grounded_v3 requires local audio, coarse plan audio and causal head timing')
        # A full training decode already needs every atom's factor vector.
        # Reuse that differentiable table for history, teacher prefixes and
        # both output branches. Keep it local to this forward, never cached
        # across optimizer updates. Pointer-only calls retain sparse lookups.
        factor_weight=self.action_factors.composed_weight() if decode_actions else None
        batch,count=candidate_features.shape[:2]
        # Mask the same target-derived controls before EVERY internal consumer.
        # Otherwise planner/workload/placement predictions leak the hidden
        # condition back into the action context through their detached paths.
        effective_condition=condition
        if self.training and self.config.condition_mask_prob>0. and condition.size(-1)>1:
            cond_mask=(torch.rand(batch,1,device=condition.device)<self.config.condition_mask_prob)
            masked_tail=torch.where(cond_mask,condition.new_full((batch,condition.size(-1)-1),.5),condition[:,1:])
            effective_condition=torch.cat((condition[:,:1],masked_tail),dim=-1)
        structure_term=None
        if self.config.structure_policy != 'off':
            from .structure import structure_dim_for_policy
            if structure_features is None or structure_features.shape != (batch,count,structure_dim_for_policy(self.config.structure_policy)):
                raise ValueError('Structure-enabled Composer requires matching candidate structure features')
            structure_features=structure_features.to(candidate_features)
            if structure_transition is None:structure_transition=torch.zeros_like(structure_features)
            if structure_transition.shape!=structure_features.shape:
                raise ValueError('structure_transition shape must match structure_features')
            structure_transition=structure_transition.to(structure_features)
            if self.training and self.config.structure_dropout:
                # One coherent mask for all candidates and all branches in
                # this training sample, never candidate-by-candidate flicker.
                keep=(torch.rand(batch,1,1,device=candidate_features.device)>=self.config.structure_dropout)
                structure_features=structure_features*keep
                structure_transition=structure_transition*keep
            structure_term=(self.structure_action_projection(structure_features)
                            +self.structure_transition_projection(structure_transition))
        valid=~candidate_mask if candidate_mask is not None else torch.ones((batch,count),device=candidate_features.device,dtype=torch.bool)
        hidden=self.audio_projection(candidate_features[...,:self.config.audio_dim])+self.time_projection(candidate_features[...,self.config.audio_dim:])
        hidden=self.audio_encoder(hidden,src_key_padding_mask=~valid)
        if structure_term is not None:
            hidden=hidden+self.structure_projection(structure_features)
        lens=self.lens(audio_lens,audio_lens_mask) if lens_embeddings is None else lens_embeddings
        if self.training and lens_embeddings is not None:
            raise ValueError('Cached audio embeddings are inference-only; training must retain lens gradients')
        placement=self.placement_head(torch.cat((lens,effective_condition[:,None].expand(-1,lens.size(1),-1)),-1)).float()
        # Explicit external overrides remain explicit evidence (e.g. an edited
        # generation curve); masking only applies to model-computed controls.
        plan=(self.planner(planner_audio,effective_condition,planner_mask) if planner_override is None else planner_override.to(hidden))
        candidate_plan=read_plan(plan,planner_indices,planner_mix)
        workload=self.predict_workload(workload_audio,effective_condition)
        decision_workload=workload if workload_override is None else workload_override.to(workload)
        audio_context=hidden+self.lens_projection(lens[:,:count])+self.condition_projection(effective_condition)[:,None]
        audio_context=audio_context+self.interval_projection(interval_audio)
        audio_context=audio_context+float(curve_strength)*(self.workload_projection(decision_workload[:,:count].detach())+self.plan_projection(candidate_plan.detach()))
        # Occupancy is chart-conditional evidence, not a mandatory note rule.
        audio_context=audio_context+self.placement_projection(placement[:,:count].sigmoid().detach())
        audio_context=self.audio_fusion(audio_context)
        if symbolic_keep is None:
            symbolic_keep=((torch.rand(batch,1,device=hidden.device)>=self.config.symbolic_dropout).to(hidden)
                if self.training else hidden.new_ones(batch,1))
        else:symbolic_keep=symbolic_keep.to(hidden)
        symbolic=torch.zeros_like(audio_context)
        last_heads=hidden.new_zeros(batch,8)
        previous_heads=hidden.new_zeros(batch,8)
        if history is not None and history.size(1):
            if history.ndim==2:history=history.unsqueeze(-1)
            padding=history_padding_mask if history_padding_mask is not None else torch.zeros(history.shape[:2],device=history.device,dtype=torch.bool)
            lengths=(~padding).sum(1)
            positions=(lengths[:,None]-1-torch.arange(history.size(1),device=history.device)[None]).clamp_min(0)
            # The checkpoint policy selects historical digit anchors or parsed
            # head angles. Factors and the original embedding retain note kinds,
            # modifiers and touch rings; center Touch has no outer angle.
            anchors=self.action_factors.history_anchor_embedding(history,self.event_embedding)
            hist_emb=(.75*anchors+.25*self.event_embedding(history)+self.action_factors.embed(history,composed_weight=factor_weight))*history.ne(0)[...,None]
            memory=hist_emb.sum(-2)+self.position_embedding(positions.clamp_max(self.config.history_events+3))
            if history_features is not None:memory=memory+self.history_feature_projection(history_features)
            safe_padding=padding.clone()
            if self.training and self.config.history_dropout>0.:
                # Drop across all history positions including t-1 and t-2 so
                # note types/modifiers cannot rely on a 2-gram copy shortcut.
                safe_padding |= torch.rand(padding.shape,device=hidden.device)<self.config.history_dropout
            rows=torch.arange(batch,device=history.device)
            tail_valid=(~safe_padding[rows,(lengths-1).clamp_min(0)])&lengths.gt(0)
            all_masked=safe_padding.all(dim=1)
            memory=memory.masked_fill(safe_padding.unsqueeze(-1),0.)
            attn_padding=safe_padding.clone()
            attn_padding[:,0]&=~all_masked
            memory=self.history_encoder(memory,src_key_padding_mask=attn_padding)
            if self.config.history_memory_policy=='rms_cap_v1':
                memory=bounded_history_memory(memory,safe_padding,self.config.history_memory_rms_bound)
            with history_attention_context(self.config.history_attention_policy,audio_context.device.type):
                attended,_=self.history_attention(audio_context,memory,memory,key_padding_mask=attn_padding,need_weights=False)
            attended=attended*(~all_masked)[:,None,None]
            tail=memory[rows,(lengths-1).clamp_min(0)]*tail_valid[:,None]
            symbolic=symbolic+(tail[:,None]+self.history_gate.tanh()*attended)*lengths.gt(0)[:,None,None]
            last_ids=history[rows,(lengths-1).clamp_min(0)]
            last_heads=self.action_factors.atom_head_lanes[last_ids].any(-2).to(hidden)*tail_valid[:,None]
            if hasattr(self,'cross_event_relative_factors'):
                previous_heads=last_heads
        if action_state is not None:
            projected_state=self.action_projection(action_state)
            symbolic=symbolic+(projected_state[:,None] if action_state.ndim==2 else projected_state)
        timing=(merge_separate_head_timing(head_timing) if self.config.head_timing_policy=='separate_pads_v2' else head_timing)
        symbolic=symbolic+self.head_timing_projection(timing.to(hidden))
        if hasattr(self,'head_timing_detail_projection'):
            symbolic=symbolic+self.head_timing_detail_projection(head_timing.to(hidden))
        trailing=self.predict_trailing(trailing_audio,effective_condition) if trailing_audio is not None else None
        if ledger is not None:
            keep=((torch.rand(batch,1,1,device=hidden.device)>=self.config.ledger_dropout).to(hidden)
                if self.training else 1.)
            ledger_context=self.ledger_projection(ledger.to(hidden))
            if trailing is not None:
                difference=trailing[:,:count].detach()-ledger[...,:4].float().clamp_min(0).log1p()
                ledger_context=ledger_context+float(curve_strength)*self.discrepancy_projection(difference.to(hidden))
            symbolic=symbolic+keep*ledger_context
        symbolic=symbolic*symbolic_keep[:,None]
        bins=torch.bucketize(candidate_features[...,self.config.audio_dim].contiguous(),hidden.new_tensor(BIN_EDGES))
        fine=self.pointer(audio_context).squeeze(-1).float()
        bin_hidden=bin_pool(audio_context,bins,valid)+self.bin_embedding.weight
        coarse=self.bin_head(bin_hidden).squeeze(-1).float()
        fine_res=self.config.history_residual_bound*torch.tanh(self.history_pointer(symbolic).squeeze(-1).float())
        bin_res=self.config.history_residual_bound*torch.tanh(self.history_bin(bin_pool(symbolic,bins,valid)).squeeze(-1).float())
        audio_logits=hierarchical_log_probs(fine,coarse,bins,valid)
        proposed=hierarchical_log_probs(fine+fine_res,coarse+bin_res,bins,valid)
        # Bound the FINAL candidate residual too. Bounding intermediate heads
        # alone does not bound odds after per-bin normalizers are subtracted.
        delta=(proposed-audio_logits)*valid
        delta=delta-delta.sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        delta=delta*valid
        correction=self.config.history_residual_bound*torch.tanh(delta/self.config.history_residual_bound)
        full_logits=(audio_logits+correction).masked_fill(~valid,-1e4).log_softmax(-1)
        if structure_term is not None:
            # Explicit configuration path, after the timing decision logits.
            # This also survives selected-prefix reconstruction below.
            audio_context=audio_context+structure_term
        if hasattr(self,'future_horizon'):
            audio_context=audio_context+self.future_horizon(
                audio_context,candidate_features,valid,audio_dim=self.config.audio_dim)
        audio_action_context=self.candidate_fusion(audio_context)
        action_context=self.candidate_fusion(audio_context+symbolic+self.action_relation(last_heads)[:,None]*symbolic_keep[:,None])
        result={'candidate_logits':full_logits,'candidate_context':action_context,
            'audio_action_context':audio_action_context,
            'audio_candidate_logits':audio_logits,'candidate_bins':bins,
            'bin_logits':coarse+bin_res,'placement_logits':placement,'plan':plan,'workload':workload,
            'pointer_residual_rms':(correction.square()*valid).sum().sqrt()/valid.sum().clamp_min(1).sqrt(),
            'plan_context_rms':candidate_plan.square().mean().sqrt()}
        if trailing is not None:result['trailing_workload']=trailing
        if hasattr(self,'cross_event_relative_factors'):
            result['previous_heads']=previous_heads*symbolic_keep
        if self.config.action_state_policy=='selected_prefix_v3':
            # Keep the committed snapshot in the pointer. Only the chosen
            # action query replaces it with state advanced to that time.
            state_term=(projected_state[:,None] if action_state.ndim==2 else projected_state) if action_state is not None else 0.
            pre=audio_context+symbolic+self.action_relation(last_heads)[:,None]*symbolic_keep[:,None]
            result['selected_state_base']=pre-state_term*symbolic_keep[:,None]
            result['selected_state_keep']=symbolic_keep
        if decode_actions:
            result.update(self.decode_candidate(action_context,candidate_indices,
                action_targets=action_targets,audio_context=audio_action_context,project_audio_actions=audio_auxiliary,
                motif_ids=motif_ids,motif_relative_beats=motif_relative_beats,motif_mask=motif_mask,
                motif_strength=motif_strength,motif_keep=symbolic_keep,
                selected_action_state=selected_action_state,state_base=result.get('selected_state_base'),
                state_keep=result.get('selected_state_keep'),factor_weight=factor_weight,
                previous_heads=result.get('previous_heads')))
        return result

    def decode_candidate(self,candidate_context,candidate_indices=None,*,action_targets=None,project_actions=True,audio_context=None,
                         motif_ids=None,motif_relative_beats=None,motif_mask=None,motif_strength=1.,motif_keep=None,
                         selected_action_state=None,state_base=None,state_keep=None,factor_weight=None,
                         project_audio_actions=True,previous_heads=None):
        if hasattr(self,'cross_event_relative_factors') and previous_heads is None:
            raise ValueError('Cross-event action decoding requires causal previous_heads from the same forward')
        if previous_heads is not None and previous_heads.shape!=(candidate_context.size(0),8):
            raise ValueError('previous_heads must match the forward batch [B,8]')
        if project_actions and factor_weight is None:
            factor_weight=self.action_factors.composed_weight()
        if candidate_indices is not None:
            rows=torch.arange(candidate_context.size(0),device=candidate_context.device)
            candidate_context=candidate_context[rows,candidate_indices]
            if audio_context is not None:
                audio_context=audio_context[rows,candidate_indices]
            if state_base is not None:state_base=state_base[rows,candidate_indices]
        result={'future':self.future_head(candidate_context),'eos_logits':self.eos_head(candidate_context).squeeze(-1),'action_context':candidate_context}
        if previous_heads is not None:result['previous_heads']=previous_heads
        if self.config.action_state_policy=='selected_prefix_v3':
            if selected_action_state is None or state_base is None:
                raise ValueError('selected_prefix_v3 requires a causal state at the selected beat')
            if selected_action_state.shape!=(*state_base.shape[:-1],self.config.action_dim):
                raise ValueError('Selected action state must match the selected candidate shape')
            keep=state_keep if state_keep is not None else state_base.new_ones(state_base.shape[0],1)
            while keep.ndim<state_base.ndim:keep=keep.unsqueeze(-2)
            candidate_context=self.candidate_fusion(state_base+self.action_projection(selected_action_state)*keep)
            result['action_context']=candidate_context
        # Motifs guide configurations only. The learned timing pointer, EOS,
        # future summary and audio-only action reference retain their inputs.
        if hasattr(self,'motif_memory') and motif_ids is not None:
            if motif_relative_beats is None or motif_mask is None:
                raise ValueError('Motif IDs require relative timing and visibility mask')
            motif_residual=self.motif_memory(
                candidate_context,lambda ids:self._embedding(ids,factor_weight=factor_weight),
                motif_ids,motif_relative_beats,motif_mask)
            # The archive is symbolic history too. Coherent history dropout
            # must not leave a second token-history route into action decoding.
            if motif_keep is not None:
                keep=motif_keep.to(motif_residual)
                while keep.ndim<motif_residual.ndim:keep=keep.unsqueeze(-2)
                motif_residual=motif_residual*keep
            candidate_context=candidate_context+float(motif_strength)*motif_residual
            result['action_context']=candidate_context
        if audio_context is not None:
            result['audio_action_context']=audio_context
        if project_actions:
            slot_embeddings=self.slot_embedding.weight
            if action_targets is not None:
                if not 1<=action_targets.size(-1)<=self.config.action_slots:
                    raise ValueError('Action target slots exceed the configured capacity')
                slot_embeddings=slot_embeddings[:action_targets.size(-1)]
            contexts=candidate_context.unsqueeze(-2)+slot_embeddings
            audio_contexts=(audio_context.unsqueeze(-2)+slot_embeddings
                            if audio_context is not None and project_audio_actions else None)
            relation=None
            if action_targets is not None:
                embedded=self._embedding(action_targets,factor_weight=factor_weight)
                prefix=embedded.cumsum(-2)-embedded
                contexts=contexts+prefix
                if audio_contexts is not None:
                    audio_contexts=audio_contexts+prefix
                heads=self.action_factors.atom_head_lanes[action_targets].to(contexts)
                relation=(heads.cumsum(-2)-heads)>0
            contexts=self.slot_fusion(contexts)
            full_logits=self.action_head(contexts)+F.linear(contexts,factor_weight)
            conserved_tables=self.conserved_action_head.detail_tables() if hasattr(self,'conserved_action_head') else None
            if conserved_tables is not None:
                full_logits=self.conserved_action_head(full_logits,contexts,tables=conserved_tables)
            if relation is not None:
                # Same-event head geometry is also a macro position decision,
                # not a trajectory/duration score to normalize away per pad.
                full_logits=full_logits+self.action_factors.relative_logits(contexts,relation)
            if hasattr(self,'cross_event_relative_factors'):
                # Relative movement is a macro starting-pad decision. Applying
                # its per-pad shift before within-group normalization would
                # cancel the entire learned branch in the grouped head.
                first=self._cross_event_logits(full_logits[...,0,:],contexts[...,0,:],previous_heads)
                full_logits=torch.cat((first.unsqueeze(-2),full_logits[...,1:,:]),-2)
            result['action_logits']=full_logits
            if audio_contexts is not None:
                audio_contexts=self.slot_fusion(audio_contexts)
                audio_logits=self.action_head(audio_contexts)+F.linear(audio_contexts,factor_weight)
                if conserved_tables is not None:
                    audio_logits=self.conserved_action_head(audio_logits,audio_contexts,tables=conserved_tables)
                if relation is not None:
                    audio_logits=audio_logits+self.action_factors.relative_logits(audio_contexts,relation)
                result['audio_action_logits']=audio_logits
        return result

    def _cross_event_logits(self,logits,context,previous_heads,strength=1.):
        """Learn button movement without changing Tap/Hold/Slide/Touch masses."""
        delta=float(strength)*self.action_factors.relative_logits(context,previous_heads,
            factor_embedding=self.cross_event_relative_factors,target_geometry=self.cross_event_geometry)
        result=logits
        for kind in range(3):
            ids=getattr(self,f'cross_event_kind_{kind}')
            if not ids.numel():continue
            original=logits.index_select(-1,ids).float()
            adjusted=original+delta.index_select(-1,ids).float()
            shift=original.logsumexp(-1,keepdim=True)-adjusted.logsumexp(-1,keepdim=True)
            adjusted=adjusted+shift
            result=result.index_copy(-1,ids,adjusted.to(logits.dtype))
        return result

    def action_slot(self,context,slot,selected_ids,*,audio_context=None,history_strength=1.,previous_heads=None):
        if audio_context is not None and float(history_strength)<1.:
            alpha=max(0.,min(1.,float(history_strength)))
            context=alpha*context+(1.-alpha)*audio_context
        if selected_ids:
            ids=torch.as_tensor(selected_ids,dtype=torch.long,device=context.device)
            context=context+self._embedding(ids).sum(0)
        slot_context=self.slot_fusion(context+self.slot_embedding.weight[slot])
        logits=self.action_head(slot_context)+self.action_factors(slot_context)
        if hasattr(self,'conserved_action_head'):
            logits=self.conserved_action_head(logits,slot_context)
        if selected_ids:
            logits=logits+self.action_factors.relative_logits(slot_context,self.action_factors.atom_head_lanes[ids].any(0))
        if slot==0 and previous_heads is not None and hasattr(self,'cross_event_relative_factors'):
            logits=self._cross_event_logits(logits,slot_context,previous_heads,history_strength)
        return logits
