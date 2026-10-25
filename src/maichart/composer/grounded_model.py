"""Audio-grounded Composer: ordered listening, hierarchical time, shared actions."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .model import ChartComposer
from .audio_lens import AudioLens
from .action_factors import CompositionalActions
from .planning import AudioPlanner,read_plan

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
        self.action_factors=CompositionalActions(list(atom_catalog),width)
        self.action_relation=nn.Linear(8,width,bias=False)
        nn.init.zeros_(self.action_relation.weight)
        # A full ledger still describes valid causal state. Its direct and
        # discrepancy routes cannot write an unbounded pointer logit anymore.

    def _embedding(self,ids):
        return (self.event_embedding(ids)+self.action_factors.embed(ids))*ids.ne(0)[...,None]

    def forward(self,candidate_features,history,condition,action_state=None,history_features=None,
                candidate_mask=None,history_padding_mask=None,candidate_indices=None,
                decode_actions=True,workload_audio=None,ledger=None,action_targets=None,
                curve_strength=1.,interval_audio=None,trailing_audio=None,workload_override=None,
                symbolic_keep=None,audio_auxiliary=True,head_timing=None,audio_lens=None,
                audio_lens_mask=None,planner_audio=None,planner_mask=None,planner_indices=None,
                planner_mix=None,planner_override=None,lens_embeddings=None,**unused):
        if (audio_lens is None and lens_embeddings is None) or planner_audio is None or head_timing is None:
            raise ValueError('audio_grounded_v3 requires local audio, coarse plan audio and causal head timing')
        batch,count=candidate_features.shape[:2]
        valid=~candidate_mask if candidate_mask is not None else torch.ones((batch,count),device=candidate_features.device,dtype=torch.bool)
        hidden=self.audio_projection(candidate_features[...,:self.config.audio_dim])+self.time_projection(candidate_features[...,self.config.audio_dim:])
        hidden=self.audio_encoder(hidden,src_key_padding_mask=~valid)
        lens=self.lens(audio_lens,audio_lens_mask) if lens_embeddings is None else lens_embeddings
        if self.training and lens_embeddings is not None:
            raise ValueError('Cached audio embeddings are inference-only; training must retain lens gradients')
        placement=self.placement_head(torch.cat((lens,condition[:,None].expand(-1,lens.size(1),-1)),-1)).float()
        plan=(self.planner(planner_audio,condition,planner_mask) if planner_override is None else planner_override.to(hidden))
        candidate_plan=read_plan(plan,planner_indices,planner_mix)
        workload=self.predict_workload(workload_audio,condition)
        decision_workload=workload if workload_override is None else workload_override.to(workload)
        audio_context=hidden+self.lens_projection(lens[:,:count])+self.condition_projection(condition)[:,None]
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
        if history is not None and history.size(1):
            if history.ndim==2:history=history.unsqueeze(-1)
            padding=history_padding_mask if history_padding_mask is not None else torch.zeros(history.shape[:2],device=history.device,dtype=torch.bool)
            lengths=(~padding).sum(1)
            positions=(lengths[:,None]-1-torch.arange(history.size(1),device=history.device)[None]).clamp_min(0)
            # Anchor history token embeddings to their physical button lane (1..8)
            # while action_factors.embed(history) and history_features carry the
            # note kind/modifier factors, preventing 18,454-ID Touch/EX copy lock.
            lane_ids=self.action_factors._lane_anchor_ids[history]
            hist_emb=(.75*self.event_embedding(lane_ids)+.25*self.event_embedding(history)+self.action_factors.embed(history))*history.ne(0)[...,None]
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
            attended,_=self.history_attention(audio_context,memory,memory,key_padding_mask=attn_padding,need_weights=False)
            attended=attended*(~all_masked)[:,None,None]
            tail=memory[rows,(lengths-1).clamp_min(0)]*tail_valid[:,None]
            symbolic=symbolic+(tail[:,None]+self.history_gate.tanh()*attended)*lengths.gt(0)[:,None,None]
            last_ids=history[rows,(lengths-1).clamp_min(0)]
            last_heads=self.action_factors.atom_head_lanes[last_ids].any(-2).to(hidden)*tail_valid[:,None]
        if action_state is not None:symbolic=symbolic+self.action_projection(action_state)[:,None]
        symbolic=symbolic+self.head_timing_projection(head_timing.to(hidden))
        trailing=self.predict_trailing(trailing_audio,condition) if trailing_audio is not None else None
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
        audio_action_context=self.candidate_fusion(audio_context)
        action_context=self.candidate_fusion(audio_context+symbolic+self.action_relation(last_heads)[:,None]*symbolic_keep[:,None])
        result={'candidate_logits':full_logits,'candidate_context':action_context,
            'audio_action_context':audio_action_context,
            'audio_candidate_logits':audio_logits,'candidate_bins':bins,
            'bin_logits':coarse+bin_res,'placement_logits':placement,'plan':plan,'workload':workload,
            'pointer_residual_rms':(correction.square()*valid).sum().sqrt()/valid.sum().clamp_min(1).sqrt(),
            'plan_context_rms':candidate_plan.square().mean().sqrt()}
        if trailing is not None:result['trailing_workload']=trailing
        if decode_actions:
            result.update(self.decode_candidate(action_context,candidate_indices,
                action_targets=action_targets,audio_context=audio_action_context))
        return result

    def decode_candidate(self,candidate_context,candidate_indices=None,*,action_targets=None,project_actions=True,audio_context=None):
        if candidate_indices is not None:
            rows=torch.arange(candidate_context.size(0),device=candidate_context.device)
            candidate_context=candidate_context[rows,candidate_indices]
            if audio_context is not None:
                audio_context=audio_context[rows,candidate_indices]
        result={'future':self.future_head(candidate_context),'eos_logits':self.eos_head(candidate_context).squeeze(-1),'action_context':candidate_context}
        if audio_context is not None:
            result['audio_action_context']=audio_context
        if project_actions:
            slot_embeddings=self.slot_embedding.weight
            if action_targets is not None:
                if not 1<=action_targets.size(-1)<=self.config.action_slots:
                    raise ValueError('Action target slots exceed the configured capacity')
                slot_embeddings=slot_embeddings[:action_targets.size(-1)]
            contexts=candidate_context.unsqueeze(-2)+slot_embeddings
            audio_contexts=audio_context.unsqueeze(-2)+slot_embeddings if audio_context is not None else None
            relation=None
            if action_targets is not None:
                embedded=self._embedding(action_targets)
                prefix=embedded.cumsum(-2)-embedded
                contexts=contexts+prefix
                if audio_contexts is not None:
                    audio_contexts=audio_contexts+prefix
                heads=self.action_factors.atom_head_lanes[action_targets].to(contexts)
                relation=(heads.cumsum(-2)-heads)>0
            contexts=self.slot_fusion(contexts)
            full_logits=self.action_head(contexts)+self.action_factors(contexts)
            if relation is not None:
                full_logits=full_logits+self.action_factors.relative_logits(contexts,relation)
            result['action_logits']=full_logits
            if audio_contexts is not None:
                audio_contexts=self.slot_fusion(audio_contexts)
                audio_logits=self.action_head(audio_contexts)+self.action_factors(audio_contexts)
                if relation is not None:
                    audio_logits=audio_logits+self.action_factors.relative_logits(audio_contexts,relation)
                result['audio_action_logits']=audio_logits
        return result

    def action_slot(self,context,slot,selected_ids,*,audio_context=None,history_strength=1.):
        if audio_context is not None and float(history_strength)<1.:
            alpha=max(0.,min(1.,float(history_strength)))
            context=alpha*context+(1.-alpha)*audio_context
        if selected_ids:
            ids=torch.as_tensor(selected_ids,dtype=torch.long,device=context.device)
            context=context+self._embedding(ids).sum(0)
        slot_context=self.slot_fusion(context+self.slot_embedding.weight[slot])
        logits=self.action_head(slot_context)+self.action_factors(slot_context)
        if selected_ids:
            logits=logits+self.action_factors.relative_logits(slot_context,self.action_factors.atom_head_lanes[ids].any(0))
        return logits
