"""One phrase choice, a shared causal executor, and loss-only future relations.

No target sequence is read here. The small mode axis belongs to output heads,
never to the audio/history encoders or the large detail vocabulary.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PhraseExecution(nn.Module):
    def __init__(self, width, phrase_width, codec, modes=4):
        super().__init__()
        self.modes=modes
        self.routes=codec.num_shapes-1+31
        small=min(128,width)
        self.prior=nn.Sequential(nn.LayerNorm(phrase_width),nn.Linear(phrase_width,modes))
        self.intent=nn.Linear(phrase_width,small,bias=False)
        self.current=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,small))
        self.choices=nn.Parameter(torch.randn(modes,small)*.04)
        self.executor=nn.Sequential(nn.GELU(),nn.Linear(small,small),nn.GELU(),nn.LayerNorm(small))
        self.route=nn.Linear(small,self.routes,bias=False)
        self.lanes=nn.Linear(small,8,bias=False)
        # Each mode chooses how to read old geometry, including a null/new
        # writing alternative. It never forces the corresponding transformation.
        self.reference_choice=nn.Linear(small,17)
        self.reference_gain=nn.Parameter(torch.tensor(.05))
        from .stream_phrase_training import RELATION_DIM,RELATION_HORIZONS
        self.relations=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,small),nn.GELU(),
                                    nn.Linear(small,RELATION_DIM*RELATION_HORIZONS))
        nn.init.zeros_(self.prior[-1].weight);nn.init.zeros_(self.prior[-1].bias)
        nn.init.normal_(self.route.weight,std=.005);nn.init.normal_(self.lanes.weight,std=.005)
        # Real head sets, not differences between unrelated codec phase origins.
        patterns=torch.zeros(codec.num_shapes,8,8)
        for shape,slots in enumerate(codec.shape_catalog):
            if not shape:continue
            for phase in range(8):
                for kind,head in slots:
                    if kind in ('tap','hold','slide'):patterns[shape,phase,(head+phase)%8]=1.
        patterns=patterns/patterns.sum(-1,keepdim=True).clamp_min(1).sqrt()
        self.register_buffer('phase_sets',patterns,persistent=False)
        transforms=torch.tensor([[(sign*j+shift)%8 for j in range(8)]
                                 for sign in (1,-1) for shift in range(8)])
        self.register_buffer('transforms',transforms,persistent=False)

    def forward(self,action,phrase):
        intent=phrase['phrase_intent']
        # Candidate preparation can contain a last-position-only phrase view.
        if intent.shape[:-1]!=action.shape[:-1]:
            if intent.size(-2)==1:action=action[...,-1:,:]
            else:raise ValueError('Phrase and action query layouts differ')
        anchor=self.intent(intent)
        chosen=anchor[...,None,:]+self.choices
        state=self.executor(self.current(action)[...,None,:]+chosen)
        lanes=self.lanes(state).float()
        reference=phrase.get('phrase_aligned_heads')
        if reference is not None:
            transformed=reference.float()[...,self.transforms]
            options=torch.cat((transformed.new_zeros(*transformed.shape[:-2],1,8),transformed),-2)
            probabilities=self.reference_choice(chosen).float().softmax(-1)
            lanes=lanes+self.reference_gain*(probabilities@options)
        lanes=lanes-lanes.mean(-1,keepdim=True)
        return dict(prior_logits=self.prior(intent).float(),route_bias=self.route(state).float(),lane_bias=lanes)

    def phase_bias(self,lanes,shapes):
        sets=self.phase_sets[shapes]
        if lanes.ndim==sets.ndim-1:
            return (sets*lanes[...,None,:]).sum(-1)
        return (sets[...,None,:,:]*lanes[...,None,:]).sum(-1)


def whole_phrase_configuration_loss(mode_nll,prior_logits,batch,weights,normalizer):
    """Marginalize one mode after accumulating a complete trusted phrase.

    Weighted event scores preserve the existing chart/window budget. For
    fragments or ambiguous recovery rows only the local mixture is trained;
    they are never called complete-phrase likelihood observations.
    """
    prior=prior_logits.float().log_softmax(-1)
    local=-torch.logsumexp(prior-mode_nll.float(),-1)
    groups=batch.get('training_phrase_group')
    complete=batch.get('training_phrase_complete')
    zero=mode_nll.sum()*0.
    if groups is None or complete is None or batch['training_phrase_group_weight'].numel()==0:
        return (local*weights).sum()/normalizer,dict(phrase_complete_groups=zero,
            phrase_mode_posterior_entropy=zero,phrase_mode_prior_entropy=(-(prior.exp()*prior).sum(-1)*weights).sum()/weights.sum().clamp_min(1))
    valid=complete.bool()&groups.ge(0)&weights.gt(0)
    count=batch['training_phrase_group_weight'].numel()
    # Fixed group count is supplied by CPU collation; no GPU max/item sync.
    index=groups.clamp_min(0).flatten()
    mass=(weights*valid).flatten()
    summed=mode_nll.new_zeros(count,mode_nll.size(-1)).index_add(0,index,mode_nll.flatten(0,1)*mass[:,None])
    group_mass=weights.new_zeros(count).index_add(0,index,mass)
    first=valid&batch['training_phrase_position'].eq(0)
    group_prior=prior.new_zeros(count,prior.size(-1)).index_add(0,index,prior.flatten(0,1)*first.flatten()[:,None])
    has_first=weights.new_zeros(count).index_add(0,index,first.flatten().to(weights)).gt(0)
    active=group_mass.gt(0)&has_first
    # A missing/zero-weight first event cannot silently invent a uniform prior.
    use=valid&active[groups.clamp_min(0)]
    joint=group_prior-summed
    grouped=-torch.logsumexp(joint,-1)
    total=((local*weights).masked_fill(use,0).sum()+grouped.masked_fill(~active,0).sum())/normalizer
    posterior=joint.log_softmax(-1)
    entropy=-(posterior.exp()*posterior).sum(-1)
    return total,dict(phrase_complete_groups=active.sum().float(),
        phrase_mode_posterior_entropy=entropy.masked_fill(~active,0).sum()/active.sum().clamp_min(1),
        phrase_mode_prior_entropy=(-(prior.exp()*prior).sum(-1)*weights).sum()/weights.sum().clamp_min(1))


def future_relation_loss(prediction,batch,weights,normalizer):
    if 'target_relation' not in batch:return prediction.sum()*0.
    target=batch['target_relation'].float()
    mask=batch['target_relation_mask'].bool()&weights.gt(0)[...,None,None]
    safe_target=target.masked_fill(~mask,0.)
    classification=F.binary_cross_entropy_with_logits(prediction.float(),safe_target.clamp(0,1),reduction='none')
    timing=F.smooth_l1_loss(F.softplus(prediction[...,16:18].float()),safe_target[...,16:18],reduction='none')
    error=torch.cat((classification[...,:16],timing,classification[...,18:]),-1)
    per_event=(error*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)
    return (per_event*weights).sum()/normalizer
