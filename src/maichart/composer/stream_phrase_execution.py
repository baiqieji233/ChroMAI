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
            choice_logits=self.reference_choice(chosen).float()
            if hasattr(self,'reference_evidence'):
                null=phrase.get('phrase_aligned_null_weight',torch.ones_like(intent[...,0]))
                weights=phrase.get('phrase_reference_weights')
                available=1-weights[...,0] if weights is not None else torch.zeros_like(null)
                evidence=torch.stack((1-null,available),-1).to(chosen)
                choice_logits=choice_logits+self.reference_evidence(evidence)[...,None,:]
            probabilities=choice_logits.softmax(-1)
            lanes=lanes+self.reference_gain*(probabilities@options)
        lanes=lanes-lanes.mean(-1,keepdim=True)
        return dict(prior_logits=self.prior(intent).float(),route_bias=self.route(state).float(),lane_bias=lanes)

    def phase_bias(self,lanes,shapes):
        sets=self.phase_sets[shapes]
        if lanes.ndim==sets.ndim-1:
            return (sets*lanes[...,None,:]).sum(-1)
        return (sets[...,None,:,:]*lanes[...,None,:]).sum(-1)


def whole_phrase_configuration_loss(mode_nll,prior_logits,batch,weights,normalizer, *,
                                    song_logits=None,song_phrase_logits=None):
    """Marginalize one mode after accumulating a complete trusted phrase.

    Weighted event scores preserve the existing chart/window budget. For
    fragments or ambiguous recovery rows only the local mixture is trained;
    they are never called complete-phrase likelihood observations.
    """
    if song_logits is not None and song_phrase_logits is not None:
        return song_response_configuration_loss(mode_nll,prior_logits,batch,weights,normalizer,
                                                song_logits,song_phrase_logits)
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


def song_response_configuration_loss(mode_nll,prior_logits,batch,weights,normalizer,song_logits,song_phrase_logits):
    """Nested likelihood of sampled complete phrases, never a whole-chart claim.

    Each target retains its original weight. The song prior is charged once
    per sampled group; complete phrase modes are marginalized inside it.
    """
    # Unpaired, incomplete and recovery observations use a local mixture of
    # response-conditioned phrase priors, without claiming song evidence.
    phrase_conditional=song_phrase_logits.float().log_softmax(-1)
    effective_prior=torch.logsumexp(song_logits.float().log_softmax(-1)[:,None,:,None]+phrase_conditional,-2)
    ordinary,metrics=whole_phrase_configuration_loss(mode_nll,effective_prior,batch,weights,normalizer)
    song_logp=song_logits.float().log_softmax(-1)
    zero=ordinary.detach()*0
    metrics.update(song_response_groups=zero,song_response_sampled_phrases=zero,
        song_response_posterior_entropy=zero,
        song_response_prior_entropy=-(song_logp.exp()*song_logp).sum(-1).mean())
    response=batch.get('training_song_response_group')
    count=int(batch.get('_song_response_group_count',0))
    if response is None or not count:return ordinary,metrics
    groups=batch['training_phrase_group'];complete=batch['training_phrase_complete'].bool()
    n=response.numel();index=groups.clamp_min(0).flatten()
    valid=complete & groups.ge(0) & weights.gt(0)
    first=valid & batch['training_phrase_position'].eq(0)
    mass=(weights*valid).flatten()
    summed=mode_nll.new_zeros(n,mode_nll.size(-1)).index_add(0,index,mode_nll.flatten(0,1)*mass[:,None])
    first_count=weights.new_zeros(n).index_add(0,index,first.flatten().to(weights))
    eligible=response.ge(0)&first_count.eq(1)
    phrase_prior=song_phrase_logits.float().log_softmax(-1)
    shape=(n,*phrase_prior.shape[-2:])
    pp=phrase_prior.new_zeros(shape).index_add(0,index,phrase_prior.flatten(0,1)*first.flatten()[:,None,None])
    base=effective_prior
    bp=base.new_zeros(n,base.size(-1)).index_add(0,index,base.flatten(0,1)*first.flatten()[:,None])
    old=-torch.logsumexp(bp-summed,-1)
    conditional=-torch.logsumexp(pp-summed[:,None,:],-1)
    rid=response.clamp_min(0)
    group_sizes=weights.new_zeros(count).index_add(0,rid,eligible.to(weights))
    active=eligible & group_sizes[rid].ge(2)
    conditional_sum=conditional.new_zeros(count,conditional.size(-1)).index_add(0,rid,conditional*active[:,None])
    song_rows=song_logits[:,None,:].expand(*groups.shape,song_logits.size(-1)).float().log_softmax(-1)
    sp=song_rows.new_zeros(n,song_rows.size(-1)).index_add(0,index,song_rows.flatten(0,1)*first.flatten()[:,None])
    # All rows in one chart have the same audio+condition prior; average is
    # invariant to which phrase the CPU grouping lists first.
    group_prior=sp.new_zeros(count,sp.size(-1)).index_add(0,rid,sp*active[:,None])/group_sizes.clamp_min(1)[:,None]
    group_active=group_sizes.ge(2)
    joint=group_prior-conditional_sum
    nested=-torch.logsumexp(joint,-1)
    adjustment=(nested.masked_fill(~group_active,0).sum()-old.masked_fill(~active,0).sum())/normalizer
    posterior=joint.log_softmax(-1)
    metrics.update(song_response_groups=group_active.sum().float(),
        song_response_sampled_phrases=active.sum().float(),
        song_response_posterior_entropy=(-(posterior.exp()*posterior).sum(-1)).masked_fill(~group_active,0).sum()/group_active.sum().clamp_min(1))
    return ordinary+adjustment,metrics


def unified_profile_response_loss(mode_nll,song_phrase_logits,batch,weights,normalizer,*,
        prior_logits,component_profile_nll,profile_chart_weights,profile_weight=1.,raw_joint_diagnostics=False):
    """Balanced hierarchical variational objective, not raw sequence joint NLL.

    One persistent m explains a complete phrase. Its variational data term
    and prior KL both carry the phrase's weight W: F_gz = W * softmin_q(mean
    weighted event NLL). Shared z uses the corresponding chart data/KL budget.
    Numerical profile evidence enters once per chart, only as supervision.
    Optional raw sequence-joint diagnostics are detached and off by default.
    """
    if normalizer<=0:raise ValueError('Unified loss needs a positive target budget')
    chart_inverse=batch['unified_chart_inverse'].long()
    count=prior_logits.size(0)
    if component_profile_nll.shape!=prior_logits.shape:raise ValueError('Unified profile component shape mismatch')
    if song_phrase_logits.size(-2)!=prior_logits.size(-1):raise ValueError('Unified latent component mismatch')
    conditional=song_phrase_logits.float().log_softmax(-1)
    local=-torch.logsumexp(conditional-mode_nll.float()[:,:,None,:],-1)
    chart_nll=local.new_zeros(count,local.size(-1)).index_add(0,chart_inverse,(local*weights[...,None]).sum(1))
    raw_chart_nll=chart_nll.detach().clone() if raw_joint_diagnostics else None
    zero=chart_nll.sum()*0.
    complete_groups=zero
    phrase_mode_scores=None
    groups=batch.get('training_phrase_group')
    complete=batch.get('training_phrase_complete')
    group_weights=batch.get('training_phrase_group_weight')
    if groups is not None and complete is not None and group_weights is not None and group_weights.numel():
        group_count=group_weights.numel();index=groups.clamp_min(0).flatten()
        valid=complete.bool()&groups.ge(0)&weights.gt(0)
        first=valid&batch['training_phrase_position'].eq(0)
        first_count=weights.new_zeros(group_count).index_add(0,index,first.flatten().to(weights))
        active=first_count.eq(1)
        mass=(weights*valid).flatten()
        summed=mode_nll.new_zeros(group_count,mode_nll.size(-1)).index_add(0,index,mode_nll.flatten(0,1)*mass[:,None])
        pp=conditional.new_zeros(group_count,*conditional.shape[-2:]).index_add(0,index,conditional.flatten(0,1)*first.flatten()[:,None,None])
        phrase_mass=weights.new_zeros(group_count).index_add(0,index,mass)
        mean_mode_nll=summed/phrase_mass.clamp_min(1e-8)[:,None]
        phrase_mode_scores=pp-mean_mode_nll[:,None,:]
        phrase_nll=(-phrase_mode_scores.logsumexp(-1)*phrase_mass[:,None] if mode_nll.size(-1)>1 else
                    summed[:,0,None].expand(-1,conditional.size(-2)))
        old=local.new_zeros(group_count,local.size(-1)).index_add(0,index,local.flatten(0,1)*mass[:,None])
        chart_rows=chart_inverse[:,None].expand_as(groups)
        group_chart=torch.zeros(group_count,device=index.device,dtype=torch.long).index_add(0,index,(chart_rows*first).flatten())
        adjustment=(phrase_nll-old).masked_fill(~active[:,None],0.)
        chart_nll=chart_nll.index_add(0,group_chart,adjustment)
        if raw_joint_diagnostics:
            with torch.no_grad():
                raw_phrase=-torch.logsumexp(pp.detach()-summed.detach()[:,None,:],-1)
                raw_adjustment=(raw_phrase-old.detach()).masked_fill(~active[:,None],0.)
                raw_chart_nll=raw_chart_nll.index_add(0,group_chart,raw_adjustment)
        complete_groups=active.sum().float()
    profile_mass=profile_chart_weights.to(chart_nll)*float(profile_weight)*float(normalizer)
    profile_nll=component_profile_nll.float()
    prior=prior_logits.float().log_softmax(-1)
    energy=chart_nll+profile_mass[:,None]*profile_nll
    chart_mass=profile_chart_weights.to(prior)
    if chart_mass.shape!=(count,):raise ValueError('Unified chart weights must match unique charts')
    valid_mass=torch.isfinite(chart_mass)&chart_mass.ge(0)
    if chart_mass.device.type=='cuda':torch._assert_async(valid_mass.all(),'Invalid unified chart weight')
    elif not bool(valid_mass.all()):raise ValueError('Invalid unified chart weight')
    positive=chart_mass.gt(0)
    # L_c = E_r[energy_c]/D + w_c KL(r||q_c). Its exact minimizing
    # posterior uses energy/(D*w), AFTER the balanced persistent-m objective.
    # The inner m and outer z therefore obey their actual variational budgets.
    budget=(chart_mass*float(normalizer)).masked_fill(~positive,1.)
    scores=prior-energy/budget[:,None]
    minimum=energy.amin(-1)
    # At w=0 the KL vanishes, so the exact limit is any distribution on the
    # minimum-energy components. Prior-relative tie breaking also gives q when
    # there is no evidence. Event evidence is retained rather than dropped.
    zero_mass_scores=prior.masked_fill(energy>minimum[:,None],-torch.inf)
    scores=torch.where(positive[:,None],scores,zero_mass_scores)
    posterior=scores.log_softmax(-1).detach()
    responsibility=posterior.exp()
    safe_posterior=posterior.masked_fill(responsibility.eq(0),0.)
    data_term=(responsibility*energy).sum()/normalizer
    prior_ce=(-(responsibility*prior).sum(-1)*chart_mass).sum()
    prior_kl=((responsibility*(safe_posterior-prior)).sum(-1)*chart_mass).sum()
    loss=data_term+prior_kl
    # Independent closed-form minimum of the same variational objective.
    optimum=(-scores.logsumexp(-1)*chart_mass).masked_fill(~positive,0.)
    optimum=optimum+minimum.masked_fill(positive,0.)/normalizer
    chart_event_mass=weights.new_zeros(count).index_add(0,chart_inverse,weights.sum(1))
    metrics=dict(unified_complete_phrase_groups=complete_groups,
        unified_chart_count=chart_nll.new_tensor(float(count)),
        unified_balanced_variational_objective=optimum.sum().detach(),
        unified_zero_prior_mass_charts=(~positive).sum().float(),
        unified_zero_prior_mass_event_charts=((~positive)&chart_event_mass.gt(0)).sum().float(),
        unified_data_term=data_term,
        unified_prior_kl=prior_kl,unified_prior_ce=prior_ce,
        unified_prior_weight_mass=chart_mass.sum(),
        unified_prior_entropy=(-(prior.exp()*prior).sum(-1)*chart_mass).sum()/chart_mass.sum().clamp_min(1e-8),
        unified_posterior_entropy=(-(responsibility*safe_posterior).sum(-1)).mean(),
        unified_profile_nll=(responsibility*profile_nll).sum(-1).mul(chart_mass).sum(),
        unified_configuration_nll=(responsibility*chart_nll).sum()/normalizer)
    with torch.no_grad():
        metrics.update(unified_phrase_mode_prior_kl=zero.detach(),
            unified_phrase_mode_posterior_entropy=zero.detach(),
            unified_phrase_mode_prior_entropy=zero.detach(),unified_phrase_mode_weight_mass=zero.detach())
        if phrase_mode_scores is not None:
            mode_logp=phrase_mode_scores.detach().log_softmax(-1)
            mode_p=mode_logp.exp()
            mode_prior=pp.detach()
            group_z=responsibility[group_chart]
            group_mass=phrase_mass.masked_fill(~active,0.)
            probability_mass=group_mass[:,None]*group_z
            kl=(mode_p*(mode_logp-mode_prior)).sum(-1)
            entropy=-(mode_p*mode_logp).sum(-1)
            prior_entropy=-(mode_prior.exp()*mode_prior).sum(-1)
            mass_total=group_mass.sum().clamp_min(1e-8)
            metrics.update(unified_phrase_mode_prior_kl=(kl*probability_mass).sum()/normalizer,
                unified_phrase_mode_posterior_entropy=(entropy*probability_mass).sum()/mass_total,
                unified_phrase_mode_prior_entropy=(prior_entropy*probability_mass).sum()/mass_total,
                unified_phrase_mode_weight_mass=group_mass.sum()/normalizer)
        if raw_joint_diagnostics:
            raw_energy=raw_chart_nll+profile_mass.detach()[:,None]*profile_nll.detach()
            metrics['unified_original_sequence_joint_nll']=-(prior.detach()-raw_energy).logsumexp(-1).sum()/normalizer
    return loss,metrics
