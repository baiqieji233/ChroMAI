"""Masked mixture distribution for one immutable, whole-song control profile."""
import itertools
import json
import numpy as np
import torch
from torch import nn
from maichart.generation.conditions import CONDITION_KEYS, GenerationCondition


def pool_audio(cache):
    valid=cache['global_mask'].to(cache['global'].dtype)
    return (cache['global']*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[...,None]


def feasible_profile(values, known):
    """Project five mutually exclusive note types subject to exact marginal locks."""
    x=np.asarray(values,dtype=np.float64).copy();known=np.asarray(known,dtype=bool)
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):raise ValueError('Song controls must be finite in [0,1]')
    if known[7] and known[12] and abs(x[7]-x[12])>1e-7:raise ValueError('Contradictory chord and chord_bias locks')
    chord=x[7] if known[7] else x[12] if known[12] else (x[7]+x[12])/2
    x[7]=x[12]=chord
    # tap, slide, hold, touch, touch_hold; hold/touch marginals overlap.
    rows=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,1],[0,0,0,1,1]],dtype=float)
    idx=[i for i in range(4) if known[8+i]]
    a=np.concatenate([np.ones((1,5)),rows[idx]],axis=0)
    b=np.array([1.]+[x[8+i] for i in idx])
    target=np.linalg.lstsq(rows,x[8:12],rcond=None)[0]
    lower=np.zeros(5)
    lower[1]=max([x[i] for i in (4,5) if known[i]]+[0.])
    if known[3] and x[3]>0:lower[1]=max(lower[1],1e-6)
    b=b-a@lower;target=target-lower
    best=None
    for flags in itertools.product((False,True),repeat=5):
        free=np.flatnonzero(flags)
        if not len(free):continue
        af=a[:,free];v=target[free]+af.T@np.linalg.lstsq(af@af.T,b-af@target[free],rcond=None)[0]
        if np.min(v)<-1e-7 or np.max(np.abs(af@v-b))>1e-7:continue
        p=np.zeros(5);p[free]=np.maximum(v,0);score=np.square(p-target).sum()
        if best is None or score<best[0]:best=(score,p)
    if best is None:raise ValueError('Contradictory note-type locks: tap/slide/hold/touch must admit five-class probabilities')
    x[8:12]=rows@(best[1]+lower)
    for i in (4,5):
        if known[i] and x[i]>x[9]+1e-7:raise ValueError('Slide radar lock exceeds slide_bias')
        if not known[i]:x[i]=min(x[i],x[9])
    if x[9]<=1e-12:
        if known[3] and x[3]>0:raise ValueError('Positive sweep requires a positive slide_bias')
        x[3]=0.
    return x


class SongProfileCompleter(nn.Module):
    """Categorical mixture with masked joint score and feasible projected draws.

    The score is the normalized pre-projection categorical likelihood, not an
    analytic likelihood for the many-to-one feasibility projection.
    """
    bins=129
    def __init__(self,d_model,components=4):
        super().__init__();self.components=int(components)
        self.network=nn.Sequential(nn.Linear(d_model+32,d_model),nn.GELU(),nn.Linear(d_model,self.components*(1+15*self.bins)))
    @staticmethod
    def bind_known(condition,known_mask):
        condition=condition.clone();known_mask=known_mask.bool().clone();known_mask[:,0]=True
        both=known_mask[:,7]&known_mask[:,12]
        contradiction=(both & ((condition[:,7]-condition[:,12]).abs()>1e-6)).any()
        if condition.device.type=='cuda':
            torch._assert_async(~contradiction,'Contradictory chord and chord_bias locks')
        elif bool(contradiction):raise ValueError('Contradictory chord and chord_bias locks')
        chord=torch.where(known_mask[:,12]&~known_mask[:,7],condition[:,12],condition[:,7])
        tied=known_mask[:,7]|known_mask[:,12]
        condition[:,7]=condition[:,12]=chord
        known_mask[:,7]=known_mask[:,12]=tied
        return condition,known_mask
    def forward(self,audio,condition,known_mask):
        condition,known_mask=self.bind_known(condition,known_mask)
        if isinstance(audio,dict):audio=pool_audio(audio)
        mask=known_mask.to(condition.dtype)
        out=self.network(torch.cat((audio,condition*mask,mask),-1)).float().reshape(-1,self.components,1+15*self.bins)
        return dict(mixture_logits=out[:,:,0],bin_logits=out[:,:,1:].reshape(-1,self.components,15,self.bins))
    def _evidence_scores(self,out,condition,known_mask):
        targets=(condition[:,1:].clamp(0,1)*(self.bins-1)).round().long()
        log_probs=out['bin_logits'].log_softmax(-1)
        logp=log_probs.gather(-1,targets[:,None,:,None].expand(-1,self.components,-1,1)).squeeze(-1)
        unknown=(~known_mask[:,1:].bool()).to(logp.dtype)
        unknown[:,11]=0  # The tied chord statistic contributes once.
        normalizer=unknown.sum(-1).clamp_min(1)
        mixture=out['mixture_logits'].log_softmax(-1)
        joint=(logp*unknown[:,None]).sum(-1)+mixture
        joint_nll=-joint.logsumexp(-1)/normalizer
        marginal=(log_probs.exp()*mixture.exp()[:,:,None,None]).sum(1)
        grid=torch.linspace(0,1,self.bins,device=condition.device,dtype=marginal.dtype)
        observed=(grid[None,None,:]>=condition[:,1:,None]).to(marginal.dtype)
        # The terminal CDF is identically one. Uniform-grid integration gives
        # a numerical proper score that distinguishes nearby from distant bins.
        crps_fields=(marginal.cumsum(-1)[...,:-1]-observed[...,:-1]).square().sum(-1)/(self.bins-1)
        crps=(crps_fields*unknown).sum(-1)/normalizer
        return dict(joint_nll=joint_nll,crps=crps,component_profile_nll=-(logp*unknown[:,None]).sum(-1)/normalizer[:,None])
    def loss_parts(self,audio,condition,known_mask=None):
        """Proper joint categorical score plus ordered marginal CRPS per chart."""
        if known_mask is None:
            known_mask=torch.rand_like(condition)<torch.rand((len(condition),1),device=condition.device)
            known_mask[:,0]=True
        condition,known_mask=self.bind_known(condition,known_mask)
        scores=self._evidence_scores(self(audio,condition,known_mask),condition,known_mask)
        return dict(joint_nll=scores['joint_nll'],crps=scores['crps'])
    def unified_evidence(self,audio,condition,known_mask=None,*,only_level_probability=.5):
        """Causal shared prior plus training-only profile evidence for one z.

        Full target controls enter the score, never the deployed prior input.
        Exactly the same mask conditions both the profile and response reads.
        """
        probability=float(only_level_probability)
        if not 0<=probability<=1:raise ValueError('Invalid only-level mask probability')
        if known_mask is None:known_mask=self.sample_known_mask(condition,probability)
        condition,known_mask=self.bind_known(condition,known_mask)
        output=self(audio,condition,known_mask)
        scores=self._evidence_scores(output,condition,known_mask)
        return dict(prior_logits=output['mixture_logits'],bin_logits=output['bin_logits'],
            component_profile_nll=scores['component_profile_nll'],crps=scores['crps'],
            known_mask=known_mask,causal_condition=condition*known_mask.to(condition.dtype))
    @staticmethod
    def sample_known_mask(condition,only_level_probability=.5):
        known=torch.rand_like(condition)<torch.rand((len(condition),1),device=condition.device)
        known=known & ~(torch.rand((len(condition),1),device=condition.device)<float(only_level_probability))
        known[:,0]=True
        return known
    def loss(self,audio,condition,known_mask=None,reduction='mean',crps_weight=0.):
        weight=float(crps_weight)
        if not np.isfinite(weight) or weight<0:raise ValueError('Invalid profile CRPS weight')
        parts=self.loss_parts(audio,condition,known_mask)
        loss=parts['joint_nll']+weight*parts['crps']
        return loss if reduction=='none' else loss.mean()
    @torch.no_grad()
    def sample(self,audio,condition,known_mask,generator=None,*,policy='component_mean_v2',return_diagnostics=False,component_choice=None):
        if policy not in ('component_mean_v2','categorical_v1'):raise ValueError('Invalid profile sampling policy')
        condition,known_mask=self.bind_known(condition,known_mask)
        out=self(audio,condition,known_mask)
        mixture=out['mixture_logits'].softmax(-1)
        if component_choice is None:modes=torch.multinomial(mixture,1,generator=generator).squeeze(-1)
        else:
            modes=torch.as_tensor(component_choice,device=condition.device,dtype=torch.long).expand(len(condition))
            if bool(((modes<0)|(modes>=self.components)).any()):raise ValueError('Invalid unified profile component')
        logits=out['bin_logits'][torch.arange(len(condition),device=condition.device),modes]
        probabilities=logits.softmax(-1)
        grid=torch.linspace(0,1,self.bins,device=condition.device,dtype=probabilities.dtype)
        means=(probabilities*grid).sum(-1)
        variance=(probabilities*(grid-means[...,None]).square()).sum(-1)
        draws=(means if policy=='component_mean_v2' else
            torch.multinomial(probabilities.reshape(-1,self.bins),1,generator=generator).reshape(-1,15)/(self.bins-1))
        values=condition.clone();values[:,1:]=torch.where(known_mask[:,1:].bool(),condition[:,1:],draws)
        values[:,12]=values[:,7]
        projected=[feasible_profile(v,m) for v,m in zip(values.cpu().numpy(),known_mask.cpu().numpy())]
        result=torch.as_tensor(np.stack(projected),device=condition.device,dtype=condition.dtype)
        if not return_diagnostics:return result
        rows=[]
        cdf=probabilities.cumsum(-1)
        for row in range(len(condition)):
            fields={}
            for index,key in enumerate(CONDITION_KEYS[1:]):
                # chord_bias reports its canonical tied chord distribution.
                j=6 if index==11 else index
                p=probabilities[row,j]
                quantiles={str(q):float(grid[torch.searchsorted(cdf[row,j].contiguous(),p.new_tensor(q)).clamp_max(self.bins-1)]) for q in (.05,.5,.95)}
                fields[key]=dict(mean=float(means[row,j]),std=float(variance[row,j].sqrt()),
                    zero_probability=float(p[0]),one_probability=float(p[-1]),
                    probability_ge_0_8=float(p[grid>=.8].sum()),quantiles=quantiles,
                    locked=bool(known_mask[row,index+1]))
            rows.append(dict(selected_component=int(modes[row]),mixture_probabilities=mixture[row].cpu().tolist(),controls=fields))
        return result,dict(policy=policy,rows=rows)


def resolve_song_profile(model,cache,condition,args,payload):
    source=getattr(args,'song_style_source','auto')
    if source not in ('auto','checkpoint','manual'):raise ValueError('Invalid song_style_source')
    raw=getattr(args,'song_style_locks',None)
    locks=json.loads(raw) if isinstance(raw,str) and raw.strip() else (raw or {})
    if not isinstance(locks,dict):raise ValueError('song_style_locks must be a JSON object')
    vector=condition.to_vector(device=cache['global'].device)[None]
    known=torch.zeros_like(vector,dtype=torch.bool);known[:,0]=True
    for key,value in locks.items():
        if key not in CONDITION_KEYS or key=='level':raise ValueError(f'Invalid style lock: {key}')
        value=float(value)
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError(f'Invalid style lock value: {key}')
        index=CONDITION_KEYS.index(key);vector[0,index]=value;known[0,index]=True
    explicit_file=bool(getattr(args,'condition_file',None) and not getattr(args,'condition_file_defaults',False))
    # Legacy manual/file values are preferences; only explicit JSON locks assert
    # algebraic equalities. Preserve their original values when auto is disabled.
    available=getattr(model,'song_profile',None) is not None and getattr(model.config,'song_profile_policy','off')!='off'
    seed=getattr(args,'song_style_seed',None)
    seed=int(args.seed if seed is None else seed)
    diagnostics=None
    unified=getattr(model.config,'song_response_policy','off')=='unified_profile_response_v2'
    if unified and not -1<=int(getattr(args,'song_response_mode',-1))<model.config.song_response_modes:
        raise ValueError('Invalid unified whole-song choice')
    source_vector,source_known=model.song_profile.bind_known(vector,known) if unified else (vector,known)
    if source=='auto' and available and not explicit_file:
        generator=torch.Generator(device=vector.device).manual_seed(seed)
        override=int(getattr(args,'song_response_mode',-1)) if unified else -1
        vector,diagnostics=model.song_profile.sample(cache,vector,known,generator=generator,
            policy='component_mean_v2',return_diagnostics=True,
            component_choice=override if override>=0 else None);effective='learned_masked_joint_v1'
    else:
        if locks:feasible_profile(vector[0].cpu().numpy(),known[0].cpu().numpy())
        effective='condition_file' if explicit_file else 'manual' if source=='manual' else 'checkpoint_defaults'
    values=dict(zip(CONDITION_KEYS,vector[0].cpu().tolist()));values['level']*=15
    result=GenerationCondition(level=values['level'],difficulty=condition.difficulty,
        radar={k:values[k] for k in CONDITION_KEYS[1:8]},style={k:values[k] for k in CONDITION_KEYS[8:]})
    if effective!='learned_masked_joint_v1':
        original=condition.to_dict()
        for key,value in locks.items():
            original['radar' if key in CONDITION_KEYS[1:8] else 'style'][key]=float(value)
        if 'chord' in locks:original['style']['chord_bias']=float(locks['chord'])
        elif 'chord_bias' in locks:original['radar']['chord']=float(locks['chord_bias'])
        result=GenerationCondition.from_dict(original)
    if unified:
        if diagnostics is not None:
            row=diagnostics['rows'][0]
            prior=torch.as_tensor([row['mixture_probabilities']],device=vector.device,dtype=vector.dtype).clamp_min(1e-30).log()
            choice=int(row['selected_component'])
        else:
            source_vector=result.to_vector(device=vector.device)[None]
            source_known=torch.ones_like(source_vector,dtype=torch.bool)
            evidence=model.song_profile.unified_evidence(cache,source_vector,source_known)
            prior=evidence['prior_logits']
            override=int(getattr(args,'song_response_mode',-1))
            if override>=prior.size(-1):raise ValueError('Invalid unified whole-song choice')
            generator=torch.Generator(device=vector.device).manual_seed(seed)
            choice=override if override>=0 else int(torch.multinomial(prior.softmax(-1),1,generator=generator)[0,0])
        cache['unified_style']=dict(prior_logits=prior,causal_condition=source_vector*source_known,
                                   known_mask=source_known,choice=choice)
    return result,dict(requested_source=source,effective_source=effective,seed=seed,locks=locks,
        effective_controls=result.to_dict(),frozen_for_song=True,
        distribution=('mixture_component_mean_with_feasibility_projection_v2' if effective=='learned_masked_joint_v1' else 'legacy_preferences'),
        sampling_policy=('component_mean_v2' if diagnostics else None),prediction_diagnostics=diagnostics,
        unified_choice=(choice if unified else None),
        training_score=('unified_configuration_profile_generalized_em_plus_crps' if unified else
            'masked_joint_categorical_nll_plus_ordered_crps' if getattr(model.config,'song_profile_policy','off')=='masked_ordered_joint_v2' else 'masked_preprojection_categorical_nll'))
