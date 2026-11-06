"""Masked mixture distribution for one immutable, whole-song control profile."""
import itertools
import json
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _profile_projection_tables():
    """Compile all 16 lock patterns / 32 nonnegative active sets on CPU once."""
    marginals=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,1],[0,0,0,1,1]],dtype=np.float64)
    full=np.concatenate((np.ones((1,5)),marginals),axis=0)
    project=np.zeros((16,32,5,5),dtype=np.float64)
    rhs=np.zeros_like(project)
    patterns=list(itertools.product((False,True),repeat=4))
    for mask,pattern in enumerate(patterns):
        constraints=np.array([0]+[i+1 for i,locked in enumerate(pattern) if locked])
        matrix=full[constraints]
        for flags,free_pattern in enumerate(itertools.product((False,True),repeat=5)):
            free=np.flatnonzero(free_pattern)
            if not len(free):continue
            reduced=matrix[:,free]
            correction=reduced.T@np.linalg.pinv(reduced@reduced.T)
            project[mask,flags][np.ix_(free,free)]=np.eye(len(free))-correction@reduced
            rhs[mask,flags][np.ix_(free,constraints)]=correction
    return tuple(torch.from_numpy(array) for array in
        (project,rhs,full,np.linalg.pinv(marginals)))


def _profile_assert(valid,message):
    if valid.device.type=='cuda':torch._assert_async(valid.all(),message)
    elif not bool(valid.all()):raise ValueError(message)


def feasible_profile_tensor(values,known,*,tables=None):
    """Batched equivalent of feasible_profile, without host transfers/solves.

    The 5-class convex projection is small enough to evaluate every active
    set in parallel. FP64 here keeps the original 1e-7 feasibility tolerance;
    all trainable predictions are detached by the completion caller already.
    """
    if values.ndim!=2 or values.size(-1)!=16 or known.shape!=values.shape:
        raise ValueError('Profile projection requires [batch,16] values and locks')
    if tables is None:tables=tuple(table.to(device=values.device) for table in _profile_projection_tables())
    projection,rhs,full,target_map=tables
    x=values.double();known=known.bool()
    _profile_assert(torch.isfinite(x)&x.ge(0)&x.le(1),'Song controls must be finite in [0,1]')
    _profile_assert(~(known[:,7]&known[:,12])|(x[:,7]-x[:,12]).abs().le(1e-7),
        'Contradictory chord and chord_bias locks')
    chord=torch.where(known[:,7],x[:,7],torch.where(known[:,12],x[:,12],.5*(x[:,7]+x[:,12])))
    lower_slide=torch.maximum(torch.where(known[:,4],x[:,4],0.),torch.where(known[:,5],x[:,5],0.))
    lower_slide=torch.maximum(lower_slide,torch.where(known[:,3]&x[:,3].gt(0),1e-6,0.))
    lower=torch.zeros_like(x[:,:5]);lower[:,1]=lower_slide
    target=x[:,8:12]@target_map.T-lower
    b=torch.cat((torch.ones_like(x[:,:1]),x[:,8:12]),-1)-lower@full.T
    code=(known[:,8:12].long()*known.new_tensor([8,4,2,1],dtype=torch.long)).sum(-1)
    candidate=torch.einsum('bfij,bj->bfi',projection[code],target)+torch.einsum('bfij,bj->bfi',rhs[code],b)
    residual=candidate@full.T-b[:,None,:]
    required=torch.cat((torch.ones_like(known[:,:1]),known[:,8:12]),-1)
    valid=candidate.amin(-1).ge(-1e-7)&residual.masked_fill(~required[:,None,:],0.).abs().amax(-1).le(1e-7)
    valid[:,0]=False
    nonnegative=candidate.clamp_min(0)
    score=(nonnegative-target[:,None,:]).square().sum(-1).masked_fill(~valid,torch.inf)
    _profile_assert(valid.any(-1),'Contradictory note-type locks: tap/slide/hold/touch must admit five-class probabilities')
    selected=nonnegative.gather(1,score.argmin(-1)[:,None,None].expand(-1,1,5)).squeeze(1)+lower
    result=x.clone();result[:,8:12]=selected@full[1:].T
    result[:,7]=chord;result[:,12]=chord
    for index in (4,5):
        _profile_assert(~known[:,index]|x[:,index].le(result[:,9]+1e-7),'Slide radar lock exceeds slide_bias')
        result[:,index]=torch.where(known[:,index],x[:,index],torch.minimum(x[:,index],result[:,9]))
    no_slide=result[:,9].le(1e-12)
    _profile_assert(~(no_slide&known[:,3]&x[:,3].gt(0)),'Positive sweep requires a positive slide_bias')
    result[:,3]=torch.where(no_slide,torch.zeros_like(result[:,3]),result[:,3])
    # Marginal equalities above are solved in FP64; restore explicit inputs
    # so float32 slider values remain bitwise unchanged where possible.
    result=torch.where(known,x,result)
    result[:,7]=chord;result[:,12]=chord
    return result.to(values.dtype)


class SongProfileCompleter(nn.Module):
    """Categorical mixture with masked joint score and feasible projected draws.

    The score is the normalized pre-projection categorical likelihood, not an
    analytic likelihood for the many-to-one feasibility projection.
    """
    bins=129
    def __init__(self,d_model,components=4):
        super().__init__();self.components=int(components)
        self.network=nn.Sequential(nn.Linear(d_model+32,d_model),nn.GELU(),nn.Linear(d_model,self.components*(1+15*self.bins)))
        for name,table in zip(('_projection_matrix','_projection_rhs','_projection_constraints','_projection_target'),_profile_projection_tables()):
            self.register_buffer(name,table,persistent=False)
    def project_controls(self,values,known):
        tables=(self._projection_matrix,self._projection_rhs,self._projection_constraints,self._projection_target)
        # Explicit model.half()/bfloat16() also casts floating buffers. Keep
        # the small exact projection tables in FP64, restoring them once per
        # device from immutable CPU constants rather than quantized matrices.
        if tables[0].dtype!=torch.float64:
            cached=getattr(self,'_execution_projection_tables',None)
            if cached is None or cached[0].device!=values.device:
                cached=tuple(table.to(device=values.device) for table in _profile_projection_tables())
                self._execution_projection_tables=cached
            tables=cached
        return feasible_profile_tensor(values,known,tables=tables)
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
    def unified_evidence(self,audio,condition,known_mask=None,*,only_level_probability=.5,prior_condition=None):
        """Causal shared prior plus training-only profile evidence for one z.

        Full target controls enter the score, never the deployed prior input.
        Exactly the same mask conditions both the profile and response reads.
        """
        probability=float(only_level_probability)
        if not 0<=probability<=1:raise ValueError('Invalid only-level mask probability')
        if known_mask is None:known_mask=self.sample_known_mask(condition,probability)
        condition,known_mask=self.bind_known(condition,known_mask)
        prior_values=condition if prior_condition is None else self.bind_known(prior_condition,known_mask)[0]
        output=self(audio,prior_values,known_mask)
        scores=self._evidence_scores(output,condition,known_mask)
        return dict(prior_logits=output['mixture_logits'],bin_logits=output['bin_logits'],
            component_profile_nll=scores['component_profile_nll'],crps=scores['crps'],
            joint_profile_nll=scores['joint_nll'],
            known_mask=known_mask,causal_condition=prior_values*known_mask.to(prior_values.dtype))

    def execution_controls(self,evidence,condition,choice):
        """Match the deployed component-mean completion; explicit locks win."""
        bins=evidence['bin_logits']
        rows=torch.arange(bins.size(0),device=bins.device)
        grid=torch.linspace(0,1,self.bins,device=bins.device)
        mean=(bins[rows,choice].softmax(-1)*grid).sum(-1).detach()
        prediction=torch.cat((condition[:,:1],mean),-1)
        completed=torch.where(evidence['known_mask'],condition,prediction)
        # Chord radar and style are one statistic, never two conflicting locks.
        mask=evidence['known_mask'];value=torch.where(mask[:,7],condition[:,7],
            torch.where(mask[:,12],condition[:,12],completed[:,7]))
        completed=completed.clone();completed[:,7]=completed[:,12]=value
        return self.project_controls(completed,evidence['known_mask'])
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
        result=self.project_controls(values,known_mask)
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
def sampled_event_choice_credit(prior_logits, selected, energy, mass, chart,
        normalizer, *, committed=None, baseline=0.):
    """Unbiased score-function credit for the *executed* song choice.

    Per-chart leave-one-out costs provide a baseline independent of that
    chart's sampled choice. A previous-batch baseline handles one-chart rows.
    The zero-valued surrogate adds the missing selection gradient, without
    counting the actor's event likelihood twice or fabricating another pass.
    """
    import torch
    count=prior_logits.size(0)
    costs=prior_logits.new_zeros(count,dtype=torch.float32).index_add(0,chart.long(),energy.float())
    budgets=costs.new_zeros(count).index_add(0,chart.long(),mass.float())
    active=budgets.gt(0)
    if committed is not None:active=active & committed.lt(0)
    unit=costs/budgets.clamp_min(1e-8)
    number=active.sum()
    others=(unit*active).sum()-unit*active
    reference=torch.where(number>1,others/(number-1).clamp_min(1),
        torch.as_tensor(baseline,device=costs.device,dtype=costs.dtype).detach())
    advantage=((costs-reference*budgets)*active).detach()
    log_probability=prior_logits.float().log_softmax(-1).gather(-1,selected[:,None]).squeeze(-1)
    surrogate=((log_probability-log_probability.detach())*advantage).sum()/float(normalizer)
    return surrogate,dict(song_choice_credit_charts=number.float(),
        song_choice_credit_abs_advantage=advantage.abs().sum()/float(normalizer),
        song_choice_credit_unit_cost=(unit*active).sum()/number.clamp_min(1))
