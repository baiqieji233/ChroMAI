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
    def loss(self,audio,condition,known_mask=None, reduction="mean"):
        if known_mask is None:
            known_mask=torch.rand_like(condition)<torch.rand((len(condition),1),device=condition.device)
            known_mask[:,0]=True
        condition,known_mask=self.bind_known(condition,known_mask)
        out=self(audio,condition,known_mask)
        targets=(condition[:,1:].clamp(0,1)*(self.bins-1)).round().long()
        logp=out['bin_logits'].log_softmax(-1).gather(-1,targets[:,None,:,None].expand(-1,self.components,-1,1)).squeeze(-1)
        unknown=(~known_mask[:,1:].bool()).to(logp.dtype)
        unknown[:,11]=0  # chord_bias duplicates chord; score the physical statistic once
        joint=(logp*unknown[:,None]).sum(-1)+out['mixture_logits'].log_softmax(-1)
        loss=-joint.logsumexp(-1)/unknown.sum(-1).clamp_min(1)
        return loss if reduction=='none' else loss.mean()
    @torch.no_grad()
    def sample(self,audio,condition,known_mask,generator=None):
        condition,known_mask=self.bind_known(condition,known_mask)
        out=self(audio,condition,known_mask)
        modes=torch.multinomial(out['mixture_logits'].softmax(-1),1,generator=generator).squeeze(-1)
        logits=out['bin_logits'][torch.arange(len(condition),device=condition.device),modes]
        draws=torch.multinomial(logits.softmax(-1).reshape(-1,self.bins),1,generator=generator).reshape(-1,15)/(self.bins-1)
        values=condition.clone();values[:,1:]=torch.where(known_mask[:,1:].bool(),condition[:,1:],draws)
        values[:,12]=values[:,7]
        projected=[feasible_profile(v,m) for v,m in zip(values.cpu().numpy(),known_mask.cpu().numpy())]
        return torch.as_tensor(np.stack(projected),device=condition.device,dtype=condition.dtype)


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
    if source=='auto' and available and not explicit_file:
        generator=torch.Generator(device=vector.device).manual_seed(seed)
        vector=model.song_profile.sample(cache,vector,known,generator=generator);effective='learned_masked_joint_v1'
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
    return result,dict(requested_source=source,effective_source=effective,seed=seed,locks=locks,
        effective_controls=result.to_dict(),frozen_for_song=True,
        distribution=('129_bin_mixture_with_feasibility_projection' if effective=='learned_masked_joint_v1' else 'legacy_preferences'),
        training_score='masked_preprojection_categorical_nll')
