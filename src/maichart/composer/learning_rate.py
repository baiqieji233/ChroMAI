"""Stateless, checkpointed learning-rate plans indexed by successful updates."""
from __future__ import annotations
import math

SCHEMA='composer_lr_schedule_v1'
KINDS={'constant','cosine'}


def validate_plan(plan):
    if not isinstance(plan,dict) or plan.get('schema')!=SCHEMA or plan.get('kind') not in KINDS:
        raise ValueError('Unsupported Composer learning-rate plan')
    for key in ('warmup_steps','total_steps'):
        if type(plan.get(key)) is not int or plan[key]<(1 if key=='total_steps' else 0):
            raise ValueError(f'Invalid learning-rate {key}')
    if plan['warmup_steps']>=plan['total_steps']:
        raise ValueError('warmup_steps must be smaller than total optimizer updates')
    if not math.isfinite(float(plan.get('base_lr',0))) or plan['base_lr']<=0:
        raise ValueError('Learning-rate base must be positive and finite')
    ratio=float(plan.get('min_lr_ratio',float('nan')))
    if not math.isfinite(ratio) or not 0<=ratio<=1:
        raise ValueError('min_lr_ratio must be in [0,1]')
    return plan


def resolve_plan(*,requested='auto',warmup_steps=None,min_lr_ratio=None,base_lr,total_steps,
                 resume=False,saved=None):
    if requested not in {'auto',*KINDS}: raise ValueError('Unknown learning-rate schedule')
    if saved is not None: validate_plan(saved)
    inherited=saved if resume else None
    kind=(inherited['kind'] if inherited else 'constant') if requested=='auto' else requested
    warmup=(inherited['warmup_steps'] if inherited else min(500,total_steps//20) if kind=='cosine' else 0) if warmup_steps is None else warmup_steps
    ratio=(inherited['min_lr_ratio'] if inherited else .1 if kind=='cosine' else 1.) if min_lr_ratio is None else min_lr_ratio
    if resume:
        previous=inherited or dict(schema=SCHEMA,kind='constant',warmup_steps=0,min_lr_ratio=1.,
                                   base_lr=base_lr,total_steps=total_steps)
        if any(left!=right for left,right in ((kind,previous['kind']),(warmup,previous['warmup_steps']),(ratio,previous['min_lr_ratio']))):
            raise ValueError('Resume inherits its learning-rate schedule; use finetune with a new directory')
        if previous['kind']!='constant' or previous['warmup_steps']:
            if base_lr!=previous['base_lr']:
                raise ValueError('Resume inherits scheduled base_lr; use finetune with a new directory')
        total_steps=previous['total_steps']
    return validate_plan(dict(schema=SCHEMA,kind=kind,base_lr=float(base_lr),
        warmup_steps=warmup,total_steps=total_steps,min_lr_ratio=float(ratio)))


def learning_rate(plan,completed_updates):
    """Rate for the next update; failed/nonfinite updates do not advance time."""
    if completed_updates<0: raise ValueError('completed_updates must be nonnegative')
    warmup=plan['warmup_steps'];base=plan['base_lr']
    if completed_updates<warmup: return base*(completed_updates+1)/warmup
    if plan['kind']=='constant': return base
    span=plan['total_steps']-warmup-1
    progress=(min(1.,max(0.,(completed_updates-warmup)/span)) if span>0
              else 1. if warmup or completed_updates else 0.)
    return base*(plan['min_lr_ratio']+(1.-plan['min_lr_ratio'])*.5*(1.+math.cos(math.pi*progress)))
