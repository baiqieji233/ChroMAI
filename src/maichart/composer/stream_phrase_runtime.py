"""Prefix-owned soft execution choices, never shared across speculative rollouts."""
import math
from collections import Counter

import torch


def execution_options(model, context, phrase_output, batch, action, generator, *, temperature=1., beat=None):
    if getattr(getattr(model,'config',None),'phrase_execution_policy','off')!='persistent_modes_v1':return {}
    temperature=float(temperature)
    if not math.isfinite(temperature) or temperature<0:raise ValueError('invalid phrase mode temperature')
    # The mode and the symbolic history share the known departure phrase.
    # A proposed landing in the next phrase must not choose a new mode early.
    start=float(batch['phrase_start_beat'][0,-1])
    prefix=context._committed
    signature=(len(prefix),id(prefix[-1]) if prefix else None)
    state=getattr(context,'_execution_runtime',None)
    if state is None or state['signature']!=signature:
        state=dict(signature=signature,pending={},at_beat={})
        context._execution_runtime=state
    if start not in state['pending']:
        mode=next((s.phrase_mode for s in reversed(prefix)
                   if getattr(s,'phrase_mode_start',None)==start and getattr(s,'phrase_mode',None) is not None),None)
        if mode is None:
            scores=model.execution_prior(phrase_output,batch)[0,-1]
            if not bool(torch.isfinite(scores).any()):raise ValueError('No finite phrase execution mode')
            mode=(int(scores.argmax()) if temperature==0 else
                  int(torch.multinomial((scores.float()/temperature).softmax(-1),1,generator=generator)))
        if not 0<=int(mode)<int(model.config.phrase_modes):raise ValueError('Invalid committed phrase mode')
        state['pending'][start]=int(mode)
    mode=state['pending'][start]
    if beat is None:beat=float(batch.get('candidate_step_beat',batch.get('target_beat',batch['step_beat']))[0,-1])
    state['at_beat'][float(beat)]=(mode,start)
    return model.execution_proposal(action,phrase_output,batch,mode)


def execution_metadata(context, beat):
    state=getattr(context,'_execution_runtime',None)
    if state is None:return {}
    value=state['at_beat'].get(float(beat))
    return {} if value is None else dict(phrase_mode=value[0],phrase_mode_start=value[1])


def execution_report(steps):
    choices={}
    notes=Counter()
    for step in steps:
        mode=getattr(step,'phrase_mode',None)
        start=getattr(step,'phrase_mode_start',None)
        if mode is not None and start is not None:
            choices.setdefault(start,int(mode));notes[int(mode)]+=1
    return dict(phrase_choices=len(choices),mode_phrase_counts=dict(Counter(choices.values())),
                mode_note_counts=dict(notes),choices=[dict(start_beat=start,mode=mode) for start,mode in sorted(choices.items())])
