"""Unweighted echo calibration, using the exact model-side evidence function."""
from __future__ import annotations

import torch

from .phase_echo import phase_echo_features, ECHO_LAGS, ECHO_PERMUTATIONS


def strongest_echo(features, target):
    row=torch.arange(len(target),device=target.device)
    sources=features['candidate_source'][row,target]
    present=features['has_candidate_source'][row,target]
    scores=features['evidence'].masked_fill(~present[:,:,None],-1).flatten(1)
    evidence,best=scores.max(-1)
    source=sources[row,best.div(4,rounding_mode='floor')]
    perm=torch.tensor(ECHO_PERMUTATIONS,device=target.device)[best.remainder(4)]
    return evidence,source.gather(1,perm)


class EchoMetrics:
    def __init__(self):
        self.copy = torch.zeros(5, 4, dtype=torch.float64)
        self.timing = torch.zeros(5, 3, dtype=torch.float64)

    @torch.no_grad()
    def update(self, model, batch, output):
        if not hasattr(model, 'action_factors') or 'history_beat_ages' not in batch:
            return
        heads = model.action_factors.atom_head_lanes.bool()
        history = batch['history']
        lanes = heads[history].any(-2) if history.ndim == 3 else heads[history]
        valid = ~batch['history_padding_mask']
        gaps = batch['candidate_features'][...,model.config.audio_dim]
        f = phase_echo_features(lanes, batch['history_beat_ages'], valid, gaps)
        logits = output['action_logits'].float()
        if logits.ndim == 3:
            first = logits[:,0]
        else:
            indices = output['action_slot_indices']
            selected = indices[:,1] == 0
            first = logits.new_empty(history.size(0), logits.size(-1))
            first[indices[selected,0]] = logits[selected]
        row = torch.arange(history.size(0),device=history.device)
        target = batch['candidate_target']
        best_e,target_set=strongest_echo(f,target)
        bits = 1 << torch.arange(8,device=history.device)
        atom_bits = (heads.long()*bits).sum(-1)
        source_bits = (target_set.long()*bits).sum(-1)
        allowed = atom_bits.ne(0)[None] & (atom_bits[None].bitwise_and(~source_bits[:,None]) == 0)
        mass = (first.softmax(-1)*allowed).sum(-1)
        authored = heads[batch['action_target']].any(-2)
        full_hit = (authored == target_set).all(-1)
        first_bits = atom_bits[batch['action_target'][:,0]]
        first_hit = first_bits.ne(0) & (first_bits.bitwise_and(~source_bits) == 0)
        use = (best_e>=0) & (f['button_count']>=4) & authored.any(-1)
        for key in ('sample_weight','action_weight','transition_weight'):
            if key in batch:use &= batch[key]>0
        rows = torch.stack((torch.ones_like(mass),mass,full_hit.float(),first_hit.float()),-1).double()
        totals = torch.zeros(5,4,dtype=torch.float64,device=history.device)
        totals.index_add_(0,best_e.clamp_min(0),rows*use[:,None])
        self.copy = self.copy.to(totals.device) + totals
        pointer = output['candidate_logits'].float().softmax(-1)
        prediction = f['rhythm_prediction'] & ~batch['candidate_padding_mask'][...,None]
        timing_mass = (pointer[...,None]*prediction).sum(1)
        timing_hit = f['rhythm_prediction'][row,target]
        # Each lag is a separate rhythm prediction, matching echo_evidence.py.
        use_r = f['has_rhythm_source'] & (f['event_count']>=4)[:,None]
        for key in ('sample_weight','pointer_weight','transition_weight'):
            if key in batch:use_r &= (batch[key]>0)[:,None]
        rows_r = torch.stack((torch.ones_like(timing_mass),timing_mass,timing_hit.float()),-1).double()
        totals_r = torch.zeros(5,3,dtype=torch.float64,device=history.device)
        totals_r.index_add_(0,f['rhythm_evidence'].flatten(),(rows_r*use_r[...,None]).reshape(-1,3))
        self.timing = self.timing.to(totals_r.device) + totals_r

    def report(self, model=None):
        def rows(values, names):
            return {str(e):dict(count=int(row[0]),**{name:float(row[i+1]/row[0]) if row[0] else None
                          for i,name in enumerate(names)}) for e,row in enumerate(values)}
        result=dict(echo_copy_mass_by_E=rows(self.copy.cpu(),('model_mass','authored_copy_rate','authored_first_slot_subset_rate')),
                    echo_timing_mass_by_R=rows(self.timing.cpu(),('model_mass','authored_copy_rate')),
                    echo_definition='phase_echo_v1; raw T=1 first-slot probabilities; exact full-event authored copy also reported; rhythm counts each available lag',
                    echo_lags=list(ECHO_LAGS))
        prior=getattr(model,'echo_prior',None)
        if prior is not None:
            from .phase_echo import ECHO_PARAM_SCALE
            result['echo_prior_calibration']={
                'table_E_scaled':(prior.table_E.detach()*ECHO_PARAM_SCALE).cpu().tolist(),
                'rhythm_table_scaled':(prior.rtable.detach()*ECHO_PARAM_SCALE).cpu().tolist(),
                'lane_audio_weight_norm':float(prior.ctx.weight.detach().float().norm()),
                'timing_audio_weight_norm':float(prior.rctx.weight.detach().float().norm())}
        return result


def echo_evidence_profile(points):
    """Full-prefix generated-chart calibration, matching echo_evidence.py.

    Collect only the events needed for the four-event evidence and each exact
    source/next-onset query. This retains full-prefix semantics without an
    unbounded full-song quadratic tensor. The shared feature function remains
    fully batched; no model inference or audio/cache reads are performed.
    """
    from fractions import Fraction
    from .events import event_metadata
    merged={}
    for point in points:
        beat,text=(point.beat,point.note_text) if hasattr(point,'beat') else point
        beat=Fraction(beat)
        try:
            heads={int(x)-1 for x in event_metadata(str(text)).heads if len(x)==1 and x in '12345678'}
        except ValueError:
            # As in the surrounding structure report, an unsupported atom is
            # still an onset but supplies no button-layout evidence.
            heads=set()
        merged.setdefault(beat,set()).update(heads)
    beats=sorted(merged);positions={b:i for i,b in enumerate(beats)}
    source_lanes=torch.zeros(len(beats),8,dtype=torch.bool)
    for i,beat in enumerate(beats):
        for lane in merged[beat]:source_lanes[i,lane]=True
    recent_buttons=[];records=[]
    for n,beat in enumerate(beats):
        if n:
            recent=range(max(0,n-4),n)
            needed=set(recent)|set(recent_buttons[-4:])
            for k in ECHO_LAGS:
                for j in list(recent)+recent_buttons[-4:]:
                    i=positions.get(beats[j]-k)
                    if i is not None:needed.add(i)
                i=positions.get(beat-k)
                if i is not None and i<n:needed.add(i)
                anchor=positions.get(beats[n-1]-k)
                if anchor is not None:
                    needed.add(anchor)
                    if anchor+1<n:needed.add(anchor+1)
            records.append((n,sorted(needed),len(recent_buttons)>=4))
        if merged[beat]:recent_buttons.append(n)
    counts=torch.zeros(5,dtype=torch.long);hits=counts.clone()
    for start in range(0,len(records),64):
        rows=records[start:start+64];width=max(len(r[1]) for r in rows);b=len(rows)
        lanes=torch.zeros(b,width,8,dtype=torch.bool);ages=torch.zeros(b,width);valid=torch.zeros(b,width,dtype=torch.bool)
        gaps=torch.zeros(b,1);actual=torch.zeros(b,8,dtype=torch.bool);eligible=torch.zeros(b,dtype=torch.bool)
        for row,(n,indices,enough) in enumerate(rows):
            size=len(indices);lanes[row,:size]=source_lanes[indices];valid[row,:size]=True
            ages[row,:size]=torch.tensor([float(beats[n-1]-beats[i]) for i in indices])
            gaps[row,0]=float(beats[n]-beats[n-1]);actual[row]=source_lanes[n]
            eligible[row]=enough and bool(merged[beats[n]])
        f=phase_echo_features(lanes,ages,valid,gaps)
        e,target=strongest_echo(f,torch.zeros(b,dtype=torch.long))
        use=eligible & e.ge(0)
        counts.index_add_(0,e[use],torch.ones_like(e[use]))
        hits.index_add_(0,e[use],(target[use]==actual[use]).all(-1).long())
    total=int(counts.sum())
    return {'scope':'full committed prefix, strongest available lag then transform; no 4/4 assumption',
            'events':total,'by_E':{str(e):{'count':int(counts[e]),'share':float(counts[e]/total) if total else 0.,
                'copy_rate':float(hits[e]/counts[e]) if counts[e] else None} for e in range(5)}}
