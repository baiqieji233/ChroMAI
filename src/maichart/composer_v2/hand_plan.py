"""Causal event-level plan: read the completed group, anticipate the next layout.

The observed state and the prospective plan are predictions, not chart labels.
Only the loss reads authored group/transition targets.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .codec import KINDS, atom_components, explicit_note_atoms
from .ordered_configuration import POSITIONS
from .spatial_history import DiskSpatialHistory
from . import hand_memory

POLICY = 'contact_plan_v1'
CHOICES = ('auto', 'off', POLICY, *hand_memory.POLICIES)
BASE_FIELDS = ('chord', 'slide', 'hold', 'touch', 'ongoing_hold', 'ongoing_trace',
          'release_before_successor', 'successor_chord', 'successor_adjacent',
          'successor_return', 'left_strike', 'same_striking_hand')
SIDE = len(BASE_FIELDS)
SUPPORT = SIDE + 8
FIELDS = (*BASE_FIELDS, *(f'left_on_{i}' for i in range(1,9)), *(f'supported_on_{i}' for i in range(1,9)))
AUX_WEIGHT = .1
MECHANICS = 'mechanics'


@lru_cache(maxsize=8192)
def rim_taps(text):
    """Map scorably equivalent EX/Break Tap heads to physical pads."""
    result=[]
    for atom in explicit_note_atoms(text):
        parts=atom_components(atom)
        if parts[0]!='N:tap' or any(
                t.startswith('S:') and not set(t[2:]).issubset('bx') for t in parts):
            return ()
        positions=[t[2:] for t in parts if t.startswith('P:')]
        if len(positions)!=1 or positions[0] not in tuple('12345678'):
            return ()
        result.append((atom,int(positions[0])))
    return tuple(result) if len(result)<=2 and len({k for _,k in result})==len(result) else ()


def _tap_episode(song, configuration, block):
    """One bounded, fully reviewed example per 32 events; no implied Slide owner."""
    from maichart.event_timing import EventTempo
    from .hand_flow import recognize_hands
    # Single/double rim taps are the first supported weak-label domain.
    best, run = [], []
    for i in range(block,min(block+32,len(song.points))):
        if not rim_taps(song.points[i].note_text):
            run=[]
            continue
        if run and configuration.seconds[i]-configuration.seconds[i-1]>.45:
            run=[]
        run.append(i)
        if len(run)>len(best):
            best=run.copy()
        if len(best)==12:
            break
    if len(best)<5:
        return {}
    start,stop=configuration.seconds[[best[0],best[-1]]]
    if (configuration.quality.covered(start,np.nextafter(stop,np.inf),1)<=0 or
            len(configuration._overlapping_sustains(start,stop+1e-8))):
        return {}
    tempo=[EventTempo(b,v) for b,v in zip(song.clock.beats,song.clock.bpms)]
    result=recognize_hands([song.points[i] for i in best],tempo,song.clock,start-1e-6,stop+1e-6,
                           stop+1e-6,beam_width=8,max_solutions=3)
    if not result['coverage']['complete']:
        return {}
    # The first displayed path supplies contact identities. Its probabilities
    # compare the cheapest complete paths on each side across the retained
    # search graph; they do not depend on a majority vote among display paths.
    frames={round(row['second'],5): row for row in result['solutions'][0]['frames']}
    labels={}
    for i in best[2:-1]:
        row=frames.get(round(float(configuration.seconds[i]),5))
        if row is None:
            continue
        contacts=[c for side in ('left','right') for c in row[side]
                  if c['kind']=='button' and c['phase']=='pulse']
        fields={}
        for contact in contacts:
            lane=contact['lane']
            probability=contact.get('p_left')
            if not 1<=lane<=8:
                continue
            # A pruned rival cannot certify a hand. Ambiguity with both paths
            # surviving can teach a soft side target, including exactly 0.5.
            fields[SUPPORT+lane-1]=0.
            if contact.get('seen_both') and probability is not None:
                fields[SIDE+lane-1]=float(probability)
                fields[SUPPORT+lane-1]=abs(2.*probability-1.)
                if len(contacts)==1:
                    fields[10]=float(probability)
        relation=row.get('relation_evidence')
        if relation and relation['seen_both'] and relation['same'] is not None:
            fields[11]=float(relation['same'])
        labels[i]=fields
    return labels


def add_hand_consensus(values,weights,base,dataset,song,configuration,begin,end):
    if not hasattr(song,'_hand_episodes'):
        song._hand_episodes={}
    for block in range(begin//32*32,end,32):
        if block not in song._hand_episodes:
            song._hand_episodes[block]=_tap_episode(song,configuration,block)
        for i,answers in song._hand_episodes[block].items():
            if begin<=i<end:
                for j,target in answers.items():
                    values[i-begin,j]=target
                    weights[i-begin,j]=base[i-begin]


def trace(probabilities, note_text, *, supported_context=True):
    """Readable priors for the keys actually emitted. A confidence is a model estimate."""
    if len(probabilities)!=len(FIELDS):
        raise ValueError('Incomplete hand plan probability vector')
    pads=rim_taps(note_text) if supported_context else ()
    supported=bool(pads)
    result=[]
    for p,lane in (pads or [(atom,None) for atom in explicit_note_atoms(note_text)]):
        if not supported:
            result.append(dict(contact=p,p_left=None,p_right=None,weak_support=0.,assignment=None,
                               reason='outside_weak_label_domain'))
            continue
        left=float(probabilities[SIDE+lane-1]);support=float(probabilities[SUPPORT+lane-1])
        result.append(dict(contact=p,lane=lane,p_left=left,p_right=1.-left,weak_support=support,
                           assignment=('left' if left>.5 else 'right') if support>=.5 and abs(left-.5)>=.15 else None))
    return dict(contacts=result, confidence_basis='uncalibrated_weak_teacher',
                same_striking_hand=float(probabilities[11]) if supported and len(pads)==1 else None)


def resolve(requested, *, saved=None, resume=False):
    previous = (saved or {}).get('hand_plan_policy', 'off')
    if requested not in CHOICES or previous not in CHOICES[1:]:
        raise ValueError('未知的手顺预判设置')
    selected = previous if requested == 'auto' else requested
    if resume and previous != selected:
        raise ValueError('续训沿用原手顺预判设置，更改请在新目录微调')
    return selected


class HandPlan(nn.Module):
    def __init__(self, codec, width):
        super().__init__()
        self.observe = nn.Linear(width, len(FIELDS))
        self.predict = nn.Linear(width + len(FIELDS), len(FIELDS))
        # A fixed physical inventory keeps output columns stable across vocabulary expansion.
        names = [*(f'N:{k}' for k in KINDS), *(f'P:{p}' for p in POSITIONS)]
        self.execute = nn.Linear(len(FIELDS)*2, len(names), bias=False)
        nn.init.zeros_(self.execute.weight)
        with torch.no_grad():
            for layer in (self.observe,self.predict):
                layer.weight[SIDE:].zero_()
                layer.bias[SIDE:SUPPORT].zero_()
                layer.bias[SUPPORT:].fill_(-2.)
        selected = [(codec.ids[n], j) for j,n in enumerate(names) if n in codec.ids]
        self.register_buffer('ids', torch.tensor([i for i,_ in selected]), persistent=False)
        self.register_buffer('columns', torch.tensor([j for _,j in selected]), persistent=False)
        self.register_buffer('buttons',torch.tensor([names.index(f'P:{k}') for k in range(1,9)
                                                    if f'P:{k}' in codec.ids]),persistent=False)
        regions = {codec.ctx_id: 1, codec.ref_id: 2, codec.tgt_id: 3}
        self.register_buffer('region',torch.tensor([regions.get(i,0) for i in range(len(codec.tokens))]),persistent=False)
        self.register_buffer('phase',torch.tensor([t.startswith('F:') for t in codec.tokens]),persistent=False)
        self.register_buffer('pad',torch.tensor([POSITIONS.index(t[2:])+1
            if t.startswith('P:') and t[2:] in POSITIONS[:8] else 0 for t in codec.tokens]),persistent=False)
        self.register_buffer('non_tap',torch.tensor([
            t.startswith('N:') and t!='N:tap' or t.startswith('P:') and t[2:] not in POSITIONS[:8]
            or t.startswith('S:') and not set(t[2:]).issubset('bx') for t in codec.tokens]),persistent=False)
        self.register_buffer('lane_numbers',torch.arange(1,9),persistent=False)
        self.register_buffer('layout_query',torch.tensor([
            t.startswith(('F:', 'N:', 'P:', 'S:', 'D:', 'V:')) or t in ('[END_NOTE]', '[END_DURATION]')
            for t in codec.tokens]),persistent=False)
        self.end = codec.event_end_id
        self.at = codec.at_id

    def forward(self, hidden, ids, start, state, mechanics=None, visible=None):
        if mechanics is not None:
            hidden = torch.where(mechanics[:,None,None],hidden.detach(),hidden)
        observed = self.observe(hidden)
        region = self.region[ids].cummax(1).values
        if start:
            region = torch.maximum(region,state['region'][:,start-1,None])
        idx = torch.arange(1,1+ids.shape[1],device=ids.device)

        def carry(rows, use, key):
            before = (state[key][:,start-1:start] if start else rows.new_zeros(len(rows),1,len(FIELDS)))
            latest = (use * idx).cummax(1).values
            values = torch.cat((before,rows),dim=1).gather(1,latest[...,None].expand_as(rows))
            if state is not None:
                DiskSpatialHistory._remember(state,key,values,start)
            return values

        if state is not None:
            DiskSpatialHistory._remember(state,'region',region,start)
        real_history = region.eq(1) | region.eq(3)
        if visible is None:
            visible = torch.ones_like(ids,dtype=torch.bool)
        def last_position(mask, key):
            last = (mask*(idx+start)).cummax(1).values
            if start:
                last = torch.maximum(last,state[key][:,start-1,None])
            if state is not None:
                DiskSpatialHistory._remember(state,key,last,start)
            return last
        opened = last_position(ids.eq(self.at),'opened')
        intact = opened > last_position(~visible,'masked')
        tap = intact & (opened>last_position(self.non_tap[ids],'other'))
        touched=(self.pad[ids][...,None].eq(self.lane_numbers)*(idx+start)[:,None]).cummax(1).values
        if start:
            touched=torch.maximum(touched,state['pads'][:,start-1:start])
        if state is not None:
            DiskSpatialHistory._remember(state,'pads',touched,start)
        chosen=touched>opened[...,None]
        n=chosen.sum(-1)
        single=(n==1)&tap
        chosen=chosen & tap[...,None] & (n<=2)[...,None]
        def for_committed_group(value):
            # The pre-note distribution has eight counterfactual answers. Send
            # only the channels for pads actually played into the next group.
            return torch.cat((value[...,:10]*intact[...,None],value[...,10:SIDE]*single[...,None],
                              value[...,SIDE:SUPPORT]*chosen,value[...,SUPPORT:]*chosen),dim=-1)
        def supported(value):
            # Aggregate weak left/same-hand targets are diagnostic. Only
            # lane-conditional assignments have their own learned support.
            return torch.cat((value[...,:10], torch.zeros_like(value[...,10:SIDE]),
                              value[...,SIDE:SUPPORT]*(value[...,SUPPORT:]+.5),
                              value[...,SUPPORT:]),dim=-1)
        previous = carry(for_committed_group(observed.sigmoid()-.5), ids.eq(self.end) & real_history, 'completed')
        predicted = self.predict(torch.cat((hidden,supported(previous).to(hidden.dtype)),dim=-1))
        # The plan is decided on F:<phase>, before the event type and every key.
        plan = carry((predicted.sigmoid()-.5)*intact[...,None], self.phase[ids] & real_history,'plan')
        # Commit the *same emitted forecast and its confidence* as a separate
        # causal control memory when the group closes. Reference tokens never
        # overwrite it. A full decode, prefix compaction and a branch rewind can
        # therefore all reconstruct this context without peeking at hand labels.
        committed = carry(for_committed_group(plan), ids.eq(self.end) & real_history, 'committed_plan')
        adjustment = self.execute(torch.cat((supported(plan),supported(committed)),dim=-1))
        adjustment = adjustment * (region.eq(3) & self.layout_query[ids])[...,None]
        return adjustment, predicted, observed


def add_item(item, dataset, song, configuration, begin, end, *, authored_history=True):
    if end <= begin:
        return
    from . import motion_writing
    codec = dataset.codec
    ids = item['input_ids'].numpy()
    start = np.flatnonzero(ids == codec.tgt_id)[0]+1
    queries = start+1+np.flatnonzero(ids[start:] == codec.phase_id)
    completed = start+np.flatnonzero(ids[start:] == codec.event_end_id)
    if len(queries) != end-begin:
        raise ValueError('Hand plan queries must align to complete authored groups')
    rows = np.arange(begin,end)
    values = np.zeros((len(rows),len(FIELDS)),np.float32)
    weights = np.zeros_like(values)
    base = item['token_weights'][queries].numpy()
    for j,i in enumerate(rows):
        if base[j]<=0:
            continue
        second = configuration.seconds[i]
        fields, key = motion_writing.contact(song.points[i].note_text)
        names = {motion_writing.EVENT_FIELDS[f] for f in fields}
        values[j,:4] = ('chord' in names,'slide' in names,'hold' in names or 'touch_hold' in names,
                         'touch' in names or 'touch_hold' in names)
        weights[j,:4] = base[j]
        next_sec = configuration.seconds[i+1] if i+1<len(song.points) else second
        overlaps = configuration._overlapping_sustains(second,next_sec+1e-8)
        active = overlaps[(overlaps[:,0] < second-1e-7) & (overlaps[:,1] <= second+1e-7)
                          & (overlaps[:,2] > second+1e-7)]
        # Reviewed-only sources: a masked sustain's origin cannot supply an answer.
        trusted = (not len(active) or configuration.quality.at(active[:,0])[:,1].min()>0)
        if trusted:
            values[j,4:6] = (np.any(active[:,3]<2),np.any(active[:,3]==2))
            weights[j,4:6] = base[j]
        if i+1<len(song.points):
            seam = min(base[j],configuration.quality.covered(second,np.nextafter(next_sec,np.inf),1))
            other, next_key = motion_writing.contact(song.points[i+1].note_text)
            values[j,6:9] = (np.any(active[:,2] <= next_sec+1e-7),
                               motion_writing.EVENT_FIELDS.index('chord') in other,
                               bool(key and next_key and (next_key-key)%8 in (1,7)))
            weights[j,6:9] = (seam if trusted else 0.,seam,seam if key and next_key else 0.)
            if i and key and next_key:
                previous = motion_writing.contact(song.points[i-1].note_text)[1]
                if previous and previous!=key:
                    values[j,9] = previous == next_key
                    weights[j,9] = min(seam,configuration.quality.covered(configuration.seconds[i-1],second,1))
    memory = getattr(dataset,'hand_plan_policy','off') in hand_memory.POLICIES
    if authored_history and not memory:
        add_hand_consensus(values, weights, base, dataset, song, configuration, begin, end)
    elif authored_history and getattr(song,'hand_memory_labels',None) is not None:
        # Keep the inherited prospective readout trainable, using the same offline
        # evidence for actual rim Tap heads rather than invoking a second solver.
        for j,i in enumerate(rows):
            pads=sorted(k for _,k in rim_taps(song.points[i].note_text))
            for slot,pad in enumerate(pads):
                grade=int(song.hand_memory_labels[i,slot])
                if grade==hand_memory.MISSING:
                    continue
                if grade==hand_memory.NO_RIVAL:
                    grade=0
                values[j,SUPPORT+pad-1]=abs(grade)/2.
                weights[j,SUPPORT+pad-1]=base[j]
                if grade:
                    values[j,SIDE+pad-1]=float(grade>0)
                    weights[j,SIDE+pad-1]=base[j]*abs(grade)/2.
    elif not authored_history and len(weights):
        # These values depend on the earlier hand/trace or previous button.
        # The replay view pairs nominally compatible, not identical, histories.
        weights[:,4:7] = 0.
        weights[0,9] = 0.
    weights *= float(item['sample_weight'])/len(BASE_FIELDS)
    item['hand_positions'] = torch.from_numpy(queries)
    # A training slice can finish just before END_EVENT's input (last label).
    item['hand_completed'] = torch.tensor([completed[j] if j<len(completed) else -1 for j in range(len(rows))])
    item['hand_targets'] = torch.from_numpy(values)
    item['hand_weights'] = torch.from_numpy(weights)
    if memory:
        hand_memory.add_targets(item,song,configuration,begin,end,authored_history)


def collate(items,batch):
    records = [(i,it) for i,it in enumerate(items) if 'hand_positions' in it]
    if not records:
        return
    for key in ('hand_positions','hand_completed'):
        batch[key] = torch.cat([torch.stack((torch.full_like(it[key],i),it[key]),dim=1) for i,it in records])
    for key in ('hand_targets','hand_weights'):
        batch[key] = torch.cat([it[key] for _,it in records])
    batch['mechanics_rows'] = torch.tensor([bool(it.get('mechanics_only')) for it in items])
    batch['mechanics_mass'] = float(sum(float(it['hand_weights'].sum()) for _,it in records if it.get('mechanics_only')))
    hand_memory.collate(items,batch)


def supervision(predicted, observed, batch):
    rows, indices = batch['hand_positions'].unbind(-1)
    completed = batch['hand_completed'][:,1]
    target, weights = batch['hand_targets'],batch['hand_weights']
    before = F.binary_cross_entropy_with_logits(predicted[rows,indices].float(),target,reduction='none')
    after = F.binary_cross_entropy_with_logits(observed[rows,completed.clamp_min(0)].float(),target,reduction='none')
    # For unfinished inputs retain the full prior loss, with no posterior target.
    error = torch.where(completed[:,None]>=0,(before+after)*.5,before)
    return (error*weights).sum(),weights.sum()


def calibration_sums(predicted, batch):
    """Lane-conditional prospective predictions vs two-sided solver evidence only."""
    rows,indices=batch['hand_positions'].unbind(-1)
    p=predicted[rows,indices,SIDE:SUPPORT].float().sigmoid()
    target=batch['hand_targets'][:,SIDE:SUPPORT]
    valid=batch['hand_weights'][:,SIDE:SUPPORT].gt(0).float()
    confidence=torch.maximum(p,1.-p)
    teacher_agreement=torch.where(p>=.5,target,1.-target)
    terms=[valid.sum(),((p-target).square()*valid).sum(),((.5-target).square()*valid).sum()]
    for low,hi in ((.5,.65),(.65,.8),(.8,1.01)):
        mask=valid*((confidence>=low)&(confidence<hi))
        terms.extend((mask.sum(),(confidence*mask).sum(),(teacher_agreement*mask).sum()))
    return torch.stack(terms)


def calibration_report(sums):
    n,squared,baseline,*bins=sums
    return dict(weak_contacts=int(n),mean_squared_probability_error=squared/n if n else None,
                neutral_error=baseline/n if n else None,
                bins=[dict(lower=low,count=int(k),predicted=p/k if k else None,
                           weak_agreement=q/k if k else None)
                      for low,(k,p,q) in zip((.5,.65,.8),zip(bins[::3],bins[1::3],bins[2::3]))])
