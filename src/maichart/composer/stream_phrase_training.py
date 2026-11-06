"""Causal phrase block scheduling and training-only set-relation labels.

Only the boundary/count block planner is shared with inference. Relation
labels must never enter the causal decoder, including through derived state.
"""
import math
from bisect import bisect_right
from functools import lru_cache

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .stream_continuation import typed_event

TRAINING_WINDOW_POLICIES = ('block_v1', 'phrase_complete_v1')
RELATION_POLICIES = ('off', 'future_sets_v1')
RELATION_HORIZONS = 2
RELATION_DIM = 22
# [0:8] future physical button heads, [8:16] all set cross-correlations
# against the last committed NOTE (one value per circular displacement),
# [16:18] log1p beat/second distance from the causal cursor,
# [18] WAIT before that NOTE, [19] no such future NOTE (completion),
# [20:22] future NOTE has hold/slide. Symmetric sets keep every displacement.


class CausalPhraseBlockIndex:
    """Incremental history anchor using only committed count and current clock.

    Short audio phrases share one block until K targets have already elapsed.
    A subsequent phrase boundary resets it; the hard cap can reset mid-phrase.
    No future phrase event count or target arrival participates in this choice.
    """
    def __init__(self,boundaries,minimum,maximum):
        self.boundaries=boundaries;self.minimum=int(minimum);self.maximum=int(maximum)
        self.block_start=0;self.segment=None;self.index=-1

    def advance(self,index,cursor):
        segment=min(len(self.boundaries)-2,max(0,bisect_right(self.boundaries,float(cursor))-1))
        if index==self.index:return self.block_start
        if index!=self.index+1:raise ValueError('Phrase block index must advance one committed step at a time')
        elapsed=index-self.block_start
        if elapsed>=self.maximum or (self.segment is not None and segment!=self.segment and elapsed>=self.minimum):
            self.block_start=index
        self.segment=segment;self.index=index
        return self.block_start


def phrase_training_blocks(stream,boundaries,minimum,maximum):
    """Pack whole short phrases, and mark every cap-split phrase incomplete."""
    planner=CausalPhraseBlockIndex(boundaries,minimum,maximum)
    anchors=np.asarray([planner.advance(i,step.start) for i,step in enumerate(stream)],dtype=np.int64)
    segments=np.clip(np.searchsorted(boundaries,[float(s.start) for s in stream],side='right')-1,
                     0,len(boundaries)-2).astype(np.int64)
    starts=np.flatnonzero(np.r_[True,anchors[1:]!=anchors[:-1]]) if len(stream) else np.empty(0,dtype=np.int64)
    spans=[(int(start),int(stop)) for start,stop in zip(starts,np.r_[starts[1:],len(stream)])]
    positions=np.zeros(len(stream),dtype=np.int64);complete=np.zeros(len(stream),dtype=np.bool_)
    phrase_starts=np.flatnonzero(np.r_[True,segments[1:]!=segments[:-1]]) if len(stream) else np.empty(0,dtype=np.int64)
    for start,stop in zip(phrase_starts,np.r_[phrase_starts[1:],len(stream)]):
        positions[start:stop]=np.arange(stop-start)
        complete[start:stop]=anchors[start]==anchors[stop-1]
    return spans,segments,positions,complete


def phrase_training_spans(stream, boundaries, maximum):
    """Disjoint target spans; long phrases retain labels but disable mode NLL.

    A bounded fragment is never advertised as a complete phrase. Its group is
    still identified for auditing, but complete=False excludes it from exact
    whole-phrase latent marginalization rather than choosing modes per slice.
    """
    # Departure time is already known before timing is sampled. Grouping by
    # the target arrival would let future timing select the model's history.
    segments = np.clip(np.searchsorted(boundaries, [float(s.start) for s in stream],
                                      side='right')-1, 0, len(boundaries)-2)
    result = []
    start = 0
    while start < len(stream):
        stop = start+1
        while stop < len(stream) and segments[stop] == segments[start]:
            stop += 1
        complete = stop-start <= maximum
        for lo in range(start, stop, maximum):
            result.append((lo, min(lo+maximum, stop), int(segments[start]), complete, start))
        start = stop
    return result


def future_relation_index(stream, unknown):
    """Compact O(events) index, reusable even when dense song labels won't fit."""
    count = len(stream)
    next_notes = np.full((count+1, 2), -1, dtype=np.int64)
    wait_prefix = np.zeros(count+1, dtype=np.int64)
    unknown_prefix = np.zeros(count+1, dtype=np.int64)
    previous = np.full(count, -1, dtype=np.int64)
    prior = -1
    for i, step in enumerate(stream):
        previous[i] = prior
        if step.kind == 2: prior = i
        wait_prefix[i+1] = wait_prefix[i]+int(step.kind == 3)
        unknown_prefix[i+1] = unknown_prefix[i]+int(unknown[i])
    for i in range(count-1, -1, -1):
        next_notes[i] = (i, next_notes[i+1,0]) if stream[i].kind == 2 else next_notes[i+1]
    return next_notes,wait_prefix,unknown_prefix,previous


@lru_cache(maxsize=4096)
def _head_cross_correlation(old,buttons):
    """Immutable normalized rotation labels shared by repeated head sets."""
    denominator=max(len(old),len(buttons))
    return tuple(sum(((b-1+shift)%8+1) in buttons for b in old)/denominator
                 for shift in range(8))


def future_relation_rows(stream, unknown, first, end, *, index=None):
    """O(requested rows) with a cached index; CPU labels, no model inputs."""
    count=len(stream)
    next_notes,wait_prefix,unknown_prefix,previous=(future_relation_index(stream,unknown) if index is None else index)
    values = np.zeros((end-first, RELATION_HORIZONS, RELATION_DIM),dtype=np.float32)
    masks = np.zeros(values.shape,dtype=np.bool_)
    for local, i in enumerate(range(first, end)):
        p = previous[i]
        old = typed_event(stream[p].text)[1] if p >= 0 and not unknown[p] else frozenset()
        for h, j in enumerate(next_notes[i]):
            stop = int(j)+1 if j >= 0 else count
            valid = bool(unknown_prefix[stop] == unknown_prefix[i])
            masks[local,h,19] = valid
            values[local,h,19] = float(j < 0)
            if j < 0 or not valid: continue
            target = stream[j]
            types, buttons, _ = typed_event(target.text)
            if not types: continue
            masks[local,h,:8] = True
            for button in buttons: values[local,h,button-1] = 1.
            if old and buttons:
                masks[local,h,8:16] = True
                values[local,h,8:16] = _head_cross_correlation(old,buttons)
            values[local,h,16] = math.log1p(max(0.,float(target.end-stream[i].start)))
            values[local,h,17] = math.log1p(max(0.,target.end_second-stream[i].start_second))
            values[local,h,18] = float(wait_prefix[j] > wait_prefix[i])
            values[local,h,20] = float('hold' in types)
            values[local,h,21] = float('slide' in types)
            masks[local,h,16:19] = True
            masks[local,h,20:22] = True
    return torch.from_numpy(values),torch.from_numpy(masks)


def collate_phrase_training(items, output):
    """Group indices are batch local; only complete primary targets are active."""
    if not any('training_phrase_group' in item for item in items): return
    if not all('training_phrase_group' in item for item in items):
        raise ValueError('Cannot mix phrase training window policies')
    if any('_training_chart_id' in item for item in items):
        chart_ids=[int(item.get('_training_chart_id',-1)) for item in items]
        output['training_chart_id']=torch.tensor(chart_ids,dtype=torch.long)
        seen_charts=set();first_chart=[]
        for chart in chart_ids:
            first_chart.append(chart>=0 and chart not in seen_charts)
            if chart>=0:seen_charts.add(chart)
        output['profile_chart_first']=torch.tensor(first_chart,dtype=torch.bool)
        output['profile_chart_rows']=torch.tensor([i for i,first in enumerate(first_chart) if first],dtype=torch.long)
    groups, positions, complete, weights = [], [], [], []
    chart_phrases = {}
    for item in items:
        active = item['loss_mask'] & item['sample_weight'].gt(0) & item['training_phrase_group'].ge(0)
        mapped=torch.full_like(item['training_phrase_group'],-1)
        completed=torch.zeros_like(active)
        positions.append(item['training_phrase_position'])
        trusted=item.get('configuration_mask',torch.ones_like(active)).bool()
        core=item['loss_mask'] & item['training_phrase_group'].ge(0)
        for segment in torch.unique(item['training_phrase_group'][core]).tolist():
            members=core & item['training_phrase_group'].eq(segment)
            valid=members & active
            mapped[valid]=len(weights)
            whole_trusted=bool(trusted[members].all())
            completed |= item['training_phrase_complete'] & valid & whole_trusted
            group_id=len(weights)
            if whole_trusted and bool((item['training_phrase_complete'] & valid).any()):
                chart=item.get('_training_chart_id')
                if chart is not None:
                    chart_phrases.setdefault(int(chart), []).append((int(segment),group_id))
            weights.append(item['sample_weight'][valid].sum())
        groups.append(mapped);complete.append(completed)
    output['training_phrase_group'] = pad_sequence(groups,batch_first=True,padding_value=-1)
    output['training_phrase_position'] = pad_sequence(positions,batch_first=True,padding_value=-1)
    output['training_phrase_complete'] = pad_sequence(complete,batch_first=True)
    output['training_phrase_group_weight'] = torch.stack(weights) if weights else torch.zeros(0)
    # Separate chart identity from shared audio identity. Spread each small
    # group over its available phrase range rather than adjacent windows.
    response=torch.full((len(weights),),-1,dtype=torch.long)
    response_count=0
    sizes={int(item.get('_song_response_group_size',4)) for item in items}
    if len(sizes)!=1:raise ValueError('Cannot mix song response group sizes')
    group_size=sizes.pop()
    if not 2<=group_size<=4:raise ValueError('Invalid song response group size')
    for entries in chart_phrases.values():
        entries=sorted(entries)
        # Repeated seam observations of a phrase cannot become separate evidence.
        unique=[];seen=set()
        for segment,gid in entries:
            if segment not in seen:unique.append((segment,gid));seen.add(segment)
        partitions=max(1,(len(unique)+group_size-1)//group_size)
        for offset in range(partitions):
            selected=unique[offset::partitions]
            if len(selected)<2 or selected[-1][0]-selected[0][0]<2:continue
            for _,gid in selected:response[gid]=response_count
            response_count+=1
    output['training_song_response_group']=response
    output['_song_response_group_count']=response_count
    # Variable K must be normalized by the actual primary target budget.
    output['_loss_normalizer'] = sum(float(item['_training_target_budget'].sum()) for item in items)


def collate_future_relations(items, output):
    if not any('target_relation' in item for item in items): return
    if not all('target_relation' in item for item in items):
        raise ValueError('Cannot mix future relation supervision policies')
    for key in ('target_relation','target_relation_mask'):
        values=[item[key] for item in items]
        if key=='target_relation_mask':
            values=[value & item.get('configuration_mask',torch.ones(value.size(0),dtype=torch.bool))[:,None,None]
                    for item,value in zip(items,values)]
        output[key] = pad_sequence(values,batch_first=True)


def collate_draft_loss_groups(items,output):
    """Loss-only phrase ownership for departure time and landing marks.

    Grouping never changes sample weights or primary target multiplicities.
    A crossing NOTE can carry time evidence for one phrase and configuration
    evidence for another. Counterfactual prefixes never share a mode group.
    """
    if 'phrase_start_beat' not in output:return
    shape=output['loss_mask'].shape
    departure=torch.full(shape,-1,dtype=torch.long);landing=torch.full_like(departure,-1)
    registry={};entries=[]
    charts=output.get('unified_chart_inverse',torch.arange(len(items))).tolist()
    for row,item in enumerate(items):
        changed=bool(item.get('_counterfactual_history',False) or item.get('_draft_prefix_isolated',False))
        chart=int(item.get('_training_chart_id',item.get('song_index',-1)))
        owner=('row',row) if changed or chart<0 else ('chart',chart)
        expected=item.get('_draft_loss_expected',{})
        active=item['loss_mask'].bool()&item['sample_weight'].gt(0)
        kinds=item['target_kind'];trusted=item.get('configuration_mask',torch.ones_like(active)).bool()
        # Family-only recovery evidence owns a landing decision too. This is
        # grouping only; geometry and relation supervision keep their masks.
        trusted=trusted|item.get('configuration_family_mask',torch.zeros_like(active)).bool()
        indices=item.get('step_index',torch.arange(len(active)))
        for side,prefix,mapping in ((0,'',departure),(1,'candidate_',landing)):
            starts=item.get(prefix+'phrase_start_beat',item['phrase_start_beat'])
            ends=item.get(prefix+'phrase_end_beat',item['phrase_end_beat'])
            selected=active if side==0 else active&kinds.eq(2)&trusted
            modes=item.get(prefix+'draft_committed_mode',torch.full_like(kinds,-1))
            for position in selected.nonzero(as_tuple=False).flatten().tolist():
                bounds=(float(starts[position]),float(ends[position]));mode=int(modes[position])
                key=(owner,*bounds,mode)
                if key not in registry:
                    registry[key]=len(entries)
                    entries.append(dict(row=row,position=position,side=side,chart=charts[row],
                        time=float(item['step_beat'][position]),changed=changed,
                        expected=expected.get(bounds),observed=[set(),set()],key=key,mode=mode))
                group=registry[key];entry=entries[group];mapping[row,position]=group
                query_time=float(item['step_beat'][position])
                if (query_time,side)<(entry['time'],entry['side']):
                    entry.update(row=row,position=position,side=side,time=query_time)
                entry['observed'][side].add(int(indices[position]))
                if entry['expected'] is None:entry['expected']=expected.get(bounds)
    output['draft_departure_group']=departure;output['draft_landing_group']=landing
    output['draft_loss_group_count']=len(entries)
    for name in ('rows','positions','side','chart','mode'):
        source={'rows':'row','positions':'position'}.get(name,name)
        output['draft_loss_group_'+name]=torch.tensor([entry[source] for entry in entries],dtype=torch.long)
    output['draft_loss_group_complete']=torch.tensor([
        not entry['changed'] and entry['expected'] is not None and entry['expected'][2]
        and len(entry['observed'][0])==entry['expected'][0]
        and len(entry['observed'][1])==entry['expected'][1] for entry in entries],dtype=torch.bool)
    output['draft_loss_group_keys']=[entry['key'] for entry in entries]
    output['draft_loss_group_decisions']=torch.tensor([
        len(entry['observed'][0]|entry['observed'][1]) for entry in entries],dtype=torch.long)


def normalize_phrase_plan_weights(items,output):
    """Plan/phrase losses divide by B; action loss divides by target budget D.

    A plan row's original mass is core_event_mass/K, so multiplying by B*K/D
    preserves the old relative coefficient when phrase lengths change. Seam
    repeats already split event weights and do not create extra plan mass.
    """
    output['_phrase_budget_scale']=1.
    if not all('_training_reference_targets' in item for item in items):return
    targets={int(item['_training_reference_targets']) for item in items}
    if len(targets)!=1:raise ValueError('Cannot mix reference target budgets')
    target_count=targets.pop()
    denominator=float(output['_loss_normalizer'])
    if target_count<=0 or denominator<=0:raise ValueError('Invalid phrase plan target budget')
    scale=len(items)*target_count/denominator
    for key in ('audio_plan_weight','audio_plan_blank_weight','target_phrase_weight'):
        if key in output:output[key]*=scale
    output['_phrase_budget_scale']=scale
