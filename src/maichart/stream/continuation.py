"""Shared event-aligned replay and authored cycle-exit supervision."""


def replay_budget(config):
    if getattr(config, 'continuation_profile', 'reset_v1') != 'overlap_v1':
        return 0
    # Keep room for newly supervised output even in small diagnostic models.
    return min(config.context_replay_tokens, max(0, config.max_tokens // 2))


def replay_start(ids, end, budget, at_id):
    """Earliest whole event fitting the suffix; never begin inside a cell."""
    lower = max(1, end - max(0, budget))
    return next((i for i in range(lower, end) if ids[i] == at_id), end)


def continuation_ranges(ids, max_tokens, at_id, eos_id, overlap=0):
    from .data import _target_ranges
    # A single long event remains legal; its replay shrinks to fit.
    positions = [i for i,t in enumerate(ids) if t == at_id] + [len(ids)-1, len(ids)]
    longest = max(b-a for a,b in zip(positions, positions[1:]))
    if longest>max_tokens:
        raise ValueError(f'单个配置需要 {longest} token，超过流上下文上限 {max_tokens}')
    capacity = max(max_tokens-overlap, longest)
    for start,end in _target_ranges(ids, capacity, at_id, eos_id):
        warm = replay_start(ids, start, min(overlap, max_tokens-(end-start)), at_id)
        yield warm,start,end


def cycle_exit_targets(ids, codec, max_period=24):
    """Mark the first divergent token when an authored chart exits a cycle.

    No generated prefix is paired with an authored answer. Legitimate repeated
    cells receive no extra target; only a real departure from two prior cycles
    is marked. Multiple matching periods share one token target.
    """
    from fractions import Fraction
    from .symmetry import event_lanes
    starts = [i for i,t in enumerate(ids) if t == codec.at_id]
    cells=[];shapes=[];note_starts=[];classes=[];lanes=[];beats=[];deltas=[];durations=[]
    for a,b in zip(starts, starts[1:]+[len(ids)-1]):
        note=ids.index(codec.note_id,a,b)+1
        beat_text=''.join(codec.tokens[t][2:] for t in ids[a+1:note-1] if codec.tokens[t].startswith('P:'))
        beats.append(Fraction(beat_text) if beat_text else Fraction(0))
        deltas.append(beats[-1]-beats[-2] if len(beats)>1 else None)
        cells.append(tuple(ids[note:b]));note_starts.append(note)
        classes.append(_structure_classes(ids[note:b],codec))
        durations.append(tuple(t for t,k in zip(ids[note:b],classes[-1]) if k=='duration'))
        raw_text=''.join(codec.tokens[t][2:] for t in ids[note:b] if codec.tokens[t].startswith('N:'))
        lanes.append(event_lanes(raw_text))
        # Only note characters, never END ('<end>'[2:] contains '>', which
        # used to mark every event as a Slide) and never duration digits.
        structure=[c for c,k in zip((codec.tokens[t][2:] for t in ids[note:b]),classes[-1])
                   if k not in ('duration','<end>')]
        text=''.join(structure)
        shapes.append((bool('/' in text),bool('b' in text),bool('h' in text),
                       bool(any(c in text for c in '-<>^pqszvwV')),bool(any(c in text for c in 'ABCDE')),
                       len(text)))
    result=[False]*len(ids)
    for i,current in enumerate(cells):
        for period in range(1,min(max_period,(i-1)//2)+1):
            if (deltas[i-2*period:i-period] == deltas[i-period:i]
                and deltas[i] != deltas[i-period]):
                expected_pos=codec.position(beats[i-1]+deltas[i-period])[1:]
                actual_pos=ids[starts[i]+1:note_starts[i]]
                divergence=next((j for j,(a,b) in enumerate(zip(actual_pos,expected_pos)) if a!=b),None)
                if divergence is not None:
                    result[starts[i]+1+divergence]=True
        for period in range(1,min(max_period,i//2)+1):
            if cells[i-2*period:i-period] == cells[i-period:i]:
                expected=cells[i-period]
                if current == expected:
                    continue
                divergence=next((j for j,(a,b) in enumerate(zip(current,expected)) if a!=b),None)
                if divergence is not None:
                    result[note_starts[i]+divergence]=True
            elif (lanes[i-2*period:i-period] == lanes[i-period:i]
                  and all(lanes[i-period:i])
                  and lanes[i] != lanes[i-period]):
                expected=cells[i-period]
                divergence=next((j for j,(a,b) in enumerate(zip(current,expected)) if a!=b),None)
                if divergence is not None:
                    result[note_starts[i]+divergence]=True
            if (durations[i-2*period:i-period] == durations[i-period:i]
                and all(durations[i-period:i])
                and durations[i] != durations[i-period]):
                if durations[i]:
                    our_dur_pos=[j for j,k in enumerate(classes[i]) if k=='duration']
                    div_idx=next((our_dur_pos[idx] for idx,(a,b) in enumerate(zip(durations[i],durations[i-period])) if a!=b),
                                 our_dur_pos[0] if our_dur_pos else None)
                    if div_idx is not None:
                        result[note_starts[i]+div_idx]=True
        if i >= 2 and not lanes[i-1] and not lanes[i-2] and lanes[i] and current:
            lane_pos=next((j for j,k in enumerate(classes[i]) if k=='lane'),0)
            result[note_starts[i]+lane_pos]=True
        if i >= 6 and all(lanes[i-6:i]) and lanes[i] and current:
            confined=set().union(*lanes[i-6:i])
            out_lanes=set(lanes[i])-confined
            if 1 <= len(confined) <= 4 and out_lanes:
                escape_pos=next((j for j,(t,k) in enumerate(zip(current,classes[i]))
                                 if k=='lane' and int(codec.tokens[t][2:]) in out_lanes),None)
                if escape_pos is not None:
                    result[note_starts[i]+escape_pos]=True
        # Also supervise exits from repeated structural shapes when the exact
        # lane/modifier text differs. The repeated prefix itself stays valid.
        # Lane digits are compared as one class: comparing raw tokens would
        # mark the first lane digit (it nearly always differs), not the place
        # where the configuration's structure actually departs.
        for period in range(1,min(max_period,i//2)+1):
            if shapes[i-2*period:i-period] != shapes[i-period:i]:
                continue
            if shapes[i] == shapes[i-period]:
                continue
            ours,theirs=classes[i],classes[i-period]
            divergence=next((j for j,(a,b) in enumerate(zip(ours,theirs)) if a!=b),
                            min(len(ours),len(theirs)))
            if current:
                result[note_starts[i]+min(divergence,len(current)-1)]=True
            break
    return result


def _structure_classes(cell,codec):
    """Per-token structure classes: lane digits collapse, everything else stays."""
    values=[];depth=0
    for token in cell:
        text=codec.tokens[token]
        char=text[2:] if text.startswith('N:') else text
        if char=='[':depth+=1
        if depth:values.append('duration')
        elif char in '12345678' and len(char)==1 and not (values and values[-1] in 'ABCDE'):values.append('lane')
        elif char in '12345678' and len(char)==1:values.append('area_index')
        else:values.append(char)
        if char==']':depth=max(0,depth-1)
    return tuple(values)
