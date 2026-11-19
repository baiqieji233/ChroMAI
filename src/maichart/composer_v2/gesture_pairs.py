"""Same-song comparisons of an ongoing rim gesture at matched authored onsets."""
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from maichart.event_timing import EventPoint
from . import motion_writing

PROBE_EVENTS = 3


def continuation(keys):
    a,b = keys[-2:]
    if rim_family(keys) == 'alternation':
        return (a,b,a)
    step = (b-a) % 8
    return tuple((b-1+n*step) % 8+1 for n in range(1,PROBE_EVENTS+1))

def rim_family(keys):
    """Four observable contacts, independent of subdivision and absolute starting pad."""
    if len(keys) != 4 or not all(keys):
        return None
    a, b, c, d = keys
    if a == b == c == d:
        return 'jack'
    if a == c and b == d:
        return 'alternation'
    steps = ((b-a) % 8, (c-b) % 8, (d-c) % 8)
    if steps in ((1, 1, 1), (7, 7, 7)):
        return 'sweep'
    return None


def inventory(song, chart):
    """Index clean examples once per chart, without forming a Cartesian product."""
    groups = defaultdict(lambda: ([], []))
    contacts = [motion_writing.contact(p.note_text) for p in song.points]
    keys = [key for _,key in contacts]
    for i in range(4, len(keys)-3):
        past = keys[i-4:i]
        family = rim_family(past)
        if family is None:
            continue
        beats = chart.beats[i-4:i+4]
        gaps = tuple(b-a for a,b in zip(beats,beats[1:]))
        # Stay inside one movement. A later re-entry is not a gesture-change label.
        if min(gaps) <= 0 or max(gaps) > 1:
            continue
        future = rim_family(keys[i:i+4])
        expected = [str(k) for k in continuation(past)]
        follows = [p.note_text for p in song.points[i:i+PROBE_EVENTS]] == expected
        if future == family and follows:
            side = 0
        elif ((future is not None and future != family) or
              (keys[i] == 0 and sum(k == 0 for k in keys[i:i+4]) >= 2)):
            # An isolated accent or a transposed copy of the same gesture is ambiguous.
            if follows:
                continue
            side = 1
        else:
            continue
        if any(song.points[j].note_text != str(keys[j]) for j in range(i-4,i)):
            continue
        start, cue, stop = chart.seconds[[i-4,i,i+3]]
        if chart.quality.covered(start,np.nextafter(stop,np.inf)) <= 0:
            continue
        sustains = chart._overlapping_sustains(start,cue)
        if len(sustains) and np.any(sustains[:,2] > start):
            continue
        clock = tuple(np.round(np.diff(chart.seconds[i-4:i+1]),3))
        signature = tuple(past), gaps[:4], clock
        groups[signature][side].append(i)
    # Every retained positive has at least one disjoint negative.
    pairs = []
    for (past,_,_), (positive,negative) in groups.items():
        if not negative:
            continue
        legal = [i for i in positive if negative[0] <= i-8 or negative[-1] >= i+8]
        if legal:
            pairs.append((past,legal,negative))
    return pairs


def training_item(dataset, song, chart, seed):
    if not hasattr(chart,'gesture_pairs'):
        chart.gesture_pairs = inventory(song,chart)
    if not chart.gesture_pairs:
        return None
    rng = np.random.default_rng(seed)
    past,positive,negative = chart.gesture_pairs[rng.integers(len(chart.gesture_pairs))]
    a = positive[rng.integers(len(positive))]
    left, right = bisect_right(negative,a-8), bisect_left(negative,a+8)
    k = int(rng.integers(left+len(negative)-right))
    b = negative[k if k < left else k+right-left]
    pads = continuation(past)
    items = []
    codec = dataset.codec
    offsets = [beat-chart.beats[a] for beat in chart.beats[a:a+PROBE_EVENTS]]
    for i in (a,b):
        cursor = chart.beats[i-1]
        if cursor-chart.beats[i-4] > dataset.history_beats:
            return None
        candidate = [EventPoint(chart.beats[i]+offset,str(key)) for offset,key in zip(offsets,pads)]
        if (candidate[-1].beat > chart.beats[i+3] or candidate[-1].beat >= song.end_beat
                or any(f'F:{p.beat % 1}' not in codec.ids for p in candidate)):
            return None
        prefix_length = 5+len(codec.encode_points(song.points[i-4:i]))
        if prefix_length > dataset.max_tokens//2 or prefix_length+len(codec.encode_points(candidate))-1 > dataset.max_tokens:
            return None
        item = dataset.build_stream(song,song.points[i-4:i],cursor,target=candidate,
                                    training=True,reference_start=None)
        ids = item['input_ids'].tolist()
        start = ids.index(codec.tgt_id)
        if ids[ids.index(codec.ctx_id)+1:ids.index(codec.ref_id)] != codec.encode_points(song.points[i-4:i]):
            return None
        queries = [pos+1 for pos in range(start,len(ids)) if ids[pos]==codec.phase_id]
        # Only the first onset is supervised under a wholly authored prefix.
        # No ordinary CE on any part of the hypothetical configuration.
        item['labels'][queries[0]:] = -100
        item['token_weights'][queries[0]:] = 0
        item['gesture_queries'] = queries
        if not item['token_weights'].any():
            return None
        for field in ('condition','condition_known'):
            item[field] = item[field].clone()
            item[field][1:] = 0
        item['curve_teacher_allowed'] = False
        dataset._writing_targets(item,chart,song,i,i+1,layout=False)
        items.append(item)
    # Identical native alternatives are evaluated in both musical contexts.
    tap = codec.ids['N:tap']
    choices = []
    for pad in pads:
        key = codec.ids[f'P:{pad}']
        after_pad = set(codec.allowed_atom_next([tap,key])) - {key}
        choices.append(((tap,codec.kind_ids), (key,codec.domains['<button>']),
                        (codec.end_note_id,after_pad), (codec.event_end_id,codec.kind_ids | {codec.event_end_id})))
    items[0]['gesture_peer'] = items[1]
    trust = min(chart.quality.covered(chart.seconds[i-4],np.nextafter(chart.seconds[i+3],np.inf)) for i in (a,b))
    items[0]['gesture_choices'] = choices, min(1.,trust)
    return items[0]


def expand(items):
    return [part for item in items for part in ((item,item['gesture_peer']) if 'gesture_peer' in item else (item,))]


def collate(items,batch):
    positions, domains, weights = [], [], []
    for row,item in enumerate(items):
        if 'gesture_choices' not in item:
            continue
        choices,trust = item['gesture_choices']
        for offset,part in enumerate(items[row:row+2]):
            for q, event_choices in zip(part['gesture_queries'],choices,strict=True):
                for step,(chosen,domain) in enumerate(event_choices):
                    positions.append((row+offset,q+step))
                    domains.append([chosen,*sorted(domain-{chosen})])
        weights.append(.25*trust*min(float(p['token_weights'].sum()*p['sample_weight']) for p in items[row:row+2]))
    if weights:
        batch['gesture_positions'] = torch.tensor(positions,dtype=torch.long)
        batch['gesture_domains'] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(v,dtype=torch.long) for v in domains],batch_first=True,padding_value=-1)
        batch['gesture_weights'] = torch.tensor(weights,dtype=torch.float32)


def loss(logits,batch):
    rows,pos = batch['gesture_positions'].unbind(1)
    ids = batch['gesture_domains']
    subset = logits[rows[:,None],pos[:,None],ids.clamp_min(0)].float().masked_fill(ids<0,-torch.inf)
    logp = (subset[:,0]-subset.logsumexp(1)).reshape(-1,2,PROBE_EVENTS,4).sum(-1).mean(-1)
    return (F.softplus(logp[:,1]-logp[:,0])*batch['gesture_weights']).sum()
