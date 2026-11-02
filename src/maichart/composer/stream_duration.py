"""Bounded acoustic interval reads for statically proven detail durations."""
from functools import lru_cache
import math

import torch
from torch import nn


@lru_cache(maxsize=65536)
def _duration_coefficients(atom):
    from maichart.generation.muri.parser import parse_markers
    values = []
    for bpm in (60., 120., 180.):
        try:
            markers, errors, _ = parse_markers(f'({bpm}){{4}}{atom},E')
        except (ValueError, TypeError, KeyError):
            return None
        sustained = [m for m in markers if m.get('type') in ('hold', 'touch_hold', 'touchhold', 'slide', 'wifi')]
        if errors or len(sustained) != 1:
            return None
        marker = sustained[0]
        if marker.get('type') in ('slide', 'wifi'):
            # Compound trajectories are deliberately outside this first contract.
            if len(marker.get('slide_segment_keys', [])) > 1:
                return None
            begin = marker.get('slide_trace_second', -1.)
        else:
            begin = marker.get('second', 0.)
        end = marker.get('end_second', -1.)
        origin = marker.get('second', 0.)
        if not all(math.isfinite(float(v)) for v in (begin, end, origin)) or begin < origin or end <= begin:
            return None
        values.append((float(begin-origin), float(end-origin)))
    coefficients = []
    for coordinate in range(2):
        a, b, c = (row[coordinate] for row in values)
        beats = (a-b)*2.
        seconds = a-beats
        if beats < -1e-7 or seconds < -1e-7 or abs(beats/3.+seconds-c) > 1e-6:
            return None
        coefficients.extend((max(0., beats), max(0., seconds)))
    return coefficients


def select_duration_context(context, indices):
    if context is None: return None
    return {key: value if key == 'cache' else value.index_select(0, indices)
            for key, value in context.items()}


def prepare_interval_clock(cache):
    """Prepare once per forward; clocks have no shape/slot dependence."""
    seconds = cache['seconds'].double()
    mask = cache['mask']
    lengths = mask.sum(-1).clamp_min(1)
    last = seconds.gather(1, (lengths-1)[:, None])[:, 0]
    stride = last.max().clamp_min(0)+2.
    offsets = torch.arange(seconds.size(0), device=seconds.device, dtype=seconds.dtype)*stride
    times = seconds.masked_fill(~mask, stride-1.)+offsets[:, None]
    return dict(lengths=lengths,last=last,offsets=offsets,times=times.flatten().contiguous(),width=seconds.size(1))


def _read_intervals(cache, queries, inverse, field='local'):
    """Search one flattened bank, avoiding a full-song clock copy per event."""
    clock=cache.get('_duration_clock')
    if clock is None:clock=prepare_interval_clock(cache)
    lengths,last,offsets,times=(clock[key] for key in ('lengths','last','offsets','times'))
    clipped = queries.double().clamp_min(0.)
    clipped = torch.minimum(clipped, last[inverse, None, None])
    shifted = clipped+offsets[inverse, None, None]
    flat = shifted.flatten().contiguous()
    right = torch.searchsorted(times, flat, right=True)
    base = (inverse*clock['width'])[:, None, None].expand_as(queries).flatten()
    upper = base+lengths[inverse, None, None].expand_as(queries).flatten()-1
    right = torch.minimum(right, upper)
    left = torch.maximum(right-1, base)
    lo, hi = times[left], times[right]
    mix = ((flat-lo)/(hi-lo).clamp_min(1e-8)).clamp(0, 1).float()
    bank = cache[field].flatten(0, 1)
    a, b = bank[left], bank[right]
    return (a+(b-a)*mix[:, None]).reshape(*queries.shape, bank.size(-1))


class DurationAudioScorer(nn.Module):
    def __init__(self, width, codec, *, feature_policy='legacy_v1'):
        super().__init__()
        if feature_policy not in ('legacy_v1', 'log_time_v2'):
            raise ValueError(f'Unknown duration feature policy: {feature_policy}')
        self.feature_policy = feature_policy
        self.score = nn.Linear(32 * 4 + 2, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self._tables = {}
        self._candidate_indices = {}
        self._catalog_candidates = 0
        self._supported_candidates = 0
        for shape, slots in enumerate(codec.shape_catalog):
            if not shape: continue
            for slot, (kind, head) in enumerate(slots):
                table = []
                for suffix in codec.detail_catalog[shape][slot]:
                    atom = str(head+1)+suffix
                    # Catalog regular forms are phase equivariant: times do not
                    # depend on the rotation. Unsupported forms remain untouched.
                    value = _duration_coefficients(atom) if kind in ('hold', 'slide') else None
                    table.append([*(value or [0., 0., 0., 0.]), float(value is not None)])
                self._catalog_candidates += len(table)
                self._supported_candidates += sum(bool(row[4]) for row in table)
                if not any(row[4] for row in table): continue
                name = f'duration_{shape}_{slot}'
                unique, inverse = torch.unique(torch.tensor(table, dtype=torch.float32), dim=0, return_inverse=True)
                self.register_buffer(name, unique, persistent=False)
                index_name = f'duration_index_{shape}_{slot}'
                self.register_buffer(index_name, inverse, persistent=False)
                self._tables[shape, slot] = name
                self._candidate_indices[shape, slot] = index_name

    def forward(self, model, logits, shape, slot, context):
        name = self._tables.get((shape, slot))
        if context is None or name is None: return logits
        # Logit autocast must not quantize physical interval boundaries to BF16.
        table = getattr(self, name).to(device=logits.device, dtype=torch.float32)
        second = context['second']
        spb = context['seconds_per_beat']
        begin = second[:, None] + spb[:, None]*table[None, :, 0] + table[None, :, 1]
        end = second[:, None] + spb[:, None]*table[None, :, 2] + table[None, :, 3]
        fractions = second.new_tensor([0., .125, .25, .375, .5, .625, .75, .875, 1.])
        queries = begin[..., None]+(end-begin)[..., None]*fractions
        cache = context['cache']
        if 'delta_keys' in cache:
            audio = _read_intervals(cache, queries, context['inverse'], field='delta_keys')
        else:
            source = cache['local'][..., :32]
            if source.size(-1) < 32:
                source = torch.nn.functional.pad(source, (0, 32-source.size(-1)))
            audio = _read_intervals({**cache, 'duration_audio_keys': source}, queries,
                                    context['inverse'], field='duration_audio_keys')
        # Stable intent selects acoustic dimensions across the entire phrase.
        attention = context['attention'][:, None, None, :]
        audio = audio * attention
        pooled = torch.cat((audio[..., :3, :].mean(-2), audio[..., 3:6, :].mean(-2),
                            audio[..., 6:, :].mean(-2), audio[..., -1, :]-audio[..., 0, :]), -1)
        time_features = torch.stack((begin-second[:, None], end-begin), -1)
        if self.feature_policy == 'log_time_v2':
            # Physical seconds previously entered an unbounded linear score:
            # a rare long hold could dominate both logits and weight gradients.
            # Smooth, strictly monotone compression retains exact interval
            # ordering without clipping durations or changing acoustic reads.
            time_features = time_features.sign()*torch.log1p(time_features.abs())
        features = torch.cat((pooled, time_features), -1)
        supported = table[None, :, 4].bool() & context['valid'][:, None] & (end <= context['end_second'][:, None])
        residual = self.score(features).squeeze(-1).masked_fill(~supported, 0.)
        indices = getattr(self, self._candidate_indices[shape, slot])
        return logits + residual.index_select(-1, indices).to(logits)

    def coverage(self):
        return {'supported': self._supported_candidates,
                'catalog_candidates': self._catalog_candidates,
                'literal_candidates_supported': False}
