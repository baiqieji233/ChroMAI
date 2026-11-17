"""Relative musical evidence for candidates that have already passed the native guard."""
from bisect import bisect_left
import math

import numpy as np
import torch

from . import motion_writing


def prepare_song(sampler, dataset, song):
    """Only the small audio-only writing readout runs here, once per song."""
    if hasattr(song, '_rank_plan'):
        return
    song._rank_plan = None
    head = getattr(sampler.model, 'local_writing', None)
    if head is None:
        return
    stop = math.floor(float(song.end_beat))
    if stop < 2:
        return
    predictions = []
    with torch.inference_mode():
        for start in range(0, stop, 256):
            anchors = list(range(start, min(start+256, stop)))
            batch = {k:v.to(sampler.device) for k,v in dataset.writing_features(song, anchors).items()}
            batch['writing_owner'] = torch.zeros(len(anchors), device=sampler.device, dtype=torch.long)
            batch['condition'] = sampler.batch['condition']
            logits = head.plan(sampler.cache['song'], batch)['writing_logits']
            predictions.append(logits[:, 0].float().sigmoid())
        predicted = torch.cat(predictions).cpu().numpy()
    song._rank_plan = head.fields, predicted, predicted[:-1].mean(0)


def _accent(points, song):
    if len(points) < 3:
        return 0.
    accents = (motion_writing.CONTACT.index('chord'),motion_writing.CONTACT.index('break_head'))
    marked = np.array([any(i in motion_writing.contact(p.note_text)[0] for i in accents) for p in points])
    if not marked.any() or marked.all():
        return 0.
    times = song.clock.at_floats(np.asarray([float(p.beat) for p in points]))
    # The same transient readout for marked and unmarked heads.
    frame_t, flux = song._phrase_flux
    strength = np.interp((times[:,None] + np.array([-.01,0,.01])).ravel(),
                         frame_t, flux).reshape(-1,3).max(1)
    scale = strength.max()
    return float((strength[marked].mean()-strength[~marked].mean())/scale) if scale > 1e-8 else 0.


def _agreement(prefix, points, song, start, stop):
    plan = getattr(song, '_rank_plan', None)
    if plan is None:
        return 0., 0
    fields, predicted, background = plan
    anchor_start = math.floor(start)
    first = bisect_left(prefix,anchor_start,key=lambda point: point.beat)
    earlier = [p for p in prefix[max(0,first-2):] if not points or p.beat < points[0].beat]
    sequence = [*earlier, *points]
    beats = [p.beat for p in sequence]
    tables = motion_writing.sums(sequence)
    offsets = {name: i for i, name in enumerate(fields)}
    parts = []
    for anchor in range(anchor_start, min(math.floor(stop)-1, len(predicted)-1), 2):
        lo, hi = bisect_left(beats, anchor), bisect_left(beats, anchor+2)
        if lo == hi:
            continue
        counts, denom = motion_writing.ordered_counts(tables, lo, hi)
        if 'no_new_head' in offsets:
            names = motion_writing.ORDER_FIELDS
        else:
            n = len(motion_writing.EVENT_FIELDS)
            counts, denom = counts[:n]+counts[n:], denom[:n]+denom[n:]
            names = [f'kind_{x}' if x in motion_writing.CONTACT[:5] else x for x in motion_writing.EVENT_FIELDS]
        indices = [i for i, name in enumerate(names) if name in offsets and denom[i] > 0]
        if not indices:
            continue
        cols = [offsets[names[i]] for i in indices]
        observed = counts[indices] / denom[indices]
        # A flat predicted arrangement casts no vote. Compare each candidate to
        # the same song baseline, rather than favouring globally frequent moves.
        improvement = (background[cols]-observed)**2 - (predicted[anchor,cols]-observed)**2
        if 'no_new_head' in offsets:
            feature_ids=np.asarray(indices)%len(motion_writing.EVENT_FIELDS)
            families=np.searchsorted((len(motion_writing.CONTACT),len(motion_writing.CONTACT)+len(motion_writing.MOVEMENT)),
                                     feature_ids,side='right')
            parts.append(float(np.mean([improvement[families==f].mean() for f in np.unique(families)])))
        else:
            parts.append(float(improvement.mean()))
    return (float(np.mean(parts)), len(parts)) if parts else (0., 0)


def score(prefix, points, song, start, stop, timing, *, policy, complete_span=True):
    inside = [p for p in points if start <= p.beat < stop]
    # Include the committed boundary head if it belongs to this musical interval.
    arrangement, spans = (_agreement(prefix, inside, song, start, stop)
                          if complete_span and policy=='music_plan_v1' else (0., 0))
    accent = _accent(inside, song)
    value = .5 + .45 * (timing-.5) + .15 * accent + .40 * arrangement
    return value, dict(ranking_policy=policy, timing_support=timing,
                       accent_alignment=accent, local_motion_agreement=arrangement, comparable_intervals=spans)
