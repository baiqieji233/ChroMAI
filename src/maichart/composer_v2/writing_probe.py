"""Bounded, teacher-forced evaluation. Oracle plans never enter generation."""
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from . import local_writing


def compare(model, dataset, song, song_index, item, batch):
    head = model.local_writing
    local_writing.add_targets(item, dataset._configuration(song_index), song,
                              model.config.future_configuration_policy)
    targets, weights = item['writing_targets'], item['writing_weights']
    valid = weights.gt(0)
    cache = model.encode_audio(batch)
    predicted = cache['writing_logits'].float().sigmoid()
    ordinary = model.decode(batch, cache, incremental=False).float()
    end = max(1, int(np.floor(float(song.end_beat))))
    anchors = np.asarray(item['anchors'])
    shifted = (anchors + max(1, end//2)) % end
    alternate = dataset.writing_features(song, shifted)
    alternate.update(writing_owner=batch['writing_owner'], condition=batch['condition'])
    elsewhere = head.plan(cache['song'], alternate)['writing_logits'].float().sigmoid()
    # Compare on this chart's real supervised features at the same anchors only.
    w = weights * torch.as_tensor((anchors >= 0) & (anchors < end) & (anchors != shifted))[:,None,None]
    errors = Counter(mass=float(w.sum()),
                     local=float(((predicted-targets).square()*w).sum()),
                     other_music=float(((elsewhere-targets).square()*w).sum()))
    labels = batch['labels']
    tiles = torch.as_tensor(batch['tile_ids_cpu'])
    covered = valid.any(-1).any(-1)
    positive = labels.ne(-100) & batch['token_weights'].gt(0) & covered[tiles.clamp_min(0)] & tiles.ge(0)
    layout = positive & model.writing_query_tokens[batch['input_ids']] & ~model.time_tokens[labels.clamp_min(0)]
    masks = {'layout': layout}
    if hasattr(head, 'timing'):
        time_known = valid[:, :, local_writing.TIME_COLUMNS].any(-1).any(-1)
        masks['entry'] = positive & time_known[tiles.clamp_min(0)] & model.writing_time_tokens[batch['input_ids']]
    oracle = torch.where(valid, targets, predicted)
    geometry_logits = cache.get('slide_geometry_logits')
    geometry = geometry_logits.float().sigmoid() if geometry_logits is not None else None
    # Hold the separately learned slide geometry constant in both probes.
    disabled = ({'writing_context': torch.zeros_like(cache['writing_context'])} if geometry is None
                else head.project(torch.zeros_like(predicted), geometry=geometry))
    modes = {'predicted': None, 'authored_hint': head.project(oracle, geometry=geometry),
             'disabled': disabled}
    metrics = {}
    for name, change in modes.items():
        logits = ordinary if change is None else model.decode(batch, {**cache, **change}, incremental=False).float()
        losses = F.cross_entropy(logits.flatten(0,1), labels.flatten(), ignore_index=-100,
                                 reduction='none').reshape_as(labels)
        for role, mask in masks.items():
            metrics[f'{name}_{role}'] = Counter(count=int(mask.sum()), nll=float(losses[mask].sum()))
    return ordinary, errors, metrics


def summarize(errors, metrics, windows):
    mass = errors['mass']
    return dict(windows=windows, feature_weight=mass,
                plan_brier={k:errors[k]/mass if mass else None for k in ('local','other_music')},
                execution={k:dict(tokens=v['count'], mean_nll=v['nll']/v['count'] if v['count'] else None)
                           for k,v in metrics.items()},
                scope='teacher_forced_same_prefix; authored_hint_is_diagnostic_only')
