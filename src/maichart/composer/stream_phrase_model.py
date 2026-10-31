"""Continuous phrase intentions from audio and completed causal references.

Targets are deliberately absent from this module's forward input selection.
Audio sampling is bounded (nine points per interval), reusing encoded banks.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .stream_phrase import PHRASE_FEATURE_DIM


class CausalPhraseContext(nn.Module):
    def __init__(self, width, condition_dim):
        super().__init__()
        model_width, width = width, min(width, 128)
        self.phrase_width = width
        self.audio = nn.Sequential(nn.Linear(32 * 4, width), nn.GELU(), nn.LayerNorm(width))
        self.writing = nn.Sequential(nn.Linear(PHRASE_FEATURE_DIM, width), nn.GELU(), nn.LayerNorm(width))
        self.shared = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.style = nn.Sequential(nn.Linear(condition_dim, width), nn.LayerNorm(width))
        self.query = nn.Linear(width * 2, width)
        self.key = nn.Linear(width, width)
        self.null_reference = nn.Parameter(torch.zeros(width))
        self.intent = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width))
        self.progress = nn.Sequential(nn.Linear(PHRASE_FEATURE_DIM, width), nn.GELU(), nn.LayerNorm(width))
        self.predict = nn.Linear(width, PHRASE_FEATURE_DIM)
        self.intent_output = nn.Linear(width, model_width, bias=False)
        self.progress_output = nn.Linear(width, model_width, bias=False)
        self.style_output = nn.Linear(width, model_width, bias=False)
        self.tracking = nn.Linear(width, width)
        self.tracking_acoustic = nn.Linear(width, 32)
        self.tracking_output = nn.Linear(width, model_width, bias=False)
        for layer in (self.intent_output, self.progress_output, self.style_output, self.tracking_output):
            nn.init.zeros_(layer.weight)

    def _music(self, model, cache, inverse, start, end):
        fractions = start.new_tensor([0., .125, .25, .375, .5, .625, .75, .875, 1.])
        positions = start[..., None] + (end-start).clamp_min(0)[..., None]*fractions
        if 'delta_keys' in cache:
            values = model._read_audio(cache, positions, inverse, clock='beats', field='delta_keys')
        else:
            # Slice the shared bank before gathering any phrase intervals.
            source = cache['local'][..., :32]
            if source.size(-1) < 32: source = F.pad(source, (0, 32-source.size(-1)))
            compact = {**cache, 'phrase_audio_keys': source}
            values = model._read_audio(compact, positions, inverse, clock='beats', field='phrase_audio_keys')
        # Preserve coarse temporal order instead of collapsing the phrase to a mean.
        pooled = torch.stack((values[..., :3, :].mean(-2), values[..., 3:5, :].mean(-2),
                              values[..., 5:7, :].mean(-2), values[..., 7:, :].mean(-2)), -2)
        return self.audio(pooled.flatten(-2))

    def forward(self, model, cache, inverse, batch, tracking_strength=0.):
        start, end = batch['phrase_start_beat'], batch['phrase_end_beat']
        features = batch['phrase_memory_features'].float()
        if features.size(-1) != PHRASE_FEATURE_DIM or features.size(-2) > 4:
            raise ValueError('Phrase memory requires PHRASE_FEATURE_DIM features and at most four references')
        memory_start, memory_end = batch['phrase_memory_start_beat'], batch['phrase_memory_end_beat']
        valid = batch['phrase_memory_mask'].bool() & (memory_end <= start[..., None]) & (memory_end > memory_start)
        anchor = self._music(model, cache, inverse, start, end)
        music = self._music(model, cache, inverse, memory_start, memory_end)
        # Mask before normalization so masked recovery rows cannot poison the path.
        writing = self.writing(torch.nan_to_num(features).masked_fill(~valid[..., None], 0.))
        references = self.shared(torch.cat((music, writing), -1))
        references = references.masked_fill(~valid[..., None], 0.)
        null = self.null_reference.expand(*anchor.shape[:-1], 1, anchor.size(-1))
        references = torch.cat((null, references), -2)
        style = self.style(batch['condition'].float())[:, None].expand_as(anchor)
        query = self.query(torch.cat((anchor, style), -1))
        scores = (query[..., None, :] * self.key(references)).sum(-1)/math.sqrt(anchor.size(-1))
        allowed = torch.cat((torch.ones_like(valid[..., :1]), valid), -1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        weights = scores.float().softmax(-1).to(references)
        retrieved = (weights[..., None] * references).sum(-2)
        intent = self.intent(torch.cat((anchor, retrieved, style), -1))
        progress = self.progress(torch.nan_to_num(batch['phrase_progress'].float()))
        residual = self.intent_output(intent) + self.progress_output(progress) + self.style_output(style)
        # This attends to encoded acoustic subspaces, not separated instruments.
        strength = torch.as_tensor(tracking_strength, device=anchor.device, dtype=anchor.dtype)
        if strength.ndim == 1: strength = strength[:, None]
        if strength.ndim == 2: strength = strength[..., None]
        tracked = anchor * self.tracking(intent).sigmoid()
        residual = residual + strength * self.tracking_output(tracked)
        prediction = self.predict(intent)
        representation = self.shared(torch.cat((anchor, self.writing(prediction)), -1))
        return residual, dict(phrase_intent=intent, phrase_reference_weights=weights,
            phrase_prediction=prediction, phrase_repr=representation,
            phrase_reference_repr=references, phrase_reference_mask=allowed)


def phrase_auxiliary_loss(output, batch):
    """One weighted feature loss; no duplicate event budget or recovery labels."""
    prediction = output['phrase_prediction'].float()
    if 'target_phrase_features' not in batch:
        return prediction.sum()*0.
    target = batch['target_phrase_features'].float()
    mask = batch['target_phrase_mask'].bool()
    weight = batch['target_phrase_weight'].float()
    if mask.ndim == prediction.ndim-1: mask = mask[..., None]
    if weight.ndim == prediction.ndim-1: weight = weight[..., None]
    active = mask & batch['step_mask'].bool()[..., None] & torch.isfinite(target)
    mass = torch.nan_to_num(weight, nan=0., posinf=0., neginf=0.).clamp_min(0).expand_as(prediction)
    mass = mass.masked_fill(~active, 0.)
    target = torch.nan_to_num(target).masked_fill(~active, 0.)
    errors = F.smooth_l1_loss(prediction, target, reduction='none')
    # Dataset weights already split a phrase's budget over its repeated queries.
    # Renormalizing by their sum would cancel that split (and chart balancing).
    return (errors * mass).sum()/max(1, prediction.size(0)*prediction.size(-1))
