"""Bounded whole-Fraction lookahead using batched causal decoder caches."""
from __future__ import annotations

from fractions import Fraction
import math

import torch


def position_complexity_cost(text, lower):
    try:
        value = Fraction(text)
        delta = value - lower
        return min(12., math.log2(value.denominator) + .5 * math.log2(delta.denominator))
    except (ValueError, ZeroDivisionError):
        return 0.


def _select_cache(caches, indices):
    if caches is None:
        return None
    return [tuple(value.index_select(0, indices) if value is not None else None
                  for value in layer) for layer in caches]


def search_position(model, codec, grammar, root_logits, root_cache, memory, mask, *,
                    consumed, width=4, complexity=.2, timing_features=None):
    """Return the best complete Fraction, never a nearest musical grid.

    Score full token probabilities before grammar renormalization: forcing a
    low-probability denominator after an overshot numerator must not make that
    path look certain. All hypotheses at a depth share one batched CUDA call.
    Search caches are speculative and never replace the committed caller cache.
    """
    prefixes = ['']
    scores = [0.]
    logits = root_logits[None]
    caches = root_cache
    best = None
    expanded = 0
    for depth in range(grammar.characters + 1):
        probabilities = logits.float().log_softmax(-1)
        # One transfer per depth, rather than a host sync per candidate token.
        log_probs = probabilities.detach().cpu().tolist()
        candidates = []
        for index, (prefix, score) in enumerate(zip(prefixes, scores)):
            for token in grammar.allowed(prefix, codec):
                probability = log_probs[index][token]
                if not math.isfinite(probability):
                    continue
                cumulative = score + probability
                if token == codec.note_id:
                    rank = cumulative - complexity * position_complexity_cost(prefix, grammar.lower)
                    if best is None or rank > best[0]:
                        best = (rank, prefix, cumulative)
                elif depth < grammar.characters:
                    candidates.append((cumulative, prefix + codec.tokens[token][2:], index, token))
        if not candidates:
            break
        candidates.sort(key=lambda row: row[0], reverse=True)
        candidates = candidates[:width]
        # Every future probability/cost is <=0, so no unfinished hypothesis can
        # improve a completed candidate above this upper bound.
        if best is not None and candidates[0][0] <= best[0]:
            break
        if consumed+depth >= model.config.max_tokens:
            break
        indices = torch.tensor([row[2] for row in candidates], device=memory.device)
        tokens = torch.tensor([[row[3]] for row in candidates], device=memory.device)
        prefixes = [row[1] for row in candidates]
        scores = [row[0] for row in candidates]
        cache_batch = _select_cache(caches, indices)
        batch_size = len(candidates)
        batch_memory = memory.expand(batch_size, -1, -1)
        batch_mask = None if mask is None else mask.expand(batch_size, -1)
        kwargs = ({'timing_features':timing_features.expand(batch_size,-1,-1)}
                  if timing_features is not None else {})
        prediction, caches = model.decode_step(tokens, consumed + depth, batch_memory, batch_mask, cache_batch,**kwargs)
        logits = prediction[:, -1]
        expanded += batch_size
    return (None if best is None else best[1]), {'expanded_hypotheses': expanded,
        'beam_width': width, 'complete_score': None if best is None else best[0]}
