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
                    consumed, width=4, complexity=.2, timing_features=None,
                    temperature=0., generator=None, canonical_only=False, adaptive=False,
                    candidate_cost=None):
    initial_width, expansions = width, 0
    while True:
        value, status = _search_position(model, codec, grammar, root_logits, root_cache,
            memory, mask, consumed=consumed, width=width, complexity=complexity,
            timing_features=timing_features, temperature=temperature, generator=None,
            canonical_only=canonical_only, candidate_cost=candidate_cost)
        expansions += status['expanded_hypotheses']
        bound = status['discarded_prefix_score_bound']
        best = status['best_complete_score']
        # A pruned prefix with a higher probability upper bound may beat the
        # completed fraction. Widen only these uncertain searches, never as a
        # function of gap size or audio loudness. Resource budget stays bounded.
        if not adaptive or width >= 32 or bound is None or (best is not None and bound <= best-12*temperature):
            break
        width = min(32, width*2)
    ranked = status['candidates']
    if ranked and temperature > 0:
        weights = torch.tensor([row['score'] for row in ranked], device=memory.device)
        choice = int(torch.multinomial(torch.softmax((weights-weights.max())/temperature, 0),
                                     1, generator=generator))
        selected = ranked[choice]
        value = selected['beat']
        status['complete_score'] = selected['score']
        status['selected_log_probability'] = selected['log_probability']
    status.update(initial_beam_width=initial_width, expanded_hypotheses=expansions,
                  search_certified=bound is None or (best is not None and bound <= best),
                  candidates=status['candidates'][:8])
    return value, status


def _search_position(model, codec, grammar, root_logits, root_cache, memory, mask, *,
                     consumed, width, complexity, timing_features,
                     temperature, generator, canonical_only, candidate_cost):
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
    completed = {}
    expanded = 0
    discarded_bound = None
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
                    if canonical_only and str(Fraction(prefix)) != prefix:
                        continue
                    rank = cumulative - complexity * position_complexity_cost(prefix, grammar.lower)
                    if candidate_cost is not None:
                        penalty = float(candidate_cost(Fraction(prefix)))
                        if not math.isfinite(penalty) or penalty < 0:
                            raise ValueError('Position candidate cost must be finite and nonnegative')
                        rank -= penalty
                    completed[prefix] = (rank, cumulative)
                    if best is None or rank > best[0]:
                        best = (rank, prefix, cumulative)
                elif depth < grammar.characters:
                    candidates.append((cumulative, prefix + codec.tokens[token][2:], index, token))
        if not candidates:
            break
        candidates.sort(key=lambda row: row[0], reverse=True)
        if len(candidates) > width:
            score = candidates[width][0]
            discarded_bound = score if discarded_bound is None else max(discarded_bound, score)
        candidates = candidates[:width]
        # Every future probability/cost is <=0, so no unfinished hypothesis can
        # improve a completed candidate above this upper bound.
        if best is not None and candidates[0][0] <= best[0] - 12 * temperature:
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
        if isinstance(prediction, dict):
            prediction = prediction['logits']
        logits = prediction[:, -1]
        expanded += batch_size
    ranked = sorted(completed.items(), key=lambda row: row[1][0], reverse=True)
    selected = None if best is None else best[1]
    return selected, {'expanded_hypotheses': expanded, 'beam_width': width,
        'discarded_prefix_score_bound': discarded_bound,
        'best_complete_score': None if best is None else best[0],
        'complete_score': None if selected is None else completed[selected][0],
        'selected_log_probability': None if selected is None else completed[selected][1],
        'candidates': [{'beat': text, 'score': rank, 'log_probability': probability}
                       for text, (rank, probability) in ranked]}
