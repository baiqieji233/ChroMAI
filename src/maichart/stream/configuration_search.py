"""Sample complete configurations before applying terminal style preferences.

Proposals come from the grammar-constrained model, without prefix potentials.
Importance resampling uses only the terminal preference, never the model
probability a second time. Duplicate proposals retain their probability mass.
This is a bounded approximation, not an exhaustive configuration search.
"""
from __future__ import annotations

import torch


def _expand_cache(cache, count):
    if cache is None:
        return None
    return [tuple(value.expand(count, *value.shape[1:]) if torch.is_tensor(value) else value
                  for value in layer) for layer in cache]


def _select_cache(cache, row, length):
    if cache is None:
        return None
    return [tuple(value[row:row+1, :length] if index < 2 else value[row:row+1]
                  if torch.is_tensor(value) else value
                  for index, value in enumerate(layer)) for layer in cache]


def resample_terminal(scores, generator, *, device):
    """All equal scores preserve the proposal law, including multiplicities."""
    values = torch.as_tensor(scores, dtype=torch.float32, device=device)
    if len(scores) == 1:
        return 0
    return int(torch.multinomial(values.softmax(0), 1, generator=generator))


@torch.inference_mode()
def propose_configurations(model, codec, logits, cache, memory, memory_mask, *,
        consumed, max_tokens, grammar, restricted, event_bias, temperature,
        candidates, generator, minimum_mass=0., timing_features=None, check=None):
    """Return one complete cell and its cache *before* consuming END.

Every branch has the same known NOTE clock. Nothing is committed here. A
context overflow is reported separately so callers can replay the event at a
resource boundary instead of silently preferring short configurations.
"""
    count = candidates if temperature > 0 else 1
    device = logits.device
    current = logits[None].expand(count, -1)
    caches = _expand_cache(cache, count)
    memory = memory.expand(count, *memory.shape[1:])
    memory_mask = memory_mask.expand(count, *memory_mask.shape[1:])
    if timing_features is not None:
        timing_features = timing_features.expand(count, *timing_features.shape[1:])
    chars = [(i, t[2:]) for i, t in enumerate(codec.tokens) if t.startswith('N:')]
    texts, tokens = ['']*count, [[] for _ in range(count)]
    running, complete = set(range(count)), []
    choices_cache = {}
    rejected, expansions = [], 0
    position = consumed
    while running:
        if position >= max_tokens:
            return None, {'reason': 'context_space', 'proposals': count, 'expanded_tokens': expansions}
        next_tokens = torch.full((count, 1), codec.pad_id, dtype=torch.long, device=device)
        for row in sorted(running):
            text = texts[row]
            if text not in choices_cache:
                base = [i for i, c in chars if grammar.can_continue(text+c)]
                legal = [i for i, c in chars if restricted.can_continue(text+c)]
                if grammar.is_complete(text): base.append(codec.end_id)
                if restricted.is_complete(text): legal.append(codec.end_id)
                choices_cache[text] = legal, base
            allowed, base = choices_cache[text]
            mass = (float(torch.exp(torch.logsumexp(current[row, allowed], 0)
                        - torch.logsumexp(current[row, base], 0)))
                    if minimum_mass and allowed and base and len(allowed)<len(base) else 1. if allowed else 0.)
            if not allowed or mass < minimum_mass:
                running.remove(row)
                rejected.append('low_probability_configuration_constraints' if allowed else 'configuration_constraints_exhausted')
                continue
            values = current[row, allowed]
            index = (int(values.argmax()) if temperature == 0 else
                     int(torch.multinomial((values/temperature).softmax(0), 1, generator=generator)))
            token = allowed[index]
            tokens[row].append(token); expansions += 1
            if token == codec.end_id:
                running.remove(row)
                if check is None or check(text):
                    complete.append((row, position, event_bias.terminal_score(text)))
                else:
                    rejected.append('reference_constraints_exhausted')
            else:
                texts[row] += codec.tokens[token][2:]
                next_tokens[row, 0] = token
        if not running:
            break
        kwargs = {'timing_features': timing_features} if timing_features is not None else {}
        output, caches = model.decode_step(next_tokens, position, memory, memory_mask, caches, **kwargs)
        current = output['logits'][:, -1].float()
        position += 1
    if not complete:
        return None, {'reason': rejected[-1] if rejected else 'configuration_constraints_exhausted',
                      'proposals': count, 'expanded_tokens': expansions}
    chosen = resample_terminal([item[2] for item in complete], generator, device=device)
    row, length, score = complete[chosen]
    return {'text': texts[row], 'tokens': tokens[row], 'cache': _select_cache(caches, row, length),
            'consumed': length}, {'reason': 'complete', 'proposals': count, 'accepted': len(complete),
                                'expanded_tokens': expansions, 'terminal_score': score}
