"""Sample complete configurations before applying terminal style preferences.

Proposals come from the grammar-constrained model: one row from the model law
alone, the others with bounded prefix potentials added. Importance resampling
uses the terminal preference plus the exact log-ratio between the model law
and the proposal mixture, so the model probability is never counted twice and
guided rows only change which configurations are explored. Duplicate proposals
retain their probability mass. This is a bounded approximation, not an
exhaustive configuration search.
"""
from __future__ import annotations

import math

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
        candidates, generator, minimum_mass=0., timing_features=None, check=None,
        exploration_temperature=0.):
    """Return one complete cell and its cache *before* consuming END.

Every branch has the same known NOTE clock. Nothing is committed here. A
context overflow is reported separately so callers can replay the event at a
resource boundary instead of silently preferring short configurations.

Proposal law: row 0 samples the model alone; the other rows add the bounded
prefix potentials of event_bias. When the model is nearly certain of a single
Tap, guided rows still explore the Holds, Slides and chords the terminal
preference needs to compare. With active repeat pressure, one guided row may
use a higher exploration temperature; all paths are scored under all proposal
laws. Importance weights use the actual mixture density, so resampling targets
exp(terminal) times the model law exactly as before (finite candidate error remains).
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
    # A single proposal cannot be reweighted, so it is the guided one.
    guided = [count == 1 or row > 0 for row in range(count)]
    explore=bool(count>1 and exploration_temperature>temperature and
                 getattr(event_bias,'exploration_requested',False))
    explore_row=count-1 if explore else -1
    native_logp, guided_logp = [0.]*count, [0.]*count
    exploration_logp=[0.]*count
    running, complete = set(range(count)), []
    choices_cache = {}
    rejected, expansions = [], 0
    position = consumed
    scale = max(float(temperature), 1e-6)
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
            token_text=[None if token==codec.end_id else codec.tokens[token][2:] for token in allowed]
            native = current[row, allowed] + torch.as_tensor(event_bias.token_bias(text, token_text),device=device,dtype=current.dtype)
            steering = torch.as_tensor(event_bias.biases(text, token_text), device=device, dtype=current.dtype)
            nudged = native + steering
            values = nudged if guided[row] else native
            row_temperature=exploration_temperature if row==explore_row else temperature
            index = (values.argmax() if row_temperature == 0 else
                     torch.multinomial((values/row_temperature).softmax(0), 1, generator=generator).squeeze(0))
            # One readback instead of synchronizing the index and both law
            # scores separately. Sampling law and RNG draws stay unchanged.
            measures=[index.float(),(native/scale).log_softmax(0)[index],(nudged/scale).log_softmax(0)[index]]
            if explore:measures.append((nudged/exploration_temperature).log_softmax(0)[index])
            measured=torch.stack(measures).cpu().tolist()
            picked,native_score,guided_score=measured[:3]
            index=int(picked)
            native_logp[row] += native_score
            guided_logp[row] += guided_score
            if explore:exploration_logp[row]+=measured[3]
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
    weights = [score + _mixture_correction(native_logp[row], guided_logp[row], count,
                   exploration_logp[row] if explore else None)
               for row, _, score in complete]
    chosen = resample_terminal(weights, generator, device=device)
    row, length, score = complete[chosen]
    from .repetition import describe
    structures={describe(texts[i])[0] for i,_,_ in complete if describe(texts[i]) is not None}
    repeat_penalty=getattr(event_bias,'repeat_penalty',lambda text:0.)
    all_repeated=bool(getattr(event_bias,'exploration_requested',False)
                      and all(repeat_penalty(texts[i])>0 for i,_,_ in complete))
    return {'text': texts[row], 'tokens': tokens[row], 'cache': _select_cache(caches, row, length),
            'consumed': length}, {'reason': 'complete', 'proposals': count, 'accepted': len(complete),
                                'expanded_tokens': expansions, 'terminal_score': score,
                                'guided_proposals': sum(guided), 'chosen_guided': bool(guided[row]),
                                'exploration_proposals':int(explore),'candidate_structures':len(structures),
                                'all_candidates_repeat':all_repeated}


def _mixture_correction(native, guided, count, exploration=None):
    """log p(cell)-log q_mix(cell), for two or three actual proposal laws."""
    if count <= 1:
        return 0.
    if exploration is not None:
        terms=[native-math.log(count),exploration-math.log(count)]
        if count>2:terms.append(guided+math.log((count-2.)/count))
        top=max(terms)
        return native-(top+math.log(sum(math.exp(value-top) for value in terms)))
    native_share = math.log(1./count)
    guided_share = math.log((count-1.)/count)
    a, b = native_share+native, guided_share+guided
    top = max(a, b)
    return native-(top+math.log(math.exp(a-top)+math.exp(b-top)))
