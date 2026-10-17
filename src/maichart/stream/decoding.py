"""Bounded continuous decoding over one immutable audio encoding."""
from __future__ import annotations

from fractions import Fraction
import json
import math
import time

import numpy as np
import torch

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.form.rational_audio import beat_at_second
from maichart.form.rational_generate import PositionGrammar, _spacing_beat_gap
from maichart.form.rational_position_search import search_position
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.sustain_state import SustainStateIndex
from .memory import StreamMemoryBank
from .timing import CanonicalPositionGrammar, interval_consistency_cost
from .context import prefix_action_state
from .constraints import BasicConfigurationGuard
from .reference_guard import ReferenceCandidateGuard


def validate_options(args):
    for name in ('temperature', 'position_temperature'):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0 <= value <= 3:
            raise ValueError(f'{name} 必须是 0–3 的有限数值')
    for name in ('first', 'level'):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f'{name} 必须是有限数值')
    if args.bpm is not None and (not math.isfinite(float(args.bpm)) or args.bpm <= 0):
        raise ValueError('BPM 必须是正的有限数值')
    for name, low, high in (('max_denominator', 1, 4096),
                            ('max_interval_denominator', 0, 4096),
                            ('max_events', 0, 1_000_000), ('seed', 0, 2**63-1),
                            ('position_beam_size', 1, 8)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 {low}–{high} 的整数')
    if isinstance(args.reference_check_attempts,bool) or not isinstance(args.reference_check_attempts,int) or not 0 <= args.reference_check_attempts <= 32:
        raise ValueError('reference_check_attempts 必须是 0–32 的整数')
    if isinstance(args.max_button_heads, bool) or not isinstance(args.max_button_heads, int) or not 0 <= args.max_button_heads <= 8:
        raise ValueError('max_button_heads 必须是 0–8 的整数')
    if args.hold_protection not in {'on', 'off'}:
        raise ValueError('hold_protection 必须为 on 或 off')
    if args.trim_exact_silence not in {'on','off'}:
        raise ValueError('trim_exact_silence 必须为 on 或 off')
    if args.generation_end_sec is not None and (not math.isfinite(args.generation_end_sec) or args.generation_end_sec <= 0):
        raise ValueError('generation_end_sec 必须是正的有限秒数')
    if not math.isfinite(args.min_spacing_ms) or not 0 <= args.min_spacing_ms <= 1000:
        raise ValueError('min_spacing_ms 必须是 0–1000 的有限数值')
    for name in ('style_balance_strength', 'local_chord_strength', 'physical_feasibility_strength'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 3:
            raise ValueError(f'{name} 必须是 0–3 的有限数值')
    for name, high in (('position_complexity', 2.), ('early_eos_penalty', 8.),
                       ('position_interval_strength', 8.)):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= high:
            raise ValueError(f'{name} 必须是 0–{high:g} 的有限数值')


def sample_token(logits, allowed, temperature, generator):
    """Zero temperature is greedy for every grammar state, including EOS."""
    if not allowed:
        raise ValueError('No legal stream token continuation')
    values = logits[allowed]
    if temperature == 0:
        return allowed[int(values.argmax())]
    index = int(torch.multinomial(torch.softmax(values/temperature, dim=-1), 1, generator=generator))
    return allowed[index]


@torch.inference_mode()
def compose_stream(model, codec, song, audio_values, audio_times, condition, atoms, args):
    """Return complete committed events plus an explicit termination status.

    Only an event boundary may rotate the resource context. All self-attention
    caches refer to exactly the consumed prefix; cross-attention memory and its
    history embeddings stay identical for the full cache lifetime.
    """
    device = next(model.parameters()).device
    model.eval()
    audio = audio_values[None].to(device)
    control = condition.to_vector(device=device)[None]
    audio_mask = torch.zeros(audio.shape[:2], dtype=torch.bool, device=device)
    encoded_audio = model.encode_audio(audio, control, audio_padding_mask=audio_mask)
    base_projection = model.audio_projection(audio)[0]
    bank = StreamMemoryBank(max_entries=max(256, model.config.max_stream_memory),
                            max_retrieval=model.config.max_stream_memory)
    guidance = TypeGuidance(atoms, condition, [], planner_strength=0.,
        style_strength=args.style_balance_strength, local_chord_strength=args.local_chord_strength,
        physical_strength=args.physical_feasibility_strength)
    grammar = guidance.grammar
    hard_guard = BasicConfigurationGuard(grammar, song.tempo, song.first,
        max_button_heads=args.max_button_heads, protect_holds=args.hold_protection == 'on',
        max_action_end=getattr(args,'effective_end_sec',song.audio.duration_sec))
    cell_grammar = None
    reference_guard = ReferenceCandidateGuard(song.tempo,song.first) if args.reference_check_attempts else None
    note_bans,note_choices = {},[]
    note_attempts = 0
    chars = [(index, text[2:]) for index, text in enumerate(codec.tokens) if text.startswith('N:')]
    points, ids, history_tokens = [], [codec.bos_id], []
    previous = Fraction(-1)
    origin = max(Fraction(0), beat_at_second(0., song.tempo, song.first))
    end = beat_at_second(getattr(args,'effective_end_sec',song.audio.duration_sec), song.tempo, song.first)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    position = None
    state, spelling, header_start = 'event', '', 0
    cache, consumed = None, 0
    rotations = emitted_tokens = canonicalizations = 0
    event_bias = None
    interval_prediction = None
    searches, search_expansions, search_trace, event_trace = 0, 0, [], []
    started = time.perf_counter()
    characters = max(4, len(str(math.ceil(end*args.max_denominator)))
                     + 1 + len(str(args.max_denominator)))

    def audio_index(beat):
        second = float(seconds_at_beat(max(Fraction(0), beat), song.tempo, song.first))
        right = int(np.searchsorted(audio_times, second))
        choices = [i for i in (right-1, right) if 0 <= i < len(audio_times)]
        return min(choices, key=lambda i: abs(float(audio_times[i])-second)) if choices else 0

    def refresh_context():
        action = audio.new_zeros((1, 1, 50))
        action_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
        if points:
            index = SustainStateIndex(points, song.tempo, song.first)
            states, padding, _ = prefix_action_state(index, points, len(points))
            action, action_mask = states[None].to(device), padding[None].to(device)
        history = torch.tensor([history_tokens], dtype=torch.long, device=device) if history_tokens else None
        summary, summary_mask = (model.summarize_tokens(history, return_padding_mask=True)
                                 if history is not None else (None, None))
        # Training supplies summarize_tokens(history) only. The old retrieval
        # vectors (raw token embedding + audio projection) were never trained as
        # decoder memory and could replace that summary entirely after a roll.
        # Keep the causal archive for diagnostics, but use the trained summary.
        memory, padding = model.encode(audio, control, audio_padding_mask=audio_mask,
            action_state=action, action_state_padding_mask=action_mask,
            stream_memory=summary, stream_memory_padding_mask=summary_mask, encoded_audio=encoded_audio)
        return model.prepare_decode_memory(memory, padding, history=history)

    def retry_note():
        nonlocal ids,cache,consumed,spelling,note_choices,note_attempts
        if note_attempts >= args.reference_check_attempts:
            return False
        # Ban one uncertain decision from this rejected candidate. All earlier
        # committed events and the sampled Fraction stay intact.
        choices = [(margin,prefix,token) for margin,prefix,token in note_choices
                   if token not in note_bans.get(prefix,set())]
        if not choices:
            return False
        _,prefix,token = min(choices,key=lambda choice:choice[0])
        note_bans.setdefault(prefix,set()).add(token)
        body = header_start+len(codec.position(position))
        ids = ids[:body]
        # Retain speculative cross K/V, rewinding self K/V to before NOTE.
        if isinstance(cache,list):
            cache = [(k[:,:body-1],v[:,:body-1],*rest) for k,v,*rest in cache]
            consumed = body-1
        else:
            cache,consumed = None,0
        spelling,note_choices = '',[]
        return True

    memory, memory_mask = refresh_context()
    note_prefix_cache = {}
    if end <= origin:
        return [], {'closed': True, 'incomplete': False, 'reason': 'empty_audio_range',
                    'resource_rotations': 0, 'tokens': 0}
    while True:
        if state == 'event':
            if args.max_events and len(points) >= args.max_events:
                reason, incomplete = 'event_budget', True
                break
            needs_search_space = (args.position_beam_size > 1
                and len(ids) + characters + 2 > model.config.max_tokens)
            if points and len(ids) > 1 and (len(ids) >= model.config.max_tokens - 1 or needs_search_space):
                history_tokens = (history_tokens + (ids if rotations == 0 else ids[1:]))[-model.config.max_history_tokens:]
                memory, memory_mask = refresh_context()
                ids, cache, consumed = [codec.bos_id], None, 0
                rotations += 1
            gap = (_spacing_beat_gap(previous, song.tempo, args.min_spacing_ms / 1000.)
                   if previous >= 0 else Fraction(0))
            # Every admissible fraction fits this spelling budget; 40 digits
            # only made impossible numerical branches expensive to explore.
            grammar_type = CanonicalPositionGrammar if args.position_beam_size > 1 else PositionGrammar
            position_grammar = grammar_type(previous, end, characters=characters,
                max_denominator=args.max_denominator,
                max_delta_denominator=args.max_interval_denominator or None,
                delta_reference=previous if previous >= 0 else Fraction(0),
                min_spacing=gap, allowed_ranges=((origin, end),))
            possible = position_grammar.allowed('', codec)
            if not possible:
                tail = float(seconds_at_beat(end, song.tempo, song.first)
                             - seconds_at_beat(max(origin, previous), song.tempo, song.first))
                incomplete = tail > args.min_spacing_ms / 1000. + 1e-8
                reason = 'position_constraints_exhausted' if incomplete else 'audio_range_exhausted'
                break
        if len(ids) > model.config.max_tokens:
            # The length of the next configuration is not known at its start.
            # Rotate at its saved boundary and replay the already sampled
            # prefix, preserving every decision rather than truncating a note
            # or making a resource limit act as a musical ending.
            if header_start <= 1:
                raise ValueError('单个配置超过流上下文 token 上限；请增大「流上下文 token 上限」后重试。')
            history_tokens = (history_tokens + ids[0 if rotations == 0 else 1:header_start])[-model.config.max_history_tokens:]
            pending = ids[header_start:]
            memory, memory_mask = refresh_context()
            ids, cache, consumed, header_start = [codec.bos_id] + pending, None, 0, 1
            rotations += 1
            if len(ids) > model.config.max_tokens:
                raise ValueError('单个配置超过流上下文 token 上限；请增大「流上下文 token 上限」后重试。')
        while consumed < len(ids):
            token = torch.tensor([[ids[consumed]]], dtype=torch.long, device=device)
            output, cache = model.decode_step(token, consumed, memory, memory_mask, cache)
            consumed += 1
        logits = output['logits'][0, -1].float()
        if state == 'event':
            allowed = [codec.eos_id, codec.at_id]
            scores = logits.clone()
            scores[codec.eos_id] -= args.early_eos_penalty
            if len(event_trace) < 64:
                event_trace.append({'after_beat': str(previous) if points else None,
                    'raw_eos_minus_at': float(logits[codec.eos_id]-logits[codec.at_id])})
            selected = sample_token(scores, allowed, args.position_temperature, rng)
            ids.append(selected); emitted_tokens += 1
            if selected == codec.eos_id:
                reason, incomplete = 'eos', False
                break
            header_start = len(ids)-1
            interval_prediction = (float(output['plan'][0, -1, 0])
                if previous >= 0 and args.position_interval_strength else None)
            state, spelling = 'position', ''
        elif state == 'position':
            if not spelling and args.position_beam_size > 1:
                def interval_cost(beat):
                    return interval_consistency_cost(beat-previous, interval_prediction,
                                                     args.position_interval_strength)
                chosen, info = search_position(model, codec, position_grammar, logits, cache,
                    memory, memory_mask, consumed=consumed, width=args.position_beam_size,
                    complexity=args.position_complexity, temperature=args.position_temperature,
                    generator=rng, canonical_only=True, adaptive=True, candidate_cost=interval_cost,
                    proposals=([Fraction(math.floor(previous)+1)] + [previous+(b.beat-a.beat)
                        for a,b in reversed(list(zip(points[-33:],points[-33:][1:])))]
                        if previous >= 0 else []))
                info['interval_prediction_beats_clipped_8'] = (None if interval_prediction is None
                    else min(1., max(0., interval_prediction))*8.)
                searches += 1
                search_expansions += info['expanded_hypotheses']
                if chosen is None:
                    if header_start > 1:
                        # A search may lack space for a whole fraction in the
                        # remaining compute context. Retry at its exact saved
                        # event boundary with the same musical lower bound.
                        history_tokens = (history_tokens + ids[0 if rotations == 0 else 1:header_start])[-model.config.max_history_tokens:]
                        memory, memory_mask = refresh_context()
                        ids, cache, consumed, header_start = [codec.bos_id, codec.at_id], None, 0, 1
                        rotations += 1
                        continue
                    reason, incomplete = 'position_search_exhausted', True
                    break
                if len(search_trace) < 64 or (previous >= 0 and float(seconds_at_beat(Fraction(chosen), song.tempo, song.first)
                        - seconds_at_beat(previous, song.tempo, song.first)) >= 4.):
                    search_trace.append({'previous_beat': str(previous), 'chosen_beat': chosen, **info})
                spelling = chosen
                selected = codec.note_id
                ids.extend(codec.position(Fraction(chosen))[1:])
                emitted_tokens += len(codec.position(Fraction(chosen)))-1
            else:
                allowed = position_grammar.allowed(spelling, codec)
                selected = sample_token(logits, allowed, args.position_temperature, rng)
                ids.append(selected); emitted_tokens += 1
            if selected == codec.note_id:
                position = Fraction(spelling)
                canonical = codec.position(position)
                if ids[header_start:] != canonical:
                    ids[header_start:] = canonical
                    # Reconsume the canonical prefix with identical memory;
                    # caches from discarded spellings must not survive.
                    cache, consumed = None, 0
                    canonicalizations += 1
                second = float(seconds_at_beat(position, song.tempo, song.first))
                if guidance.local_chords is not None:
                    guidance.local_chords.cues[position] = (second, 1.)
                if guidance.physical is not None:
                    guidance.physical.register(position, second, song.tempo, song.first)
                event_bias = guidance.event(position)
                cell_grammar = hard_guard.event(position,button_capacity=(
                    reference_guard.button_capacity(position) if reference_guard is not None else None))
                note_bans,note_choices,note_attempts = {},[],0
                state, spelling = 'note', ''
            else:
                spelling += codec.tokens[selected][2:]
        else:
            prefix_key = (spelling,cell_grammar.forbidden,cell_grammar.button_capacity,
                          cell_grammar.held_buttons,cell_grammar.second)
            allowed = note_prefix_cache.get(prefix_key)
            if allowed is None:
                base_allowed = [index for index, char in chars if grammar.can_continue(spelling+char)]
                allowed = [index for index in base_allowed if cell_grammar.can_continue(spelling+codec.tokens[index][2:])]
                hard_guard.blocked_token_choices += len(base_allowed)-len(allowed)
                if cell_grammar.is_complete(spelling):
                    allowed.append(codec.end_id)
                if len(note_prefix_cache) >= 4096:
                    note_prefix_cache.clear()
                note_prefix_cache[prefix_key] = allowed
            allowed = [token for token in allowed if token not in note_bans.get(spelling,set())]
            if not allowed:
                if reference_guard is not None:
                    note_attempts += 1
                    if retry_note(): continue
                reason, incomplete = 'configuration_constraints_exhausted', True
                break
            continuations = [None if token == codec.end_id else codec.tokens[token][2:] for token in allowed]
            hints = event_bias.biases(spelling, continuations)
            scores = logits.clone()
            scores[allowed] += torch.as_tensor(hints, device=device, dtype=logits.dtype)
            selected = sample_token(scores, allowed, args.temperature, rng)
            if reference_guard is not None and len(allowed)>1:
                alternatives = [token for token in allowed if token != selected]
                margin = float(scores[selected]-scores[alternatives].max())
                note_choices.append((margin,spelling,selected))
            ids.append(selected); emitted_tokens += 1
            if selected == codec.end_id:
                point = EventPoint(position, spelling)
                if reference_guard is not None:
                    note_attempts += 1
                    accepted,_ = reference_guard.check(point)
                    if not accepted:
                        if retry_note(): continue
                        reason,incomplete = 'reference_constraints_exhausted',True
                        break
                points.append(point)
                previous = position
                guidance.commit(spelling, position)
                hard_guard.commit(point)
                if reference_guard is not None: reference_guard.commit(point)
                vector = model.token_embedding(torch.tensor(codec.cell(point), device=device)).mean(0)
                bank.append(position, spelling, vector + base_projection[audio_index(position)])
                if len(points) % 32 == 0:
                    print('stream_arrange_progress '+json.dumps({'events': len(points),
                        'beat': str(position), 'audio_second': float(seconds_at_beat(position, song.tempo, song.first)),
                        'audio_duration_sec': song.audio.duration_sec,
                        'elapsed_sec': round(time.perf_counter()-started, 2),
                        'resource_rotations': rotations, 'search_expansions': search_expansions}), flush=True)
                state, spelling = 'event', ''
            else:
                spelling += codec.tokens[selected][2:]
    return points, {'closed': not incomplete, 'incomplete': incomplete, 'reason': reason,
                    'resource_rotations': rotations, 'tokens': emitted_tokens,
                    'equivalent_fraction_normalizations': canonicalizations,
                    'audio_encoder_calls': 1, 'guidance': guidance.report(),
                    'memory_policy': 'training_history_summary_v1',
                    'basic_constraints': hard_guard.report(),
                    'reference_constraints': reference_guard.report() if reference_guard is not None else {'enabled':False},
                    'position_search': {'mode': 'whole_fraction' if args.position_beam_size > 1 else 'character',
                        'searches': searches, 'expanded_hypotheses': search_expansions,
                        'trace_first_64': search_trace}, 'event_decisions_first_64': event_trace}
