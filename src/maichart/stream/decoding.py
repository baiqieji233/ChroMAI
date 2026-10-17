"""Bounded continuous decoding over one immutable audio encoding."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import torch

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.form.rational_audio import beat_at_second
from maichart.form.rational_generate import PositionGrammar, _spacing_beat_gap
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.sustain_state import SustainStateIndex
from .memory import StreamMemoryBank


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
                            ('max_events', 0, 1_000_000), ('seed', 0, 2**63-1)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 {low}–{high} 的整数')
    if not math.isfinite(args.min_spacing_ms) or not 0 <= args.min_spacing_ms <= 1000:
        raise ValueError('min_spacing_ms 必须是 0–1000 的有限数值')
    for name in ('style_balance_strength', 'local_chord_strength', 'physical_feasibility_strength'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 3:
            raise ValueError(f'{name} 必须是 0–3 的有限数值')


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
    chars = [(index, text[2:]) for index, text in enumerate(codec.tokens) if text.startswith('N:')]
    points, ids, history_tokens = [], [codec.bos_id], []
    previous = Fraction(-1)
    origin = max(Fraction(0), beat_at_second(0., song.tempo, song.first))
    end = beat_at_second(song.audio.duration_sec, song.tempo, song.first)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    position = None
    state, spelling, header_start = 'event', '', 0
    cache, consumed = None, 0
    rotations = emitted_tokens = canonicalizations = 0
    event_bias = None

    def audio_index(beat):
        second = float(seconds_at_beat(max(Fraction(0), beat), song.tempo, song.first))
        right = int(np.searchsorted(audio_times, second))
        choices = [i for i in (right-1, right) if 0 <= i < len(audio_times)]
        return min(choices, key=lambda i: abs(float(audio_times[i])-second)) if choices else 0

    def refresh_context():
        action = audio.new_zeros((1, 1, 50))
        action_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
        if points:
            # Include the last committed head; at(head) itself is strictly
            # causal and would still hide a Hold started at exactly that beat.
            index = SustainStateIndex(points, song.tempo, song.first)
            states, padding, _ = index.at(points[-1].beat + Fraction(1, 1_000_000))
            action, action_mask = states[None].to(device), padding[None].to(device)
        history = torch.tensor([history_tokens], dtype=torch.long, device=device) if history_tokens else None
        summary = model.summarize_tokens(history) if history is not None else None
        archive = bank.tensor(base_projection[audio_index(previous)], current_beat=previous)
        if archive is not None:
            archive = archive[None].to(device)
            summary = archive if summary is None else torch.cat((summary, archive), 1)
            summary = summary[:, -model.config.max_stream_memory:]
        memory, padding = model.encode(audio, control, audio_padding_mask=audio_mask,
            action_state=action, action_state_padding_mask=action_mask,
            stream_memory=summary, encoded_audio=encoded_audio)
        return model.prepare_decode_memory(memory, padding, history=history)

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
            if len(ids) >= model.config.max_tokens - 1 and points:
                history_tokens = (history_tokens + ids)[-model.config.max_history_tokens:]
                memory, memory_mask = refresh_context()
                ids, cache, consumed = [codec.bos_id], None, 0
                rotations += 1
            gap = (_spacing_beat_gap(previous, song.tempo, args.min_spacing_ms / 1000.)
                   if previous >= 0 else Fraction(0))
            position_grammar = PositionGrammar(previous, end, characters=40,
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
            history_tokens = (history_tokens + ids[:header_start])[-model.config.max_history_tokens:]
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
            if points:
                # Preserve the existing bounded anti-early-EOS heuristic.
                scores[codec.eos_id] -= min(8., max(0., float(end-previous)) / 8.)
            selected = sample_token(scores, allowed, args.position_temperature, rng)
            ids.append(selected); emitted_tokens += 1
            if selected == codec.eos_id:
                reason, incomplete = 'eos', False
                break
            header_start = len(ids)-1
            state, spelling = 'position', ''
        elif state == 'position':
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
                state, spelling = 'note', ''
            else:
                spelling += codec.tokens[selected][2:]
        else:
            allowed = note_prefix_cache.get(spelling)
            if allowed is None:
                allowed = [index for index, char in chars if grammar.can_continue(spelling+char)]
                if grammar.is_complete(spelling):
                    allowed.append(codec.end_id)
                if len(note_prefix_cache) >= 4096:
                    note_prefix_cache.clear()
                note_prefix_cache[spelling] = allowed
            if not allowed:
                raise ValueError(f'No valid note continuation for {spelling!r}')
            continuations = [None if token == codec.end_id else codec.tokens[token][2:] for token in allowed]
            hints = event_bias.biases(spelling, continuations)
            scores = logits.clone()
            scores[allowed] += torch.as_tensor(hints, device=device, dtype=logits.dtype)
            selected = sample_token(scores, allowed, args.temperature, rng)
            ids.append(selected); emitted_tokens += 1
            if selected == codec.end_id:
                point = EventPoint(position, spelling)
                points.append(point)
                previous = position
                guidance.commit(spelling, position)
                vector = model.token_embedding(torch.tensor(codec.cell(point), device=device)).mean(0)
                bank.append(position, spelling, vector + base_projection[audio_index(position)])
                state, spelling = 'event', ''
            else:
                spelling += codec.tokens[selected][2:]
    return points, {'closed': not incomplete, 'incomplete': incomplete, 'reason': reason,
                    'resource_rotations': rotations, 'tokens': emitted_tokens,
                    'equivalent_fraction_normalizations': canonicalizations,
                    'audio_encoder_calls': 1, 'guidance': guidance.report()}
