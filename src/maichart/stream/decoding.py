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
from maichart.form.rational_generate import _spacing_beat_gap
from maichart.form.rational_position_search import search_position
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.sustain_state import SustainStateIndex
from .memory import StreamMemoryBank
from .timing import CanonicalPositionGrammar, interval_consistency_cost, rhythm_cycle_cost,interval_reliability
from .context import prefix_action_state
from .constraints import BasicConfigurationGuard
from .reference_guard import ReferenceCandidateGuard
from .ending import LearnedEnding
from .configuration_search import propose_configurations
from .continuation import replay_budget,replay_start


def validate_options(args):
    for name,default in (('eos_temperature',1.),('position_audio_strength',.25),('repetition_strength',.6)):
        value=float(getattr(args,name,default))
        if not math.isfinite(value) or not 0<=value<=3:
            raise ValueError(f'{name} 必须是 0–3 的有限数值')
    for name,default in (('repetition_min_seconds',4.),('repetition_min_beats',8.)):
        value=float(getattr(args,name,default))
        if not math.isfinite(value) or not 0<value<=120:
            raise ValueError(f'{name} 必须大于0且不超过120')
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
                            ('position_beam_size', 1, 8),('position_complexity_floor',1,4096),
                            ('event_reconsider_attempts',0,16)):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 {low}–{high} 的整数')
    candidates=getattr(args,'configuration_candidates',4)
    if isinstance(candidates,bool) or not isinstance(candidates,int) or not 1<=candidates<=16:
        raise ValueError('configuration_candidates 必须是 1–16 的整数')
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
    for name in ('style_balance_strength', 'local_chord_strength', 'physical_feasibility_strength','learned_ending_strength'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 3:
            raise ValueError(f'{name} 必须是 0–3 的有限数值')
    section=getattr(args,'section_type_strength',1.0)
    if not math.isfinite(section) or not 0<=section<=3:
        raise ValueError('section_type_strength 必须是 0–3 的有限数值')
    rhythm_diversity=float(getattr(args,'rhythm_diversity_strength',.6))
    if not math.isfinite(rhythm_diversity) or not 0<=rhythm_diversity<=3:
        raise ValueError('rhythm_diversity_strength 必须是 0–3 的有限数值')
    gap_penalty=float(getattr(args,'unplanned_gap_penalty_strength',0.))
    if not math.isfinite(gap_penalty) or not 0<=gap_penalty<=3:
        raise ValueError('unplanned_gap_penalty_strength 必须是 0–3 的有限数值')
    intent=getattr(args,'plan_intent_strength',1.0)
    if not math.isfinite(intent) or not 0<=intent<=1:
        raise ValueError('plan_intent_strength 必须是 0–1 的有限数值')
    if not math.isfinite(args.constraint_min_probability) or not 0<=args.constraint_min_probability<=1:
        raise ValueError('constraint_min_probability 必须为 0–1 的有限数值')
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


def eos_probability(margin,temperature):
    """Actual two-token probability after the independent temperature."""
    return (float(margin>=0) if temperature==0 else
            math.exp(-float(np.logaddexp(0.,-float(margin)/temperature))))


def rewind_decoder_cache(cache, prefix_length):
    """Discard speculative self attention, retain immutable cross attention."""
    if not isinstance(cache,list):
        return None,0
    return [(k[:,:prefix_length],v[:,:prefix_length],*rest) for k,v,*rest in cache],prefix_length


def search_workspace_limits(beam, candidates, free_gib=None):
    """Limit concurrent hypotheses, never disable Fraction scoring."""
    if free_gib is not None and free_gib < 4.:
        return 1, 1
    if free_gib is not None and free_gib < 6.:
        return min(2, beam), 1
    return beam, candidates


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
    grounded=None
    if model.config.grounding_profile!='none':
        from .grounded_inference import GroundedInference
        grounded=GroundedInference(model,codec,song,encoded_audio,audio_times=audio_times)
        model=grounded
    base_projection = model.audio_projection(audio)[0] if grounded is None else None
    bank = StreamMemoryBank(max_entries=max(256, model.config.max_stream_memory),
                            max_retrieval=model.config.max_stream_memory)
    section_strength=float(getattr(args,'section_type_strength',1.0))
    section_profile=None
    if ((section_strength and args.style_balance_strength) or args.local_chord_strength
            or getattr(args,'repetition_strength',.6)):
        from .section_profile import SectionTypeProfile
        section_profile=SectionTypeProfile(encoded_audio[2][0].sigmoid().float().cpu().numpy(),audio_times)
    plan_intent_strength=float(getattr(args,'plan_intent_strength',1.0))
    guidance = TypeGuidance(atoms, condition, [], planner_strength=0.,
        style_strength=args.style_balance_strength, local_chord_strength=args.local_chord_strength,
        physical_strength=args.physical_feasibility_strength,
        section_profile=section_profile if section_strength and args.style_balance_strength else None,
        section_strength=section_strength if section_profile is not None and args.style_balance_strength else 0.,
        plan_intent_strength=plan_intent_strength, max_star_slides=2)
    plan_intent=None
    from .repetition import RepetitionGuard
    repetition=RepetitionGuard(section_profile,strength=getattr(args,'repetition_strength',.6),
        min_seconds=getattr(args,'repetition_min_seconds',4.),min_beats=getattr(args,'repetition_min_beats',8.))
    position_audio_strength=float(getattr(args,'position_audio_strength',.25))
    position_audio_enabled=bool(position_audio_strength and grounded is not None
        and getattr(args,'alignment_training_verified',False))
    position_audio_events=0;position_evidence=None
    eos_temperature=float(getattr(args,'eos_temperature',args.position_temperature))
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
    ending=(LearnedEnding(audio_times,encoded_audio[2][0,:,0].sigmoid().float().cpu().numpy(),
                          getattr(args,'effective_end_sec',song.audio.duration_sec),args.learned_ending_strength)
            if float(getattr(args,'effective_end_sec',song.audio.duration_sec))>0 else None)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    proposal_rng = torch.Generator(device=device).manual_seed((args.seed+104729)%(2**63-1))
    terminal_guidance=bool(args.temperature>0 and getattr(args,'configuration_candidates',4)>1
        and (args.style_balance_strength or args.local_chord_strength or args.physical_feasibility_strength))
    proposal_events=proposal_tokens=proposal_count=0
    position = None
    state, spelling, header_start = 'event', '', 0
    cache, consumed = None, 0
    rotations = emitted_tokens = canonicalizations = 0
    rotation_trace=[]
    event_bias = None
    interval_prediction = None
    interval_policy=interval_reliability(getattr(args,'interval_metrics',None))
    searches, search_expansions, search_trace, event_trace = 0, 0, [], []
    rejected_positions,event_retries,event_retry_count = set(),[],0
    event_trace_tail=[]
    memory_guard_events=[]
    started = time.perf_counter()
    def workspace_limits():
        beam=args.position_beam_size; candidates=getattr(args,'configuration_candidates',4)
        free_gib=None
        if device.type=='cuda':
            free,_=torch.cuda.mem_get_info(device)
            free_gib=float(free)/1024**3
        beam,candidates=search_workspace_limits(beam,candidates,free_gib)
        if beam!=args.position_beam_size or candidates!=getattr(args,'configuration_candidates',4):
            if not memory_guard_events or memory_guard_events[-1]['beam']!=beam or memory_guard_events[-1]['candidates']!=candidates:
                memory_guard_events.append({'free_gib':None if free_gib is None else round(free_gib,3),
                    'beam':beam,'candidates':candidates,'position_policy':'whole_fraction',
                    'events':len(points),'after_beat':str(previous) if points else None})
        return beam,candidates
    # Search just beyond the file as a stop alternative. Otherwise the grammar
    # turns an intended next whole beat beyond EOF into a tiny legal fraction
    # in the last milliseconds. Such a position is never emitted as a note.
    search_end=end+8
    characters = max(4, len(str(math.ceil(search_end*args.max_denominator)))
                     + 1 + len(str(args.max_denominator)))

    def audio_index(beat):
        second = float(seconds_at_beat(max(Fraction(0), beat), song.tempo, song.first))
        right = int(np.searchsorted(audio_times, second))
        choices = [i for i in (right-1, right) if 0 <= i < len(audio_times)]
        return min(choices, key=lambda i: abs(float(audio_times[i])-second)) if choices else 0

    def refresh_context(context_points=None):
        context_points=points if context_points is None else context_points
        if grounded is not None:grounded.begin_context(context_points,history_tokens)
        action = audio.new_zeros((1, 1, 50))
        action_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
        if context_points:
            index = ((grounded.replay_state or grounded.state) if grounded is not None
                     else SustainStateIndex(context_points, song.tempo, song.first))
            states, padding, _ = prefix_action_state(index, context_points, len(context_points))
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
        kwargs={'history_clock':grounded.history_clock} if grounded is not None else {}
        return model.prepare_decode_memory(memory, padding, history=history,**kwargs)

    def rotate_context(pending=(), *, space_failure=False):
        nonlocal ids,history_tokens,memory,memory_mask,cache,consumed,header_start,rotations
        budget=replay_budget(model.config)
        if space_failure and rotation_trace and rotation_trace[-1]['events']==len(points):
            budget=0  # A long cell must not cause endless replay retries.
        full=codec.encode(points)[:-1]
        warm=replay_start(full,len(full),min(budget,max(0,model.config.max_tokens-len(pending)-2)),codec.at_id)
        count=sum(t==codec.at_id for t in full[:warm])
        history_tokens=full[max(0,warm-model.config.max_history_tokens):warm]
        # Release the old self/cross attention workspace before constructing
        # the replacement context. Otherwise a rotation briefly holds both
        # generations of KV tensors and can push Windows into shared memory.
        cache=None; consumed=0
        old_memory,old_mask=memory,memory_mask
        memory=None; memory_mask=None
        del old_memory,old_mask
        memory,memory_mask=refresh_context(points[:count])
        suffix=full[warm:]
        ids=[codec.bos_id]+suffix+list(pending)
        header_start=1+len(suffix)
        rotations+=1
        record={'events':len(points),'after_beat':str(previous),
            'audio_second':float(seconds_at_beat(previous,song.tempo,song.first)),
            'replayed_events':len(points)-count,'replayed_tokens':len(suffix),
            'history_tokens':len(history_tokens),'reason':'space_retry' if space_failure else 'context_capacity'}
        rotation_trace.append(record)
        print('stream_context_rotation '+json.dumps(record),flush=True)

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

    def retry_event(cause,probability=None):
        nonlocal ids,cache,consumed,state,spelling,event_retry_count
        if position is None or event_retry_count>=args.event_reconsider_attempts:
            return False
        rejected_positions.add(position)
        event_retry_count+=1
        event_retries.append({'beat':str(position),'reason':cause,'retained_model_mass':probability,
                              'attempt':event_retry_count})
        # Rewind the complete speculative event, not any committed music.
        # Cross memory still belongs to this resource context; no rejected
        # action, style count or archive item has been committed yet.
        ids=ids[:header_start]
        # Keep the prefix K/V and reconsume its final token to recover the
        # next-event distribution. Replaying an entire 1024-token block on
        # every failed candidate dominated generation time in the real audit.
        cache,consumed=rewind_decoder_cache(cache,header_start-1)
        # Local/physical helpers may have evaluated the rejected future time;
        # rebuild them from authored points before trying an earlier proposal.
        guidance.replay(points,song.tempo,song.first,
            curve_factor=section_profile.density_factor if section_profile is not None else None)
        state,spelling='event',''
        return True

    memory, memory_mask = refresh_context()
    note_prefix_cache = {}
    if end <= origin:
        return [], {'closed': True, 'incomplete': False, 'reason': 'empty_audio_range',
                    'resource_rotations': 0, 'tokens': 0}
    while True:
        if state == 'event':
            effective_beam,effective_candidates=workspace_limits()
            if args.max_events and len(points) >= args.max_events:
                reason, incomplete = 'event_budget', True
                break
            needs_search_space = len(ids) + characters + 2 > model.config.max_tokens
            if points and len(ids) > 1 and (len(ids) >= model.config.max_tokens - 1 or needs_search_space):
                rotate_context()
            gap = (_spacing_beat_gap(previous, song.tempo, args.min_spacing_ms / 1000.)
                   if previous >= 0 else Fraction(0))
            # Every admissible fraction fits this spelling budget; 40 digits
            # only made impossible numerical branches expensive to explore.
            position_grammar = CanonicalPositionGrammar(previous, search_end, characters=characters,
                max_denominator=args.max_denominator,
                max_delta_denominator=args.max_interval_denominator or None,
                delta_reference=previous if previous >= 0 else Fraction(0),
                min_spacing=gap, allowed_ranges=((origin, search_end),),
                valid_position=lambda value:value not in rejected_positions)
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
            pending = ids[header_start:]
            rotate_context(pending,space_failure=True)
            if len(ids) > model.config.max_tokens:
                raise ValueError('单个配置超过流上下文 token 上限；请增大「流上下文 token 上限」后重试。')
        while consumed < len(ids):
            token = torch.tensor([[ids[consumed]]], dtype=torch.long, device=device)
            if grounded is not None:grounded.set_prefix(ids)
            output, cache = model.decode_step(token, consumed, memory, memory_mask, cache)
            consumed += 1
        logits = output['logits'][0, -1].float()
        if state == 'event':
            allowed = [codec.eos_id, codec.at_id]
            # Read boundary diagnostics and the complete plan together. These
            # CPU consumers used to synchronize CUDA separately six times.
            plan=output.get('plan')
            boundary_values=torch.cat((logits[allowed],plan[0,-1].float()
                if plan is not None else logits.new_empty(0))).cpu().tolist()
            scores = logits.clone()
            scores[codec.eos_id] -= args.early_eos_penalty
            raw_eos_margin=boundary_values[0]-boundary_values[1]
            raw_margin=raw_eos_margin-args.early_eos_penalty
            guided_margin=raw_margin
            if ending is not None and ending.strength:
                clock=float(seconds_at_beat(previous,song.tempo,song.first)) if points else 0.
                guided_margin=ending.eos_logit(raw_margin,clock)
                scores[codec.eos_id]=scores[codec.at_id]+guided_margin
            decision={'after_beat':str(previous) if points else None,
                'raw_eos_minus_at':raw_eos_margin,
                'guided_eos_minus_at':guided_margin,'eos_temperature':eos_temperature,
                'eos_probability':eos_probability(guided_margin,eos_temperature)}
            if len(event_trace)<64:event_trace.append(decision)
            event_trace_tail.append(decision);del event_trace_tail[:-32]
            selected = sample_token(scores, allowed, eos_temperature, rng)
            ids.append(selected); emitted_tokens += 1
            if selected == codec.eos_id:
                reason, incomplete = 'eos', False
                break
            header_start = len(ids)-1
            position_evidence=(grounded.position_evidence(output['hidden'],control,strength=position_audio_strength)
                               if position_audio_enabled else None)
            if position_evidence is not None:position_audio_events+=1
            interval_prediction = (boundary_values[2]
                if previous >= 0 and plan is not None and
                (args.position_interval_strength or getattr(args,'unplanned_gap_penalty_strength',0.)) else None)
            # Same row the plan target supervises: the next event's intent.
            plan_intent = (np.asarray(boundary_values[5:],dtype=np.float32)
                           if plan_intent_strength and plan is not None else None)
            state, spelling = 'position', ''
        elif state == 'position':
            effective_beam,effective_candidates=workspace_limits()
            rhythm_diversity_strength=float(getattr(args,'rhythm_diversity_strength',.6))
            gap_penalty_strength=float(getattr(args,'unplanned_gap_penalty_strength',0.))
            if not spelling:
                def interval_cost(beat):
                    return (interval_consistency_cost(beat-previous, interval_prediction,
                                                      args.position_interval_strength,
                                                      gap_penalty_strength,
                                                      reliability=interval_policy['multiplier'],cap=.5)
                            + rhythm_cycle_cost(beat, points, previous, rhythm_diversity_strength)
                            + (position_evidence.cost(seconds_at_beat(beat,song.tempo,song.first))
                               if position_evidence is not None else 0.))
                chosen, info = search_position(model, codec, position_grammar, logits, cache,
                    memory, memory_mask, consumed=consumed, width=effective_beam,
                    complexity=args.position_complexity, temperature=args.position_temperature,
                    complexity_floor=args.position_complexity_floor,
                    timing_features=grounded.timing_features if grounded is not None else None,
                    generator=rng, canonical_only=True, adaptive=True, candidate_cost=interval_cost,
                    max_width=effective_beam,
                    proposal_trigger=lambda beat: (interval_prediction is not None
                        and 0 <= interval_prediction < 1.
                        and float(beat-previous) > max(interval_prediction*16.,interval_prediction*8.+4.)),
                    proposals=([Fraction(math.floor(previous)+1)]
                        + [previous+step for step in (Fraction(1,4),Fraction(1,2),Fraction(1,3),Fraction(3,4),
                                                      Fraction(1,6),Fraction(1,8),Fraction(1),Fraction(2))]
                        + [previous+(b.beat-a.beat) for a,b in reversed(list(zip(points[-17:],points[-17:][1:])))]
                        if previous >= 0 else []))
                info['interval_prediction_beats_clipped_8'] = (None if interval_prediction is None
                    else min(1., max(0., interval_prediction))*8.)
                info['interval_prediction_raw']=interval_prediction
                info['interval_shaping_multiplier']=interval_policy['multiplier']
                if position_evidence is not None:info['audio_pointer']=position_evidence.report()
                searches += 1
                search_expansions += info['expanded_hypotheses']
                if chosen is None:
                    if header_start > 1:
                        # A search may lack space for a whole fraction in the
                        # remaining compute context. Retry at its exact saved
                        # event boundary with the same musical lower bound.
                        rotate_context([codec.at_id],space_failure=True)
                        continue
                    reason, incomplete = 'position_search_exhausted', True
                    break
                if len(search_trace) < 64 or info.get('proposal_triggered') or (previous >= 0 and float(seconds_at_beat(Fraction(chosen), song.tempo, song.first)
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
                if position>=end:
                    reason,incomplete='predicted_position_after_audio_end',False
                    break
                canonical = codec.position(position)
                if ids[header_start:] != canonical:
                    ids[header_start:] = canonical
                    # Reconsume the canonical prefix with identical memory;
                    # caches from discarded spellings must not survive.
                    cache, consumed = None, 0
                    canonicalizations += 1
                second = float(seconds_at_beat(position, song.tempo, song.first))
                if guidance.local_chords is not None:
                    guidance.local_chords.cues[position] = (
                        second, section_profile.density_factor(second) if section_profile is not None else 1.)
                if guidance.physical is not None:
                    guidance.physical.register(position, second, song.tempo, song.first)
                event_bias = guidance.event(position, second, intent=plan_intent)
                event_bias = repetition.event(event_bias,position,second)
                terminal_guidance=bool(args.temperature>0 and getattr(args,'configuration_candidates',4)>1
                    and (args.style_balance_strength or args.local_chord_strength or args.physical_feasibility_strength
                         or event_bias.cases))
                cell_grammar = hard_guard.event(position,
                    button_capacity=(reference_guard.button_capacity(position) if reference_guard is not None else None),
                    slide_launch_buttons=(reference_guard.slide_launch_buttons(position)
                                          if reference_guard is not None else frozenset()))
                note_bans,note_choices,note_attempts = {},[],0
                state, spelling = 'note', ''
            else:
                spelling += codec.tokens[selected][2:]
        else:
            if terminal_guidance and not spelling:
                def check_configuration(text):
                    nonlocal note_attempts
                    if reference_guard is not None:note_attempts+=1
                    return reference_guard.check(EventPoint(position,text))[0] if reference_guard is not None else True
                candidate_budget=effective_candidates
                if reference_guard is not None:candidate_budget=min(candidate_budget,max(1,args.reference_check_attempts-note_attempts))
                picked,search=propose_configurations(model,codec,logits,cache,memory,memory_mask,
                    consumed=consumed,max_tokens=model.config.max_tokens,grammar=grammar,restricted=cell_grammar,
                    event_bias=event_bias,temperature=args.temperature,candidates=candidate_budget,
                    generator=proposal_rng,minimum_mass=args.constraint_min_probability,
                    timing_features=grounded.timing_features if grounded is not None else None,
                    check=check_configuration)
                proposal_events+=1;proposal_tokens+=search['expanded_tokens'];proposal_count+=search['proposals']
                if search['reason']=='context_space':
                    if header_start<=1:
                        raise ValueError('完整配置候选超过流上下文 token 上限；请增大「流上下文 token 上限」。')
                    pending=ids[header_start:]
                    rotate_context(pending,space_failure=True)
                    continue
                if picked is None:
                    if (reference_guard is not None and search['reason']=='reference_constraints_exhausted'
                            and note_attempts<args.reference_check_attempts):continue
                    if retry_event(search['reason']):continue
                    reason,incomplete=search['reason'],True
                    break
                spelling=picked['text'];ids.extend(picked['tokens'])
                cache,consumed=picked['cache'],picked['consumed']
                emitted_tokens+=len(picked['tokens'])
                selected=codec.end_id
            else:
                selected=None
            if selected is None:
                prefix_key = (spelling,cell_grammar.forbidden,cell_grammar.button_capacity,
                              cell_grammar.held_buttons,cell_grammar.slide_launch_buttons,cell_grammar.second)
                cached_allowed = note_prefix_cache.get(prefix_key)
                if cached_allowed is None:
                    base_allowed = [index for index, char in chars if grammar.can_continue(spelling+char)]
                    allowed = [index for index in base_allowed if cell_grammar.can_continue(spelling+codec.tokens[index][2:])]
                    hard_guard.blocked_token_choices += len(base_allowed)-len(allowed)
                    if grammar.is_complete(spelling):base_allowed.append(codec.end_id)
                    if cell_grammar.is_complete(spelling):allowed.append(codec.end_id)
                    if len(note_prefix_cache) >= 4096:note_prefix_cache.clear()
                    note_prefix_cache[prefix_key] = (allowed,base_allowed)
                else:
                    allowed,base_allowed=cached_allowed
                if allowed:
                    token_text=[None if token==codec.end_id else codec.tokens[token][2:] for token in allowed]
                    values=event_bias.token_bias(spelling,token_text)
                    # No terminal resampling here: use the bounded prefix
                    # potentials so style, local-chord and physical guidance
                    # still act instead of silently switching off.
                    values=[a+b for a,b in zip(values,event_bias.biases(spelling,token_text))]
                    bias=torch.as_tensor(values,device=logits.device,dtype=logits.dtype)
                    scores=logits.clone();scores[allowed]+=bias
                else:
                    scores=logits
                if args.constraint_min_probability and base_allowed and len(allowed)<len(base_allowed):
                    kept=(float(torch.exp(torch.logsumexp(logits[allowed],0)-torch.logsumexp(logits[base_allowed],0)))
                          if allowed else 0.)
                    if kept<args.constraint_min_probability:
                        if retry_event('hard_mask_discarded_likely_configuration',kept):continue
                        reason,incomplete='low_probability_configuration_constraints',True
                        break
                allowed = [token for token in allowed if token not in note_bans.get(spelling,set())]
                if not allowed:
                    if reference_guard is not None:
                        note_attempts += 1
                        if retry_note(): continue
                    if retry_event('configuration_constraints_exhausted'):continue
                    reason, incomplete = 'configuration_constraints_exhausted', True
                    break
                selected = sample_token(scores, allowed, args.temperature, rng)
                if reference_guard is not None and len(allowed)>1:
                    alternatives = [token for token in allowed if token != selected]
                    margin = float(logits[selected]-logits[alternatives].max())
                    note_choices.append((margin,spelling,selected))
                ids.append(selected); emitted_tokens += 1
            if selected == codec.end_id:
                point = EventPoint(position, spelling)
                if reference_guard is not None and not terminal_guidance:
                    note_attempts += 1
                    accepted,_ = reference_guard.check(point)
                    if not accepted:
                        if retry_note(): continue
                        if retry_event('reference_constraints_exhausted'):continue
                        reason,incomplete = 'reference_constraints_exhausted',True
                        break
                points.append(point)
                previous = position
                guidance.commit(spelling, position)
                repetition.commit(spelling,position,seconds_at_beat(position,song.tempo,song.first))
                hard_guard.commit(point)
                if reference_guard is not None: reference_guard.commit(point)
                if grounded is not None:grounded.commit(point)
                rejected_positions.clear();event_retry_count=0
                if grounded is None:
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
                    'context_rotations':rotation_trace,
                    'memory_guard_events':memory_guard_events,
                    'continuation_profile':getattr(model.config,'continuation_profile','reset_v1'),
                    'interval_guidance':{**interval_policy,'normal_shaping_cap':.5,
                        'unplanned_gap_cost':'independent_user_control'},
                    'equivalent_fraction_normalizations': canonicalizations,
                    'audio_encoder_calls': 1, 'guidance': guidance.report(),
                    'configuration_search':{'mode':'complete_cell_importance_resampling',
                        'candidate_budget':getattr(args,'configuration_candidates',4),
                        'guided_events':proposal_events,'proposals':proposal_count,'expanded_tokens':proposal_tokens,
                        'prefix_style_bias':False,'greedy_terminal_guidance':False},
                    'grounding':grounded.report() if grounded is not None else {'profile':'none'},
                    'memory_policy': 'training_history_summary_v1',
                    'basic_constraints': hard_guard.report(),
                    'reference_constraints': reference_guard.report() if reference_guard is not None else {'enabled':False},
                    'learned_ending':ending.report() if ending is not None else {'enabled':False},
                    'eos_temperature':eos_temperature,'repetition':repetition.report(),
                    'position_audio_guidance':{'enabled':position_audio_enabled,'requested_strength':position_audio_strength,
                        'effective_strength':position_audio_strength if position_audio_enabled else 0.,
                        'supervision_verified':bool(getattr(args,'alignment_training_verified',False)),
                        'events':position_audio_events,'source':'authored_next_event_pointer',
                        'fallback_reason':None if position_audio_enabled else 'disabled_or_no_verified_alignment_training'},
                    'event_reconsideration':{'retries':event_retries,'total_retries':len(event_retries),
                        'minimum_retained_probability':args.constraint_min_probability,
                        'budget_per_event':args.event_reconsider_attempts},
                    'position_search': {'mode': 'whole_fraction',
                        'searches': searches, 'expanded_hypotheses': search_expansions,
                        'trace_first_64': search_trace}, 'event_decisions_first_64': event_trace,
                    'event_decisions_last_32':event_trace_tail}
