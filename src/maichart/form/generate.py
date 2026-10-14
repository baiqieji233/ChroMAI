"""Full-song onset draft, causal arrangement, and bounded neural revisions."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
import torch

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.form.data import input_example, collate_form, form_audio
from maichart.form.revision import EditPatch, RevisionBudget, apply_patch_candidate
from maichart.form.onset_selection import (
    local_peak_mask, merge_close_onsets, interval_statistics,
    odds_modulate_probabilities, onset_gap_diagnostics,
)
from maichart.form.clock import profile_seconds, chart_curve_to_audio
from maichart.generation.event_audio import segment_frame_times, _normalize_tempo_map, _time_to_beat
from maichart.generation.event_cli import _NoteGrammar
from maichart.generation.event_quality import assess_event_chart

MAX_NOTE_CHARACTERS = 256


def to_device(batch, device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def exact_candidate(second,tempo,first,denominators=(1,2,3,4,5,6,8,12,16,24,32,48)):
    beat=_time_to_beat(float(second)-first,_normalize_tempo_map(tempo,tempo[0].bpm))
    candidates={Fraction(round(beat*d),d) for d in denominators}
    candidates={b for b in candidates if b>=0}
    errors={b:abs(seconds_at_beat(b,tempo,first)-second) for b in candidates}
    # A 50 Hz peak has half-frame timing uncertainty. Prefer a simpler exact
    # subdivision inside that uncertainty instead of turning e.g. a 1/3 beat
    # rounded to 160 ms into an arbitrary 5/16 beat. This is grid inference,
    # not a gameplay judgement tolerance.
    tolerance=max(.0100001,min(errors.values())+1e-9)
    nearby=[b for b in candidates if errors[b]<=tolerance]
    return min(nearby,key=lambda b:(b.denominator,errors[b],b))


def _curve_factor(profile, second, *, strength=1.0, span=0.5, mode='follow'):
    if profile is None:
        return 1.0
    if mode not in {'follow', 'hard'}:
        raise ValueError('song-form intensity curve mode must be follow or hard')
    values = np.asarray([float(point['intensity']) for point in profile['points']], dtype=np.float64)
    times = np.asarray([float(point['time_sec']) for point in profile['points']], dtype=np.float64)
    intensity = float(np.interp(float(second), times, values))
    factor = 2.0 * intensity
    limit = max(0.0, min(1.0, float(span)))
    if mode != 'hard':
        factor = min(1.0 + limit, max(1.0 - limit, factor))
    return 1.0 + float(strength) * (factor - 1.0)


def _curve_condition(song, span, profile, *, strength=1.0, span_limit=0.5, mode='follow'):
    """Return span-local four-channel load values and availability mask."""
    if profile is None:
        return None, None, None
    from maichart.form.curve import LOAD_CURVE_CHANNELS, load_curve_bin_centers
    import torch
    resolution = float(profile.get('curve_resolution_sec', 2.0))
    duration = float(song.audio.duration_sec)
    low = float(seconds_at_beat(span.start, song.tempo, song.first))-float(song.first)
    high = float(seconds_at_beat(span.end, song.tempo, song.first))-float(song.first)
    chart_duration = max(0.,duration-float(song.first))
    # Use the same bin centers and half-open span selection as training.
    times = load_curve_bin_centers(chart_duration, resolution).astype(np.float32)
    absolute_times = chart_curve_to_audio(times,song.first)
    source_clock = profile_seconds(profile,absolute_times,song.first)
    components = profile.get('components') or {}
    scalar = np.asarray([float(point['intensity']) for point in profile['points']], dtype=np.float32)
    scalar_times = np.asarray([float(point['time_sec']) for point in profile['points']], dtype=np.float32)
    values = np.zeros((len(times), 4), dtype=np.float32)
    mask = np.zeros((len(times), 4), dtype=np.bool_)
    for channel, name in enumerate(LOAD_CURVE_CHANNELS):
        if name in components:
            raw = np.asarray(components[name], dtype=np.float32)
            source_times = np.asarray([float(point['time_sec']) for point in profile['points']], dtype=np.float32)
            values[:, channel] = np.interp(source_clock, source_times, raw)
            mask[:, channel] = True
        else:
            values[:, channel] = np.interp(source_clock, scalar_times, scalar)
            mask[:, channel] = channel == 0
    for index, time in enumerate(source_clock):
        # Only channel zero is a density. The other channels are proportions
        # of note heads, so intensity scaling must not change their units.
        values[index, 0] *= np.float32(_curve_factor(profile, time, strength=strength, span=span_limit, mode=mode))
    values[(absolute_times < 0)|(absolute_times >= duration)] = 0
    values = np.clip(values, 0.0, 1.0)
    selected = (times >= low) & (times < high)
    return (torch.from_numpy(values[selected]), torch.from_numpy(mask[selected]),
            torch.from_numpy(times[selected] - np.float32(low)))


def plan_onsets(planner,song,condition,*,threshold=.5,device='cpu',intensity_curve=None,
                intensity_curve_strength=1.0,intensity_curve_span=.5,intensity_curve_mode='follow',
                onset_merge_ms=40.,rhythm_snap_enabled=False,rhythm_snap_ms=15.,
                rhythm_snap_divisions=(4,8,12,16,24,32),rhythm_snap_mode='sequence'):
    """One song-wide neural prediction, before any arrangement chunks exist."""
    if not 0<threshold<1: raise ValueError('onset threshold must be in (0,1)')
    merge_close_onsets([], onset_merge_ms)
    from maichart.form.rhythm_snap import validate_snap_ms, approximate_onsets, normalize_divisions
    validate_snap_ms(rhythm_snap_ms)
    rhythm_snap_divisions = normalize_divisions(rhythm_snap_divisions)
    audio=form_audio(song,song.spans[0].start,song.spans[-1].end)
    example=input_example(song,0,[],condition=condition)
    with torch.inference_mode():
        prediction=planner(audio[None].to(device),example['form'][None].to(device),condition.to_vector(device=device)[None])
    scores=prediction['onset_logits'][0].sigmoid().cpu().numpy()
    types=prediction['type_logits'][0].softmax(-1).cpu().numpy()
    durations=prediction['sustain_seconds'][0].cpu().numpy()
    times=segment_frame_times(seconds_at_beat(song.spans[0].start,song.tempo,song.first),seconds_at_beat(song.spans[-1].end,song.tempo,song.first))
    from maichart.generation.intensity import music_activity
    activity=music_activity(song.audio)
    active=np.interp(times,activity.times,activity.active().astype(float),left=0,right=0) if len(activity.times) else np.ones_like(times)
    audio_valid = (times >= 0) & (times < float(song.audio.duration_sec))
    active = np.where(audio_valid, active, 0.0)
    peaks = local_peak_mask(scores)
    curve_times = profile_seconds(intensity_curve,times,song.first) if intensity_curve is not None else times
    curve_factors = np.ones_like(scores, dtype=np.float64)
    curve_intensities = None
    if intensity_curve is not None:
        curve_times_in_profile = np.asarray(
            [float(point['time_sec']) for point in intensity_curve['points']], dtype=np.float64)
        curve_values_in_profile = np.asarray(
            [float(point['intensity']) for point in intensity_curve['points']], dtype=np.float64)
        curve_intensities = np.interp(curve_times, curve_times_in_profile, curve_values_in_profile)
        curve_factors = np.asarray([
            _curve_factor(intensity_curve, float(time), strength=intensity_curve_strength,
                          span=intensity_curve_span, mode=intensity_curve_mode)
            for time in curve_times
        ], dtype=np.float64)
        # Rhythm snapping consumes raw-audio seconds, whereas a saved legacy
        # curve's rest regions were measured from the chart start.
        offset = float(song.first) if intensity_curve.get('time_axis','chart_relative_v1') == 'chart_relative_v1' else 0.
        audio_rests = [{'start_sec':float(r['start_sec'])+offset,'end_sec':float(r['end_sec'])+offset}
                       for r in intensity_curve.get('rest_regions', [])]
        curve_rest_mask = np.asarray([
            any(float(region['start_sec']) <= float(time) < float(region['end_sec'])
                for region in intensity_curve.get('rest_regions', []))
            for time in curve_times
        ], dtype=np.bool_)
    else:
        audio_rests = []
        curve_rest_mask = np.zeros_like(scores, dtype=np.bool_)
    adjusted_scores = odds_modulate_probabilities(scores, curve_factors)
    # The planner predicts authored selection; an independent audible-span gate
    # prevents a random high score in file-leading/trailing silence.
    selected={}
    candidate_count = 0
    rest_rejected_after_quantization = 0
    for i,score in enumerate(scores):
        if not peaks[i] or active[i]<.05 or not 0<=times[i]<song.audio.duration_sec: continue
        raw_score = float(score)
        if intensity_curve is not None:
            if curve_rest_mask[i]:
                continue
            score = float(adjusted_scores[i])
        if score < threshold: continue
        beat=exact_candidate(times[i],song.tempo,song.first)
        quantized_second = float(seconds_at_beat(beat,song.tempo,song.first))
        if not 0<=quantized_second<song.audio.duration_sec: continue
        if beat>=song.spans[-1].end: continue
        if any(float(region['start_sec']) <= quantized_second < float(region['end_sec'])
               for region in audio_rests):
            rest_rejected_after_quantization += 1
            continue
        record={'beat':str(beat),'second':float(seconds_at_beat(beat,song.tempo,song.first)),
                'score':float(score),'raw_score':raw_score,'source_second':float(times[i]),
                'curve_factor':float(curve_factors[i]),
                'type_probabilities':types[i].tolist(),'sustain_seconds':float(durations[i])}
        candidate_count += 1
        if beat not in selected or selected[beat]['score']<score: selected[beat]=record
    before = [selected[b] for b in sorted(selected)]
    onsets = merge_close_onsets(before, onset_merge_ms)
    selection = {'algorithm':'raw_peaks_curve_odds_gate_temporal_nms_v3',
                 'curve_modulation':'odds_factors' if intensity_curve is not None else 'identity',
                 'onset_merge_ms':float(onset_merge_ms),
                 'raw_local_peaks':int(peaks.sum()),'eligible_candidates':candidate_count,
                 'quantization_duplicates':candidate_count-len(before),
                 'rest_rejected_after_quantization':rest_rejected_after_quantization,
                 'suppressed_close_onsets':len(before)-len(onsets),
                 'before_merge':interval_statistics(before),'after_merge':interval_statistics(onsets)}
    if rhythm_snap_enabled:
        onsets, selection['rhythm_snap'] = approximate_onsets(onsets,song.tempo,song.first,
            tolerance_ms=rhythm_snap_ms,min_interval_ms=onset_merge_ms,end_beat=song.spans[-1].end,
            duration_sec=song.audio.duration_sec,
            rest_regions=audio_rests,divisions=rhythm_snap_divisions,mode=rhythm_snap_mode)
        selection['after_rhythm_snap'] = interval_statistics(onsets)
        print('form_rhythm_snap '+json.dumps(selection['rhythm_snap']),flush=True)
    else:
        selection['rhythm_snap'] = {'enabled':False}
    final = [item for item in onsets if not any(
        float(region['start_sec']) <= float(item['second']) < float(region['end_sec'])
        for region in audio_rests)]
    selection['rest_rejected_after_snap'] = len(onsets)-len(final)
    onsets = final
    diagnostics = onset_gap_diagnostics(
        times=times, probabilities=scores, adjusted=adjusted_scores, peaks=peaks,
        activity=active, rest_mask=curve_rest_mask, rest_regions=audio_rests,
        onsets=onsets, threshold=threshold, curve_intensity=curve_intensities,
        curve_factors=curve_factors,
        start_sec=max(0.0,float(seconds_at_beat(song.spans[0].start,song.tempo,song.first))),
        end_sec=min(float(song.audio.duration_sec),
                    float(seconds_at_beat(song.spans[-1].end,song.tempo,song.first))),
    )
    diagnostics['curve_modulation'] = selection['curve_modulation']
    selection['onset_diagnostics'] = diagnostics
    print('form_onset_diagnostics '+json.dumps(diagnostics),flush=True)
    if rhythm_snap_enabled:
        selection['after_rhythm_snap'] = interval_statistics(onsets)
    print('form_onsets '+json.dumps(selection), flush=True)
    return {'schema':'maichart-full-song-blueprint-v1','onsets':onsets,'selection':selection,
            'threshold':threshold,'frame_seconds':.02,'musical_units':'whole_song',
            'quantization':'simple_rational_within_half_frame_uncertainty','limitations':[
                'Onset probabilities are weighted-training scores, not calibrated confidence.',
                'Broad types and durations are provisional cues; actual layout is decoded separately.']}


def arrange_span(model,codec,example,onsets,atoms,*,origin,temperature=.7,seed=0,device='cpu',type_guidance=None):
    """Arrange the given whole-song skeleton without redeciding each chunk's rhythm.

    Exact position tokens are forced from the blueprint; the model composes
    note atoms. At token budget, return a continuation point without changing
    the musical segment or asking the planner to invent a new rhythm.
    """
    grammar=type_guidance.grammar if type_guidance is not None else _NoteGrammar.from_atoms(atoms)
    batch=to_device(collate_form([example]),device)
    rng=torch.Generator(device=device).manual_seed(seed)
    max_tokens=model.config.max_tokens
    points=[];ids=[codec.bos_id];fallbacks=0
    chars=[(i,t[2:]) for i,t in enumerate(codec.tokens) if t.startswith('N:')]
    model.eval()
    with torch.inference_mode():
        memory,mask=model.encode(batch)
        cache=None;consumed=0
        def next_logits():
            nonlocal cache,consumed
            while consumed < len(ids):
                logits,cache=model.decode_step(torch.tensor([[ids[consumed]]],device=device),consumed,memory,mask,cache)
                consumed+=1
            return logits[0,-1]
        for beat in onsets:
            header=codec.position(beat-origin)
            if len(ids)+len(header)+4>=max_tokens: break
            event_bias = type_guidance.event(beat) if type_guidance is not None else None
            ids.extend(header);text=''
            body_start = len(ids)
            for _ in range(MAX_NOTE_CHARACTERS):
                allowed=[i for i,c in chars if grammar.can_continue(text+c)]
                if grammar.is_complete(text): allowed.append(codec.end_id)
                if not allowed: raise ValueError(f'No valid atom continuation for {text!r}')
                if len(ids)>=max_tokens-1:
                    if not grammar.is_complete(text):
                        # Do not commit half an atom or reroll a whole musical
                        # region. The caller continues this onset with fresh
                        # compute capacity and all previously accepted notes.
                        return points, fallbacks
                    token=codec.end_id
                else:
                    logits=next_logits()
                    values=logits[allowed]
                    if event_bias is not None:
                        biases=event_bias.biases(text,[None if token==codec.end_id else codec.tokens[token][2:] for token in allowed])
                        values=values+torch.as_tensor(biases,device=values.device,dtype=values.dtype)
                    if temperature<=0: token=allowed[int(values.argmax())]
                    else:
                        probabilities=torch.softmax(values/temperature,dim=-1)
                        token=allowed[int(torch.multinomial(probabilities,1,generator=rng))]
                ids.append(token)
                if token==codec.end_id:
                    points.append(EventPoint(beat,text))
                    if type_guidance is not None: type_guidance.commit(text,beat)
                    break
                text+=codec.tokens[token][2:]
            else:
                available = max_tokens-body_start-1
                fallback = text if grammar.is_complete(text) and len(text) <= available else next(
                    (atom for atom in sorted(grammar.atoms,key=lambda atom:(len(atom),atom))
                     if len(atom) <= available and grammar.is_complete(atom)), None)
                if fallback is None:
                    return points, fallbacks
                # Replace the speculative body in both tokens and causal KV
                # memory; later notes must observe the chart actually emitted.
                ids = ids[:body_start]+[codec.ids['N:'+char] for char in fallback]+[codec.end_id]
                if hasattr(model,'rewind_decode_cache'):
                    cache=model.rewind_decode_cache(cache,body_start)
                    consumed=body_start
                else:
                    cache=None;consumed=0
                points.append(EventPoint(beat,fallback));fallbacks += 1
                if type_guidance is not None: type_guidance.commit(fallback,beat)
                print('form_decoder_fallback '+json.dumps({'beat':str(beat),'discarded_chars':len(text),
                    'replacement_chars':len(fallback),'context_repaired':True}),flush=True)
    return points, fallbacks


def generate_chart(model,planner,codec,song,condition,atoms,*,threshold=.5,memory_strength=1.,temperature=.7,seed=0,device='cpu',blueprint=None,
                   intensity_curve=None,intensity_curve_strength=1.0,intensity_curve_span=.5,intensity_curve_mode='follow',
                   onset_merge_ms=40.,rhythm_snap_enabled=False,rhythm_snap_ms=15.,
                   rhythm_snap_divisions=(4,8,12,16,24,32),
                   planner_type_strength=1.,style_balance_strength=1.,rhythm_snap_mode='sequence',
                   local_chord_strength=1.):
    from maichart.form.type_guidance import TypeGuidance,validate_strength
    planner_type_strength=validate_strength(planner_type_strength,'planner_type_strength')
    style_balance_strength=validate_strength(style_balance_strength,'style_balance_strength')
    local_chord_strength=validate_strength(local_chord_strength,'local_chord_strength')
    blueprint=blueprint or plan_onsets(planner,song,condition,threshold=threshold,device=device,
                                       intensity_curve=intensity_curve,
                                       intensity_curve_strength=intensity_curve_strength,
                                       intensity_curve_span=intensity_curve_span,
                                       intensity_curve_mode=intensity_curve_mode,onset_merge_ms=onset_merge_ms,
                                       rhythm_snap_enabled=rhythm_snap_enabled,rhythm_snap_ms=rhythm_snap_ms,
                                       rhythm_snap_divisions=rhythm_snap_divisions,rhythm_snap_mode=rhythm_snap_mode)
    onsets=[Fraction(item['beat']) for item in blueprint['onsets']]
    if onsets!=sorted(set(onsets)): raise ValueError('Blueprint beats must be sorted and unique')
    guidance_cues = [{**item, 'second':float(seconds_at_beat(Fraction(item['beat']),song.tempo,song.first))}
                     for item in blueprint['onsets']]
    type_guidance = TypeGuidance(atoms,condition,guidance_cues,
        planner_strength=planner_type_strength,style_strength=style_balance_strength,
        local_chord_strength=local_chord_strength) if (
        planner_type_strength or style_balance_strength or local_chord_strength) else None
    accepted=[];records=[]
    for span in song.spans:
        desired=[b for b in onsets if span.start<=b<span.end]
        passes=0
        context_records=[]
        span_fallbacks = 0
        while desired:
            # Continue at the first not-yet-written event. Keep the region and
            # its full-song motif reference, and expose accepted local events.
            from dataclasses import replace
            current=span if passes==0 else replace(span,start=desired[0])
            source_spans=song.spans
            if passes: song.spans=[current if s.index==span.index else s for s in source_spans]
            try:
                example=input_example(song,span.index,accepted,condition=condition,codec=codec,
                    max_memory_tokens=model.config.max_memory_tokens,memory_strength=memory_strength)
                context_records.append(dict(example['meta']))
                curve_values, curve_mask, curve_times = _curve_condition(
                    song, current, intensity_curve, strength=intensity_curve_strength,
                    span_limit=intensity_curve_span, mode=intensity_curve_mode)
                if curve_values is not None:
                    # ``collate_form`` adds the batch dimension; keep each
                    # example at [T, 4] / [T] here.
                    example['load_curve'] = curve_values
                    example['load_curve_condition_mask'] = curve_mask
                    example['load_curve_times_sec'] = curve_times
                from maichart.form.revision import DraftPoint
                future=[DraftPoint(b,None,True) for b in onsets if b>=current.start]
                example['skeleton']=torch.tensor(codec.bounded_memory(future,origin=current.start,
                    max_tokens=model.config.max_memory_tokens,newest=False))
                generated, fallback_count=arrange_span(model,codec,example,desired,atoms,origin=current.start,
                    temperature=temperature,seed=seed+span.index*1009+passes,device=device,type_guidance=type_guidance)
            finally: song.spans=source_spans
            if not generated: raise ValueError('Decoder made no progress within the configured token budget')
            span_fallbacks += fallback_count
            accepted.extend(generated);desired=desired[len(generated):];passes+=1
        records.append({'span':span.index,'start':str(span.start),'end':str(span.end),
                        'resource_passes':passes,'onsets':sum(span.start<=b<span.end for b in onsets),
                        'decoder_fallbacks':span_fallbacks,'contexts':context_records})
        print('form_arrange '+json.dumps(records[-1]),flush=True)
    if not accepted: raise ValueError('Whole-song planner selected no onsets; inspect the plan before adjusting threshold')
    quality=assess_event_chart(accepted,song.tempo,first=song.first,duration_sec=song.audio.duration_sec,target_condition=condition)
    from maichart.form.local_load import local_load_statistics
    quality['local_load'] = local_load_statistics(accepted,song.tempo,song.first,
                                                 target_chord_ratio=condition.style['chord_bias'])
    if not quality.get('syntax_valid'): raise ValueError('New-model output failed exact syntax verification')
    type_report=type_guidance.report() if type_guidance is not None else {
        'schema':'song_form_type_guidance_v1','applied':False,
        'planner_type_strength':0.,'style_balance_strength':0.,'hard_quota':False}
    print('form_type_guidance '+json.dumps(type_report),flush=True)
    curve_report = None
    if intensity_curve is not None:
        curve_report = {
            'applied': True,
            'method': intensity_curve.get('method'),
            'strength': float(intensity_curve_strength),
            'span': float(intensity_curve_span),
            'mode': intensity_curve_mode,
            'rest_regions': list(intensity_curve.get('rest_regions', [])),
            'guidance': 'onset soft budget plus span-local layout condition',
            'quality_guarantee': False,
        }
    return accepted,{'blueprint':blueprint,'arrangement':records,'quality':quality,
                     'decoder_fallbacks':sum(int(record.get('decoder_fallbacks',0)) for record in records),
                     'memory_strength':memory_strength,
                     'intensity_curve':curve_report,'type_guidance':type_report}


def revise_chart(model,codec,song,points,condition,atoms,*,start,end,device='cpu',seed=0,temperature=.5,max_checks=8,memory_strength=1.):
    from maichart.form.segmentation import _metric_span
    from maichart.form.data import FormSong
    from maichart.form.repair import _assessment
    from maichart.generation.post_process import PostProcessConfig,detect_chart_conflicts
    start,end=Fraction(start),Fraction(end)
    if start<0 or end<=start or end>song.spans[-1].end: raise ValueError('Invalid edit range')
    history=[p for p in points if p.beat<start]
    region=[p for p in points if start<=p.beat<end]
    right=[p for p in points if p.beat>=end]
    if not region: raise ValueError('Edit region has no existing onsets')
    # Explicit edit region replaces compute geometry only; it does not import
    # future targets into draft mode. Revision is a separate task contract.
    region_index=next(i for i,r in enumerate(song.regions) if r.start<=start<r.end)
    edit=_metric_span(0,start,end,song.meters,1.,'explicit_edit')
    # Keep the original whole-song audio graph and reference identities. An
    # edit is a local compute query, not a new truncated song with new motifs.
    editing=FormSong(song.audio,song.tempo,song.first,song.meters,[edit],song.relations,song.audit,
                     song.regions,[region_index])
    example=input_example(editing,0,history,condition=condition,codec=codec,task=1,
        draft=region,right=right,max_memory_tokens=model.config.max_memory_tokens,memory_strength=memory_strength)
    from maichart.form.revision import DraftPoint
    example['skeleton']=torch.tensor(codec.bounded_memory(
        [DraftPoint(p.beat,None,True) for p in points if p.beat>=start],origin=start,
        max_tokens=model.config.max_memory_tokens,newest=False))
    candidate,fallback_count=arrange_span(model,codec,example,[p.beat for p in region],atoms,origin=start,
        temperature=temperature,seed=seed,device=device)
    if len(candidate)!=len(region): raise ValueError('Edit exceeds decoder budget; select a smaller region')
    config=PostProcessConfig(enable_conflict_detection=True,conflict_engine='miacode')
    result=apply_patch_candidate(points,EditPatch(start,end,tuple(candidate),'neural_revision'),
        lambda chart:_assessment(detect_chart_conflicts(chart,song.tempo,song.first,config)),
        budget=RevisionBudget(max_checks=max_checks,max_edits=8,deadline_secs=10.))
    report={'accepted':result.accepted,'reason':result.reason,'checks':result.checks,'edits':result.edits,
            'decoder_fallbacks':fallback_count,
            'before':asdict(result.before) if result.before else None,'after':asdict(result.after) if result.after else None}
    # The unaccepted proposal is retained for review, never called preferred.
    report['proposal']=[p.to_dict() for p in candidate]
    return list(result.points),report
