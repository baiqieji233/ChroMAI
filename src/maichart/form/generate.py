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
from maichart.generation.event_audio import segment_frame_times, _normalize_tempo_map, _time_to_beat
from maichart.generation.event_cli import _NoteGrammar
from maichart.generation.event_quality import assess_event_chart


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
    from maichart.form.curve import LOAD_CURVE_CHANNELS
    import torch
    resolution = float(profile.get('curve_resolution_sec', 2.0))
    duration = float(song.audio.duration_sec)
    low = max(0.0, float(seconds_at_beat(span.start, song.tempo, song.first)) - float(song.first))
    high = min(duration, float(seconds_at_beat(span.end, song.tempo, song.first)) - float(song.first))
    count = max(1, int(math.ceil(duration / resolution)))
    times = np.asarray([min(duration, (i + 0.5) * resolution) for i in range(count)], dtype=np.float32)
    components = profile.get('components') or {}
    scalar = np.asarray([float(point['intensity']) for point in profile['points']], dtype=np.float32)
    scalar_times = np.asarray([float(point['time_sec']) for point in profile['points']], dtype=np.float32)
    values = np.zeros((len(times), 4), dtype=np.float32)
    mask = np.zeros((len(times), 4), dtype=np.bool_)
    for channel, name in enumerate(LOAD_CURVE_CHANNELS):
        if name in components:
            raw = np.asarray(components[name], dtype=np.float32)
            source_times = np.asarray([float(point['time_sec']) for point in profile['points']], dtype=np.float32)
            values[:, channel] = np.interp(times, source_times, raw)
            mask[:, channel] = True
        else:
            values[:, channel] = np.interp(times, scalar_times, scalar)
            mask[:, channel] = channel == 0
    for index, time in enumerate(times):
        values[index] *= np.float32(_curve_factor(profile, time, strength=strength, span=span_limit, mode=mode))
    values = np.clip(values, 0.0, 1.0)
    selected = (times >= low - resolution) & (times < high + resolution)
    if not np.any(selected):
        selected[np.argmin(np.abs(times - (low + high) * .5))] = True
    return (torch.from_numpy(values[selected]), torch.from_numpy(mask[selected]),
            torch.from_numpy(times[selected] - np.float32(low)))


def plan_onsets(planner,song,condition,*,threshold=.5,device='cpu',intensity_curve=None,
                intensity_curve_strength=1.0,intensity_curve_span=.5,intensity_curve_mode='follow'):
    """One song-wide neural prediction, before any arrangement chunks exist."""
    if not 0<threshold<1: raise ValueError('onset threshold must be in (0,1)')
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
    # The planner predicts authored selection; an independent audible-span gate
    # prevents a random high score in file-leading/trailing silence.
    selected={}
    for i,score in enumerate(scores):
        if score<threshold or active[i]<.05 or not 0<=times[i]<song.audio.duration_sec: continue
        if intensity_curve is not None:
            if any(float(region['start_sec']) <= float(times[i]) < float(region['end_sec'])
                   for region in intensity_curve.get('rest_regions', [])):
                continue
            score *= _curve_factor(intensity_curve, float(times[i]), strength=intensity_curve_strength,
                                   span=intensity_curve_span, mode=intensity_curve_mode)
            if score < threshold:
                continue
        if i and score<=scores[i-1]: continue
        if i+1<len(scores) and score<scores[i+1]: continue
        beat=exact_candidate(times[i],song.tempo,song.first)
        if not 0<=seconds_at_beat(beat,song.tempo,song.first)<song.audio.duration_sec: continue
        if beat>=song.spans[-1].end: continue
        record={'beat':str(beat),'second':float(seconds_at_beat(beat,song.tempo,song.first)),
                'score':float(score),'type_probabilities':types[i].tolist(),'sustain_seconds':float(durations[i])}
        if beat not in selected or selected[beat]['score']<score: selected[beat]=record
    return {'schema':'maichart-full-song-blueprint-v1','onsets':[selected[b] for b in sorted(selected)],
            'threshold':threshold,'frame_seconds':.02,'musical_units':'whole_song',
            'quantization':'simple_rational_within_half_frame_uncertainty','limitations':[
                'Onset probabilities are weighted-training scores, not calibrated confidence.',
                'Broad types and durations are provisional cues; actual layout is decoded separately.']}


def arrange_span(model,codec,example,onsets,atoms,*,origin,temperature=.7,seed=0,device='cpu'):
    """Arrange the given whole-song skeleton without redeciding each chunk's rhythm.

    Exact position tokens are forced from the blueprint; the model composes
    note atoms. At token budget, return a continuation point without changing
    the musical segment or asking the planner to invent a new rhythm.
    """
    grammar=_NoteGrammar.from_atoms(atoms)
    batch=to_device(collate_form([example]),device)
    rng=torch.Generator(device=device).manual_seed(seed)
    max_tokens=model.config.max_tokens
    points=[];ids=[codec.bos_id]
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
            ids.extend(header);text=''
            for _ in range(256):
                allowed=[i for i,c in chars if grammar.can_continue(text+c)]
                if grammar.is_complete(text): allowed.append(codec.end_id)
                if not allowed: raise ValueError(f'No valid atom continuation for {text!r}')
                if len(ids)>=max_tokens-1:
                    if not grammar.is_complete(text):
                        # Do not commit half an atom or reroll a whole musical
                        # region. The caller continues this onset with fresh
                        # compute capacity and all previously accepted notes.
                        return points
                    token=codec.end_id
                else:
                    logits=next_logits()
                    values=logits[allowed]
                    if temperature<=0: token=allowed[int(values.argmax())]
                    else:
                        probabilities=torch.softmax(values/temperature,dim=-1)
                        token=allowed[int(torch.multinomial(probabilities,1,generator=rng))]
                ids.append(token)
                if token==codec.end_id:
                    points.append(EventPoint(beat,text));break
                text+=codec.tokens[token][2:]
            else: raise ValueError('Maximum note text length reached')
    return points


def generate_chart(model,planner,codec,song,condition,atoms,*,threshold=.5,memory_strength=1.,temperature=.7,seed=0,device='cpu',blueprint=None,
                   intensity_curve=None,intensity_curve_strength=1.0,intensity_curve_span=.5,intensity_curve_mode='follow'):
    blueprint=blueprint or plan_onsets(planner,song,condition,threshold=threshold,device=device,
                                       intensity_curve=intensity_curve,
                                       intensity_curve_strength=intensity_curve_strength,
                                       intensity_curve_span=intensity_curve_span,
                                       intensity_curve_mode=intensity_curve_mode)
    onsets=[Fraction(item['beat']) for item in blueprint['onsets']]
    if onsets!=sorted(set(onsets)): raise ValueError('Blueprint beats must be sorted and unique')
    accepted=[];records=[]
    for span in song.spans:
        desired=[b for b in onsets if span.start<=b<span.end]
        passes=0
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
                curve_values, curve_mask, curve_times = _curve_condition(
                    song, current, intensity_curve, strength=intensity_curve_strength,
                    span_limit=intensity_curve_span, mode=intensity_curve_mode)
                if curve_values is not None:
                    example['load_curve'] = curve_values.unsqueeze(0)
                    example['load_curve_condition_mask'] = curve_mask.unsqueeze(0)
                    example['load_curve_times_sec'] = curve_times.unsqueeze(0)
                from maichart.form.revision import DraftPoint
                future=[DraftPoint(b,None,True) for b in onsets if b>=current.start]
                example['skeleton']=torch.tensor(codec.bounded_memory(future,origin=current.start,
                    max_tokens=model.config.max_memory_tokens,newest=False))
                generated=arrange_span(model,codec,example,desired,atoms,origin=current.start,
                    temperature=temperature,seed=seed+span.index*1009+passes,device=device)
            finally: song.spans=source_spans
            if not generated: raise ValueError('Decoder made no progress within the configured token budget')
            accepted.extend(generated);desired=desired[len(generated):];passes+=1
        records.append({'span':span.index,'start':str(span.start),'end':str(span.end),
                        'resource_passes':passes,'onsets':sum(span.start<=b<span.end for b in onsets)})
        print('form_arrange '+json.dumps(records[-1]),flush=True)
    if not accepted: raise ValueError('Whole-song planner selected no onsets; inspect the plan before adjusting threshold')
    quality=assess_event_chart(accepted,song.tempo,first=song.first,duration_sec=song.audio.duration_sec,target_condition=condition)
    if not quality.get('syntax_valid'): raise ValueError('New-model output failed exact syntax verification')
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
    return accepted,{'blueprint':blueprint,'arrangement':records,'quality':quality,'memory_strength':memory_strength,
                     'intensity_curve':curve_report}


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
    candidate=arrange_span(model,codec,example,[p.beat for p in region],atoms,origin=start,
        temperature=temperature,seed=seed,device=device)
    if len(candidate)!=len(region): raise ValueError('Edit exceeds decoder budget; select a smaller region')
    config=PostProcessConfig(enable_conflict_detection=True,conflict_engine='miacode')
    result=apply_patch_candidate(points,EditPatch(start,end,tuple(candidate),'neural_revision'),
        lambda chart:_assessment(detect_chart_conflicts(chart,song.tempo,song.first,config)),
        budget=RevisionBudget(max_checks=max_checks,max_edits=8,deadline_secs=10.))
    report={'accepted':result.accepted,'reason':result.reason,'checks':result.checks,'edits':result.edits,
            'before':asdict(result.before) if result.before else None,'after':asdict(result.after) if result.after else None}
    # The unaccepted proposal is retained for review, never called preferred.
    report['proposal']=[p.to_dict() for p in candidate]
    return list(result.points),report
