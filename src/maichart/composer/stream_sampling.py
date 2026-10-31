"""Shared legal stream sampling primitives, independent of authored future."""
from fractions import Fraction
import torch

from maichart.event_timing import EventPoint
from .stream_data import NOTE, WAIT, EOS, second_at_beat
from .stream_timing import temper_grid_logits


def kind_logits_for_audio(logits, cursor, audio_end):
    values = logits.float().clone()
    values[:2] = -torch.inf
    if audio_end-cursor > 2:
        values[EOS] = -torch.inf
    else:
        values[WAIT] = -torch.inf
    return values


def note_delta_logits_for_audio(logits, deltas, cursor, audio_end, clock, duration, *, first_step):
    values = logits.float().clone()
    if len(deltas) != values.numel():
        raise ValueError('Delta logits and vocabulary have different sizes')
    rejected=[]
    for index, delta in enumerate(deltas):
        delta = delta if isinstance(delta,Fraction) else Fraction(str(delta))
        second = second_at_beat(clock,cursor+delta)
        rejected.append(index == 0 or delta < 0 or (delta == 0 and not first_step)
                or cursor+delta >= audio_end or second < 0 or second >= duration)
    return values.masked_fill(torch.as_tensor(rejected,device=values.device,dtype=torch.bool),-torch.inf)


def sample_logits(logits, temperature, generator):
    if not bool(torch.isfinite(logits).any()): return None
    if temperature <= 0: return int(logits.argmax().item())
    return int(torch.multinomial((logits.float()/temperature).softmax(-1),1,generator=generator).item())


def delta_sampling_logits(logits,temperature,repeat_delta_id=-1):
    """Temper continue/switch mass separately from the switch alternatives.

    A flat temperature on factorized probabilities would inflate the lone
    copy class just because switch probability is split over many deltas.
    At T=1 this is the identical legal conditional distribution.
    """
    scores=logits.float()
    index=int(repeat_delta_id)
    if temperature<=0 or index<0 or index>=scores.numel() or not bool(torch.isfinite(scores[index])):
        return scores if temperature<=0 else scores/temperature
    other=scores.clone();other[index]=-torch.inf
    if not bool(torch.isfinite(other).any()):return scores
    masses=torch.stack((scores[index],other.logsumexp(-1)))
    group=(masses/temperature).log_softmax(-1)
    result=(other/temperature).log_softmax(-1)+group[1]
    result[index]=group[0]
    return result


def sample_guarded_event(model, context, reference_phase, beat, guard, generator,
                         *, temperature=.8, attempts=8, accepts_extra=None, reference_phase_mask=None,
                         motion_phase_distribution=None,button_load=None,family_context=None,family_logits=None,
                         motion_keep_logit=None,motion_reference_shape=None,
                         template_keep_logit=None,template_reference_shape=None,duration_context=None):
    """Try at most attempts complete events; commit the accepted point once."""
    if not 1 <= attempts <= 8:
        raise ValueError('Guarded action attempts must be between 1 and 8')
    blocked = set()
    permanent = set()
    family=None;template_keep_override=None
    if getattr(getattr(model,'config',None),'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'):
        family,template_keep_override=model.sample_event_intent(context,family_logits=family_logits,
            template_keep_logit=template_keep_logit,template_reference_shape=template_reference_shape,
            temperature=temperature,generator=generator)
        if family is None:return None,0
    elif getattr(getattr(model,'config',None),'event_family_policy','legacy_v1') in ('audio_semantic_v2','audio_prior_transition_v3'):
        if family_context is None and family_logits is None:raise ValueError('Audio semantic decoding requires candidate music context')
        options={'family_logits':family_logits} if family_logits is not None else {}
        family=model.sample_event_family(family_context,temperature=temperature,generator=generator,**options)
        if family is None:return None,0
    for attempt in range(attempts):
        options = dict(reference_phase=reference_phase, temperature=temperature, generator=generator)
        if duration_context is not None:options['duration_context']=duration_context
        if family is not None:options.update(family_context=family_context,family_override=family)
        if family_logits is not None:options['family_logits']=family_logits
        if reference_phase_mask is not None:
            options['reference_phase_mask'] = reference_phase_mask
        if motion_phase_distribution is not None:
            options['motion_phase_distribution'] = motion_phase_distribution
        if button_load is not None:options['button_load']=button_load
        if motion_keep_logit is not None:options['motion_keep_logit']=motion_keep_logit
        if motion_reference_shape is not None:options['motion_reference_shape']=motion_reference_shape
        if template_keep_logit is not None:options['template_keep_logit']=template_keep_logit
        if template_reference_shape is not None:options['template_reference_shape']=template_reference_shape
        if template_keep_override is not None:options['template_keep_override']=template_keep_override
        if blocked:
            options['blocked_shape_phases'] = blocked
        event = model.decode_event(context, **options)
        if event is None and blocked != permanent:
            # Temporary phase traversal must not permanently forbid another
            # duration/trajectory/modifier choice at an otherwise usable phase.
            blocked = set(permanent)
            options['blocked_shape_phases'] = blocked
            event = model.decode_event(context, **options)
        if event is None:
            return None, attempt
        point = EventPoint(beat,event['text'])
        if (accepts_extra is None or accepts_extra(point)) and guard.accepts(point):
            guard.commit(point)
            return point, attempt
        if 'shape' in event and 'phase' in event:
            pair = ((0, 0, int(event['family'])) if int(event['shape'])==0 and 'family' in event
                    else (int(event['shape']), int(event['phase'])))
            blocked.add(pair)
            codec = getattr(model, 'codec', None)
            if codec is not None and pair[0] != codec.fallback_shape_id and all(
                    size == 1 for size in codec.detail_sizes[pair[0]]):
                # Only a singleton detail catalog proves that this pair cannot
                # become legal through a different detail sample.
                mask = codec.phase_equivalents(pair[0], pair[1], event['details'])
                equivalents = {(pair[0], phase) for phase, same in enumerate(mask) if same}
                permanent.update(equivalents)
                blocked.update(equivalents)
    return None, attempts
