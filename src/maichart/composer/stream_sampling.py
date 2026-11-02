"""Shared legal stream sampling primitives, independent of authored future."""
from fractions import Fraction
import torch

from maichart.event_timing import EventPoint
from .stream_data import NOTE, WAIT, EOS, second_at_beat
from .stream_timing import temper_grid_logits


class GreedyDecisionFrontier:
    """Bounded original-probability prefix replay, not exact global k-best."""
    def __init__(self):
        self.pending=[(0.,())];self.seen={()};self.decisions=[];self.queue_peak=1

    def begin(self):
        if not self.pending:return False
        self.pending.sort(key=lambda item:(-item[0],item[1]))
        self.path=self.pending.pop(0)[1];self.decisions=[]
        return True

    def choose(self,logits):
        scores=logits.float().reshape(-1)
        if not bool(torch.isfinite(scores).any()):raise ValueError('No legal deterministic action decision')
        position=len(self.decisions)
        selected=self.path[position] if position<len(self.path) else int(scores.argmax())
        if not bool(torch.isfinite(scores[selected])):raise ValueError('Invalid deterministic replay prefix')
        # The common successful first proposal needs no top-k transfer. Keep
        # detached action logits and expand alternatives only after rejection.
        self.decisions.append((selected,scores.detach()))
        return selected

    def reject(self):
        prefix=[];score=0.
        self.seen.add(tuple(decision[0] for decision in self.decisions))
        for position,(selected,scores) in enumerate(self.decisions):
            log_probs=scores.log_softmax(-1)
            count=min(8,int(torch.isfinite(scores).sum()))
            # Earlier forced choices belong to sibling partitions already
            # queued by their parent. Reversing them would replay failed leaves.
            alternatives=(sorted((int(index),float(log_probs[index])) for index in
                torch.topk(scores,count).indices.tolist()) if position>=len(self.path) else [])
            for alternative,alternative_probability in alternatives:
                branch=tuple(prefix+[alternative])
                if alternative!=selected and branch not in self.seen:
                    self.seen.add(branch);self.pending.append((score+alternative_probability,branch))
                    self.pending.sort(key=lambda item:(-item[0],item[1]))
                    self.pending=self.pending[:64]
            prefix.append(selected);score+=float(log_probs[selected])
        self.pending.sort(key=lambda item:(-item[0],item[1]))
        self.pending=self.pending[:64];self.queue_peak=max(self.queue_peak,len(self.pending))


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
                         template_keep_logit=None,template_reference_shape=None,duration_context=None,
                         execution_route_log_probs=None,execution_lane_bias=None,detail_mode=None):
    """Try at most attempts complete events; commit the accepted point once."""
    if not 1 <= attempts <= 8:
        raise ValueError('Guarded action attempts must be between 1 and 8')
    permanent = set()
    import inspect
    frontier=(GreedyDecisionFrontier() if temperature<=0 and
        'decision_chooser' in inspect.signature(model.decode_event).parameters else None)
    trace=[]
    guard.last_sampling_trace=trace
    for attempt in range(attempts):
        if frontier is not None and not frontier.begin():break
        # A rejected complete action supplies no proof that its music family
        # or keep decision is impossible. Redraw that entire joint decision at
        # the SAME onset, before considering another time. Only exact singleton
        # supports proven impossible below may be removed from later proposals.
        family=None;template_keep_override=None
        sampled_family=False
        if getattr(getattr(model,'config',None),'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'):
            sampled_family=True
            family,template_keep_override=model.sample_event_intent(context,family_logits=family_logits,
                template_keep_logit=template_keep_logit,template_reference_shape=template_reference_shape,
                temperature=temperature,generator=generator,
                **({'decision_chooser':frontier.choose} if frontier is not None else {}))
        elif getattr(getattr(model,'config',None),'event_family_policy','legacy_v1') in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4','shared_continuation_v1'):
            sampled_family=True
            if family_context is None and family_logits is None:raise ValueError('Audio semantic decoding requires candidate music context')
            family_options={'family_logits':family_logits} if family_logits is not None else {}
            if frontier is not None:
                scores=(model.event_family_scores(family_context) if family_logits is None else family_logits)
                scores=scores.reshape(-1).masked_fill(~model.semantic_family_supported,-torch.inf)
                family=frontier.choose(scores)+1 if bool(torch.isfinite(scores).any()) else None
            else:
                family=model.sample_event_family(family_context,temperature=temperature,generator=generator,**family_options)
        options = dict(reference_phase=reference_phase, temperature=temperature, generator=generator)
        if execution_route_log_probs is not None:options['execution_route_log_probs']=execution_route_log_probs
        if execution_lane_bias is not None:options['execution_lane_bias']=execution_lane_bias
        if detail_mode is not None:options['detail_mode']=int(detail_mode)
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
        if frontier is not None:options['decision_chooser']=frontier.choose
        if permanent and frontier is None:
            options['blocked_shape_phases'] = set(permanent)
        event = None if sampled_family and family is None else model.decode_event(context, **options)
        if event is None:
            trace.append(dict(attempt=attempt,beat=str(beat),family=None if family is None else int(family),
                template_keep=None if template_keep_override is None else bool(template_keep_override),
                accepted=False,rejection_reason='no_decodable_support'))
            if frontier is not None:
                frontier.reject()
                trace[-1].update(decision_count=len(frontier.decisions),queue_peak=frontier.queue_peak)
            continue
        point = EventPoint(beat,event['text'])
        extra_ok=accepts_extra is None or accepts_extra(point)
        accepted=extra_ok and guard.accepts(point)
        trace.append(dict(attempt=attempt,beat=str(beat),text=event['text'],
            family=int(event['family']) if 'family' in event else (None if family is None else int(family)),
            shape=int(event['shape']) if 'shape' in event else None,
            phase=int(event['phase']) if 'phase' in event else None,
            template_keep=None if template_keep_override is None else bool(template_keep_override),
            accepted=bool(accepted),rejection_reason=None if accepted else
                ('extra_constraint' if not extra_ok else getattr(guard,'last_reason',None) or 'physical_guard')))
        if accepted:
            if frontier is not None:trace[-1].update(decision_count=len(frontier.decisions),queue_peak=frontier.queue_peak)
            guard.commit(point)
            return point, attempt
        if frontier is not None:
            frontier.reject()
            trace[-1].update(deterministic_decisions=[decision[0] for decision in frontier.decisions],
                decision_count=len(frontier.decisions),queue_peak=frontier.queue_peak)
        if 'shape' in event and 'phase' in event:
            pair = ((0, 0, int(event['family'])) if int(event['shape'])==0 and 'family' in event
                    else (int(event['shape']), int(event['phase'])))
            codec = getattr(model, 'codec', None)
            if codec is not None and pair[0] != codec.fallback_shape_id and all(
                    size == 1 for size in codec.detail_sizes[pair[0]]):
                # Only a singleton detail catalog proves that this pair cannot
                # become legal through a different detail sample.
                mask = codec.phase_equivalents(pair[0], pair[1], event['details'])
                equivalents = {(pair[0], phase) for phase, same in enumerate(mask) if same}
                permanent.update(equivalents)
    return None, len(trace)
