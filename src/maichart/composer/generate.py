"""Generate a structured event chart with chart_composer_v1."""
from __future__ import annotations
import argparse,hashlib,json,math
from collections import Counter,OrderedDict
from itertools import groupby
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch

from maichart.event_timing import EventTempo,EventPoint,seconds_at_beat
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio
from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS
from maichart.generation.event_compile import compile_event_points
from maichart.generation.event_cli import _NoteGrammar
from maichart.stream.grounded_inference import IncrementalState
from maichart.generation.timeline import normalize_tempo_map
from maichart.stream.audio_bounds import generation_bounds
from maichart.generation.post_process import PostProcessConfig,process_chart_points
from .checkpoint import load
from .action_guidance import (action_logits_with_touch_guard,
    TOUCH_HISTORY_GUIDANCE_POLICY,TOUCH_HISTORY_LOG_ODDS_BOUND)
from . import ACTION_SAMPLING_POLICY
from .candidates import (COMPOSER_AUDIO_FEATURE_POLICY,candidate_beats,
                          candidate_features,composer_audio_nodes,interp_candidate_audio, MUSIC_CANDIDATE_POLICY_V6)
from .model import ChartComposer, WORKLOAD_PROFILES, MUSIC_PROFILES, GROUNDED_PROFILE
from .generation_settings import resolve_seed,resolve_condition
from maichart.tempo_clock import TempoClock
from .acoustics import AcousticContext
from .workload import build_workload_targets, causal_ledger, event_statistics
from .guard import EventGuard
from .events import (HEAD_PADS, event_metadata, head_timing_features,
                     history_event_features)
from .candidates import MUSIC_CANDIDATE_POLICY

GROUNDED_POINTER_SAMPLING_POLICY = 'joint_t1_bin_mass_within_bin_temperature_v1'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bpm',type=float);p.add_argument('--first',type=float,default=0.);p.add_argument('--level',type=float,default=13.)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=None,
        help='omit for a new random seed; supply an integer for reproducibility');p.add_argument('--device',default='auto')
    p.add_argument('--pointer-temperature',type=float,default=None)
    p.add_argument('--touch-history-guard',action=argparse.BooleanOptionalAction,default=True,
        help='audio-grounded positive-temperature decoding: limit symbolic Touch probability feedback')
    p.add_argument('--touch-history-bound',type=float,default=TOUCH_HISTORY_LOG_ODDS_BOUND,
        help='Touch positive log-odds amplification cap in [0,4]; lower means stronger protection, not a Touch ratio')
    p.add_argument('--action-temperature',type=float,default=None)
    p.add_argument('--seed-randomized',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--max-events',type=int,default=0);p.add_argument('--condition-file');p.add_argument('--tempo-map')
    p.add_argument('--curve-strength',type=float,default=1.)
    p.add_argument('--pointer-history-strength',type=float,default=1.,
        help='pointer audio anchor: 1 preserves full-context route, 0 uses audio-only route')
    p.add_argument('--min-same-head-ms',type=float,default=50.,
        help='minimum same-pad head spacing in milliseconds; 0 disables this generation safety policy')
    p.add_argument('--intercept-tap-on-slide',action=argparse.BooleanOptionalAction,default=False,
        help='intercept TapOnSlide in EventGuard and post-process; default only reports it')
    return p


def _weight_fingerprint(state):
    digest=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        digest.update(key.encode())
        if not torch.is_tensor(tensor):
            digest.update(json.dumps(tensor,sort_keys=True,separators=(',',':')).encode());continue
        digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
        values=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(values))
    return digest.hexdigest()


def _device(value):return torch.device('cuda' if value=='auto' and torch.cuda.is_available() else ('cpu' if value=='auto' else value))


def _interp_features(values,times,song,candidates,previous,*,clock=None,feature_policy=None,rich_time=False):
    clock=clock or TempoClock(song.tempo,song.first)
    seconds=np.asarray([clock.at(b) for b in candidates],dtype=np.float64)
    acoustic=interp_candidate_audio(values,times,seconds,suppress_attack_bridges=feature_policy=='attack_confidence_v2')
    time_part=candidate_features(candidates,previous,song.tempo,song.first,clock=clock,candidate_seconds=seconds,rich_time=rich_time)
    return torch.from_numpy(np.concatenate((acoustic,time_part),axis=1))


def _history_features(history, atoms, tempo, first, action_slots, *, clock=None,preceding_beat=Fraction(0),parsed=False):
    rows=[];previous_beat=preceding_beat
    previous_second=float(clock.at(previous_beat) if clock is not None else seconds_at_beat(previous_beat,tempo,first))
    for ids, point in history:
        text='/'.join(atoms[index-1] for index in ids if 0 < index <= len(atoms))
        second=float(clock.at(point) if clock is not None else seconds_at_beat(point,tempo,first))
        rows.append(history_event_features(text, point-previous_beat, second-previous_second, parsed=parsed))
        previous_beat=point; previous_second=second
    return rows


def _head_metadata(atoms,grammar):
    pads=[f'{group}{lane}' for group in 'ABDE' for lane in range(1,9)]+['C']
    pad_bits={pad:1<<i for i,pad in enumerate(pads)}
    def encode(heads):
        bits=0
        for head in heads:
            head=str(head);bits |= pad_bits.get(f'A{head}' if head.isdigit() else head,0)
        return bits
    bits=np.asarray([0]+[encode(grammar.heads.get(atom,())) for atom in atoms],dtype=np.uint64)
    sizes=np.asarray([int(value).bit_count() for value in bits])
    is_touch=torch.tensor([False]+[a[0] in 'ABCDE' for a in atoms],dtype=torch.bool)
    is_slide=torch.tensor([False]+[any(c in a for c in '-<>^pqszvwV') for a in atoms],dtype=torch.bool)
    is_hold=torch.tensor([False]+[('h' in a and a[0] not in 'ABCDE' and not any(c in a for c in '-<>^pqszvwV')) for a in atoms],dtype=torch.bool)
    is_tap=torch.tensor([False]+[not (t or s or h) for t,s,h in zip(is_touch[1:].tolist(),is_slide[1:].tolist(),is_hold[1:].tolist())],dtype=torch.bool)
    # Ring choice has the same calibration problem as note kind: pooling all
    # Touch atoms before top-k can erase B/C/D while eight common E pads occupy
    # the retained tail. Preserve learned ring mass; temperature/truncation
    # choose the pad, modifier and duration within each ring. This is not a
    # quota: the model may still assign a ring zero/negligible probability.
    touch_rings=tuple(torch.tensor([False]+[a[0]==ring for a in atoms],dtype=torch.bool)
                      for ring in 'ABCDE')
    groups=(is_tap,is_hold,is_slide,*touch_rings)
    return bits,sizes,encode,{atom:i+1 for i,atom in enumerate(atoms)},groups


def _truncate_probabilities(probabilities, *, top_p=.90, min_p=.04, max_k=64):
    """Prune the noise tail within a distribution while preserving tied options."""
    if probabilities.numel() <= 16:
        return probabilities
    top_prob = probabilities.max()
    if not torch.isfinite(top_prob) or top_prob <= 0.:
        return probabilities
    sorted_probs, _ = probabilities.sort(descending=True)
    cumsum = sorted_probs.cumsum(-1)
    included = (cumsum - sorted_probs) < top_p
    included[0] = True
    if 0 < max_k < sorted_probs.numel():
        included[max_k:] = False
    nucleus_floor = sorted_probs[included].min()
    threshold = torch.maximum(top_prob * min_p, nucleus_floor) - 1e-12
    keep = probabilities >= threshold
    if not keep.any():
        return probabilities
    filtered = torch.where(keep, probabilities, torch.zeros_like(probabilities))
    return filtered / filtered.sum().clamp_min(1e-12)


def _grounded_pointer_probabilities(logits, bins, temperature, *, eligible=None,
                                    top_p=.92, min_p=.05, max_k=16):
    """Keep current legal T=1 bin mass; temper/prune only within each bin.

    Retry callers mask the original joint logits again, so rejected candidates
    lose their mass before the remaining bins are conditionally normalized.
    """
    if logits.ndim != 1 or bins.shape != logits.shape:
        raise ValueError('Pointer logits and bins must be matching vectors')
    if not math.isfinite(float(temperature)) or temperature < 0:
        raise ValueError('Pointer temperature must be finite and nonnegative')
    logits = logits.float()
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError('Pointer logits contain NaN or positive infinity')
    if bins.dtype != torch.long or bool(((bins < 0) | (bins >= 6)).any()):
        raise ValueError('Pointer bins must be long indices in [0, 6)')
    if eligible is not None:
        if eligible.shape != logits.shape or eligible.dtype != torch.bool:
            raise ValueError('Pointer eligibility must be a matching boolean vector')
        logits = logits.masked_fill(~eligible, float('-inf'))
    valid = torch.isfinite(logits)
    probabilities = torch.zeros_like(logits)
    if not bool(valid.any()):
        return probabilities
    if temperature == 0:
        probabilities[logits.argmax()] = 1.
        return probabilities
    joint = logits.softmax(-1)
    for group in range(6):
        members = valid & bins.eq(group)
        if not bool(members.any()):
            continue
        values = logits[members]
        within = ((values-values.max()) / max(float(temperature), 1e-6)).softmax(-1)
        within = _truncate_probabilities(within, top_p=top_p, min_p=min_p, max_k=max_k)
        probabilities[members] = joint[members].sum() * within
    return probabilities


def _category_truncate_actions(masked_logits, temperature, groups):
    """Preserve T=1 kind/ring mass, truncating only inside each disjoint group.

    Legacy callers may still supply four coarse kind masks. Production uses
    three button kinds followed by the five Touch rings.
    """
    macro_probs = masked_logits.softmax(-1)
    if groups is None:
        return (masked_logits / max(float(temperature), 1e-6)).softmax(-1)
    out = torch.zeros_like(macro_probs)
    out[0] = macro_probs[0]
    limits=(16,12,16)+tuple(8 for _ in groups[3:])
    for mask, max_k in zip(groups, limits):
        m = mask.to(masked_logits.device) & masked_logits.isfinite()
        if not m.any():
            continue
        cat_mass = macro_probs[m].sum()
        if cat_mass <= 0:
            continue
        # Normalize inside the category: a global low-temperature softmax can
        # underflow every atom of a valid category before its mass is restored.
        sub_logits = masked_logits[m]
        sub = ((sub_logits-sub_logits.max()) / max(float(temperature), 1e-6)).softmax(-1)
        sub = _truncate_probabilities(sub, top_p=.85, min_p=.08, max_k=max_k)
        out[m] = cat_mass * sub
    total = out.sum()
    return out / total if total > 0. else (masked_logits / max(float(temperature), 1e-6)).softmax(-1)


def _candidate_playability(candidates, base_playable, committed_previous, policy):
    """Apply the existing phase policy relative to committed chart history."""
    if policy != MUSIC_CANDIDATE_POLICY_V6:
        return base_playable
    canonical_denoms={1,2,3,4,6,8}
    phase_playable=[ok and beat.denominator in canonical_denoms
        and (committed_previous is None or (beat-committed_previous).denominator in canonical_denoms)
        for ok,beat in zip(base_playable,candidates)]
    # Preserve the existing fallback policy; only its history anchor changes.
    return phase_playable if any(phase_playable) else base_playable


def _compile_composer_chart(points, tempo, first, condition):
    # Retain every input tempo anchor even if EOS/max_events stops note output
    # earlier. Constant-tempo exports keep their existing one-beat tail.
    last_anchor=max([Fraction(0), *(point.beat for point in points),
                     *(change.beat for change in tempo)])
    export_end=Fraction(math.ceil(last_anchor)+1)
    return compile_event_points(points,tempo_changes=tempo,first=first,
        end_beat=export_end,level=str(condition.level),difficulty=condition.difficulty)


def _sample_actions(logits, atoms, grammar, temperature, generator, occupied=(), blocked=(), max_heads=2,diagnostics=None,
                    slot_decoder=None,blocked_first=(),head_metadata=None,canonical_order=False):
    selected=[]
    used_buttons=set(occupied)
    for slot,row in enumerate(logits):
        if slot_decoder is not None: row=slot_decoder(slot,selected)
        if head_metadata is None: head_metadata=_head_metadata(atoms,grammar)
        if len(head_metadata)==5:
            bits,sizes,encode,lookup,groups=head_metadata
        else:
            bits,sizes,encode,lookup=head_metadata;groups=None
        used=np.uint64(encode(used_buttons));blocked_bits=np.uint64(encode(blocked))
        mask_np=((bits & (used|blocked_bits))==0)&(sizes+int(used).bit_count()<=max_heads)&(sizes>0)
        mask_np[0]=slot>0
        if canonical_order and selected:
            mask_np[1:lookup[selected[-1]]+1]=False
        if slot==0 and blocked_first: mask_np[list(blocked_first)]=False
        for atom in selected: mask_np[lookup[atom]]=False
        allowed=np.flatnonzero(mask_np).tolist()
        if not allowed:
            break
        values=row.float().clone()
        mask=torch.full_like(values,float('-inf'))
        mask[torch.tensor(allowed,device=values.device)]=values[torch.tensor(allowed,device=values.device)]
        if temperature > 0:
            probabilities=_category_truncate_actions(mask,temperature,groups)
            if diagnostics is not None and slot==0:
                diagnostics['action_top_probability_sum']+=probabilities.max().detach()
                diagnostics['action_decisions']+=1
            choice=int(torch.multinomial(probabilities,1,generator=generator).item())
        else:
            choice=int(mask.argmax().item())
        if choice == 0:
            break
        selected_atom=atoms[choice-1]
        selected.append(selected_atom)
        used_buttons.update(f'A{head}' if str(head).isdigit() else str(head)
                            for head in (grammar.heads.get(selected_atom) or ()))
    return selected


def _active_constraints(state, beat, tempo, first, *, clock=None):
    """Return concrete source heads still occupied at the proposed beat.

    SustainStateIndex also exposes a hand-space pad coverage feature. That is
    useful as a soft model input but must not be treated as exact key
    ownership for a hard same-head filter.
    """
    second=float(clock.at(beat) if clock is not None else seconds_at_beat(beat,tempo,first)); result=set(); blocked=set(); active_hands=0
    for marker in getattr(state,'markers',()):
        onset=float(marker.get('second',0.)); end=float(marker.get('end_second',-1.))
        if onset > second or end <= second + 1e-8:
            continue
        kind=marker.get('type')
        if kind == 'hold':
            lane=marker.get('lane')
            if lane is not None:
                result.add(f'A{int(lane)}'); blocked.update({str(int(lane)),f'A{int(lane)}'})
        elif kind=='touch_hold' and marker.get('touch_pad'):
            result.add(str(marker['touch_pad'])); blocked.add(str(marker['touch_pad']))
        # Slide paths and waiting intervals are not exclusive key ownership.
        # The candidate transaction uses the existing runtime judge instead.
    return result,blocked,active_hands


def _active_heads(state, beat, tempo, first, *, clock=None):
    """Backward-compatible view of currently held concrete heads."""
    return _active_constraints(state, beat, tempo, first, clock=clock)[0]


def density_observation(points, clock):
    seconds=np.asarray([clock.at(p.beat) for p in points], dtype=np.float64)
    intervals=np.diff(seconds)
    last={};same=[]
    for point,second in zip(points,seconds):
        for pad in event_metadata(point.note_text).head_pad_indices:
            if pad in last: same.append(second-last[pad])
            last[pad]=second
    return {'min_event_interval_ms':float(intervals.min()*1000) if len(intervals) else None,
        'min_same_head_interval_ms':float(min(same)*1000) if same else None,
        'same_head_intervals_below_50ms':int(sum(value<.05-1e-9 for value in same)),
        'event_intervals_below_25ms':int(np.count_nonzero(intervals<.025-1e-9))}


def blend_pointer_logits(full_logits, audio_logits, strength):
    """Geometric probability interpolation; endpoints preserve exact logits.

    This scales the full-vs-audio difference; it does not put an absolute
    bound on that difference, and does not establish musical quality.
    """
    strength=float(strength)
    if not math.isfinite(strength) or not 0. <= strength <= 1.:
        raise ValueError('pointer_history_strength must be between 0 and 1')
    if strength==1.:return full_logits
    if strength==0.:return audio_logits
    if full_logits.shape!=audio_logits.shape:
        raise ValueError('Full and audio pointer logits must have the same shape')
    return audio_logits + strength * (full_logits - audio_logits)


def _pointer_js(left, right):
    left=left.float().softmax(-1); right=right.float().softmax(-1)
    mean=(left+right)*.5
    return (.5*(left*(left.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
            + .5*(right*(right.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1))


@torch.inference_mode()
def generate(args):
    if not math.isfinite(args.temperature) or args.temperature<0:
        raise ValueError('temperature 必须是非负有限值；0 为贪心解码')
    if int(getattr(args,'max_events',0)) < 0:
        raise ValueError('max_events 必须是非负整数')
    pointer_value=getattr(args,'pointer_temperature',None)
    action_value=getattr(args,'action_temperature',None)
    pointer_temperature=float(pointer_value if pointer_value is not None else args.temperature)
    action_temperature=float(action_value if action_value is not None else args.temperature)
    for name,value in (('pointer_temperature',pointer_temperature),('action_temperature',action_temperature)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} 必须是非负有限值；0 为贪心解码')
    curve_strength=float(getattr(args,'curve_strength',1.))
    if not math.isfinite(curve_strength) or not 0 <= curve_strength <= 2:
        raise ValueError('curve_strength must be between 0 and 2')
    pointer_history_strength=float(getattr(args,'pointer_history_strength',1.))
    if not math.isfinite(pointer_history_strength) or not 0. <= pointer_history_strength <= 1.:
        raise ValueError('pointer_history_strength must be between 0 and 1')
    min_same_head_ms=float(getattr(args,'min_same_head_ms',50.))
    if not math.isfinite(min_same_head_ms) or not 0 <= min_same_head_ms <= 200:
        raise ValueError('min_same_head_ms must be between 0 and 200')
    seed,seed_randomized=resolve_seed(args.seed)
    seed_randomized=seed_randomized or bool(getattr(args,'seed_randomized',False))
    condition,condition_source=resolve_condition(args.level,condition_file=args.condition_file)
    touch_history_bound=float(getattr(args,'touch_history_bound',TOUCH_HISTORY_LOG_ODDS_BOUND))
    if not math.isfinite(touch_history_bound) or not 0<=touch_history_bound<=4:
        raise ValueError('touch_history_bound must be finite and between 0 and 4')
    if condition.difficulty not in range(1,7):
        raise ValueError('difficulty must be an integer from 1 to 6')
    device=_device(args.device);model,payload=load(args.checkpoint,device=device);model.eval()
    weights_sha256=_weight_fingerprint(payload['model_state_dict'])
    # The model owns copied parameters. Release file mappings now so Windows
    # training can replace composer_last.pt while a generation is running.
    payload.pop('model_state_dict',None)
    atoms=payload.get('atom_catalog') or []
    atom_to_id={atom:i+1 for i,atom in enumerate(atoms)}
    learned=model.config.architecture_profile in WORKLOAD_PROFILES
    music_v2=model.config.architecture_profile in MUSIC_PROFILES
    grounded=model.config.architecture_profile==GROUNDED_PROFILE
    touch_history_guard_requested=bool(getattr(args,'touch_history_guard',True))
    touch_history_guard=touch_history_guard_requested and grounded and action_temperature>0
    if not music_v2 and pointer_history_strength!=1.:
        raise ValueError('pointer_history_strength below 1 requires a music_context_v2 checkpoint')
    feature_policy=payload.get('audio_feature_policy','legacy')
    candidate_policy=payload.get('candidate_policy') or (MUSIC_CANDIDATE_POLICY_V6 if music_v2 else ('absolute_recovery_v4' if learned else ('phase_v3' if feature_policy=='attack_confidence_v2' else 'relative_v1')))
    trained_candidate_policy=candidate_policy
    if music_v2 and candidate_policy==MUSIC_CANDIDATE_POLICY:
        candidate_policy=MUSIC_CANDIDATE_POLICY_V6
    parsed_features=bool(model.config.head_timing_dim)
    if not atoms: raise ValueError('Composer checkpoint has no atom catalog')
    audio=extract_rational_audio(args.audio,cache_dir=Path(args.checkpoint).parent/'audio_cache',evidence_profile='balanced_evidence_v1')
    tempo=(normalize_tempo_map(json.loads(Path(args.tempo_map).read_text(encoding='utf-8-sig')))
           if args.tempo_map else [EventTempo(Fraction(0),args.bpm or audio.tempo_bpm)])
    song=organize_song(audio,tempo,args.first,max_seconds=999.,quality_profile='musical_evidence_v1')
    clock=TempoClock(song.tempo,song.first)
    values,times=composer_audio_nodes(song,policy=COMPOSER_AUDIO_FEATURE_POLICY if learned else 'legacy')
    bounds=generation_bounds(audio,trim_exact_silence=True);end_sec=bounds['effective_end_sec']
    points=[];history=[];previous=None;max_events=args.max_events or 100000
    action_index=IncrementalState(song.tempo,song.first)
    acoustic_context=AcousticContext(values,times,audio.duration_sec,validated_bank=learned) if learned else None
    if grounded:
        from .audio_lens import LensBank
        from .planning import PlanGrid
        lens_bank=LensBank(values,times,audio.duration_sec)
        plan_grid=PlanGrid(clock,end_sec,acoustic_context)
        planner_audio=torch.from_numpy(plan_grid.audio)[None].to(device)
        planner_mask=torch.zeros(planner_audio.shape[:2],device=device,dtype=torch.bool)
        planner_prediction=model.planner(planner_audio,condition.to_vector(device=device)[None],planner_mask)
        lens_cache=OrderedDict()
        lens_cache_hits=lens_cache_misses=0
    intercept_tap_on_slide=bool(getattr(args,'intercept_tap_on_slide',False))
    guard=EventGuard(song.tempo,song.first,end_sec,min_same_head_sec=min_same_head_ms/1000.,intercept_tap_on_slide=intercept_tap_on_slide)
    last_head_seconds=np.full(len(HEAD_PADS),np.nan)
    last_head_beats=np.full(len(HEAD_PADS),np.nan)
    history_seconds=[];history_heads=[];head_prefix=[0.];skipped_candidates=0;stop_reason='audio_range_exhausted'
    context_coverage=[]
    grammar=_NoteGrammar.from_atoms(atoms)
    head_metadata=_head_metadata(atoms,grammar)
    touch_mask=torch.tensor([False]+[atom[0] in 'ABCDE' for atom in atoms],dtype=torch.bool,device=device)
    rng=torch.Generator(device=device).manual_seed(seed)
    sampling={'pointer_top_probability_sum':torch.zeros((),device=device),
              'action_top_probability_sum':torch.zeros((),device=device),'fallback_count':0,
              'pointer_decisions':0,'action_decisions':0,
              'audio_pointer_js_sum':torch.zeros((),device=device),'audio_pointer_js_count':0,
              'touch_guard_proposals':0,'touch_guard_safety_rejections':0}
    print('composer_generation_runtime '+json.dumps({'seed':seed,'seed_randomized':seed_randomized,
        'temperature':args.temperature,'pointer_temperature':pointer_temperature,
        'action_temperature':action_temperature,
        'pointer_history_strength':pointer_history_strength,
        'sampling_mode':('greedy' if pointer_temperature==0 and action_temperature==0 else 'temperature'),
        'condition':condition.to_dict(),'condition_source':condition_source,
        'architecture_profile':model.config.architecture_profile,
        'head_timing_dim':model.config.head_timing_dim,'candidate_policy':candidate_policy,
        'trained_candidate_policy':trained_candidate_policy,'min_same_head_ms':min_same_head_ms,'intercept_tap_on_slide':intercept_tap_on_slide,
        'action_sampling_policy':ACTION_SAMPLING_POLICY,
        'pointer_sampling_policy':GROUNDED_POINTER_SAMPLING_POLICY if grounded else 'global_temperature_truncation_v1',
        'touch_history_guard':touch_history_guard,
        'touch_history_bound_requested':touch_history_bound,
        'touch_history_guidance_policy':TOUCH_HISTORY_GUIDANCE_POLICY if touch_history_guard else 'off',
        'checkpoint_epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps'),
        'weights_sha256':weights_sha256}),flush=True)
    control=condition.to_vector(device=device)[None]
    while len(points)<max_events:
        candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,audio_end_sec=end_sec,
            candidate_count=model.config.candidate_count,clock=clock,policy=candidate_policy)
        if not candidates:break
        constraints=([_active_constraints(action_index,cand,song.tempo,song.first,clock=clock) for cand in candidates]
                      if getattr(action_index,'markers',None) else [(set(),set(),0) for _ in candidates])
        base_playable=[len(held)<2 and 0.<=float(clock.at(beat))<end_sec
                       for beat,(held,_blocked,_active_hands) in zip(candidates,constraints)]
        feature_previous=history[-1][1] if history else None
        playable=_candidate_playability(candidates,base_playable,feature_previous,candidate_policy)
        if not any(playable):
            previous=candidates[0]
            skipped_candidates+=1
            continue
        cand_mask=torch.tensor([[not ok for ok in playable]],dtype=torch.bool,device=device)
        feats=_interp_features(values,times,song,candidates,feature_previous,clock=clock,
            feature_policy=feature_policy,rich_time=learned)[None].to(device)
        candidate_seconds=np.asarray([clock.at(beat) for beat in candidates])
        extra={}
        if learned:
            from .workload import prefix_ledger
            extra={'workload_audio':torch.from_numpy(acoustic_context.at(candidate_seconds))[None].to(device),
                'ledger':torch.from_numpy(prefix_ledger(history_seconds,head_prefix,candidate_seconds))[None].to(device),
                'curve_strength':curve_strength}
            if music_v2:
                committed_second=history_seconds[-1] if history_seconds else 0.
                extra.update(interval_audio=torch.from_numpy(acoustic_context.interval(committed_second,candidate_seconds))[None].to(device),
                             trailing_audio=torch.from_numpy(acoustic_context.trailing(candidate_seconds))[None].to(device))
            if model.config.head_timing_dim:
                extra['head_timing']=torch.from_numpy(head_timing_features(last_head_seconds,last_head_beats,
                    candidate_seconds,[float(beat) for beat in candidates]))[None].to(device)
            if grounded:
                indices,mix=plan_grid.locate(candidate_seconds)
                keys=[float(t) for t in candidate_seconds]
                missing=list(dict.fromkeys(key for key in keys if key not in lens_cache))
                if missing:
                    new_patches,new_valid=lens_bank.query(missing)
                    encoded=model.lens(torch.from_numpy(new_patches)[None].to(device),
                        torch.from_numpy(new_valid)[None].to(device))[0]
                    for key,value in zip(missing,encoded):lens_cache[key]=value.detach().clone()
                lens_cache_misses+=len(missing);lens_cache_hits+=len(keys)-len(missing)
                embedded=torch.stack([lens_cache[key] for key in keys])[None]
                for key in keys:lens_cache.move_to_end(key)
                while len(lens_cache)>8192:lens_cache.popitem(last=False)
                extra.update(planner_audio=planner_audio,planner_mask=planner_mask,
                    planner_indices=torch.from_numpy(indices)[None].to(device),
                    planner_mix=torch.from_numpy(mix)[None].to(device),planner_override=planner_prediction,
                    lens_embeddings=embedded)
        history_tail=history[-model.config.history_events:]
        if history_tail:
            hist_atoms=torch.tensor([row for row,_ in history_tail],dtype=torch.long,device=device)[None]
            preceding_beat=history[-model.config.history_events-1][1] if len(history)>model.config.history_events else Fraction(0)
            hist_features=torch.tensor(_history_features(history_tail,atoms,song.tempo,song.first,model.config.action_slots,
                clock=clock,preceding_beat=preceding_beat,parsed=parsed_features),dtype=torch.float32,device=device)[None]
        else:
            hist_atoms=torch.zeros((1,0,model.config.action_slots),dtype=torch.long,device=device)
            hist_features=torch.zeros((1,0,8),dtype=torch.float32,device=device)
        committed_beat=history[-1][1] if history else Fraction(0)
        states,mask,metadata=action_index.at(committed_beat,include_current=True)
        active=states[~mask]
        action=active.mean(0).to(device) if active.numel() else torch.zeros(50,device=device)
        output=model(feats,hist_atoms,control,action_state=action[None],history_features=hist_features,
                     candidate_mask=cand_mask,
                     history_padding_mask=torch.zeros((1,hist_atoms.size(1)),dtype=torch.bool,device=device),
                     decode_actions=False,audio_auxiliary=music_v2,**extra)
        full_pointer=output['candidate_logits'][0].float()
        audio_pointer=output['audio_candidate_logits'][0].float() if 'audio_candidate_logits' in output else full_pointer
        if music_v2 and 'audio_candidate_logits' in output:
            sampling['audio_pointer_js_sum'] += _pointer_js(full_pointer[None],audio_pointer[None]).sum().detach()
            sampling['audio_pointer_js_count'] += 1
        logits=blend_pointer_logits(full_pointer,audio_pointer,pointer_history_strength)
        if not grounded:
            logits=logits/max(pointer_temperature,1e-6)
        if grounded and (torch.isnan(logits).any() or torch.isposinf(logits).any()):
            raise ValueError('Pointer logits contain NaN or positive infinity')
        if not all(playable):
            logits=logits.masked_fill(cand_mask[0],float('-inf'))
        selected=[]
        for attempt in range(min(len(candidates),8)):
            if not torch.isfinite(logits).any(): break
            pointer_probs=(_grounded_pointer_probabilities(logits,output['candidate_bins'][0],pointer_temperature)
                           if grounded else _truncate_probabilities(logits.softmax(-1),top_p=.92,min_p=.05,max_k=16)
                           if pointer_temperature>0 else None)
            if pointer_temperature>0:
                sampling['pointer_top_probability_sum']+=pointer_probs.max().detach()
                sampling['pointer_decisions']+=1
            choice=int(torch.multinomial(pointer_probs,1,generator=rng).item()) if pointer_temperature>0 else int(logits.argmax())
            beat=candidates[choice]
            cand_sec=float(clock.at(beat))
            decision=model.decode_candidate(output['candidate_context'],torch.tensor([choice],device=device),
                project_actions=not learned,audio_context=output.get('audio_action_context'))
            occupied,blocked,_=constraints[choice]
            cooldown_blocked=set(blocked)|{pad for pad,prior in guard.last_head_seconds.items()
                if cand_sec-prior<guard.min_same_head_sec-1e-9}
            audio_act=decision['audio_action_context'][0] if 'audio_action_context' in decision else None
            decoder_diagnostics={}
            decoder=(lambda slot,chosen: action_logits_with_touch_guard(model,decision['action_context'][0],slot,
                [atom_to_id[atom] for atom in chosen],audio_context=audio_act,
                history_strength=pointer_history_strength,enabled=touch_history_guard,touch_mask=touch_mask,
                diagnostics=decoder_diagnostics,log_odds_bound=touch_history_bound)) if learned else None
            blocked_first=set()
            for retry in range(6):
                decoder_diagnostics.clear()
                slot_rows=(torch.zeros((model.config.action_slots,len(atoms)+1),device=device)
                           if learned else decision['action_logits'][0])
                proposed=_sample_actions(slot_rows,atoms,grammar,action_temperature,rng,
                    occupied=occupied,blocked=cooldown_blocked,max_heads=2,diagnostics=sampling,
                    slot_decoder=decoder,blocked_first=blocked_first,head_metadata=head_metadata,
                    canonical_order=music_v2)
                if proposed:
                    guided=bool(decoder_diagnostics.get('applied'))
                    sampling['touch_guard_proposals']+=int(guided)
                    event_proposal=EventPoint(beat,'/'.join(proposed))
                    accepted=(guard.accepts(event_proposal,protect_reported_hard=True) if guided
                              else guard.accepts(event_proposal))
                    if accepted:
                        selected=proposed;break
                    if guided and guard.last_reason=='guided_proposal_new_reported_hard_conflict':
                        sampling['touch_guard_safety_rejections']+=1
                sampling['fallback_count']+=1
                if proposed:
                    blocked_first.add(atom_to_id[proposed[0]])
                    if guard.last_reason=='repeated_head_too_soon':
                        for head in event_metadata('/'.join(proposed)).heads:
                            cooldown_blocked.add(f'A{head}' if str(head).isdigit() else str(head))
            if selected: break
            logits[choice]=float('-inf')
        if not selected:
            # Advance the search cursor; committed chart/history remain intact.
            # A temporarily blocked action must never end the whole song.
            previous=candidates[0]
            skipped_candidates+=1
            continue
        text='/'.join(selected)
        event=EventPoint(beat,text)
        points.append(event); action_index.append(event);guard.commit(event)
        for pad in event_metadata(text).head_pad_indices:
            last_head_seconds[pad]=float(clock.at(beat));last_head_beats[pad]=float(beat)
        history_seconds.append(float(clock.at(beat)));history_heads.append(event_statistics(text)[0])
        head_prefix.append(head_prefix[-1]+history_heads[-1])
        if history_tail:
            context_coverage.append((float(clock.at(history_tail[-1][1])-clock.at(history_tail[0][1])),
                                     float(history_tail[-1][1]-history_tail[0][1])))
        else:
            context_coverage.append((0.,0.))
        atom_ids=tuple([atom_to_id[atom] for atom in selected]+[0]*(model.config.action_slots-len(selected)))
        history.append((atom_ids,beat));previous=beat
        eos_probability=float(decision['eos_logits'][0].sigmoid())
        beat_sec=float(clock.at(beat))
        if beat_sec>=end_sec or (eos_probability >= (.98 if learned else .9)
                and (learned or beat_sec>=max(0.,end_sec-2.))):
            stop_reason='learned_eos'
            break
    raw_density=density_observation(points,clock)
    playability = process_chart_points(
        points, song.tempo, args.first,
        PostProcessConfig(
            enable_conflict_detection=True,
            detect_hard_conflicts=True,
            detect_soft_conflicts=False,
            conflict_engine='miacode',
            conflict_repair_mode='local',
            conflict_retries=0,
            note_repair_mode='delete',
            conflict_kind_modes={
                'TapOnSlide':'enforce' if intercept_tap_on_slide else 'report', 'Overlap':'enforce',
                'MultiTouch':'report', 'SlideHeadTap':'report', 'SlideTooFast':'report',
            },
        ),
    )
    points = playability.points
    text,_=_compile_composer_chart(points,song.tempo,args.first,condition)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    configurations=Counter(point.note_text for point in points)
    report={'schema':'maichart-chart-composer-generation-v2','family':'chart_composer_v1','event_count':len(points),
            'architecture_profile':model.config.architecture_profile,
            'checkpoint':str(Path(args.checkpoint).resolve()),'device':str(device),'seed':seed,
            'seed_randomized':seed_randomized,'temperature':args.temperature,
            'pointer_temperature':pointer_temperature,'action_temperature':action_temperature,
            'condition':condition.to_dict(),'condition_source':condition_source,
            'checkpoint_epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps'),
            'weights_sha256':weights_sha256,'chart_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'events_sha256':hashlib.sha256(json.dumps([(str(p.beat),p.note_text) for p in points],
                ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest(),
            'sampling_diagnostics':{'pointer_mean_top_probability':float(sampling['pointer_top_probability_sum'])/max(1,sampling['pointer_decisions']) if pointer_temperature>0 else None,
                'action_mean_top_probability':float(sampling['action_top_probability_sum'])/max(1,sampling['action_decisions']) if action_temperature>0 else None,
                'fallback_count':sampling['fallback_count'],
                'touch_guard_proposals':sampling['touch_guard_proposals'],
                'touch_guard_safety_rejections':sampling['touch_guard_safety_rejections'],
                'mean_audio_full_pointer_js':float(sampling['audio_pointer_js_sum'])/max(1,sampling['audio_pointer_js_count']) if sampling['audio_pointer_js_count'] else None},
            'configuration_observation':{'unique_texts':len(configurations),
                'top_texts':configurations.most_common(5),
                'max_consecutive_same_text':max((sum(1 for _ in group) for _,group in groupby(point.note_text for point in points)),default=0)},
            'audio_bounds':bounds,'structured_event_generation':True,'hard_constraints':'exact_heads_and_runtime_miacode_transactions',
            'audio_feature_policy':feature_policy,'candidate_policy':candidate_policy,
            'trained_candidate_policy':trained_candidate_policy,'min_same_head_ms':min_same_head_ms,
            'action_sampling_policy':ACTION_SAMPLING_POLICY,
            'pointer_sampling_policy':GROUNDED_POINTER_SAMPLING_POLICY if grounded else 'global_temperature_truncation_v1',
            'touch_history_guard':touch_history_guard,
            'touch_history_guard_requested':touch_history_guard_requested,
            'touch_history_bound_requested':touch_history_bound,
            'touch_history_guidance_policy':TOUCH_HISTORY_GUIDANCE_POLICY if touch_history_guard else 'off',
            'touch_history_log_odds_bound':touch_history_bound if touch_history_guard else None,
            'density_observation':{'before_postprocess':raw_density,'exported':density_observation(points,clock)},
            'curve_strength':curve_strength,'pointer_history_strength':pointer_history_strength,
            'audio_pointer_training_weight':payload.get('args',{}).get('audio_pointer_weight'),
            'pointer_route':'audio_plus_full_context_residual' if music_v2 else 'full_context',
            'stop_reason':'max_events' if len(points)>=max_events else stop_reason,
            'guard':{'checks':guard.checks,'rejections':guard.rejections,'skipped_candidates':skipped_candidates,
                'rejection_reasons':dict(guard.reason_counts)},
            'playability':playability.to_dict()}
    if context_coverage:
        report['history_coverage']={'events':model.config.history_events,
            'seconds_p10_p50_p90':np.percentile(np.asarray(context_coverage)[:,0],[10,50,90]).tolist(),
            'beats_p10_p50_p90':np.percentile(np.asarray(context_coverage)[:,1],[10,50,90]).tolist(),
            'scope':'committed_decoder_context_not_a_quality_gate'}
    if learned:
        curve_times=np.arange(.25,float(audio.duration_sec),.5)
        curve=model.predict_workload(torch.from_numpy(acoustic_context.at(curve_times))[None].to(device),control)
        curve=curve[0].float().cpu().numpy()
        report['learned_workload_curve']={'times_sec':curve_times.tolist(),
            'event_rate':np.expm1(curve[:,0]).tolist(),'head_rate':np.expm1(curve[:,1]).tolist(),
            'hold_fraction':curve[:,2].tolist(),'slide_fraction':curve[:,3].tolist(),
            'touch_fraction':curve[:,4].tolist(),'source':'audio_and_requested_condition'}
        actual=build_workload_targets(points,song.tempo,song.first,curve_times,audio.duration_sec)
        predicted_heads=np.expm1(curve[:,1]);actual_heads=np.expm1(actual[:,1])
        correlation=(float(np.corrcoef(predicted_heads,actual_heads)[0,1])
            if len(curve_times)>1 and np.std(predicted_heads)>1e-8 and np.std(actual_heads)>1e-8 else None)
        report['workload_observation']={
            'actual_event_rate':np.expm1(actual[:,0]).tolist(),
            'actual_head_rate':actual_heads.tolist(),
            'mean_head_rate_error':float(np.mean(np.abs(predicted_heads-actual_heads))) if len(actual) else None,
            'head_rate_correlation':correlation,
            'scope':'generated_chart_vs_predicted_workload_not_a_musical_quality_score'}
    if grounded:
        report['audio_lens_cache']={'hits':lens_cache_hits,'misses':lens_cache_misses,'capacity':8192}
        report['coarse_audio_plan']={'times_sec':plan_grid.centers.tolist(),
            'values':planner_prediction[0].cpu().tolist(),
            'channels':['log_event_rate','log_head_rate','hold','slide','touch','rest'],
            'interpolated':True,'source':'audio_and_condition_only'}
    output.with_suffix('.composer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    args=parser().parse_args(argv);print(json.dumps(generate(args),ensure_ascii=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
