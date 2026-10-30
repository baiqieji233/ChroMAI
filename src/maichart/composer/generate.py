"""Generate a structured event chart with chart_composer_v1."""
from __future__ import annotations
import argparse,hashlib,json,math,time
from collections import Counter, OrderedDict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from maichart.event_timing import EventTempo,EventPoint,seconds_at_beat
from maichart.form.data import organize_song
from maichart.form.timing import normalize_meter_map
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
    TOUCH_HISTORY_GUIDANCE_POLICY,TOUCH_HISTORY_LOG_ODDS_BOUND,LEGACY_TOUCH_HISTORY_GUIDANCE_POLICY)
from . import ACTION_SAMPLING_POLICY
from .candidates import (COMPOSER_AUDIO_FEATURE_POLICY,candidate_beats,
                          candidate_features,composer_audio_nodes,interp_candidate_audio,
                          MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6, is_playable_candidate_transition,
                          GRID_LEVEL_COUNT, candidate_grid_levels, rhythm_family_state, family_switch_levels)
from .model import ChartComposer, WORKLOAD_PROFILES, MUSIC_PROFILES, GROUNDED_PROFILE
from .generation_settings import resolve_seed,resolve_condition
from .configuration_observation import configuration_observation
from .structure_metrics import structure_metrics
from .checkpoint import generation_motif_input_policy
from .sampling_contract import (RETRY_POLICIES,POINTER_GRID_POLICY,POINTER_GRID_POLICIES,
                                REPETITION_GUARD_POLICY,REPETITION_GUARD_POLICIES,MICROSTEP_ACTION_SAMPLING_POLICY)
from maichart.tempo_clock import TempoClock
from .acoustics import AcousticContext
from .workload import build_workload_targets, causal_ledger, event_statistics
from .guard import EventGuard,first_rejected_prefix
from . import LEGACY_ACTION_SAMPLING_POLICY,PREVIOUS_ACTION_SAMPLING_POLICY,MASKED_TOUCH_ACTION_SAMPLING_POLICY
from .events import (HEAD_PADS, event_metadata, head_timing_features,
                     history_event_features,head_pads_for_policy,head_pad_indices_for_policy)
from .curve import CURVE_MODES, CURVE_SPAN_MODES, apply_curve_to_composer

LEGACY_GROUNDED_POINTER_SAMPLING_POLICY = 'local_bin_ratios_far_tail_pruned_v3'
GRID_GROUNDED_POINTER_SAMPLING_POLICY = 'local_bin_grid_level_mass_far_tail_pruned_v4'
FAMILY_GROUNDED_POINTER_SAMPLING_POLICY = 'local_bin_grid_family_coherent_far_tail_pruned_v5'
MICROSTEP_GROUNDED_POINTER_SAMPLING_POLICY = 'local_bin_grid_family_microstep_far_tail_pruned_v6'
GROUNDED_POINTER_SAMPLING_POLICY = MICROSTEP_GROUNDED_POINTER_SAMPLING_POLICY
# grid_level_mass_v1: inside each distance bin keep the T=1 mass of every rhythm
# grid level (integer/half/quarter/eighth/third/sixth); temperature and
# truncation act only within a level. grid_family_coherent_v2 additionally
# applies temperature to switching rhythm family (binary <-> ternary) relative
# to the committed prefix, so only confident switches survive a low
# temperature. v3 also tempers entering the 1/8-beat absolute grid, without
# deleting that level or obstructing an exit. 'off' restores the old sampler.
POINTER_GRID_LEVEL_MIN_P = .05
# official_tail_v2 retains official run tails and adds a soft speed cost for
# compact per-lane chord-member repeats / already fast runs, on every slot.
REPETITION_HISTORY_EVENTS = 12
# (p95, p99) run lengths measured on 400 official Master charts, button lanes
# only (2026-10-29, .scratch/repeat_runs.py): identical lane-set runs (AA..),
# ABAB ping-pong, ABC cycles, and the most-hit lane among 6 events in 4 beats.
OFFICIAL_RUN_TAILS = {'repeat': (3, 5), 'pingpong': (6, 9), 'cycle3': (5, 6), 'saturation': (4, 5)}


def pointer_sampling_policy_name(grid_policy):
    return {'grid_family_coherent_v3':MICROSTEP_GROUNDED_POINTER_SAMPLING_POLICY,
            'grid_family_coherent_v2':FAMILY_GROUNDED_POINTER_SAMPLING_POLICY,
            'grid_level_mass_v1':GRID_GROUNDED_POINTER_SAMPLING_POLICY}.get(grid_policy,LEGACY_GROUNDED_POINTER_SAMPLING_POLICY)


def pointer_switch_levels(grid_policy, committed_beats, device):
    """Temper family changes and, for v3, entry into the microstep grid."""
    if grid_policy not in ('grid_family_coherent_v2','grid_family_coherent_v3'):
        return None
    beats=list(committed_beats)
    entering=grid_policy=='grid_family_coherent_v3' and (not beats or Fraction(beats[-1]).denominator!=8)
    return torch.from_numpy(family_switch_levels(rhythm_family_state(beats),entering_microstep=entering)).to(device)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--survival-calibration',choices=('auto','trained','integrated'),default='auto')
    p.add_argument('--density-feedback-policy',choices=('auto','off','causal_count_debt_v1'),default='auto')
    p.add_argument('--audio',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bpm',type=float);p.add_argument('--first',type=float,default=0.);p.add_argument('--level',type=float,default=13.)
    p.add_argument('--temperature',type=float,default=.8);p.add_argument('--seed',type=int,default=None,
        help='omit for a new random seed; supply an integer for reproducibility');p.add_argument('--device',default='auto')
    p.add_argument('--pointer-temperature',type=float,default=None)
    p.add_argument('--contact-policy',choices=('auto','off','diagnostic_v1','contact_groups_v1'),default='auto')
    p.add_argument('--touch-layout-policy',choices=('open','legacy_two_heads'),default='open',
        help='open separates Touch occupancy from the two-button budget; does not guarantee physical hand feasibility')
    p.add_argument('--touch-history-guard',action=argparse.BooleanOptionalAction,default=True,
        help='audio-grounded positive-temperature decoding: limit symbolic Touch probability feedback')
    p.add_argument('--touch-guidance-policy',choices=('legal_masked_v3','legacy_v2'),default='legal_masked_v3',
        help='legacy_v2 is retained for reproducible historical diagnosis only')
    p.add_argument('--touch-history-bound',type=float,default=TOUCH_HISTORY_LOG_ODDS_BOUND,
        help='Touch positive log-odds amplification cap in [0,4]; lower means stronger protection, not a Touch ratio')
    p.add_argument('--action-temperature',type=float,default=None)
    p.add_argument('--seed-randomized',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--max-events',type=int,default=0);p.add_argument('--condition-file');p.add_argument('--tempo-map')
    p.add_argument('--curve-strength',type=float,default=1.)
    p.add_argument('--meter-map',default=None,help='optional JSON meter map shared with curve prediction')
    p.add_argument('--intensity-curve-strength',type=float,default=1.,help='manual curve influence; 0 disables it without changing native workload guidance')
    p.add_argument('--intensity-curve-chord-multiplier',type=float,default=1.,help='notes per event multiplier [0.25,4]; 1 keeps learned chords')
    p.add_argument('--intensity-curve-file',default=None,help='optional JSON intensity curve file for workload/planner guidance')
    p.add_argument('--intensity-curve-mode',choices=CURVE_MODES,default='follow',
        help='how strictly the intensity curve guides workload and planner predictions')
    p.add_argument('--intensity-curve-span',choices=CURVE_SPAN_MODES,default='full',
        help='whether the intensity curve spans the full audio duration or trimmed audible bounds')
    p.add_argument('--motif-strength',type=float,default=1.,help='trained optional configuration memory residual, [0,2]; 0 disables')
    p.add_argument('--pointer-history-strength',type=float,default=1.,
        help='pointer audio anchor: 1 preserves full-context route, 0 uses audio-only route')
    p.add_argument('--action-history-strength',type=float,default=1.,
        help='independent action history/audio blend in [0,1]; 1 keeps full action context')
    p.add_argument('--quality-context-strength',type=float,default=1.,
        help='trained cooperative configuration context strength in [0,2]')
    p.add_argument('--quality-plan-strength',type=float,default=1.,
        help='trained short-phrase configuration plan strength in [0,2]')
    p.add_argument('--quality-lookahead-strength',type=float,default=0.,
        help='optional bounded physical preview in [0,1]; 0 keeps ordinary sampling')
    p.add_argument('--action-type-temperature',type=float,default=1.,
        help='positive action-temperature sampling: Tap/Hold/Slide/Touch type temperature, 1 keeps calibrated mass')
    p.add_argument('--action-type-min-p',type=float,default=0.,
        help='relative probability floor for four action types, 0 disables; preserves conditional Touch rings')
    p.add_argument('--min-same-head-ms',type=float,default=50.,
        help='minimum same-pad head spacing in milliseconds; 0 disables this generation safety policy')
    p.add_argument('--action-retry-policy',choices=('auto',*sorted(RETRY_POLICIES)),default='auto',
        help='auto inherits the checkpoint sampling contract; failed_prefix_v2 keeps valid first atoms when a later combination fails')
    p.add_argument('--intercept-tap-on-slide',action=argparse.BooleanOptionalAction,default=False,
        help='intercept TapOnSlide in EventGuard and post-process; default only reports it')
    p.add_argument('--pointer-grid-policy',choices=POINTER_GRID_POLICIES,default=POINTER_GRID_POLICY,
        help='grid_family_coherent_v3 tempers binary/ternary switches and entry into the 32nd-note grid; '
             'v2 tempers family switches only; grid_level_mass_v1 ignores the prefix; off uses the legacy sampler')
    p.add_argument('--repetition-guard',choices=REPETITION_GUARD_POLICIES,default=REPETITION_GUARD_POLICY,
        help='official_tail_v2 damps official run tails on every slot and fast compact per-lane repetition; legacy_v1 keeps first-slot damping')
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


_CANONICAL_SIMAI_DURATION_DIVISORS = frozenset({1, 2, 3, 4, 6, 8, 12, 16, 24, 32})


@lru_cache(maxsize=8)
def _canonical_duration_mask(atoms: tuple[str, ...]) -> np.ndarray:
    """Return bool mask [len(atoms)+1] keeping STOP and atoms with standard playable Simai durations.

    Allows standard subdivisions {1,2,3,4,6,8,12,16,24} and short {32} durations
    ([32:1]..[32:4], which appear 500+ times in official charts), plus long Holds
    up to 16 beats (4 measures, e.g. 6h[1:3], Ch[1:4]), while filtering out
    gimmick divisors (48, 64, 96, 384), awkward [32:13] slides, and >8-beat slides.
    """
    mask = np.ones(len(atoms) + 1, dtype=bool)
    for idx, atom in enumerate(atoms, start=1):
        if '[' not in atom:
            continue
        is_slide = any(ch in atom for ch in '-<>^pqszvwV')
        max_beats = 8.0 if is_slide else 16.0
        pos = 0
        ok = True
        while True:
            left = atom.find('[', pos)
            if left < 0:
                break
            right = atom.find(']', left + 1)
            if right < 0:
                ok = False
                break
            spec = atom[left + 1:right].strip()
            parts = spec.split(':')
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                ok = False
                break
            div = int(parts[0])
            num = int(parts[1])
            if div not in _CANONICAL_SIMAI_DURATION_DIVISORS:
                ok = False
                break
            if div == 32 and (num < 1 or num > 4):
                ok = False
                break
            if (4.0 * num / div) > max_beats:
                ok = False
                break
            pos = right + 1
        mask[idx] = ok
    return mask


@lru_cache(maxsize=8)
def _no_bracket_atom_mask(atoms: tuple[str, ...]) -> np.ndarray:
    """Return bool mask [len(atoms)+1] keeping STOP and pure Tap/Touch atoms without duration brackets."""
    mask = np.ones(len(atoms) + 1, dtype=bool)
    for idx, atom in enumerate(atoms, start=1):
        if '[' in atom:
            mask[idx] = False
    return mask


def _head_metadata(atoms,grammar,touch_layout_policy='legacy_two_heads'):
    pads=[str(lane) for lane in range(1,9)]+[f'{group}{lane}' for group in 'ABDE' for lane in range(1,9)]+['C']
    if touch_layout_policy=='legacy_two_heads': pads=pads[8:]
    pad_bits={pad:1<<i for i,pad in enumerate(pads)}
    def encode(heads):
        bits=0
        for head in heads:
            head=str(head)
            if touch_layout_policy=='legacy_two_heads' and head.isdigit(): head=f'A{head}'
            bits |= pad_bits.get(head,0)
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
    if probabilities.numel() <= 1:
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


def _grid_level_conserved_within(values, member_levels, temperature, *, top_p, min_p, max_k, level_min_p,
                                 switch_levels=None):
    """One bin: keep T=1 mass per rhythm grid level, temper/prune only inside a level.

    A global within-bin temperature squares (T=.5) the minority grid levels
    before min-p pruning, so every triplet or 16th position below roughly 22%
    of the bin's favourite was deleted outright and generated rhythm collapsed
    onto integer/half beats. Levels below ``level_min_p`` of the strongest level
    are still treated as noise and removed.

    ``switch_levels`` (bool [GRID_LEVEL_COUNT]) marks levels that would switch
    rhythm family. Their combined mass is tempered against the rest
    (M_s^(1/T) : M_k^(1/T)), proportions inside each side kept. Conserving every
    level at T=1 reproduced the model's thinly spread, low-confidence ternary
    entries: 27.8% of generated ternary onsets were isolated vs 6.2% in official
    charts. Continuation and shared integer beats keep their T=1 mass; T=1 is
    identical to plain level conservation.
    """
    base = (values - values.max()).softmax(-1)
    levels = member_levels.to(device=values.device, dtype=torch.long)
    masses = torch.zeros(GRID_LEVEL_COUNT, dtype=base.dtype, device=base.device).scatter_add_(0, levels, base)
    # Noise is defined on the model's own T=1 level masses, before tempering.
    keep = (masses > 0) & (masses >= masses.max() * float(level_min_p) - 1e-12)
    masses = torch.where(keep, masses, torch.zeros_like(masses))
    masses = masses / masses.sum().clamp_min(1e-30)
    if switch_levels is not None and float(temperature) != 1.:
        switch = switch_levels.to(device=masses.device, dtype=torch.bool)
        switch_mass, keep_mass = masses[switch].sum(), masses[~switch].sum()
        if float(switch_mass) > 0. and float(keep_mass) > 0.:
            split = (torch.stack((switch_mass, keep_mass)).log() / max(float(temperature), 1e-6)).softmax(-1)
            masses = torch.where(switch, masses * (split[0] / switch_mass), masses * (split[1] / keep_mass))
    out = torch.zeros_like(base)
    for level in torch.nonzero(masses > 0).flatten().tolist():
        member = levels.eq(level)
        sub = values[member]
        sub = ((sub - sub.max()) / max(float(temperature), 1e-6)).softmax(-1)
        sub = _truncate_probabilities(sub, top_p=top_p, min_p=min_p, max_k=max_k)
        out[member] = masses[level] * sub
    return out


def _grounded_pointer_probabilities(logits, bins, temperature, *, eligible=None,
                                    top_p=.92, min_p=.05, max_k=16,
                                    bin_temperature=1., bin_min_p=0.,
                                    levels=None, level_min_p=POINTER_GRID_LEVEL_MIN_P, switch_levels=None):
    """Keep current legal T=1 bin mass by default; temper/prune within each bin.

    Retry callers mask the original joint logits again, so rejected candidates
    lose their mass before the remaining bins are conditionally normalized.
    When ``bin_temperature != 1`` or ``bin_min_p > 0``, coarse bin masses are
    also tempered/pruned so 1-2% noise in long-rest bins (4..8, >=8 beats) cannot
    cause random mid-phrase blank stretches.

    With ``levels`` (``candidate_grid_levels``), each bin additionally keeps
    the T=1 mass of every rhythm grid level; temperature only chooses inside a
    level. ``switch_levels`` (``family_switch_levels``) also tempers the mass of
    levels that switch rhythm family. Without ``levels`` the v3 within-bin
    sampler is unchanged.
    """
    if logits.ndim != 1 or bins.shape != logits.shape:
        raise ValueError('Pointer logits and bins must be matching vectors')
    if levels is not None:
        if levels.shape != logits.shape or levels.dtype != torch.long:
            raise ValueError('Pointer grid levels must be a matching long vector')
        if bool(((levels < 0) | (levels >= GRID_LEVEL_COUNT)).any()):
            raise ValueError(f'Pointer grid levels must be in [0, {GRID_LEVEL_COUNT})')
        if not math.isfinite(float(level_min_p)) or not 0. <= float(level_min_p) <= 1.:
            raise ValueError('level_min_p must be finite and in [0,1]')
        levels = levels.to(logits.device)
        if switch_levels is not None and (switch_levels.shape != (GRID_LEVEL_COUNT,) or switch_levels.dtype != torch.bool):
            raise ValueError(f'switch_levels must be a boolean vector of length {GRID_LEVEL_COUNT}')
    elif switch_levels is not None:
        raise ValueError('switch_levels requires grid levels')
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
    bin_masses = torch.stack([joint[valid & bins.eq(group)].sum() for group in range(6)])
    if (float(bin_temperature) != 1. or float(bin_min_p) > 0.) and bool((bin_masses > 0).any()):
        active = bin_masses > 0
        if float(bin_temperature) != 1.:
            scaled = torch.full_like(bin_masses, float('-inf'))
            scaled[active] = bin_masses[active].clamp_min(1e-30).log() / max(float(bin_temperature), 1e-6)
            bin_masses = scaled.softmax(-1)
        if float(bin_min_p) > 0.:
            # Preserve T=1.0 relative mass across local/beat bins (Bin 0..2, <=2 beats)
            # Local-bin ratios stay unchanged; renormalizing after removal can
            # still increase their total mass. Only prune tiny noise tails
            # on long-jump bins (Bin 3..5, >2 beats).
            long_bin = torch.arange(6, device=bin_masses.device) >= 3
            keep_bins = (~long_bin) | (bin_masses >= (bin_masses.max() * float(bin_min_p) - 1e-12))
            if bool(keep_bins.any()):
                bin_masses = torch.where(keep_bins, bin_masses, torch.zeros_like(bin_masses))
                bin_masses = bin_masses / bin_masses.sum().clamp_min(1e-12)
    for group in range(6):
        members = valid & bins.eq(group)
        if not bool(members.any()):
            continue
        values = logits[members]
        if levels is None:
            within = ((values-values.max()) / max(float(temperature), 1e-6)).softmax(-1)
            within = _truncate_probabilities(within, top_p=top_p, min_p=min_p, max_k=max_k)
        else:
            within = _grid_level_conserved_within(values, levels[members], temperature,
                top_p=top_p, min_p=min_p, max_k=max_k, level_min_p=level_min_p, switch_levels=switch_levels)
        probabilities[members] = bin_masses[group] * within
    return probabilities


def _controlled_action_type_masses(macro_probs, groups, type_temperature, type_min_p):
    """Optional coarse type adjustment, retaining conditional Touch-ring mass."""
    if not math.isfinite(float(type_temperature)) or not 0<=type_temperature<=2:
        raise ValueError('action_type_temperature must be in [0,2]')
    if not math.isfinite(float(type_min_p)) or not 0<=type_min_p<=1:
        raise ValueError('action_type_min_p must be in [0,1]')
    masses=None
    if groups is not None and (type_temperature!=1. or type_min_p!=0.):
        masses=torch.stack([macro_probs[mask.to(macro_probs.device)].sum() for mask in groups])
        # Three button kinds plus one Touch kind. Tempering each Touch ring
        # separately would reintroduce a bias toward its most common ring.
        coarse=torch.cat((masses[:3],masses[3:].sum().reshape(1)))
        total=coarse.sum()
        if total>0:
            if type_temperature==0.:
                selected=torch.zeros_like(coarse);selected[coarse.argmax()]=1.
            else:
                selected=(coarse.log()/max(float(type_temperature),1e-6)).softmax(-1)
                selected=torch.where(selected>=selected.max()*float(type_min_p),selected,0.)
                selected=selected/selected.sum().clamp_min(1e-12)
            adjusted=selected*total
            touch=masses[3:]/coarse[3].clamp_min(1e-30)*adjusted[3]
            masses=torch.cat((adjusted[:3],touch))
    return masses


def _category_truncate_actions(masked_logits, temperature, groups, *, type_temperature=1.,type_min_p=0.):
    """Preserve legacy T=1 kind/ring mass, tempering within those categories."""
    macro_probs = masked_logits.softmax(-1)
    masses=_controlled_action_type_masses(macro_probs,groups,type_temperature,type_min_p)
    if groups is None:
        return (masked_logits / max(float(temperature), 1e-6)).softmax(-1)
    out = torch.zeros_like(macro_probs)
    out[0] = macro_probs[0]
    limits=(16,12,16)+tuple(8 for _ in groups[3:])
    for group_index,(mask, max_k) in enumerate(zip(groups, limits)):
        m = mask.to(masked_logits.device) & masked_logits.isfinite()
        if not m.any():
            continue
        cat_mass = macro_probs[m].sum() if masses is None else masses[group_index]
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


@lru_cache(maxsize=8)
def _macro_sampler_layout(group_ids,version):
    """Pack immutable catalog buckets once, avoiding per-bucket GPU syncs."""
    ids=group_ids.detach().cpu().tolist()
    if min(ids)<0:raise ValueError('Macro group IDs must be nonnegative')
    buckets=[[] for _ in range(max(ids)+1)]
    for atom_id,group in enumerate(ids):buckets[group].append(atom_id)
    width=max(map(len,buckets))
    positions=torch.zeros(len(buckets),width,device=group_ids.device,dtype=torch.long)
    padding=torch.ones_like(positions,dtype=torch.bool)
    for group,members in enumerate(buckets):
        positions[group,:len(members)]=torch.tensor(members,device=group_ids.device)
        padding[group,:len(members)]=False
    return positions,padding


def _conserved_action_probabilities(masked_logits, temperature, macro_group_ids, groups=None, *,
                                    type_temperature=1.,type_min_p=0.):
    """Keep legal starting-pad bucket mass; temper/prune only bucket details.

    Legality and Touch guidance have already changed the distribution on
    purpose. Type controls may further change its coarse kind mass. Neither
    atom temperature nor detail top-p/top-k may silently change pad buckets.
    Zero temperature chooses the best bucket, then its best legal atom.
    """
    values=masked_logits.float()
    group_ids=macro_group_ids.to(device=values.device,dtype=torch.long)
    if values.ndim!=1 or group_ids.shape!=values.shape or not group_ids.numel():
        raise ValueError('Grouped action sampling requires matching nonempty vectors')
    if not math.isfinite(float(temperature)) or temperature<0:
        raise ValueError('Action temperature must be finite and nonnegative')
    valid=torch.isfinite(values)
    out=torch.zeros_like(values)
    if not bool(valid.any()):return out
    version=None if torch.is_inference(group_ids) else group_ids._version
    positions,padding=_macro_sampler_layout(group_ids,version)
    packed=values[positions].masked_fill(padding,float('-inf'))
    group_mass_logits=packed.logsumexp(-1)
    group_probs=group_mass_logits.softmax(-1)
    if temperature==0:
        chosen_group=group_mass_logits.argmax()
        chosen_atom=values.masked_fill(~(valid&group_ids.eq(chosen_group)),float('-inf')).argmax()
        out[chosen_atom]=1.
        return out
    atom_probs=values.softmax(-1)
    masses=_controlled_action_type_masses(atom_probs,groups,type_temperature,type_min_p)
    if masses is not None:
        atom_scale=torch.ones_like(values)
        # Every catalog bucket belongs to one coarse kind/ring. Its first
        # atom supplies that intentional type-control scale, even if illegal.
        # Use the category mask rather than only legal members for that lookup.
        for group_index,mask in enumerate(groups):
            original=atom_probs[mask.to(device=values.device)].sum()
            atom_scale[mask.to(device=values.device)]=masses[group_index]/original.clamp_min(1e-30)
        group_probs=group_probs*atom_scale[positions[:,0]]
        group_probs=group_probs/group_probs.sum().clamp_min(1e-30)
    present=torch.isfinite(group_mass_logits)
    safe_packed=torch.where(present[:,None],packed,torch.zeros_like(packed))
    within=((safe_packed-safe_packed.max(-1,keepdim=True).values)/max(float(temperature),1e-6)).softmax(-1)
    within=within.masked_fill(padding|~present[:,None],0.)
    max_k=torch.full_like(group_mass_logits,16,dtype=torch.long)
    if groups is not None:
        first=positions[:,0]
        max_k=torch.where(groups[1].to(values.device)[first],12,max_k)
        if len(groups)>3:
            touch=torch.stack([mask.to(values.device)[first] for mask in groups[3:]]).any(0)
            max_k=torch.where(touch,8,max_k)
    # Batched equivalent of _truncate_probabilities. There are no Python loops
    # or scalar device synchronizations per macro bucket.
    sorted_probs=within.sort(descending=True,dim=-1).values
    ranks=torch.arange(within.size(-1),device=values.device)[None]
    included=((sorted_probs.cumsum(-1)-sorted_probs)<.85)&(ranks<max_k[:,None])
    included[:,0]=True
    floor=sorted_probs.masked_fill(~included,float('inf')).min(-1,keepdim=True).values
    threshold=torch.maximum(within.max(-1,keepdim=True).values*.08,floor)-1e-12
    filtered=within.masked_fill(within<threshold,0.)
    filtered=filtered/filtered.sum(-1,keepdim=True).clamp_min(1e-30)
    counts=(~padding&torch.isfinite(packed)).sum(-1)
    within=torch.where((counts<=1)[:,None],within,filtered)
    out.scatter_add_(0,positions.reshape(-1),(within*group_probs[:,None]).reshape(-1))
    return out/out.sum().clamp_min(1e-30)


def _candidate_playability(candidates, base_playable, committed_previous, policy):
    """Apply the existing phase policy relative to committed chart history."""
    if policy != MUSIC_CANDIDATE_POLICY_V6:
        return base_playable
    phase_playable=[ok and is_playable_candidate_transition(beat, committed_previous, policy)
        for ok,beat in zip(base_playable,candidates)]
    # Preserve the existing fallback policy; only its history anchor changes.
    return phase_playable if any(phase_playable) else base_playable


def _filter_rhythm_and_active_span_candidates(
    candidates,
    playable,
    candidate_seconds,
    committed_previous,
    committed_second,
    *,
    policy,
    committed_note_text=None,
    curve_controller=None,
    curve_times=None,
    song_workload=None,
    plan_centers=None,
    plan_values=None,
    learned_rest=False,
):
    """Suppress cross-family {48}/{96} intervals, invalid {32} entries, and mid-phrase long holes."""
    if policy != MUSIC_CANDIDATE_POLICY_V6 or committed_previous is None or not any(playable):
        return playable
    from .candidates import CANONICAL_BEAT_DENOMINATORS
    prev = committed_previous if isinstance(committed_previous, Fraction) else Fraction(committed_previous)
    prev_sec = float(committed_second) if committed_second is not None else 0.0
    # 1. Prefer canonical interval denominators so ternary<->binary switches happen at {4}/{8} anchors
    #    instead of emitting {48} (denom=12) or {96} (denom=24) in Simai.
    canonical_delta = [
        ok and (beat - prev).denominator in CANONICAL_BEAT_DENOMINATORS
        for ok, beat in zip(playable, candidates)
    ]
    current = canonical_delta if any(canonical_delta) else list(playable)
    # 2. Only initiate a {32} (1/8-beat) step when physical step >= 25ms AND the anchor note at
    #    `prev` is a clean single-head Tap/Touch (never a Chord, Hold, or Slide, which in Simai
    #    would compile as `{32}1/7,2` or `{32}5h[8:1],4x`).
    if prev.denominator != 8:
        # Compact chords such as "17" are just as multi-headed as "1/7".
        prev_is_complex = bool(committed_note_text and (
            '[' in committed_note_text or '/' in committed_note_text
            or event_metadata(committed_note_text).chord))
        safe_subdiv = [
            ok and not (
                beat.denominator == 8
                and (prev_is_complex or (float(sec) - prev_sec) < 0.025)
            )
            for ok, beat, sec in zip(current, candidates, candidate_seconds)
        ]
        if any(safe_subdiv):
            current = safe_subdiv
    # 3. Active-span long-jump guard: allow normal 1-measure (4-beat) breaths and genuine musical
    #    breakdowns/rests; only block >= 6-beat (>= 1.8s) holes when the skipped span is
    #    continuously active throughout without any rest or low-intensity dip.
    if learned_rest:
        return current
    has_near = any(ok and (beat - prev) < Fraction(6, 1) for ok, beat in zip(current, candidates))
    if has_near and committed_second is not None:
        curve_active = curve_controller is not None and bool(getattr(curve_controller, 'active', False))
        no_long_hole = []
        for ok, beat, sec in zip(current, candidates, candidate_seconds):
            if not ok:
                no_long_hole.append(False)
                continue
            delta_b = beat - prev
            delta_s = float(sec) - prev_sec
            if delta_b < Fraction(6, 1) or delta_s < 1.8:
                no_long_hole.append(True)
                continue
            span_lo = prev_sec + 0.25
            span_hi = float(sec) - 0.25
            span_is_active = False
            if span_hi > span_lo:
                if curve_active:
                    probe = np.linspace(span_lo, span_hi, max(2, min(16, int((span_hi - span_lo) / 0.25) + 1)))
                    target = curve_controller.target_array(probe)
                    rest_flags = curve_controller.rest_mask(probe)
                    if not np.any(rest_flags) and float(np.mean(target)) >= 0.32 and float(np.min(target)) >= 0.20:
                        span_is_active = True
                elif plan_centers is not None and plan_values is not None and len(plan_centers) > 0:
                    mask = (plan_centers >= span_lo) & (plan_centers <= span_hi)
                    if np.any(mask):
                        heads = np.expm1(np.clip(plan_values[mask, 1], 0., 5.))
                        rest = np.clip(plan_values[mask, 5], 0., 1.)
                        if np.any((heads >= 1.0) & (rest < 0.55)) and not np.any((heads < 0.5) & (rest >= 0.55)):
                            span_is_active = True
                elif curve_times is not None and song_workload is not None and len(curve_times) > 0:
                    mask = (curve_times >= span_lo) & (curve_times <= span_hi)
                    if np.any(mask):
                        heads = np.expm1(np.clip(song_workload[mask, 1], 0., 5.))
                        if float(np.mean(heads)) >= 1.8 and float(np.min(heads)) >= 0.8:
                            span_is_active = True
            no_long_hole.append(not span_is_active)
        if any(no_long_hole):
            current = no_long_hole
    return current if any(current) else playable


def _pointer_bin_settings(temperature, *, learned_rest=False):
    """Shared generation/recovery coarse-distance sampling controls.

    Keep bin_temperature=1.0 to preserve local-bin ratios; removing long-bin
    tail mass still renormalizes the full distribution. This is not exact
    conservation of overall density.
    """
    return {'bin_temperature': 1.0, 'bin_min_p': .04 if float(temperature) > 0 and not learned_rest else 0.}


def _microstep_head_limit(beat, previous):
    return 1 if previous is not None and beat-previous <= Fraction(1, 8) else None


def _compile_composer_chart(points, tempo, first, condition):
    # Retain every input tempo anchor even if EOS/max_events stops note output
    # earlier. Constant-tempo exports keep their existing one-beat tail.
    last_anchor=max([Fraction(0), *(point.beat for point in points),
                     *(change.beat for change in tempo)])
    export_end=Fraction(math.ceil(last_anchor)+1)
    return compile_event_points(points,tempo_changes=tempo,first=first,
        end_beat=export_end,level=str(condition.level),difficulty=condition.difficulty)


def _anti_repetition_logit_bias(recent_history, beat, bits, button_bits, *, device, policy=REPETITION_GUARD_POLICY, per_lane_runs=False, repetition_clock=None):
    """First-slot log-odds damping [len(bits)] against runaway repetition loops, or None."""
    if policy == 'off':
        return None
    if policy == 'legacy_v1':
        return _legacy_anti_repetition_logit_bias(recent_history, beat, bits, button_bits, device=device)
    if policy != 'official_tail_v2':
        raise ValueError('Unknown repetition_guard policy')
    return _official_tail_repetition_bias(recent_history, beat, bits, button_bits, device=device, per_lane_runs=per_lane_runs,repetition_clock=repetition_clock)


def _recent_button_bits(recent_history, bits, button_bits, limit):
    """Most recent first: combined button-lane bits and beat of each event."""
    evt_bits, evt_beats = [], []
    for ids, b in reversed(recent_history[-limit:]):
        active = [int(i) for i in ids if 0 < int(i) < len(bits)]
        combined = int(np.bitwise_or.reduce(bits[active])) if active else 0
        evt_bits.append(combined & int(button_bits))
        evt_beats.append(b)
    return evt_bits, evt_beats


def _official_run_strength(resulting_length, kind, moderate, strong):
    p95, p99 = OFFICIAL_RUN_TAILS[kind]
    return strong if resulting_length > p99 else (moderate if resulting_length > p95 else 0.)


def _official_tail_repetition_bias(recent_history, beat, bits, button_bits, *, device, per_lane_runs=False,repetition_clock=None):
    """Damp a repetition only when continuing it would exceed official run-length tails.

    legacy_v1 damped the third same-lane hit (AA -> A), A,B,A -> B, quick
    single-tap bounces and every chained double tap. Official Master charts
    use those constantly: after 4,4,5 the
    next event is 5 in 42.5% of cases (generated: 8.5%), ABAB ping-pong is 4.3%
    of decisions (generated: 0.5%), and same-lane runs of 3 are the p90. Here a
    run inside the official p95 is left to the model; beyond p95 it gets the old
    moderate strength and beyond p99 the old strong strength, so runaway loops
    are still broken. Run definitions match .scratch/repeat_runs.py.
    """
    if not recent_history or (len(recent_history) < 2 and not per_lane_runs):
        return None
    if (beat - recent_history[-1][1]) > Fraction(2, 1):
        return None
    e, t = _recent_button_bits(recent_history, bits, button_bits, REPETITION_HISTORY_EVENTS)
    if e[0] == 0:
        return None
    penalties = np.zeros(len(bits), dtype=np.float32)
    if per_lane_runs:
        def quick_gap(later,earlier):
            gap=(float(repetition_clock.at(later))-float(repetition_clock.at(earlier))
                 if repetition_clock is not None else float(later-earlier))
            limit=.11 if repetition_clock is not None else .25
            return gap,limit
        gap,limit=quick_gap(beat,t[0])
        if 0.<gap<=limit:
            # A normal isolated double tap is retained. Repeating a chord member
            # or extending an already fast lane run receives a soft speed cost.
            repeated=int(e[0]) if int(e[0]).bit_count()>1 else 0
            if len(e)>1:
                prior_gap,_=quick_gap(t[0],t[1])
                if 0.<prior_gap<=limit:repeated |= int(e[0]) & int(e[1])
            if repeated:
                speed=.5+1.0*(1.-gap/limit)
                penalties[(bits & np.uint64(repeated))!=0]-=speed
    # 1. Identical lane-set run (A, A, ... -> A), consecutive gaps <= 2 beats.
    if per_lane_runs:
        for lane_idx in range(8):
            lane_bit = int(button_bits) & (1 << lane_idx)
            if not lane_bit or not (e[0] & lane_bit):
                continue
            run = 1
            while run < len(e) and (e[run] & lane_bit) and (t[run-1] - t[run]) <= Fraction(2, 1):
                run += 1
            strength = _official_run_strength(run + 1, 'repeat', 1.65, 2.35)
            if strength:
                penalties[(bits & np.uint64(lane_bit)) != 0] -= strength
    else:
        run = 1
        while run < len(e) and e[run] == e[0] and (t[run-1] - t[run]) <= Fraction(2, 1):
            run += 1
        strength = _official_run_strength(run + 1, 'repeat', 1.65, 2.35)
        if strength:
            penalties[(bits & np.uint64(e[0])) != 0] -= strength
    # 2. Ping-pong (.., A, B, A -> B): alternation length in events.
    if len(e)>1 and e[1] and e[1] != e[0] and (t[0] - t[1]) <= Fraction(2, 1) and (beat - t[1]) <= Fraction(3, 1):
        length = 2
        while (length < len(e) and e[length] == e[length-2] and e[length] != e[length-1]
               and (t[length-2] - t[length]) <= Fraction(3, 1)):
            length += 1
        strength = _official_run_strength(length + 1, 'pingpong', 1.75, 2.45)
        if strength:
            b1 = np.uint64(e[1])
            penalties[(bits & b1) == b1] -= strength
    # 3. Three-cycle (A, B, C, A, B -> C).
    if len(e) >= 3 and e[2] and e[1] and len({e[0], e[1], e[2]}) == 3:
        length = 3
        while length < len(e) and e[length] == e[length-3] and len({e[length-3], e[length-2], e[length-1]}) == 3:
            length += 1
        strength = _official_run_strength(length + 1, 'cycle3', 1.50, 2.35)
        if strength:
            b2 = np.uint64(e[2])
            penalties[(bits & b2) == b2] -= strength
    # 4. One lane hit by too many of the last 6 events inside 4 beats.
    window = [eb for eb, bt in zip(e[:5], t[:5]) if (beat - bt) <= Fraction(4, 1) and eb]
    if len(window) + 1 >= 4:
        limit = OFFICIAL_RUN_TAILS['saturation'][0]
        for lane_idx in range(8):
            lane_bit = int(button_bits) & (1 << lane_idx)
            if not lane_bit:
                continue
            excess = sum(1 for eb in window if eb & lane_bit) + 1 - limit
            if excess > 0:
                penalties[(bits & np.uint64(lane_bit)) != 0] -= 0.40 * excess
    if not np.any(penalties != 0.0):
        return None
    return torch.from_numpy(penalties).to(device=device)


def _legacy_anti_repetition_logit_bias(recent_history, beat, bits, button_bits, *, device):
    """legacy_v1: bounded log-odds damping to break AA / ABA / ABAB / ABCABC self-reinforcing loops."""
    if not recent_history or len(recent_history) < 2:
        return None
    last_beat = recent_history[-1][1]
    if (beat - last_beat) > Fraction(2, 1):
        return None
    evt_bits = []
    evt_beats = []
    for ids, b in reversed(recent_history[-8:]):
        active = [int(i) for i in ids if 0 < int(i) < len(bits)]
        combined = int(np.bitwise_or.reduce(bits[active])) if active else 0
        evt_bits.append(combined & int(button_bits))
        evt_beats.append(b)
    if len(evt_bits) < 2 or evt_bits[0] == 0:
        return None

    penalties = np.zeros(len(bits), dtype=np.float32)
    b0 = np.uint64(evt_bits[0])
    b1 = np.uint64(evt_bits[1])

    # 1. Consecutive same head-set loop (A, A -> A) or chained double-taps (A, A, B -> B) or >= {16} jackhammer:
    if evt_bits[0] == evt_bits[1]:
        streak = 2
        for prev_b in evt_bits[2:]:
            if prev_b == evt_bits[0]:
                streak += 1
            else:
                break
        strength = 1.65 if streak == 2 else 2.35
        match = (bits & b0) != 0
        penalties[match] -= strength
    elif int(b0).bit_count() == 1:
        if (beat - evt_beats[0]) <= Fraction(1, 4):
            # Damp >= {16} single-tap jackhammers (A -> A within <= 1/4 beat)
            penalties[bits == b0] -= 0.90
        elif len(evt_bits) >= 3 and evt_bits[1] != 0 and evt_bits[1] == evt_bits[2] and (beat - evt_beats[2]) <= Fraction(2, 1):
            # Previous pair was already a double-tap (A, A, B -> B, e.g. 4, 4, 5 -> 5): damp chaining double-taps
            penalties[(bits & b0) == b0] -= 1.35

    # 2. Alternating 2-cycle ping-pong loop (A, B, A -> B, or preventing A, B -> A when A is already repeated):
    if len(evt_bits) >= 3 and evt_bits[0] == evt_bits[2] and evt_bits[1] != 0 and (beat - evt_beats[2]) <= Fraction(3, 1):
        # Last 3 events are A, B, A -> penalize B (which would complete ABAB) and A (which would do A, B, A, A)
        b_strength = 2.45 if (len(evt_bits) >= 4 and evt_bits[1] == evt_bits[3]) else 1.75
        match_b = (bits & b1) == b1
        penalties[match_b] -= b_strength
    elif evt_bits[0] != evt_bits[1] and evt_bits[1] != 0 and (beat - evt_beats[1]) <= Fraction(3, 2):
        # Last 2 events are A, B -> if A has already appeared >= 2 times in recent 6 events, damp bouncing right back to A (ABA)
        a_count = sum(1 for x in evt_bits[1:6] if x == evt_bits[1])
        if a_count >= 2:
            match_a = (bits & b1) == b1
            penalties[match_a] -= 1.35
        elif (beat - evt_beats[1]) <= Fraction(1, 1) and int(b0).bit_count() == 1 and int(b1).bit_count() == 1:
            # Mildly discourage immediate 1-tap ABA bounce within 1 beat so streams flow instead of ping-ponging
            match_a = (bits == b1)
            penalties[match_a] -= 0.85

    # 3. 3-cycle loop (A, B, C, A, B -> C):
    if len(evt_bits) >= 5 and evt_bits[0] == evt_bits[3] and evt_bits[1] == evt_bits[4] and evt_bits[2] != 0:
        b2 = np.uint64(evt_bits[2])
        penalties[(bits & b2) == b2] -= 1.50

    # 4. Short-window single-lane saturation (within 4 beats):
    window_bits = [eb for eb, bt in zip(evt_bits[:6], evt_beats[:6]) if (beat - bt) <= Fraction(4, 1) and eb != 0]
    if len(window_bits) >= 4:
        for lane_idx in range(8):
            lane_bit = np.uint64(int(button_bits)) & np.uint64(1 << lane_idx)
            if lane_bit == 0:
                continue
            hits = sum(1 for eb in window_bits if (np.uint64(eb) & lane_bit) != 0)
            if hits >= 3:
                penalties[(bits & lane_bit) != 0] -= 0.40 * (hits - 2)

    if not np.any(penalties != 0.0):
        return None
    return torch.from_numpy(penalties).to(device=device)


def _action_legal_mask(*, bits, sizes, button_sizes, encode, lookup, used_buttons, blocked,
                       selected, selected_head_count, slot, atoms, max_heads=2,
                       max_total_heads=None, touch_layout_policy='open', occupied=(), contact_budget=None,
                       contact_policy='off', canonical_order=False, blocked_first=(),
                       blocked_prefixes=(), canonical_dur_mask=None, no_bracket_mask=None):
    """Shared physical/grammar mask, also used to measure preview probability loss."""
    used=np.uint64(encode(used_buttons));blocked_bits=np.uint64(encode(blocked))
    button_bits=np.uint64(encode(str(lane) for lane in range(1,9)))
    budget_ok=(sizes+int(used).bit_count()<=max_heads if touch_layout_policy=='legacy_two_heads'
               else button_sizes+int(used & button_bits).bit_count()<=max_heads)
    if max_total_heads is not None:budget_ok &= sizes+selected_head_count<=max_total_heads
    mask=((bits & (used|blocked_bits))==0)&budget_ok&(sizes>0)
    if contact_budget is not None and contact_policy=='contact_groups_v1':
        mask &= contact_budget.mask(atoms,selected,occupied=occupied,policy=contact_policy)
    mask[0]=slot>0
    if canonical_order and selected:mask[1:lookup[selected[-1]]+1]=False
    if slot==0 and blocked_first:mask[list(blocked_first)]=False
    prefix=tuple(selected)
    for rejected in blocked_prefixes:
        if len(rejected)==slot+1 and tuple(rejected[:-1])==prefix:mask[lookup[rejected[-1]]]=False
    for atom in selected:mask[lookup[atom]]=False
    if canonical_dur_mask is not None and (mask[1:] & canonical_dur_mask[1:]).any():mask &= canonical_dur_mask
    if no_bracket_mask is not None and (mask[1:] & no_bracket_mask[1:]).any():mask &= no_bracket_mask
    return mask


def _sample_actions(logits, atoms, grammar, temperature, generator, occupied=(), blocked=(), max_heads=2,diagnostics=None,
                    slot_decoder=None,blocked_first=(),head_metadata=None,canonical_order=False,touch_layout_policy='open',
                    type_temperature=1.,type_min_p=0.,blocked_prefixes=(),contact_budget=None,contact_policy='off',
                    slot_mask_decoder=None,macro_group_ids=None,max_total_heads=None,microstep=False,
                    recent_history=None,beat=None,slot_atom_masks=None,min_legal_mass=0.,
                    repetition_guard=REPETITION_GUARD_POLICY, per_lane_runs=False,repetition_clock=None):
    if touch_layout_policy not in {'open','legacy_two_heads'}:
        raise ValueError('unknown touch_layout_policy')
    if not math.isfinite(float(min_legal_mass)) or not 0. <= min_legal_mass <= 1.:
        raise ValueError('min_legal_mass must be finite and in [0,1]')
    selected=[]
    selected_head_count=0
    used_buttons=set(occupied)
    if head_metadata is None: head_metadata=_head_metadata(atoms,grammar,touch_layout_policy)
    if len(head_metadata)==5:
        bits,sizes,encode,lookup,groups=head_metadata
    else:
        bits,sizes,encode,lookup=head_metadata;groups=None
    button_bits=np.uint64(encode(str(lane) for lane in range(1,9)))
    button_sizes=np.asarray([int(value & button_bits).bit_count() for value in bits])
    canonical_dur_mask=_canonical_duration_mask(tuple(atoms))
    no_bracket_mask=_no_bracket_atom_mask(tuple(atoms)) if microstep else None
    anti_rep_bias=(_anti_repetition_logit_bias(recent_history,beat,bits,button_bits,device=logits.device,
                                               policy=repetition_guard, per_lane_runs=per_lane_runs,repetition_clock=repetition_clock)
                   if temperature>0 and recent_history and beat is not None else None)
    for slot,row in enumerate(logits):
        mask_np=_action_legal_mask(bits=bits,sizes=sizes,button_sizes=button_sizes,encode=encode,
            lookup=lookup,used_buttons=used_buttons,blocked=blocked,selected=selected,
            selected_head_count=selected_head_count,slot=slot,atoms=atoms,max_heads=max_heads,
            max_total_heads=max_total_heads,touch_layout_policy=touch_layout_policy,
            occupied=occupied,contact_budget=contact_budget,contact_policy=contact_policy,canonical_order=canonical_order,
            blocked_first=blocked_first,blocked_prefixes=blocked_prefixes,
            canonical_dur_mask=canonical_dur_mask,no_bracket_mask=no_bracket_mask)
        if slot_atom_masks is not None:
            mask_np &= slot_atom_masks[slot]
        allowed=np.flatnonzero(mask_np).tolist()
        if not allowed:
            break
        if slot_mask_decoder is not None:row=slot_mask_decoder(slot,selected,mask_np)
        elif slot_decoder is not None:row=slot_decoder(slot,selected)
        values=row.float().clone()
        if anti_rep_bias is not None and (slot==0 or repetition_guard=='official_tail_v2'):
            values = values + anti_rep_bias
        if slot==0 and min_legal_mass>0.:
            # Assess support before physical/duration filtering renormalizes
            # tiny residual choices. STOP is not an onset and must not dilute
            # the first-slot denominator. Keep the original T=1 distribution;
            # this gate does not change the sampler's type or detail masses.
            onset=torch.as_tensor(sizes>0,device=values.device,dtype=torch.bool)
            onset[0]=False
            onset &= torch.isfinite(values)
            legal=onset & torch.as_tensor(mask_np,device=values.device,dtype=torch.bool)
            retained=(float(torch.exp(values[legal].logsumexp(0)-values[onset].logsumexp(0)))
                      if bool(legal.any()) else 0.)
            if diagnostics is not None:
                diagnostics['min_retained_mass']=min(diagnostics.get('min_retained_mass',1.),retained)
            if retained<float(min_legal_mass):
                if diagnostics is not None:
                    diagnostics['low_action_support_rejections']=diagnostics.get('low_action_support_rejections',0)+1
                return []
        mask=torch.full_like(values,float('-inf'))
        mask[torch.tensor(allowed,device=values.device)]=values[torch.tensor(allowed,device=values.device)]
        if temperature > 0 or macro_group_ids is not None:
            probabilities=(_conserved_action_probabilities(mask,temperature,macro_group_ids,groups,
                type_temperature=type_temperature,type_min_p=type_min_p) if macro_group_ids is not None else
                _category_truncate_actions(mask,temperature,groups,type_temperature=type_temperature,type_min_p=type_min_p))
            if diagnostics is not None and slot==0:
                diagnostics['action_top_probability_sum']+=probabilities.max().detach()
                diagnostics['action_decisions']+=1
            choice=(int(torch.multinomial(probabilities,1,generator=generator).item())
                    if temperature>0 else int(probabilities.argmax().item()))
        else:
            choice=int(mask.argmax().item())
        if choice == 0:
            break
        selected_atom=atoms[choice-1]
        if contact_budget is not None and contact_policy!='off' and diagnostics is not None:
            decision=contact_budget.decision(selected,selected_atom,occupied=occupied,blocked=blocked,policy=contact_policy)
            reasons=diagnostics.setdefault('contact_reasons',{})
            reasons[decision['reason']]=reasons.get(decision['reason'],0)+1
            diagnostics['contact_unknown']=diagnostics.get('contact_unknown',0)+int(decision['verdict']=='unknown')
        selected.append(selected_atom)
        selected_head_count += int(sizes[choice])
        used_buttons.update(str(head)
                            for head in (grammar.heads.get(selected_atom) or ()))
    return selected


def _action_sequence_log_probability(selected, logits, atoms, grammar, temperature, *,
                                     slot_mask_decoder, head_metadata, **options):
    """Score one current event without forcing it or excluding its competitor.

    Uses the same conditional masks and sampling distribution as generation,
    including the first STOP. No RNG draw and no branch-specific intent mask.
    """
    bits,sizes,encode,lookup=head_metadata[:4]
    groups=head_metadata[4] if len(head_metadata)==5 else None
    button_bits=np.uint64(encode(str(lane) for lane in range(1,9)))
    button_sizes=np.asarray([int(value & button_bits).bit_count() for value in bits])
    used=set(options.get('occupied',()));prefix=[];head_count=0;terms=[]
    bias=(_anti_repetition_logit_bias(options.get('recent_history'),options.get('beat'),bits,button_bits,device=logits.device,
                                      policy=options.get('repetition_guard',REPETITION_GUARD_POLICY),
                                      per_lane_runs=options.get('per_lane_runs',False),repetition_clock=options.get('repetition_clock'))
          if temperature>0 and options.get('recent_history') and options.get('beat') is not None else None)
    targets=[lookup[atom] for atom in selected]+([0] if len(selected)<len(logits) else [])
    for slot,target in enumerate(targets):
        legal=_action_legal_mask(bits=bits,sizes=sizes,button_sizes=button_sizes,encode=encode,lookup=lookup,
            used_buttons=used,blocked=options.get('blocked',()),selected=prefix,selected_head_count=head_count,
            slot=slot,atoms=atoms,max_heads=options.get('max_heads',2),max_total_heads=options.get('max_total_heads'),
            touch_layout_policy=options.get('touch_layout_policy','open'),occupied=options.get('occupied',()),
            contact_budget=options.get('contact_budget'),contact_policy=options.get('contact_policy','off'),
            canonical_order=options.get('canonical_order',False),blocked_first=options.get('blocked_first',()),
            blocked_prefixes=options.get('blocked_prefixes',()),canonical_dur_mask=_canonical_duration_mask(tuple(atoms)),
            no_bracket_mask=_no_bracket_atom_mask(tuple(atoms)) if options.get('microstep',False) else None)
        if not legal[target]:return float('-inf')
        values=slot_mask_decoder(slot,prefix,legal).float()
        if bias is not None and (slot==0 or options.get('repetition_guard',REPETITION_GUARD_POLICY)=='official_tail_v2'):values=values+bias
        values=values.masked_fill(~torch.from_numpy(legal).to(values.device),float('-inf'))
        group_ids=options.get('macro_group_ids')
        if group_ids is not None:
            probs=_conserved_action_probabilities(values,temperature,group_ids,groups,
                type_temperature=options.get('type_temperature',1.),type_min_p=options.get('type_min_p',0.))
        elif temperature>0:
            probs=_category_truncate_actions(values,temperature,groups,
                type_temperature=options.get('type_temperature',1.),type_min_p=options.get('type_min_p',0.))
        else:
            probs=torch.zeros_like(values);probs[values.argmax()]=1.
        if temperature<=0:
            terms.append(torch.where(probs.argmax()==target,probs.new_zeros(()),probs.new_full((),float('-inf'))))
        else:terms.append(probs[target].log())
        if target:
            atom=atoms[target-1];prefix.append(atom);head_count+=int(sizes[target])
            used.update(str(head) for head in (grammar.heads.get(atom) or ()))
    return float(torch.stack(terms).sum()) if terms else float('-inf')


@lru_cache(maxsize=8)
def _atom_intent_groups(atoms):
    """Compact type/head-count classes used only by optional preview draws."""
    signatures={};groups=[-1]
    for atom in atoms:
        meta=event_metadata(atom)
        key=(meta.type_counts,len(meta.heads))
        groups.append(signatures.setdefault(key,len(signatures)))
    return np.asarray(groups,dtype=np.int64)


def _same_intent_slot_masks(atoms,selected,action_slots):
    groups=_atom_intent_groups(tuple(atoms))
    lookup={atom:i+1 for i,atom in enumerate(atoms)}
    masks=np.zeros((action_slots,len(atoms)+1),dtype=bool)
    for slot,atom in enumerate(selected):
        masks[slot]=groups==groups[lookup[atom]]
    masks[len(selected):,0]=True
    return masks


def _active_constraints(state, beat, tempo, first, *, clock=None,touch_layout_policy='legacy_two_heads'):
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
                if touch_layout_policy=='legacy_two_heads':
                    result.add(f'A{int(lane)}'); blocked.update({str(int(lane)),f'A{int(lane)}'})
                else:
                    result.add(str(int(lane))); blocked.add(str(int(lane)))
        elif kind=='touch_hold' and marker.get('touch_pad'):
            result.add(str(marker['touch_pad'])); blocked.add(str(marker['touch_pad']))
        # Slide paths and waiting intervals are not exclusive key ownership.
        # The candidate transaction uses the existing runtime judge instead.
    return result,blocked,active_hands


def _active_heads(state, beat, tempo, first, *, clock=None):
    """Backward-compatible view of currently held concrete heads."""
    return _active_constraints(state, beat, tempo, first, clock=clock)[0]


def _touch_head_sets(atoms,grammar):
    return tuple(frozenset(str(head) for head in grammar.heads[atom]) for atom in atoms
        if grammar.heads.get(atom) and all(not str(head).isdigit() for head in grammar.heads[atom]))


def _candidate_has_capacity(occupied,blocked,touch_layout_policy,touch_heads=()):
    if touch_layout_policy=='legacy_two_heads':return len(occupied)<2
    if sum(str(head).isdigit() for head in occupied)<2:return True
    # Do not search unusable times in a button-only vocabulary. When both
    # buttons are held, an actual unoccupied Touch action may still be chosen.
    unavailable=set(occupied)|set(blocked)
    return any(not (heads & unavailable) for heads in touch_heads)


def density_observation(points, clock, *, touch_layout_policy='legacy_two_heads'):
    seconds=np.asarray([clock.at(p.beat) for p in points], dtype=np.float64)
    intervals=np.diff(seconds)
    last={};same=[]
    for point,second in zip(points,seconds):
        meta=event_metadata(point.note_text)
        pads=meta.head_pad_indices if touch_layout_policy=='legacy_two_heads' else meta.heads
        for pad in pads:
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
    return strength * full_logits + (1. - strength) * audio_logits


def _sample_survival_tail(full_logits, audio_logits, full_tail, audio_tail, strength, rng):
    """Sample the joint event/window-tail distribution before conditional timing."""
    full=torch.cat((full_logits.float(),full_tail.float().reshape(1)))
    audio=torch.cat((audio_logits.float(),audio_tail.float().reshape(1)))
    logits=blend_pointer_logits(full,audio,strength)
    if torch.equal(full_tail,audio_tail):
        # Integrated policies share one audio-derived event mass. Blending
        # conditional timing distributions must not alter the waiting odds.
        normalizer=logits[:-1].logsumexp(-1)
        event=logits[:-1]-torch.where(torch.isfinite(normalizer),normalizer,torch.zeros_like(normalizer))
        mass=(-torch.expm1(full_tail.double())).clamp_min(1e-30).log().float()
        logits=torch.cat((event+mass,full_tail.float().reshape(1)))
    probabilities=logits.softmax(-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum()<=0:
        raise ValueError('Survival distribution has no finite probability mass')
    return int(torch.multinomial(probabilities,1,generator=rng).item())==full_logits.numel()


def _pointer_js(left, right):
    left=left.float().softmax(-1); right=right.float().softmax(-1)
    mean=(left+right)*.5
    return (.5*(left*(left.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
            + .5*(right*(right.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1))


def _has_future_planned_activity(after_sec, end_sec, *, curve_times=None, song_workload=None,
                                 plan_centers=None, plan_values=None, curve_controller=None):
    """Check whether the model's own workload or coarse planner predicts significant new-note activity ahead.

    ``after_sec`` must be the last committed note-head timestamp (``last_head_sec``),
    never a sustain tail timestamp (``last_action_end_sec``).
    """
    horizon_start = float(after_sec) + 0.25
    horizon_end = float(end_sec)
    if horizon_end <= horizon_start:
        return False, None
    if plan_centers is not None and plan_values is not None and len(plan_centers) > 0:
        centers = np.asarray(plan_centers, dtype=np.float64)
        mask = (centers >= horizon_start) & (centers < horizon_end)
        if curve_controller is not None and getattr(curve_controller, 'active', False):
            mask &= ~curve_controller.rest_mask(centers)
        if np.any(mask):
            future_heads = np.expm1(np.clip(plan_values[mask, 1], 0., 5.))
            future_rest = np.clip(plan_values[mask, 5], 0., 1.)
            active_bins = (future_heads >= 0.65) & (future_rest < 0.70)
            if np.any(active_bins):
                return True, 'planner_activity_remaining'
    if curve_times is not None and song_workload is not None and len(curve_times) > 0:
        times_arr = np.asarray(curve_times, dtype=np.float64)
        mask = (times_arr >= horizon_start) & (times_arr < horizon_end)
        if curve_controller is not None and getattr(curve_controller, 'active', False):
            mask &= ~curve_controller.rest_mask(times_arr)
        if np.any(mask):
            future_heads = np.expm1(np.clip(song_workload[mask, 1], 0., 5.))
            if np.any(future_heads >= 0.75):
                return True, 'workload_activity_remaining'
    return False, None


def evaluate_eos_proposal(
    last_head_sec: float,
    end_sec: float,
    *,
    eos_probability: float,
    eos_threshold: float = 0.98,
    last_action_end_sec: float | None = None,
    has_future_candidates: bool = True,
    curve_controller=None,
    curve_times=None,
    song_workload=None,
    plan_centers=None,
    plan_values=None,
) -> dict[str, object]:
    """Evaluate whether decoding should stop at ``last_head_sec``.

    ``last_action_end_sec`` (sustain/slide tail) is intentionally decoupled from
    new-note demand: an active Hold/Slide locks only its occupied pads and must
    never suppress future note-head checks or prematurely trigger outro EOS.
    """
    del last_action_end_sec
    head_sec = float(last_head_sec)
    limit_sec = float(end_sec)
    if head_sec >= limit_sec or not bool(has_future_candidates):
        return {
            'should_stop': True,
            'stop_reason': 'audio_range_exhausted',
            'proposed': False,
            'accepted': False,
            'veto_reason': None,
        }
    if float(eos_probability) < float(eos_threshold):
        return {
            'should_stop': False,
            'stop_reason': None,
            'proposed': False,
            'accepted': False,
            'veto_reason': None,
        }
    veto_reason = None
    curve_active = curve_controller is not None and bool(
        getattr(curve_controller, 'active', getattr(curve_controller, 'enabled', False))
    )
    if curve_active and curve_controller.has_future_activity(head_sec, limit_sec):
        veto_reason = 'curve_activity_remaining'
    else:
        has_activity, activity_reason = _has_future_planned_activity(
            head_sec,
            limit_sec,
            curve_times=curve_times,
            song_workload=song_workload,
            plan_centers=plan_centers,
            plan_values=plan_values,
            curve_controller=curve_controller,
        )
        if has_activity:
            veto_reason = activity_reason
    if veto_reason is not None:
        return {
            'should_stop': False,
            'stop_reason': None,
            'proposed': True,
            'accepted': False,
            'veto_reason': veto_reason,
        }
    return {
        'should_stop': True,
        'stop_reason': 'learned_eos',
        'proposed': True,
        'accepted': True,
        'veto_reason': None,
    }


@torch.inference_mode()
def generate(args):
    from .checkpoint import read_payload
    try:new_stream=read_payload(args.checkpoint).get('schema')=='maichart-composer-event-set-checkpoint-v1'
    except (OSError,RuntimeError,ValueError,EOFError):new_stream=False
    if new_stream:
        from .stream_generate import generate as generate_stream
        for key,default in dict(pointer_temperature=args.temperature,action_temperature=args.temperature,
                intensity_curve_mode='follow',intensity_curve_strength=.65,touch_layout_policy='open',min_same_head_ms=50.).items():
            if getattr(args,key,None) is None:setattr(args,key,default)
        return generate_stream(args)
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
    action_history_strength=float(getattr(args,'action_history_strength',1.))
    quality_context_strength=float(getattr(args,'quality_context_strength',1.))
    quality_plan_strength=float(getattr(args,'quality_plan_strength',1.))
    quality_lookahead_strength=float(getattr(args,'quality_lookahead_strength',0.))
    type_temperature=float(getattr(args,'action_type_temperature',1.))
    type_min_p=float(getattr(args,'action_type_min_p',0.))
    pointer_grid_policy=getattr(args,'pointer_grid_policy',POINTER_GRID_POLICY)
    if pointer_grid_policy not in POINTER_GRID_POLICIES:raise ValueError('Unknown pointer_grid_policy')
    repetition_guard=getattr(args,'repetition_guard',REPETITION_GUARD_POLICY)
    if repetition_guard not in REPETITION_GUARD_POLICIES:raise ValueError('Unknown repetition_guard')
    for name,value,maximum in (('action_history_strength',action_history_strength,1.),
                               ('action_type_temperature',type_temperature,2.),('action_type_min_p',type_min_p,1.),
                               ('quality_context_strength',quality_context_strength,2.),
                               ('quality_plan_strength',quality_plan_strength,2.),
                               ('quality_lookahead_strength',quality_lookahead_strength,1.)):
        if not math.isfinite(value) or not 0<=value<=maximum:raise ValueError(f'{name} must be in [0,{maximum}]')
    min_same_head_ms=float(getattr(args,'min_same_head_ms',50.))
    if not math.isfinite(min_same_head_ms) or not 0 <= min_same_head_ms <= 200:
        raise ValueError('min_same_head_ms must be between 0 and 200')
    seed,seed_randomized=resolve_seed(args.seed)
    seed_randomized=seed_randomized or bool(getattr(args,'seed_randomized',False))
    condition,condition_source=resolve_condition(args.level,condition_file=args.condition_file)
    touch_layout_policy=str(getattr(args,'touch_layout_policy','open'))
    if touch_layout_policy not in {'open','legacy_two_heads'}:
        raise ValueError('unknown touch_layout_policy')
    motif_strength=float(getattr(args,'motif_strength',1.))
    if not math.isfinite(motif_strength) or not 0<=motif_strength<=2:
        raise ValueError('motif_strength must be finite and between 0 and 2')
    touch_history_bound=float(getattr(args,'touch_history_bound',TOUCH_HISTORY_LOG_ODDS_BOUND))
    if not math.isfinite(touch_history_bound) or not 0<=touch_history_bound<=4:
        raise ValueError('touch_history_bound must be finite and between 0 and 4')
    if condition.difficulty not in range(1,7):
        raise ValueError('difficulty must be an integer from 1 to 6')
    device=_device(args.device);model,payload=load(args.checkpoint,device=device);model.eval()
    trained_timing_policy=getattr(model.config,'timing_distribution_policy','hierarchical_v1')
    survival_calibration=getattr(args,'survival_calibration','auto')
    if survival_calibration not in ('auto','trained','integrated'):raise ValueError('Unknown survival_calibration')
    if trained_timing_policy=='survival_v1' and survival_calibration=='integrated':
        model.config.timing_distribution_policy='survival_integrated_v2'
    elif survival_calibration=='integrated' and trained_timing_policy!='survival_integrated_v2':
        raise ValueError('Integrated calibration requires survival-trained weights')
    trained_density_feedback=getattr(model.config,'density_feedback_policy','off')
    feedback_override=getattr(args,'density_feedback_policy','auto')
    if feedback_override not in ('auto','off','causal_count_debt_v1'):raise ValueError('Unknown density_feedback_policy')
    if feedback_override!='auto':model.config.density_feedback_policy=feedback_override
    if model.config.density_feedback_policy!='off' and model.config.timing_distribution_policy!='survival_integrated_v2':
        raise ValueError('Count feedback requires integrated survival; use a matching checkpoint or explicit integrated calibration')
    from .structure_backend import validate_backend_identity
    validate_backend_identity(model.config)
    from .structure import structure_dim_for_policy
    structure_feature_dim=(structure_dim_for_policy(model.config.structure_policy)
                           if model.config.structure_policy!='off' else 0)
    contact_policy=getattr(args,'contact_policy','auto')
    if contact_policy=='auto':contact_policy=model.config.contact_policy
    weights_sha256=_weight_fingerprint(payload['model_state_dict'])
    # The model owns copied parameters. Release file mappings now so Windows
    # training can replace composer_last.pt while a generation is running.
    payload.pop('model_state_dict',None)
    atoms=payload.get('atom_catalog') or []
    atom_to_id={atom:i+1 for i,atom in enumerate(atoms)}
    learned=model.config.architecture_profile in WORKLOAD_PROFILES
    music_v2=model.config.architecture_profile in MUSIC_PROFILES
    grounded=model.config.architecture_profile==GROUNDED_PROFILE
    effective_quality_policy=(getattr(model.config,'configuration_quality_policy','off') if grounded else 'off')
    min_action_support=.01 if effective_quality_policy in {'cooperative_v2','song_paired_v3'} else 0.
    effective_lookahead_strength=quality_lookahead_strength if effective_quality_policy!='off' else 0.
    retry_policy=getattr(args,'action_retry_policy','auto')
    if retry_policy=='auto':
        retry_policy=('failed_prefix_v2' if payload.get('action_sampling_policy') in {MICROSTEP_ACTION_SAMPLING_POLICY,ACTION_SAMPLING_POLICY,
                      MASKED_TOUCH_ACTION_SAMPLING_POLICY,PREVIOUS_ACTION_SAMPLING_POLICY}
                      else 'legacy_first_atom_v1')
    if retry_policy not in RETRY_POLICIES:raise ValueError('Unknown action_retry_policy')
    legacy_guidance=getattr(args,'touch_guidance_policy','legal_masked_v3')=='legacy_v2'
    effective_touch_policy=LEGACY_TOUCH_HISTORY_GUIDANCE_POLICY if legacy_guidance else TOUCH_HISTORY_GUIDANCE_POLICY
    musical_samplers=pointer_grid_policy in {'grid_family_coherent_v2','grid_family_coherent_v3'} and repetition_guard==REPETITION_GUARD_POLICY
    runtime_sampling_policy=((PREVIOUS_ACTION_SAMPLING_POLICY if legacy_guidance else
                              (MICROSTEP_ACTION_SAMPLING_POLICY if pointer_grid_policy=='grid_family_coherent_v3' else ACTION_SAMPLING_POLICY) if musical_samplers else MASKED_TOUCH_ACTION_SAMPLING_POLICY)
                            if retry_policy=='failed_prefix_v2' else LEGACY_ACTION_SAMPLING_POLICY)
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
    motif_input_policy=generation_motif_input_policy(payload)
    meter_file=getattr(args,'meter_map',None)
    meters=normalize_meter_map(json.loads(Path(meter_file).read_text(encoding='utf-8-sig'))) if meter_file else None
    song=organize_song(audio,tempo,args.first,meters=meters,max_seconds=999.,quality_profile='musical_evidence_v1',
        recurrence_descriptor_policy=motif_input_policy)
    clock=TempoClock(song.tempo,song.first)
    values,times=composer_audio_nodes(song,policy=COMPOSER_AUDIO_FEATURE_POLICY if learned else 'legacy')
    bounds=generation_bounds(audio,trim_exact_silence=True);end_sec=bounds['effective_end_sec']
    points=[];history=[];previous=None;max_events=args.max_events or 100000
    action_index=IncrementalState(song.tempo,song.first)
    motif_index=None
    if model.config.motif_memory_policy!='off':
        from .motif_memory import ComposerMotifIndex
        motif_index=ComposerMotifIndex(getattr(song,'relations',None),[],np.zeros((0,model.config.action_slots),dtype=np.int64))
    motif_queries=motif_matched_queries=0
    control=condition.to_vector(device=device)[None]
    acoustic_context=AcousticContext(values,times,audio.duration_sec,validated_bank=learned) if learned else None
    curve_times=None
    song_workload_curve=None
    baseline_head_rate=3.0
    if learned:
        curve_times=np.arange(.25,float(audio.duration_sec),.5)
        song_workload_curve=model.predict_workload(torch.from_numpy(acoustic_context.at(curve_times))[None].to(device),control)[0].float().cpu().numpy()
        pos_heads=np.expm1(np.clip(song_workload_curve[:,1],0.,5.))
        pos_heads=pos_heads[pos_heads>0.15]
        if len(pos_heads):
            baseline_head_rate=float(np.mean(pos_heads))
    curve_file=getattr(args,'intensity_curve_file',None) or getattr(args,'intensity_curve',None)
    curve_mode=str(getattr(args,'intensity_curve_mode','follow') or 'follow')
    curve_span=str(getattr(args,'intensity_curve_span','full') or 'full')
    curve_controller=apply_curve_to_composer(
        curve_file,
        duration_sec=float(audio.duration_sec),
        effective_start_sec=float(bounds.get('effective_start_sec',0.)),
        effective_end_sec=float(end_sec),
        mode=curve_mode,
        span_mode=curve_span,
        strength=float(getattr(args,'intensity_curve_strength',1.)),
        baseline_head_rate=baseline_head_rate,
        chord_multiplier=float(getattr(args,'intensity_curve_chord_multiplier',1.)),
    )
    plan_centers_np=None
    plan_values_np=None
    if grounded:
        from .audio_lens import LensBank
        from .planning import PlanGrid
        lens_bank=LensBank(values,times,audio.duration_sec)
        plan_grid=PlanGrid(clock,end_sec,acoustic_context)
        planner_audio=torch.from_numpy(plan_grid.audio)[None].to(device)
        planner_mask=torch.zeros(planner_audio.shape[:2],device=device,dtype=torch.bool)
        prepared_song_state=None
        if effective_quality_policy=='song_paired_v3':
            from .quality_inputs import plan_clock_features
            planner_clock=plan_clock_features(plan_grid,clock)[None].to(device)
            prepared_song_state=model.planner.encode(planner_audio,planner_mask,planner_clock)
            planner_prediction,_=model.planner.heads(prepared_song_state,control)
        else:
            planner_prediction=model.planner(planner_audio,control,planner_mask)
        native_planner_prediction=planner_prediction
        if curve_controller.enabled:
            planner_prediction=curve_controller.transform_planner(planner_prediction,plan_grid.centers)
        plan_centers_np=np.asarray(plan_grid.centers,dtype=np.float64)
        plan_values_np=planner_prediction[0].float().cpu().numpy()
        lens_cache=OrderedDict()
        lens_cache_hits=lens_cache_misses=0
    intercept_tap_on_slide=bool(getattr(args,'intercept_tap_on_slide',False))
    guard=EventGuard(song.tempo,song.first,end_sec,min_same_head_sec=min_same_head_ms/1000.,intercept_tap_on_slide=intercept_tap_on_slide,touch_layout_policy=touch_layout_policy)
    last_head_seconds=np.full(len(head_pads_for_policy(model.config.head_timing_policy)),np.nan)
    last_head_beats=np.full(len(last_head_seconds),np.nan)
    history_seconds=[];history_heads=[];head_prefix=[0.];skipped_candidates=0;survival_skip_count=0;stop_reason='audio_range_exhausted'
    committed_action_end_sec=0.
    eos_proposals=0;eos_accepted=0;eos_veto_reasons=Counter()
    context_coverage=[]
    grammar=_NoteGrammar.from_atoms(atoms)
    from .contacts import ContactBudget
    contact_budget=ContactBudget(grammar) if contact_policy!='off' else None
    head_metadata=_head_metadata(atoms,grammar,touch_layout_policy)
    touch_heads=_touch_head_sets(atoms,grammar)
    touch_mask=torch.tensor([False]+[atom[0] in 'ABCDE' for atom in atoms],dtype=torch.bool,device=device)
    rng=torch.Generator(device=device).manual_seed(seed)
    lookahead_stats={'policy':'off','requested_strength':quality_lookahead_strength,
        'effective_strength':effective_lookahead_strength,'proposals':0,'future_trials':0,
        'alternate_draws':0,'switches':0,'seconds':0.,'reasons':{}}
    if effective_lookahead_strength>0:
        from .lookahead import LOOKAHEAD_POLICY
        lookahead_stats['policy']=LOOKAHEAD_POLICY
    sampling={'rhythm_family_states':Counter(),'pointer_top_probability_sum':torch.zeros((),device=device),
              'action_top_probability_sum':torch.zeros((),device=device),'fallback_count':0,
              'pointer_decisions':0,'action_decisions':0,
              'audio_pointer_js_sum':torch.zeros((),device=device),'audio_pointer_js_count':0,
              'touch_guard_proposals':0,'touch_guard_safety_rejections':0}
    print('composer_generation_runtime '+json.dumps({'seed':seed,'seed_randomized':seed_randomized,
        'temperature':args.temperature,'pointer_temperature':pointer_temperature,
        'action_temperature':action_temperature,
        'pointer_history_strength':pointer_history_strength,
        'action_history_strength':action_history_strength,'action_type_temperature':type_temperature,'action_type_min_p':type_min_p,
        'effective_quality_policy':effective_quality_policy,
        'configuration_quality_policy':effective_quality_policy,
        'quality_context_strength':quality_context_strength,'quality_plan_strength':quality_plan_strength,
        'quality_lookahead_strength':effective_lookahead_strength,
        'first_slot_min_legal_mass':min_action_support,
        'sampling_mode':('greedy' if pointer_temperature==0 and action_temperature==0 else 'temperature'),
        'condition':condition.to_dict(),'condition_source':condition_source,
        'architecture_profile':model.config.architecture_profile,
        'head_timing_dim':model.config.head_timing_dim,'candidate_policy':candidate_policy,
        'trained_candidate_policy':trained_candidate_policy,'min_same_head_ms':min_same_head_ms,'intercept_tap_on_slide':intercept_tap_on_slide,
        'action_sampling_policy':runtime_sampling_policy,
        'rhythm_decode_rules':'rhythm_blank_guard_v2',
        'trained_action_sampling_policy':payload.get('action_sampling_policy'),
        'action_retry_policy':retry_policy,
        'history_anchor_policy':getattr(model.config,'history_anchor_policy','legacy_digit_v1'),
        'history_memory_policy':model.config.history_memory_policy,
        'history_memory_rms_bound':model.config.history_memory_rms_bound,
        'history_attention_policy':model.config.history_attention_policy,
        'action_state_policy':model.config.action_state_policy,'slide_factor_policy':model.config.slide_factor_policy,
        'motif_memory_policy':model.config.motif_memory_policy,'motif_strength':motif_strength,
        'pointer_sampling_policy':pointer_sampling_policy_name(pointer_grid_policy) if grounded else 'global_temperature_truncation_v1',
        'pointer_grid_policy':pointer_grid_policy if grounded else 'off','repetition_guard':repetition_guard,
        'touch_layout_policy':touch_layout_policy,
        'contact_policy':contact_policy,'structure_policy':model.config.structure_policy,
        'future_horizon_policy':model.config.future_horizon_policy,
        'echo_prior_policy':model.config.echo_prior_policy,
        'timing_distribution_policy':getattr(model.config,'timing_distribution_policy','hierarchical_v1'),
        'trained_timing_distribution_policy':trained_timing_policy,'survival_calibration':survival_calibration,
        'survival_rate_contract':('fixed_50ms_grid_plan_rest_prior_preserved_tail' if model.config.timing_distribution_policy=='survival_integrated_v2' else 'trained'),
        'runtime_timing_score_contract':('rate_prior_separate_distance_classifier_unclipped_fine' if model.config.timing_distribution_policy=='survival_integrated_v2' else 'trained_midpoint_bounded_shape'),
        'button_load_policy':getattr(model.config,'button_load_policy','off'),
        'fine_timing_policy':getattr(model.config,'fine_timing_policy','off'),
        'action_head_policy':model.config.action_head_policy,
        'structure_schema':model.config.structure_schema,'structure_feature_dim':structure_feature_dim,
        'structure_backend':model.config.structure_backend,
        'tempo_source':('provided_map' if args.tempo_map else 'provided_bpm' if args.bpm else 'audio_estimate'),
        'touch_layout_hand_feasibility_guaranteed':False,
        'touch_history_guard':touch_history_guard,
        'touch_history_bound_requested':touch_history_bound,
        'touch_history_guidance_policy':effective_touch_policy if touch_history_guard else 'off',
        'intensity_curve_enabled':curve_controller.enabled,
        'intensity_curve_mode':curve_controller.mode if curve_controller.enabled else 'off',
        'intensity_curve_span':curve_controller.span_mode if curve_controller.enabled else 'full',
        'intensity_curve_strength':curve_controller.strength,
        'intensity_curve_chord_multiplier':curve_controller.chord_multiplier,
        'checkpoint_epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps'),
        'weights_sha256':weights_sha256}),flush=True)
    while len(points)<max_events:
        candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,audio_end_sec=end_sec,
            candidate_count=model.config.candidate_count,clock=clock,policy=candidate_policy)
        if not candidates:break
        candidate_seconds=np.asarray([clock.at(beat) for beat in candidates])
        constraints=([_active_constraints(action_index,cand,song.tempo,song.first,clock=clock,touch_layout_policy=touch_layout_policy) for cand in candidates]
                      if getattr(action_index,'markers',None) else [(set(),set(),0) for _ in candidates])
        base_playable=[_candidate_has_capacity(held,_blocked,touch_layout_policy,touch_heads) and 0.<=float(sec)<end_sec
                       and not curve_controller.is_in_rest_region(float(sec))
                       for beat,sec,(held,_blocked,_active_hands) in zip(candidates,candidate_seconds,constraints)]
        feature_previous=history[-1][1] if history else None
        playable=_candidate_playability(candidates,base_playable,feature_previous,candidate_policy)
        if grounded and any(playable):
            playable=_filter_rhythm_and_active_span_candidates(
                candidates,
                playable,
                candidate_seconds,
                feature_previous,
                history_seconds[-1] if history_seconds else None,
                policy=candidate_policy,
                learned_rest=getattr(model.config,'timing_distribution_policy','hierarchical_v1') in ('survival_v1','survival_integrated_v2'),
                committed_note_text=points[-1].note_text if points else None,
                curve_controller=curve_controller,
                curve_times=curve_times,
                song_workload=song_workload_curve,
                plan_centers=plan_centers_np,
                plan_values=plan_values_np,
            )
        if not any(playable):
            # Advance across mid-song rest regions or temporarily unplayable windows while preserving chart history.
            previous=candidates[0]
            skipped_candidates+=1
            continue
        cand_mask=torch.tensor([[not ok for ok in playable]],dtype=torch.bool,device=device)
        # Keep the model's internal encoder/candidate_competition mask aligned with
        # training (`data.py` / `recovery.py`), where all in-bounds proposals are
        # visible to neighbor competition and playability masks the output logits.
        model_cand_mask=(torch.tensor([[(float(sec)<0.) or (float(sec)>=end_sec) for sec in candidate_seconds]],
                                      dtype=torch.bool,device=device)
                         if grounded else cand_mask)
        feats=_interp_features(values,times,song,candidates,feature_previous,clock=clock,
            feature_policy=feature_policy,rich_time=learned)[None].to(device)
        extra={}
        if getattr(model.config,'timing_distribution_policy','hierarchical_v1') in ('survival_v1','survival_integrated_v2'):
            anchor=clock.at(feature_previous if feature_previous is not None else Fraction(0))
            start=max(0.,float(clock.at(previous)-anchor)) if previous is not None else 0.
            extra['survival_start']=torch.tensor([start],device=device,dtype=torch.float32)
        window_total_heads_budget=None
        if model.config.structure_policy!='off':
            from .structure import structure_features_and_transition
            features,transition=structure_features_and_transition(song,candidates,candidate_seconds,
                float(feature_previous) if feature_previous is not None else None,
                float(clock.at(feature_previous)) if feature_previous is not None else None,
                policy=model.config.structure_policy,backend=model.config.structure_backend,
                cache_dir=Path(args.checkpoint).parent/'audio_cache')
            extra['structure_features']=torch.from_numpy(features)[None].to(device)
            extra['structure_transition']=torch.from_numpy(transition)[None].to(device)
        if learned:
            from .workload import prefix_ledger
            extra.update({'workload_audio':torch.from_numpy(acoustic_context.at(candidate_seconds))[None].to(device),
                'ledger':torch.from_numpy(prefix_ledger(history_seconds,head_prefix,candidate_seconds))[None].to(device),
                'curve_strength':curve_strength})
            if music_v2:
                committed_second=history_seconds[-1] if history_seconds else 0.
                extra.update(interval_audio=torch.from_numpy(acoustic_context.interval(committed_second,candidate_seconds))[None].to(device),
                             trailing_audio=torch.from_numpy(acoustic_context.trailing(candidate_seconds))[None].to(device))
            if curve_controller.enabled:
                base_wl=model.predict_workload(extra['workload_audio'],control)
                base_tr=(model.predict_trailing(extra['trailing_audio'],control)
                         if music_v2 and 'trailing_audio' in extra and hasattr(model,'predict_trailing') else None)
                window_overrides=curve_controller.build_window_overrides(
                    candidate_seconds=candidate_seconds,model_workload=base_wl,model_trailing=base_tr)
                if window_overrides['workload_override'] is not None:
                    extra['workload_override']=window_overrides['workload_override']
                if window_overrides['trailing_override'] is not None:
                    extra['trailing_override']=window_overrides['trailing_override']
                window_total_heads_budget=window_overrides['total_heads_budget']
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
                    planner_mix=torch.from_numpy(mix)[None].to(device),
                    planner_override=(planner_prediction if effective_quality_policy!='song_paired_v3' or curve_controller.enabled else None),
                    lens_embeddings=embedded)
                if effective_quality_policy=='song_paired_v3':
                    extra['prepared_song_state']=prepared_song_state
        if getattr(model.config,'timing_distribution_policy','hierarchical_v1')=='survival_integrated_v2':
            from .density_grid import survival_grid_inputs
            absolute_start=max(0.,float(clock.at(previous))) if previous is not None else max(0.,float(clock.at(Fraction(0))))
            grid=survival_grid_inputs(acoustic_context,absolute_start,candidate_seconds,end_sec,plan_grid=plan_grid,
                prefix_seconds=history_seconds if model.config.density_feedback_policy=='causal_count_debt_v1' else None)
            grid_keys=('survival_audio','survival_edges','survival_candidate_seconds',
                       'survival_start_seconds','survival_plan_indices','survival_plan_mix')
            extra.update({key:torch.as_tensor(grid[key])[None].to(device) for key in grid_keys})
            if 'survival_prefix_counts' in grid:
                extra['survival_prefix_counts']=torch.from_numpy(grid['survival_prefix_counts'])[None].to(device)
            if curve_controller.active and not curve_controller._neutral():
                from .planning import read_plan
                # Fuse native predictions first, then apply the actual editor
                # change once; the displayed/action planner was edited above.
                grid_plan=read_plan(native_planner_prediction,extra['survival_plan_indices'],extra['survival_plan_mix'])
                grid_rates=model.predict_survival_grid(extra['survival_audio'],control,plan_at_grid=grid_plan)
                grid_workload=model.predict_workload(extra['survival_audio'],control)
                grid_heads=torch.expm1(grid_workload[...,1].float().clamp(0.,8.))
                grid_seconds=(grid['survival_edges'][:-1]+grid['survival_edges'][1:])*.5
                extra['survival_grid_rates_override']=curve_controller.transform_survival_rates(grid_rates,grid_heads,grid_seconds)
        history_tail=history[-model.config.history_events:]
        if effective_quality_policy!='off':
            from .quality_inputs import quality_context_inputs
            quality_start=max(0,len(points)-model.config.history_events) if effective_quality_policy=='song_paired_v3' else 0
            quality=quality_context_inputs(points=points[quality_start:],
                include_future=getattr(model,'compact_actions',False),
                atom_ids=np.asarray([ids for ids,_ in history[quality_start:]],dtype=np.int64).reshape(-1,model.config.action_slots),
                candidate_beats=candidates,candidate_seconds=candidate_seconds,state_index=action_index,
                audio_features=values,audio_times=times,acoustic_context=acoustic_context,
                clock=clock,audio_duration=audio.duration_sec,policy=effective_quality_policy,
                plan_grid=plan_grid if grounded else None)
            extra.update(quality_inputs={k:v[None].to(device) for k,v in quality.items()},
                quality_context_strength=quality_context_strength,quality_plan_strength=quality_plan_strength)
        if getattr(model,'compact_actions',False) and 'quality_inputs' not in extra:
            from .quality_inputs import fixed_future_audio_inputs
            extra['quality_inputs']={k:v[None].to(device) for k,v in fixed_future_audio_inputs(acoustic_context,clock,candidates,audio.duration_sec).items()}
        if grounded:
            from .history_clock import history_beat_ages
            extra['history_beat_ages']=torch.from_numpy(history_beat_ages(
                [beat for _,beat in history_tail]))[None].to(device)
            extra['history_second_ages']=torch.tensor(
                [[float(clock.at(history[-1][1]))-float(clock.at(beat)) for _,beat in history_tail]],
                dtype=torch.float32,device=device)
        if history_tail:
            hist_atoms=torch.tensor([row for row,_ in history_tail],dtype=torch.long,device=device)[None]
            preceding_beat=history[-model.config.history_events-1][1] if len(history)>model.config.history_events else Fraction(0)
            hist_features=torch.tensor(_history_features(history_tail,atoms,song.tempo,song.first,model.config.action_slots,
                clock=clock,preceding_beat=preceding_beat,parsed=parsed_features),dtype=torch.float32,device=device)[None]
        else:
            hist_atoms=torch.zeros((1,0,model.config.action_slots),dtype=torch.long,device=device)
            hist_features=torch.zeros((1,0,8),dtype=torch.float32,device=device)
        committed_beat=history[-1][1] if history else Fraction(0)
        from .context_features import composer_action_state
        action=composer_action_state(action_index,len(history),candidates,model.config.action_state_policy).to(device)
        output=model(feats,hist_atoms,control,action_state=action[None],history_features=hist_features,
                     candidate_mask=model_cand_mask,
                     history_padding_mask=torch.zeros((1,hist_atoms.size(1)),dtype=torch.bool,device=device),
                     decode_actions=False,audio_auxiliary=music_v2,**extra)
        full_pointer=output['candidate_logits'][0].float()
        if 'density_feedback_log_penalty' in output:
            sampling['density_feedback_penalty_sum']=sampling.get('density_feedback_penalty_sum',0.)+output['density_feedback_log_penalty'].mean().detach()
            sampling['density_feedback_windows']=sampling.get('density_feedback_windows',0)+1
        audio_pointer=output['audio_candidate_logits'][0].float() if 'audio_candidate_logits' in output else full_pointer
        if music_v2 and 'audio_candidate_logits' in output:
            sampling['audio_pointer_js_sum'] += _pointer_js(full_pointer[None],audio_pointer[None]).sum().detach()
            sampling['audio_pointer_js_count'] += 1
        if getattr(model.config,'timing_distribution_policy','hierarchical_v1') in ('survival_v1','survival_integrated_v2'):
            if _sample_survival_tail(full_pointer,audio_pointer,output['no_event_log_prob'][0],
                    output.get('audio_no_event_log_prob',output['no_event_log_prob'])[0],
                    pointer_history_strength,rng):
                if previous is not None and candidates[-1]<=previous:
                    raise ValueError('Survival candidate window did not advance')
                previous=candidates[-1]
                survival_skip_count+=1
                continue
        logits=blend_pointer_logits(full_pointer,audio_pointer,pointer_history_strength)
        if not grounded:
            logits=logits/max(pointer_temperature,1e-6)
        if grounded and (torch.isnan(logits).any() or torch.isposinf(logits).any()):
            raise ValueError('Pointer logits contain NaN or positive infinity')
        if not all(playable):
            logits=logits.masked_fill(cand_mask[0],float('-inf'))
        pointer_levels=(torch.from_numpy(candidate_grid_levels(candidates,logits.numel())).to(logits.device)
                        if grounded and pointer_grid_policy!='off' else None)
        pointer_switch=(pointer_switch_levels(pointer_grid_policy,[beat for _,beat in history[-32:]],logits.device)
                        if pointer_levels is not None else None)
        if pointer_switch is not None:
            sampling['rhythm_family_states'][rhythm_family_state([beat for _,beat in history[-32:]])]+=1
        selected=[]
        for attempt in range(min(len(candidates),8)):
            if not torch.isfinite(logits).any(): break
            pointer_probs=(_grounded_pointer_probabilities(logits,output['candidate_bins'][0],pointer_temperature,
                                                           levels=pointer_levels,switch_levels=pointer_switch,
                                                           **_pointer_bin_settings(pointer_temperature,learned_rest='next_event_log_probs' in output))
                           if grounded else _truncate_probabilities(logits.softmax(-1),top_p=.92,min_p=.05,max_k=16)
                           if pointer_temperature>0 else None)
            if pointer_temperature>0:
                sampling['pointer_top_probability_sum']+=pointer_probs.max().detach()
                sampling['pointer_decisions']+=1
            choice=int(torch.multinomial(pointer_probs,1,generator=rng).item()) if pointer_temperature>0 else int(logits.argmax())
            beat=candidates[choice]
            total_heads_limit=None
            if window_total_heads_budget is not None:
                expectation=float(window_total_heads_budget[choice])
                lower=max(1,math.floor(expectation))
                # Only explicit chord reduction consumes this draw. The default
                # curve keeps exactly the old action sampling/random sequence.
                fraction=expectation-lower
                total_heads_limit=lower+(int(float(torch.rand((),device=device,generator=rng))<fraction) if fraction>0. else 0)
            micro_limit=_microstep_head_limit(beat,feature_previous)
            if micro_limit is not None:
                total_heads_limit=micro_limit
            cand_sec=float(clock.at(beat))
            memory={}
            if motif_index is not None:
                from .context_features import motif_tensors
                memory=motif_tensors(motif_index,beat,len(history),model.config.history_events,model.config.motif_max_events)
                motif_queries+=1;motif_matched_queries+=int(memory['motif_mask'].any())
                memory={k:v[None].to(device) for k,v in memory.items()}
            if model.config.action_state_policy=='selected_prefix_v3':
                from .context_features import selected_action_state
                memory.update(selected_action_state=selected_action_state(action_index,len(history),beat)[None].to(device),
                    state_base=output['selected_state_base'],state_keep=output['selected_state_keep'])
            decision=model.decode_candidate(output['candidate_context'],torch.tensor([choice],device=device),
                 project_actions=not learned,audio_context=output.get('audio_action_context'),
                 **({**memory,'motif_strength':motif_strength,
                     'previous_heads':output.get('previous_heads'),
                     **({'echo_heads':output['echo_lane_bias']} if 'echo_lane_bias' in output else {}),
                     **({'button_load_features':output['button_load_features']} if 'button_load_features' in output else {})} if grounded else {}))
            occupied,blocked,_=constraints[choice]
            cooldown_blocked=set(blocked)|{pad for pad,prior in guard.last_head_seconds.items()
                if cand_sec-prior<guard.min_same_head_sec-1e-9}
            audio_act=decision['audio_action_context'][0] if 'audio_action_context' in decision else None
            decoder_diagnostics={}
            decoder=(lambda slot,chosen,legal_mask: action_logits_with_touch_guard(model,decision['action_context'][0],slot,
                [atom_to_id[atom] for atom in chosen],audio_context=audio_act,
                 history_strength=action_history_strength,enabled=touch_history_guard,touch_mask=touch_mask,
                diagnostics=decoder_diagnostics,log_odds_bound=touch_history_bound,
                legal_mask=legal_mask,legacy=getattr(args,'touch_guidance_policy','legal_masked_v3')=='legacy_v2',
                previous_heads=(decision['previous_heads'][0] if 'previous_heads' in decision else None),
                echo_heads=(decision['echo_heads'][0] if 'echo_heads' in decision else None),
                button_load_features=(decision['button_load_features'][0] if 'button_load_features' in decision else None))) if learned else None
            blocked_first=set()
            blocked_prefixes=set()
            action_options=dict(occupied=occupied,blocked=cooldown_blocked,max_heads=2,
                min_legal_mass=min_action_support,
                max_total_heads=total_heads_limit,blocked_first=blocked_first,head_metadata=head_metadata,
                canonical_order=music_v2 and getattr(model.config,'action_head_policy','') != 'compact_grouped_v3',touch_layout_policy=touch_layout_policy,
                contact_budget=contact_budget,contact_policy=contact_policy,
                type_temperature=type_temperature,type_min_p=type_min_p,blocked_prefixes=blocked_prefixes,
                macro_group_ids=getattr(getattr(model,'conserved_action_head',None),'atom_group_id',None),
                microstep=(micro_limit is not None),recent_history=history[-REPETITION_HISTORY_EVENTS:],beat=beat,
                repetition_guard=repetition_guard,
                per_lane_runs=getattr(model.config,'action_head_policy','') == 'compact_grouped_v3',repetition_clock=clock)
            for retry in range(6):
                decoder_diagnostics.clear()
                slot_rows=(torch.zeros((model.config.action_slots,len(atoms)+1),device=device)
                           if learned else decision['action_logits'][0])
                proposed=_sample_actions(slot_rows,atoms,grammar,action_temperature,rng,
                    diagnostics=sampling,slot_mask_decoder=decoder,**action_options)
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
                if not proposed and min_action_support>0:
                    # This candidate has no supported first action. Repeating
                    # an identical draw six times cannot restore its support.
                    break
                sampling['fallback_count']+=1
                if proposed:
                    if retry_policy=='failed_prefix_v2':
                        blocked_prefixes.add(first_rejected_prefix(guard,beat,proposed,protect_reported_hard=guided))
                    else:blocked_first.add(atom_to_id[proposed[0]])
                    if guard.last_reason=='repeated_head_too_soon':
                        for head in event_metadata('/'.join(proposed)).heads:
                            pad=f'A{head}' if touch_layout_policy=='legacy_two_heads' and str(head).isdigit() else str(head)
                            prior=guard.last_head_seconds.get(pad)
                            if retry_policy=='legacy_first_atom_v1' or (prior is not None and cand_sec-prior<guard.min_same_head_sec-1e-9):
                                cooldown_blocked.add(pad)
            if selected: break
            logits[choice]=float('-inf')
        if not selected:
            # Advance the search cursor; committed chart/history remain intact.
            # A temporarily blocked action must never end the whole song.
            previous=candidates[0]
            skipped_candidates+=1
            continue
        if effective_lookahead_strength>0 and 'quality_gap_logits' in output:
            from .lookahead import short_gap_candidate,bounded_physical_lookahead,FutureAssessment
            preview_started=time.perf_counter()
            future_allowed=_candidate_playability(candidates,base_playable,beat,candidate_policy)
            future_choice=short_gap_candidate(candidates,choice,output['quality_gap_logits'][0,choice],
                allowed=future_allowed)
            if future_choice is not None:
                baseline_event=EventPoint(beat,'/'.join(selected))
                variants={baseline_event.note_text:list(selected)}
                reference_future={}
                # Private draws reuse the current action context; they do not
                # change main sampling diagnostics or its random stream.
                def preview_decoder(current_decision):
                    audio_value=current_decision.get('audio_action_context')
                    return lambda slot,chosen,legal: action_logits_with_touch_guard(
                        model,current_decision['action_context'][0],slot,
                        [atom_to_id[atom] for atom in chosen],
                        audio_context=audio_value[0] if audio_value is not None else None,
                        history_strength=action_history_strength,enabled=touch_history_guard,
                        touch_mask=touch_mask,log_odds_bound=touch_history_bound,legal_mask=legal,
                        legacy=legacy_guidance,previous_heads=current_decision['previous_heads'][0],
                        echo_heads=current_decision['echo_heads'][0] if 'echo_heads' in current_decision else None,
                        button_load_features=current_decision['button_load_features'][0] if 'button_load_features' in current_decision else None)

                def alternative_factory(private_rng):
                    options=dict(action_options)
                    options['blocked_prefixes']=set(blocked_prefixes)|{tuple(selected)}
                    options['slot_atom_masks']=_same_intent_slot_masks(atoms,selected,model.config.action_slots)
                    alternative=_sample_actions(slot_rows,atoms,grammar,action_temperature,private_rng,
                        slot_mask_decoder=preview_decoder(decision),**options)
                    if not alternative:return None
                    event=EventPoint(beat,'/'.join(alternative))
                    variants[event.note_text]=alternative
                    return event

                def current_probability_factory(current):
                    return _action_sequence_log_probability(variants[current.note_text],slot_rows,atoms,grammar,
                        action_temperature,slot_mask_decoder=preview_decoder(decision),**action_options)

                def future_factory(branch_guard,current,future_beat,private_rng):
                    # Rebuild only the guard's bounded live prefix.  Native
                    # markers supply true waiting/occupancy/release semantics.
                    branch_state=IncrementalState(song.tempo,song.first)
                    for point,_,_,_ in branch_guard.prefix:branch_state.append(point)
                    future_second=float(clock.at(future_beat))
                    held,blocked_future,_=_active_constraints(branch_state,future_beat,song.tempo,song.first,
                        clock=clock,touch_layout_policy=touch_layout_policy)
                    blocked_future=set(blocked_future)|{pad for pad,prior in branch_guard.last_head_seconds.items()
                        if future_second-prior<branch_guard.min_same_head_sec-1e-9}
                    previous_heads=torch.zeros((1,8),device=device)
                    for head in event_metadata(current.note_text).heads:
                        if head in '12345678' and len(head)==1:previous_heads[0,int(head)-1]=1
                    future_memory={'previous_heads':previous_heads}
                    if 'echo_lane_bias' in output:future_memory['echo_heads']=output['echo_lane_bias']
                    if 'button_load_features' in output:future_memory['button_load_features']=output['button_load_features']
                    if model.config.action_state_policy=='selected_prefix_v3':
                        from .context_features import selected_action_state
                        future_memory.update(selected_action_state=selected_action_state(
                            branch_state,len(branch_state.points),future_beat)[None].to(device),
                            state_base=output['selected_state_base'],state_keep=output['selected_state_keep'])
                    # The cached context still encodes the earlier prefix. It
                    # proposes an action; the native transaction judges it.
                    # No backbone rerun or claimed next-step posterior is used.
                    future_decision=model.decode_candidate(output['candidate_context'],
                        torch.tensor([future_choice],device=device),project_actions=False,
                        audio_context=output.get('audio_action_context'),**future_memory)
                    future_limit=_microstep_head_limit(future_beat,current.beat)
                    total_limit=future_limit
                    if total_limit is None and window_total_heads_budget is not None:
                        expectation=float(window_total_heads_budget[future_choice]);lower=max(1,math.floor(expectation))
                        fraction=expectation-lower
                        total_limit=lower+(int(float(torch.rand((),device=device,generator=private_rng))<fraction)
                            if fraction>0. else 0)
                    # Evaluate both branches against ONE distribution before
                    # either proposed current event adds occupancy/cooldown.
                    # A legal sample after masking alone hides probability lost
                    # to that mask (e.g. .85 preferred mass replaced by .02).
                    bits,sizes,encode,lookup=head_metadata[:4]
                    button_bits=np.uint64(encode(str(lane) for lane in range(1,9)))
                    mask_options=dict(bits=bits,sizes=sizes,
                        button_sizes=np.asarray([int(value & button_bits).bit_count() for value in bits]),
                        encode=encode,lookup=lookup,selected=(),selected_head_count=0,slot=0,atoms=atoms,
                        max_total_heads=total_limit,touch_layout_policy=touch_layout_policy,
                        contact_budget=contact_budget,contact_policy=contact_policy,
                        canonical_dur_mask=_canonical_duration_mask(tuple(atoms)),
                        no_bracket_mask=_no_bracket_atom_mask(tuple(atoms)) if future_limit is not None else None)
                    if not reference_future:
                        prior_held,prior_blocked,_=_active_constraints(action_index,future_beat,song.tempo,song.first,
                            clock=clock,touch_layout_policy=touch_layout_policy)
                        prior_blocked=set(prior_blocked)|{pad for pad,prior in guard.last_head_seconds.items()
                            if future_second-prior<guard.min_same_head_sec-1e-9}
                        prior_mask=_action_legal_mask(used_buttons=prior_held,occupied=prior_held,
                            blocked=prior_blocked,**mask_options)
                        prior_memory={'previous_heads':output.get('previous_heads')}
                        if 'echo_lane_bias' in output:prior_memory['echo_heads']=output['echo_lane_bias']
                        if 'button_load_features' in output:prior_memory['button_load_features']=output['button_load_features']
                        if model.config.action_state_policy=='selected_prefix_v3':
                            prior_memory.update(selected_action_state=selected_action_state(
                                action_index,len(history),future_beat)[None].to(device),
                                state_base=output['selected_state_base'],state_keep=output['selected_state_keep'])
                        prior_decision=model.decode_candidate(output['candidate_context'],
                            torch.tensor([future_choice],device=device),project_actions=False,
                            audio_context=output.get('audio_action_context'),**prior_memory)
                        prior_logits=preview_decoder(prior_decision)(0,[],prior_mask).float()
                        bias=_anti_repetition_logit_bias(history[-REPETITION_HISTORY_EVENTS:],future_beat,bits,button_bits,
                                                         device=device,policy=repetition_guard,
                                                         per_lane_runs=action_options['per_lane_runs'],repetition_clock=clock)
                        if action_temperature>0 and bias is not None:prior_logits=prior_logits+bias
                        prior_logits=prior_logits.masked_fill(~torch.from_numpy(prior_mask).to(device),float('-inf'))
                        group_ids=action_options['macro_group_ids']
                        groups=head_metadata[4] if len(head_metadata)==5 else None
                        if not prior_mask.any():
                            probabilities=torch.zeros_like(prior_logits)
                        elif group_ids is not None:
                            probabilities=_conserved_action_probabilities(prior_logits,action_temperature,group_ids,groups,
                                type_temperature=type_temperature,type_min_p=type_min_p)
                        elif action_temperature>0:
                            probabilities=_category_truncate_actions(prior_logits,action_temperature,groups,
                                type_temperature=type_temperature,type_min_p=type_min_p)
                        else:
                            probabilities=torch.zeros_like(prior_logits);probabilities[prior_logits.argmax()]=1.
                        if action_temperature<=0 and prior_mask.any():
                            winner=probabilities.argmax();probabilities=torch.zeros_like(probabilities);probabilities[winner]=1.
                        reference_future['probabilities']=probabilities
                    branch_mask=_action_legal_mask(used_buttons=held,occupied=held,blocked=blocked_future,**mask_options)
                    retained=float(reference_future['probabilities'].masked_fill(
                        ~torch.from_numpy(branch_mask).to(device),0.).sum().clamp(0.,1.))
                    ids=tuple([atom_to_id[atom] for atom in variants[current.note_text]]+
                        [0]*(model.config.action_slots-len(variants[current.note_text])))
                    options=dict(action_options,occupied=held,blocked=blocked_future,
                        blocked_first=(),blocked_prefixes=(),max_total_heads=total_limit,
                        microstep=future_limit is not None,recent_history=history[-(REPETITION_HISTORY_EVENTS-1):]+[(ids,current.beat)],beat=future_beat)
                    proposed_future=_sample_actions(slot_rows,atoms,grammar,action_temperature,private_rng,
                        slot_mask_decoder=preview_decoder(future_decision),**options)
                    future_event=EventPoint(future_beat,'/'.join(proposed_future)) if proposed_future else None
                    return FutureAssessment(future_event,retained)

                preview=bounded_physical_lookahead(baseline_event,future_beat=candidates[future_choice],
                    guard=guard,generator=rng,alternative_factory=alternative_factory,future_factory=future_factory,
                    current_probability_factory=current_probability_factory,
                    strength=effective_lookahead_strength,protect_reported_hard=touch_history_guard)
                lookahead_stats['proposals']+=1
                lookahead_stats['future_trials']+=preview.future_trials
                lookahead_stats['alternate_draws']+=preview.alternate_draws
                lookahead_stats['switches']+=int(preview.switched)
                if preview.baseline_retained_probability is not None:
                    lookahead_stats['retained_mass_checks']=lookahead_stats.get('retained_mass_checks',0)+1
                    lookahead_stats['baseline_retained_mass_sum']=lookahead_stats.get('baseline_retained_mass_sum',0.)+preview.baseline_retained_probability
                if preview.alternative_retained_probability is not None:
                    lookahead_stats['alternative_mass_checks']=lookahead_stats.get('alternative_mass_checks',0)+1
                    lookahead_stats['alternative_retained_mass_sum']=lookahead_stats.get('alternative_retained_mass_sum',0.)+preview.alternative_retained_probability
                if preview.current_log_probability_ratio is not None:
                    lookahead_stats['current_support_checks']=lookahead_stats.get('current_support_checks',0)+1
                if preview.reason=='insufficient_current_support':
                    lookahead_stats['unsupported_alternatives']=lookahead_stats.get('unsupported_alternatives',0)+1
                reasons=lookahead_stats['reasons'];reasons[preview.reason]=reasons.get(preview.reason,0)+1
                if preview.switched:selected=variants[preview.event.note_text]
            lookahead_stats['seconds']+=time.perf_counter()-preview_started
        text='/'.join(selected)
        event=EventPoint(beat,text)
        points.append(event); action_index.append(event);guard.commit(event)
        if guard.prefix:
            committed_action_end_sec=max(committed_action_end_sec,float(guard.prefix[-1][2]))
        for pad in head_pad_indices_for_policy(event_metadata(text),model.config.head_timing_policy):
            last_head_seconds[pad]=float(clock.at(beat));last_head_beats[pad]=float(beat)
        history_seconds.append(float(clock.at(beat)));history_heads.append(event_statistics(text)[0])
        head_prefix.append(head_prefix[-1]+history_heads[-1])
        if history_tail:
            context_coverage.append((float(clock.at(history_tail[-1][1])-clock.at(history_tail[0][1])),
                                     float(history_tail[-1][1]-history_tail[0][1])))
        else:
            context_coverage.append((0.,0.))
        atom_ids=tuple([atom_to_id[atom] for atom in selected]+[0]*(model.config.action_slots-len(selected)))
        if motif_index is not None:motif_index.append(beat,atom_ids)
        history.append((atom_ids,beat));previous=beat
        eos_probability=float(decision['eos_logits'][0].sigmoid())
        beat_sec=float(clock.at(beat))
        eos_decision=evaluate_eos_proposal(
            beat_sec,
            end_sec,
            eos_probability=eos_probability,
            eos_threshold=(.98 if learned else .9),
            last_action_end_sec=committed_action_end_sec,
            curve_controller=curve_controller,
            curve_times=curve_times,
            song_workload=song_workload_curve,
            plan_centers=plan_centers_np,
            plan_values=plan_values_np,
        )
        if eos_decision['proposed']:
            eos_proposals+=1
            if eos_decision['veto_reason'] is not None:
                eos_veto_reasons[eos_decision['veto_reason']]+=1
            elif eos_decision['accepted']:
                eos_accepted+=1
        if eos_decision['should_stop']:
            stop_reason=str(eos_decision['stop_reason'])
            break
    raw_density=density_observation(points,clock,touch_layout_policy=touch_layout_policy)
    raw_configuration_observation=configuration_observation(points)
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
    last_head_sec=float(clock.at(points[-1].beat)) if points else None
    last_action_end_sec=None
    if points:
        last_action_end_sec=max(float(guard._lifetime(pt)[1]) for pt in points)
    text,_=_compile_composer_chart(points,song.tempo,args.first,condition)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    report={'schema':'maichart-chart-composer-generation-v2','family':'chart_composer_v1','event_count':len(points),
            'architecture_profile':model.config.architecture_profile,
            'checkpoint':str(Path(args.checkpoint).resolve()),'device':str(device),'seed':seed,
            'seed_randomized':seed_randomized,'temperature':args.temperature,
            'pointer_temperature':pointer_temperature,'action_temperature':action_temperature,
            'configuration_quality_policy':effective_quality_policy,'effective_quality_policy':effective_quality_policy,
            'quality_context_strength':quality_context_strength,'quality_plan_strength':quality_plan_strength,
            'quality_lookahead_strength':effective_lookahead_strength,'quality_lookahead':lookahead_stats,
            'first_slot_min_legal_mass':min_action_support,
            'condition':condition.to_dict(),'condition_source':condition_source,
            'checkpoint_epoch':payload.get('epoch'),'optimizer_steps':payload.get('optimizer_steps'),
            'weights_sha256':weights_sha256,'chart_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'events_sha256':hashlib.sha256(json.dumps([(str(p.beat),p.note_text) for p in points],
                ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest(),
            'sampling_diagnostics':{'pointer_mean_top_probability':float(sampling['pointer_top_probability_sum'])/max(1,sampling['pointer_decisions']) if pointer_temperature>0 else None,
                'action_mean_top_probability':float(sampling['action_top_probability_sum'])/max(1,sampling['action_decisions']) if action_temperature>0 else None,
                'fallback_count':sampling['fallback_count'],
                'rhythm_family_states':dict(sampling['rhythm_family_states']),
                'touch_guard_proposals':sampling['touch_guard_proposals'],
                'touch_guard_safety_rejections':sampling['touch_guard_safety_rejections'],
                'contact_unknown':sampling.get('contact_unknown',0),'contact_reasons':sampling.get('contact_reasons',{}),
                'low_action_support_rejections':sampling.get('low_action_support_rejections',0),
                'min_retained_action_mass':sampling.get('min_retained_mass'),
                'mean_audio_full_pointer_js':float(sampling['audio_pointer_js_sum'])/max(1,sampling['audio_pointer_js_count']) if sampling['audio_pointer_js_count'] else None},
            'configuration_observation':configuration_observation(points),
            'configuration_observation_before_postprocess':raw_configuration_observation,
            'structure_metrics':structure_metrics(points),
            'audio_bounds':bounds,'last_head_sec':last_head_sec,'last_action_end_sec':last_action_end_sec,
            'effective_end_sec':float(end_sec),
            'eos_diagnostics':{'eos_proposals':eos_proposals,'eos_accepted':eos_accepted,'eos_veto_reasons':dict(eos_veto_reasons)},
            'structured_event_generation':True,'hard_constraints':'exact_heads_and_runtime_miacode_transactions',
            'audio_feature_policy':feature_policy,'candidate_policy':candidate_policy,
            'trained_candidate_policy':trained_candidate_policy,'min_same_head_ms':min_same_head_ms,
            'action_sampling_policy':runtime_sampling_policy,
            'rhythm_decode_rules':'rhythm_blank_guard_v2',
            'trained_action_sampling_policy':payload.get('action_sampling_policy'),
            'history_anchor_policy':getattr(model.config,'history_anchor_policy','legacy_digit_v1'),
            'history_memory_policy':model.config.history_memory_policy,
            'history_memory_rms_bound':model.config.history_memory_rms_bound,
            'history_attention_policy':model.config.history_attention_policy,
            'action_state_policy':model.config.action_state_policy,'slide_factor_policy':model.config.slide_factor_policy,
            'motif_memory_policy':model.config.motif_memory_policy,'motif_strength':motif_strength,
            'motif_retrieval':{'queries':motif_queries,'matched_queries':motif_matched_queries,
                'max_events':model.config.motif_max_events,'scope':'committed_archive_configuration_only'},
            'pointer_sampling_policy':pointer_sampling_policy_name(pointer_grid_policy) if grounded else 'global_temperature_truncation_v1',
            'pointer_grid_policy':pointer_grid_policy if grounded else 'off','repetition_guard':repetition_guard,
            'touch_layout_policy':touch_layout_policy,
            'contact_policy':contact_policy,'structure_policy':model.config.structure_policy,
            'future_horizon_policy':model.config.future_horizon_policy,
        'echo_prior_policy':model.config.echo_prior_policy,
        'timing_distribution_policy':getattr(model.config,'timing_distribution_policy','hierarchical_v1'),
        'trained_timing_distribution_policy':trained_timing_policy,'survival_calibration':survival_calibration,
        'survival_rate_contract':('fixed_50ms_grid_plan_rest_prior_preserved_tail' if model.config.timing_distribution_policy=='survival_integrated_v2' else 'trained'),
        'runtime_timing_score_contract':('rate_prior_separate_distance_classifier_unclipped_fine' if model.config.timing_distribution_policy=='survival_integrated_v2' else 'trained_midpoint_bounded_shape'),
        'trained_timing_rate_contract':(payload.get('training_objective_recipe') or {}).get('timing_rate_contract'),
        'button_load_policy':getattr(model.config,'button_load_policy','off'),
        'fine_timing_policy':getattr(model.config,'fine_timing_policy','off'),
            'action_head_policy':model.config.action_head_policy,
            'structure_schema':model.config.structure_schema,'structure_feature_dim':structure_feature_dim,
            'structure_backend':model.config.structure_backend,
            'tempo_source':('provided_map' if args.tempo_map else 'provided_bpm' if args.bpm else 'audio_estimate'),
            'touch_layout_hand_feasibility_guaranteed':False,
            'touch_history_guard':touch_history_guard,
            'touch_history_guard_requested':touch_history_guard_requested,
            'touch_history_bound_requested':touch_history_bound,
            'touch_history_guidance_policy':effective_touch_policy if touch_history_guard else 'off',
            'touch_history_log_odds_bound':touch_history_bound if touch_history_guard else None,
            'density_observation':{'before_postprocess':raw_density,'exported':density_observation(points,clock,touch_layout_policy=touch_layout_policy)},
            'curve_strength':curve_strength,'pointer_history_strength':pointer_history_strength,
            'action_history_strength':action_history_strength,'action_type_temperature':type_temperature,'action_type_min_p':type_min_p,
            'audio_pointer_training_weight':payload.get('args',{}).get('audio_pointer_weight'),
            'trained_action_objective_policy':payload.get('args',{}).get('action_objective_policy','event_balanced_v1'),
            'action_retry_policy':retry_policy,
            'head_timing_policy':model.config.head_timing_policy,
            'cross_event_relation_policy':model.config.cross_event_relation_policy,
            'trained_motif_input_policy':(payload.get('motif_input_policy','legacy_temporal_unit_v1')
                if model.config.motif_memory_policy!='off' else 'off'),
            'runtime_motif_input_policy':(motif_input_policy if model.config.motif_memory_policy!='off' else 'off'),
            'pointer_route':'audio_plus_full_context_residual' if music_v2 else 'full_context',
            'stop_reason':'max_events' if len(points)>=max_events else stop_reason,
            'guard':{'checks':guard.checks,'rejections':guard.rejections,'skipped_candidates':skipped_candidates,
                     'survival_skip_count':survival_skip_count,
                     'retry_prefix_checks':getattr(guard,'retry_prefix_checks',0),
                'rejection_reasons':dict(guard.reason_counts)},
            'playability':playability.to_dict()}
    report['density_feedback_policy']=model.config.density_feedback_policy
    report['trained_density_feedback_policy']=trained_density_feedback
    report['density_feedback_observation']={
        'query_windows':sampling.get('density_feedback_windows',0),
        'mean_grid_log_penalty':float(sampling.get('density_feedback_penalty_sum',0.))/max(1,sampling.get('density_feedback_windows',0)),
        'scope':'past_and_future_query_grids_not_a_musical_quality_score'}
    if context_coverage:
        report['history_coverage']={'events':model.config.history_events,
            'seconds_p10_p50_p90':np.percentile(np.asarray(context_coverage)[:,0],[10,50,90]).tolist(),
            'beats_p10_p50_p90':np.percentile(np.asarray(context_coverage)[:,1],[10,50,90]).tolist(),
            'scope':'committed_decoder_context_not_a_quality_gate'}
    if learned:
        curve=song_workload_curve
        report['learned_workload_curve']={'times_sec':curve_times.tolist(),
            'event_rate':np.expm1(curve[:,0]).tolist(),'head_rate':np.expm1(curve[:,1]).tolist(),
            'hold_fraction':curve[:,2].tolist(),'slide_fraction':curve[:,3].tolist(),
            'touch_fraction':curve[:,4].tolist(),'source':'audio_and_requested_condition',
            'used_directly_as_timing_rate':model.config.timing_distribution_policy!='survival_integrated_v2'}
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
        report['intensity_curve']=curve_controller.summarize_diagnostics(
            points=points,clock=clock,curve_times=curve_times,actual_head_rates=actual_heads)
    if grounded:
        report['audio_lens_cache']={'hits':lens_cache_hits,'misses':lens_cache_misses,'capacity':8192}
        report['coarse_audio_plan']={'times_sec':plan_grid.centers.tolist(),
            'values':planner_prediction[0].cpu().tolist(),
            'channels':['log_event_rate','log_head_rate','hold','slide','touch','rest'],
            'interpolated':True,'source':('intensity_curve_guided' if curve_controller.enabled else 'audio_and_condition_only')}
    output.with_suffix('.composer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main(argv=None):
    args=parser().parse_args(argv);print(json.dumps(generate(args),ensure_ascii=False),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
