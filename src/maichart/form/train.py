"""Train song-form planning and layout with observable, resumable updates."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from maichart.form.data import FormDataset, collate_form, form_audio
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.layout_metrics import layout_diagnostics
from maichart.form import checkpoint
from maichart.form.quality_profile import (
    LEGACY, MUSICAL_EVIDENCE, DEFAULT_ALIGNMENT_WEIGHT,
    DEFAULT_ALIGNMENT_TOLERANCE_MS, descriptor_dim, enhanced,
)
from maichart.form.training_runtime import (
    TRAINING_SEMANTICS, EpochBatches, Progress, UpdateEngine, capture_rng, emit,
    resolve_precision, restore_rng, validate_cpu_batch,
)

# The token vocabulary is fixed by the rational schema, so the per-event
# position objective can share one codec instance instead of rebuilding it for
# every batch.
from maichart.form.codec import RationalCodec as _RationalCodec

_CODEC_FOR_LOSS = _RationalCodec()

# Keep the rational Planner's training evidence distribution identical to
# curve prediction and generation.  The contract lives with rational audio so
# every caller uses one source of truth.
from maichart.form.rational_audio import (
    PLANNER_MAX_AUDIO_NODES as RATIONAL_PLANNER_MAX_AUDIO_NODES,
    planner_audio_queries,
)


def _rational_planner_audio(song):
    """Build the bounded Planner query clock shared with generation."""
    return planner_audio_queries(song)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model-family', choices=('song_form_v1','song_form_v2'), default='song_form_v2')
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--cache-dir', default='cache/song_form')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--prepare-workers', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--cuda-memory-budget-gb', type=float, default=0.0,
                   help='CUDA allocator budget in GiB; 0 uses 60%% of total VRAM, capped at 95%%')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--blueprint-batch-size', type=int, default=None,
                   help='v2 whole-song Planner charts per update; default 4 for fresh training, legacy resume keeps 1')
    p.add_argument('--blueprint-workers', type=int, default=0,
                   help='Planner input preparation threads; 0 chooses up to four workers, one lock per audio source')
    p.add_argument('--blueprint-prefetch', type=int, default=8,
                   help='maximum prepared or pending whole-song Planner charts')
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--limit-charts', type=int, default=0)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--dim-feedforward', type=int, default=0, help='0 uses 3*d-model for v2 or 4*d-model for v1')
    p.add_argument('--planner-width', type=int, default=128, help='independent workload Planner width for v2')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--grad-accumulation', type=int, default=1)
    p.add_argument('--save-every', type=int, default=0)
    p.add_argument('--checkpoint-interval-steps', type=int, default=500,
                   help='save resumable form_last.pt every N successful updates; 0 disables interval saves')
    p.add_argument('--condition-drop-prob', type=float, default=.1)
    p.add_argument('--blueprint-loss-weight', type=float, default=1.)
    p.add_argument('--load-curve-loss-weight', type=float, default=1.)
    p.add_argument('--relation-loss-weight', type=float, default=.1)
    p.add_argument('--event-decision-loss-weight',type=float,default=None,
                   help='v2 auxiliary AT/EOS objective; fresh/finetune default .25, old strict resume retains 0')
    p.add_argument('--rhythm-geometry-profile', choices=('auto','none','exact_fraction_v1','exact_fraction_v2'), default='auto',
                   help='checkpointed v2 numeric Fraction geometry contract; enable only for fresh/finetune')
    p.add_argument('--rhythm-loss-weight', type=float, default=None,
                   help='masked next-event Fraction gap/phase loss; fresh exact profile defaults to .1')
    p.add_argument('--position-event-loss-weight', type=float, default=None,
                   help='average each event Fraction spelling before averaging events, so a long '
                        'denominator is not supervised harder than a one-digit integer; '
                        'fresh v2 defaults to .25, resume inherits')
    p.add_argument('--quality-profile', choices=('auto', LEGACY, MUSICAL_EVIDENCE), default='auto',
                   help='fresh v2 uses balanced musical evidence; resume inherits; finetune upgrades only explicitly')
    p.add_argument('--position-alignment-loss-weight', type=float, default=None,
                   help='authored event/audio correspondence auxiliary objective; enhanced default .15')
    p.add_argument('--position-alignment-tolerance-ms', type=float, default=None,
                   help='maximum authored event/query labeling uncertainty; never an output timing grid')
    p.add_argument('--planner-context-seconds', type=float, default=None,
                   help='left/right audio neighborhoods on actual query timestamps; enhanced default 1 second')
    p.add_argument('--curve-bin-seconds', type=float, default=None,
                   help='training workload bin duration, independent of curve editor display resolution')
    p.add_argument('--quality-enabled', dest='quality_enabled', action='store_true', default=True)
    p.add_argument('--no-quality-enabled', dest='quality_enabled', action='store_false')
    p.add_argument('--max-memory-tokens', type=int, default=512)
    p.add_argument('--max-region-seconds', type=float, default=14.)
    p.add_argument('--meter-map', help='per-song music_structure.json may override this map')
    p.add_argument('--init-checkpoint')
    p.add_argument('--resume')
    p.add_argument('--resume-data-policy', choices=('strict', 'restart_epoch'), default='strict',
                   help='strict continuation, or preserve source/optimizer state and restart the current epoch after preparation changes')
    p.add_argument('--precision', choices=('auto', 'fp32', 'bf16', 'fp16'), default='auto')
    p.add_argument('--loader-workers', type=int, default=0)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--layout-prefetch-batches', type=int, default=None,
                   help='in-process single-thread CPU batch lookahead for loader-workers=0; '
                        'default uses prefetch-factor, 0 disables for comparison')
    p.add_argument('--audio-cache-gb', type=float, default=1.)
    p.add_argument('--audio-cache-mode', choices=('lru', 'resident'), default='lru')
    p.add_argument('--sample-cache-mb', type=int, default=256)
    p.add_argument('--log-interval', type=float, default=10.)
    # Training input perturbation to reduce train-generation gap
    p.add_argument('--perturb-onset-drop-prob', type=float, default=0.0,
                   help='Probability of dropping each onset during training (0-0.3 recommended)')
    p.add_argument('--perturb-onset-add-prob', type=float, default=0.0,
                   help='Probability of adding spurious onsets (0-0.2 recommended)')
    p.add_argument('--perturb-onset-shift-ms', type=float, default=0.0,
                   help='Gaussian shift std for onset timing in milliseconds (0-20 recommended)')
    p.add_argument('--perturb-curve-noise-std', type=float, default=0.0,
                   help='Gaussian noise std for load curve values (0-0.1 recommended)')
    p.add_argument('--perturb-curve-dropout-prob', type=float, default=0.0,
                   help='Probability of zeroing curve frames (0-0.2 recommended)')
    p.add_argument('--perturb-history-corruption-prob', type=float, default=0.0,
                   help='Probability of corrupting history notes (0-0.1 recommended)')
    p.add_argument('--perturb-apply-prob', type=float, default=1.0,
                   help='Overall probability of applying perturbations (0-1)')
    # Repetition penalty and diversity loss
    p.add_argument('--repetition-penalty-weight', type=float, default=0.0,
                   help='Weight for repetition penalty loss (0-0.2 recommended)')
    p.add_argument('--diversity-loss-weight', type=float, default=0.0,
                   help='Weight for pattern diversity loss (0-0.1 recommended)')
    return p


def _validate(args):
    positive = ('epochs', 'batch_size', 'lr', 'grad_accumulation', 'prefetch_factor', 'log_interval', 'blueprint_prefetch')
    nonnegative = ('max_steps', 'limit_charts', 'weight_decay', 'prepare_workers', 'loader_workers',
                   'audio_cache_gb', 'sample_cache_mb', 'checkpoint_interval_steps', 'save_every',
                   'blueprint_loss_weight', 'load_curve_loss_weight', 'relation_loss_weight', 'blueprint_workers')
    for name in positive + nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
            raise ValueError(f'Invalid training setting: {name}')
    if not math.isfinite(args.cuda_memory_budget_gb) or args.cuda_memory_budget_gb < 0:
        raise ValueError('cuda_memory_budget_gb must be nonnegative')
    quality_limits = {
        'perturb_onset_drop_prob': .5, 'perturb_onset_add_prob': .5,
        'perturb_onset_shift_ms': 50., 'perturb_curve_noise_std': .3,
        'perturb_curve_dropout_prob': .5, 'perturb_history_corruption_prob': .3,
        'perturb_apply_prob': 1., 'repetition_penalty_weight': .5,
        'diversity_loss_weight': .3,
    }
    for name, maximum in quality_limits.items():
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f'{name} must be between 0 and {maximum}')
    if args.repetition_penalty_weight:
        # The helper returns a constant without a gradient path to the logits,
        # and teacher-forced batches have no generated prefix to compare.
        raise ValueError('repetition_penalty_weight is not a trainable objective yet; keep it at 0')
    if args.blueprint_batch_size is not None and args.blueprint_batch_size < 1:
        raise ValueError('blueprint_batch_size must be positive')
    if args.blueprint_workers > 4:
        raise ValueError('blueprint_workers cannot exceed 4')
    if args.layout_prefetch_batches is not None and (isinstance(args.layout_prefetch_batches, bool)
            or not 0 <= args.layout_prefetch_batches <= 16):
        raise ValueError('layout_prefetch_batches must be between 0 and 16')
    if args.event_decision_loss_weight is not None and (not math.isfinite(args.event_decision_loss_weight)
            or not 0<=args.event_decision_loss_weight<=10):
        raise ValueError('event_decision_loss_weight must be between 0 and 10')
    if args.rhythm_geometry_profile not in {'auto','none','exact_fraction_v1','exact_fraction_v2'}:
        raise ValueError('Unknown rhythm_geometry_profile')
    if args.rhythm_loss_weight is not None and (not math.isfinite(args.rhythm_loss_weight)
            or not 0 <= args.rhythm_loss_weight <= 10):
        raise ValueError('rhythm_loss_weight must be between 0 and 10')
    if args.position_event_loss_weight is not None and (not math.isfinite(args.position_event_loss_weight)
            or not 0 <= args.position_event_loss_weight <= 10):
        raise ValueError('position_event_loss_weight must be between 0 and 10')
    if args.quality_profile not in ('auto', LEGACY, MUSICAL_EVIDENCE):
        raise ValueError('Unknown quality_profile')
    for name, minimum, maximum in (
            ('position_alignment_loss_weight', 0., 10.),
            ('position_alignment_tolerance_ms', 1., 100.),
            ('planner_context_seconds', 0., 30.), ('curve_bin_seconds', .01, 30.)):
        value = getattr(args, name, None)
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))
                                  or not minimum <= float(value) <= maximum):
            raise ValueError(f'{name} must be between {minimum} and {maximum}')
    if not math.isfinite(args.condition_drop_prob) or not 0 <= args.condition_drop_prob <= 1:
        raise ValueError('condition_drop_prob must be in [0, 1]')
    if args.resume and args.init_checkpoint:
        raise ValueError('Choose resume or finetune, not both')
    if args.resume_data_policy != 'strict' and not args.resume:
        raise ValueError('resume_data_policy only applies to resume')
    if args.audio_cache_mode == 'resident' and (args.loader_workers or args.audio_cache_gb <= 0):
        raise ValueError('Full audio residency requires loader-workers=0 and a positive audio-cache-gb budget')
    if not args.resume and not args.init_checkpoint:
        fresh_model_configs(args)


def fresh_model_configs(args):
    from maichart.form.blueprint import BlueprintConfig
    from maichart.form.codec import RationalCodec
    rational=getattr(args,'model_family','song_form_v1')=='song_form_v2'
    resolve_quality_training_settings(args)
    ff=getattr(args,'dim_feedforward',0) or (3 if rational else 4)*args.d_model
    profile = getattr(args, 'rhythm_geometry_profile', 'none')
    if profile == 'auto':
        profile = 'exact_fraction_v2' if rational else 'none'
    config = FormConfig(vocab_size=len(RationalCodec()), d_model=args.d_model, nhead=args.nhead,
        dropout=args.dropout, layers=args.layers, dim_feedforward=ff,
        max_tokens=args.max_tokens, max_memory_tokens=args.max_memory_tokens,
        rhythm_geometry_profile=profile, quality_profile=args.quality_profile,
        descriptor_dim=descriptor_dim(args.quality_profile), curve_bin_seconds=args.curve_bin_seconds,
        position_alignment_tolerance_ms=args.position_alignment_tolerance_ms)
    if rational:
        from maichart.form.rational_planner import RationalPlannerConfig
        planner_config=RationalPlannerConfig(width=getattr(args,'planner_width',128),
            quality_profile=args.quality_profile, descriptor_dim=descriptor_dim(args.quality_profile),
            context_seconds=args.planner_context_seconds)
    else:
        planner_config = BlueprintConfig(width=max(16, args.d_model//2))
    return config, planner_config


def resolve_quality_training_settings(args, old=None, model=None, planner=None):
    """Resolve one versioned input/target contract before data or optimizers.

    Config-only namespaces are accepted so WebUI preflight uses these same
    defaults. Resume always restores saved values, including when a browser
    still contains values from another model.
    """
    saved_args = (old or {}).get('args') or {}
    config = getattr(model, 'config', None)
    planner_config = getattr(planner, 'config', None)
    saved_config = (old or {}).get('model_config') or {}
    saved_planner_config = (old or {}).get('blueprint_config') or {}
    saved_profile = getattr(config, 'quality_profile', saved_config.get('quality_profile', LEGACY))
    requested = getattr(args, 'quality_profile', 'auto')
    resume = bool(getattr(args, 'resume', None))
    if requested not in ('auto', LEGACY, MUSICAL_EVIDENCE):
        raise ValueError('quality_profile must be auto, legacy, or musical_evidence_v1')
    if resume:
        profile = saved_profile
    elif getattr(args, 'model_family', 'song_form_v1') == 'song_form_v1':
        if requested not in ('auto', LEGACY):
            raise ValueError('song_form_v1 requires quality_profile=legacy')
        profile = LEGACY
    elif old:
        profile = saved_profile if requested == 'auto' else requested
        if enhanced(saved_profile) and profile == LEGACY:
            raise ValueError('quality_profile cannot be downgraded during finetune')
    else:
        profile = MUSICAL_EVIDENCE if requested == 'auto' else requested
    is_enhanced = enhanced(profile)
    same_contract = bool(old and profile == saved_profile)
    defaults = {
        'position_alignment_loss_weight': DEFAULT_ALIGNMENT_WEIGHT if is_enhanced else 0.,
        'position_alignment_tolerance_ms': DEFAULT_ALIGNMENT_TOLERANCE_MS,
        'planner_context_seconds': 1. if is_enhanced else 0.,
        'curve_bin_seconds': 1. if is_enhanced else 2.,
    }
    saved = {
        'position_alignment_loss_weight': saved_args.get('position_alignment_loss_weight', 0.),
        'position_alignment_tolerance_ms': getattr(config, 'position_alignment_tolerance_ms',
            saved_config.get('position_alignment_tolerance_ms',
                saved_args.get('position_alignment_tolerance_ms', DEFAULT_ALIGNMENT_TOLERANCE_MS))),
        'planner_context_seconds': getattr(planner_config, 'context_seconds',
                                           saved_planner_config.get('context_seconds', 0.)),
        'curve_bin_seconds': getattr(config, 'curve_bin_seconds', saved_config.get('curve_bin_seconds', 2.)),
    }
    for name, default in defaults.items():
        value = saved[name] if resume else getattr(args, name, None)
        if value is None:
            value = saved[name] if same_contract else default
        limits = {'position_alignment_loss_weight': (0., 10.),
                  'position_alignment_tolerance_ms': (1., 100.),
                  'planner_context_seconds': (0., 30.), 'curve_bin_seconds': (.01, 30.)}
        minimum, maximum = limits[name]
        if isinstance(value, bool):
            raise ValueError(f'{name} must be a number')
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a number') from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f'{name} must be between {minimum} and {maximum}')
        setattr(args, name, value)
    if not is_enhanced:
        if args.position_alignment_loss_weight or args.planner_context_seconds or args.curve_bin_seconds != 2.:
            raise ValueError('quality_profile=legacy requires position_alignment_loss_weight=0, '
                             'planner_context_seconds=0, curve_bin_seconds=2')
    args.quality_profile = profile
    return args


def resolve_rhythm_geometry_settings(args, old=None, model=None):
    """Resolve profile/loss once before dataset or optimizer construction."""
    saved_profile = (getattr(model.config, 'rhythm_geometry_profile', 'none')
                     if model is not None else 'none')
    requested_profile = args.rhythm_geometry_profile
    if args.model_family == 'song_form_v1':
        if requested_profile not in ('auto', 'none'):
            raise ValueError('song_form_v1 only supports rhythm_geometry_profile=none')
        args.rhythm_geometry_profile = 'none'
    elif getattr(args, 'resume', None):
        if requested_profile not in ('auto', saved_profile):
            raise ValueError('resume must inherit the checkpoint rhythm_geometry_profile')
        args.rhythm_geometry_profile = saved_profile
    elif old:
        args.rhythm_geometry_profile = (saved_profile if requested_profile == 'auto' else requested_profile)
    else:
        args.rhythm_geometry_profile = ('exact_fraction_v2' if requested_profile == 'auto' else requested_profile)
    if args.rhythm_loss_weight is None:
        if getattr(args, 'resume', None):
            args.rhythm_loss_weight = float((old.get('args') or {}).get('rhythm_loss_weight', 0.))
        else:
            args.rhythm_loss_weight = .1 if args.rhythm_geometry_profile != 'none' else 0.
    if old and getattr(args, 'init_checkpoint', None) and saved_profile == 'exact_fraction_v1' and args.rhythm_geometry_profile == 'none':
        raise ValueError('finetune cannot remove the checkpoint rhythm_geometry_profile')
    if args.rhythm_geometry_profile=='none' and args.rhythm_loss_weight:
        raise ValueError('rhythm_loss_weight requires an exact geometry profile')
    return args.rhythm_geometry_profile, args.rhythm_loss_weight


def layout_loss(prediction, batch, relation_weight, event_decision_weight=0., rhythm_loss_weight=0.,
                repetition_penalty_weight=0., diversity_weight=0., position_event_weight=0.,
                position_alignment_weight=0.):
    """Per-example objectives. Their sum can be accumulated across batch sizes."""
    logits = prediction['logits'].float()
    valid = batch['target'] != 0
    ce = F.cross_entropy(logits.transpose(1, 2), batch['target'], ignore_index=0, reduction='none')
    weights = batch['weight'].float()
    per_token = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    relation = F.binary_cross_entropy_with_logits(
        prediction['relation_logits'].float(), batch['relation_target'].float(), reduction='none').mean(1)
    relation = relation * batch['relation_valid'].float() * weights
    token = per_token * weights
    if repetition_penalty_weight:
        raise ValueError('repetition_penalty_weight is not a trainable objective yet; keep it at 0')

    # Add diversity loss if enabled
    diversity_loss = weights.new_zeros(weights.shape)
    if diversity_weight > 0 and 'hidden_states' in prediction:
        from maichart.form.repetition_penalty import compute_pattern_diversity_loss
        diversity_loss = compute_pattern_diversity_loss(
            prediction['hidden_states'],
            segment_size=8,
            diversity_weight=diversity_weight,
            valid_mask=valid, reduction='none',
        )
        diversity_loss = diversity_loss * weights

    loss_sum = (token + relation_weight * relation).sum() + diversity_loss.sum()
    auxiliary = None
    if event_decision_weight:
        from maichart.form.rational_objectives import event_decision_loss
        auxiliary = event_decision_loss(logits,batch)*weights
        loss_sum=loss_sum+event_decision_weight*auxiliary.sum()
    numeric_timing = None
    if rhythm_loss_weight:
        from maichart.form.rhythm_geometry import numeric_timing_loss
        if 'timing_prediction' not in prediction or 'timing_labels' not in batch or 'timing_mask' not in batch:
            raise ValueError('numeric timing objective requires an exact geometry model and dataset labels')
        numeric_timing = numeric_timing_loss(prediction['timing_prediction'], batch['timing_labels'], batch['timing_mask'], reduction='none') * weights
        loss_sum = loss_sum + rhythm_loss_weight * numeric_timing.sum()
    position_event = None
    if position_event_weight:
        # Token CE weights an event by its spelling length.  A Fraction costs
        # one digit per character, so a long denominator is supervised several
        # times as hard as a one-digit integer and the objective drifts toward
        # whichever events happen to spell long.  Average each event's position
        # spelling first so one event is one unit of supervision, then weight
        # by the same per-example weight the token objective uses.
        from maichart.form.rhythm_geometry import position_event_loss
        position_event = position_event_loss(logits, batch['target'], batch['decoder_input'],
                                             _CODEC_FOR_LOSS, reduction='none',token_ce=ce) * weights
        loss_sum = loss_sum + position_event_weight * position_event.sum()
    alignment = None
    alignment_metrics = {}
    if position_alignment_weight:
        from maichart.form.alignment import position_alignment_loss
        alignment_per_row, alignment_metrics = position_alignment_loss(prediction, batch)
        alignment = alignment_per_row * weights
        loss_sum = loss_sum + position_alignment_weight * alignment.sum()
    correct = ((logits.detach().argmax(-1) == batch['target']) & valid).sum()
    if 'decoder_input' in batch and 'task' in batch:
        diagnostic_values, diagnostic_counts = layout_diagnostics(
            logits, batch['target'], batch['decoder_input'], batch['task'], ce)
    else:
        diagnostic_values, diagnostic_counts = {}, {}
    if alignment is not None:
        draft_rows = (batch['task'] == 0) & (weights > 0)
        active_events = batch['alignment_event_mask'].any(1) & draft_rows
        authored_events = batch['alignment_event_indices'].ne(-1).any(1) & draft_rows
        active_queries = batch['position_alignment_mask'].any(1) & draft_rows
        rows_by_metric = {'position_alignment_event_accuracy': active_events,
                          'position_alignment_coverage': authored_events,
                          'position_alignment_loss': active_queries}
        for key, row_mask in rows_by_metric.items():
            values = alignment.detach() if key == 'position_alignment_loss' else alignment_metrics[key]
            count = row_mask.sum()
            diagnostic_values[key] = values.masked_fill(~row_mask, 0).sum()/count.clamp_min(1)
            diagnostic_counts[key] = count
    return loss_sum, {'token_loss': token.detach().mean(), 'relation_loss': relation.detach().mean(),
                      **({'diversity_loss': diversity_loss.detach().mean()} if diversity_weight else {}),
                      **({'event_decision_aux_loss':auxiliary.detach().mean()} if auxiliary is not None else {}),
                      **({'position_event_aux_loss':position_event.detach().mean()} if position_event is not None else {}),
                      **({'numeric_timing_aux_loss':numeric_timing.detach().mean()} if numeric_timing is not None else {}),
                      'loss': loss_sum.detach() / len(weights),
                      'token_accuracy': correct.float() / valid.sum().clamp_min(1),
                      'diagnostics': {'values': diagnostic_values, 'counts': diagnostic_counts}}


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _blueprint_input(dataset, index):
    """CPU targets retain exact tempo/first; excluded spans get zero loss weight."""
    from maichart.form.blueprint import build_blueprint_targets
    from maichart.form.curve import resample_load_curve
    from maichart.event_timing import seconds_at_beat
    from maichart.generation.event_audio import segment_frame_times
    song, points, level, weight, _, condition, *rest = dataset.songs[index]
    if getattr(song, 'audio_profile', 'legacy_50hz') == 'rational_audio_queries_v1':
        audio,times,_=_rational_planner_audio(song)
        curve=resample_load_curve(song.load_curve_times_sec,song.load_curve,times-song.first)
        boundaries=np.r_[times[0],(times[1:]+times[:-1])*.5,
            seconds_at_beat(song.spans[-1].end,song.tempo,song.first)]
        weights=np.maximum(np.diff(boundaries),0).astype(np.float32)
        weights/=max(float(weights.mean()),1e-8)
        record=rest[1] if len(rest)>1 else None
        for region in (record or {}).get('regions',[]):
            weights[(times>=region['start_sec'])&(times<region['end_sec'])]*=region.get('weight',1.)
        from maichart.form.data import _form_descriptor
        inputs={'audio':audio[None],'form':torch.from_numpy(_form_descriptor(song,0))[None],
                'condition':condition.to_vector()[None]}
        if enhanced(getattr(song, 'quality_profile', LEGACY)):
            inputs['audio_times_sec'] = torch.as_tensor(times, dtype=torch.float64)[None]
        targets={'load_curve_target':torch.from_numpy(curve)[None],
                 'load_curve_mask':torch.ones((1,len(times),4),dtype=torch.bool),
                 'frame_weight':torch.from_numpy(weights)[None]}
        validate_cpu_batch(inputs);validate_cpu_batch(targets)
        return inputs,targets,float(weight)
    audio = form_audio(song, song.spans[0].start, song.spans[-1].end)
    times = segment_frame_times(seconds_at_beat(song.spans[0].start, song.tempo, song.first),
                                seconds_at_beat(song.spans[-1].end, song.tempo, song.first))
    targets = build_blueprint_targets(points, song.tempo, song.first, times,
        frame_end_sec=min(float(song.audio.duration_sec),
                          float(seconds_at_beat(song.spans[-1].end, song.tempo, song.first))))
    curve = resample_load_curve(song.load_curve_times_sec, song.load_curve, times-float(song.first))
    targets['load_curve_target'] = torch.from_numpy(curve)
    targets['load_curve_mask'] = torch.ones_like(targets['load_curve_target'], dtype=torch.bool)
    frame_weight = np.ones(len(times), dtype=np.float32)
    record = rest[1] if len(rest) > 1 else None
    for region in (record or {}).get('regions', []):
        selected = (times >= float(region['start_sec'])) & (times < float(region['end_sec']))
        frame_weight[selected] *= float(region.get('weight', 1.))
    targets['frame_weight'] = torch.from_numpy(frame_weight)
    from maichart.form.data import _form_descriptor
    form = torch.from_numpy(_form_descriptor(song, 0))
    inputs = {'audio': audio[None], 'form': form[None],
              'condition': condition.to_vector()[None]}
    validate_cpu_batch(inputs)
    validate_cpu_batch(targets)
    return inputs, {k: v[None] for k, v in targets.items()}, float(weight)


def main(argv=None):
    args = parser().parse_args(argv)
    _validate(args)
    from maichart.form.output_guard import training_output_guard
    with training_output_guard(args.checkpoint_dir):
        return _train(args)


def _train(args):
    old = None
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    cuda_budget_gib = 0.0
    cuda_budget_error = None
    if device.type == 'cuda':
        # PyTorch rejects a bare ``cuda`` device for this API. Resolve it to
        # the active ordinal while preserving an explicit cuda:N selection.
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / 1024**3
        requested = float(args.cuda_memory_budget_gb)
        cuda_budget_gib = min(total_gib * .95, requested if requested > 0 else total_gib * .60)
        try:
            torch.cuda.set_per_process_memory_fraction(cuda_budget_gib / total_gib, device=cuda_index)
        except (RuntimeError, ValueError) as exc:
            # A cap is an optimization guard; failure to install it must not
            # abort an otherwise valid training run. Keep the requested value
            # visible and explain the fallback in the structured log.
            cuda_budget_error = str(exc)
    meters = json.loads(Path(args.meter_map).read_text(encoding='utf-8-sig')) if args.meter_map else None
    if args.resume or args.init_checkpoint:
        model, blueprint, codec, old = checkpoint.load(args.resume or args.init_checkpoint, device='cpu')
        args.model_family = old['family']
        for key in ('max_tokens', 'max_memory_tokens', 'd_model', 'layers', 'nhead', 'dropout'):
            setattr(args, key, getattr(model.config, key))
        if args.resume:
            saved_args = old.get('args') or {}
            for name in ('seed', 'weight_decay', 'grad_accumulation', 'condition_drop_prob',
                         'blueprint_loss_weight', 'load_curve_loss_weight', 'relation_loss_weight',
                          'quality_enabled', 'max_region_seconds', 'limit_charts', 'batch_size', 'max_steps',
                          'rhythm_loss_weight', 'position_event_loss_weight'):
                if name in saved_args:
                    setattr(args, name, saved_args[name])
            from maichart.form.training_input_perturbation import QUALITY_TRAINING_DEFAULTS
            for name, default in QUALITY_TRAINING_DEFAULTS.items():
                setattr(args, name, saved_args.get(name, default))
            # Never had a gradient path; zero keeps the saved run's numerics.
            args.repetition_penalty_weight = 0.
            # Resume retains the optimizer's numerical regime. Old trainers
            # were FP32; performance options remain adjustable during finetune.
            args.precision = old.get('precision', saved_args.get('precision', 'fp32'))
            meters = (old.get('dataset_contract') or {}).get('meter_map', meters)
    # The profile is architectural. Resume always inherits it; finetune only
    # migrates when the user explicitly requests exact_fraction_v1.
    saved_profile = getattr(model.config, 'rhythm_geometry_profile', 'none') if old else 'none'
    saved_quality = getattr(model.config, 'quality_profile', LEGACY) if old else LEGACY
    resolve_rhythm_geometry_settings(args, old, model if old else None)
    resolve_quality_training_settings(args, old, model if old else None, blueprint if old else None)
    quality_changed = bool(old and saved_quality != args.quality_profile)
    planner_contract_changed = bool(old and (quality_changed
        or float(getattr(model.config, 'curve_bin_seconds', 2.)) != args.curve_bin_seconds
        or float(getattr(blueprint.config, 'context_seconds', 0.)) != args.planner_context_seconds))
    alignment_contract_changed = bool(old and (quality_changed
        or float(getattr(model.config, 'position_alignment_tolerance_ms', 15.))
           != args.position_alignment_tolerance_ms))
    layout_contract_changed = bool(old and (quality_changed or planner_contract_changed
        or saved_profile != args.rhythm_geometry_profile))
    semantics=('song_form_training_v6_musical_evidence' if enhanced(args.quality_profile) else
               (('song_form_training_v5_exact_geometry' if args.rhythm_geometry_profile!='none'
                else 'song_form_training_v5_rational_inputs')
               if args.model_family=='song_form_v2' else TRAINING_SEMANTICS))
    _validate(args)
    if args.model_family=='song_form_v2':
        if args.resume:
            args.event_decision_loss_weight=(old.get('args') or {}).get('event_decision_loss_weight',0.)
        elif args.event_decision_loss_weight is None:
            args.event_decision_loss_weight=.25
        if args.position_event_loss_weight is None:
            if args.resume:
                # A resumed run keeps the objective it was trained with; adding
                # a new term mid-run would change the loss scale the optimizer
                # state was built against.
                args.position_event_loss_weight = float(
                    (old.get('args') or {}).get('position_event_loss_weight', 0.))
            else:
                args.position_event_loss_weight = .25
        saved_blueprint_batch_size = (old.get('args') or {}).get('blueprint_batch_size', 1) if args.resume else 4
        args.blueprint_batch_size = ((saved_blueprint_batch_size or 1) if args.resume
                                    else args.blueprint_batch_size or saved_blueprint_batch_size or 1)
    else:
        args.event_decision_loss_weight=0.
        # The per-event position objective groups rational Fraction spellings,
        # which only exist in the v2 codec. Resolve it to an explicit 0 so the
        # training loop never compares None.
        args.position_event_loss_weight = 0.
        args.blueprint_batch_size = 1
    planner_objective_enabled = (args.blueprint_loss_weight > 0
        and (args.model_family != 'song_form_v2' or args.load_curve_loss_weight > 0))
    precision = resolve_precision(args.precision, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    output = Path(args.checkpoint_dir)
    if old is None and (output/'form_last.pt').is_file():
        raise ValueError('从头训练请使用新输出目录；该目录已存在 form_last.pt，避免覆盖现有模型。')
    if old and args.init_checkpoint and Path(args.init_checkpoint).resolve().parent == output.resolve():
        raise ValueError('Finetune uses a separate output directory to preserve the source model')
    if args.resume and args.resume_data_policy == 'restart_epoch' and Path(args.resume).resolve().parent == output.resolve():
        raise ValueError('迁移续训请使用新的 checkpoint 输出目录，以保留来源模型。')
    from maichart.form.resume_contract import (
        contract_header, preflight_resume, verify_saved_chart_sources,
        compare_resume_contract, resolve_resume_contract, validate_resume_cursor,
    )
    current_header = contract_header(args.manifest, meters, args.max_region_seconds)
    if args.resume:
        preflight_resume(old.get('dataset_contract') or {}, current_header)
        verify_saved_chart_sources(old.get('dataset_contract') or {})
    output.mkdir(parents=True, exist_ok=True)
    blueprint_workers = (min(4, os.cpu_count() or 1) if args.blueprint_workers == 0
                         else args.blueprint_workers) if args.model_family=='song_form_v2' else 0
    emit('form_runtime', training_semantics=semantics, model_family=args.model_family, device=str(device), precision=precision,
         cache_dir=str(Path(args.cache_dir).resolve()),max_region_seconds=args.max_region_seconds,
         cuda_memory_budget_gb=round(cuda_budget_gib, 3), cuda_memory_budget_requested_gb=float(args.cuda_memory_budget_gb),
         cuda_memory_budget_error=cuda_budget_error,
         batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
         blueprint_batch_size=args.blueprint_batch_size,
         planner_objective_enabled=planner_objective_enabled,
         blueprint_workers=blueprint_workers, blueprint_prefetch=args.blueprint_prefetch,
         event_decision_loss_weight=args.event_decision_loss_weight,
         rhythm_geometry_profile=args.rhythm_geometry_profile,
         rhythm_loss_weight=args.rhythm_loss_weight,
         position_event_loss_weight=args.position_event_loss_weight,
         quality_profile=args.quality_profile,
         position_alignment_loss_weight=args.position_alignment_loss_weight,
         position_alignment_tolerance_ms=args.position_alignment_tolerance_ms,
         planner_context_seconds=args.planner_context_seconds, curve_bin_seconds=args.curve_bin_seconds,
         effective_batch_size=args.batch_size*args.grad_accumulation, loader_workers=args.loader_workers,
         prepare_workers=args.prepare_workers, prefetch_factor=args.prefetch_factor,
         layout_prefetch_batches=(args.layout_prefetch_batches if args.layout_prefetch_batches is not None
                                  else min(args.prefetch_factor, 16)) if not args.loader_workers else 0,
         cache_budget_scope='per_process', audio_cache_gb=args.audio_cache_gb,
         audio_cache_mode=args.audio_cache_mode,
         sample_cache_mb=args.sample_cache_mb, checkpoint_interval_steps=args.checkpoint_interval_steps)
    if cuda_budget_error:
        emit('form_warning', message='无法安装 CUDA 显存硬上限，训练继续使用 PyTorch 默认分配策略。',
             cuda_memory_budget_error=cuda_budget_error)
    if args.model_family == 'song_form_v2' and not planner_objective_enabled:
        emit('form_warning', message='全曲负荷规划损失已关闭：Planner 保持冻结，不计训练步数。只有来源模型已学会全曲规划时才能生成；从头训练需开启规划损失与曲线预测损失。',
             planner_pretrained=bool(old and checkpoint.planner_is_ready(old)))
    if args.loader_workers:
        emit('form_warning', message='Loader workers copy dataset metadata and each has its own cache; memory grows with worker count.')
    if old and old.get('training_semantics') != semantics:
        emit('form_warning', message='Checkpoint uses different training semantics. Prepared audio/skeleton identities are now explicit; optional exact geometry changes the versioned input contract. Use finetune for changed inputs. Source checkpoint generation keeps its own profile.',
             optimizer_restored=bool(args.resume))
    emit('form_prepare', stage='dataset', manifest=str(Path(args.manifest).resolve()),
         cache_dir=str(Path(args.cache_dir).resolve()),max_region_seconds=args.max_region_seconds,
         limit_charts=args.limit_charts, max_steps=args.max_steps, prepare_workers=args.prepare_workers)
    # Create perturbation config for training input robustness
    from maichart.form.training_input_perturbation import PerturbationConfig
    perturbation_config = PerturbationConfig(
        onset_drop_prob=args.perturb_onset_drop_prob,
        onset_add_prob=args.perturb_onset_add_prob,
        onset_shift_sigma_ms=args.perturb_onset_shift_ms,
        curve_noise_std=args.perturb_curve_noise_std,
        curve_dropout_prob=args.perturb_curve_dropout_prob,
        history_corruption_prob=args.perturb_history_corruption_prob,
        apply_prob=args.perturb_apply_prob,
        seed=args.seed + 999999  # Different seed for perturbation
    )
    if perturbation_config.is_enabled():
        emit('form_perturbation', enabled=True,
             onset_drop_prob=perturbation_config.onset_drop_prob,
             onset_add_prob=perturbation_config.onset_add_prob,
             onset_shift_ms=perturbation_config.onset_shift_sigma_ms,
             curve_noise_std=perturbation_config.curve_noise_std,
             curve_dropout_prob=perturbation_config.curve_dropout_prob,
             history_corruption_prob=perturbation_config.history_corruption_prob,
             apply_prob=perturbation_config.apply_prob)
    dataset = FormDataset(args.manifest, cache_dir=args.cache_dir, max_tokens=args.max_tokens,
                          max_memory_tokens=args.max_memory_tokens, meters=meters, seed=args.seed,
                          limit_charts=args.limit_charts, max_region_seconds=args.max_region_seconds,
                          quality_enabled=args.quality_enabled, preparation_workers=args.prepare_workers,
                          audio_cache_gb=args.audio_cache_gb, sample_cache_mb=args.sample_cache_mb,
                          audio_profile='rational_audio_queries_v1' if args.model_family=='song_form_v2' else 'legacy_50hz',
                          rhythm_geometry_profile=args.rhythm_geometry_profile,
                          quality_profile=args.quality_profile, curve_bin_seconds=args.curve_bin_seconds,
                          position_alignment_tolerance_ms=args.position_alignment_tolerance_ms,
                          perturbation_config=perturbation_config)
    contract = {**current_header, 'charts': dataset.audit['charts']}
    migration = None
    if args.resume:
        diff = compare_resume_contract(old.get('dataset_contract') or {}, contract)
        if not diff['compatible']:
            (output/'resume_contract_diff.json').write_text(json.dumps(diff,ensure_ascii=False,indent=2),encoding='utf-8')
            emit('form_resume_contract', old_charts=diff['old_chart_count'], new_charts=diff['new_chart_count'],
                 added=len(diff['added']), removed=len(diff['removed']), changed=len(diff['changed']),
                 can_restart_epoch=diff['can_restart_epoch'], policy=args.resume_data_policy,
                 report=str(output/'resume_contract_diff.json'))
        resolve_resume_contract(old.get('dataset_contract') or {}, contract, args.resume_data_policy)
        if not diff['compatible']:
            migration = {'source_checkpoint':str(Path(args.resume).resolve()),
                         'old_chart_count':diff['old_chart_count'],'new_chart_count':diff['new_chart_count'],
                         'old_position':old.get('training_position'), 'restart_epoch':int(old.get('epoch',0))+1,
                         'optimizer_restored':True,'policy':args.resume_data_policy}
    if args.audio_cache_mode == 'resident':
        residency = dataset.preload_audio(lambda **fields: emit('form_audio_cache', mode='resident', **fields))
        emit('form_audio_cache', **residency, phase='in_memory_training')
    from maichart.form.blueprint import BlueprintConfig, SongBlueprint, compute_blueprint_loss
    if old is None:
        codec = dataset.codec
        config, planner_config = fresh_model_configs(args)
        model = FormTransformer(config)
        if args.model_family=='song_form_v2':
            from maichart.form.rational_planner import RationalPlanner
            blueprint = RationalPlanner(planner_config)
        else:
            blueprint = SongBlueprint(planner_config)
    elif args.init_checkpoint:
        from dataclasses import replace
        migrated_config = replace(model.config, rhythm_geometry_profile=args.rhythm_geometry_profile,
            quality_profile=args.quality_profile, descriptor_dim=descriptor_dim(args.quality_profile),
            curve_bin_seconds=args.curve_bin_seconds,
            position_alignment_tolerance_ms=args.position_alignment_tolerance_ms)
        if migrated_config != model.config:
            model = checkpoint.load_finetune_model(args.init_checkpoint, migrated_config, device='cpu')
        if args.model_family == 'song_form_v2':
            migrated_planner_config = replace(blueprint.config, quality_profile=args.quality_profile,
                descriptor_dim=descriptor_dim(args.quality_profile), context_seconds=args.planner_context_seconds)
            if migrated_planner_config != blueprint.config:
                blueprint = checkpoint.load_finetune_planner(args.init_checkpoint, migrated_planner_config, device='cpu')
        if quality_changed:
            emit('form_quality_migration', source_profile=saved_quality, target_profile=args.quality_profile,
                 message='新音频证据、曲线密度与辅助目标需要微调；保留来源权重，规划能力将重新计数。')
    model.to(device); blueprint.to(device)
    # Inputs are checked on CPU before copying. Shapes are always checked in
    # the model; avoid per-forward GPU -> host scalar synchronizations.
    model.validate_inputs = False
    blueprint.validate_inputs = False
    from maichart.form.monitoring import training_model_summary, write_training_model_summary
    model_summary = training_model_summary(model, blueprint)
    write_training_model_summary(output, model_summary)
    emit('form_model_summary', **model_summary)
    parameters = list(model.parameters()) + list(blueprint.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(old['optimizer_state_dict'])
    engine = UpdateEngine(optimizer, parameters, device, precision)
    if args.resume and old.get('grad_scaler_state') and engine.scaler.is_enabled():
        engine.scaler.load_state_dict(old['grad_scaler_state'])
    steps = int(old.get('optimizer_steps', 0)) if old else 0
    plan_steps = int(old.get('blueprint_steps', 0)) if old else 0
    planner_effective_steps = (int(old.get('planner_effective_steps', plan_steps))
                               if old and not planner_contract_changed and checkpoint.planner_is_ready(old) else 0)
    task_training = dict(old.get('task_training', {})) if old else {'draft': 0, 'revision': 0}
    capabilities = dict(old.get('capabilities', {})) if old and old.get('training_semantics') == semantics else {}
    if layout_contract_changed:
        # Old lifetime task counts do not establish that newly added input
        # branches have received even one layout update in the new contract.
        for task_name in ('draft', 'revision'):
            capabilities[task_name] = {'steps': 0, 'contract': 'form_condition_v2'}
    if planner_contract_changed:
        capabilities.pop('load_curve', None)
    if alignment_contract_changed:
        capabilities.pop('position_alignment', None)
    history = list(old.get('training_history', [])) if args.resume else []
    start_epoch = int(old.get('epoch', 0)) if args.resume else 0
    cursor = dict(old.get('training_position') or {}) if args.resume else {}
    migrations = list(old.get('data_migrations') or []) if args.resume else []
    if migration:
        cursor = {}
        migrations.append(migration)
        emit('form_resume_migration', **migration,
             message='保留模型与优化器状态；丢弃旧样本游标，从未完成轮次的规划阶段重新开始。')
    first_items = {}
    for i, item in enumerate(dataset.items):
        first_items.setdefault(item[0], i)
    plan_indices = [i for i in range(len(dataset.songs)) if i in first_items]
    if args.max_steps:
        plan_indices = plan_indices[:args.max_steps]
    plan_total = len(plan_indices)
    layout_total = min(math.ceil(len(dataset)/args.batch_size), args.max_steps or math.ceil(len(dataset)/args.batch_size))
    if args.resume:
        validate_resume_cursor(cursor, epoch=start_epoch, layout_total=layout_total,
                               plan_total=plan_total, batch_size=args.batch_size, max_steps=args.max_steps,
                               blueprint_batch_size=args.blueprint_batch_size)
    (output/'dataset_audit.json').write_text(json.dumps(dataset.audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'form_training charts={len(dataset.songs)} examples={len(dataset)} device={device} parameters={sum(p.numel() for p in parameters)}', flush=True)
    emit('form_dataset', charts=len(dataset.songs), examples=len(dataset), draft_examples=sum(item[3] == 0 for item in dataset.items),
         revision_examples=sum(item[3] == 1 for item in dataset.items), layout_batches=layout_total,
         blueprint_charts=plan_total, optimizer_updates_per_layout_epoch=math.ceil(layout_total/args.grad_accumulation),
         skipped_reasons=dict(Counter(str(item.get('reason', 'unknown')) for item in dataset.audit.get('skipped', []))),
         prepare_elapsed_sec=round(time.perf_counter()-started, 2))
    atoms = sorted({atom for _, points, *_ in dataset.songs for point in points for atom in _atoms(point.note_text)}
                   | set(old.get('atom_catalog', []) if old else []))
    last_save_steps = steps + plan_steps
    progress = None
    metrics = {}

    def save(epoch, phase, completed, *, complete=False, reason='interval'):
        nonlocal last_save_steps
        if engine.samples:
            raise ValueError('Checkpoint requested inside an accumulation group')
        tick = time.perf_counter()
        saved = checkpoint.payload(model, blueprint, codec, epoch=epoch if complete else epoch-1,
            optimizer_steps=steps, blueprint_steps=plan_steps, task_training=dict(task_training),
            planner_effective_steps=planner_effective_steps,
            capabilities={k:dict(v) for k,v in capabilities.items()}, optimizer_state_dict=optimizer.state_dict(),
            grad_scaler_state=engine.scaler.state_dict(), dataset_contract=contract, args=vars(args),
            atom_catalog=atoms, metrics=metrics, training_history=history,
            data_migrations=migrations,
            condition_supervision='event_derived_radar_style_v1', training_semantics=semantics,
            precision=precision,
            training_position={'epoch':epoch, 'phase':'complete' if complete else phase, 'completed':completed,
                               'layout_total':layout_total, 'plan_total':plan_total, 'batch_size':args.batch_size,
                               'blueprint_batch_size':args.blueprint_batch_size,
                               'max_steps':args.max_steps}, rng_state=capture_rng(device),
            supervision='authored_draft_and_masked_reconstruction_not_aesthetic_reward')
        _atomic_save(saved, output/'form_last.pt')
        if complete and args.save_every and epoch % args.save_every == 0:
            _atomic_save(saved, output/f'form_epoch_{epoch:03d}.pt')
        elapsed = time.perf_counter()-tick
        if progress:
            progress.timing['checkpoint_sec'] += elapsed
        last_save_steps = steps + plan_steps
        emit('form_checkpoint', path=str(output/'form_last.pt'), epoch=epoch, phase=phase,
             batch=completed, optimizer_steps=steps, blueprint_steps=plan_steps,
             epoch_complete=complete, resumable=True, reason=reason, elapsed_sec=round(elapsed, 3))

    def increment_cap(name, amount, contract_name):
        value = capabilities.setdefault(name, {'steps': 0, 'contract': contract_name})
        value['steps'] += amount
        value['contract'] = contract_name

    if args.resume:
        if migration:
            restart_seed = args.seed+(start_epoch+1)*100003
            random.seed(restart_seed);np.random.seed(restart_seed);torch.manual_seed(restart_seed)
        else:
            restore_rng(old.get('rng_state'), device)
        if cursor and cursor.get('phase') != 'complete':
            emit('form_resume', **cursor, optimizer_steps=steps, blueprint_steps=plan_steps)
    for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
        dataset.seed = args.seed + epoch*100003
        model.train(); blueprint.train()
        progress = Progress(epoch, plan_total, layout_total, args.log_interval, device, precision)
        plan_start = layout_start = 0
        if cursor.get('epoch') == epoch and cursor.get('phase') != 'complete':
            if cursor.get('phase') == 'blueprint':
                plan_start = int(cursor['completed'])
            elif cursor.get('phase') == 'layout_revision':
                plan_start = plan_total
                layout_start = int(cursor['completed'])
        phase_loss_sums = dict(plan_loss=0., layout_loss=0.)
        phase_counts = dict(plan_loss=0, layout_loss=0)
        layout_metric_sums = {}
        layout_metric_counts = {}
        progress.begin('blueprint', plan_start)
        plan_batch_size = args.blueprint_batch_size
        blueprint_cache = None
        if args.model_family=='song_form_v2':
            from maichart.form.blueprint_batch import collate_blueprint
            from maichart.form.blueprint_cache import BlueprintInputCache
            from maichart.form.blueprint_prefetch import iter_prefetched_blueprints
            source_locks = {}
            source_locks_guard = threading.Lock()
            from maichart.form.rational_audio import extract_rational_audio
            blueprint_cache = (BlueprintInputCache(args.cache_dir)
                               if getattr(dataset.audio_store, 'loader', None) is extract_rational_audio
                               else None)

            def prepare_blueprint(index):
                audio = dataset.songs[index][0].audio
                source = str(getattr(audio, 'path', getattr(audio, 'source', index)))
                with source_locks_guard:
                    lock = source_locks.setdefault(source, threading.Lock())
                with lock:
                    if blueprint_cache is None:
                        return _blueprint_input(dataset, index)
                    key, identity = blueprint_cache.identity_for_dataset(
                        dataset, index, max_tokens=args.max_tokens,
                        max_region_seconds=args.max_region_seconds,
                        audio_feature_config={'profile': 'rational_audio_queries_v1',
                                              'max_nodes': RATIONAL_PLANNER_MAX_AUDIO_NODES})
                    inputs, targets, weight, _ = blueprint_cache.get_or_build(
                        key, identity, lambda: _blueprint_input(dataset, index))
                    return inputs, targets, weight

            plan_stream = iter_prefetched_blueprints(plan_indices[plan_start:], prepare_blueprint,
                                                     workers=blueprint_workers, depth=args.blueprint_prefetch)
        else:
            plan_stream = None
        stream_entered = False
        try:
            if plan_stream is not None:
                plan_stream.__enter__()
                stream_entered = True
                stream_iter = iter(plan_stream)
            else:
                stream_iter = None
            for position in range(plan_start, plan_total, plan_batch_size):
                end_position = min(position+plan_batch_size, plan_total)
                chart_count = end_position-position
                before = time.perf_counter()
                if args.model_family=='song_form_v2':
                    samples = [next(stream_iter)[1] for _ in range(chart_count)]
                    inputs, targets, weights, trainable = collate_blueprint(samples)
                    update_samples = int(trainable.sum())
                else:
                    inputs, targets, weight = _blueprint_input(dataset, plan_indices[position])
                    update_samples = int(weight > 0 and bool(targets['frame_weight'].sum() > 0))
                progress.timing['data_wait_sec'] += time.perf_counter()-before
                tick = time.perf_counter()
                inputs = {k:v.to(device, non_blocking=True) for k,v in inputs.items()}
                targets = {k:v.to(device, non_blocking=True) for k,v in targets.items()}
                if args.model_family=='song_form_v2':
                    weights = weights.to(device, non_blocking=True)
                progress.timing['h2d_sec'] += time.perf_counter()-tick
                tick = time.perf_counter()
                with engine.autocast():
                    if args.model_family=='song_form_v2':
                        null=torch.rand(inputs['condition'].shape[0],device=device)<args.condition_drop_prob
                        planner_kwargs = ({'audio_times_sec': inputs['audio_times_sec']}
                                          if enhanced(args.quality_profile) else {})
                        prediction=blueprint(inputs['audio'],inputs['form'],inputs['condition'],
                            audio_padding_mask=inputs['audio_padding_mask'],
                            form_padding_mask=inputs['form_padding_mask'],condition_null=null, **planner_kwargs)
                    else:
                        prediction = blueprint(inputs['audio'], inputs['form'], inputs['condition'])
                if args.model_family=='song_form_v2':
                    from maichart.form.rational_planner import rational_planner_loss
                    parts = rational_planner_loss(prediction,targets,load_curve_weight=args.load_curve_loss_weight,
                                                  reduction='none')
                    loss = (parts['loss'] * weights * args.blueprint_loss_weight).sum()
                    # Keep a calibration signal beside the loss.  A low loss
                    # can still hide a planner that predicts a flat workload;
                    # this diagnostic uses the same non-padding frame mask as
                    # the objective and does not affect gradients.
                    planner_valid = targets['frame_weight'] > 0
                    planner_valid &= ~inputs['audio_padding_mask']
                    planner_denominator = planner_valid.sum().clamp_min(1).to(
                        prediction['load_curve'].dtype)
                    planner_density = prediction['load_curve'][..., 0].sigmoid()
                    planner_target_density = targets['load_curve_target'][..., 0].to(
                        planner_density)
                    parts['planner_density_prediction'] = (
                        planner_density.masked_select(planner_valid).sum()
                        / planner_denominator)
                    parts['planner_density_target'] = (
                        planner_target_density.masked_select(planner_valid).sum()
                        / planner_denominator)
                    parts = {key:value.mean() for key,value in parts.items()}
                else:
                    parts = compute_blueprint_loss({k:v.float() for k,v in prediction.items()}, targets,
                        load_curve_weight=args.load_curve_loss_weight)
                    loss = parts['loss'] * weight * args.blueprint_loss_weight
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite blueprint loss')
                if planner_objective_enabled and update_samples:
                    engine.backward(loss, update_samples)
                    update = engine.step()
                    if update['applied']:
                        plan_steps += 1
                        planner_effective_steps += 1
                        if args.load_curve_loss_weight > 0:
                            increment_cap('load_curve', 1, 'load_curve_v1')
                    else:
                        emit('form_warning', message='AMP overflow: blueprint update skipped', epoch=epoch, chart=end_position)
                progress.timing['compute_sec'] += time.perf_counter()-tick
                progress.add(parts, chart_count)
                phase_loss_sums['plan_loss'] += loss.detach()
                phase_counts['plan_loss'] += chart_count
                progress.phase_samples += chart_count
                if args.checkpoint_interval_steps and steps+plan_steps-last_save_steps >= args.checkpoint_interval_steps:
                    save(epoch, 'blueprint', end_position)
                progress.log(end_position, plan_total, force=position==plan_start,
                             optimizer_steps=steps, blueprint_steps=plan_steps, blueprint_batch_size=plan_batch_size,
                         blueprint_workers=blueprint_workers,
                         blueprint_cache=blueprint_cache.stats() if blueprint_cache is not None else None,
                         lr=optimizer.param_groups[0]['lr'])
        except BaseException:
            if plan_stream is not None and stream_entered:
                plan_stream.__exit__(*sys.exc_info())
            raise
        else:
            if plan_stream is not None and stream_entered:
                plan_stream.__exit__(None, None, None)
        if plan_start < plan_total:
            if blueprint_cache is not None:
                emit('form_blueprint_cache', phase='blueprint_complete', **blueprint_cache.stats())
            save(epoch, 'layout_revision', 0, reason='blueprint_complete')
        progress.begin('layout_revision', layout_start)
        sampler = EpochBatches(len(dataset), args.batch_size, args.seed, epoch, start=layout_start, limit=args.max_steps)
        loader_options = dict(batch_sampler=sampler, collate_fn=collate_form, num_workers=args.loader_workers,
                              pin_memory=device.type=='cuda', generator=torch.Generator().manual_seed(args.seed+epoch))
        if args.loader_workers:
            loader_options['prefetch_factor'] = args.prefetch_factor
            loader_options['multiprocessing_context'] = 'spawn'
        loader = DataLoader(dataset, **loader_options)
        layout_prefetch_depth = ((args.layout_prefetch_batches if args.layout_prefetch_batches is not None
                                  else min(args.prefetch_factor, 16)) if not args.loader_workers else 0)
        if layout_prefetch_depth:
            from maichart.form.layout_prefetch import LayoutBatchPrefetch
            batch_stream = LayoutBatchPrefetch(loader, depth=layout_prefetch_depth)
        else:
            batch_stream = nullcontext(iter(loader))
        pending_tasks = Counter()
        pending_cap_tasks = Counter()
        pending_rhythm_count = None
        pending_position_event_count = None
        pending_alignment_count = None
        # A per-batch finiteness test would make the host wait for each forward
        # pass before queueing its backward or preparing the next batch. Keep
        # the flag on the device and test it once before the optimizer update.
        pending_finite = None
        last_update = {}
        with batch_stream as iterator:
            for batch_index in range(layout_start, layout_total):
                tick = time.perf_counter()
                batch = next(iterator)
                validate_cpu_batch(batch)
                batch_size = len(batch['task'])
                tokens = int((batch['target'] != 0).sum())
                task_counts = Counter(batch['task'].tolist())
                progress.timing['data_wait_sec'] += time.perf_counter()-tick
                tick = time.perf_counter()
                batch = {k:v.to(device, non_blocking=device.type=='cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                progress.timing['h2d_sec'] += time.perf_counter()-tick
                tick = time.perf_counter()
                if args.condition_drop_prob:
                    batch['condition_null'] = torch.rand(batch_size, device=device) < args.condition_drop_prob
                with engine.autocast():
                    prediction = model(batch)
                loss_sum, parts = layout_loss(prediction, batch, args.relation_loss_weight,
                    event_decision_weight=args.event_decision_loss_weight,
                    rhythm_loss_weight=args.rhythm_loss_weight,
                    repetition_penalty_weight=args.repetition_penalty_weight,
                    diversity_weight=args.diversity_loss_weight,
                    position_event_weight=args.position_event_loss_weight,
                    position_alignment_weight=args.position_alignment_loss_weight)
                diagnostics = parts.pop('diagnostics')
                finite = torch.isfinite(loss_sum.detach())
                pending_finite = finite if pending_finite is None else pending_finite & finite
                engine.backward(loss_sum, batch_size)
                pending_tasks.update(task_counts)
                for task_id in (0, 1):
                    mask = (batch['task'] == task_id) & ~batch['condition_null'] & (batch['weight'] > 0)
                    pending_cap_tasks[task_id] += mask.sum().detach()
                if args.rhythm_geometry_profile != 'none' and args.rhythm_loss_weight > 0:
                    timing_mask = batch.get('timing_mask')
                    if timing_mask is not None:
                        # Keep this on-device through the accumulation group. A
                        # row contributes only when it has a valid timing label
                        # and positive chart/region weight; null conditioning is
                        # still legitimate auxiliary supervision.
                        valid_rhythm = timing_mask.any(dim=1) & (batch['weight'] > 0)
                        count = valid_rhythm.sum()
                        pending_rhythm_count = count if pending_rhythm_count is None else pending_rhythm_count + count
                if args.position_event_loss_weight > 0:
                    # Count supervised events, not tokens: that is exactly the unit
                    # the per-event objective averages over.
                    boundary = ((batch['decoder_input'] == _CODEC_FOR_LOSS.bos_id)
                                | (batch['decoder_input'] == _CODEC_FOR_LOSS.end_id))
                    event_count = (boundary & (batch['target'] == _CODEC_FOR_LOSS.at_id)
                                   & (batch['weight'] > 0).unsqueeze(1)).sum()
                    pending_position_event_count = (event_count if pending_position_event_count is None
                                                    else pending_position_event_count + event_count)
                if args.position_alignment_loss_weight > 0:
                    aligned = (batch['alignment_event_mask']
                               & ((batch['weight'] > 0) & (batch['task'] == 0)).unsqueeze(1)).sum()
                    pending_alignment_count = (aligned if pending_alignment_count is None
                                               else pending_alignment_count + aligned)
                update = None
                if engine.microbatches >= args.grad_accumulation or batch_index+1 == layout_total:
                    if not bool(pending_finite):
                        raise ValueError('Nonfinite layout loss')
                    pending_finite = None
                    update = engine.step()
                    last_update = update
                    if update['applied']:
                        steps += 1
                        for task_id, name in ((0, 'draft'), (1, 'revision')):
                            task_training[name] = task_training.get(name, 0) + pending_tasks[task_id]
                            conditioned_count = int(pending_cap_tasks[task_id])
                            if conditioned_count:
                                increment_cap(name, conditioned_count, 'form_condition_v2')
                        if pending_rhythm_count is not None and bool(pending_rhythm_count.item()):
                            increment_cap('rhythm_geometry', int(pending_rhythm_count.item()), args.rhythm_geometry_profile)
                        if pending_position_event_count is not None and bool(pending_position_event_count.item()):
                            increment_cap('position_event', int(pending_position_event_count.item()), 'per_event_fraction_v1')
                        if pending_alignment_count is not None and bool(pending_alignment_count.item()):
                            increment_cap('position_alignment', int(pending_alignment_count.item()), 'authored_audio_queries_v1')
                    else:
                        emit('form_warning', message='AMP overflow: accumulated layout update skipped', epoch=epoch, batch=batch_index+1)
                    pending_tasks.clear(); pending_cap_tasks.clear(); pending_rhythm_count = None
                    pending_position_event_count = None
                    pending_alignment_count = None
                progress.timing['compute_sec'] += time.perf_counter()-tick
                progress.phase_samples += batch_size
                progress.phase_tokens += tokens
                progress.add(parts, batch_size, weights={'token_accuracy': tokens})
                progress.add(diagnostics['values'], weights=diagnostics['counts'])
                for key, value in diagnostics['values'].items():
                    count = diagnostics['counts'][key]
                    layout_metric_sums[key] = layout_metric_sums.get(key, 0.) + value.detach() * count
                    layout_metric_counts[key] = layout_metric_counts.get(key, 0) + count
                phase_loss_sums['layout_loss'] += parts['loss']*batch_size
                phase_counts['layout_loss'] += batch_size
                if update and args.checkpoint_interval_steps and steps+plan_steps-last_save_steps >= args.checkpoint_interval_steps:
                    save(epoch, 'layout_revision', batch_index+1)
                progress.log(batch_index+1, layout_total, optimizer_steps=steps, blueprint_steps=plan_steps,
                             accumulation_pending=engine.microbatches, grad_accumulation=args.grad_accumulation,
                             lr=optimizer.param_groups[0]['lr'], cache=dataset.cache_stats(),
                             cache_scope='main_process_only' if args.loader_workers else 'training_process',
                             layout_prefetch=iterator.stats() if layout_prefetch_depth else {'mode': 'synchronous'},
                             grad_norm=last_update.get('grad_norm'), amp_scale=last_update.get('amp_scale'))
        del iterator, loader
        metrics = {'epoch':epoch,
                   'draft_revision_loss':float(phase_loss_sums['layout_loss'])/max(1,phase_counts['layout_loss']),
                   'whole_song_plan_loss':float(phase_loss_sums['plan_loss'])/max(1,phase_counts['plan_loss']),
                   'optimizer_steps':steps, 'blueprint_steps':plan_steps, 'task_training':dict(task_training),
                   'elapsed_sec':round(time.perf_counter()-progress.started,2),
                   'partial_epoch_metrics':bool(plan_start or layout_start)}
        metrics['metric_counts'] = {}
        for key, value in layout_metric_sums.items():
            count = layout_metric_counts[key]
            metrics[key] = float((value / count.clamp_min(1)).detach().cpu())
            metrics['metric_counts'][key] = int(count.detach().cpu())
        history.append(metrics)
        save(epoch, 'layout_revision', layout_total, complete=True, reason='epoch_complete')
        (output/'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        emit('form_epoch', **metrics)
        cursor = {}
    return 0


def _atoms(text):
    from maichart.notes import _split_top_level_slashes
    return _split_top_level_slashes(text)


if __name__ == '__main__':
    raise SystemExit(main())
