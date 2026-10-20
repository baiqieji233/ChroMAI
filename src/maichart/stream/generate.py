"""Generate a continuous exact-Fraction chart with chart_stream_v1."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from maichart.event_timing import EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio, planner_audio_queries, beat_at_second
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.event_compile import compile_event_points
from maichart.generation.timeline import normalize_tempo_map
from .checkpoint import load
from .decoding import compose_stream, validate_options
from .runtime import resolve_device
from .timing import export_end_timing, timing_diagnostics
from .reference_guard import summarize_reference
from maichart.generation.miacode_python import analyze_maidata
from .audio_bounds import generation_bounds


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--bpm', type=float)
    p.add_argument('--first', type=float, default=0.)
    p.add_argument('--level', type=float, default=13.)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--position-temperature', type=float, default=.15)
    p.add_argument('--position-beam-size', type=int, default=4)
    p.add_argument('--position-complexity', type=float, default=.35)
    p.add_argument('--position-complexity-floor',type=int,default=8)
    p.add_argument('--position-interval-strength', type=float, default=1.)
    p.add_argument('--unplanned-gap-penalty-strength', type=float, default=.5)
    p.add_argument('--rhythm-diversity-strength', type=float, default=.6)
    p.add_argument('--early-eos-penalty', type=float, default=0.)
    p.add_argument('--learned-ending-strength',type=float,default=1.)
    p.add_argument('--constraint-min-probability',type=float,default=.05)
    p.add_argument('--event-reconsider-attempts',type=int,default=4)
    p.add_argument('--max-events', type=int, default=0)
    p.add_argument('--max-denominator', type=int, default=384)
    p.add_argument('--max-interval-denominator', type=int, default=96)
    p.add_argument('--min-spacing-ms', type=float, default=20.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--condition-file')
    p.add_argument('--tempo-map')
    p.add_argument('--meter-map')
    p.add_argument('--style-balance-strength', type=float, default=0.)
    p.add_argument('--configuration-candidates', type=int, default=4)
    p.add_argument('--local-chord-strength', type=float, default=0.)
    p.add_argument('--physical-feasibility-strength', type=float, default=0.)
    p.add_argument('--section-type-strength', type=float, default=1.)
    p.add_argument('--plan-intent-strength', type=float, default=1.)
    p.add_argument('--max-button-heads', type=int, default=2)
    p.add_argument('--hold-protection', choices=('on', 'off'), default='on')
    p.add_argument('--reference-check-attempts', type=int, default=8)
    p.add_argument('--generation-end-sec',type=float)
    p.add_argument('--trim-exact-silence',choices=('on','off'),default='on')
    return p


def generate(args):
    validate_options(args)
    device = resolve_device(args.device)
    model, payload = load(args.checkpoint, device)
    model.eval()
    atoms = payload.get('atom_catalog') or [str(i) for i in range(1, 9)]
    # Optimizer tensors stay on CPU when loading, then can be released entirely
    # during inference. Only immutable model metadata is needed in the report.
    model_config = asdict(model.config)
    checkpoint_training = {key: payload.get(key) for key in ('epoch', 'optimizer_steps', 'training_position')}
    checkpoint_training['training_semantics'] = (payload.get('args') or {}).get('training_semantics', 'legacy')
    del payload
    audio = extract_rational_audio(args.audio, cache_dir=Path(args.checkpoint).parent/'audio_cache',
                                   evidence_profile='balanced_evidence_v1')
    audio.feature_policy = 'native_measured_v2'
    tempo_file = getattr(args, 'tempo_map', None)
    meter_file = getattr(args, 'meter_map', None)
    tempo = (normalize_tempo_map(json.loads(Path(tempo_file).read_text(encoding='utf-8-sig')))
             if tempo_file else [EventTempo(Fraction(0), args.bpm or audio.tempo_bpm)])
    meters = json.loads(Path(meter_file).read_text(encoding='utf-8-sig')) if meter_file else None
    song = organize_song(audio, tempo, args.first, meters=meters, max_seconds=999.,
                         quality_profile='musical_evidence_v1')
    if not song.spans:
        raise ValueError('当前 first/BPM 设置下没有可生成的音频范围')
    values, times, _ = planner_audio_queries(song)
    bounds = generation_bounds(audio,requested_end=args.generation_end_sec,
                               trim_exact_silence=args.trim_exact_silence == 'on')
    args.effective_end_sec = bounds['effective_end_sec']
    condition = GenerationCondition(level=args.level, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    if args.condition_file:
        condition = GenerationCondition.from_dict(json.loads(Path(args.condition_file).read_text(encoding='utf-8-sig')))
    print('stream_runtime '+json.dumps({'device': str(device), 'seed': args.seed,
        'max_tokens': model.config.max_tokens, 'max_history_tokens': model.config.max_history_tokens}), flush=True)
    points, status = compose_stream(model, RationalCodec(), song, values, times, condition, atoms, args)
    audio_end = beat_at_second(bounds['effective_end_sec'], song.tempo, song.first)
    export_end, export_timing = export_end_timing(audio_end, song.tempo, song.first)
    text, _ = compile_event_points(points, tempo_changes=song.tempo, bpm=song.tempo[0].bpm,
        first=args.first, end_beat=export_end, level=str(condition.level))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = summarize_reference(analyze_maidata(text))
    review_needed = bool(reference['runtime_hard_count'] or not reference['parse_ok'])
    report = {'schema': 'maichart-chart-stream-generation-v1', 'family': 'chart_stream_v1',
        'event_count': len(points), 'audio_duration': audio.duration_sec,
        'audio_bounds':bounds,
        'audio_end_beat': str(audio_end), 'stream_continuity': True,
        'checkpoint': str(Path(args.checkpoint).resolve()), 'model_config': model_config,
        'checkpoint_training': checkpoint_training,
        'device': str(device), 'seed': args.seed, 'status': status,
        'audio_evidence_profile': audio.evidence_profile, 'audio_feature_policy': audio.feature_policy,
        'generation_status': ('incomplete' if status['incomplete'] else 'needs_review' if review_needed else 'completed'),
        'reference_playability':reference,
        'first': args.first, 'tempo_map': [{'beat': str(t.beat), 'bpm': t.bpm} for t in song.tempo],
        'export_timing': export_timing,
        'timing_diagnostics': timing_diagnostics(points, song.tempo, song.first, bounds['effective_end_sec'], text),
        'generation_controls': {'style_balance_strength': args.style_balance_strength,
            'position_complexity_floor':args.position_complexity_floor,
            'learned_ending_strength':args.learned_ending_strength,
            'constraint_min_probability':args.constraint_min_probability,
            'event_reconsider_attempts':args.event_reconsider_attempts,
            'position_beam_size': args.position_beam_size, 'position_complexity': args.position_complexity,
            'position_interval_strength': args.position_interval_strength,
            'unplanned_gap_penalty_strength': args.unplanned_gap_penalty_strength,
            'rhythm_diversity_strength': float(getattr(args, 'rhythm_diversity_strength', .6)),
            'position_temperature': args.position_temperature, 'early_eos_penalty': args.early_eos_penalty,
            'max_denominator': args.max_denominator, 'max_interval_denominator': args.max_interval_denominator,
            'min_spacing_ms': args.min_spacing_ms,
            'interval_consistency_policy': 'censored_log_ratio_v1',
            'max_button_heads': args.max_button_heads, 'hold_protection': args.hold_protection,
            'reference_check_attempts':args.reference_check_attempts,
            'local_chord_strength_requested': args.local_chord_strength,
            'local_chord_strength_effective': args.local_chord_strength,
            'physical_feasibility_strength': args.physical_feasibility_strength,
            'section_type_strength': float(getattr(args, 'section_type_strength', 1.)),
            'plan_intent_strength': float(getattr(args, 'plan_intent_strength', 1.))},
        'limitations': ['Model quality requires trained chart_stream weights.',
                        'Soft physical guidance does not certify playability.']}
    output.write_text(text, encoding='utf-8')
    output.with_suffix('.stream.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    print(json.dumps(generate(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
