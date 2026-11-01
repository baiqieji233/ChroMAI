"""Serializable generation settings shared by training validation and the CLI."""
import math

# File/audio/condition/tempo inputs come from each validation song, not a
# global generation request. These are the independently editable controls.
CONTROLS = {
    'search_policy': (str, 'phrase_v1', ('legacy', 'phrase_v1')),
    'physical_policy': (str, 'runtime_hard_v2', ('legacy', 'runtime_hard_v2')),
    'phrase_candidates': (int, 1, (1, 2, 4)),
    'phrase_write_beats': (float, 8., None),
    'phrase_commit_beats': (float, 4., None),
    'phrase_step_budget': (int, 256, None),
    'phrase_repair_budget': (int, 8, None),
    'phrase_rollback_budget': (int, 4, None),
    'phrase_time_budget_sec': (float, 30., None),
    'softcomfort_strength': (float, 1., None),
    'audio_tracking_strength': (float, 0., None),
    'pointer_temperature': (float, None, None),
    'action_temperature': (float, None, None),
    'min_same_head_ms': (float, 50., None),
    'touch_layout_policy': (str, 'open', ('open', 'legacy_two_heads')),
    'max_events': (int, 0, None),
}


def add_arguments(parser):
    for name, (kind, default, choices) in CONTROLS.items():
        options = dict(type=kind, default=default)
        if choices is not None: options['choices'] = choices
        parser.add_argument('--generation-eval-'+name.replace('_', '-'), **options)


def from_training_args(args):
    values = {name: getattr(args, 'generation_eval_'+name, default)
              for name, (_, default, _) in CONTROLS.items()}
    return validate(values)


def validate(settings):
    unknown = set(settings)-set(CONTROLS)
    if unknown: raise ValueError(f'Unknown generation evaluation controls: {sorted(unknown)}')
    values = {name: settings.get(name, default) for name, (_, default, _) in CONTROLS.items()}
    for name, (kind, _, choices) in CONTROLS.items():
        value = values[name]
        if value is None and name in ('pointer_temperature', 'action_temperature'): continue
        if choices is not None:
            if value not in choices: raise ValueError('Invalid generation_eval_'+name)
        elif not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError('Invalid generation_eval_'+name)
        if kind is int and int(value) != value: raise ValueError('Invalid generation_eval_'+name)
    if not 0 < values['phrase_commit_beats'] <= values['phrase_write_beats']:
        raise ValueError('Validation phrase commit span must be positive and <= write span')
    if values['phrase_step_budget'] < 1 or values['phrase_time_budget_sec'] <= 0:
        raise ValueError('Validation generation needs positive step/time budgets')
    if values['audio_tracking_strength'] > 1 or values['softcomfort_strength'] > 2:
        raise ValueError('Validation tracking/comfort strength exceeds its supported range')
    if values['min_same_head_ms'] > 200:
        raise ValueError('Validation same-head interval must be between 0 and 200 ms')
    return values
