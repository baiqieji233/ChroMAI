"""Lightweight CLI/WebUI contract; importing it must not load the tensor runtime."""
import math

POLICY = 'phase_audio_v1'
CHOICES = ('auto', 'off', POLICY)


def resolve_policy(requested, parent=None, *, resume=False):
    previous = (parent or {}).get('onset_readout_policy', 'off')
    # Resume replays the original contract. Fresh and continuous finetune use
    # the latest readout; legacy window checkpoints have no compatible input.
    latest = POLICY if parent is None or parent.get('sequence_policy') == 'continuous_events_v1' else previous
    policy = (previous if resume else latest) if requested in (None, 'auto') else requested
    if policy not in ('off', POLICY):
        raise ValueError('Unknown onset readout policy')
    if resume and policy != previous:
        raise ValueError('Changing onset readout requires finetune in a new directory')
    return policy, parent is not None and policy != previous


def effective_strength(requested, policy):
    value = (1. if policy == POLICY else 0.) if requested is None else float(requested)
    if not math.isfinite(value) or not 0 <= value <= 2:
        raise ValueError('起音参考强度必须在 0 到 2 之间')
    if policy != POLICY and value != 0:
        raise ValueError('此检查点未训练独立声学读出；请先在新目录微调启用')
    return value
