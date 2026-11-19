"""Composer v2 learning rate computed from its persisted update-attempt index."""
import math

from .defaults import LR_SCHEDULES


def validate(schedule, warmup_steps, decay_end_step):
    if schedule not in LR_SCHEDULES:
        raise ValueError("Unknown lr_schedule")
    if decay_end_step < 0:
        raise ValueError("衰减结束步数须为非负整数")
    if decay_end_step and (schedule == 'constant' or decay_end_step <= warmup_steps):
        raise ValueError("衰减结束步数须大于预热更新数，恒定学习率请填 0")


def factor(attempt, schedule, warmup_steps, total_updates, min_lr_ratio, decay_end_step=0):
    if attempt < warmup_steps:
        return (attempt + 1) / warmup_steps
    if schedule == 'constant':
        return 1.

    # Use the original cosine endpoint for existing checkpoint contracts.
    inclusive_end = bool(decay_end_step or schedule != 'cosine')
    end = (decay_end_step or total_updates) - inclusive_end
    if inclusive_end and attempt >= end:
        return min_lr_ratio
    progress = min(1., (attempt - warmup_steps) / max(1, end - warmup_steps))
    if schedule == 'linear':
        remaining = 1. - progress
    elif schedule == 'quadratic':
        remaining = (1. - progress) ** 2
    elif schedule == 'exponential':
        remaining = math.expm1(3. * (1. - progress)) / math.expm1(3.)
    else:
        return min_lr_ratio + (1 - min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * remaining
