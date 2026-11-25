"""Disk-geometry experiment and its backward-compatible checkpoint contract."""
from .defaults import MOTION_WRITING_POLICIES


SLIDES = 'slide_geometry_v1'
DISK = 'disk_geometry_v1'
CHOICES = ('auto', 'off', SLIDES, DISK)
GEOMETRY_LOSS_WEIGHT = .025


def resolve(requested, *, saved, resume, sequence, writing):
    previous = (saved or {}).get('spatial_policy', 'off')
    if requested not in CHOICES or previous not in CHOICES[1:]:
        raise ValueError('未知的盘面几何学习设置')
    policy = previous if requested == 'auto' else requested
    if resume and policy != previous:
        raise ValueError('断点续训须继承盘面几何设置，更改请在新目录微调')
    if policy != 'off' and (sequence != 'continuous_events_v1' or writing not in MOTION_WRITING_POLICIES):
        raise ValueError('盘面几何学习需要连续模型及 v3、空拍与重音或 v4 局部写法')
    return policy


def migrate_parameters(model, state, previous, requested):
    """Allow only tensor families specifically enabled/disabled by this change."""
    geometry = ('local_writing.geometry_predictor.', 'local_writing.geometry_projection.')
    changes = ((geometry, previous != 'off', requested != 'off'),
               (('spatial_history.',), previous == DISK, requested == DISK))
    missing = set()
    for prefixes, old, new in changes:
        if old and not new:
            state = {k: v for k, v in state.items() if not k.startswith(prefixes)}
        elif new and not old:
            missing.update(k for k in model.state_dict() if k.startswith(prefixes))
    return state, missing
