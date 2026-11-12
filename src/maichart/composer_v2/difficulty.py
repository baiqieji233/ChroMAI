"""Difficulty selection for Composer v2 training and validation."""
from dataclasses import dataclass
import math


DEFAULT_MIN_LEVEL = 12.
DECLARED_MAX_LEVEL = 15.
FITTED_MAX_LEVEL = 16.
CONDITION_MAX_LEVEL = 20.
FITTED_SOURCES = ('chartgate', 'fitted')
SOURCES = ('auto', 'declared', *FITTED_SOURCES)


@dataclass(frozen=True)
class DifficultySelection:
    source: str
    min_level: float
    max_level: float

    def contract(self, scorer=None):
        if self == DEFAULT_SELECTION:
            return None
        result = dict(source=self.source, min_level=self.min_level, max_level=self.max_level)
        if self.source in FITTED_SOURCES:
            result['scorer'] = scorer.identity
        return result


DEFAULT_SELECTION = DifficultySelection('declared', DEFAULT_MIN_LEVEL, DECLARED_MAX_LEVEL)


def resolve_selection(source='auto', min_level=None, max_level=None, *, parent=None, resume=False):
    saved = (parent or {}).get('difficulty_selection') or {}
    if source not in SOURCES:
        raise ValueError('未知定数来源')
    if source == 'auto':
        source = saved.get('source', 'declared') if resume else 'declared'
    fitted = source in FITTED_SOURCES
    ceiling = CONDITION_MAX_LEVEL if fitted else DECLARED_MAX_LEVEL
    defaults = (DEFAULT_MIN_LEVEL, FITTED_MAX_LEVEL if fitted else DECLARED_MAX_LEVEL)
    bounds = []
    for name, raw, default in zip(('min_level', 'max_level'), (min_level, max_level), defaults):
        if raw is None:
            raw = saved.get(name, default) if resume else default
        if isinstance(raw, bool):
            raise ValueError('定数范围必须是数字')
        try:
            bounds.append(float(raw))
        except (TypeError, ValueError) as error:
            raise ValueError('定数范围必须是数字') from error
    low, high = bounds
    if not all(map(math.isfinite, bounds)) or not 1. <= low <= high <= ceiling:
        raise ValueError(f'定数范围须满足 1 ≤ 最低定数 ≤ 最高定数 ≤ {ceiling:g}')
    selection = DifficultySelection(source, low, high)
    if resume and selection != DifficultySelection(saved.get('source', 'declared'),
            saved.get('min_level', DEFAULT_MIN_LEVEL), saved.get('max_level', DECLARED_MAX_LEVEL)):
        raise ValueError('断点续训需要沿用检查点的定数来源和范围。切换定数来源可用新目录微调。')
    return selection


def scorer_path_key(source):
    return 'chartgate_dir' if source == 'chartgate' else 'open_mdp_dir'


def locate_scorer(source, cache_dir, directory=None, saved_directory=None):
    from pathlib import Path
    if not directory and saved_directory and Path(saved_directory).is_dir():
        directory = saved_directory
    if source == 'chartgate':
        from .chartgate import locate_checkout
        return locate_checkout(cache_dir, directory)
    from .open_mdp import locate_checkout
    return locate_checkout(directory)


def make_scorer(source, directory, cache_dir, *, read_only=False):
    if source == 'chartgate':
        from .chartgate import ChartGateScorer
        return ChartGateScorer(directory, cache_dir, read_only=read_only)
    from .open_mdp import OpenMdpScorer
    return OpenMdpScorer(directory, cache_dir, read_only=read_only)
