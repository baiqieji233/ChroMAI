"""Difficulty selection for Composer v2 training and validation."""
from dataclasses import dataclass
import math


DEFAULT_MIN_LEVEL = 12.
DECLARED_MAX_LEVEL = 15.
FITTED_MAX_LEVEL = 16.
CONDITION_MAX_LEVEL = 20.
FITTED_SOURCES = ('chartgate', 'fitted')
SOURCES = ('auto', 'declared', *FITTED_SOURCES)
DIFFICULTY_SLOTS = (4, 5, 6)
DEFAULT_DIFFICULTY_SLOTS = (5, 6)
LEGACY_DIFFICULTY_SLOTS = (4, 5, 6)
CHARTGATE_SLOTS = (5, 6)


@dataclass(frozen=True)
class DifficultySelection:
    source: str
    min_level: float
    max_level: float
    min_declared_level: float = 0.
    difficulty_slots: tuple[int, ...] = DEFAULT_DIFFICULTY_SLOTS

    def dataset_options(self):
        return dict(min_level=self.min_level, max_level=self.max_level,
                    min_declared_level=self.min_declared_level, difficulty_slots=self.difficulty_slots)

    def split_exclusion(self, difficulty):
        """Apply metadata checks before assigning songs to automatic holdouts."""
        from maichart.training.event_dataset import numeric_level
        if not isinstance(difficulty, dict) or difficulty.get('has_chart') is False:
            return 'no_chart'
        try:
            slot = int(difficulty.get('difficulty_index', difficulty.get('index', 0)))
        except (TypeError, ValueError, OverflowError):
            return 'invalid_slot'
        declared = difficulty.get('level_raw', difficulty.get('level'))
        supported = CHARTGATE_SLOTS if self.source == 'chartgate' else None
        reason = metadata_exclusion(slot, declared, self.difficulty_slots, supported, self.min_declared_level)
        if reason is not None or self.source != 'declared':
            return reason
        try:
            level = numeric_level(declared)
        except ValueError:
            return 'invalid_declared_level'
        if not self.min_level <= level <= self.max_level:
            return 'declared_level_out_of_range'

    def contract(self, scorer=None):
        if self == LEGACY_SELECTION:
            return None
        result = dict(source=self.source, min_level=self.min_level, max_level=self.max_level)
        if self.source in FITTED_SOURCES:
            result['scorer'] = scorer.identity
        if self.min_declared_level:
            result['min_declared_level'] = self.min_declared_level
        if self.difficulty_slots != LEGACY_DIFFICULTY_SLOTS:
            result['difficulty_slots'] = list(self.difficulty_slots)
        return result


DEFAULT_SELECTION = DifficultySelection('declared', DEFAULT_MIN_LEVEL, DECLARED_MAX_LEVEL, 0., DEFAULT_DIFFICULTY_SLOTS)
LEGACY_SELECTION = DifficultySelection('declared', DEFAULT_MIN_LEVEL, DECLARED_MAX_LEVEL, 0., LEGACY_DIFFICULTY_SLOTS)


def metadata_exclusion(slot, declared, slots, supported_slots, min_declared_level):
    if slots is not None and slot not in slots:
        return 'difficulty_slot_excluded'
    if supported_slots is not None and slot not in supported_slots:
        return 'difficulty_slot_unsupported'
    if min_declared_level:
        from maichart.training.event_dataset import numeric_level
        try:
            level = numeric_level(declared)
        except ValueError:
            return 'invalid_declared_level'
        if not math.isfinite(level):
            return 'invalid_declared_level'
        if level < min_declared_level:
            return 'declared_level_below_minimum'
    return None


def _number(raw, label):
    if isinstance(raw, bool):
        raise ValueError(f'{label}必须是数字')
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{label}必须是数字') from error


def resolve_selection(source='auto', min_level=None, max_level=None, *,
                      min_declared_level=None, difficulty_slots=None, parent=None, resume=False):
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
        bounds.append(_number(raw, '定数范围'))
    low, high = bounds
    if not all(map(math.isfinite, bounds)) or not 1. <= low <= high <= ceiling:
        raise ValueError(f'定数范围须满足 1 ≤ 最低定数 ≤ 最高定数 ≤ {ceiling:g}')
    if min_declared_level is None:
        min_declared_level = saved.get('min_declared_level', 0.) if resume else float(fitted)
    declared_minimum = _number(min_declared_level, '最低标称定数')
    if not 0. <= declared_minimum <= CONDITION_MAX_LEVEL:
        raise ValueError('最低标称定数须在 0–20 之间，0 表示关闭标称校验')
    if declared_minimum and not fitted:
        raise ValueError('最低标称定数用于拟合定数筛选')
    if difficulty_slots is None:
        difficulty_slots = saved.get('difficulty_slots', LEGACY_DIFFICULTY_SLOTS) if resume else DEFAULT_DIFFICULTY_SLOTS
    if not isinstance(difficulty_slots, (list, tuple)) or not difficulty_slots or any(
        type(slot) is not int or slot not in DIFFICULTY_SLOTS for slot in difficulty_slots
    ):
        raise ValueError('请选择至少一个难度槽位（4 EXPERT、5 MASTER、6 Re:MASTER）')
    slots = tuple(sorted(set(difficulty_slots)))
    if source == 'chartgate' and not set(slots).intersection(CHARTGATE_SLOTS):
        raise ValueError('Chart;Gate 需要至少选择 MASTER 或 Re:MASTER')
    selection = DifficultySelection(source, low, high, declared_minimum, slots)
    if resume and selection != DifficultySelection(saved.get('source', 'declared'),
            saved.get('min_level', DEFAULT_MIN_LEVEL), saved.get('max_level', DECLARED_MAX_LEVEL),
            saved.get('min_declared_level', 0.), tuple(saved.get('difficulty_slots', LEGACY_DIFFICULTY_SLOTS))):
        raise ValueError('断点续训需要沿用检查点的难度筛选。修改筛选条件可用新目录微调。')
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
