"""Audio-conditioned chart generation utilities."""
from importlib import import_module

__all__ = [
    "AUDIO_FEATURE_KEYS",
    "CONDITION_KEYS",
    "RADAR_KEYS",
    "STYLE_KEYS",
    "ConditionedChartTransformer",
    "ConditionedChartTransformerConfig",
    "GenerationCondition",
    "condition_from_targets",
    "derive_radar_from_targets",
    "derive_style_from_targets",
]

_MODEL_EXPORTS = {"AUDIO_FEATURE_KEYS", "ConditionedChartTransformer", "ConditionedChartTransformerConfig"}


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Chart parsing and physical analysis share this package with the PyTorch model.
    module = import_module(".model" if name in _MODEL_EXPORTS else ".conditions", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
