"""Compatibility import for the Blueprint CPU prefetch API.

The implementation lives in :mod:`maichart.form.blueprint_prefetch`; this
module keeps the first integration name stable for callers already using it.
"""

from .blueprint_prefetch import (  # noqa: F401
    BlueprintPrefetchStats,
    OrderedBlueprintPrefetch,
    iter_prefetched_blueprints,
    prefetch_blueprints,
)

__all__ = [
    "BlueprintPrefetchStats",
    "OrderedBlueprintPrefetch",
    "iter_prefetched_blueprints",
    "prefetch_blueprints",
]
