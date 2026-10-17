"""Release oversized idle CUDA caches at complete optimizer boundaries.

This guard changes neither the allocator's hard limit nor live tensors.  A
large cached allocation can still count against the Windows GPU residency
budget after PyTorch has released its tensors, so the driver-facing footprint
and ``memory_allocated`` do not necessarily tell the same story.
"""
from __future__ import annotations

import math

import torch


_GIB = 1024 ** 3


class CudaCacheGuard:
    """Bound cache trimming without changing training or allocator semantics.

    Call :meth:`check` only after an optimizer update has completed and its
    gradients have been released.  The caller owns this boundary; this class
    does not inspect model tensors, synchronize them to the CPU, modify RNG,
    or change the requested CUDA memory limit.  ``empty_cache`` itself may
    wait for outstanding allocator work.

    Trimming requires both a large reserved footprint and substantial idle
    space.  A cooldown bounds repeated trim/reallocate cycles for variable
    length batches.  CPU construction/checking never accesses CUDA APIs.
    """

    def __init__(self, device, budget_gib, *, cooldown_updates=8):
        if isinstance(budget_gib, bool) or not math.isfinite(budget_gib) or budget_gib < 0:
            raise ValueError('CUDA cache budget must be finite and nonnegative')
        if (isinstance(cooldown_updates, bool) or not isinstance(cooldown_updates, int)
                or cooldown_updates < 1):
            raise ValueError('CUDA cache cooldown must be a positive update count')
        self.device = torch.device(device)
        self.cooldown_updates = cooldown_updates
        self.checks = 0
        self.trim_count = 0
        self._last_trim_check = -cooldown_updates
        self._device_index = None
        self.total_bytes = 0
        self.high_water_bytes = 0
        if self.device.type == 'cuda':
            self._device_index = (self.device.index if self.device.index is not None
                                  else torch.cuda.current_device())
            self.total_bytes = int(torch.cuda.get_device_properties(self._device_index).total_memory)
            if self.total_bytes <= 0:
                raise ValueError('CUDA device must have positive total memory')
            # A zero budget means no explicit user hard cap. It does not
            # disable driver headroom; the 70% device threshold still applies.
            budget_threshold = budget_gib * _GIB * .90 if budget_gib else self.total_bytes
            self.high_water_bytes = int(min(self.total_bytes * .70, budget_threshold))

    def policy(self):
        return {'enabled': self._device_index is not None,
                'reserved_high_water_gib': round(self.high_water_bytes / _GIB, 3),
                'minimum_inactive_gib': .5,
                'minimum_inactive_fraction': .2,
                'cooldown_updates': self.cooldown_updates}

    def _snapshot(self):
        allocated = int(torch.cuda.memory_allocated(self._device_index))
        reserved = int(torch.cuda.memory_reserved(self._device_index))
        return {'allocated_gib': round(allocated / _GIB, 3),
                'reserved_gib': round(reserved / _GIB, 3),
                'inactive_gib': round(max(0, reserved - allocated) / _GIB, 3)}, allocated, reserved

    def check(self):
        """Trim idle blocks if necessary; return one event, otherwise ``None``."""
        if self._device_index is None:
            return None
        self.checks += 1
        if self.checks - self._last_trim_check < self.cooldown_updates:
            return None
        before, allocated, reserved = self._snapshot()
        inactive = max(0, reserved - allocated)
        if (reserved < self.high_water_bytes
                or inactive < max(.5 * _GIB, reserved * .20)):
            return None
        # empty_cache operates on the current device. Resolve a bare `cuda`
        # once at construction, and preserve the caller's original device.
        with torch.cuda.device(self._device_index):
            torch.cuda.empty_cache()
        after, _, after_reserved = self._snapshot()
        self.trim_count += 1
        self._last_trim_check = self.checks
        return {'reason': 'oversized_inactive_cuda_cache',
                'before': before, 'after': after,
                'released_gib': round(max(0, reserved - after_reserved) / _GIB, 3),
                'reserved_high_water_gib': round(self.high_water_bytes / _GIB, 3),
                'trim_count': self.trim_count}


__all__ = ['CudaCacheGuard']
