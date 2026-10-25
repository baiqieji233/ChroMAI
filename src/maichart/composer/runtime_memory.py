"""Host memory headroom for Windows CUDA allocations and CPU audio caches."""
from __future__ import annotations

import ctypes
import sys

GIB = 1024 ** 3


def host_memory():
    if sys.platform != 'win32':
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32)] + [
            (name, ctypes.c_uint64) for name in (
                'physical', 'free_physical', 'commit_limit', 'free_commit',
                'virtual', 'free_virtual', 'extended')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {key: int(getattr(status, key)) for key in (
        'physical', 'free_physical', 'commit_limit', 'free_commit')}


def cache_capacity(requested, resident, memory):
    """Return an absolute cache budget with explicit host-memory reserves."""
    if memory is None:
        return requested
    # ``resident + growth`` allowed a cache that had already reached its
    # requested size to remain there while free physical memory collapsed.
    # Keep at least 2 GiB of physical and 4 GiB of commit headroom instead.
    physical_budget = max(0, memory['free_physical'] - 2 * GIB)
    commit_budget = max(0, memory['free_commit'] - 4 * GIB)
    return min(requested, physical_budget, commit_budget)


def memory_gib(memory):
    return None if memory is None else {key + '_gib': round(value / GIB, 3)
                                        for key, value in memory.items()}
