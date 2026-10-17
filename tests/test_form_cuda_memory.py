from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from maichart.form.cuda_memory import CudaCacheGuard


_GIB = 1024 ** 3


class FakeCuda:
    def __init__(self, total=12, allocated=1, reserved=10):
        self.total = int(total * _GIB)
        self.allocated = int(allocated * _GIB)
        self.reserved = int(reserved * _GIB)
        self.calls = []
        self.device_index = None

    def current_device(self):
        self.calls.append(('current_device',))
        return 3

    def get_device_properties(self, device):
        self.calls.append(('properties', device))
        return SimpleNamespace(total_memory=self.total)

    def memory_allocated(self, device):
        self.calls.append(('allocated', device))
        return self.allocated

    def memory_reserved(self, device):
        self.calls.append(('reserved', device))
        return self.reserved

    @contextmanager
    def device(self, index):
        previous = self.device_index
        self.device_index = index
        self.calls.append(('enter_device', index))
        try:
            yield
        finally:
            self.device_index = previous
            self.calls.append(('leave_device', index))

    def empty_cache(self):
        self.calls.append(('empty_cache', self.device_index))
        self.reserved = self.allocated


@pytest.fixture
def fake_cuda(monkeypatch):
    fake = FakeCuda()
    for name in ('current_device', 'get_device_properties', 'memory_allocated',
                 'memory_reserved', 'device', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, getattr(fake, name))
    return fake


def test_cpu_is_a_cuda_free_noop(fake_cuda):
    guard = CudaCacheGuard('cpu', 8)
    assert guard.check() is None
    assert guard.policy()['enabled'] is False
    assert fake_cuda.calls == []


def test_guard_releases_only_idle_cache_without_altering_rng(fake_cuda):
    guard = CudaCacheGuard(torch.device('cuda'), 10.5)
    state = torch.get_rng_state().clone()
    event = guard.check()
    assert event == {
        'reason': 'oversized_inactive_cuda_cache',
        'before': {'allocated_gib': 1., 'reserved_gib': 10., 'inactive_gib': 9.},
        'after': {'allocated_gib': 1., 'reserved_gib': 1., 'inactive_gib': 0.},
        'released_gib': 9., 'reserved_high_water_gib': 8.4, 'trim_count': 1,
    }
    assert ('empty_cache', 3) in fake_cuda.calls
    assert fake_cuda.device_index is None
    assert fake_cuda.allocated == _GIB
    assert torch.equal(state, torch.get_rng_state())


def test_explicit_device_does_not_use_current_ordinal(fake_cuda):
    guard = CudaCacheGuard('cuda:1', 8)
    assert guard.policy()['reserved_high_water_gib'] == 7.2
    assert guard.check()['trim_count'] == 1
    assert ('current_device',) not in fake_cuda.calls
    assert ('properties', 1) in fake_cuda.calls
    assert ('empty_cache', 1) in fake_cuda.calls


@pytest.mark.parametrize('allocated,reserved', [(1, 7), (7.9, 8), (9, 10), (10, 10)])
def test_small_cache_or_mostly_live_memory_is_untouched(fake_cuda, allocated, reserved):
    fake_cuda.allocated, fake_cuda.reserved = int(allocated*_GIB), int(reserved*_GIB)
    guard = CudaCacheGuard('cuda:0', 10.5)
    assert guard.check() is None
    assert not any(name == 'empty_cache' for name, *_ in fake_cuda.calls)


def test_minimum_idle_half_gib_prevents_small_allocation_thrashing(fake_cuda):
    fake_cuda.allocated, fake_cuda.reserved = int(1.5*_GIB), int(1.9*_GIB)
    guard = CudaCacheGuard('cuda:0', 2.)
    assert guard.check() is None


def test_cooldown_bounds_repeated_allocator_flushes(fake_cuda):
    guard = CudaCacheGuard('cuda:0', 8, cooldown_updates=8)
    assert guard.check()['trim_count'] == 1
    fake_cuda.reserved = 10*_GIB
    calls_after_trim = len(fake_cuda.calls)
    for _ in range(7):
        assert guard.check() is None
    assert len(fake_cuda.calls) == calls_after_trim
    assert guard.check()['trim_count'] == 2


def test_zero_budget_retains_device_headroom(fake_cuda):
    guard = CudaCacheGuard('cuda:0', 0)
    assert guard.policy()['reserved_high_water_gib'] == 8.4
    assert guard.check()['released_gib'] == 9.


@pytest.mark.parametrize('budget', [-1, float('nan'), float('inf'), True])
def test_invalid_budget_is_rejected_before_cuda_calls(fake_cuda, budget):
    with pytest.raises(ValueError):
        CudaCacheGuard('cuda', budget)
    assert fake_cuda.calls == []


@pytest.mark.parametrize('cooldown', [0, -1, True, 1.5])
def test_invalid_cooldown_is_rejected(fake_cuda, cooldown):
    with pytest.raises(ValueError):
        CudaCacheGuard('cuda', 8, cooldown_updates=cooldown)
    assert fake_cuda.calls == []


def test_memory_snapshot_exposes_cache_and_whole_device_headroom(fake_cuda, monkeypatch):
    from maichart.form.training_runtime import memory_snapshot
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (2*_GIB, 12*_GIB))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 6*_GIB)
    assert memory_snapshot(torch.device('cuda')) == {
        'allocated_gib': 1., 'reserved_gib': 10., 'inactive_cached_gib': 9.,
        'device_free_gib': 2., 'device_total_gib': 12., 'peak_allocated_gib': 6.}


def test_webui_records_cache_events_without_replacing_training_phase():
    import json
    from maichart.webui.form_api import update_form_progress
    task = SimpleNamespace(kind='form_train', result={'phase':'layout_revision'})
    event = {'reason':'oversized_inactive_cuda_cache','released_gib':3.,'trim_count':2}
    update_form_progress(task, 'form_cuda_cache ' + json.dumps(event))
    assert task.result == {'phase':'layout_revision', 'cuda_cache_guard':event}
