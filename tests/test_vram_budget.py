"""The VRAM guard must bound the *peak* reservation, not the sampled value.

Observed failure: every batch shape gets its own cached block, so the reservation
grew monotonically.  Between steps the sampled value was 4.33 GiB while
``max_memory_reserved`` reached 17.21 GiB on a 12 GiB card; Windows backed the
excess with shared GPU memory over PCIe (measured 425 ms/micro-batch while
spilling versus 228 ms healthy, and the original 38-minute first epoch).
"""

import torch

from maichart.training import train_events


class _FakeProps:
    total_memory = 12 * 1024 ** 3


def _cuda_stubs(monkeypatch, *, peak: float, reserved: float, after_free: float):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: _FakeProps())
    state = {"peak": peak, "reserved": reserved, "freed": 0, "resets": 0}
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: state["peak"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: state["reserved"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1.25 * 1024 ** 3)

    def reset_peak_memory_stats(*_args, **_kwargs):
        state["resets"] += 1
        state["peak"] = state["reserved"]

    def empty_cache():
        state["freed"] += 1
        state["reserved"] = after_free

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    return state


def test_auto_budget_leaves_room_for_the_desktop(monkeypatch):
    _cuda_stubs(monkeypatch, peak=0, reserved=0, after_free=0)
    assert train_events.vram_budget_bytes(0.0, torch.device("cuda")) == int(12 * 1024 ** 3 * 0.6)
    assert train_events.vram_budget_bytes(5.0, torch.device("cuda")) == 5 * 1024 ** 3
    assert train_events.vram_budget_bytes(0.0, torch.device("cpu")) == 0


def test_peak_over_budget_releases_cached_blocks(monkeypatch):
    """The sampled value is below budget; only the peak reveals the growth."""

    state = _cuda_stubs(monkeypatch, peak=17.2 * 1024 ** 3, reserved=4.3 * 1024 ** 3,
                        after_free=1.1 * 1024 ** 3)
    freed = train_events.enforce_vram_budget(torch.device("cuda"), int(7.2 * 1024 ** 3),
                                             reset_peak=True)
    assert state["freed"] == 1 and state["resets"] == 1
    assert 3.0 < freed < 3.4, freed
    assert state["peak"] == 1.1 * 1024 ** 3, "the peak window must restart after a flush"


def test_under_budget_does_nothing(monkeypatch):
    state = _cuda_stubs(monkeypatch, peak=3.0 * 1024 ** 3, reserved=3.0 * 1024 ** 3, after_free=0)
    assert train_events.enforce_vram_budget(torch.device("cuda"), int(7.2 * 1024 ** 3)) == 0.0
    assert state["freed"] == 0
    assert train_events.enforce_vram_budget(torch.device("cpu"), int(7.2 * 1024 ** 3)) == 0.0
    assert train_events.enforce_vram_budget(torch.device("cuda"), 0) == 0.0


def test_memory_report_uses_gib(monkeypatch):
    _cuda_stubs(monkeypatch, peak=5 * 1024 ** 3, reserved=5 * 1024 ** 3, after_free=0)
    report = train_events.device_memory_report(torch.device("cuda"))
    assert report["allocated_gib"] == 1.25 and report["reserved_gib"] == 5.0
    assert train_events.device_memory_report(torch.device("cpu"))["peak_gib"] == 0.0


def test_hard_cap_uses_a_device_index(monkeypatch):
    """`set_per_process_memory_fraction` rejects a bare `cuda` device.

    Passing the unresolved device raised ValueError and the cap was silently
    skipped, so the run stayed uncapped and spilled into shared GPU memory.
    """

    calls = []
    _cuda_stubs(monkeypatch, peak=0, reserved=0, after_free=0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, index: calls.append((fraction, index)))

    applied = train_events.apply_vram_hard_cap(torch.device("cuda"), int(7.2 * 1024 ** 3))
    assert calls and calls[0][1] == 0, calls
    assert abs(calls[0][0] - 0.6) < 1e-6, calls
    assert applied == int(12 * 1024 ** 3 * 0.6), applied
    # An explicit index is forwarded untouched, and non-CUDA is a no-op.
    train_events.apply_vram_hard_cap(torch.device("cuda:1"), int(6 * 1024 ** 3))
    assert calls[-1][1] == 1
    assert train_events.apply_vram_hard_cap(torch.device("cpu"), int(6 * 1024 ** 3)) == 0


def test_hard_cap_failure_is_reported_not_fatal(monkeypatch):
    _cuda_stubs(monkeypatch, peak=0, reserved=0, after_free=0)

    def boom(*_args):
        raise RuntimeError("no cap for you")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", boom)
    assert train_events.apply_vram_hard_cap(torch.device("cuda"), int(6 * 1024 ** 3)) == 0


def test_trainer_flushes_periodically_and_reports_vram():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/maichart/training/train_events.py").read_text(encoding="utf-8")
    assert "cuda_memory epoch=" in source
    assert '"--cuda-memory-budget-gb"' in source and '"--cuda-cache-flush-batches"' in source
    assert "vram_flush_batches=args.cuda_cache_flush_batches" in source
    assert "batch_index % vram_flush_batches == 0" in source
    assert "reset_peak=True" in source
    assert "apply_vram_hard_cap(device, budget_bytes)" in source
    assert '"peak_gib"' in source


def test_flush_interval_defaults_are_sane():
    parser = train_events.build_parser()
    args = parser.parse_args(["--manifest", "m.json"])
    assert args.cuda_cache_flush_batches == 100
    assert args.cuda_memory_budget_gb == 0.0
    assert args.prefetch_batches == 2
    assert args.no_length_bucketing is False
