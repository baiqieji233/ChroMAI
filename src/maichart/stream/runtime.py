"""Device resolution shared by stream loading, curve inference and generation."""
from __future__ import annotations

import torch
import os
from contextlib import contextmanager, nullcontext


@contextmanager
def training_cpu_budget(requested, device):
    """Bound nested CPU pools while independent songs prepare in threads.

    The scope owns the whole training run, not each worker: changing BLAS
    limits in worker contexts would race because these pools are global.
    """
    previous=torch.get_num_threads()
    count=int(requested) or min(2 if device.type=='cuda' else 4,os.cpu_count() or 1)
    torch.set_num_threads(count)
    try:
        from threadpoolctl import threadpool_limits
        context=threadpool_limits(limits=count,user_api='blas')
        controlled=True
    except ImportError:
        context=nullcontext();controlled=False
    try:
        with context:yield {'torch_threads':count,'blas_threads':count if controlled else None}
    finally:torch.set_num_threads(previous)


def resolve_device(requested='auto'):
    if requested is None or str(requested).strip().lower() == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError('生成设备必须为 auto、cpu 或 cuda[:编号]') from exc
    if device.type not in {'cpu', 'cuda'} or (device.type == 'cpu' and device.index is not None):
        raise ValueError('生成设备必须为 auto、cpu 或 cuda[:编号]')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA 不可用，请选择 CPU 或自动设备')
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f'CUDA 设备编号不可用：{index}')
        device = torch.device('cuda', index)
    return device
