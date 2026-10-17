"""Device resolution shared by stream loading, curve inference and generation."""
from __future__ import annotations

import torch


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
