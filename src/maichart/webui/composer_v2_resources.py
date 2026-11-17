"""Pre-launch sizing for Composer v2, without moving a model to the GPU."""

import csv
from functools import lru_cache
import os
import shutil
import subprocess
import time

from maichart.composer_v2.training_memory import checkpointing_scope

MIB = 2**20
GIB = 2**30
SONG_FRAMES = 160


@lru_cache(maxsize=1)
def _nvidia_memory(_time_bucket):
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return ()
    try:
        result = subprocess.run(
            [executable, '--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.free,compute_cap',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, errors='replace', timeout=2, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 7:
            continue
        index, uuid, name, pci, total, free, capability = row
        try:
            total_bytes = int(total) * MIB
        except ValueError:
            continue
        free_bytes = int(free) * MIB if free.isdecimal() else None
        major = capability.split('.')[0]
        devices.append(dict(index=index, uuid=uuid, name=name, pci=pci,
                            total_bytes=total_bytes, free_bytes=free_bytes,
                            capability=int(major) if major.isdecimal() else None))
    return tuple(devices)


def cuda_memory():
    """Report free memory only when the CUDA target can be identified."""
    devices = _nvidia_memory(int(time.monotonic() // 5))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        first = visible.split(',')[0].strip()
        if first.startswith('GPU-'):
            matches = [device for device in devices if device['uuid'].startswith(first)]
            return matches[0] if len(matches) == 1 else None
        if not first.isdecimal():
            return None
        if os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID':
            ordered = sorted(devices, key=lambda device: device['pci'])
            return ordered[int(first)] if int(first) < len(ordered) else None
        return devices[0] if len(devices) == 1 and first == '0' else None
    if len(devices) == 1:
        return devices[0]
    if devices and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID':
        return min(devices, key=lambda device: device['pci'])
    return None


def _context_shape(config, long_batch):
    # Model two input lengths instead of presuming every batch reaches max_tokens.
    tokens = config.max_tokens if long_batch else max(1, config.max_tokens // 2)
    if config.sequence_policy == 'window_v1':
        frames = (config.history_beats + config.window_beats + config.future_beats) * config.ticks_per_beat
        return tokens, 1, max(1, (frames + 1) // 2)
    beats = config.history_beats + config.window_beats + 2
    if not long_batch:
        beats = max(1, (beats + 1) // 2)
    tiles = min(tokens, (beats + config.local_audio_block_beats - 1) // config.local_audio_block_beats)
    # Continuous-event tiles have 24 samples per beat and a stride-2 convolution.
    return tokens, tiles, (config.future_beats + 4) * 12


def _training_peak(config, vocab, parameters, trainable, batch, act_bytes, recompute,
                   flash_decoder, long_batch):
    tokens, tiles, local_frames = _context_shape(config, long_batch)
    width, ffn, heads = config.d_model, config.dim_feedforward, config.nhead
    sequence = batch * tokens
    local = batch * tiles * local_frames
    song = batch * SONG_FRAMES
    song_attentions = config.layers // 2

    # FP32 weights, gradients and two AdamW moment buffers. AdamW's CUDA foreach
    # update can also temporarily hold an extra parameter-sized buffer.
    parameters_bytes = 4 * parameters
    resident = parameters_bytes + 12 * trainable
    optimizer_peak = resident + 4 * trainable

    # Account for saved projections, residuals, normalized inputs, FFN and
    # dropout tensors. These factors approximate the live tape, not all FLOPs.
    encoder_layer = (8 + 7 * act_bytes) * width + (1 + 2 * act_bytes) * ffn
    decoder_layer = (16 + 10 * act_bytes) * width + (1 + 2 * act_bytes) * ffn
    audio_inputs = 4 * width * (3 * local + song)
    audio_saved = 0
    if trainable == parameters:  # music_reading freezes both encoders
        if recompute == 'all':
            audio_saved = 4 * width * (local * config.audio_layers + song * config.song_layers)
        else:
            audio_saved = encoder_layer * (local * config.audio_layers + song * config.song_layers)

    local_scores = sequence * heads * local_frames
    song_scores = sequence * heads * SONG_FRAMES
    self_mask = batch * tokens**2 if not flash_decoder else 0
    decoder_saved = 4 * sequence * width * config.layers
    if recompute == 'off':
        decoder_saved = (decoder_layer * sequence + act_bytes * (2 * local * width + local_scores)
                         + self_mask) * config.layers
        decoder_saved += act_bytes * (2 * song * width + song_scores) * song_attentions
    # Local and relative song attention may recompute an FP32 score matrix in
    # backward. Reserve scratch for one layer, not a dense matrix in every layer.
    bias_workspace = 8 * max(local_scores, song_scores if song_attentions else 0)
    self_workspace = (4 if long_batch else 1) * heads * self_mask
    working_layer = max(encoder_layer * local, decoder_layer * sequence)
    vocabulary_loss = sequence * vocab * (act_bytes + 8)
    autocast_weights = 2 * parameters if act_bytes == 2 else 0
    backward_peak = (resident + audio_inputs + audio_saved + decoder_saved + working_layer
                     + max(bias_workspace, self_workspace) + vocabulary_loss + autocast_weights)
    return max(optimizer_peak, backward_peak)


def _generation_peak(config, vocab, parameters, long_batch):
    tokens, tiles, local_frames = _context_shape(config, long_batch)
    width, layers = config.d_model, config.layers
    weights = 4 * parameters
    if config.sequence_policy != 'continuous_events_v1':
        tiles = 1
    # One process per simultaneous chart. Weights and cached projection weights
    # are shared by that process's decoder; successive proposals reuse buffers.
    projections = 4 * 3 * width * (width + 1) * layers
    kv = 2 * 4 * width * (layers * (tokens + tiles * local_frames) + (layers // 2) * SONG_FRAMES)
    caches = kv * (2 if long_batch else 1)  # primary and refreshed trial context can coexist
    attention = (4 if long_batch else 1) * config.nhead * tokens**2
    prefill = 4 * max(tokens * vocab, tokens * config.dim_feedforward, tiles * local_frames * width)
    return weights + projections + caches + attention + prefill


def estimate_memory(config, vocab, parameters, trainable, *, mode, requested_device,
                    requested_precision='auto', batch=1, activation_checkpointing='auto',
                    decoder_flash_attention='off', cuda_budget=0., audio_device='auto',
                    inference_backend='auto', count=1, parallel=1):
    import torch

    training = mode != 'generate'
    cuda_available = torch.cuda.is_available()
    xpu_available = training and hasattr(torch, 'xpu') and torch.xpu.is_available()
    device = requested_device
    if device == 'auto':
        device = 'cuda' if cuda_available else 'xpu' if xpu_available else 'cpu'
    audio_cuda = config.audio_backend == 'mert' and (
        audio_device == 'cuda' or audio_device == 'auto' and cuda_available and (training or device == 'cuda'))
    cuda = cuda_memory() if cuda_available and (device == 'cuda' or audio_cuda) else None
    total = cuda['total_bytes'] if cuda else None
    capability = cuda['capability'] if cuda else None
    scope = checkpointing_scope(activation_checkpointing, device, total) if training else 'off'
    precision = requested_precision if training else 'fp32'
    if precision == 'auto':
        precision = 'bf16' if device == 'xpu' or capability is not None and capability >= 8 else 'fp32'
    if training and requested_precision == 'auto' and device == 'cuda' and capability is None:
        # The job chooses BF16 where supported. Size both paths when discovery is unavailable.
        precision = 'auto'
    workers = 1 if training else min(count, parallel)
    gpu_decoder = device in ('cuda', 'xpu')
    flash = (decoder_flash_attention == 'on' and device == 'cuda' and precision != 'fp32'
             and (capability is None or capability >= 8)
             and config.d_model // config.nhead <= 256
             and torch.backends.cuda.is_flash_attention_available())
    peak = []
    for long_batch in (False, True):
        if not gpu_decoder:
            peak.append(0)
            continue
        if training:
            act_bytes = 4 if precision == 'fp32' or precision == 'auto' and long_batch else 2
            size = _training_peak(config, vocab, parameters, trainable, batch, act_bytes,
                                  scope, flash and act_bytes == 2, long_batch)
        else:
            size = _generation_peak(config, vocab, parameters, long_batch)
        # CUDA/runtime allocations and the caching allocator depend on the host.
        reserve = .08 if not long_batch else .15
        context = (192 if not long_batch else 320) * MIB
        peak.append(workers * round(size * (1 + reserve) + context))

    # MERT is a separate preparation phase. The serial multi-chart route has
    # already loaded Composer while preparing audio. Allow for that overlap.
    audio_peak = None
    if audio_cuda:
        audio_peak = [round(.65 * GIB), round(1.85 * GIB)]
        if mode == 'generate' and device == 'cuda' and count > 1 and workers == 1:
            audio_peak = [value + 4 * parameters for value in audio_peak]
    openvino_gpu = False
    if device == 'cpu' and (inference_backend in ('auto', 'openvino-gpu') and not training
                            or config.audio_backend == 'mert' and audio_device in ('auto', 'openvino-gpu') and not audio_cuda):
        from maichart.composer_v2.acceleration import openvino_devices
        openvino_gpu = (inference_backend == 'openvino-gpu' or audio_device == 'openvino-gpu'
                        or any(name.split('.')[0] == 'GPU' for name in openvino_devices()))
    combined = peak if device == 'xpu' else [max(value, audio) for value, audio in zip(peak, audio_peak or (0, 0))]
    cuda_report = {key: cuda[key] for key in ('name', 'free_bytes', 'total_bytes')} if cuda else None
    return dict(device=device, precision=precision, peak_bytes=combined, model_peak_bytes=peak,
                audio_peak_bytes=audio_peak, workers=workers, activation_checkpointing=scope,
                tokens_per_sample=[_context_shape(config, long_batch)[0] for long_batch in (False, True)],
                batch_size=batch if training else None, cuda=cuda_report,
                cuda_available=cuda_available, xpu_available=xpu_available,
                cuda_budget_bytes=round(cuda_budget * GIB) if cuda_budget and (device == 'cuda' or audio_cuda) else None,
                openvino_gpu=openvino_gpu)
