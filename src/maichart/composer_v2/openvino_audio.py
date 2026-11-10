"""MERT block-8 inference on Intel GPU/NPU, with persistent IR/compile caches.

The output deliberately precedes the encoder's final layer norm, matching the
original block-8 hook. No padding or altered chunk boundaries are introduced.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np


EXPORT_VERSION = 'mert_block8_v1'
FULL_SAMPLES = 374 * 320 + 400


def first_eight(model):
    import torch

    class FirstEight(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = model.feature_extractor
            self.projection = model.feature_projection
            self.position = model.encoder.pos_conv_embed
            self.layers = torch.nn.ModuleList(list(model.encoder.layers[:8]))

        def forward(self, samples):
            value = self.projection(self.features(samples).transpose(1, 2))
            value = value + self.position(value)
            for layer in self.layers:
                value = layer(value)[0]
            return value

    return FirstEight().eval().requires_grad_(False)


class OpenVINOMERT:
    def __init__(self, cache_dir, model_identity, device, load_model):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError('Intel acceleration requires: python -m pip install -e ".[intel-acceleration]"') from exc
        self.ov = ov
        self.core = ov.Core()
        self.device = {'openvino-gpu': 'GPU', 'openvino-npu': 'NPU'}[device]
        if not any(d.split('.')[0] == self.device for d in self.core.available_devices):
            raise RuntimeError(f'OpenVINO {self.device} unavailable; detected {self.core.available_devices}')
        signature = dict(export=EXPORT_VERSION, model=model_identity, openvino=ov.__version__)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.directory = Path(cache_dir) / 'openvino_mert' / key
        self.path = self.directory / 'model.xml'
        self.load_model = load_model
        self.full_request = None

    def _ir(self):
        if not self.path.is_file():
            import torch
            self.directory.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.mert-export-', dir=self.directory.parent))
            try:
                model, _ = self.load_model()
                wrapper = first_eight(model)
                with torch.inference_mode(), warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    warnings.simplefilter('ignore', torch.jit.TracerWarning)
                    example = torch.zeros(1, FULL_SAMPLES)
                    traced = torch.jit.trace(wrapper, (example,), check_trace=False)
                    ir = self.ov.convert_model(traced, input=[[1, FULL_SAMPLES]])
                    self.ov.save_model(ir, temporary / 'model.xml', compress_to_fp16=True)
                (temporary / 'identity.json').write_text(json.dumps(dict(export=EXPORT_VERSION), indent=2), encoding='utf-8')
                try:
                    temporary.rename(self.directory)
                except OSError:
                    if not self.path.is_file():
                        raise
                del wrapper, traced, ir
                gc.collect()
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return self.core.read_model(self.path)

    def infer(self, samples):
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        # Static complete chunks give the best end-to-end latency on both Intel
        # devices. GPU dynamic edge kernels add compilation cost for each song's
        # tail length; CPU edges avoid that and preserve temporal GroupNorm.
        if samples.shape != (1, FULL_SAMPLES):
            return None
        request = self.full_request
        if request is None:
            model = self._ir()
            compiled = self.core.compile_model(model, self.device, {
                'PERFORMANCE_HINT': 'LATENCY',
                'CACHE_DIR': str(self.directory / ('compiled_' + self.device.lower() + '_full')),
            })
            request = compiled.create_infer_request()
            self.full_request=request
            logging.getLogger(__name__).warning('MERT acceleration active: OpenVINO %s, export/compile cache %s', self.device, self.directory)
        # Copy the output: InferRequest owns and reuses its output buffers.
        return np.array(request.infer([samples])[0], dtype=np.float32, copy=True)
