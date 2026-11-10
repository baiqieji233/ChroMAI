"""Fuse one incremental decoder step; retain host-owned, rewindable KV caches.

Prefill and multi-token queries keep the original implementation. Returning
only the new K/V vectors avoids copying the whole history back from a device.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import warnings

import numpy as np
import torch
from torch import nn


EXPORT_VERSION = 'composer_decoder_step_v3_static'


class DecoderStep(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))

    def forward(self, x, condition, beats, token_mask, local_mask, local_beats,
                song_mask, song_beats, *memories):
        offset = 0
        appended = []
        for layer in self.layers:
            past_k, past_v, local_k, local_v = memories[offset:offset+4]
            offset += 4
            value = layer.norms[0](x + condition[:, None])
            new_k, new_v = layer.self_attention.memory(value)
            appended.extend((new_k, new_v))
            keys = (torch.cat((past_k, new_k), -2), torch.cat((past_v, new_v), -2))
            x = x + layer.self_attention(value, kv=keys, key_mask=token_mask)
            query = layer.norms[1](x + condition[:, None])
            x = x + layer.local_attention(query, kv=(local_k, local_v), key_mask=local_mask,
                                           query_beats=beats, key_beats=local_beats)
            if layer.song_attention is not None:
                song_k, song_v = memories[offset:offset+2]
                offset += 2
                x = x + layer.song_attention(layer.norms[2](x + condition[:, None]),
                    kv=(song_k, song_v), key_mask=song_mask, query_beats=beats, key_beats=song_beats)
            x = x + layer.ff(layer.norms[3](x))
        return (x, *appended)


class DecoderRuntime:
    def __init__(self, model, args):
        self.model = model
        self.requested = getattr(args, 'inference_backend', 'auto')
        self.cache_dir = Path(args.cache_dir) / 'openvino_decoder'
        self.selected = None
        self.request = None
        self.failed = False
        self.report = None
        self.autotune_threads = not getattr(args, 'cpu_threads', 0)
        self.threads = torch.get_num_threads()
        self._gpu_requests = {}
        self._gpu_buffers = {}
        self._gpu_ir = self._gpu_directory = None
        self._past_indices = set()
        offset = 8
        for layer in model.decoder:
            self._past_indices.update((offset,offset+1))
            offset += 4 + (2 if layer.song_attention is not None else 0)

    def _inputs(self, x, condition, beats, token_mask, audio, state, tile):
        view = slice(tile, tile+1)
        inputs = [x, condition, beats, token_mask, audio['local_mask'][view],
                  audio['local_beats'][view], audio['song_mask'], audio['song_beats']]
        for layer, memory in zip(self.model.decoder, state['layers']):
            if 'self_kv' not in memory:
                return None
            local = memory.get('local_kv')
            if local is None or local[0].size(0) != audio['local'].size(0):
                local = layer.local_attention.memory(audio['local'])
                memory['local_kv'] = local
            inputs.extend((*memory['self_kv'], *(v[view] for v in local)))
            if layer.song_attention is not None:
                if 'song_kv' not in memory:
                    memory['song_kv'] = layer.song_attention.memory(audio['song'])
                inputs.extend(memory['song_kv'])
        return inputs

    def _ir(self, wrapper, inputs, ov):
        signature = hashlib.sha256()
        signature.update((EXPORT_VERSION + ov.__version__ + str(torch.__version__)).encode())
        signature.update(json.dumps(self.model.config.to_dict(), sort_keys=True).encode())
        for name, tensor in wrapper.state_dict().items():
            signature.update(name.encode())
            signature.update(tensor.detach().contiguous().numpy().tobytes())
        directory = self.cache_dir / signature.hexdigest()
        path = directory / 'model.xml'
        if not path.is_file():
            directory.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.decoder-export-', dir=directory.parent))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    warnings.simplefilter('ignore', torch.jit.TracerWarning)
                    traced = torch.jit.trace(wrapper, tuple(inputs), check_trace=False)
                ir = ov.convert_model(traced, input=[list(v.shape) for v in inputs])
                shapes = {}
                for i, value in enumerate(inputs):
                    shape = list(value.shape)
                    if i in (3, 4, 5, 6, 7):
                        shape[1] = -1
                    elif i >= 8:
                        shape[2] = -1
                    shapes[i] = shape
                ir.reshape(shapes)
                ov.save_model(ir, temporary / 'model.xml', compress_to_fp16=False)
                try:
                    temporary.rename(directory)
                except OSError:
                    if not path.is_file():
                        raise
                del traced, ir
                gc.collect()
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        return directory, ov.Core().read_model(path)

    @staticmethod
    def _arrays(inputs):
        return [np.ascontiguousarray(value.detach().numpy()) for value in inputs]

    def _gpu_inputs(self, inputs):
        length=inputs[3].shape[1]-1
        bucket=64
        while bucket<length:
            bucket*=2
        arrays=[]
        for index,value in enumerate(inputs):
            raw=value.detach().numpy()
            if index==3 or index in self._past_indices:
                shape=list(raw.shape)
                shape[1 if index==3 else 2]=bucket+1 if index==3 else bucket
                key=(index,tuple(shape))
                array=self._gpu_buffers.get(key)
                if array is None:
                    array=np.zeros(shape,dtype=raw.dtype);self._gpu_buffers[key]=array
                if index==3:
                    array.fill(False);array[:,:length]=raw[:,:length];array[:,-1]=raw[:,-1]
                else:
                    array[:,:,:length]=raw
                arrays.append(array)
            else:
                arrays.append(np.ascontiguousarray(raw))
        return arrays

    def _gpu_request(self, arrays):
        key=tuple(tuple(v.shape) for v in arrays)
        if key not in self._gpu_requests:
            model=self._gpu_ir.clone()
            model.reshape({i:list(v.shape) for i,v in enumerate(arrays)})
            compiled=self._ov_core.compile_model(model,'GPU',{
                'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f16',
                'CACHE_DIR':str(self._gpu_directory/'compiled_gpu_static')})
            self._gpu_requests[key]=compiled.create_infer_request()
        return self._gpu_requests[key]

    @staticmethod
    def _infer(request, arrays):
        result = request.infer(arrays)
        return [np.array(result[i], copy=True) for i in range(len(result))]

    def _torch_runner(self, inputs):
        """Measure the existing reusable CPU cache, rather than a slower concat."""
        x,condition,beats,mask,local_mask,local_beats,song_mask,song_beats=inputs[:8]
        audio=dict(local=x.new_empty(1,local_mask.size(1),x.size(-1)),local_mask=local_mask,
                   local_beats=local_beats,song=x.new_empty(1,song_mask.size(1),x.size(-1)),
                   song_mask=song_mask,song_beats=song_beats)
        states=[];offset=8
        for layer in self.model.decoder:
            past=inputs[offset:offset+2]
            buffers=tuple(value.new_empty((*value.shape[:2],value.size(2)+1,value.size(3))) for value in past)
            for buffer,value in zip(buffers,past):buffer[:,:,:value.size(2)].copy_(value)
            memory=dict(self_kv_storage=buffers,local_kv=tuple(inputs[offset+2:offset+4]))
            offset+=4
            if layer.song_attention is not None:
                memory['song_kv']=tuple(inputs[offset:offset+2]);offset+=2
            states.append(memory)
        length=mask.size(1)-1
        def run():
            value=x
            for layer,memory in zip(self.model.decoder,states):
                memory['self_kv']=tuple(buffer[:,:,:length] for buffer in memory['self_kv_storage'])
                value=layer(value,condition,mask,beats,audio,state=memory,groups=0)
            return value
        return run

    def _select(self, inputs):
        import openvino as ov
        wrapper = DecoderStep(self.model.decoder).eval()
        reference = tuple(wrapper(*inputs))
        torch_run=self._torch_runner(inputs)
        directory, ir = self._ir(wrapper, inputs, ov)
        core = ov.Core()
        self._ov_core=core
        self._gpu_ir,self._gpu_directory=ir,directory
        hardware=[]
        for device in core.available_devices:
            if device.split('.')[0] not in ('CPU','GPU'):
                continue
            details=[device,str(core.get_property(device,'FULL_DEVICE_NAME'))]
            try:details.append(str(core.get_property(device,'DRIVER_VERSION')))
            except RuntimeError:pass
            hardware.append(details)
        profile=dict(hardware=hardware,requested=self.requested,threads=torch.get_num_threads(),
                     autotune=self.autotune_threads,benchmark='stateful_torch_v1')
        key=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()[:24]
        selection_path=directory/('selection_'+key+'.json')
        if self.requested=='auto':
            try:
                saved=json.loads(selection_path.read_text(encoding='utf-8'))
                if saved['selected']=='torch':
                    self.selected='torch';self.threads=saved['cpu_threads']
                    torch.set_num_threads(self.threads)
                    self.report={**saved,'cached_selection':True}
                    self._gpu_ir=None
                    print('composer_v2_acceleration '+json.dumps(self.report),flush=True)
                    return
                # Accelerators still validate their first actual input each run.
                preferred=saved['selected']
            except (OSError,ValueError,KeyError,TypeError):
                preferred=None
        else:preferred=self.requested
        measurements = {}
        candidates = {}
        thread_measurements = {}
        if self.autotune_threads:
            maximum = torch.get_num_threads()
            for threads in sorted({1, min(2, maximum), min(4, maximum), maximum}):
                torch.set_num_threads(threads)
                torch_run()
                times = []
                for _ in range(4):
                    began = time.perf_counter(); torch_run(); times.append(time.perf_counter()-began)
                thread_measurements[threads] = float(np.median(times))*1000.
            self.threads = min(thread_measurements, key=thread_measurements.get)
            torch.set_num_threads(self.threads)
        for _ in range(2):
            torch_run()
        times = []
        for _ in range(3):
            began = time.perf_counter(); torch_run(); times.append(time.perf_counter()-began)
        measurements['torch'] = dict(median_ms=float(np.median(times))*1000.)
        arrays = self._arrays(inputs)
        devices = ['CPU', 'GPU'] if self.requested == 'auto' and preferred is None else [preferred.split('-')[-1].upper()]
        available = {d.split('.')[0] for d in core.available_devices}
        for device in devices:
            if device not in available:
                continue
            name = 'openvino-' + device.lower()
            try:
                properties = {'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f32' if device=='CPU' else 'f16',
                              'CACHE_DIR': str(directory / ('compiled_' + device.lower()))}
                if device == 'CPU':
                    properties['INFERENCE_NUM_THREADS'] = torch.get_num_threads()
                if device=='GPU':
                    device_arrays=self._gpu_inputs(inputs)
                    request=self._gpu_request(device_arrays)
                else:
                    compiled = core.compile_model(ir, device, properties)
                    request = compiled.create_infer_request()
                    device_arrays=arrays
                values = self._infer(request, device_arrays)
                error = max(float(np.linalg.norm(value-ref.detach().numpy()) /
                                  max(1e-6, np.linalg.norm(ref.detach().numpy())))
                            for value, ref in zip(values, reference))
                if not np.isfinite(error) or error > .002:
                    raise ValueError(f'Decoder validation relative L2 error {error} exceeds 0.002')
                for _ in range(2):
                    self._infer(request, device_arrays)
                times = []
                for _ in range(4):
                    began = time.perf_counter()
                    # Include host packing and output copies in backend selection.
                    self._infer(request, self._gpu_inputs(inputs) if device=='GPU' else self._arrays(inputs))
                    times.append(time.perf_counter()-began)
                measurements[name] = dict(median_ms=float(np.median(times))*1000., relative_l2_error=error)
                candidates[name] = request
            except Exception as exc:
                measurements[name] = dict(error=str(exc))
                logging.getLogger(__name__).warning('Decoder backend %s unavailable: %s', name, exc)
        if self.requested == 'auto':
            best = min((key for key, value in measurements.items() if 'median_ms' in value),
                       key=lambda key: measurements[key]['median_ms'])
            # Avoid selecting an accelerator on timing noise alone.
            if measurements[best]['median_ms'] > measurements['torch']['median_ms'] * .9:
                best = 'torch'
        else:
            best = self.requested
            if best not in candidates:
                raise RuntimeError(f'Decoder backend {best} failed: {measurements.get(best)}')
        self.selected = best
        self.request = candidates.get(best)
        if best!='openvino-gpu':
            self._gpu_requests.clear();self._gpu_buffers.clear();self._gpu_ir=None
        self.report = dict(requested=self.requested, selected=best, measurements=measurements,
                           cpu_threads=self.threads, cpu_thread_measurements=thread_measurements,
                           scope='single_token_decoder_layers', precision='fp16' if best=='openvino-gpu' else 'fp32')
        temporary=selection_path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_text(json.dumps(self.report,indent=2),encoding='utf-8')
            temporary.replace(selection_path)
        finally:
            temporary.unlink(missing_ok=True)
        print('composer_v2_acceleration ' + json.dumps(self.report), flush=True)

    def step(self, x, condition, beats, token_mask, audio, state, groups):
        if self.failed or self.selected == 'torch' or not isinstance(groups, int) or x.shape[:2] != (1, 1):
            return None
        try:
            inputs = self._inputs(x, condition, beats, token_mask, audio, state, groups)
            if inputs is None:
                return None
            if self.selected is None:
                self._select(inputs)
            if self.request is None:
                return None
            if self.selected=='openvino-gpu':
                arrays=self._gpu_inputs(inputs)
                outputs=self._infer(self._gpu_request(arrays),arrays)
            else:
                outputs = self._infer(self.request, self._arrays(inputs))
            if len(outputs) != 1+2*len(self.model.decoder) or not all(np.isfinite(v).all() for v in outputs):
                raise ValueError('Invalid OpenVINO decoder outputs')
            if outputs[0].shape!=tuple(x.shape):
                raise ValueError('Invalid OpenVINO decoder hidden-state shape')
            for i,layer in enumerate(self.model.decoder):
                expected=(1,layer.self_attention.heads,1,layer.self_attention.dim)
                if any(value.shape!=expected for value in outputs[1+2*i:3+2*i]):
                    raise ValueError('Invalid OpenVINO decoder KV shape')
        except Exception as exc:
            self.failed = True
            if self.requested != 'auto':
                raise
            if self.report is not None:
                self.report.update(fallback='torch',error=str(exc))
            logging.getLogger(__name__).exception('Decoder acceleration failed; retaining PyTorch inference')
            return None
        # Commit only after every output has been checked. Allocation failures
        # during commit propagate, rather than replaying a partially appended step.
        for i,(layer,memory) in enumerate(zip(self.model.decoder,state['layers'])):
            new=tuple(torch.from_numpy(v) for v in outputs[1+2*i:3+2*i])
            memory['self_kv'],_=layer._append_cpu_kv(new,memory)
        return torch.from_numpy(outputs[0])
