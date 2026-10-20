"""Bounded partial-network CUDA graphs, independent of checkpoint semantics.

Only the dense Transformer blocks are captured. Rational clocks, retrieval,
supervision and the optimizer remain live. There is one shape slot per block;
other shapes use ordinary execution rather than growing a graph cache forever.
"""
from __future__ import annotations

import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class _Encoder(nn.Module):
    def __init__(self, block):
        super().__init__(); self.block = block

    def forward(self, values, padding):
        return self.block(values, src_key_padding_mask=padding, is_causal=False)


class _Decoder(nn.Module):
    def __init__(self, block):
        super().__init__(); self.block = block

    def forward(self, values, memory, padding):
        length = values.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=values.device).triu(1)
        return self.block(values, memory, tgt_mask=causal,
                          memory_key_padding_mask=padding, tgt_is_causal=True)


class TrainingGraphRunner:
    def __init__(self, *, mode='auto', batch_size=8, max_tokens=1024,
                 max_audio_tokens=1024, budget_gb=2., reserve_gb=1.5, report=None):
        if mode not in {'auto', 'eager', 'cuda_graph'}:
            raise ValueError('Unknown stream training execution mode')
        if not math.isfinite(budget_gb) or budget_gb < 0:
            raise ValueError('CUDA graph budget must be a finite nonnegative value')
        self.mode, self.batch_size = mode, batch_size
        self.max_tokens, self.max_audio_tokens = max_tokens, max_audio_tokens
        self.budget = int(budget_gb * 1024**3)
        self.reserve = int(reserve_gb * 1024**3)
        self.report = report or (lambda value: print('stream_execution '+json.dumps(value), flush=True))
        self.slots, self.disabled, self.observations = {}, {}, {}
        self.hits, self.fallbacks, self.bytes = {}, {}, 0
        self.prepared = False
        self.initializing = False

    def _notice(self, name, reason, **values):
        self.report({'block': name, 'reason': reason, **values})

    @staticmethod
    def _signature(args):
        enabled=torch.is_autocast_enabled('cuda')
        return (enabled, torch.get_autocast_dtype('cuda') if enabled else None,
                tuple((tuple(value.shape), value.dtype, value.requires_grad) for value in args))

    def _execute(self, name, block, args):
        signature = self._signature(args)
        if name in self.slots:
            expected, graphed = self.slots[name]
            if signature == expected:
                self._own_parameter_gradients(graphed)
                self.hits[name] = self.hits.get(name, 0)+1
                return graphed(*args)
            return None
        if name in self.disabled:
            return None
        if self.prepared:
            # Never capture while CPU prefetch workers may allocate pinned
            # storage. CUDA's global stream capture forbids such concurrency.
            return None
        seen = self.observations.setdefault(name, {})
        seen[signature] = seen.get(signature, 0)+1
        # Let the regular path initialize its kernels before graph warmup.
        if seen[signature] < 2:
            return None

        device = args[0].device
        free, _ = torch.cuda.mem_get_info(device)
        if self.bytes >= self.budget or free <= self.reserve:
            self.disabled[name] = 'memory_budget'
            self._notice(name, 'memory_budget', free_gib=free/1024**3)
            return None
        # Do not start capture when the estimated block activations would
        # consume the remaining device headroom. This is a guard, not a tensor
        # or music-content budget. A real graph allocation is checked below.
        layers = len(block.layers)
        estimate = args[0].numel()*layers*12*args[0].element_size()
        if free < self.reserve+estimate:
            self.disabled[name] = 'memory_headroom'
            self._notice(name, 'memory_headroom', estimated_gib=estimate/1024**3)
            return None
        wrapper = _Encoder(block) if name == 'encoder' else _Decoder(block)
        wrapper.train(block.training)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        static = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in args)
        started = time.monotonic()
        # Capture warmup must neither consume real dropout draws nor write to
        # existing accumulated parameter gradients. make_graphed_callables
        # uses autograd.grad, leaving .grad owned by the outer optimizer.
        try:
            enabled = torch.is_autocast_enabled('cuda')
            dtype = torch.get_autocast_dtype('cuda')
            with torch.random.fork_rng(devices=[device]), torch.autocast(
                    'cuda', enabled=enabled, dtype=dtype, cache_enabled=False):
                graphed = torch.cuda.make_graphed_callables(wrapper, static, num_warmup_iters=2)
            # Graph-private pools can retain workspace that no longer counts
            # as an allocated tensor. Budget the reserved increase as well.
            added = max(0, torch.cuda.memory_allocated(device)-allocated,
                        torch.cuda.memory_reserved(device)-reserved)
            if self.bytes+added > self.budget:
                self.disabled[name] = 'graph_budget_exceeded'
                self._notice(name, 'graph_budget_exceeded', added_gib=added/1024**3)
                # make_graphed_callables installs a closure on its wrapper;
                # break that ownership cycle before abandoning a capture.
                graphed.forward = type(graphed).forward.__get__(graphed)
                return None
            self.slots[name] = (signature, graphed)
            self.bytes += added
            self.hits.setdefault(name, 0)
            self._notice(name, 'captured', extra_gib=added/1024**3,
                         seconds=time.monotonic()-started,
                         shape=[list(value.shape) for value in args])
            if self.initializing:
                return None
            self._own_parameter_gradients(graphed)
            self.hits[name]+=1
            return graphed(*args)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            self.disabled[name] = str(exc).splitlines()[0][:240]
            self._notice(name, 'capture_unavailable', error=self.disabled[name])
            return None

    @staticmethod
    def _own_parameter_gradients(graphed):
        # The graph API returns reusable static gradient tensors. If a leaf
        # has grad=None, autograd can adopt that returned storage directly;
        # the next graph replay would overwrite the accumulated gradient.
        # Independent zeroed leaf buffers force accumulation into owned memory.
        # Existing gradients (including a partially accumulated group) stay put.
        for parameter in graphed.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad=torch.zeros_like(parameter)

    def prepare(self, model, precision):
        """Capture standard dense shapes once, before starting data workers.

        No authored labels or real activations are needed for a shape capture.
        LayerNorm has no running statistics. Random generators are restored
        around graph setup; optimizer/parameter gradients remain untouched.
        """
        device=next(model.parameters()).device
        # The real-batch probe on the current Windows wheel did not show a
        # graph speedup. Keep automatic mode conservative; explicit graph mode
        # remains available for a speed/memory comparison on the user's device.
        if self.mode!='cuda_graph' or not self.budget or device.type!='cuda':
            self.prepared=True
            return self.stats()
        width=model.config.d_model
        self.initializing=True
        memory_length=((model.config.max_audio_tokens+model.config.max_history_tokens
                        +model.config.max_stream_memory+32+63)//64)*64
        def values(length):
            return torch.zeros(self.batch_size,length,width,device=device,requires_grad=True)
        def mask(length):
            return torch.zeros(self.batch_size,length,device=device,dtype=torch.bool)
        with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                            enabled=precision in {'bf16','fp16'}):
            for name,block in (('encoder',model.audio_encoder),('decoder',model.decoder)):
                args=((values(self.max_audio_tokens),mask(self.max_audio_tokens)) if name=='encoder'
                      else (values(self.max_tokens),values(memory_length),mask(memory_length)))
                signature=self._signature(args)
                self.observations.setdefault(name,{})[signature]=2
                self._execute(name,block,args)
        self.prepared=True
        self.initializing=False
        return self.stats()

    def _eligible(self, block, values):
        return (self.mode == 'cuda_graph' and self.budget > 0 and block.training
                and torch.is_grad_enabled() and values.is_cuda
                and values.size(0) == self.batch_size)

    def encoder(self, block, values, padding):
        if not self._eligible(block, values):
            return block(values, src_key_padding_mask=padding, is_causal=False)
        length = values.size(1)
        if length > self.max_audio_tokens:
            return block(values, src_key_padding_mask=padding, is_causal=False)
        padded = values if length==self.max_audio_tokens else F.pad(values, (0, 0, 0, self.max_audio_tokens-length))
        mask = padding if length==self.max_audio_tokens else F.pad(padding, (0, self.max_audio_tokens-length), value=True)
        result = self._execute('encoder', block, (padded, mask))
        if result is None:
            self.fallbacks['encoder'] = self.fallbacks.get('encoder', 0)+1
            return block(values, src_key_padding_mask=padding, is_causal=False)
        return result[:, :length]

    def decoder(self, block, values, memory, padding):
        length = values.size(1)
        if self._eligible(block, values) and length <= self.max_tokens:
            # Add only trailing masked memory; retain all real history/actions.
            memory_length = ((memory.size(1)+63)//64)*64
            padded = values if length==self.max_tokens else F.pad(values, (0, 0, 0, self.max_tokens-length))
            evidence = memory if memory_length==memory.size(1) else F.pad(memory, (0, 0, 0, memory_length-memory.size(1)))
            mask = padding if memory_length==memory.size(1) else F.pad(padding, (0, memory_length-memory.size(1)), value=True)
            result = self._execute('decoder', block, (padded, evidence, mask))
            if result is not None:
                return result[:, :length]
            self.fallbacks['decoder'] = self.fallbacks.get('decoder', 0)+1
        causal = torch.ones(length, length, dtype=torch.bool, device=values.device).triu(1)
        return block(values, memory, tgt_mask=causal,
                     memory_key_padding_mask=padding, tgt_is_causal=True)

    def stats(self):
        return {'requested': self.mode, 'captured': sorted(self.slots),
                'replays': dict(self.hits), 'eager_batches': dict(self.fallbacks),
                'disabled': dict(self.disabled), 'extra_bytes': self.bytes,
                'budget_bytes': self.budget}


def install_training_graphs(model, **settings):
    # This plain Python runner is not a child Module: checkpoint key paths and
    # optimizer parameter ownership must remain exactly the original ones.
    runner = TrainingGraphRunner(**settings)
    object.__setattr__(model, '_training_graph_runner', runner)
    return runner


class MetricAccumulator:
    """Sum scalar metrics by dtype in a few kernels, preserving integer counts."""
    def __init__(self):
        self.groups = {}
        self.names = ()

    def update(self, metrics):
        if not self.names:
            self.names = tuple(metrics)
            grouped = {}
            for name, value in metrics.items():
                grouped.setdefault((value.device, value.dtype), []).append(name)
            for key, names in grouped.items():
                self.groups[key] = (tuple(names), torch.stack([metrics[name].detach() for name in names]))
            return
        if tuple(metrics) != self.names:
            raise ValueError('Training metrics changed during accumulation')
        for names, total in self.groups.values():
            total.add_(torch.stack([metrics[name].detach() for name in names]))

    def as_dict(self):
        values = {name: total[index] for names, total in self.groups.values()
                  for index, name in enumerate(names)}
        return {name: values[name] for name in self.names}
