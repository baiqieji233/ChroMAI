"""Group fully prepared examples while preserving optimizer-update membership."""
from itertools import islice
import math
import torch

from .data import LengthBucketBatchSampler


FULL_INPUT = 'full_input_v1'
TARGET_ONLY = 'target_only_v1'
CHOICES = ('auto', FULL_INPUT, TARGET_ONLY)


class PreparedDataset:
    def __init__(self, source):
        self.source = source
        self.items = {}

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.items.pop(index)


class FullInputBatchSampler:
    """Sort within one optimizer update, then collate into smaller batches."""
    def __init__(self, dataset, batch_size, accumulation, *, source_batch_size, seed, order):
        self.batch_size = int(batch_size)
        self.accumulation = int(accumulation)
        self.effective_batch = self.batch_size * self.accumulation
        if min(self.batch_size, self.accumulation, source_batch_size) < 1 or self.effective_batch % source_batch_size:
            raise ValueError('Source batch size must divide the optimizer batch')
        self.source_batches_per_update = self.effective_batch // source_batch_size
        self.base = LengthBucketBatchSampler(dataset, source_batch_size, seed=seed, shuffle=True, order=order)
        self.prepared = PreparedDataset(dataset)
        self.start_batch = 0

    def __len__(self):
        return math.ceil(len(self.prepared) / self.batch_size)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        if self.start_batch == len(self):
            return
        if self.start_batch < 0 or self.start_batch > len(self) or self.start_batch % self.accumulation:
            raise ValueError('Full-input batching resumes at optimizer-update boundaries')
        self.base.start_batch = (self.start_batch // self.accumulation) * self.source_batches_per_update
        batches = iter(self.base)
        try:
            while True:
                source_batches = list(islice(batches, self.source_batches_per_update))
                group = [int(index) for batch in source_batches for index in batch]
                if not group:
                    break
                self.prepared.items.update((index, self.prepared.source[index]) for index in group)
                for source_batch in source_batches:
                    items = [self.prepared.items[int(index)] for index in source_batch]
                    if 'curve_counts' in items[0]:
                        preserve_curve_weights(items)
                group.sort(key=lambda index: len(self.prepared.items[index]['input_ids']))
                for begin in range(0, len(group), self.batch_size):
                    yield group[begin:begin+self.batch_size]
        finally:
            self.prepared.items.clear()


def preserve_curve_weights(items):
    if any('gesture_peer' in item for item in items):
        from .gesture_pairs import expand
        items=expand(items)
    """Keep the existing curve numerator when rearranging a source minibatch."""
    token_mass = 0.
    quality_mass = 0.
    for item in items:
        quality = float(item.get('sample_weight', 1.))
        active = item['labels'].ne(-100)
        token_mass += float((item.get('token_weights', active.float()) * active).double().sum()) * quality
        widths = item['curve_widths']
        exposure = widths[...,None] * item.get('curve_quality_weights',torch.ones_like(item['curve_counts']))
        quality_mass += quality * bool((exposure * (widths > 0)[...,None]).sum() > 0)
    scale = max(1.,token_mass) / max(1e-6,quality_mass)
    for item in items:
        item['curve_loss_scale'] = torch.tensor(scale,dtype=torch.float32)


def resolve_layout(requested, batch_size, accumulation, parent=None):
    if requested not in CHOICES:
        raise ValueError('Unknown training batch layout')
    saved = (parent or {}).get('training_contract', {})
    old_layout = saved.get('batch_layout', TARGET_ONLY)
    layout = (old_layout if parent else FULL_INPUT) if requested == 'auto' else requested
    changed_batch = bool(parent and (batch_size, accumulation) != (saved['batch_size'], saved['grad_accumulation']))
    if parent and (changed_batch or layout != old_layout):
        if layout != FULL_INPUT:
            raise ValueError('调整续训批大小时，请将「批次分组」选为「按完整输入长度」。')
        if batch_size * accumulation != saved['batch_size'] * saved['grad_accumulation']:
            raise ValueError('续训时，单卡批大小 × 梯度累积须保持一致。')
    source_size = saved.get('batch_source_size', saved.get('batch_size', batch_size))
    validation_size = saved.get('validation_batch_size', saved.get('batch_size', batch_size))
    return layout, source_size, validation_size


def resume_cursor(parent_state, old_contract, new_contract, batches_per_epoch):
    old_cursor = int(parent_state['batch_cursor'])
    old_accumulation = int(old_contract['grad_accumulation'])
    if old_cursor % old_accumulation:
        raise ValueError('续训检查点不在一次完整参数更新的边界。')
    cursor = old_cursor // old_accumulation * int(new_contract['grad_accumulation'])
    if cursor > batches_per_epoch:
        raise ValueError('续训位置超出当前轮次。')
    return cursor


def compatible_contract(saved, current):
    if saved == current:
        return True
    if current.get('batch_layout') != FULL_INPUT:
        return False
    if saved['batch_size'] * saved['grad_accumulation'] != current['batch_size'] * current['grad_accumulation']:
        return False
    if current['batch_source_size'] != saved.get('batch_source_size', saved['batch_size']):
        return False
    if current['validation_batch_size'] != saved.get('validation_batch_size', saved['batch_size']):
        return False
    allowed = {'batch_size', 'grad_accumulation', 'batch_layout', 'batch_source_size', 'validation_batch_size'}
    return {key:value for key,value in saved.items() if key not in allowed} == {
        key:value for key,value in current.items() if key not in allowed}
