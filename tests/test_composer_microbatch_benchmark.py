import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location('microbatch_bench',
    Path(__file__).resolve().parents[1] / 'tools' / 'benchmark_composer_microbatch.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_effective_batch_partition_preserves_order():
    items = list(range(256))
    for width, expected in [(64, 4), (128, 2), (256, 1)]:
        groups = bench.chunks(items, width)
        assert len(groups) == expected
        assert [x for group in groups for x in group] == items
    with pytest.raises(ValueError):
        bench.chunks(items, 100)


def test_stress_is_explicit_and_does_not_mutate_fixed_inputs():
    sample = {'action_target': torch.tensor([7] + [0] * 32),
              'history': torch.zeros(3, 33, dtype=torch.long)}
    inputs = [sample, sample]
    stress = bench.stress_items(inputs)
    assert sample['action_target'].sum() == 7
    assert sample['history'].sum() == 0
    assert stress[0]['action_target'].ne(0).sum() == 33
    assert stress[0]['history'][-1].ne(0).sum() == 33
    assert stress[1] is sample
