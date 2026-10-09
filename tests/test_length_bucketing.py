"""Length bucketing must shorten padding without changing what the epoch sees.

Random batching pads every batch to its longest item; measured windows span
13..756 tokens, so random batches made the GPU compute ~2.2x the real tokens and
peaked the activation memory on the unluckiest mix.  Bucketing groups
length-neighbours inside shuffled chunks.
"""

import random
from pathlib import Path

import pytest

from maichart.training.train_events import (
    LengthBucketedBatchSampler,
    segment_lengths,
)


def test_every_index_is_visited_exactly_once_per_epoch():
    sampler = LengthBucketedBatchSampler([10] * 50 + [200] * 50, 8, shuffle=True, seed=1)
    seen = [index for batch in sampler for index in batch]
    assert sorted(seen) == list(range(100)), "an epoch must cover the dataset exactly once"
    assert len(sampler) == 13, "ceil(100 / 8)"


def test_bucketing_cuts_the_padding_waste():
    """The average batch maximum must fall far below the global maximum."""

    rng = random.Random(0)
    lengths = [rng.choice([13, 60, 120, 250, 400, 756]) for _ in range(2_000)]

    def average_batch_max(batches, values):
        return sum(max(values[i] for i in batch) for batch in batches) / len(batches)

    random_sampler = LengthBucketedBatchSampler(lengths, 16, shuffle=False, seed=0)
    # shuffle=False still sorts; use a plain random batching for the comparison.
    rng2 = random.Random(0)
    shuffled = list(range(len(lengths)))
    rng2.shuffle(shuffled)
    random_batches = [shuffled[i:i + 16] for i in range(0, len(shuffled), 16)]
    bucketed = list(LengthBucketedBatchSampler(lengths, 16, shuffle=True, seed=0))

    random_max = average_batch_max(random_batches, lengths)
    bucketed_max = average_batch_max(bucketed, lengths)
    assert bucketed_max < random_max * 0.7, (
        f"bucketing should cut padding waste: random={random_max:.0f} bucketed={bucketed_max:.0f}")
    assert len(random_sampler) == (len(shuffled) + 15) // 16


def test_chunk_order_is_shuffled_but_locally_sorted():
    lengths = list(range(0, 1000, 10)) * 3
    # chunk_batches=2 -> chunk = 20 samples, so 15 chunks over 300 samples.
    batches = list(LengthBucketedBatchSampler(lengths, 10, shuffle=True, seed=7, chunk_batches=2))
    first_batch_of_chunks = [batches[i][0] for i in range(0, len(batches), 2)]
    positions = [int(lengths[index] / 10) for index in first_batch_of_chunks]
    assert positions != sorted(positions), "chunks must be shuffled, not globally sorted"
    # Within a chunk lengths stay close: the spread of a chunk is much smaller
    # than the spread of the whole dataset.
    chunk = [lengths[index] for batch in batches[:2] for index in batch]
    assert max(chunk) - min(chunk) < 0.6 * (max(lengths) - min(lengths))


def test_epoch_order_comes_from_the_global_rng():
    """Same global seed reproduces the order; successive epochs reshuffle.

    The old sampler used its own ``Random(seed)``, which replayed an identical
    batch order in every epoch (no reshuffling at all) and could not continue a
    resumed run's sequence.  The order now comes from the global RNG, which the
    trainer seeds and checkpoints.
    """

    lengths = [random.Random(3).randint(10, 500) for _ in range(400)]
    sampler = LengthBucketedBatchSampler(lengths, 16, shuffle=True, seed=11)
    random.seed(99)
    first = list(sampler)
    random.seed(99)
    second = list(sampler)
    assert first == second, "re-seeding must reproduce the sequence (resume contract)"
    third = list(sampler)
    assert third != first, "each epoch must reshuffle"
    for order in (first, third):
        assert sorted(index for batch in order for index in batch) == list(range(len(lengths)))


def test_prefetch_loader_preserves_batches_and_order():
    """The prefetch thread must not reorder, drop, or duplicate batches."""

    from maichart.training.train_events import PrefetchLoader

    class FakeLoader:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __iter__(self):
            for item in self.items:
                yield item

    items = [{"batch": index} for index in range(25)]
    wrapped = PrefetchLoader(FakeLoader(items), depth=2)
    assert len(wrapped) == 25
    assert [item["batch"] for item in wrapped] == list(range(25))
    # Second epoch: the stop flag must not survive the first iteration, or every
    # epoch after the first silently yields nothing (batches=0).
    assert [item["batch"] for item in wrapped] == list(range(25))
    assert [item["batch"] for item in wrapped] == list(range(25))
    # Empty and single-item loaders must terminate cleanly.
    assert list(PrefetchLoader(FakeLoader([]), depth=2)) == []
    assert [item["batch"] for item in PrefetchLoader(FakeLoader([{"batch": 7}]), depth=3)] == [7]


def test_prefetch_loader_survives_abandoning_an_epoch():
    """Breaking out of the loop early must not poison later epochs."""

    from maichart.training.train_events import PrefetchLoader

    class FakeLoader:
        def __len__(self):
            return 10

        def __iter__(self):
            for index in range(10):
                yield index

    wrapped = PrefetchLoader(FakeLoader(), depth=2)
    for index, _ in enumerate(wrapped):
        if index == 2:
            break
    assert list(wrapped) == list(range(10))


def test_epoch_reports_the_data_gpu_split():
    """The epoch line must expose loader time vs GPU time."""

    source = (Path(__file__).resolve().parents[1] / "src/maichart/training/train_events.py").read_text(encoding="utf-8")
    assert '"data_seconds"' in source and '"gpu_seconds"' in source
    assert "data_s=" in source and "gpu_s=" in source
    assert "torch.cuda.Event(enable_timing=True)" in source
    assert '"--prefetch-batches"' in source


def test_bucketing_keeps_working_on_small_corpora():
    """A corpus smaller than the old chunk size must still get bucketing.

    chunk = min(batch * chunk_batches, max(batch * 2, n // 8)), so a 1000-window
    corpus is split into 8 chunks instead of one (which had made bucketing a
    no-op equal to random batching).
    """

    lengths = [10] * 500 + [700] * 500
    batches = list(LengthBucketedBatchSampler(lengths, 16, shuffle=True, seed=1))
    assert sorted(index for batch in batches for index in batch) == list(range(1000))
    # With 8 chunks of 125 the extreme groups never meet: no batch mixes a
    # 10-token window with a 700-token one.
    mixed = [batch for batch in batches
             if any(lengths[i] == 10 for i in batch) and any(lengths[i] == 700 for i in batch)]
    assert not mixed, f"{len(mixed)} batches mixed the extremes"


def test_segment_lengths_cover_both_families():
    class Fake:
        def __init__(self, segments):
            self.segments = segments

    # cross-window 7-tuple: (sample, start, end, window, prefix, prefix_len, offset)
    ctx = Fake([(0, 0, 16, [1] * 30, [2] * 5, 7, 0), (0, 16, 32, [1] * 40, [2] * 3, 5, 0)])
    assert segment_lengths(ctx) == [35, 43]
    # plain 4-tuple: (sample, start, end, tokens)
    plain = Fake([(0, 0, 16, [1] * 12)])
    assert segment_lengths(plain) == [12]


def test_trainer_records_throughput_and_buckets_by_default():
    source = (Path(__file__).resolve().parents[1] / "src/maichart/training/train_events.py").read_text(encoding="utf-8")
    assert "ms_per_batch" in source and "train_seconds" in source
    assert '"--no-length-bucketing"' in source
    assert "bucket=not args.no_length_bucketing" in source
    assert '"PYTORCH_CUDA_ALLOC_CONF"' in (
        Path(__file__).resolve().parents[1] / "src/maichart/webui/server.py").read_text(encoding="utf-8")
