"""CPU regressions for changing microbatch size without replaying samples."""
from itertools import chain
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from maichart.composer import train


class IndexDataset:
    def __init__(self, counts=(8, 11, 7, 13)):
        self.samples = [SimpleNamespace(song_index=song)
                        for song, count in enumerate(counts) for _ in range(count)]
        self.accessed = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.accessed.append(index)
        return index


def _loader(dataset, generator, sampler_size, batch_size, skip=0):
    sampler = train.SongLocalBatchSampler(dataset, sampler_size, generator, locality=4)
    if sampler_size != batch_size:
        sampler = train.RebatchedSampler(sampler, batch_size, len(dataset))
    return DataLoader(dataset, batch_sampler=train.RemainingBatches(sampler, skip),
                      generator=generator, collate_fn=list, num_workers=0)


def _flatten(batches):
    return list(chain.from_iterable(batches))


def _args(**changes):
    values = dict(resume='checkpoint.pt', batch_size=64, grad_accumulation=6,
                  max_steps=0, rebatch_resume=True)
    values.update(changes)
    return SimpleNamespace(**values)


def _checkpoint(*, batch_size=96, accumulation=4, batch=5200, complete=False,
                max_steps=0, **state_changes):
    state = dict(epoch_complete=complete, batch_in_epoch=batch,
                 loader_generator_state=torch.Generator().manual_seed(17).get_state())
    state.update(state_changes)
    return {'args': {'batch_size': batch_size, 'grad_accumulation': accumulation,
                     'max_steps': max_steps}, 'training_state': state}


def test_real_checkpoint_layout_maps_legacy_cursor_without_consuming_rng():
    before = torch.get_rng_state().clone()
    assert train.resume_batch_layout(_args(), _checkpoint(), 949015) == (7800, 96, 499200)
    assert torch.equal(before, torch.get_rng_state())


def test_existing_rebatched_checkpoint_preserves_original_sampler_without_flag():
    saved = _checkpoint(batch_size=64, accumulation=6, batch=7800,
                        sample_in_epoch=499200, sampler_batch_size=96)
    assert train.resume_batch_layout(_args(rebatch_resume=False), saved, 949015) == (7800, 96, 499200)


@pytest.mark.parametrize('old,args,message', [
    (_checkpoint(), _args(rebatch_resume=False), 'explicitly'),
    (_checkpoint(), _args(grad_accumulation=4), 'effective batch'),
    (_checkpoint(max_steps=100), _args(), 'full epochs'),
    (_checkpoint(), _args(max_steps=100), 'full epochs'),
    (_checkpoint(batch=2), _args(), 'complete optimizer'),
    (_checkpoint(sample_in_epoch=499201), _args(), 'inconsistent sample cursor'),
    (_checkpoint(batch=-4), _args(), 'inconsistent sample cursor'),
    (_checkpoint(sampler_batch_size=0), _args(), 'inconsistent sample cursor'),
])
def test_unsafe_rebatch_layouts_are_rejected(old, args, message):
    with pytest.raises(ValueError, match=message):
        train.resume_batch_layout(args, old, 949015)


@pytest.mark.parametrize('cursor', [949015, 949016])
def test_partial_resume_rejects_cursor_at_or_past_dataset_end(cursor):
    saved = _checkpoint(batch_size=1, accumulation=1, batch=cursor,
                        sample_in_epoch=cursor, sampler_batch_size=1)
    with pytest.raises(ValueError, match='inconsistent sample cursor'):
        train.resume_batch_layout(_args(batch_size=1, grad_accumulation=1), saved, 949015)


def test_completed_checkpoint_starts_new_sampler_and_zero_cursor():
    saved = _checkpoint(batch_size=64, accumulation=6, batch=14829,
                        complete=True, sample_in_epoch=949015, sampler_batch_size=96)
    assert train.resume_batch_layout(_args(rebatch_resume=False), saved, 949015) == (0, 64, 0)


def test_rebatch_requires_epoch_loader_rng_to_preserve_order():
    saved=_checkpoint(loader_generator_state=None)
    with pytest.raises(ValueError,match='loader RNG'):
        train.resume_batch_layout(_args(),saved,949015)


def test_versioned_sampler_state_blocks_silent_legacy_resume_and_roundtrips():
    rng=torch.Generator().manual_seed(137).get_state()
    encoded=train.checkpoint_loader_rng_state(rng,96,64,False)
    assert encoded['schema']=='composer_epoch_sampler_v1'
    restored=train.resume_loader_rng_state({'loader_generator_state':encoded,'sampler_batch_size':96})
    assert torch.equal(restored,rng)
    with pytest.raises(TypeError):
        torch.Generator().set_state(encoded)  # legacy code fails before updates
    with pytest.raises(ValueError,match='sampler state'):
        train.resume_loader_rng_state({'loader_generator_state':encoded,'sampler_batch_size':64})
    assert train.checkpoint_loader_rng_state(rng,96,64,True) is rng


@pytest.mark.parametrize('batch_size', [1, 4, 8, 64])
def test_rebatched_sampler_retains_unequal_song_sequence_and_partial_tail(batch_size):
    dataset = IndexDataset()
    original = list(train.SongLocalBatchSampler(dataset, 6,
                    torch.Generator().manual_seed(51913), locality=4))
    sampler = train.RebatchedSampler(original, batch_size, len(dataset))
    actual = list(sampler)
    assert _flatten(actual) == _flatten(original)
    assert len(actual) == len(sampler)
    assert sorted(_flatten(actual)) == list(range(len(dataset)))
    assert all(len(batch) == batch_size for batch in actual[:-1])
    assert 1 <= len(actual[-1]) <= batch_size


def test_dataloader_base_seed_and_skipping_preserve_exact_remaining_sequence():
    # Iterating DataLoader consumes a base seed before the song sampler shuffles.
    # Comparing direct sampler iterations alone would miss this ordering bug.
    original = IndexDataset()
    original_generator = torch.Generator().manual_seed(51913)
    epoch_start = original_generator.get_state().clone()
    full = _flatten(list(_loader(original, original_generator, 6, 6)))
    resumed = IndexDataset()
    restored = torch.Generator().set_state(epoch_start)
    remaining = _flatten(list(_loader(resumed, restored, 6, 4, skip=3)))
    assert remaining == full[12:]
    assert resumed.accessed == full[12:]  # skipped samples never reach __getitem__
    assert torch.equal(restored.get_state(), original_generator.get_state())
    # Ensure this fixture detects the naive implementation that reshuffles at 4.
    naive = _flatten(list(_loader(IndexDataset(),
                    torch.Generator().set_state(epoch_start), 4, 4)))
    assert naive != full


def test_two_mid_epoch_restarts_and_next_epoch_use_saved_sequence(tmp_path, monkeypatch):
    class TinyDataset(IndexDataset):
        codec_atoms = ['1', '2']
        fingerprint = 'rebatch-sequence-fixture'
        candidate_misses = 0
        candidate_total = 39
        candidate_miss_examples = []
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.epoch = 0
            self.visits = []
            self.action_slots = self.action_slots_requested = kwargs['action_slots']
            self.max_required_action_slots = 1
            self.__class__.instances.append(self)

        def set_epoch(self, epoch):
            self.epoch = epoch

        def cache_stats(self):
            return {}

        def __getitem__(self, index):
            self.visits.append((self.epoch, index))
            return {'candidate_features': torch.full((16, 141), index / 100.),
                    'candidate_target': torch.tensor(index % 16),
                    'candidate_mask': torch.zeros(16, dtype=torch.bool),
                    'action_target': torch.tensor([1, 0]),
                    'history': torch.zeros((0, 2), dtype=torch.long),
                    'history_features': torch.zeros((0, 8)),
                    'action_state': torch.zeros(50), 'head_timing': torch.zeros((16, 99)),
                    'condition': torch.zeros(16), 'future_target': torch.zeros(8),
                    'eos_target': torch.tensor(float(index == 38)),
                    'song_index': torch.tensor(self.samples[index].song_index),
                    'event_index': torch.tensor(index)}

    monkeypatch.setattr(train, 'ComposerDataset', TinyDataset)
    original_save = train.save
    arguments = ['--manifest', 'fixture.json', '--epochs', '1', '--d-model', '8',
                 '--layers', '1', '--nhead', '2', '--dim-feedforward', '16', '--dropout', '0',
                 '--history-events', '8', '--candidate-count', '16', '--action-slots', '2',
                 '--device', 'cpu', '--precision', 'fp32', '--prefetch-batches', '0',
                 '--checkpoint-interval-steps', '1', '--architecture-profile', 'music_context_v2',
                 '--seed', '51913']

    def stop_after(batch_number):
        def save_then_interrupt(*args, **kwargs):
            result = original_save(*args, **kwargs)
            if kwargs['training_state']['batch_in_epoch'] == batch_number:
                raise InterruptedError('simulate restart at a persisted optimizer boundary')
            return result
        monkeypatch.setattr(train, 'save', save_then_interrupt)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        stop_after(2)
        with pytest.raises(InterruptedError):
            train.main(arguments + ['--checkpoint-dir', str(tmp_path / 'old'),
                                     '--batch-size', '6', '--grad-accumulation', '2'])
        old_path = tmp_path / 'old/composer_last.pt'
        old = torch.load(old_path, map_location='cpu', weights_only=False)
        assert old['training_state']['sample_in_epoch'] == 12
        assert old['training_state']['sampler_batch_size'] == 6
        epoch_rng = old['training_state']['loader_generator_state']
        expected_generator = torch.Generator().set_state(epoch_rng)
        expected_first = _flatten(list(_loader(IndexDataset(), expected_generator, 6, 6)))

        stop_after(6)
        with pytest.raises(InterruptedError):
            train.main(arguments + ['--checkpoint-dir', str(tmp_path / 'rebatch'),
                '--batch-size', '4', '--grad-accumulation', '3', '--rebatch-resume',
                '--resume', str(old_path)])
        rebatch_path = tmp_path / 'rebatch/composer_last.pt'
        changed = torch.load(rebatch_path, map_location='cpu', weights_only=False)
        assert changed['optimizer_steps'] == 2
        assert changed['training_state']['batch_in_epoch'] == 6
        assert changed['training_state']['sample_in_epoch'] == 24
        assert changed['training_state']['sampler_batch_size'] == 6
        assert changed['training_state']['loader_generator_state']['schema']=='composer_epoch_sampler_v1'
        assert torch.equal(epoch_rng, train.resume_loader_rng_state(changed['training_state']))

        monkeypatch.setattr(train, 'save', original_save)
        assert train.main(arguments + ['--checkpoint-dir', str(tmp_path / 'continued'),
            '--batch-size', '4', '--grad-accumulation', '3', '--epochs', '2',
            '--resume', str(rebatch_path)]) == 0
        final = torch.load(tmp_path / 'continued/composer_last.pt', map_location='cpu', weights_only=False)
        first, second, third = TinyDataset.instances
        assert first.visits == [(1, index) for index in expected_first[:12]]
        assert second.visits == [(1, index) for index in expected_first[12:24]]
        assert [index for epoch, index in third.visits if epoch == 1] == expected_first[24:]
        expected_second = _flatten(list(_loader(IndexDataset(), expected_generator, 4, 4)))
        assert [index for epoch, index in third.visits if epoch == 2] == expected_second
        assert final['epoch'] == 2 and final['optimizer_steps'] == 8
        assert final['training_state']['epoch_complete']
        assert final['training_state']['sample_in_epoch'] == 39  # final batch has 3 samples
        assert final['training_state']['sampler_batch_size'] == 4
        assert torch.equal(final['training_state']['loader_generator_state'], expected_generator.get_state())
    finally:
        torch.set_num_threads(previous_threads)
