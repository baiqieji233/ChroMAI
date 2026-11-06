"""Data contracts independent of heavyweight offline encoders."""
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer_v2.data import (
    AUDIO_LOCAL_BATCH_ORDER, ComposerV2Dataset, ComposerV2Song, LengthBucketBatchSampler, collate_composer_v2,
)
from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


class Codec:
    pad_id, bos_id, ctx_id, ref_id, tgt_id, eos_id, carry_id = range(7)

    def encode_points(self, points, origin, region):
        result = []
        for point in points:
            result += [100 + int(point.beat - origin), 200]
        return result

    def token_clocks(self, ids):
        clock, result = 0., []
        for token in ids:
            if 60 <= token < 150:
                clock = token - 100
            result.append(clock)
        return result


class Audio:
    duration_sec = 200.

    def sample(self, seconds):
        return (np.zeros((len(seconds), 98), np.float32),
                np.zeros((len(seconds), 1024), np.float32),
                np.zeros(len(seconds), bool))

    def summary(self, max_tokens=160):
        return {'mel': np.zeros((2, 98), np.float32),
                'semantic': np.zeros((2, 1024), np.float32),
                'semantic_valid': np.zeros(2, bool),
                'times_sec': np.array([0., self.duration_sec])}


def fixture_dataset(max_tokens=128):
    data = ComposerV2Dataset.__new__(ComposerV2Dataset)
    data.condition_policy = 'radar_style_v1'
    data.codec = Codec()
    data.window_beats, data.history_beats, data.future_beats = 16, 16, 4
    data.max_tokens = max_tokens
    points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(16), '2'),
              EventPoint(Fraction(31), '3')]
    song = ComposerV2Song('test', Path('chart'), Path('audio'), 5, 13., points,
        TempoClock([EventTempo(Fraction(0), 120.), EventTempo(Fraction(20), 60.)], .75),
        Fraction(32), np.zeros(16, np.float32), audio=Audio())
    song.summary = {'mel': np.zeros((2, 98), np.float32),
                    'semantic': np.zeros((2, 1024), np.float32),
                    'semantic_valid': np.zeros(2, bool),
                    'times_sec': np.array([0., 100.]), 'beats': np.array([0., 100.])}
    data.songs, data.windows = [song], [(0, Fraction(0)), (0, Fraction(16))]
    return data, song


def test_target_only_loss_and_boundary_assignment():
    data, song = fixture_dataset()
    first = data.build_window(song, song.points, 0, training=True)
    second = data.build_window(song, song.points, 16, training=True)
    assert first['labels'][first['labels'] != -100].tolist() == [100, 200, data.codec.eos_id]
    assert second['labels'][second['labels'] != -100].tolist() == [100, 200, 115, 200, data.codec.eos_id]
    assert len(first['token_beats']) == len(first['input_ids'])
    # The final history component's input predicts TGT; TGT predicts first target time.
    assert second['input_ids'][(second['labels'] != -100).nonzero()[0]] == data.codec.tgt_id


def test_chart_cache_reuses_exact_points_and_sustain_and_invalidates_edits(tmp_path):
    source = tmp_path / 'maidata.txt'
    original = '&first=0.125\n&lv_5=13\n&inote_5=(120){8}1h[4:1],2-5[8:1],(180)3,,,E\n'
    source.write_text(original, encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [{
        'maidata_path': str(source), 'audio_path': str(tmp_path / 'audio.wav'),
        'difficulties': [{'index': 5, 'level': '13'}],
    }]}), encoding='utf-8')

    def load():
        return ComposerV2Dataset(manifest, tmp_path / 'cache', prepare=False)

    cold, warm = load(), load()
    assert cold.audit['preparation']['chart_cache_hits'] == 0
    assert warm.audit['preparation']['chart_cache_hits'] == 1
    a, b = cold.songs[0], warm.songs[0]
    assert (a.points, a.authored_end_beat) == (b.points, b.authored_end_beat)
    assert a.sustain.markers == b.sustain.markers
    assert a.sustain.tempo_changes == b.sustain.tempo_changes
    states, mask, metadata = a.sustain.at(Fraction(3, 4), visible_before_beat=Fraction(3, 4))
    warm_states, warm_mask, warm_metadata = b.sustain.at(Fraction(3, 4), visible_before_beat=Fraction(3, 4))
    assert torch.equal(states, warm_states)
    assert torch.equal(mask, warm_mask)
    assert metadata == warm_metadata

    cache_file, = (tmp_path / 'cache').rglob('*.json.zlib')
    cache_file.write_bytes(b'broken')
    assert load().audit['preparation']['chart_cache_hits'] == 0
    assert load().audit['preparation']['chart_cache_hits'] == 1

    source.write_text(original.replace('0.125', '0.375'), encoding='utf-8')
    changed = load()
    assert changed.audit['preparation']['chart_cache_hits'] == 0
    assert changed.songs[0].sustain.first == 0.375


def test_empty_window_still_supervises_eos():
    data, song = fixture_dataset()
    item = data.build_window(song, song.points, 40, training=True)
    assert item['labels'][item['labels'] != -100].tolist() == [data.codec.eos_id]


def test_target_budget_never_silently_deletes_notes():
    data, song = fixture_dataset(max_tokens=5)
    with pytest.raises(ValueError, match='Complete target/carry'):
        data.build_window(song, song.points, 16, training=True)


def test_train_and_inference_prefix_are_identical_despite_target_density():
    data, song = fixture_dataset(max_tokens=18)
    # Dense immediate history must be cropped by one fixed budget in both modes.
    song.points = [EventPoint(Fraction(i), str(i % 8 + 1)) for i in range(16)] + [
        EventPoint(Fraction(16), '2'), EventPoint(Fraction(31), '3')]
    training = data.build_window(song, song.points, 16, training=True)
    inference = data.build_window(song, song.points, 16, training=False)
    prefix_length = int((training['labels'] != -100).nonzero()[0]) + 1
    assert inference['input_ids'].tolist() == training['input_ids'][:prefix_length].tolist()
    assert len(inference['input_ids']) <= data.max_tokens // 2


def test_padding_masks_and_bucket_coverage():
    data, song = fixture_dataset()
    items = [data.build_window(song, song.points, start, training=True) for start in (0, 16)]
    batch = collate_composer_v2(items)
    assert batch['input_ids'].shape == batch['labels'].shape == batch['token_mask'].shape
    assert torch.all(batch['labels'][~batch['token_mask']] == -100)
    sampler = LengthBucketBatchSampler(data, 1, seed=5)
    assert sorted(i for group in sampler for i in group) == [0, 1]


def test_audio_local_batch_order_keeps_exact_epoch_coverage_and_resume_cursor():
    audio = [SimpleNamespace(audio=SimpleNamespace(identity={'audio_sha256': str(index // 2)}))
             for index in range(96)]
    windows = [(index, offset) for index in range(len(audio)) for offset in range(index % 5 + 1)]

    class SizedDataset(SimpleNamespace):
        def __len__(self):
            return len(self.windows)

    dataset = SizedDataset(songs=audio, windows=windows, lengths=[10 + i % 7 for i in range(len(windows))])
    sampler = LengthBucketBatchSampler(dataset, 7, seed=123, order=AUDIO_LOCAL_BATCH_ORDER)
    first = list(sampler)
    assert len(first) == len(sampler)
    assert all(len(batch) == 7 for batch in first[:-1])
    assert sorted(index for batch in first for index in batch) == list(range(len(windows)))
    # Complete batches in the first block can only draw from its 16 sources.
    assert len({windows[index][0] // 2 for batch in first[:8] for index in batch}) <= 16
    sampler.start_batch = 9
    assert list(sampler) == first[9:]
    sampler.start_batch = 0
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_resume_seeks_to_saved_batch_without_fetching_consumed_samples():
    from torch.utils.data import DataLoader
    from maichart.composer_v2.train import _batches

    class Dataset:
        lengths = [3, 1, 2, 4, 5, 3, 1, 2]

        def __init__(self):
            self.fetched = []

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, index):
            self.fetched.append(int(index))
            return int(index)

    dataset = Dataset()
    sampler = LengthBucketBatchSampler(dataset, 2, seed=5)
    sampler.set_epoch(3)
    expected = list(sampler)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=list)
    assert list(_batches(loader, skip=2)) == list(enumerate(expected[2:], start=2))
    assert dataset.fetched == [int(index) for batch in expected[2:] for index in batch]
    assert list(sampler) == expected


def test_real_tempo_and_offset_determine_audio_queries():
    data, song = fixture_dataset()
    seen = []
    original = song.audio.sample
    song.audio.sample = lambda seconds: (seen.append(seconds.copy()) or original(seconds))
    data.build_window(song, song.points, 16, training=True)
    # Local node zero is beat 0, not audio zero, despite tempo changing inside window.
    assert seen[-1][0] == pytest.approx(.75)
    assert song.clock.at(Fraction(24)) == pytest.approx(14.75)


def prepare_fixture(data, song):
    class Store:
        def prepare(self, path):
            return song.audio
        def close(self):
            pass
    data.audio_store = Store()
    data.audit = {'skipped': []}


def test_authored_end_after_audio_only_caps_empty_tail_not_source_notes():
    data, song = fixture_dataset()
    song.authored_end_beat = Fraction(1000)
    song.end_beat = Fraction(1000)
    song.audio.duration_sec = 30.
    original = list(song.points)
    prepare_fixture(data, song)
    data.prepare()
    assert song.end_beat < song.authored_end_beat
    assert song.points == original
    assert data.audit['audio_extents'][0]['authored_end_beat'] == '1000'


def test_audio_end_onset_rejects_whole_chart_instead_of_teaching_rest():
    data, song = fixture_dataset()
    song.audio.duration_sec = song.clock.at(Fraction(31))
    prepare_fixture(data, song)
    with pytest.raises(ValueError, match='No usable'):
        data.prepare()
    assert data.songs == []
    assert data.audit['skipped'][0]['reason'] == 'authored_onset_at_or_after_exclusive_audio_end'


def test_sustain_end_risk_is_reported_without_rejecting_authored_duration():
    data, song = fixture_dataset()
    song.audio.duration_sec = 30.
    song.sustain = SimpleNamespace(markers=[{'end_second': 30.1}])
    prepare_fixture(data, song)
    data.prepare()
    assert data.songs == [song]
    assert data.audit['sustain_audio_end_risks'][0]['guard_tolerance_sec'] == 1e-8


def test_training_condition_uses_cached_flux_on_the_real_audio_first():
    data, song = fixture_dataset()
    times = np.arange(0., 200., .01)
    song.audio.mel_times_sec = times
    song.audio.mel = np.zeros((len(times), 98), dtype=np.float32)
    for point in song.points:
        song.audio.mel[int(round(song.clock.at(point.beat) / .01)), 96] = 1.
    prepare_fixture(data, song)
    data.prepare()
    from maichart.composer_v2.conditions import CONDITION_KEYS
    index = CONDITION_KEYS.index('melody_following')
    assert song.condition_known[index] and song.condition[index] == 1.
