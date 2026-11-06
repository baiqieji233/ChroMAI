"""Small CPU/cache contract checks; no model download or MERT execution."""
import hashlib
import json

import numpy as np
import pytest

from maichart.composer_v2.audio import AudioFeatures, AudioFeatureStore


def _features(backend='mert'):
    times = np.array([0., 1., 2.])
    mel = np.repeat(np.array([0., 10., 20.])[:, None], 98, axis=1)
    semantic = np.repeat(np.array([0., 2., 4.])[:, None], 1024, axis=1)
    return AudioFeatures(mel, times, semantic, times, 2., {'backend': backend})


def _write_bank(store, identity):
    bank = store._directory(identity)
    bank.mkdir(parents=True)
    features = _features(identity['backend'])
    for key in ('mel', 'mel_times_sec', 'semantic', 'semantic_times_sec'):
        np.save(bank / (key + '.npy'), getattr(features, key), allow_pickle=False)
    (bank / 'index.json').write_text(json.dumps(dict(identity=identity, duration_sec=2.)), encoding='utf-8')
    return bank


def test_sampling_uses_real_timestamps_and_masks_outside_song():
    mel, semantic, valid = _features().sample(np.array([.25, 1.5, 3.]))
    np.testing.assert_allclose(mel[:, 0], [2.5, 15., 20.])
    np.testing.assert_allclose(semantic[:, 0], [.5, 3., 4.])
    np.testing.assert_array_equal(valid, [True, True, False])


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_device_interpolation_preserves_numpy_results_exactly(dtype):
    import torch

    rng = np.random.default_rng(81)
    times = np.cumsum(rng.uniform(.01, .03, 500))
    values = rng.normal(size=(500, 1024)).astype(dtype)
    seconds = np.concatenate(([-1., times[-1] + 1.], times[10:20], rng.uniform(0, times[-1], 150)))
    right = np.searchsorted(times, seconds, side='right')
    left = np.clip(right - 1, 0, len(times) - 1)
    right = np.clip(right, 0, len(times) - 1)
    weight = np.clip((seconds - times[left]) / np.maximum(times[right] - times[left], 1e-12), 0., 1.)
    expected = np.asarray(values[left], dtype=np.float32) * (1. - weight[:, None])
    expected += np.asarray(values[right], dtype=np.float32) * weight[:, None]
    features = AudioFeatures(np.zeros((500, 98), np.float32), times, values, times, times[-1], {'backend':'mert'})
    _, left, right, weight, valid = features.sample_for_training(seconds)
    for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        a = torch.as_tensor(left, device=device).float().double()
        b = torch.as_tensor(right, device=device).float().double()
        w = torch.as_tensor(weight, device=device)[:, None]
        actual = (a * (1. - w) + b * w).float().cpu().numpy()
        np.testing.assert_array_equal(actual, expected.astype(np.float32))
    np.testing.assert_array_equal(valid, features.sample(seconds)[2])


def test_native_semantic_mask_is_unavailable():
    features = _features('native')
    assert not features.sample(np.array([.5]))[2].any()
    assert not features.summary(2)['semantic_valid'].any()


def test_summary_preserves_song_timeline():
    result = _features().summary(max_tokens=2)
    np.testing.assert_allclose(result['times_sec'], [.5, 1.5])
    np.testing.assert_allclose(result['position'], [.25, .75])
    assert result['mel'].shape == (2, 98)
    assert result['semantic'].shape == (2, 1024)


def test_load_never_builds_missing_cache(tmp_path):
    source = tmp_path / 'audio.bin'
    source.write_bytes(b'audio identity fixture')
    store = AudioFeatureStore(tmp_path / 'cache', backend='native')
    with pytest.raises(FileNotFoundError, match='offline prepare'):
        store.load(source)
    assert not (tmp_path / 'cache').exists()


def test_cache_key_tracks_audio_content_and_backend(tmp_path):
    source = tmp_path / 'audio.bin'
    source.write_bytes(b'first')
    store = AudioFeatureStore(tmp_path / 'cache', backend='native')
    first = store._identity(source)
    source.write_bytes(b'second')
    second = store._identity(source)
    assert store._directory(first) != store._directory(second)
    mert = AudioFeatureStore(tmp_path / 'cache', backend='mert')
    assert mert._directory(mert._identity(source)) != store._directory(second)


def test_evicted_bank_reference_reopens_the_same_values(tmp_path, monkeypatch):
    from maichart.composer_v2 import audio

    monkeypatch.setattr(audio, 'MAX_OPEN_BANKS', 1)
    store = AudioFeatureStore(tmp_path)
    first = _features()
    reference = store.reference(first)
    opened = []

    def read(directory, identity):
        opened.append(directory)
        return first if directory == reference.directory else _features()

    monkeypatch.setattr(store, '_read', read)
    expected = reference.sample([.5])[1]
    store._mapped(tmp_path / 'another_bank', first.identity)
    assert store.mapping_stats()['open_banks'] == 1
    np.testing.assert_array_equal(reference.sample([.5])[1], expected)
    assert opened == [reference.directory, tmp_path / 'another_bank', reference.directory]


def test_zero_audio_budget_skips_retaining_bank_mappings(tmp_path, monkeypatch):
    store = AudioFeatureStore(tmp_path, max_mapped_bytes=0)
    features = _features()
    monkeypatch.setattr(store, '_read', lambda directory, identity: features)
    reference = store.reference(features)
    np.testing.assert_array_equal(reference.sample([.5])[1], features.sample([.5])[1])
    assert store.mapping_stats()['open_banks'] == 0
    assert store.mapping_stats()['mapped_bytes'] == 0


def test_mapping_count_can_exceed_16_and_shrinks_on_host_pressure(tmp_path, monkeypatch):
    from maichart.composer_v2 import audio

    gib = 2**30
    memory = dict(free_physical=12*gib, free_commit=12*gib)
    monkeypatch.setattr(audio, 'host_memory', lambda: memory)
    store = AudioFeatureStore(tmp_path, max_mapped_bytes=gib)
    features = _features()
    monkeypatch.setattr(store, '_read', lambda directory, identity: features)
    for i in range(32):
        store._mapped(tmp_path / str(i), features.identity)
    assert store.mapping_stats()['open_banks'] == 32
    store._mapped(tmp_path / '0', features.identity)
    assert store.mapping_stats()['map_hits'] == 1

    memory['free_physical'] = gib
    store._memory_check_at = 0.
    store._mapped(tmp_path / '0', features.identity)
    stats = store.mapping_stats()
    assert stats['effective_budget_bytes'] == stats['mapped_bytes'] == stats['open_banks'] == 0
    assert stats['evictions'] == 32


def test_reopen_reuses_validation_but_rechecks_changed_timeline(tmp_path):
    store = AudioFeatureStore(tmp_path, max_mapped_bytes=0)
    identity = {'backend': 'mert'}
    bank = _write_bank(store, identity)
    first = store._mapped(bank, identity).sample([.5])[1]
    np.testing.assert_array_equal(store._mapped(bank, identity).sample([.5])[1], first)
    assert store.mapping_stats()['bank_opens'] == 2
    assert store.mapping_stats()['bank_validations'] == 1

    bad_times = bank / 'new_times.npy'
    np.save(bad_times, np.array([0., 2., 1.]), allow_pickle=False)
    bad_times.replace(bank / 'mel_times_sec.npy')
    with pytest.raises(ValueError, match='Invalid mel cache timeline'):
        store._mapped(bank, identity)


def test_prepare_reuses_persisted_audio_sha_without_changing_identity(tmp_path, monkeypatch):
    from maichart.composer_v2 import audio
    from maichart.form.cache_support import SourceDigests

    monkeypatch.setattr(audio, 'source_digests', SourceDigests)
    source = tmp_path / 'audio.bin'
    source.write_bytes(b'original bytes')
    root = tmp_path / 'cache'
    first_store = AudioFeatureStore(root, backend='native')
    identity = first_store._identity(source)
    assert identity['audio_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    _write_bank(first_store, identity)
    assert first_store.prepare(source).identity == identity
    assert first_store._source_digests.stats()['full_reads'] == 1

    second_store = AudioFeatureStore(root, backend='native')
    assert second_store.prepare(source).identity == identity
    assert second_store._source_digests.stats()['full_reads'] == 0
    source.write_bytes(b'changed bytes')
    assert second_store._identity(source)['audio_sha256'] != identity['audio_sha256']
    assert second_store._source_digests.stats()['full_reads'] == 1


def test_summary_sidecar_reuses_identical_arrays_and_rebuilds_after_corruption(tmp_path):
    identity = {'backend': 'mert'}
    store = AudioFeatureStore(tmp_path, max_mapped_bytes=0)
    bank = _write_bank(store, identity)
    prepared = store._mapped(bank, identity)
    reference = store.reference(prepared)
    expected = _features().summary(160)
    for key, array in reference.summary().items():
        np.testing.assert_array_equal(array, expected[key])
    assert store.mapping_stats()['bank_opens'] == 1
    del prepared

    another = AudioFeatureStore(tmp_path, max_mapped_bytes=0)
    reference = another.reference(_features())
    for key, array in reference.summary().items():
        np.testing.assert_array_equal(array, expected[key])
    assert another.mapping_stats()['bank_opens'] == 0
    assert another.mapping_stats()['summary_hits'] == 1
    (bank / 'summary_160_v1.npz').write_bytes(b'')
    for key, array in reference.summary().items():
        np.testing.assert_array_equal(array, expected[key])
    assert another.mapping_stats()['bank_opens'] == 1

    updated = np.load(bank / 'semantic.npy') * 2
    replacement = bank / 'replacement.npy'
    np.save(replacement, updated, allow_pickle=False)
    replacement.replace(bank / 'semantic.npy')
    np.testing.assert_array_equal(reference.summary()['semantic'], expected['semantic'] * 2)
    assert another.mapping_stats()['summary_hits'] == 1
    assert another.mapping_stats()['bank_opens'] == 2


def test_mert_extraction_stops_at_the_captured_layer(tmp_path, monkeypatch):
    import torch

    calls = []

    class Layer(torch.nn.Module):
        def forward(self, state):
            calls.append(self)
            return (state + 1.,)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList(Layer() for _ in range(24))

        def forward(self, input_values, **kwargs):
            state = input_values
            for layer in self.encoder.layers:
                state = layer(state)[0]
            return state

    def processor(chunk, **kwargs):
        frames = (len(chunk) - 400) // 320 + 1
        return {'input_values': torch.zeros((1, frames, 1024))}

    model = Model()
    store = AudioFeatureStore(tmp_path)
    monkeypatch.setattr(store, '_encoder', lambda: (model, processor))
    store._extract_semantic(torch.zeros(720), .03, tmp_path)
    assert calls == list(model.encoder.layers[:8])
    np.testing.assert_array_equal(np.load(tmp_path / 'semantic.npy'), np.full((2, 1024), 8., np.float16))
    assert not model.encoder.layers[7]._forward_hooks
