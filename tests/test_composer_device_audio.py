"""Reference equivalence; these tests intentionally support CPU-only CI."""
import json

import numpy as np
import pytest
import torch

from maichart.composer.acoustics import AcousticContext
from maichart.composer.audio_lens import LensBank
from maichart.composer.device_audio import AudioQueryMaterializer, _DeviceBank


def _context(times):
    times = np.asarray(times, dtype=np.float64)
    values = np.random.default_rng(41).normal(size=(len(times), 120)).astype(np.float32)
    context = AcousticContext(values, times, 1.01)
    context.lens_bank = LensBank(values, times, 1.01)
    return context


@pytest.mark.parametrize('times', [[], [.13], [0., .071, .19, .400000001, .73, 1.01]])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_device_queries_preserve_irregular_clock_edges_and_reference(times, device):
    context = _context(times)
    bank = _DeviceBank(context, device)
    seconds = np.array([-.3, 0., .13, .19, .400000001, .73, 1.01, 1.4], np.float64)
    query = torch.tensor(seconds, device=device)
    for method in ('at', 'trailing'):
        actual = getattr(bank, method)(query)
        assert actual.dtype == torch.float32
        np.testing.assert_allclose(actual.cpu(), getattr(context, method)(seconds), atol=2e-6, rtol=2e-6)
    for previous in (-1., .13, .73, 2.):
        actual = bank.interval(query.new_tensor(previous), query)
        np.testing.assert_allclose(actual.cpu(), context.interval(previous, seconds), atol=2e-6, rtol=2e-6)
    patches, valid = bank.lens(query)
    expected, mask = context.lens_bank.query(seconds)
    np.testing.assert_array_equal(valid.cpu(), mask)
    np.testing.assert_allclose(patches.cpu(), expected, atol=2e-6, rtol=2e-6)


def test_materializer_deduplicates_rows_and_bounds_lru():
    a, b = _context([0., .1, .4]), _context([0., .2, .6])
    size = _DeviceBank(a, 'cpu').nbytes
    materializer = AudioQueryMaterializer(budget_bytes=size)
    first = materializer._get(a, 'cpu')
    assert materializer._get(a, 'cpu') is first
    materializer._get(b, 'cpu')
    assert materializer.resident_bytes <= size
    assert len(materializer.cache) == 1
    oversized = AudioQueryMaterializer(budget_bytes=1)
    oversized._get(a, 'cpu')
    assert oversized.resident_bytes == 0 and not oversized.cache
    batch = {'_audio_banks': (a, b), '_audio_bank_rows': ((0, 2), (1,)),
             '_audio_music': True, '_audio_grounded': True}
    seconds = torch.tensor([[.1, .2, .3], [.4, .5, .6], [.2, .3, .4]], dtype=torch.float64)
    tensors = {'audio_query_seconds': seconds, 'audio_previous_seconds': torch.tensor([0., .2, .1], dtype=torch.float64),
               'candidate_features': torch.zeros(3, 2, 120)}
    assert materializer.materialize(batch, tensors) is tensors
    for row, context in enumerate((a, b, a)):
        np.testing.assert_allclose(tensors['workload_audio'][row], context.at(seconds[row].numpy()), atol=2e-6)
        np.testing.assert_allclose(tensors['interval_audio'][row], context.interval(
            tensors['audio_previous_seconds'][row], seconds[row, :2].numpy()), atol=2e-6)


def test_dataset_deferred_collate_materialize_matches_dense_real_chain(tmp_path):
    import soundfile as sf
    from maichart.composer.data import ComposerDataset, collate_composer
    from maichart.composer.model import GROUNDED_PROFILE
    sr = 8000
    clock = np.arange(sr*2)/sr
    sf.write(tmp_path/'track.wav', (.1*np.sin(2*np.pi*440*clock)).astype(np.float32), sr)
    (tmp_path/'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}1,2,3,4,5,6,7,8,E\n', encoding='utf8')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs': [{'song_id': 'device-audio-fixture',
        'maidata_path': str(tmp_path/'maidata.txt'), 'audio_path': str(tmp_path/'track.wav'),
        'difficulties': [{'difficulty_index': 5, 'level_raw': '12'}]}]}), encoding='utf8')
    dataset = ComposerDataset(manifest, cache_dir=tmp_path/'cache', candidate_count=16,
        history_events=8, action_slots=4, preparation_workers=1, architecture_profile=GROUNDED_PROFILE)
    dense = collate_composer([dataset[0], dataset[1], dataset[0]])
    dataset.defer_audio_queries = True
    batch = collate_composer([dataset[0], dataset[1], dataset[0]])
    assert 'audio_lens' not in batch and 'workload_audio' not in batch
    assert len(batch['_audio_banks']) == 1
    tensors = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    AudioQueryMaterializer().materialize(batch, tensors)
    for key, expected in dense.items():
        if torch.is_tensor(expected):
            torch.testing.assert_close(tensors[key], expected, atol=2e-6, rtol=2e-6)
    assert batch['planner_unique_indices'].tolist() == [0]
    assert batch['planner_inverse'].tolist() == [0, 0, 0]
