"""CPU contract tests for versioned musical-evidence data and caches."""
from fractions import Fraction as F
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form import data
from maichart.form.curve import build_load_curve, encode_density, decode_density
from maichart.form.quality_profile import MUSICAL_EVIDENCE
from maichart.form.timing import build_spans
from test_form_pipeline import fixture_audio


def _manifest(tmp_path):
    folder = tmp_path / 'song'
    folder.mkdir()
    chart = folder / 'maidata.txt'
    chart.write_text('&title=fixture\n&wholebpm=120\n&lv_5=13\n'
                     '&inote_5=(120){8}1,2,3,4,5,6,7,8,1h[4:2],2,3,4,E\n', encoding='utf-8')
    audio = folder / 'track.wav'
    audio.write_bytes(b'synthetic audio fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [{'song_id': 'fixture',
        'maidata_path': 'song/maidata.txt', 'audio_path': 'song/track.wav',
        'difficulties': [{'difficulty_index': 5, 'level_raw': '13'}]}]}), encoding='utf-8')
    return manifest


@pytest.fixture
def dataset_builder(tmp_path, monkeypatch):
    from maichart.form.preparation_cache import PreparationCache
    monkeypatch.setattr(PreparationCache, '_legacy_roots', lambda self: [self.root])
    manifest = _manifest(tmp_path)

    def loader(*args, **kwargs):
        return fixture_audio(4)

    monkeypatch.setattr(data, 'extract_event_audio', loader)

    def build(**kwargs):
        return data.FormDataset(manifest, cache_dir=tmp_path / 'cache', audio_loader=loader,
                                max_tokens=256, max_region_seconds=3, **kwargs)
    return build


def test_new_density_scale_distinguishes_densities_above_eight_heads():
    nps = np.array([0., 2., 8., 16., 64., 256.])
    encoded = encode_density(nps, 'musical_log_v1')
    assert encoded[2] == pytest.approx(.5)
    assert encoded[3] == pytest.approx(2/3)
    assert np.all(np.diff(encoded) > 0)
    np.testing.assert_allclose(decode_density(encoded, 'musical_log_v1'), nps)
    np.testing.assert_array_equal(encode_density([8., 16.], 'legacy_linear_v1'), [1., 1.])


@pytest.mark.parametrize('first', [-1., 0., 1.])
def test_enhanced_curve_uses_one_second_bins_and_audio_bounds(first):
    tempo = [EventTempo(F(0), 60)]
    points = [EventPoint(F(index, 16), '1') for index in range(64)]
    times, values = build_load_curve(points, tempo, first, 4., quality_profile=MUSICAL_EVIDENCE)
    assert len(times) == int(4-first)
    assert values.shape == (len(times), 4)
    positive = values[:, 0] > 0
    assert np.all(values[positive, 0] < 1)
    assert np.all(np.isfinite(values))
    assert values[positive, 0].max() == pytest.approx(2/3)


def test_enhanced_song_descriptor_and_bounds_keep_targets_outside_inputs():
    song = data.organize_song(fixture_audio(4), [EventTempo(F(0), 120)], .25,
        spans=build_spans(F(4), [], bars_per_span=1), quality_profile=MUSICAL_EVIDENCE)
    example = data.input_example(song, 0, [])
    assert song.curve_bin_seconds == 1.
    assert example['form'].shape[-1] == 515
    assert len(example['audio_times_sec']) == len(example['audio'])
    torch.testing.assert_close(example['span_bounds'], torch.tensor([4/64, 2/20]))
    assert 'position_alignment_target' not in example
    assert 'alignment_event_indices' not in example
    assert example['skeleton'].tolist() == [0]


def test_cache_profiles_are_distinct_and_warm_restore_full_layout_descriptors(dataset_builder):
    legacy = dataset_builder()
    legacy_warm = dataset_builder(quality_profile='legacy', curve_bin_seconds=2.)
    assert legacy_warm.preparation_cache_stats['hits'] == 1
    upgraded = dataset_builder(quality_profile=MUSICAL_EVIDENCE)
    assert upgraded.preparation_cache_stats['hits'] == 0
    assert upgraded.preparation_cache_stats['rebuilt'] == 1
    assert upgraded.songs[0][0].curve_bin_seconds == 1.
    assert legacy.songs[0][0].curve_bin_seconds == 2.
    warm = dataset_builder(quality_profile=MUSICAL_EVIDENCE)
    assert warm.preparation_cache_stats['hits'] == 1
    np.testing.assert_array_equal(warm.songs[0][0].relations.layout_descriptors,
                                  upgraded.songs[0][0].relations.layout_descriptors)
    assert warm.songs[0][5].metadata['duration_semantics_profile'] == 'miacode_head_bpm_v1'
    for index in range(len(upgraded)):
        original, restored = upgraded[index], warm[index]
        for key, value in original.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, restored[key], rtol=0, atol=0)
    changed = dataset_builder(quality_profile=MUSICAL_EVIDENCE, curve_bin_seconds=.5)
    assert changed.preparation_cache_stats['hits'] == 0


def test_alignment_targets_collate_separately_from_audio_and_have_false_padding(dataset_builder):
    dataset = dataset_builder(quality_profile=MUSICAL_EVIDENCE)
    first = dataset[0]
    shorter = dict(first)
    for key in ('audio', 'audio_times_sec', 'position_alignment_target', 'position_alignment_mask'):
        shorter[key] = first[key][:3]
    for key in ('alignment_event_indices', 'alignment_audio_index', 'alignment_event_mask'):
        shorter[key] = first[key][:1]
    batch = data.collate_form([first, shorter])
    assert batch['audio_times_sec'].shape == batch['position_alignment_target'].shape
    assert batch['audio_times_sec_padding_mask'][1, 3:].all()
    assert not batch['position_alignment_mask'][1, 3:].any()
    assert not batch['alignment_event_mask'][1, 1:].any()
    assert batch['alignment_event_indices'][1, 1:].eq(-1).all()
    assert batch['span_bounds'].shape == (2, 2)
    assert batch['position_alignment_target'].max() > 0


def test_blueprint_cache_identity_tracks_quality_layout_and_bin_without_changing_legacy(dataset_builder, tmp_path):
    from maichart.form.blueprint_cache import BlueprintInputCache
    cache = BlueprintInputCache(tmp_path / 'blueprints')
    legacy = dataset_builder()
    old_key, old_identity = cache.identity_for_dataset(legacy, 0)
    assert 'quality_profile' not in old_identity
    assert 'curve_bin_seconds' not in old_identity
    current = dataset_builder(quality_profile=MUSICAL_EVIDENCE)
    key, identity = cache.identity_for_dataset(current, 0)
    assert key != old_key
    assert identity['quality_profile'] == MUSICAL_EVIDENCE
    assert identity['density_profile'] == 'musical_log_v1'
    assert identity['audio_evidence_profile'] == 'balanced_evidence_v1'
    current.songs[0][0].relations.layout_descriptors[0, -1] += .1
    assert cache.identity_for_dataset(current, 0)[0] != key


def test_audio_store_keeps_native_loader_identity_and_passes_evidence_profile(tmp_path, monkeypatch):
    from maichart.form import rational_audio
    source = tmp_path / 'source.wav'
    source.write_bytes(b'fixture')
    calls = []

    def loader(path, cache_dir=None, *, evidence_profile='legacy'):
        calls.append(('extract', evidence_profile))
        return fixture_audio(1)

    def path_for(path, cache_dir, *, evidence_profile='legacy'):
        calls.append(('path', evidence_profile))
        return tmp_path / 'absent.npz'

    def candidates(path, cache_dir, *, evidence_profile='legacy'):
        calls.append(('candidates', evidence_profile))
        return []

    monkeypatch.setattr(rational_audio, 'extract_rational_audio', loader)
    monkeypatch.setattr(rational_audio, 'cache_path', path_for)
    monkeypatch.setattr(rational_audio, '_cache_candidates', candidates)
    store = data._AudioStore(loader, tmp_path, quality_profile=MUSICAL_EVIDENCE)
    store.get(source)
    assert store.loader is rational_audio.extract_rational_audio
    assert calls == [('path', 'balanced_evidence_v1'), ('candidates', 'balanced_evidence_v1'),
                     ('extract', 'balanced_evidence_v1')]


def test_enhanced_alignment_tolerance_is_validated(dataset_builder):
    with pytest.raises(ValueError, match='position_alignment_tolerance_ms'):
        dataset_builder(quality_profile=MUSICAL_EVIDENCE, position_alignment_tolerance_ms=0)


def test_learned_curve_passes_enhanced_clock_and_exports_density_units(tmp_path, monkeypatch):
    from maichart.form import checkpoint, rational_audio
    from maichart.form.codec import RationalCodec
    from maichart.form.curve import generate_learned_profile
    source, model_path = tmp_path / 'audio.wav', tmp_path / 'model.pt'
    source.write_bytes(b'fixture source identity')
    model_path.write_bytes(b'fixture checkpoint identity')
    audio = fixture_audio(4)
    extractions = []

    def extract(path, cache_dir=None, *, evidence_profile='legacy'):
        extractions.append(evidence_profile)
        return audio

    class Planner:
        def eval(self):
            pass

        def __call__(self, values, form, condition, *, audio_times_sec):
            assert form.shape[-1] == 515
            torch.testing.assert_close(audio_times_sec, torch.tensor([[1., 3.]], dtype=torch.float64))
            density = torch.tensor([.5, 2/3])
            return {'load_curve': torch.logit(density)[None, :, None].expand(1, 2, 4)}

    payload = {'family': 'song_form_v2', 'codec': RationalCodec().to_dict(),
               'optimizer_steps': 1, 'blueprint_steps': 1,
               'capabilities': {'load_curve': {'steps': 1, 'contract': checkpoint.INPUT_CONTRACT['load_curve']}}}
    model = SimpleNamespace(config=SimpleNamespace(quality_profile=MUSICAL_EVIDENCE,
                                                   curve_bin_seconds=1., rhythm_geometry_profile='none'))
    monkeypatch.setattr(checkpoint, 'load', lambda *args, **kwargs: (model, Planner(), None, payload))
    monkeypatch.setattr(rational_audio, 'extract_rational_audio', extract)
    monkeypatch.setattr(rational_audio, 'planner_audio_queries',
                        lambda song: (torch.zeros((2, 120)), np.array([1., 3.]), [F(2), F(6)]))
    profile = generate_learned_profile(source, model_path, bpm=120, resolution_sec=2.)
    assert extractions == ['balanced_evidence_v1']
    assert profile['quality_profile'] == MUSICAL_EVIDENCE
    assert profile['curve_bin_seconds'] == 1.
    assert profile['density_profile'] == 'musical_log_v1'
    at = {point['time_sec']: point['intensity'] for point in profile['points']}
    assert at[1.] == pytest.approx(1/3)
    assert at[3.] == pytest.approx(2/3)
