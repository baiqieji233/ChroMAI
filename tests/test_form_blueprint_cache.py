import torch
from fractions import Fraction
from types import SimpleNamespace


def test_blueprint_input_cache_roundtrip_and_identity(tmp_path):
    from maichart.form.blueprint_cache import BlueprintInputCache

    cache = BlueprintInputCache(tmp_path)
    chart = {'source': 'song/maidata.txt', 'difficulty': 5,
             'chart_sha256': 'chart', 'audio_sha256': 'audio'}
    key, identity = cache.identity(
        chart, audio_profile='rational_audio_queries_v1', max_tokens=2048,
        max_memory_tokens=640, max_region_seconds=14.,
        tempo_map=[['0', 120.]], first=0.25,
        condition=[0.5, 0.1], quality={'weight': 1.}, curve_sha256='curve')
    inputs = {'audio': torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
              'form': torch.ones((1, 2, 7)), 'condition': torch.zeros((1, 2))}
    targets = {'load_curve_target': torch.full((1, 3, 4), .25),
               'load_curve_mask': torch.ones((1, 3, 4), dtype=torch.bool),
               'frame_weight': torch.ones((1, 3))}
    assert cache.load(key, identity) is None
    assert cache.save(key, identity, inputs, targets, 1.5) is None
    result = cache.load(key, identity)
    assert result['_weight'] == 1.5
    for name, value in {**{f'input_{k}': v for k, v in inputs.items()},
                        **{f'target_{k}': v for k, v in targets.items()}}.items():
        torch.testing.assert_close(result[name], value)
    assert cache.stats()['hits'] == 1


def test_blueprint_input_cache_rejects_changed_tempo_or_first(tmp_path):
    from maichart.form.blueprint_cache import BlueprintInputCache

    cache = BlueprintInputCache(tmp_path)
    chart = {'source': 'song/maidata.txt', 'difficulty': 5,
             'chart_sha256': 'chart', 'audio_sha256': 'audio'}
    common = dict(audio_profile='rational_audio_queries_v1', max_tokens=2048,
                  max_memory_tokens=640, max_region_seconds=14.,
                  tempo_map=[['0', 120.]], first=0., condition=[1.])
    key, identity = cache.identity(chart, **common)
    cache.save(key, identity, {'audio': torch.zeros((1, 1, 120))},
               {'load_curve_target': torch.zeros((1, 1, 4)),
                'load_curve_mask': torch.ones((1, 1, 4), dtype=torch.bool),
                'frame_weight': torch.ones((1, 1))}, 1.)
    changed = dict(common, first=.5)
    changed_key, changed_identity = cache.identity(chart, **changed)
    assert changed_key != key
    assert cache.load(changed_key, changed_identity) is None


def test_identity_for_dataset_tracks_timing_condition_quality_and_curve_without_audio(tmp_path):
    from maichart.event_timing import EventTempo
    from maichart.form.blueprint_cache import BlueprintInputCache

    class Span:
        def __init__(self, start, end):
            self.start, self.end = Fraction(start), Fraction(end)
        def to_dict(self):
            return {'start': str(self.start), 'end': str(self.end)}

    span = Span(0, 4)
    song = SimpleNamespace(
        tempo=[EventTempo(Fraction(0), 120.)], first=0., meters=[],
        spans=[span], regions=[span], load_curve_times_sec=torch.tensor([0., 1.]).numpy(),
        load_curve=torch.zeros((2, 4)).numpy(),
        relations=SimpleNamespace(descriptors=torch.zeros((1, 504)).numpy()),
        audio_profile='rational_audio_queries_v1',
        audio=SimpleNamespace(__getattr__=lambda *_: (_ for _ in ()).throw(AssertionError('lazy audio touched'))),
    )
    condition = SimpleNamespace(to_vector=lambda: torch.tensor([0.5, 1.]))
    record = {'label': 'excellent', 'weight': 1.5, 'regions': [{'start_sec': 0., 'end_sec': 1., 'weight': 2.}]}
    dataset = SimpleNamespace(
        songs=[(song, [], 13., 1.5, None, condition, [(0., 1.)], record)],
        audit={'charts': [{'source': 'song/maidata.txt', 'difficulty': 5,
                           'chart_sha256': 'chart', 'audio_sha256': 'audio'}]},
        max_memory_tokens=640,
    )
    cache = BlueprintInputCache(tmp_path)
    original, _ = cache.identity_for_dataset(dataset, 0, max_tokens=2048, max_region_seconds=14.)
    song.first = .5
    changed_first, _ = cache.identity_for_dataset(dataset, 0, max_tokens=2048, max_region_seconds=14.)
    assert original != changed_first
    song.first = 0.
    record['weight'] = 2.
    changed_quality, _ = cache.identity_for_dataset(dataset, 0, max_tokens=2048, max_region_seconds=14.)
    assert original != changed_quality
