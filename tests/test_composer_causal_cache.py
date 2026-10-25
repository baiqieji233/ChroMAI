"""Actual preparation cache persistence of exact causal source positions."""
import copy
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from maichart.composer.prefix_state import committed_prefix_states
from maichart.form.preparation_cache import _decode
from maichart.generation.sustain_state import SustainStateIndex
from test_form_preparation_cache import cached_source


def test_preparation_cache_exact_source_fraction_and_legacy_absence(cached_source):
    build,chart,_,_ = cached_source
    chart.write_text(chart.read_text(encoding='utf-8').replace('{12}1,2,3,','{12}1,2h[4:4],3,'),encoding='utf-8')
    cold = build()
    assert cold.preparation_cache_stats['write_errors'] == 0
    root = Path(cold.preparation_cache_stats['path'])
    index_path = next(root.glob('*.index.json'))
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    metadata = payload['result']
    assert metadata['state_markers']
    assert all(isinstance(marker['source_beat'],str) for marker in metadata['state_markers'])
    with np.load(index_path.with_name(index_path.name.replace('.index.json','.'+payload['arrays_sha256']+'.npz')),allow_pickle=False) as source:
        arrays = {key:source[key].copy() for key in source.files}
    restored = _decode(metadata,arrays)['song'][4]
    assert all(isinstance(marker['source_beat'],Fraction) for marker in restored.markers)
    assert any(marker['source_beat'].denominator == 3 for marker in restored.markers)
    points = cold.songs[0][1]
    cutoff = len(points)
    query = [Fraction(2),Fraction(3)]
    torch.testing.assert_close(committed_prefix_states(restored,query,committed_events=cutoff),
        committed_prefix_states(cold.songs[0][4],query,committed_events=cutoff),rtol=0,atol=0)
    assert build().preparation_cache_stats['hits'] == 1
    legacy = copy.deepcopy(metadata)
    for marker in legacy['state_markers']:
        marker.pop('source_beat')
    old = _decode(legacy,arrays)['song'][4]
    assert all('source_beat' not in marker for marker in old.markers)
    # Legacy checkpoints still query without causal cutoff; new APIs fail closed
    # until Composer's lazy table explicitly rebuilds the exact-source index.
    old.at(Fraction(2),include_current=True)
    with pytest.raises(ValueError,match='rebuilt index'):
        committed_prefix_states(old,query,committed_events=cutoff)
    rebuilt = SustainStateIndex(old.points,old.tempo_changes,old.first)
    torch.testing.assert_close(committed_prefix_states(rebuilt,query,committed_events=cutoff),
                               committed_prefix_states(restored,query,committed_events=cutoff))
