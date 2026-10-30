"""Numbered center Touch aliases must not crash causal head-table preparation."""
from fractions import Fraction
import json

import numpy as np
import pytest

from maichart.composer.events import event_metadata, head_pad_indices_for_policy
from maichart.generation.muri.parser import parse_markers


@pytest.mark.parametrize('lane', range(1, 9))
@pytest.mark.parametrize('suffix', ['', 'f', 'h[16:1]', 'fh[16:1]'])
def test_center_alias_uses_same_physical_pad_as_native_judge(lane, suffix):
    alias, center = f'C{lane}{suffix}', f'C{suffix}'
    for policy in ('merged_pads_v1', 'separate_pads_v2'):
        assert head_pad_indices_for_policy(event_metadata(alias), policy) == head_pad_indices_for_policy(event_metadata(center), policy)
    assert event_metadata(alias).atoms == (alias,)
    markers, errors, _ = parse_markers(f'(120){{4}}{alias},', first=0.)
    assert not errors
    assert markers[0]['touch_pad'] == 'C'


def test_aliases_deduplicate_the_same_physical_center():
    for policy in ('merged_pads_v1', 'separate_pads_v2'):
        assert head_pad_indices_for_policy(event_metadata('C1/C8/C'), policy) == head_pad_indices_for_policy(event_metadata('C'), policy)
        assert head_pad_indices_for_policy(event_metadata('A1'), policy) != head_pad_indices_for_policy(event_metadata('C1'), policy)


def test_real_dataset_builds_center_recency_with_existing_atom_spellings(tmp_path):
    import soundfile as sf
    from maichart.composer.data import ComposerDataset
    sr = 8000
    wave = (.1 * np.sin(2 * np.pi * 440 * np.arange(sr*4) / sr)).astype(np.float32)
    sf.write(tmp_path/'track.wav', wave, sr)
    (tmp_path/'maidata.txt').write_text('&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}C1,C2h[16:1],C3f,C,1,2,E\n', encoding='utf8')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'songs': [{'song_id': 'center-alias',
        'maidata_path': str(tmp_path/'maidata.txt'), 'audio_path': str(tmp_path/'track.wav'),
        'difficulties': [{'difficulty_index': 5, 'level_raw': '12'}]}]}), encoding='utf8')
    dataset = ComposerDataset(manifest, cache_dir=tmp_path/'cache', candidate_count=16,
        history_events=8, action_slots=4, preparation_workers=1,
        architecture_profile='audio_grounded_v3', head_timing_policy='separate_pads_v2')
    assert 'C1' in dataset.codec_atoms
    table = dataset._song_table(0)
    center = head_pad_indices_for_policy(event_metadata('C'), 'separate_pads_v2')[0]
    assert table['points'][0].note_text == 'C1'
    assert np.isnan(table['last_head_seconds'][0, center])
    assert table['last_head_seconds'][1, center] == 0.
    for index in range(len(dataset)):
        dataset[index]  # exercises the real path that raised KeyError
