import json
from fractions import Fraction

import torch

from maichart.event_timing import EventPoint, event_points_to_joint_tokens, joint_tokens_to_event_points
from maichart.stream.model import ChartStreamTransformer, StreamConfig
from maichart.stream.checkpoint import save, load
from maichart.generation.model_compatibility import checkpoint_capabilities


def test_stream_forward_and_history_summary_are_finite():
    model = ChartStreamTransformer(StreamConfig(vocab_size=55, d_model=32, nhead=4,
        layers=1, dim_feedforward=64, max_tokens=32, max_history_tokens=8, max_stream_memory=4))
    batch = {'audio': torch.randn(2, 6, 120), 'condition': torch.randn(2, 16),
        'audio_padding_mask': torch.zeros(2, 6, dtype=torch.bool),
        'condition_null': torch.zeros(2, dtype=torch.bool),
        'decoder_input': torch.randint(0, 55, (2, 12)),
        'target': torch.randint(0, 55, (2, 12)),
        'history': torch.randint(0, 55, (2, 8)),
        'history_padding_mask': torch.zeros(2, 8, dtype=torch.bool)}
    output = model(batch)
    assert output['logits'].shape == (2, 12, 55)
    assert torch.isfinite(output['logits']).all()
    (output['logits'].square().mean() + output['plan'].square().mean()).backward()


def test_stream_event_tokens_preserve_fraction_and_configuration():
    points = [EventPoint(Fraction(1, 6), '1/5'), EventPoint(Fraction(2, 3), '3h[4:1]')]
    tokens = event_points_to_joint_tokens(points)
    restored = joint_tokens_to_event_points(tokens)
    assert [(p.beat, p.note_text) for p in restored] == [(p.beat, p.note_text) for p in points]


def test_stream_checkpoint_is_independent(tmp_path):
    model = ChartStreamTransformer(StreamConfig(vocab_size=55, d_model=16, nhead=2,
        layers=1, dim_feedforward=32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    path = tmp_path / 'stream_last.pt'
    save(path, model, optimizer, epoch=1, args={'x': 1}, manifest=manifest)
    restored, payload = load(path)
    assert payload['family'] == 'chart_stream_v1'
    assert restored.config == model.config
    caps = checkpoint_capabilities(payload)
    assert caps['model_family'] == 'chart_stream_v1' and caps['valid']


def test_stream_dataset_builds_continuation_chunks_without_chunk_eos(tmp_path, monkeypatch):
    import json
    import soundfile as sf
    import numpy as np
    from maichart.stream.data import StreamDataset

    song = tmp_path / 'song'; song.mkdir()
    sr = 22050
    signal = np.zeros(sr * 3, dtype=np.float32)
    sf.write(song / 'track.wav', signal, sr)
    (song / 'maidata.txt').write_text(
        '&wholebpm=120\n&lv_5=13\n&inote_5=(120)' + ''.join('{8}1,2,3,4,' for _ in range(20)) + 'E\n',
        encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [dict(song_id='stream', maidata_path='song/maidata.txt',
        audio_path='song/track.wav', difficulties=[dict(difficulty_index=5, level_raw='13')])]}), encoding='utf-8')
    dataset = StreamDataset(manifest, cache_dir=tmp_path / 'cache', max_tokens=32,
        max_history_tokens=8, max_region_seconds=15, preparation_workers=1, audio_cache_gb=.2)
    assert len(dataset) > 1
    assert all(not (chunk.final is False and chunk.target[-1] == dataset.codec.eos_id)
               for chunk in dataset.chunks)
    item = dataset[0]
    assert item['action_state'].shape[-1] == 50
