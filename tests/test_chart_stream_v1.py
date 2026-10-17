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
