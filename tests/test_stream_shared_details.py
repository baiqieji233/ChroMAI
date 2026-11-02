import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_shared_details import SharedDetailEmbedding, detail_components


def test_components_share_relative_path_and_retain_exact_duration_and_modifiers():
    assert detail_components('slide', 0, '-3[4:1]') == detail_components('slide', 1, '-4[4:1]')
    assert detail_components('hold', 0, 'h[#0.12345]') != detail_components('hold', 0, 'h[#0.12346]')
    assert detail_components('hold', 0, 'bh[4:1]') != detail_components('hold', 0, 'hb[4:1]')


def test_catalog_order_exact_decode_and_shared_feedback_gradients():
    events = ['1h[4:1]', '1h[8:1]', '1bh[4:1]', '1h[4:1]/3', '1h[8:1]/3']
    codec = EventSetCodec.fit(events)
    module = SharedDetailEmbedding(codec)
    bank = module.encode_catalog()
    shared_rows = []
    losses = []
    for text in events:
        encoded = codec.encode(text)
        shape = encoded['shape']
        assert shape != 0
        for slot, index in enumerate(encoded['details']):
            weights = module.candidate_weights(shape, slot, bank)
            torch.testing.assert_close(module.encode_shape(shape)[slot], weights)
            torch.testing.assert_close(module.candidate_weights(shape, slot), weights)
            assert weights.shape == (codec.detail_sizes[shape][slot], 32)
            feedback = module.embedding_for(shape, slot, torch.tensor([index]), bank)
            torch.testing.assert_close(feedback[0], weights[index])
            losses.append((feedback*torch.arange(32)).sum())
            if codec.detail_catalog[shape][slot][index] == 'h[4:1]':
                shared_rows.append(feedback)
        decoded = codec.decode(shape, encoded['phase'], encoded['details'])
        assert codec.encode(decoded)['details'] == encoded['details']
    assert len(shared_rows) >= 2
    torch.testing.assert_close(shared_rows[0], shared_rows[-1], rtol=0, atol=0)
    sum(losses).backward()
    for embedding in module.component_embeddings:
        assert embedding.weight.grad is not None
        assert torch.isfinite(embedding.weight.grad).all()
        assert embedding.weight.grad.abs().sum() > 0
