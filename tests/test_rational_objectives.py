import torch

from maichart.form.codec import RationalCodec
from maichart.form.rational_objectives import event_decision_loss


def _batch(codec, target_rows):
    width = max(len(row) for row in target_rows)
    decoder = []
    target = []
    for row in target_rows:
        decoder.append(row[:-1] + [codec.pad_id] * (width - len(row)))
        target.append(row[1:] + [codec.pad_id] * (width - len(row)))
    return {"decoder_input": torch.tensor(decoder), "target": torch.tensor(target)}


def test_event_decision_loss_empty_and_nonempty_boundaries_have_expected_shape():
    codec = RationalCodec()
    # BOS -> AT -> ... -> END -> EOS; second row is an empty BOS -> EOS cell.
    batch = _batch(codec, [
        [codec.bos_id, codec.at_id, codec.note_id, 18, codec.end_id, codec.eos_id],
        [codec.bos_id, codec.eos_id],
    ])
    logits = torch.zeros(2, 5, len(codec), requires_grad=True)
    values = event_decision_loss(logits, batch)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert values[0] > 0 and values[1] > 0
    values.sum().backward()
    assert logits.grad[0, 0, codec.at_id] < 0
    assert logits.grad[1, 0, codec.eos_id] < 0


def test_event_decision_loss_ignores_nonboundary_and_padding():
    codec = RationalCodec()
    batch = {
        "decoder_input": torch.tensor([[codec.bos_id, codec.at_id, codec.pad_id]]),
        "target": torch.tensor([[codec.at_id, codec.eos_id, codec.pad_id]]),
    }
    logits = torch.zeros(1, 3, len(codec))
    value = event_decision_loss(logits, batch)
    # Only BOS -> AT is a boundary decision; AT -> EOS and padding are ignored.
    assert value.shape == (1,)
    assert value.item() > 0


def test_event_decision_loss_without_boundaries_is_finite_zero():
    codec = RationalCodec()
    batch = {
        "decoder_input": torch.tensor([[codec.at_id, codec.note_id]]),
        "target": torch.tensor([[codec.note_id, codec.end_id]]),
    }
    logits = torch.randn(1, 2, len(codec))
    value = event_decision_loss(logits, batch)
    assert value.tolist() == [0.0]
