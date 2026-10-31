"""Phrase causality contracts; intentionally small CPU fixtures."""
import torch

from maichart.composer.stream_phrase import PHRASE_FEATURE_DIM
from maichart.composer.stream_phrase_model import CausalPhraseContext, phrase_auxiliary_loss


class AudioReader:
    @staticmethod
    def _read_audio(cache, queries, inverse, **kwargs):
        # Same interval always has the same music, independent of decoder prefix.
        return queries.float()[..., None].expand(*queries.shape, 32) / 16


def phrase_batch():
    f = PHRASE_FEATURE_DIM
    return dict(phrase_start_beat=torch.zeros(1, 2), phrase_end_beat=torch.full((1, 2), 4.),
        phrase_progress=torch.randn(1, 2, f), phrase_memory_features=torch.zeros(1, 2, 4, f),
        phrase_memory_start_beat=torch.zeros(1, 2, 4), phrase_memory_end_beat=torch.zeros(1, 2, 4),
        phrase_memory_mask=torch.zeros(1, 2, 4, dtype=torch.bool), condition=torch.zeros(1, 3),
        step_mask=torch.ones(1, 2, dtype=torch.bool), target_phrase_features=torch.randn(1, 2, f),
        target_phrase_mask=torch.ones(1, 2, dtype=torch.bool), target_phrase_weight=torch.ones(1, 2))


def test_intent_is_anchor_sticky_and_targets_are_not_forward_inputs():
    module = CausalPhraseContext(8, 3).eval()
    batch = phrase_batch()
    cache = {'delta_keys': torch.zeros(1, 1, 32)}
    residual, output = module(AudioReader(), cache, torch.tensor([0]), batch)
    torch.testing.assert_close(output['phrase_intent'][:, 0], output['phrase_intent'][:, 1])
    assert torch.equal(residual, torch.zeros_like(residual))
    batch['target_phrase_features'].fill_(float('nan'))
    batch['phrase_progress'].normal_()
    _, second = module(AudioReader(), cache, torch.tensor([0]), batch)
    torch.testing.assert_close(output['phrase_intent'], second['phrase_intent'])
    assert torch.equal(output['phrase_reference_weights'][..., 0], torch.ones(1, 2))


def test_masked_recovery_target_nan_is_safe_and_receives_no_gradient():
    batch = phrase_batch()
    batch['target_phrase_mask'].zero_()
    batch['target_phrase_features'].fill_(float('nan'))
    prediction = torch.randn_like(batch['target_phrase_features'], requires_grad=True)
    loss = phrase_auxiliary_loss({'phrase_prediction': prediction}, batch)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
