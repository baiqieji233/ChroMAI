from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.generation.event_model import EventVocabulary
from maichart.generation.model_compatibility import (
    ModelFamily, checkpoint_capabilities, get_generation_config_for_model,
)
from maichart.generation.onset_guidance import LocalOnsetGuidance


class _PositionChoiceModel:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.head_calls = 0

    def eval(self):
        return self

    def encode_audio(self, audio, condition, *, condition_null, **kwargs):
        return condition_null.float().reshape(1, 1, 1).expand(1, 2, 1), None

    def local_onset_logits(self, memory):
        self.head_calls += 1
        return torch.tensor([[3., -3.]]) if memory[0, 0, 0] else torch.tensor([[-3., 3.]])

    def decode_step(self, memory, token, condition, caches, step, audio_mask=None):
        logits = torch.full((1, 1, len(self.vocabulary)), -100.)
        choices = ({"AT:0": 1., "AT:1/2": .95} if step == 0 else
                   {("CHAR:1" if step == 1 else "<event_end>" if step == 2 else "<eos>"): 20.})
        for label, value in choices.items():
            logits[0, 0, self.vocabulary.token_to_id[label]] = value
        return logits, caches


def test_decoder_uses_conditional_onsets_and_respects_null_branch_and_rest_mask():
    from maichart.generation.event_cli import _decode
    vocabulary = EventVocabulary(["AT:0", "AT:1/2", "CHAR:1"])
    context = {"frame_times": np.array([0., .5]), "start_beat": 0,
               "position_offset": 0, "tempo_changes": [EventTempo(Fraction(0), 60)],
               "first": 0, "strength": .6}
    model = _PositionChoiceModel(vocabulary)
    def run(**kwargs):
        return _decode(model, vocabulary, torch.zeros((2, 112)), torch.zeros((2, 112)),
                       torch.zeros(16), torch.device("cpu"), max_tokens=8,
                       temperature=1., sample=False, seed=7, atom_catalog=["1"],
                       position_end=Fraction(1), **kwargs)
    assert run().beats == [Fraction(0)]
    assert model.head_calls == 0
    assert run(onset_context=context).beats == [Fraction(1, 2)]
    assert run(onset_context=context, guidance_scale=0).beats == [Fraction(0)]

    def mask_rest(logits, vocab, **kwargs):
        logits = logits.clone()
        logits[vocab.token_to_id["AT:1/2"]] = float("-inf")
        return logits
    feedback = SimpleNamespace(shape_logits=mask_rest, accept_cell=lambda *a: True,
                               allow_empty_end=lambda: False)
    assert run(onset_context=context, density_feedback=feedback).beats == [Fraction(0)]


def test_position_guidance_preserves_stop_mass_and_hard_masks():
    logits = torch.tensor([0.1, 0.0, 0.2, float("-inf"), -0.1])
    guide = LocalOnsetGuidance([1, 2, 3], [-2.0, -0.1, 0.0], 0.6)
    result = guide.shape_logits(logits, state="position")
    assert result[2] - result[1] > logits[2] - logits[1]
    assert torch.isneginf(result[3])
    assert torch.equal(result[[0, 4]], logits[[0, 4]])
    assert torch.softmax(result, 0)[0] == pytest.approx(float(torch.softmax(logits, 0)[0]))
    assert torch.logsumexp(result[1:4], 0) == pytest.approx(float(torch.logsumexp(logits[1:4], 0)))
    assert guide.shape_logits(logits, state="char") is logits
    assert LocalOnsetGuidance([1, 2], [-2, 0], 0).shape_logits(logits, state="position") is logits


def test_exact_context_positions_use_tempo_map_and_first():
    vocab = EventVocabulary(["AT:16", "AT:33/2", "AT:17"])
    times = np.asarray([12.3, 12.8, 13.3], dtype=np.float32)
    head = torch.tensor([-2.0, 0.0, 2.0])
    guide = LocalOnsetGuidance.from_predictions(
        head, times, vocab, start_beat=Fraction(16), position_offset=Fraction(16),
        tempo_changes=[EventTempo(Fraction(0), 120), EventTempo(Fraction(8), 60)],
        first=0.3, strength=0.5,
    )
    scores = dict(zip(guide.token_ids.tolist(), guide.log_scores.tolist()))
    assert len(scores) == 3
    for token, index in (("AT:16", 0), ("AT:33/2", 1), ("AT:17", 2)):
        assert scores[vocab.token_to_id[token]] == pytest.approx(float(head[index].sigmoid().log()), abs=2e-6)


def test_pooled_head_scores_use_bin_centres_not_raw_frame_indices():
    vocab = EventVocabulary(["AT:1/2", "AT:5/2", "AT:9/2", "AT:13/2"])
    guide = LocalOnsetGuidance.from_predictions(
        torch.tensor([-3., 3., -3., -3.]), np.arange(8, dtype=np.float32), vocab,
        start_beat=0, position_offset=0, tempo_changes=[EventTempo(Fraction(0), 60)],
        first=0, strength=1,
    )
    result = guide.shape_logits(torch.zeros(len(vocab)), state="position")
    assert result.argmax().item() == vocab.token_to_id["AT:5/2"]


def _checkpoint(*, trained=True):
    return {
        "schema": "maichart-event-generator-checkpoint-v1",
        "model_config": {"d_model": 16, "planner_profile": "music_structure_v3", "plan_dim": 13,
                         "onset_profile": "chart_onsets_v1", "local_pooling_policy": "per_sample_v2"},
        "data_contract": {"target_schema": "event_ctx_v1",
                          "onset_params": {"profile": "chart_onsets_v1", "loss_weight": 0.2}},
        "onset_training": {"profile": "chart_onsets_v1", "trained": trained, "optimizer_steps": 12 if trained else 0},
        "model_state_dict": {
            "onset_head.0.weight": torch.zeros(8, 16), "onset_head.0.bias": torch.zeros(8),
            "onset_head.2.weight": torch.zeros(1, 8), "onset_head.2.bias": torch.zeros(1),
        },
    }


def test_head_capability_requires_trained_matching_weights():
    checkpoint = _checkpoint()
    assert checkpoint_capabilities(checkpoint)["local_onset_guidance"]
    assert not checkpoint_capabilities(_checkpoint(trained=False))["local_onset_guidance"]
    checkpoint["model_state_dict"].pop("onset_head.2.weight")
    caps = checkpoint_capabilities(checkpoint)
    assert not caps["local_onset_guidance"] and caps["errors"]


def test_legacy_generation_only_accepts_zero_onset_strength():
    checkpoint = _checkpoint()
    checkpoint["model_config"].pop("onset_profile")
    checkpoint.pop("onset_training")
    checkpoint["data_contract"].pop("onset_params")
    checkpoint["model_state_dict"] = {}
    assert checkpoint_capabilities(checkpoint)["fine_note_strength"]
    assert not checkpoint_capabilities(checkpoint)["local_onset_guidance"]
    args = SimpleNamespace(local_onset_bias_strength=0)
    assert get_generation_config_for_model(ModelFamily.EVENT_CTX_V1, args, checkpoint)["local_onset_bias_strength"] == 0
    args.local_onset_bias_strength = 0.3
    with pytest.raises(ValueError, match="已训练采音头"):
        get_generation_config_for_model(ModelFamily.EVENT_CTX_V1, args, checkpoint)
