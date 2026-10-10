import torch

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig, EventVocabulary
from maichart.training.event_checkpoints import (
    EVENT_CHECKPOINT_SCHEMA,
    compare_event_contract,
    load_event_checkpoint,
    vocabulary_signature,
)
from maichart.training.train_events import local_onset_objective, pool_onset_targets, run_epoch


def _tiny_model(*, profile="chart_onsets_v1", policy="per_sample_v2"):
    return EventConditionedTransformer(EventTransformerConfig(
        input_dim=8, condition_dim=16, vocab_size=16, d_model=16, nhead=2,
        num_layers=1, dim_feedforward=32, max_audio_tokens=4, global_audio_tokens=2,
        max_tokens=16, planner_profile="music_structure_v3", onset_profile=profile,
        local_pooling_policy=policy,
    ))


def _onset_contract():
    return {"profile": "chart_onsets_v1", "target_version": "chart_onsets_v1",
            "sigma_sec": 0.02, "loss_weight": 0.2,
            "local_pooling_policy": "per_sample_v2"}


def test_onset_targets_and_weights_are_pooled_with_the_same_bins():
    model = _tiny_model()
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    weight = torch.ones_like(target)
    valid = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    padding = torch.tensor([[False, False, False, True, True, True, True]])
    pooled_target, pooled_quality, pooled_valid = pool_onset_targets({
        "onset_target": target, "onset_weight": weight, "onset_valid": valid,
        "audio_padding_mask": padding,
    }, model)
    assert pooled_target.shape == (1, 3)
    torch.testing.assert_close(pooled_target, torch.tensor([[0.0, 1.0, 0.0]]))
    torch.testing.assert_close(pooled_quality, torch.ones_like(pooled_quality))
    torch.testing.assert_close(pooled_valid, torch.ones_like(pooled_valid))


def test_zero_quality_weight_has_no_onset_loss_or_gradient():
    logits = torch.zeros((1, 3), requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    quality = torch.zeros_like(target)
    valid = torch.ones_like(target)
    loss, stats = local_onset_objective(logits, target, quality, valid)
    loss.backward()
    assert loss.item() == 0.0
    assert stats["valid_frames"].item() == 3
    assert stats["weighted_frames"].item() == 0.0
    assert torch.count_nonzero(logits.grad) == 0


class _OneBatch:
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


class _Scaler:
    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def get_scale(self):
        return 1.0


def _training_batch(quality=1.0):
    return {
        "audio": torch.randn(1, 4, 8), "audio_padding_mask": torch.zeros(1, 4, dtype=torch.bool),
        "global_audio": torch.randn(1, 2, 8), "condition": torch.zeros(1, 16),
        "condition_null": torch.zeros(1, dtype=torch.bool),
        "decoder_input": torch.tensor([[0, 1]]), "target": torch.tensor([[1, 2]]),
        "decoder_padding_mask": torch.zeros(1, 2, dtype=torch.bool),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "plan_audio_mask": torch.zeros(1, 4, dtype=torch.bool),
        "plan_target": torch.zeros(1, 10), "quality_token_weights": torch.ones(1, 2),
        "quality_sample_weights": torch.ones(1), "quality_plan_weights": torch.ones(1),
        "onset_target": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        "onset_weight": torch.full((1, 4), quality),
        "onset_valid": torch.ones(1, 4),
    }


def test_run_epoch_trains_aux_head_and_counts_only_auxiliary_updates():
    torch.manual_seed(4)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    initial = {key: value.detach().clone() for key, value in model.onset_head.state_dict().items()}
    metrics = run_epoch(model, _OneBatch(_training_batch()), torch.device("cpu"),
                        optimizer=optimizer, scaler=_Scaler(), accumulation=1,
                        onset_loss_weight=0.2)
    assert metrics["onset_optimizer_steps_successful"] == 1
    assert metrics["onset_valid_frames"] == 4
    assert metrics["onset_bce"] > 0
    assert any(not torch.equal(initial[key], value) for key, value in model.onset_head.state_dict().items())

    frozen = {key: value.detach().clone() for key, value in model.onset_head.state_dict().items()}
    zero_weight_metrics = run_epoch(model, _OneBatch(_training_batch(quality=0.0)),
                                    torch.device("cpu"), optimizer=optimizer,
                                    scaler=_Scaler(), accumulation=1, onset_loss_weight=0.2)
    assert zero_weight_metrics["onset_optimizer_steps_successful"] == 0
    assert all(torch.equal(frozen[key], value) for key, value in model.onset_head.state_dict().items())


def test_validation_objective_includes_onset_loss_without_creating_gradients():
    torch.manual_seed(9)
    model = _tiny_model()
    batch = _training_batch()
    with_aux = run_epoch(model, _OneBatch(batch), torch.device("cpu"), onset_loss_weight=0.2)
    without_aux = run_epoch(model, _OneBatch(batch), torch.device("cpu"), onset_loss_weight=0.0)

    assert with_aux["loss"] == without_aux["loss"]
    assert abs((with_aux["objective_loss"] - without_aux["objective_loss"])
               - 0.2 * with_aux["onset_bce"]) < 1e-6
    assert all(parameter.grad is None for parameter in model.parameters())


def test_checkpoint_with_enabled_profile_must_contain_head_and_metadata(tmp_path):
    model = _tiny_model()
    vocab = EventVocabulary.from_event_charts([], segment_beats=4)
    payload = {
        "schema": EVENT_CHECKPOINT_SCHEMA,
        "model_config": {**model.config.to_dict(), "vocab_size": len(vocab)},
        "model_state_dict": model.state_dict(),
        "vocabulary": vocab.to_dict(),
        "data_contract": {"vocabulary_signature": vocabulary_signature(vocab),
                          "onset_params": _onset_contract()},
        "onset_training": {"profile": "chart_onsets_v1", "trained": False, "optimizer_steps": 0},
    }
    path = tmp_path / "complete.pt"
    torch.save(payload, path)
    assert load_event_checkpoint(path)["model_config"]["onset_profile"] == "chart_onsets_v1"

    payload["model_state_dict"] = {
        key: value for key, value in model.state_dict().items() if not key.startswith("onset_head.")
    }
    torch.save(payload, path)
    try:
        load_event_checkpoint(path)
    except ValueError as error:
        assert "weights" in str(error)
    else:  # pragma: no cover
        raise AssertionError("generation/resume loader accepted a missing trained head")


def test_onset_contract_migration_is_allowed_only_for_init():
    common = {
        "target_schema": "event_ctx_v1", "feature_profile": "event_audio_v2_rhythm",
        "input_dim": 112, "sample_rate": 22050, "hop_length": 441,
        "segment_beats": 16, "max_tokens": 1024, "vocabulary_signature": "same",
        "condition_keys": [], "dataset_signature": "same", "optimizer_state_dict": {"state": {}},
    }
    stored = {"data_contract": dict(common)}
    current = dict(common)
    current["onset_params"] = _onset_contract()
    try:
        compare_event_contract(stored, current, mode="resume")
    except ValueError as error:
        assert "onset_params" in str(error)
    else:  # pragma: no cover
        raise AssertionError("resume silently changed the onset objective")
    compare_event_contract(stored, current, mode="init")
