import random
from types import SimpleNamespace

import torch
import pytest

from maichart.training.train_events import (
    PrefetchLoader,
    _global_step_metadata,
    run_epoch,
)
from maichart.training import train_events


class _ConditionDataset:
    def __init__(self, seed):
        self._condition_rng = random.Random(seed)
        self.draws = []

    def __iter__(self):
        draw = self._condition_rng.random()
        self.draws.append(draw)
        yield {
            "audio": torch.zeros(1, 1, 1),
            "global_audio": torch.zeros(1, 1, 1),
            "decoder_input": torch.zeros(1, 1, dtype=torch.long),
            "target": torch.ones(1, 1, dtype=torch.long),
            "condition": torch.tensor([[0.5, draw]], dtype=torch.float32),
            "condition_null": torch.zeros(1, dtype=torch.bool),
            "audio_padding_mask": torch.zeros(1, 1, dtype=torch.bool),
            "decoder_padding_mask": torch.zeros(1, 1, dtype=torch.bool),
            "loss_mask": torch.ones(1, 1, dtype=torch.bool),
            "plan_target": torch.zeros(1, 6),
            "quality_sample_weights": torch.ones(1),
            "quality_plan_weights": torch.ones(1),
            "quality_token_weights": torch.ones(1, 1),
        }


class _Loader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(self.dataset)


class _PrefetchWrapper:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


class _BrokenDataset(_ConditionDataset):
    def __iter__(self):
        yield from super().__iter__()
        raise RuntimeError("feature extraction failed in prefetch producer")


class _BrokenLoader(_Loader):
    def __len__(self):
        return 2


class _TinyPlannerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(4))
        self.plan = torch.nn.Parameter(torch.zeros(9))
        self.config = SimpleNamespace(planner_profile="legacy_v1")
        self.conditions_seen = []

    def forward(self, audio, decoder_input, condition, **_kwargs):
        self.conditions_seen.append(condition.detach().cpu().clone())
        logits = self.logits.view(1, 1, 4).expand(condition.size(0), decoder_input.size(1), -1)
        plan = self.plan.view(1, -1).expand(condition.size(0), -1)
        return logits, plan


class _FakeScaler:
    def __init__(self, *, skip_step=False):
        self.skip_step = skip_step

    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, _optimizer):
        return None

    def step(self, optimizer):
        if not self.skip_step:
            optimizer.step()

    def update(self):
        return None

    def get_scale(self):
        return 1.0


def test_validation_replays_fixed_condition_mask_without_reseeding_training_rng():
    validation_dataset = _ConditionDataset(seed=7)
    validation_loader = _PrefetchWrapper(_Loader(validation_dataset))
    train_dataset = _ConditionDataset(seed=938)
    train_loader = _PrefetchWrapper(_Loader(train_dataset))
    training_rng_state = train_dataset._condition_rng.getstate()
    model = _TinyPlannerModel()

    first_validation = run_epoch(model, validation_loader, torch.device("cpu"),
                                 validation_condition_seed=0, phase="validation")
    first_condition = model.conditions_seen[-1]
    second_validation = run_epoch(model, validation_loader, torch.device("cpu"),
                                  validation_condition_seed=0, phase="validation")
    second_condition = model.conditions_seen[-1]

    assert first_condition.equal(second_condition)
    assert first_validation["validation_condition_rng_reset"] is True
    assert first_validation["validation_condition_seed"] == 0
    assert first_validation["phase"] == "validation"
    assert second_validation["validation_condition_rng_reset"] is True
    assert train_dataset._condition_rng.getstate() == training_rng_state

    expected_train_draw = random.Random(938).random()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    run_epoch(model, train_loader, torch.device("cpu"), optimizer=optimizer,
              scaler=_FakeScaler(), accumulation=1)
    assert train_dataset.draws == [expected_train_draw]


def test_optimizer_hook_distinguishes_success_from_scaler_skip():
    device = torch.device("cpu")
    model = _TinyPlannerModel()
    successful_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    successful = run_epoch(model, _Loader(_ConditionDataset(seed=1)), device,
                           optimizer=successful_optimizer, scaler=_FakeScaler(), accumulation=1)
    assert successful["optimizer_steps_attempted"] == 1
    assert successful["optimizer_steps_successful"] == 1
    assert successful["amp_skipped_steps"] == 0

    skipped_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    skipped = run_epoch(model, _Loader(_ConditionDataset(seed=2)), device,
                        optimizer=skipped_optimizer, scaler=_FakeScaler(skip_step=True),
                        accumulation=1)
    assert skipped["optimizer_steps_attempted"] == 1
    assert skipped["optimizer_steps_successful"] == 0
    assert skipped["amp_skipped_steps"] == 1


def test_epoch_metrics_separate_token_planner_config_and_objective_losses():
    model = _TinyPlannerModel()
    metrics = run_epoch(model, _Loader(_ConditionDataset(seed=3)), torch.device("cpu"))

    assert metrics["loss"] == metrics["token_loss"]
    assert metrics["objective_loss"] > metrics["token_loss"]
    assert metrics["planner_loss"] > 0
    assert metrics["configuration_loss"] > 0
    assert metrics["configuration_accuracy"] == 1.0
    assert metrics["optimizer_steps_attempted"] == 0
    assert metrics["optimizer_steps_successful"] == 0
    assert metrics["amp_skipped_steps"] == 0


def test_prefetch_dataset_exception_reaches_epoch_caller(tmp_path):
    dataset = _BrokenDataset(seed=17)
    loader = PrefetchLoader(_BrokenLoader(dataset), depth=1)

    with pytest.raises(RuntimeError, match="feature extraction failed"):
        run_epoch(_TinyPlannerModel(), loader, torch.device("cpu"), phase="validation")


def test_run_epoch_emits_phase_progress_without_waiting(monkeypatch, capsys):
    times = iter([100.0, 161.0])
    monkeypatch.setattr(train_events.time, "monotonic", lambda: next(times))

    run_epoch(_TinyPlannerModel(), _Loader(_ConditionDataset(seed=23)), torch.device("cpu"),
              phase="validation", progress_interval_seconds=60)

    assert "epoch_progress phase=validation batch=1/1 elapsed_seconds=61.0" in capsys.readouterr().out


def test_old_global_step_is_retained_as_unverified_legacy_baseline():
    semantics, legacy_baseline = _global_step_metadata({"global_step": 1301}, resume=True)
    assert semantics == "legacy_baseline_plus_successful_steps_v1"
    assert legacy_baseline == 1301


def test_global_step_metadata_marks_old_baseline_without_guessing_successful_steps():
    assert _global_step_metadata(None, resume=False) == ("successful_optimizer_steps_v1", None)
    assert _global_step_metadata({"global_step": 1301}, resume=True) == (
        "legacy_baseline_plus_successful_steps_v1", 1301,
    )
    assert _global_step_metadata({"global_step": 1304,
                                  "global_step_semantics": "successful_optimizer_steps_v1"},
                                 resume=True) == ("successful_optimizer_steps_v1", None)
