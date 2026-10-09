"""The one-click path must run preprocessing and then training, in that order.

Chaining them in the browser would orphan a multi-hour run when the page is
reloaded, so both stages live in one background process driven by a plan file.
These tests pin the ordering, the failure behaviour (a failed preprocessing must
not start training) and the family switch.
"""

import json

import pytest

from maichart.training import auto_pipeline


def test_plan_runs_preprocess_then_train(monkeypatch, tmp_path, capsys):
    calls = []

    monkeypatch.setattr(auto_pipeline.prepare_events, "main",
                        lambda argv: calls.append(("prepare", argv)) or 0)
    monkeypatch.setattr(auto_pipeline.train_events, "main",
                        lambda argv: calls.append(("train", argv)) or 0)

    plan = {"family": "event_v1", "prepare": ["raw", "-o", "manifest.json"],
            "train": ["--manifest", "manifest.json", "--epochs", "2"]}
    code = auto_pipeline.run_plan(plan)

    assert code == 0
    assert [kind for kind, _ in calls] == ["prepare", "train"]
    assert calls[1][1] == ["--manifest", "manifest.json", "--epochs", "2"]
    output = capsys.readouterr().out
    assert "auto_pipeline_stage=preprocess" in output
    assert "auto_pipeline_stage=train" in output


def test_failed_preprocessing_does_not_start_training(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(auto_pipeline.prepare_events, "main",
                        lambda argv: calls.append("prepare") or 2)
    monkeypatch.setattr(auto_pipeline.train_events, "main",
                        lambda argv: calls.append("train") or 0)

    code = auto_pipeline.run_plan({"prepare": ["raw"], "train": ["--manifest", "m.json"]})
    assert code == 2
    assert calls == ["prepare"], "training must not run when preprocessing fails"
    assert "auto_pipeline_preprocess_failed" in capsys.readouterr().out


def test_plan_without_prepare_starts_training_directly(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(auto_pipeline.train_events, "main",
                        lambda argv: calls.append(argv) or 0)
    code = auto_pipeline.run_plan({"train": ["--manifest", "m.json"]})
    assert code == 0 and calls
    assert "preprocess skipped" in capsys.readouterr().out


def test_plan_without_train_arguments_is_rejected():
    with pytest.raises(ValueError, match="没有训练参数"):
        auto_pipeline.run_plan({"prepare": ["raw"]})


def test_legacy_launcher_prefix_is_stripped_from_prepare(monkeypatch, capsys):
    """The wrapper runs stages in-process: the `-m <module>` launcher prefix in
    an old plan must be stripped, not handed to argparse (exit 2 "unrecognized
    arguments: -m <raw_root>" was exactly this bug)."""

    calls: list[list] = []
    monkeypatch.setattr(auto_pipeline.prepare_events, "main",
                        lambda argv: calls.append(list(argv)) or 0)
    monkeypatch.setattr(auto_pipeline.train_events, "main", lambda argv: 0)

    code = auto_pipeline.run_plan({
        "prepare": ["-m", "maichart.training.prepare_events", "raw", "--limit", "3"],
        "train": ["--manifest", "m.json"],
    })
    assert code == 0
    assert calls == [["raw", "--limit", "3"]]


def test_legacy_family_prepare_with_launcher_is_rejected():
    """A legacy-style prepare list cannot be routed in-process; fail loudly."""

    with pytest.raises(ValueError, match="启动器前缀"):
        auto_pipeline.run_plan({
            "prepare": ["-m", "maichart.cli", "preprocess", "build", "raw"],
            "train": ["--manifest", "m.json"],
        })


def test_ctx_family_switches_the_trainer(monkeypatch, tmp_path):
    """The plan carries the family so the wrapper can install the right dataset."""

    import maichart.training.train_event_ctx as ctx
    import maichart.training.train_events as train_events

    configured = []
    monkeypatch.setattr(ctx, "configure", lambda: configured.append(True))
    monkeypatch.setattr(auto_pipeline.train_events, "main", lambda argv: 0)
    monkeypatch.setattr(auto_pipeline, "prepare_events", auto_pipeline.prepare_events)

    code = auto_pipeline.run_plan({"family": "event_ctx_v1", "train": ["--manifest", "m.json"]})
    assert code == 0
    assert configured == [True], "the cross-window family must be installed before training"
    # event_v1 leaves the hooks alone.
    assert train_events.DATASET_CLASS.__name__ in {"EventDataset", "EventCtxDataset"}


def test_cli_reads_the_plan_file(monkeypatch, tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"train": ["--manifest", "m.json"]}), encoding="utf-8")
    monkeypatch.setattr(auto_pipeline.train_events, "main", lambda argv: 0)
    assert auto_pipeline.main(["--plan", str(plan_path)]) == 0

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert auto_pipeline.main(["--plan", str(broken)]) == 1
    assert "auto_pipeline_error" in capsys.readouterr().err
