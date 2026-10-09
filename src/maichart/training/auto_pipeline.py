"""Run dataset preprocessing and then training as one task.

The WebUI's one-click button needs both stages to happen without the browser
having to chain them: preprocessing takes minutes, training takes hours, and a
page reload in between must not orphan the run.  A single background process
therefore runs both, printing each stage so the task view shows where it is.

The arguments are supplied as a plan file rather than on the command line, so the
server can build the exact same argv it would use for the two separate stages:

.. code-block:: json

    {
      "family": "event_ctx_v1",           // optional; event_v1 when absent
      "prepare": ["<prepare_events argv>"],
      "train": ["<train_events argv>"]
    }

``prepare`` is skipped when it is missing or empty, which makes the same entry
point usable as "resume training with whatever manifest already exists".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from maichart.training import prepare_events, train_events


def run_plan(plan: dict) -> int:
    family = str(plan.get("family") or "event_v1")
    prepare_args = list(plan.get("prepare") or [])
    train_args = list(plan.get("train") or [])
    if not train_args:
        raise ValueError("计划里没有训练参数：无法启动")

    # Defensive: older plan files embedded the subprocess launcher (`-m <module>`)
    # in the prepare list.  This wrapper calls the stage in-process, which needs
    # only the real arguments — argparse used to reject the prefix with
    # "unrecognized arguments: -m <raw_root>" and exit 2 before any work.
    if prepare_args[:2] == ["-m", "maichart.training.prepare_events"]:
        prepare_args = prepare_args[2:]
    elif prepare_args[:1] == ["-m"] or "preprocess" in prepare_args[:1]:
        raise ValueError(
            "prepare 参数里残留了启动器前缀（-m …），且不是 event 预处理；"
            "legacy 家族的一键管线请改用两个独立任务"
        )

    if prepare_args:
        print(f"auto_pipeline_stage=preprocess family={family} argv={' '.join(prepare_args[:6])} ...", flush=True)
        code = prepare_events.main(prepare_args)
        if code != 0:
            print(f"auto_pipeline_preprocess_failed exit={code}；训练不会开始", flush=True)
            return code
    else:
        print("auto_pipeline_stage=preprocess skipped (no plan)", flush=True)

    if family == "event_ctx_v1":
        # Same trainer, different dataset and target_schema.
        from maichart.training import train_event_ctx

        train_event_ctx.configure()

    print(f"auto_pipeline_stage=train family={family}", flush=True)
    return train_events.main(train_args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess a dataset and train in one background task.")
    parser.add_argument("--plan", required=True, help="JSON file with 'prepare' and 'train' argument lists")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        return run_plan(plan)
    except Exception as exc:  # noqa: BLE001 - the task log is the report
        print(f"auto_pipeline_error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
