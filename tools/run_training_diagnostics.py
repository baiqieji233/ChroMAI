"""Wait for selected training jobs, then run bounded CPU-only diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class StateLockError(RuntimeError):
    pass


class StateFileLock:
    """Nonblocking OS lock; the lock file itself may remain after exit."""

    def __init__(self, state_path: Path) -> None:
        self.path = state_path.with_name(state_path.name + ".lock")
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise StateLockError(f"Diagnostics state is already locked: {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Diagnostics config must be a JSON object.")
    for key in ("pipeline_state", "config_sha256", "jobs", "state_path"):
        if key not in config:
            raise ValueError(f"Diagnostics config is missing {key!r}.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(config["config_sha256"])):
        raise ValueError("config_sha256 must be a 64-character SHA-256 hex digest.")
    if not isinstance(config["jobs"], list) or not config["jobs"]:
        raise ValueError("jobs must be a non-empty JSON array.")
    pipeline_path = _resolve(config["pipeline_state"], path.parent)
    state_path = _resolve(config["state_path"], path.parent)
    if state_path == pipeline_path:
        raise ValueError("state_path must not alias pipeline_state.")
    if state_path == path:
        raise ValueError("state_path must not alias the diagnostics config file.")
    protected_paths = {pipeline_path, state_path, path}
    state_input_paths = set()
    for job in config["jobs"]:
        if isinstance(job, dict):
            for key in ("checkpoint", "manifest"):
                if job.get(key):
                    input_path = _resolve(job[key], path.parent)
                    protected_paths.add(input_path)
                    state_input_paths.add(input_path)
    if state_path in state_input_paths:
        raise ValueError("state_path must not alias a configured checkpoint or manifest.")
    names = set()
    for index, job in enumerate(config["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be an object.")
        for key in ("training_job_name", "checkpoint", "kind", "manifest", "cache_dir", "output"):
            if not job.get(key):
                raise ValueError(f"jobs[{index}] is missing {key!r}.")
        if job["kind"] not in {"event_tokens", "onset_alignment"}:
            raise ValueError(f"jobs[{index}].kind must be event_tokens or onset_alignment.")
        name = str(job["training_job_name"])
        if name in names:
            raise ValueError(f"Duplicate training_job_name: {name}")
        names.add(name)
        if not re.fullmatch(r"event_epoch_\d{3}\.pt", Path(job["checkpoint"]).name):
            raise ValueError(f"jobs[{index}].checkpoint must be an immutable event_epoch_NNN.pt file.")
        checkpoint_path = _resolve(job["checkpoint"], path.parent)
        manifest_path = _resolve(job["manifest"], path.parent)
        output_path = _resolve(job["output"], path.parent)
        protected_paths.update({checkpoint_path, manifest_path})
        for previous in config["jobs"][:index]:
            protected_paths.add(_resolve(previous["output"], path.parent))
        if output_path in protected_paths:
            raise ValueError(f"jobs[{index}].output aliases a protected input or state path: {output_path}")
        protected_paths.add(output_path)
        limit = int(job.get("limit", 128))
        batch_size = int(job.get("batch_size", 4))
        if limit < 1 or batch_size not in {2, 4}:
            raise ValueError(f"jobs[{index}] requires limit >= 1 and batch_size 2 or 4.")
    poll_seconds = float(config.get("poll_seconds", 10))
    if not math.isfinite(poll_seconds) or poll_seconds < 0:
        raise ValueError("poll_seconds must be finite and nonnegative.")
    config["poll_seconds"] = poll_seconds
    return config, path.parent


def _new_state(config: dict[str, Any], config_digest: str) -> dict[str, Any]:
    return {
        "schema": "maichart-training-diagnostics-state-v1",
        "config_sha256": config_digest,
        "pipeline_config_sha256": str(config["config_sha256"]).lower(),
        "status": "waiting",
        "created_at": _now(),
        "updated_at": _now(),
        "jobs": [{
            "training_job_name": job["training_job_name"], "kind": job["kind"],
            "checkpoint": job["checkpoint"], "status": "pending",
            "checkpoint_sha256": None, "report_path": None, "error": None,
        } for job in config["jobs"]],
    }


class TrainingDiagnostics:
    def __init__(self, config: dict[str, Any], config_dir: Path, state_path: Path,
                 *, sleep_fn: Callable[[float], None] = time.sleep,
                 max_polls: int | None = None) -> None:
        self.config = config
        self.config_dir = config_dir
        self.state_path = state_path
        self.sleep_fn = sleep_fn
        self.max_polls = max_polls
        self.polls = 0
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.config_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return _new_state(self.config, self.config_digest)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != self.config_digest:
            raise ValueError("Diagnostics state belongs to a different diagnostics config.")
        if state.get("pipeline_config_sha256") != str(self.config["config_sha256"]).lower():
            raise ValueError("Diagnostics state pipeline config digest does not match this config.")
        expected = {str(job["training_job_name"]) for job in self.config["jobs"]}
        actual = {str(job.get("training_job_name")) for job in state.get("jobs", [])}
        if expected != actual:
            raise ValueError("Diagnostics state jobs do not match this config.")
        return state

    def _persist(self) -> None:
        self.state["updated_at"] = _now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_name(self.state_path.name + ".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.state_path)

    def _pipeline_snapshot(self) -> dict[str, Any] | None:
        path = _resolve(self.config["pipeline_state"], self.config_dir)
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(snapshot, dict):
            return None
        observed = snapshot.get("config_sha256")
        if observed != str(self.config["config_sha256"]).lower():
            raise ValueError(f"Pipeline config digest mismatch: expected {self.config['config_sha256']}, observed {observed}.")
        return snapshot

    @staticmethod
    def _task_status(job: dict[str, Any]) -> str | None:
        snapshot = job.get("last_snapshot")
        if isinstance(snapshot, dict) and snapshot.get("status"):
            return str(snapshot["status"])
        return None

    @staticmethod
    def _listed_checkpoint(job: dict[str, Any], checkpoint: Path) -> bool:
        snapshot = job.get("last_snapshot") or {}
        result = job.get("task_result") or snapshot.get("result") or {}
        paths = result.get("paths") or []
        target = checkpoint.resolve()
        return any(Path(path).expanduser().resolve() == target for path in paths)

    def _ready_training_jobs(self, snapshot: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str | None]:
        observed_jobs = {str(job.get("name")): job for job in snapshot.get("jobs", [])
                         if isinstance(job, dict)}
        upstream_status = str(snapshot.get("status") or "").lower()
        upstream_failed = upstream_status in {
            "failed", "cancelled", "dependency_failed", "completed_with_errors",
        }
        ready = {}
        for item in self.state["jobs"]:
            if item["status"] == "completed":
                continue
            source = observed_jobs.get(item["training_job_name"])
            if source is None or source.get("status") in {None, "pending", "submitting", "submitted", "observing"}:
                if upstream_failed:
                    item.update({"status": "failed", "error":
                                 f"Upstream pipeline ended as {upstream_status} before the selected training job completed."})
                    return {}, "failed"
                return {}, "waiting"
            status = str(source.get("status"))
            if status in {"failed", "cancelled"}:
                item.update({"status": "failed", "error": f"Training job {status}: {source.get('error') or self._task_status(source) or status}"})
                return {}, "failed"
            if status != "completed" or self._task_status(source) not in {None, "completed"}:
                if upstream_failed:
                    item.update({"status": "failed", "error":
                                 f"Upstream pipeline ended as {upstream_status} before the selected training job completed."})
                    return {}, "failed"
                return {}, "waiting"
            ready[item["training_job_name"]] = source
        return ready, None

    def _run_job(self, spec: dict[str, Any], state_job: dict[str, Any], source_job: dict[str, Any]) -> None:
        checkpoint = _resolve(spec["checkpoint"], self.config_dir)
        try:
            if not self._listed_checkpoint(source_job, checkpoint):
                raise ValueError("Completed training result does not list the configured checkpoint path.")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
            before = _sha256(checkpoint)
            state_job.update({"status": "running", "checkpoint": str(checkpoint),
                              "checkpoint_sha256": before, "report_path": None, "error": None,
                              "started_at": _now()})
            self._persist()
            import torch

            torch.set_num_threads(2)
            common = {
                "checkpoint": checkpoint,
                "manifest": _resolve(spec["manifest"], self.config_dir),
                "cache_dir": _resolve(spec["cache_dir"], self.config_dir),
                "limit": int(spec.get("limit", 128)),
                "batch_size": int(spec.get("batch_size", 4)),
                "seed": int(spec.get("seed", 1729)),
                "output": None,
            }
            if spec["kind"] == "event_tokens":
                from tools.evaluate_event_tokens import evaluate

                args = SimpleNamespace(**common, audio_ablation="shuffle_acoustic")
            elif spec["kind"] == "onset_alignment":
                from tools.evaluate_onset_alignment import evaluate

                args = SimpleNamespace(**common, quality_annotations=None,
                                       shuffle_acoustic=True, time_only=True)
            else:  # Config validation already constrains this set.
                raise ValueError(f"Unknown diagnostic kind: {spec['kind']}")
            report = evaluate(args)
            after = _sha256(checkpoint)
            if before != after:
                raise RuntimeError("Checkpoint SHA-256 changed during diagnostics.")
            output = _resolve(spec["output"], self.config_dir)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            state_job.update({"status": "completed", "checkpoint_sha256": before,
                              "checkpoint_sha256_after": after, "report_path": str(output),
                              "completed_at": _now(), "error": None})
        except Exception as exc:
            state_job.update({"status": "failed", "checkpoint": str(checkpoint),
                              "error": str(exc), "failed_at": _now()})
        self._persist()

    def run(self) -> int:
        with StateFileLock(self.state_path):
            while True:
                try:
                    snapshot = self._pipeline_snapshot()
                except ValueError as exc:
                    self.state.update({"status": "failed", "error": str(exc)})
                    self._persist()
                    return 2
                if snapshot is None:
                    result = "waiting"
                else:
                    ready, failure = self._ready_training_jobs(snapshot)
                    if failure == "failed":
                        self.state.update({"status": "failed", "error": "A required training job failed."})
                        self._persist()
                        return 2
                    if failure == "waiting":
                        result = "waiting"
                    else:
                        for spec in self.config["jobs"]:
                            state_job = next(item for item in self.state["jobs"]
                                             if item["training_job_name"] == spec["training_job_name"])
                            if state_job["status"] != "completed":
                                self._run_job(spec, state_job, ready[spec["training_job_name"]])
                        if all(job["status"] == "completed" for job in self.state["jobs"]):
                            self.state.update({"status": "completed", "error": None})
                            self._persist()
                            return 0
                        result = "diagnostic_failed"
                self.state["status"] = result
                self._persist()
                if result == "diagnostic_failed":
                    return 2
                self.polls += 1
                if self.max_polls is not None and self.polls >= self.max_polls:
                    return 3 if result == "waiting" else 2
                self.sleep_fn(self.config["poll_seconds"])


def run_config(path: Path, *, sleep_fn: Callable[[float], None] = time.sleep,
               max_polls: int | None = None) -> int:
    config, config_dir = load_config(path)
    state_path = _resolve(config["state_path"], config_dir)
    return TrainingDiagnostics(config, config_dir, state_path,
                               sleep_fn=sleep_fn, max_polls=max_polls).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        return run_config(args.config)
    except (ValueError, StateLockError) as exc:
        print(f"diagnostics_error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
