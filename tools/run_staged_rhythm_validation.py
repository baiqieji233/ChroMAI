"""Observe one training task, run ordered WebUI jobs, then describe chart outputs."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

SUCCESS_STATUSES = {"complete", "completed", "success", "succeeded", "finished", "done"}
FAILURE_STATUSES = {"failed", "error", "failure"}
CANCELLED_STATUSES = {"cancelled", "canceled"}
NONRESUMABLE_FINAL = {"completed", "completed_with_errors", "failed", "cancelled", "dependency_failed"}


class HTTPResponseError(RuntimeError):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:1000]}")
        self.code = int(code)
        self.body = body


class StateLockError(RuntimeError):
    pass


class StateFileLock:
    """Nonblocking process lock; lock-file presence alone has no meaning."""

    def __init__(self, state_path: Path) -> None:
        self.path = state_path.with_name(state_path.name + ".lock")
        self.handle = None
        self._windows_locked = False

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
                try:
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK, 13, 36}:
                        raise StateLockError(
                            f"Controller state is already locked: {self.path}; do not start another runner for this state."
                        ) from exc
                    raise
                self._windows_locked = True
            else:
                import fcntl

                try:
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in {errno.EACCES, errno.EAGAIN}:
                        raise StateLockError(
                            f"Controller state is already locked: {self.path}; do not start another runner for this state."
                        ) from exc
                    raise
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.handle is None:
            return False
        try:
            if os.name == "nt" and self._windows_locked:
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif os.name != "nt":
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None
        return False


class HTTPJsonClient:
    def request(self, method: str, url: str, *, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise HTTPResponseError(exc.code, text) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object from {method} {url}.")
        return data


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Controller config must be a JSON object.")
    for key in ("base_url", "initial_task_id", "jobs", "state_path", "report_path", "evaluation_paths"):
        if key not in config:
            raise ValueError(f"Controller config is missing {key!r}.")
    dependency_path = config.get("depends_on_state")
    dependency_digest = config.get("depends_on_config_sha256")
    if bool(dependency_path) != bool(dependency_digest):
        raise ValueError("depends_on_state and depends_on_config_sha256 must be provided together.")
    if dependency_digest is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(dependency_digest)):
        raise ValueError("depends_on_config_sha256 must be a 64-character SHA-256 hex digest.")
    if not isinstance(config["jobs"], list) or not isinstance(config["evaluation_paths"], list):
        raise ValueError("jobs and evaluation_paths must be JSON arrays.")
    for index, job in enumerate(config["jobs"]):
        if not isinstance(job, dict) or not job.get("name") or not job.get("endpoint"):
            raise ValueError(f"jobs[{index}] must contain name and endpoint.")
        if not isinstance(job.get("payload"), dict):
            raise ValueError(f"jobs[{index}].payload must be a JSON object.")
    return config, config_path.parent


def _task_status(payload: dict[str, Any]) -> str:
    value = payload.get("status", payload.get("state", "unknown"))
    return str(value).strip().lower().replace(" ", "_")


def _terminal_status(status: str) -> str | None:
    if status in SUCCESS_STATUSES:
        return "completed"
    if status in FAILURE_STATUSES:
        return "failed"
    if status in CANCELLED_STATUSES:
        return "cancelled"
    return None


def _candidate_path(path: Path, index: int, count: int) -> Path:
    if count == 1:
        return path
    return path.with_name(f"{path.stem}_candidate_{index + 1:03d}{path.suffix}")


def _file_fingerprint(path: Path) -> dict[str, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _generation_output_paths(job_config: dict[str, Any]) -> list[dict[str, str]] | None:
    payload = job_config.get("payload") or {}
    output = payload.get("output")
    if not output:
        return None
    output_path = _resolve_path(str(output), Path(__file__).resolve().parents[1])
    try:
        count = max(1, int(payload.get("candidates", 1)))
    except (TypeError, ValueError, OverflowError):
        return None
    artifacts: list[dict[str, str]] = []
    for index in range(count):
        text_path = _candidate_path(output_path, index, count)
        artifacts.append({"kind": "text", "path": str(text_path)})
        artifacts.append({"kind": "chart_json", "path": str(_candidate_path(output_path.with_suffix(".chart.json"), index, count))})
        artifacts.append({"kind": "quality_json", "path": str(text_path.with_suffix(".quality.json"))})
    return artifacts


class StagedRhythmValidation:
    def __init__(
        self,
        config: dict[str, Any],
        config_dir: Path,
        *,
        client: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], str] = _timestamp,
    ) -> None:
        self.config = config
        self.config_dir = config_dir
        self.client = client or HTTPJsonClient()
        self.sleep_fn = sleep_fn
        self.now_fn = now_fn
        self.base_url = str(config["base_url"]).rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        requested_timeout = float(config.get("http_timeout_sec", 20))
        requested_poll = float(config.get("poll_seconds", 10))
        if not math.isfinite(requested_timeout) or not math.isfinite(requested_poll):
            raise ValueError("http_timeout_sec and poll_seconds must be finite numbers.")
        self.http_timeout = min(20.0, max(0.1, requested_timeout))
        self.poll_seconds = min(30.0, max(0.0, requested_poll))
        self.state_path = _resolve_path(config["state_path"], config_dir)
        self.dependency_path = (_resolve_path(config["depends_on_state"], config_dir)
                                if config.get("depends_on_state") else None)
        self.dependency_digest = (str(config["depends_on_config_sha256"]).lower()
                                  if config.get("depends_on_config_sha256") else None)
        if (self.dependency_path is not None
                and os.path.normcase(str(self.dependency_path)) == os.path.normcase(str(self.state_path))):
            raise ValueError("depends_on_state must not resolve to this controller's own state_path.")
        self.report_path = _resolve_path(config["report_path"], config_dir)
        self.evaluation_paths = [_resolve_path(item, config_dir) for item in config["evaluation_paths"]]
        self.config_digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("config_sha256") != self.config_digest:
                raise ValueError(
                    "Config differs from the saved run state. Keep the original job queue; edit the state only "
                    "to attach a task_id after resolving a submit_uncertain job."
                )
            expected_initial_id = (str(self.config["initial_task_id"])
                                   if self.config["initial_task_id"] is not None else None)
            if state.get("initial", {}).get("task_id") != expected_initial_id:
                raise ValueError("Saved initial task_id differs from config; refusing to observe another task.")
            if len(state.get("jobs", [])) != len(self.config["jobs"]):
                raise ValueError("Saved job count differs from config.")
            if self.dependency_path is not None and not isinstance(state.get("dependency"), dict):
                # A controller from before dependency support may have written
                # a matching config digest while ignoring these optional keys.
                state["dependency"] = {
                    "path": str(self.dependency_path),
                    "expected_config_sha256": self.dependency_digest,
                    "status": "pending",
                }
            state["human_review"] = "pending"
            state.setdefault("evaluation", {})["human_review"] = "pending"
            return state
        state = {
            "schema": "maichart-staged-rhythm-validation-state-v1",
            "config_sha256": self.config_digest,
            "base_url": self.base_url,
            "status": "queued",
            "initial": {
                "name": "initial_training",
                "task_id": (str(self.config["initial_task_id"])
                            if self.config["initial_task_id"] is not None else None),
                "status": ("pending" if self.config["initial_task_id"] is not None else "not_requested"),
                "consecutive_observation_failures": 0,
            },
            "jobs": [
                {"name": job["name"], "endpoint": job["endpoint"], "status": "pending",
                 "task_id": str(job["task_id"]) if job.get("task_id") else None,
                 "result": None, "error": None}
                for job in self.config["jobs"]
            ],
            "current_stage": ("dependency" if self.dependency_path is not None else
                              "jobs" if self.config["initial_task_id"] is None else "initial_training"),
            "events": [],
            "evaluation": {"status": "pending", "path_count": len(self.evaluation_paths)},
            "dependency": ({"path": str(self.dependency_path),
                            "expected_config_sha256": self.dependency_digest,
                            "status": "pending"} if self.dependency_path is not None else None),
            "human_review": "pending",
            "created_at": self.now_fn(),
            "updated_at": self.now_fn(),
        }
        self.state = state
        initial_message = ("Controller initialized; the configured initial task will be observed by its saved ID."
                          if self.config["initial_task_id"] is not None
                          else "Controller initialized with no external initial task.")
        self._transition(state, "queued", initial_message)
        return state

    def _persist(self, *, markdown: bool = False) -> None:
        self.state["updated_at"] = self.now_fn()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)
        if markdown:
            self._write_markdown()

    def _transition(self, target: dict[str, Any], status: str, message: str) -> None:
        previous = target.get("status")
        target["status"] = status
        if previous != status:
            self.state["events"].append({"at": self.now_fn(), "stage": target.get("name", self.state.get("current_stage")),
                                         "from": previous, "to": status, "message": message})
            self._persist(markdown=True)
        else:
            self._persist()

    def _write_markdown(self) -> None:
        lines = [
            "# Staged Rhythm Validation",
            "",
            f"- Status: `{self.state.get('status')}`",
            f"- Updated: `{self.state.get('updated_at')}`",
            f"- Base URL: `{self.base_url}`",
            f"- Initial training task: `{self.state['initial']['task_id'] if self.state['initial'].get('task_id') is not None else 'not requested'}` ({self.state['initial']['status']})",
            "- Human review: pending",
            "",
            "## Jobs",
            "",
            "| Name | Endpoint | Status | Task ID | Error |",
            "|---|---|---|---|---|",
        ]
        for job in self.state["jobs"]:
            error = str(job.get("error") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {job['name']} | {job['endpoint']} | {job['status']} | {job.get('task_id') or ''} | {error} |")
        lines.extend(["", "## Stage Changes", ""])
        for event in self.state.get("events", []):
            lines.append(f"- `{event['at']}` `{event.get('stage')}`: `{event.get('from')}` -> "
                         f"`{event.get('to')}`. {event.get('message', '')}")
        dependency = self.state.get("dependency") or {}
        if dependency:
            lines.extend(["", "## Dependency", "",
                          f"- Status: `{dependency.get('status')}`",
                          f"- State path: `{dependency.get('path')}`",
                          f"- Observed controller status: `{dependency.get('observed_status', 'unavailable')}`"])
            if dependency.get("error"):
                lines.append(f"- Dependency error: {dependency['error']}")
        evaluation = self.state.get("evaluation") or {}
        lines.extend(["", "## Evaluation", "", f"- Status: `{evaluation.get('status')}`"])
        completeness = evaluation.get("artifact_completeness") or {}
        if completeness:
            lines.append(
                f"- Generation artifacts: expected {completeness.get('expected_count', 0)}; "
                f"produced {completeness.get('produced_count', 0)}; "
                f"missing {completeness.get('missing_count', 0)}; "
                f"unverifiable {completeness.get('unverifiable_count', 0)}"
            )
            for path in completeness.get("missing_paths", [])[:10]:
                lines.append(f"- Missing generation artifact: `{path}`")
            for path in completeness.get("stale_paths", [])[:10]:
                lines.append(f"- Existing artifact was not produced by this task: `{path}`")
        pairing = evaluation.get("reference_pairing") or {}
        if pairing.get("status") == "requested":
            lines.append(
                f"- Reference pairs: matched {pairing.get('matched', 0)}; "
                f"missing {pairing.get('missing', 0)}; incompatible {pairing.get('incompatible', 0)}"
            )
        if evaluation.get("report_path"):
            lines.append(f"- JSON report: `{evaluation['report_path']}`")
        if evaluation.get("review_report_path"):
            lines.append(f"- Markdown review: `{evaluation['review_report_path']}`")
        if evaluation.get("evaluated_count") is not None:
            lines.append(f"- Evaluated charts: {evaluation['evaluated_count']}; errors: {evaluation.get('error_count', 0)}")
        if evaluation.get("error"):
            lines.append(f"- Evaluation error: {evaluation['error']}")
        if evaluation.get("review_error"):
            lines.append(f"- Review rendering error: {evaluation['review_error']}")
        for error in evaluation.get("error_details", [])[:10]:
            lines.append(f"- Evaluation error: `{error.get('path', '')}`: {error.get('error', error)}")
        lines.extend(["- Human review: pending", "- No pass/fail playability conclusion was produced.", ""])
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def _url(self, endpoint: str, *, task_id: str | None = None) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            path = endpoint
        else:
            path = f"{self.base_url}/{endpoint.lstrip('/')}"
        if task_id is not None:
            separator = "&" if "?" in path else "?"
            path += separator + urllib.parse.urlencode({"id": task_id})
        return path

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.client.request(method, url, payload=payload, timeout=self.http_timeout)

    def _observe(self, stage: dict[str, Any]) -> tuple[str | None, int]:
        task_id = str(stage["task_id"])
        stage.setdefault("consecutive_observation_failures", 0)
        while True:
            try:
                snapshot = self._request("GET", self._url("/api/task", task_id=task_id))
            except Exception as exc:  # network loss never means the remote task stopped
                failures = int(stage.get("consecutive_observation_failures", 0)) + 1
                stage["consecutive_observation_failures"] = failures
                stage["last_observation_error"] = str(exc)
                self.state["status"] = "observing"
                self._persist()
                if failures >= 3:
                    self.state["status"] = "observation_unavailable"
                    stage["status"] = "observation_unavailable"
                    self.state["events"].append({
                        "at": self.now_fn(), "stage": stage.get("name"), "from": "observing",
                        "to": "observation_unavailable",
                        "message": f"3 consecutive GET observations failed; task {task_id} may still be live.",
                    })
                    self._persist(markdown=True)
                    return None, 3
                self.sleep_fn(self.poll_seconds)
                continue
            if not isinstance(snapshot, dict):
                snapshot = {"status": "unknown", "raw": snapshot}
            stage["consecutive_observation_failures"] = 0
            stage["last_snapshot"] = {
                key: snapshot.get(key)
                for key in ("task_id", "kind", "status", "progress", "message", "error", "result")
                if key in snapshot
            }
            raw_status = _task_status(snapshot)
            terminal = _terminal_status(raw_status)
            next_status = terminal or (raw_status if raw_status in {"queued", "running"} else "running")
            old_status = stage.get("status")
            stage["status"] = next_status
            stage["last_observed_at"] = self.now_fn()
            self.state["status"] = "running"
            self._persist(markdown=old_status != next_status)
            if terminal == "completed":
                return "completed", 0
            if terminal in {"failed", "cancelled"}:
                return terminal, 1
            self.sleep_fn(self.poll_seconds)

    def _submit_job(self, job_config: dict[str, Any], job_state: dict[str, Any]) -> bool:
        if job_state.get("task_id"):
            return True
        if job_state.get("status") in {"submitting", "submit_uncertain"}:
            job_state["status"] = "submit_uncertain"
            self.state["status"] = "submit_uncertain"
            self._persist(markdown=True)
            return False
        expected = (_generation_output_paths(job_config)
                    if self._is_generation_job(job_config) else None)
        if expected is not None:
            job_state["artifact_baseline"] = {
                item["path"]: _file_fingerprint(Path(item["path"])) for item in expected
            }
        self._transition(job_state, "submitting", "POST will be sent once; submission is persisted before network I/O.")
        try:
            response = self._request("POST", self._url(str(job_config["endpoint"])), job_config["payload"])
        except HTTPResponseError as exc:
            job_state["error"] = str(exc)
            if 400 <= exc.code < 500:
                self._transition(job_state, "failed", "Server rejected this request; it will not be resubmitted automatically.")
                return False
            self._transition(job_state, "submit_uncertain", "POST response was lost after a server error; no retry was sent.")
            self.state["status"] = "submit_uncertain"
            self._persist(markdown=True)
            return False
        except Exception as exc:
            job_state["error"] = str(exc)
            self._transition(job_state, "submit_uncertain", "POST outcome is unknown; no retry was sent.")
            self.state["status"] = "submit_uncertain"
            self._persist(markdown=True)
            return False
        task_id = response.get("task_id")
        if not task_id:
            job_state["error"] = response.get("error") or "POST response has no task_id"
            self._transition(job_state, "submit_uncertain", "Response did not identify a task; no retry was sent.")
            self.state["status"] = "submit_uncertain"
            self._persist(markdown=True)
            return False
        job_state["task_id"] = str(task_id)
        self._transition(job_state, "submitted", "Task ID saved; all future checks will GET this same ID.")
        return True

    @staticmethod
    def _is_training_job(job_config: dict[str, Any]) -> bool:
        endpoint = str(job_config.get("endpoint", "")).rstrip("/")
        if endpoint.startswith(("http://", "https://")):
            endpoint = urllib.parse.urlparse(endpoint).path.rstrip("/")
        else:
            endpoint = endpoint.split("?")[0]
        return endpoint == "/api/train" or str(job_config.get("kind", "")).lower() == "training"

    @staticmethod
    def _is_generation_job(job_config: dict[str, Any]) -> bool:
        endpoint = str(job_config.get("endpoint", "")).rstrip("/")
        if endpoint.startswith(("http://", "https://")):
            endpoint = urllib.parse.urlparse(endpoint).path.rstrip("/")
        else:
            endpoint = endpoint.split("?")[0]
        return endpoint == "/api/generate" or str(job_config.get("kind", "")).lower() == "generation"

    def _wait_for_dependency(self) -> bool:
        if self.dependency_path is None:
            return True
        dependency = self.state.get("dependency")
        if not isinstance(dependency, dict):
            dependency = {"path": str(self.dependency_path),
                          "expected_config_sha256": self.dependency_digest,
                          "status": "pending"}
            self.state["dependency"] = dependency
        self.state["status"] = "waiting_for_dependency"
        self.state["current_stage"] = "dependency"
        dependency.setdefault("name", "dependency")
        self._persist(markdown=True)
        while True:
            try:
                snapshot = json.loads(self.dependency_path.read_text(encoding="utf-8"))
                if not isinstance(snapshot, dict):
                    raise ValueError("dependency state root is not an object")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                dependency.update({"status": "waiting", "observed_status": "unavailable",
                                   "error": str(exc), "last_observed_at": self.now_fn()})
                self.state["status"] = "waiting_for_dependency"
                self.state["current_stage"] = "dependency"
                self._persist(markdown=True)
                self.sleep_fn(self.poll_seconds)
                continue

            observed_digest = snapshot.get("config_sha256")
            dependency.update({"observed_config_sha256": observed_digest,
                               "observed_status": snapshot.get("status", "unknown"),
                               "observed_updated_at": snapshot.get("updated_at"),
                               "last_state": snapshot, "last_observed_at": self.now_fn()})
            dependency.pop("error", None)
            if observed_digest != self.dependency_digest:
                return self._fail_dependency(
                    dependency,
                    f"config digest mismatch: expected {self.dependency_digest}, observed {observed_digest!r}")

            observed_status = str(snapshot.get("status", "unknown")).strip().lower()
            if observed_status in {"failed", "cancelled", "canceled", "completed_with_errors", "dependency_failed"}:
                return self._fail_dependency(
                    dependency, f"upstream controller ended with status {observed_status!r}")

            if observed_status == "completed":
                initial = snapshot.get("initial")
                jobs = snapshot.get("jobs")
                evaluation = snapshot.get("evaluation")
                initial_status = initial.get("status") if isinstance(initial, dict) else None
                initial_complete = initial_status in {"completed", "not_requested"}
                all_jobs_completed = (isinstance(jobs, list)
                                      and all(isinstance(job, dict) and job.get("status") == "completed"
                                              for job in jobs))
                evaluation_status = evaluation.get("status") if isinstance(evaluation, dict) else None
                evaluation_complete = evaluation_status in {"completed", "not_requested"}
                if initial_complete and all_jobs_completed and evaluation_complete:
                    dependency["error"] = None
                    self._transition(dependency, "completed", "Upstream controller and all jobs/evaluation completed.")
                    self.state["status"] = "running"
                    self.state["current_stage"] = (
                        "initial_training" if self.config["initial_task_id"] is not None else "jobs")
                    self._persist(markdown=True)
                    return True
                dependency.update({"status": "waiting", "reason": "upstream completion snapshot is incomplete",
                                   "initial_status": initial_status,
                                   "evaluation_status": evaluation_status,
                                   "job_statuses": [job.get("status") if isinstance(job, dict) else None
                                                    for job in jobs] if isinstance(jobs, list) else None})
            else:
                dependency.update({"status": "waiting", "reason": "upstream controller is not terminal"})

            self.state["status"] = "waiting_for_dependency"
            self.state["current_stage"] = "dependency"
            self._persist(markdown=True)
            self.sleep_fn(self.poll_seconds)

    def _fail_dependency(self, dependency: dict[str, Any], reason: str) -> bool:
        dependency.update({"status": "failed", "error": reason})
        self.state["status"] = "dependency_failed"
        self.state["current_stage"] = "dependency"
        self.state["events"].append({
            "at": self.now_fn(), "stage": "dependency", "from": "waiting_for_dependency",
            "to": "dependency_failed", "message": reason,
        })
        self._persist(markdown=True)
        return False

    def _artifact_completeness(self) -> dict[str, Any]:
        expected_count = produced_count = missing_count = unverifiable_count = 0
        missing_paths: list[str] = []
        stale_paths: list[str] = []
        produced_chart_paths: list[Path] = []
        job_reports = []
        for job_config, job_state in zip(self.config["jobs"], self.state["jobs"]):
            if not self._is_generation_job(job_config):
                continue
            status = job_state.get("status")
            task_result = job_state.get("task_result") or {}
            model_family = str(task_result.get("model_family") or
                               (job_config.get("payload") or {}).get("model_family") or "")
            explicit = _generation_output_paths(job_config)
            job_expected = 0
            job_produced = 0
            job_missing: list[str] = []
            job_unverifiable = 0
            job_stale: list[str] = []
            job_charts: list[str] = []
            if explicit is None:
                if status == "completed":
                    result_paths = [str(value) for value in (task_result.get("paths") or [])
                                    if Path(str(value)).is_file()]
                    chart_paths = [value for value in result_paths
                                   if ".chart" in Path(str(value)).name
                                   and str(value).endswith(".json")
                                   and ".original." not in str(value)]
                    text_paths = [value for value in result_paths
                                  if str(value).endswith(".txt") and ".original." not in str(value)
                                  and ".repaired." not in str(value)]
                    quality_paths = ([value for value in result_paths
                                      if str(value).endswith(".quality.json")]
                                     if model_family in {"event_v1", "event_ctx_v1"} else [])
                    job_produced = len(set(text_paths + chart_paths + quality_paths))
                    job_charts = chart_paths
                else:
                    chart_paths = []
                job_unverifiable = 1
            else:
                required = [item for item in explicit
                            if item["kind"] != "quality_json"
                            or model_family in {"event_v1", "event_ctx_v1"}]
                job_expected = len(required)
                baseline = job_state.get("artifact_baseline")
                for item in required:
                    path = Path(item["path"])
                    current = _file_fingerprint(path)
                    before = baseline.get(item["path"]) if isinstance(baseline, dict) else None
                    verified = (status == "completed" and isinstance(baseline, dict)
                                and item["path"] in baseline and current is not None
                                and current != before)
                    if verified:
                        job_produced += 1
                        if item["kind"] == "chart_json":
                            job_charts.append(item["path"])
                    else:
                        job_missing.append(item["path"])
                        if current is not None:
                            job_stale.append(item["path"])
                        if not isinstance(baseline, dict) or item["path"] not in baseline:
                            job_unverifiable += 1
            expected_count += job_expected
            produced_count += job_produced
            missing_count += max(0, job_expected - job_produced)
            unverifiable_count += job_unverifiable
            missing_paths.extend(job_missing)
            stale_paths.extend(job_stale)
            produced_chart_paths.extend(Path(path) for path in job_charts)
            job_reports.append({
                "name": job_config.get("name"), "status": status,
                "expected_count": job_expected, "produced_count": job_produced,
                "missing_count": max(0, job_expected - job_produced),
                "unverifiable_count": job_unverifiable,
                "missing_paths": job_missing, "stale_paths": job_stale,
            })
        return {
            "expected_count": expected_count, "produced_count": produced_count,
            "missing_count": missing_count, "unverifiable_count": unverifiable_count,
            "missing_paths": missing_paths, "stale_paths": stale_paths,
            "produced_chart_paths": [str(path) for path in dict.fromkeys(produced_chart_paths)],
            "jobs": job_reports,
        }

    def _evaluation(self) -> None:
        self.state["current_stage"] = "evaluation"
        evaluation = self.state["evaluation"]
        completeness = self._artifact_completeness()
        has_generation_jobs = any(self._is_generation_job(job) for job in self.config["jobs"])
        evaluation_inputs = ([Path(path) for path in completeness["produced_chart_paths"]]
                            if has_generation_jobs else self.evaluation_paths)
        if not has_generation_jobs and not evaluation_inputs:
            evaluation.update({"status": "not_requested", "human_review": "pending"})
            self._persist(markdown=True)
            return
        try:
            if evaluation_inputs:
                from tools.run_rhythm_evaluation import evaluate_paths

                pairing_root = None
                produced = [path.resolve() for path in evaluation_inputs]
                if self.config.get("reference_dir"):
                    for configured_root in self.evaluation_paths:
                        if configured_root.is_dir() and all(
                            path.is_relative_to(configured_root.resolve()) for path in produced
                        ):
                            pairing_root = configured_root
                            break
                payload = evaluate_paths(evaluation_inputs,
                                         reference_dir=(_resolve_path(self.config["reference_dir"], self.config_dir)
                                                        if self.config.get("reference_dir") else None),
                                         pairing_root=pairing_root)
            else:
                payload = {"schema": "maichart-rhythm-evaluation-v1",
                           "metric_policy": "descriptive metrics only; no aggregate quality score",
                           "input_count": 0, "evaluated_count": 0, "error_count": 0,
                           "reference_pairing": {"status": "not_requested", "matched": 0, "missing": 0},
                           "charts": [], "errors": []}
            report_path = _resolve_path(
                self.config.get("evaluation_report_path", self.report_path.with_suffix(".evaluation.json")),
                self.config_dir,
            )
            payload["artifact_completeness"] = completeness
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            review_path = report_path.with_suffix(".md")
            review_error = None
            try:
                from tools.render_rhythm_review import render_files

                render_files([report_path], review_path)
            except Exception as exc:
                # Keep the machine-readable evaluation even when the optional
                # human-review artifact cannot be rendered.
                review_error = str(exc)
            evaluated_count = int(payload.get("evaluated_count", 0))
            error_count = int(payload.get("error_count", 0))
            pairing = payload.get("reference_pairing") or {}
            pairing_incomplete = (pairing.get("status") == "requested"
                                  and (int(pairing.get("missing", 0)) > 0
                                       or int(pairing.get("incompatible", 0)) > 0))
            incomplete = (completeness["missing_count"] > 0
                          or completeness["unverifiable_count"] > 0
                          or pairing_incomplete)
            eval_status = ("partial" if incomplete or (evaluated_count > 0 and error_count > 0)
                           or review_error else
                           "completed" if evaluated_count > 0 and error_count == 0 else "failed")
            evaluation.update({"status": eval_status, "report_path": str(report_path),
                               "review_report_path": str(review_path),
                               "evaluated_count": evaluated_count, "error_count": error_count,
                               "error_details": list(payload.get("errors", [])),
                               "reference_pairing": pairing,
                               "artifact_completeness": completeness,
                               "human_review": "pending"})
            if review_error:
                evaluation["review_error"] = review_error
        except Exception as exc:
            evaluation.update({"status": "partial" if completeness["missing_count"] or completeness["unverifiable_count"] else "failed",
                               "error": str(exc), "artifact_completeness": completeness,
                               "human_review": "pending"})
        self._persist(markdown=True)

    def run(self) -> int:
        if self.state["status"] == "dependency_failed":
            return 2
        if self.state["status"] in NONRESUMABLE_FINAL:
            dependency = self.state.get("dependency") or {}
            dependency_already_satisfied = (
                self.dependency_path is None
                or (dependency.get("status") == "completed"
                    and dependency.get("expected_config_sha256") == self.dependency_digest)
            )
            if dependency_already_satisfied:
                return {"completed": 0, "completed_with_errors": 2}.get(self.state["status"], 1)
        if self.state["status"] in {"submit_uncertain", "observation_unavailable"}:
            # An operator may attach the discovered task ID in state.json after reconciliation.
            current_name = self.state.get("current_stage")
            if current_name == "initial_training":
                stage = self.state["initial"]
            else:
                index = int(self.state.get("current_job_index", 0))
                stage = self.state["jobs"][index]
            if self.state["status"] == "submit_uncertain" and not stage.get("task_id"):
                self._write_markdown()
                return 3
            self.state["status"] = "running"
            if stage.get("task_id") and stage.get("status") == "observation_unavailable":
                stage["status"] = "submitted"
            self._persist(markdown=True)
        try:
            if self.dependency_path is not None and not self._wait_for_dependency():
                return 2
            initial = self.state["initial"]
            if initial.get("status") not in {"completed", "not_requested"}:
                self.state["current_stage"] = "initial_training"
                self.state["current_job_index"] = -1
                initial.setdefault("name", "initial_training")
                self._transition(initial, "observing", "Observing the configured initial task ID; no POST is sent.")
                terminal, code = self._observe(initial)
                if terminal != "completed":
                    self.state["status"] = terminal or "observation_unavailable"
                    self._persist(markdown=True)
                    return code or 3
                self._transition(initial, "completed", "Initial training task completed successfully.")
            for index, (job_config, job_state) in enumerate(zip(self.config["jobs"], self.state["jobs"])):
                if not job_state.get("task_result"):
                    job_state["task_result"] = job_state.get("last_snapshot", {}).get("result") or {}
                if job_state["status"] in {"completed", "failed", "cancelled"}:
                    if job_state["status"] == "failed" and not job_config.get("continue_on_failure", False):
                        self.state["status"] = "failed"
                        self._persist(markdown=True)
                        return 1
                    continue
                self.state["current_stage"] = job_state["name"]
                self.state["current_job_index"] = index
                self.state["status"] = "running"
                self._persist(markdown=True)
                if not self._submit_job(job_config, job_state):
                    if (job_state["status"] == "failed" and job_config.get("continue_on_failure", False)
                            and not self._is_training_job(job_config)):
                        continue
                    if job_state["status"] == "failed":
                        self.state["status"] = "failed"
                        self._persist(markdown=True)
                        return 1
                    return 3
                if job_state["status"] != "completed":
                    self._transition(job_state, "observing", "Polling the returned task ID.")
                    terminal, code = self._observe(job_state)
                    job_state["task_result"] = job_state.get("last_snapshot", {}).get("result") or {}
                    if terminal != "completed":
                        job_state["error"] = job_state.get("last_snapshot", {}).get("error") or job_state.get("error")
                        if terminal in {"failed", "cancelled"}:
                            if (job_config.get("continue_on_failure", False)
                                    and not self._is_training_job(job_config)):
                                continue
                            self.state["status"] = terminal
                            self._persist(markdown=True)
                            return code
                        return 3
                    self._transition(job_state, "completed", "Job task completed successfully.")
            self.state["status"] = "evaluating"
            self.state["current_stage"] = "evaluation"
            self._persist(markdown=True)
            self._evaluation()
            failed_jobs = [job for job in self.state["jobs"]
                           if job.get("status") in {"failed", "cancelled"}]
            evaluation_failed = self.state["evaluation"].get("status") in {"failed", "partial"}
            final_status = "completed_with_errors" if failed_jobs or evaluation_failed else "completed"
            self.state["events"].append({
                "at": self.now_fn(), "stage": "final", "from": "evaluating", "to": final_status,
                "message": (f"continued_failed_jobs={len(failed_jobs)}; "
                           f"evaluation_status={self.state['evaluation'].get('status')}"),
            })
            self.state["status"] = final_status
            self._persist(markdown=True)
            return 2 if final_status == "completed_with_errors" else 0
        except KeyboardInterrupt:
            self.state["status"] = "controller_interrupted"
            self.state["events"].append({
                "at": self.now_fn(), "stage": self.state.get("current_stage"),
                "from": "running", "to": "controller_interrupted",
                "message": "Controller stopped by keyboard interrupt; the remote task was not cancelled.",
            })
            self._persist(markdown=True)
            return 130


def run_config(
    config_path: str | Path,
    *,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], str] = _timestamp,
) -> int:
    config, config_dir = _load_config(Path(config_path))
    state_path = _resolve_path(config["state_path"], config_dir)
    with StateFileLock(state_path):
        controller = StagedRhythmValidation(config, config_dir, client=client,
                                            sleep_fn=sleep_fn, now_fn=now_fn)
        return controller.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON queue config created by the operator")
    args = parser.parse_args(argv)
    try:
        return run_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
