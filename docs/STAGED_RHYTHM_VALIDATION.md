# Staged Rhythm Validation Controller

The controller observes an existing training task, then submits the ordered jobs from an operator-owned JSON config. It never submits the configured `initial_task_id`; that ID is only polled.

```json
{
  "base_url": "http://127.0.0.1:7862",
  "initial_task_id": "d32a25d95f70",
  "jobs": [
    {
      "name": "generate-song-835",
      "endpoint": "/api/generate",
      "payload": {},
      "continue_on_failure": true
    }
  ],
  "state_path": ".scratch/full_v3_stage1/controller_state.json",
  "report_path": ".scratch/full_v3_stage1/controller_report.md",
  "evaluation_report_path": ".scratch/full_v3_stage1/generated_chart_metrics.json",
  "evaluation_paths": [".scratch/full_v3_stage1/generated"],
  "poll_seconds": 10,
  "http_timeout_sec": 20
}
```

Replace each job payload with the request produced for that song. The controller sends jobs in array order. It GET-polls the task ID returned from each POST and records the ID before moving on. A failed `/api/train` job always stops later work; a generation job can set `continue_on_failure` to keep the rest of the panel moving.

Run the same config path to resume after a controller exit:

```powershell
python tools/run_staged_rhythm_validation.py ".scratch/full_v3_stage1/staged_validation.json"
```

The state file prevents repeat POSTs. If a POST times out before returning a task ID, the state becomes `submit_uncertain` and the controller will not submit it again. Reconcile that request in the WebUI, then add the discovered `task_id` to that job's state entry and set its status to `submitted`; rerun the same config. After three failed GET observations, the state becomes `observation_unavailable` with the same live task ID; rerunning later resumes polling that ID. Ctrl+C records `controller_interrupted` and leaves the server task alone.

Only one controller can use a given state path at a time. A separate `<state_path>.lock` file carries a nonblocking OS lock (`msvcrt` on Windows, `flock` on POSIX); its presence after exit is normal and does not mean the state is locked. A second runner fails with an explicit message before it reads or updates state.

HTTP timeouts are capped at 20 seconds and polling delay at 30 seconds. Completion writes a JSON chart-metrics report, a same-stem Markdown rhythm review (for example `evaluation.json` -> `evaluation.md`), and a Markdown stage log. The state records the review path as `evaluation.review_report_path`; the controller report links both machine-readable and human-review artifacts. Review rendering is derived by default and does not require a config field. If rendering fails, the JSON report is retained, `evaluation.review_error` is recorded, evaluation becomes `partial`, and the controller returns nonzero. A clean tool run returns zero only when all jobs succeeded and evaluation has no errors. Continued generation failures, partial evaluation, evaluation exceptions, or zero evaluated charts with errors finish as `completed_with_errors` and return nonzero, with counts/details retained. `human_review` remains `pending`; no automatic playability pass is produced.

For generation jobs with an explicit `payload.output`, the controller checks each expected text file and `.chart.json` file, using the WebUI candidate suffix when `candidates` is greater than one. It records file fingerprints before submission and counts only files created or changed by that task; a stale same-name file is reported as missing from the run and is excluded from evaluation. A `.quality.json` file is required only when the completed task result identifies an event model family. Jobs without an explicit output use the task result's returned paths and are marked unverifiable because the expected set cannot be inferred reliably. Any missing or unverifiable generation artifact makes evaluation partial and the controller exit nonzero. Missing or incompatible reference pairs also make evaluation partial. Explicit candidate chart files such as `song.chart_candidate_001.json` are discovered and paired. When evaluating explicit file paths, the configured evaluation directory is retained as the reference pairing root so equal song filenames in separate arms match their corresponding reference subdirectories.

A follow-on controller can skip observing an external initial task by setting `initial_task_id` to JSON `null`. To wait for an earlier local controller, set both `depends_on_state` and `depends_on_config_sha256`; copy the latter from the dependency state's `config_sha256`. The child persists `waiting_for_dependency` / `dependency` while polling that JSON state at the configured interval. It submits no jobs until the digest matches, the upstream controller is `completed`, its initial task is `completed` or `not_requested`, every upstream job is `completed`, and evaluation is `completed` or `not_requested`. A digest mismatch or upstream `failed`, `cancelled`, `completed_with_errors`, or `dependency_failed` ends the child as `dependency_failed` with a nonzero exit. Missing, unreadable, running, interrupted, and observation-unavailable dependency snapshots remain pending; rerunning the child resumes the same state and queue.
