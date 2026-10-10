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

HTTP timeouts are capped at 20 seconds and polling delay at 30 seconds. Completion writes a JSON chart-metrics report and a Markdown stage log. A clean tool run returns zero only when all jobs succeeded and evaluation has no errors. Continued generation failures, partial evaluation, evaluation exceptions, or zero evaluated charts with errors finish as `completed_with_errors` and return nonzero, with counts/details retained. `human_review` remains `pending`; no automatic playability pass is produced.
