# Training Diagnostics Observer

`tools/run_training_diagnostics.py` watches an existing staged-validation `pipeline_state.json`. It never submits work or controls a training process. A configured diagnostic starts only after its named pipeline job is `completed`, its completed result lists the exact configured immutable `event_epoch_NNN.pt`, and the checkpoint is present. Missing or temporarily unreadable state and unsubmitted/running jobs remain in `waiting`; a pipeline config digest mismatch or a failed/cancelled training job is terminal and returns nonzero.

The observer uses the evaluator APIs directly on CPU with two PyTorch threads. `event_tokens` runs the teacher-forced token evaluation with shuffled acoustic frames; `onset_alignment` runs the real, shuffled-acoustic, and time-only onset comparisons. `limit` defaults to 128 windows and `batch_size` must be 2 or 4. The accepted kinds are fixed; the config cannot name a command or Python module to execute.

Example config (replace the digest with the `config_sha256` from the source pipeline state):

```json
{
  "pipeline_state": "pipeline_state.json",
  "config_sha256": "<64-character pipeline config digest>",
  "state_path": "training_diagnostics_state.json",
  "poll_seconds": 10,
  "jobs": [
    {
      "training_job_name": "full_finetune",
      "checkpoint": "../../checkpoints/music_structure_v3_full_stage1/event_epoch_015.pt",
      "kind": "event_tokens",
      "manifest": "../../webui_workspace/datasets/full训练集/manifest.json",
      "cache_dir": "../../webui_workspace/cache/events",
      "limit": 128,
      "batch_size": 4,
      "output": "diagnostics/event_tokens_epoch_015.json"
    },
    {
      "training_job_name": "onset_finetune",
      "checkpoint": "../../checkpoints/music_structure_v3_full_stage1/event_epoch_015.pt",
      "kind": "onset_alignment",
      "manifest": "../../webui_workspace/datasets/full训练集/manifest.json",
      "cache_dir": "../../webui_workspace/cache/events",
      "limit": 128,
      "batch_size": 2,
      "output": "diagnostics/onset_epoch_015.json"
    }
  ]
}
```

Paths are resolved relative to the diagnostics config file. `config_sha256` must equal the source pipeline state's digest. The observer hashes each checkpoint immediately before and after evaluation and records both hashes, report path, status, and any error in its own state JSON. A state-file OS lock prevents concurrent observers. Completed diagnostics are a durable completion marker and are skipped on resume; failed diagnostics remain unfinished and are retried when the observer is restarted. Checkpoint read/evaluation failures affect only diagnostics and do not alter the source training pipeline state.

Run it separately from the training controller:

```powershell
python tools/run_training_diagnostics.py ".scratch/full_v3_stage1/training_diagnostics.json"
```
