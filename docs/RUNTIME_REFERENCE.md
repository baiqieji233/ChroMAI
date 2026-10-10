# MiaCode Runtime Reference

The event CLI can optionally invoke a user-provided `miacode_muri_dump` developer build after a candidate chart compiles. This is disabled by default and never runs during window decoding.

```powershell
python -m maichart.generation.event_cli `
  --runtime-reference-mode report `
  --runtime-reference-executable .scratch/muri_build/build/Release/miacode_muri_dump.exe `
  --runtime-reference-timeout-sec 30 `
  ...
```

`report` preserves the reference JSON, parser status, diagnostics, static references, visible entries, executable hash, declared MiaCode commit, and `engine_reference_only=true` / `playable_verified=false` in `quality.runtime_reference`. It never changes the chart.

`reject` rejects a candidate when a valid reference result contains a diagnostic whose exact `alert_level` is `Muri`. Reference timeout, parser failure, invalid JSON, executable failure, or input mutation also conservatively reject the candidate as `reference_failed`; these are not converted into internal hard conflicts. Rejected candidates retain their event JSON and quality report, while the existing `.history` handling removes stale advertised text output.

The reference executable is invoked with `shell=False` against a temporary complete maidata file. It receives the generated BPM/tempo metadata, `first`, and requested difficulty. The external engine is a MiaCode algorithm reference, not an official cabinet judge or player-action simulation; reports must not be interpreted as complete playability certification.
