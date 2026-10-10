# MiaCode Reference Diagnostics

`tools/evaluate_miacode_reference.py` is an offline wrapper around a user-provided headless `miacode_muri_dump` executable. The executable is a Qt developer build with a headless CLI entry point; this tool invokes that CLI directly and makes no assumption that the MiaCode GUI can be automated or queried.

The wrapper passes argument arrays to `subprocess.run` with `shell=False`, requests marker output, and runs each input separately. It records executable and maidata SHA-256 identities before/after use, the exact argv and exit code, raw stdout/stderr and decoded JSON, plus the `parse`, `errors`, `diagnostics`, `static_references`, `marker_states`, and `summary` fields. It preserves engine diagnostics as engine-reference data; it does not translate them into maichart hard conflicts, repair charts, or claim playability.

The source revision is recorded as declared reference commit `6573efb611056bccd046dd17bc2f4cceda78bf88` with `commit_verification: declared_only`. The executable hash identifies the binary that was actually invoked, but does not prove which source revision produced that binary. `engine_reference_only` is always true and `playable_verified` is always false.

```powershell
python tools/evaluate_miacode_reference.py `
  --executable "C:\dev\MiaCode\build\miacode_muri_dump.exe" `
  --maidata "webui_workspace\datasets\full训练集\song one\maidata.txt" `
  --maidata "webui_workspace\datasets\full训练集\song two\maidata.txt" `
  --difficulty 5 `
  --first -0.125 `
  --timeout-sec 30 `
  --output ".scratch\miacode_reference.json"
```

`--first` is optional and, when given, is passed as an explicit override to each invocation. The default difficulty is 5; valid values are 1 through 7. Timeout is per chart and must be in `(0, 60]` seconds. Output must be a separate JSON path and cannot alias an input or the executable. Paths with spaces and Unicode are passed as individual argv values.

The expected driver JSON schema has boolean root `ok`; `parse` with boolean `ok`, nonnegative integer `error_count` and `warning_count`, and array `errors`/`warnings`; arrays `diagnostics`, `static_references`, `marker_states`; and object `summary`. Root/parse failure, parser errors, nonzero exit, timeout, malformed JSON, or schema errors make that chart fail. All entries are still written to the report, including successes when another input fails, and the command exits 2 for partial or failed runs. Parser warnings and reference diagnostics are retained without making a successful parse fail.
