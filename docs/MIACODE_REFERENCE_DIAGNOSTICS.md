# MiaCode Reference Diagnostics

Generation and WebUI now use the in-process native judge described in
`MIACODE_NATIVE_PLAYABILITY.md`. The external wrapper below remains an offline
developer comparison tool; it is not used by the generation pipeline and does
not require the user to run MiaCode.exe.

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

## Local verified build and real charts (2026-10-11)

An independent build is available at `.scratch/muri_build/build/Release/miacode_muri_dump.exe`. Its SHA-256 is `a1346998ca675939aa49c7179562ebe14039bce7d8c466d22ce188465e2f4790`. The source checkout was verified at the declared commit with a clean worktree. `build_provenance.json` in that scratch directory records the binary, standalone CMake file and Qt base archive hashes. This local build record does not make arbitrary user-supplied binaries source-verified.

The build uses Qt 6.8.3 base in `.scratch/muri_build/qt-sdk`; it does not modify the user's running MiaCode installation. Set its DLL path for the current PowerShell process before invoking the wrapper:

```powershell
$env:PATH = (Resolve-Path .scratch/muri_build/qt-sdk/bin).Path + ';' + $env:PATH
python tools/evaluate_miacode_reference.py `
  --executable .scratch/muri_build/build/Release/miacode_muri_dump.exe `
  --maidata "webui_workspace/datasets/full训练集/Splash/11207/maidata.txt" `
  --difficulty 5 --output .scratch/ether_strike_reference.json
```

The driver has run successfully on a Slide/Tap fixture and seven real-chart inputs, including the authored Ether Strike chart and generated variants. The authored chart had zero runtime/static references; generated variants exposed runtime findings absent from the internal static check. Reports are under `.scratch/full_v3_stage1/quality_probe_epoch7/miacode_reference.json` and `quality_probe_condition_fix/miacode_reference.json`. Neither findings nor zero findings are a human-play certification. MiaCode counts helper heads and slide paths differently from maichart, so raw note counts must not be compared without matching their semantics.
