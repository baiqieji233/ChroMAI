"""Render existing run_rhythm_evaluation JSON reports as a reviewable Markdown file."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


NOT_MEASURED = "未测量"
GAPS = (("1/2", "半拍"), ("1/3", "1/3拍"), ("1/4", "1/4拍"), ("1/6", "1/6拍"))
NOTE_TYPES = ("tap", "hold", "slide", "touch", "touch_hold")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _number(value: Any, *, digits: int = 3, percent: bool = False) -> str:
    parsed = _finite_number(value)
    if parsed is None:
        return NOT_MEASURED
    if percent:
        return f"{parsed * 100:.{digits}f}%"
    if abs(parsed - round(parsed)) < 1e-9:
        return str(int(round(parsed)))
    return f"{parsed:.{digits}f}"


def _escape(value: Any) -> str:
    text = str(value if value is not None else NOT_MEASURED)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def infer_arm(chart: dict[str, Any], report_path: Path) -> str:
    explicit = chart.get("arm") or _get(chart, "metadata", "arm")
    if explicit:
        return str(explicit)
    raw = str(chart.get("path") or "")
    parts = [part for part in raw.replace("\\", "/").split("/") if part]
    positions = [index for index, part in enumerate(parts) if part.casefold() == "generated"]
    if positions and positions[-1] + 2 < len(parts):
        # Reports usually contain generated/<run>/<arm>/song.chart.json;
        # the chart's immediate parent is the arm. Keep a fallback for the
        # older generated/<arm>/song layout.
        return parts[-2]
    if positions and positions[-1] + 1 < len(parts):
        return parts[positions[-1] + 1]
    return report_path.parent.name or "unlabeled"


def _chart_number(chart: dict[str, Any], *path: str) -> float | None:
    return _finite_number(_get(chart, *path))


def _audio_after_count(chart: dict[str, Any]) -> float | None:
    """Only expose audio-end counts when evaluator had a real duration."""
    if _get(chart, "boundaries", "audio_end_check_available") is not True:
        return None
    return _chart_number(chart, "boundaries", "notes_at_or_after_audio_end")


def _playability_count(chart: dict[str, Any], key: str) -> float | None:
    status = _get(chart, "playability_evidence", "status")
    if status in (None, "unavailable", "disabled"):
        return None
    return _chart_number(chart, "playability_evidence", "counts", key)


def _artifact_completeness(report: dict[str, Any]) -> str:
    value = report.get("artifact_completeness")
    if not isinstance(value, dict):
        value = _get(report, "evaluation", "artifact_completeness")
    if not isinstance(value, dict):
        return "not reported; generation artifact completeness is unverified"
    expected = _finite_number(value.get("expected_count"))
    produced = _finite_number(value.get("produced_count"))
    missing = _finite_number(value.get("missing_count"))
    unverifiable = _finite_number(value.get("unverifiable_count"))
    if None in (expected, produced, missing, unverifiable):
        return "incomplete completeness fields; status unverified"
    if expected == produced == missing == unverifiable == 0:
        return "not applicable; no expected generation artifacts"
    if missing > 0 or unverifiable > 0 or produced < expected:
        status = "partial"
    elif produced == expected:
        status = "complete"
    else:
        status = "inconsistent counts"
    return (f"{status}: expected={_number(expected)}, produced={_number(produced)}, "
            f"missing={_number(missing)}, unverifiable={_number(unverifiable)}")


def _macro(values: list[float | None], total_charts: int, *, percent: bool = False) -> str:
    measured = [value for value in values if value is not None and math.isfinite(value)]
    if not measured:
        return f"{NOT_MEASURED} (0/{total_charts})"
    mean = sum(measured) / len(measured)
    return f"{_number(mean, percent=percent)} ({len(measured)}/{total_charts})"


def _gap_count(chart: dict[str, Any], gap: str) -> float | None:
    histogram = _get(chart, "distribution", "onset_rhythm", "gap_histogram_beats")
    named = _get(chart, "distribution", "onset_rhythm", "named_subdivision_gap_counts")
    named_key = {"1/3": "12th", "1/4": "16th", "1/6": "24th"}.get(gap)
    if named_key and isinstance(named, dict) and named_key in named:
        return _finite_number(named[named_key])
    if isinstance(histogram, dict) and gap in histogram:
        return _finite_number(histogram[gap])
    # A histogram with a sample count is complete: omitted gaps are zero.
    if isinstance(histogram, dict) and _finite_number(
            _get(chart, "distribution", "onset_rhythm", "gap_sample_count")) is not None:
        return 0.0
    return None


def _gap_share(chart: dict[str, Any], gap: str) -> float | None:
    count = _gap_count(chart, gap)
    total = _chart_number(chart, "distribution", "onset_rhythm", "gap_sample_count")
    if count is None or total is None or total <= 0:
        return None
    return count / total


def _arm_summary(arm: str, charts: list[dict[str, Any]]) -> list[str]:
    count = len(charts)
    onset_count = [
        _chart_number(chart, "distribution", "onset_rhythm", "distinct_onset_count")
        for chart in charts
    ]
    note_count = [_chart_number(chart, "distribution", "note_count") for chart in charts]
    fine_ratio = [_chart_number(chart, "distribution", "onset_rhythm", "fine_note_ratio") for chart in charts]
    lines = [f"### {arm}", "", "每曲等权宏平均；括号为有测量值的曲数/曲谱总数。缺失字段不按0处理。", "",
             "| 曲数 | 平均note数 | 平均起点数 | 细碎ratio | 最长连续空白秒 | 越界：音频前/音频后/声明结束后 |",
             "|---:|---:|---:|---:|---:|---|",
             f"| {count} | {_macro(note_count, count)} | {_macro(onset_count, count)} | "
             f"{_macro(fine_ratio, count, percent=True)} | "
             f"{_macro([_chart_number(c, 'structure', 'longest_blank_run_sec') for c in charts], count)} | "
             f"{_macro([_chart_number(c, 'boundaries', 'notes_before_audio_start') for c in charts], count)} / "
             f"{_macro([_audio_after_count(c) for c in charts], count)} / "
             f"{_macro([_chart_number(c, 'boundaries', 'notes_at_or_after_declared_end_beat') for c in charts], count)} |",
             "", "| 间隔 | 平均起点间隔数 | 平均占比 |", "|---|---:|---:|"]
    for gap, label in GAPS:
        lines.append(f"| {label} | {_macro([_gap_count(c, gap) for c in charts], count)} | "
                     f"{_macro([_gap_share(c, gap) for c in charts], count, percent=True)} |")
    lines.extend(["", "| 类型占比 | tap | hold | slide | touch | touch_hold |",
                  "|---|---:|---:|---:|---:|---:|"])
    lines.append("| 每曲等权宏平均 | " + " | ".join(
        _macro([_chart_number(c, "distribution", "type_ratios", note_type) for c in charts],
               count, percent=True) for note_type in NOTE_TYPES) + " |")
    statuses: dict[str, int] = {}
    identity_statuses: dict[str, int] = {}
    for chart in charts:
        status = str(_get(chart, "playability_evidence", "status") or "unavailable")
        statuses[status] = statuses.get(status, 0) + 1
        identity = str(_get(chart, "reference_pair", "identity_status") or
                       _get(chart, "reference_pair", "status") or "missing")
        identity_statuses[identity] = identity_statuses.get(identity, 0) + 1
    lines.extend(["", "| 静态检查状态 | 曲数 | 平均conflicts | 平均remaining_hard |",
                  "|---|---:|---:|---:|"])
    for status, n in sorted(statuses.items()):
        selected = [c for c in charts if str(_get(c, "playability_evidence", "status") or "unavailable") == status]
        lines.append(f"| {_escape(status)} | {n} | "
                     f"{_macro([_playability_count(c, 'conflicts') for c in selected], n)} | "
                     f"{_macro([_playability_count(c, 'remaining_hard') for c in selected], n)} |")
    measured_pairs = [c for c in charts
                      if _finite_number(_get(c, "reference_pair", "onset_alignment", "f1")) is not None]
    verified_pairs = [c for c in measured_pairs
                      if _get(c, "reference_pair", "identity_status") == "audio_sha256_match"]
    filename_pairs = [c for c in measured_pairs if c not in verified_pairs]
    lines.extend(["", "| 参考 onset 宏平均 | 曲数 | 平均precision | 平均recall | 平均F1 |",
                  "|---|---:|---:|---:|---:|"])
    for label, group in (("SHA-256 verified", verified_pairs),
                         ("Unverified/filename-only", filename_pairs)):
        lines.append(f"| {label} | {len(group)}/{count} | "
                     f"{_macro([_chart_number(c, 'reference_pair', 'onset_alignment', 'precision') for c in group], count, percent=True)} | "
                     f"{_macro([_chart_number(c, 'reference_pair', 'onset_alignment', 'recall') for c in group], count, percent=True)} | "
                     f"{_macro([_chart_number(c, 'reference_pair', 'onset_alignment', 'f1') for c in group], count, percent=True)} |")
    lines.append("参考配对可信度计数：" + (", ".join(f"{_escape(k)}={v}" for k, v in sorted(identity_statuses.items())) or NOT_MEASURED) + "。")
    lines.append(f"SHA-256 verified参考指标曲数：{len(verified_pairs)}/{count}；未将其他配对身份计入该行。")
    return lines


def _format_run(run: Any) -> str:
    if not isinstance(run, dict):
        return "无符合条件片段"
    start = _finite_number(run.get("start_sec"))
    end = _finite_number(run.get("end_sec"))
    if start is None or end is None:
        return "无符合条件片段"
    keys = run.get("keys")
    keys_text = f" keys={','.join(str(k) for k in keys)}" if isinstance(keys, list) else ""
    direction = f" {run['direction']}" if run.get("direction") else ""
    return (f"{start:.3f}-{end:.3f}s{direction}{keys_text} "
            f"({run.get('onset_count', NOT_MEASURED)} onsets)")


def _chart_section(entry: dict[str, Any]) -> list[str]:
    chart = entry["chart"]
    distribution = chart.get("distribution") or {}
    onset = distribution.get("onset_rhythm") or {}
    boundaries = chart.get("boundaries") or {}
    playability = chart.get("playability_evidence") or {}
    counts = playability.get("counts") or {}
    reference = chart.get("reference_pair") or {}
    alignment = reference.get("onset_alignment") or {}
    identity = chart.get("identity") or {}
    title = identity.get("title") or Path(str(chart.get("path", "chart"))).name
    level = identity.get("level")
    difficulty = identity.get("difficulty_index")
    lines = [f"### {_escape(entry['arm'])} / {_escape(title)}",
             f"路径：`{_escape(chart.get('path') or NOT_MEASURED)}`。"
             f" level={_escape(level)}, difficulty={_escape(difficulty)}, "
             f"reference={_escape(reference.get('status'))}, "
             f"pair identity={_escape(reference.get('identity_status'))}。", "",
             "| notes | onsets | fine ratio | 空白秒 | 音频前/音频后/声明结束后 |",
             "|---:|---:|---:|---:|---|",
             f"| {_number(distribution.get('note_count'))} | "
             f"{_number(onset.get('distinct_onset_count'))} | "
             f"{_number(onset.get('fine_note_ratio'), percent=True)} | "
             f"{_number(_get(chart, 'structure', 'longest_blank_run_sec'))} | "
             f"{_number(boundaries.get('notes_before_audio_start'))} / "
             f"{_number(_audio_after_count(chart))} / "
             f"{_number(boundaries.get('notes_at_or_after_declared_end_beat'))} |", "",
             "| 起点gap | count | 占比 |", "|---|---:|---:|"]
    total_gaps = _finite_number(onset.get("gap_sample_count"))
    histogram = onset.get("gap_histogram_beats") or {}
    for gap, label in GAPS:
        raw = _finite_number(histogram.get(gap))
        share = raw / total_gaps if raw is not None and total_gaps and total_gaps > 0 else None
        lines.append(f"| {label} | {_number(raw)} | {_number(share, percent=True)} |")
    types = distribution.get("type_ratios") or {}
    lines.extend(["", "| tap | hold | slide | touch | touch_hold |",
                  "|---:|---:|---:|---:|---:|",
                  "| " + " | ".join(_number(types.get(t), percent=True) for t in NOTE_TYPES) + " |",
                  "", "| 静态状态 | conflicts | hard | remaining_hard | verified |",
                  "|---|---:|---:|---:|---|",
                  f"| {_escape(playability.get('status') or 'unavailable')} | "
                  f"{_number(_playability_count(chart, 'conflicts'))} | {_number(_playability_count(chart, 'hard'))} | {_number(_playability_count(chart, 'remaining_hard'))} | "
                  f"{_escape(playability.get('verified') if 'verified' in playability else NOT_MEASURED)} |",
                  "", "| 参考配对状态 | identity | precision | recall | F1 |",
                  "|---|---|---:|---:|---:|"])
    lines.append(f"| {_escape(reference.get('status'))} | {_escape(reference.get('identity_status'))} | "
                 f"{_number(alignment.get('precision'), percent=True)} | "
                 f"{_number(alignment.get('recall'), percent=True)} | "
                 f"{_number(alignment.get('f1'), percent=True)} |")
    tolerances = reference.get("onset_alignment_by_tolerance")
    if isinstance(tolerances, dict):
        lines.extend(["", "| 起音匹配容差 | precision | recall | F1 |", "|---|---:|---:|---:|"])
        for tolerance, measured in tolerances.items():
            lines.append(f"| {_escape(tolerance)} | "
                         f"{_number(_get(measured, 'precision'), percent=True)} | "
                         f"{_number(_get(measured, 'recall'), percent=True)} | "
                         f"{_number(_get(measured, 'f1'), percent=True)} |")
        lines.append("50 ms 在快速细分中可能匹配到相邻拍位；紧容差用于检查这种影响，均不是实机判定窗。")
    spatial = onset.get("spatial_repetition") or {}
    lines.extend(["", "试听定位（秒）", "", "| 类型 | 起止时间与片段摘要 |", "|---|---|"])
    for label, key in (("顺圈", "clockwise_adjacent"),
                       ("逆圈", "counterclockwise_adjacent"),
                       ("两键交互", "alternating_two_keys")):
        lines.append(f"| {label} | {_escape(_format_run(spatial.get(key)))} |")
    error = chart.get("error")
    if error:
        lines.extend(["", f"评估错误：`{_escape(error)}`。"])
    lines.append("")
    return lines


def render_reports(inputs: list[tuple[Path, dict[str, Any]]]) -> str:
    charts: list[dict[str, Any]] = []
    report_errors = []
    for report_path, report in inputs:
        entries = report.get("charts")
        if not isinstance(entries, list):
            report_errors.append((report_path, "报告缺少charts列表"))
            continue
        for chart in entries:
            if not isinstance(chart, dict):
                continue
            charts.append({"chart": chart, "arm": infer_arm(chart, report_path),
                           "report_path": report_path})
        for error in report.get("errors") or []:
            report_errors.append((report_path, error))

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for entry in charts:
        by_arm.setdefault(entry["arm"], []).append(entry["chart"])
    lines = ["# Generated Rhythm Review", "",
             "本报告整理结构与配对证据，不计算综合分、不自动判定谱面好坏。",
             "Arm 优先取显式arm字段，否则取 generated/<run>/<arm>/song路径中的图表父目录；旧式generated/<arm>/song路径使用generated后一级。缺失arm标签显示为unlabeled。",
             "所有 arm 宏平均按曲谱等权；缺失字段跳过并显示测量分母，不当作0。",
             "静态判定区报告不是实机可玩性认证。参考P/R/F1仅适用于当前配对，配对身份状态单独展示。", "",
             f"输入报告：{len(inputs)}；曲目记录：{len(charts)}；错误记录：{len(report_errors)}。", ""]
    for report_path, report in inputs:
        completeness = _artifact_completeness(report)
        lines.append(f"- `{_escape(report_path)}`：schema={_escape(report.get('schema'))}, "
                     f"input={_number(report.get('input_count'))}, evaluated={_number(report.get('evaluated_count'))}, "
                     f"errors={_number(report.get('error_count'))}。")
        lines.append(f"  Artifact completeness: {completeness}.")
    if report_errors:
        lines.extend(["", "## 未完成记录", "", "| 输入报告 | 错误 |", "|---|---|"])
        for path, error in report_errors:
            lines.append(f"| `{_escape(path)}` | {_escape(json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else error)} |")
    if by_arm:
        lines.extend(["", "## Arm 宏平均", ""])
        for arm, arm_charts in sorted(by_arm.items()):
            lines.extend(_arm_summary(arm, arm_charts))
            lines.append("")
    lines.extend(["## 逐曲审阅", ""])
    for entry in charts:
        lines.extend(_chart_section(entry))
    return "\n".join(lines).rstrip() + "\n"


def render_files(input_paths: list[Path], output_path: Path) -> str:
    resolved_inputs = [path.resolve() for path in input_paths]
    resolved_output = output_path.resolve()
    if resolved_output in resolved_inputs:
        raise ValueError("output path must not overwrite an input evaluation report")
    loaded = []
    for path in input_paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation report root must be an object: {path}")
        loaded.append((path, payload))
    rendered = render_reports(loaded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return rendered


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path,
                        help="one or more complete run_rhythm_evaluation JSON reports")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rendered = render_files(args.reports, args.output)
    print(f"wrote {args.output} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
