"""Read a reference clock without importing any note data into generation."""
import base64
import binascii
from fractions import Fraction
import json
from math import isfinite

from maichart.event_timing import EventTempo, tokenize_event_timing
from maichart.maidata import decode_text_with_encoding, parse_maidata_metadata, resolve_chart_first
from maichart.tempo_clock import TempoClock


MAX_FILE_BYTES = 4 * 1024 * 1024


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        raise ValueError(f"{name} 必须是有限数值")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是有限数值") from exc
    if not isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def _preview(changes, first, source):
    clock = TempoClock(changes, 0 if first is None else first)
    if not all(isfinite(sec) for sec in clock.seconds):
        raise ValueError("变速表的时间超出可表示范围")
    return dict(first=first, first_source=source, rows=[
        dict(beat=str(t.beat), bpm=t.bpm, seconds=sec) for t, sec in zip(changes, clock.seconds)
    ])


def _difficulty_clock(chart, block):
    try:
        whole = _number(chart.wholebpm, "wholebpm")
    except ValueError:
        whole = None
    initial = whole if whole and whole > 0 else None
    first_raw = resolve_chart_first(chart, block.index)
    first = _number(first_raw, "首拍偏移") if first_raw is not None else 0.
    slot_key = f"first_{block.index}"
    source = slot_key if chart.fields.get(slot_key, "").strip() else "first" if first_raw is not None else "default"
    points = tokenize_event_timing(block.inote, initial or 120., first=first)
    if not initial and not any(d.directive_type == "bpm" for d in points[0].directives):
        raise ValueError("第 0 拍没有 BPM，且谱面没有可用的 &wholebpm")
    changes = []
    for point in points:
        # point.bpm is the effective value after all directives at this beat.
        if not changes or point.bpm != changes[-1].bpm:
            changes.append(EventTempo(point.beat, point.bpm))
    return _preview(changes, first, source)


def _json_clock(text):
    data = json.loads(text)
    first = None
    if isinstance(data, dict):
        if data.get("first") is not None:
            first = _number(data["first"], "首拍偏移")
        data = data.get("tempo_map", data.get("tempo_changes"))
    if not isinstance(data, list) or not data:
        raise ValueError("BPM JSON 需要非空的变速数组，或包含 tempo_map 和 first 的对象")
    changes = []
    for index, row in enumerate(data, 1):
        if not isinstance(row, dict) or not isinstance(row.get("beat"), (str, int)) or isinstance(row["beat"], bool):
            raise ValueError(f"第 {index} 行的 beat 应为整数或分数字符串")
        try:
            beat = Fraction(row["beat"])
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"第 {index} 行的拍位无效") from exc
        bpm = _number(row.get("bpm"), "BPM")
        if bpm <= 0 or (not changes and beat != 0) or (changes and beat <= changes[-1].beat):
            raise ValueError(f"第 {index} 行无效。BPM 须大于零，拍位须从 0 开始并严格递增")
        changes.append(EventTempo(beat, bpm))
    return _preview(changes, first, "json" if first is not None else "auto")


def read_reference_tempo(payload):
    encoded = payload.get("content")
    if not isinstance(encoded, str) or not encoded or len(encoded) > 4 * ((MAX_FILE_BYTES + 2) // 3):
        raise ValueError("请选择不超过 4 MiB 的 maidata 或 BPM JSON 文件")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("无法读取文件内容") from exc
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("文件超过 4 MiB")
    text, _ = decode_text_with_encoding(data)
    if text.lstrip().startswith(("{", "[")):
        return dict(title="", slots=[dict(index=None, level="", **_json_clock(text))])
    chart = parse_maidata_metadata(text)
    slots = []
    for block in chart.difficulties:
        if not block.inote or not block.inote.strip():
            continue
        slot = dict(index=block.index, level=block.level or "")
        try:
            slot.update(_difficulty_clock(chart, block))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            slot["error"] = str(exc)
        slots.append(slot)
    if not slots:
        raise ValueError("文件中没有可读取的 &inote_1 至 &inote_6 难度段")
    return dict(title=chart.title or "", slots=slots)
