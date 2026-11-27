"""Review a Simai chart's simultaneous two-hand paths in audio seconds."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys

from maichart.event_timing import parse_event_chart
from maichart.tempo_clock import TempoClock
from .hand_flow import build_motion_stream, recognize_hands


def main(argv=None):
    parser = argparse.ArgumentParser(description="查看舞萌谱面的乐句手顺和备选交接方式")
    parser.add_argument("maidata", type=Path, help="maidata.txt 的路径")
    parser.add_argument("--slot", type=int, choices=range(1, 7), default=5, help="难度槽位，默认 5（Master）")
    parser.add_argument("--start", type=float, default=0.0, help="从音频开头起算的秒数，默认 0")
    parser.add_argument("--end", type=float, help="结束秒数，默认从起点分析 20 秒")
    parser.add_argument("--output", type=Path, help="保存 JSON 诊断，省略则打印到终端")
    parser.add_argument("--html", type=Path, help="保存可交互的手顺预览 HTML")
    parser.add_argument("--audio", type=Path, help="预览音乐，默认查找谱面旁的 track.mp3 / ogg / wav / flac")
    parser.add_argument(
        "--solutions", type=int, default=3, choices=range(1, 17), help="最多显示几种接近的解，默认 3"
    )
    parser.add_argument("--serve", action="store_true", help="在本机用 MiaView_Web 打开手顺预览；Ctrl+C 结束")
    parser.add_argument("--port", type=int, default=0, help="预览服务端口；0 自动分配")
    parser.add_argument(
        "--runtime-root", type=Path, help="MiaView_Web 安装目录（默认读取项目的 webui_workspace）"
    )
    args = parser.parse_args(argv)
    path = args.maidata.resolve()
    audio = args.audio or next(
        (p for ext in ("mp3", "ogg", "wav", "flac") if (p := path.with_name("track." + ext)).is_file()), None
    )
    if audio and not audio.is_file():
        parser.error(f"找不到音频文件 {audio}")
    outputs = [p.resolve() for p in (args.output, args.html) if p]
    if len(set(outputs)) != len(outputs) or any(
        p in {path, audio.resolve() if audio else None} for p in outputs
    ):
        parser.error("请为报告和预览分别指定独立的输出文件")
    start = args.start
    stop = start + 20 if args.end is None else args.end
    if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
        parser.error("请输入有效的音频秒数，结束时间需晚于开始时间")
    chart = parse_event_chart(path, args.slot, preserve_each_groups=True)
    if not chart.points:
        parser.error("所选难度没有音符")
    clock = TempoClock(chart.tempo_changes, chart.audio_first)
    stream = build_motion_stream(chart.points, chart.tempo_changes, clock)
    observed = max(stop, clock.at(chart.points[-1].beat), max(stream.job_ends, default=stop))
    result = recognize_hands(
        chart.points,
        chart.tempo_changes,
        clock,
        start,
        stop,
        observed,
        motion_stream=stream,
        max_solutions=args.solutions,
    )
    from .hand_review_html import source_layer

    result["source_layer"] = source_layer(chart, clock, start, stop)
    title = chart.title or path.parent.name
    result.update(schema="dx-hand-phrase-v1", title=title, slot=args.slot, start=start, end=stop)
    for target in outputs:
        target.parent.mkdir(exist_ok=True, parents=True)
    if args.output or not (args.html or args.serve):
        text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    if args.html:
        from .hand_review_html import write_view

        from maichart.maidata import parse_maidata_file

        command = [
            sys.executable,
            "-m",
            "maichart.composer_v2.hand_review",
            str(path),
            "--slot",
            str(args.slot),
            "--start",
            str(start),
            "--end",
            str(stop),
            "--solutions",
            str(args.solutions),
            "--serve",
        ]
        if audio:
            command.extend(("--audio", str(audio.resolve())))
        if args.runtime_root:
            command.extend(("--runtime-root", str(args.runtime_root.resolve())))
        source_root = str(Path(__file__).resolve().parents[2])
        if os.name == "nt":

            def quote_ps(arg):
                return "'" + arg.replace("'", "''") + "'"

            launch = "$env:PYTHONPATH = " + quote_ps(source_root) + "\n& " + " ".join(map(quote_ps, command))
        else:
            launch = "PYTHONPATH=" + shlex.quote(source_root) + " " + shlex.join(command)
        write_view(
            args.html,
            result,
            title=title,
            audio_path=audio,
            start=start,
            stop=stop,
            chart_text=parse_maidata_file(path).raw,
            serve_command=launch,
        )
    if args.serve:
        from .hand_review_server import serve_review

        serve_review(result, path, audio, args.runtime_root, args.port)
    return result


if __name__ == "__main__":
    main()
