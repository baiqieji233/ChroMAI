"""Render synchronized phrase assignments in a browser or as a standalone file."""

import json
import os
from importlib import resources
from pathlib import Path
from urllib.parse import quote


def source_layer(chart, clock, start, stop):
    events = [
        dict(second=clock.at(p.beat), beat=str(p.beat), notes=p.note_text)
        for p in chart.points
        if start <= clock.at(p.beat) <= stop
    ]
    return dict(events=events)


def render_view(
    result, *, title, start, stop, audio_url=None, chart_url=None, chart_text=None, serve_command=None
):
    payload = dict(
        title=title,
        start=float(start),
        stop=float(stop),
        audio=audio_url,
        chart_url=chart_url,
        chart_text=chart_text,
        slot=int(result.get("slot", 5)),
        solutions=result["solutions"],
        unknown=result["coverage"].get("unknown_count", 0),
        model_events=result.get("model_events"),
        source=result.get("source_layer"),
        serve_command=serve_command,
    )
    # Chart titles and paths may contain HTML or a literal closing script tag.
    data = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    static = resources.files("maichart.webui").joinpath("static")
    template = static.joinpath("hand_review.html").read_text(encoding="utf-8")
    overlay = "\n".join(static.joinpath(name).read_text(encoding="utf-8")
                        for name in ("hand_overlay_motion.js", "hand_overlay.js"))
    return template.replace("HAND_REVIEW_OVERLAY", overlay, 1).replace("HAND_REVIEW_PAYLOAD", data, 1)


def write_view(path, result, *, title, audio_path=None, chart_text=None, serve_command=None, start, stop):
    destination = Path(path).resolve()
    audio_url = None
    if audio_path:
        audio = Path(audio_path).resolve()
        if destination == audio:
            raise ValueError("请为手顺预览选择独立的 HTML 文件")
        try:
            audio_url = quote(Path(os.path.relpath(audio, destination.parent)).as_posix(), safe="/")
        except ValueError:  # Different drives on Windows.
            audio_url = audio.as_uri()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_view(
            result,
            title=title,
            start=start,
            stop=stop,
            audio_url=audio_url,
            chart_text=chart_text,
            serve_command=serve_command,
        ),
        encoding="utf-8",
    )
