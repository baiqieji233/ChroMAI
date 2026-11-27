"""Loopback MiaView review for a single explicitly selected chart and track."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from maichart.webui import chart_preview, preview_assets
from .hand_review_html import render_view


def serve_review(result, chart, audio, runtime_root=None, port=0):
    if runtime_root is None:
        from maichart.webui.server import WEBUI_ROOT

        runtime_root = WEBUI_ROOT / "preview_runtime" / "miaview"
    root = Path(runtime_root).resolve()
    if not preview_assets.status(root)["available"]:
        raise ValueError("请先在 WebUI 的成谱预览中安装 MiaView_Web，或用 --runtime-root 指向其安装目录。")
    html = render_view(
        result,
        title=result["title"],
        start=result["start"],
        stop=result["end"],
        chart_url="/chart",
        audio_url="/audio" if audio else None,
    ).encode("utf-8")
    resources = {"/chart": Path(chart).resolve()}
    if audio:
        resources["/audio"] = Path(audio).resolve()

    class Handler(BaseHTTPRequestHandler):
        def serve(self, head):
            route = urlsplit(self.path).path
            if route in resources:
                if not resources[route].is_file():
                    return self.send_error(404)
                return chart_preview.serve_file(self, resources[route], head=head)
            if route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if route.startswith("/miaview/"):
                try:
                    asset = preview_assets.asset_path(root, unquote(route[len("/miaview/") :]))
                    if not asset.is_file():
                        raise FileNotFoundError()
                    return chart_preview.serve_file(self, asset, head=head, immutable=True)
                except (OSError, ValueError):
                    return self.send_error(404)
            if route == "/":
                mime = "text/html; charset=utf-8"
                content = html
            elif route == "/api/preview/status":
                mime = "application/json"
                content = json.dumps(preview_assets.status(root)).encode()
            else:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            if not head:
                self.wfile.write(content)

        def do_GET(self):
            return self.serve(False)

        def do_HEAD(self):
            return self.serve(True)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as http:
        print(f"在浏览器打开 http://127.0.0.1:{http.server_port}/  ·  Ctrl+C 关闭预览", flush=True)
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
