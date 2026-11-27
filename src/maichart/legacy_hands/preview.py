"""On-demand MiaView hand preview. Run with python -m maichart.legacy_hands.preview."""
import argparse
from collections import OrderedDict, defaultdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import socket
import threading
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
import webbrowser

from maichart.event_timing import _parse_event_metadata
from maichart.maidata import decode_text_with_encoding, parse_maidata_metadata
from maichart.webui import chart_preview, preview_assets
from .player_motion import simulate_parsed_chart
from .prediction import note_hand_records
from .annotation import annotate


STATIC = Path(chart_preview.__file__).parent / 'static'
ASSETS = {'hand_preview.html', 'hand_preview.css', 'hand_preview.js', 'hand_note_colors.js',
          'hand_overlay.js', 'hand_overlay_motion.js'}
SLOTS = range(1, 8)
MAX_CHART_BYTES = 8 * 1024 * 1024
SERVICE = 'chromai-hand-preview-v1'


class PreviewServer(ThreadingHTTPServer):
    allow_reuse_address = False
    allow_reuse_port = False

    def server_bind(self):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def project_playback(raw):
    machine = raw['machine']
    angle = math.radians(machine['tilt_deg_from_vertical'])
    radius = machine['button_radius_mm']
    frames = []
    owners = defaultdict(list)
    for row in raw['frames']:
        frame = {'second': row[0]}
        touched = {}
        for side, label in enumerate(('left', 'right'), 1):
            (x, y, z), _, sites = row[4][side-1]
            frame[label] = [dict(x=round(x/radius, 6), y=round(-(y*math.cos(angle)-z*math.sin(angle))/radius, 6),
                                 kind='hypothesis', job=-1)]
            for site in sites:
                source = site[7]
                touched[source] = touched.get(source, 0) | side
        for source, side in touched.items():
            if side != 3 and (not owners[source] or owners[source][-1][1] != side):
                owners[source].append([row[0], side])
        frames.append(frame)
    return dict(frames=frames, noteHands=note_hand_records(raw['notes'], owners),
                summary=raw['summary'], duration=frames[-1]['second'] if frames else 0.)


class ReviewCache:
    """Compute each chart/slot/style once, with bounded in-process storage."""
    def __init__(self):
        self.results = OrderedDict()
        self.lock = threading.RLock()

    def compute(self, text, metadata, slot, style):
        key = (hashlib.sha256(text.encode()).digest(), slot, style)
        with self.lock:
            if key in self.results:
                self.results.move_to_end(key)
                return self.results[key]
            chart = _parse_event_metadata(metadata, slot, strict_notes=False, preserve_each_groups=True)
            if style == 'labels':
                hands = [json.loads(self.compute(text, metadata, slot, mode))['noteHands']
                         for mode in ('continuous','relay')]
                result = annotate(chart, *hands)
            else:
                result = project_playback(simulate_parsed_chart(chart, sweep_style=style))
            payload = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
            self.results[key] = payload
            while len(self.results) > 1 and sum(map(len, self.results.values())) > 48*1024*1024:
                self.results.popitem(last=False)
            return payload


def player_status(root):
    state = preview_assets.status(root)
    state['tinted'] = False
    if state['available']:
        # A hand-enabled build can be installed alongside the normal player.
        # Use its own engine and WASM together without changing current.json.
        releases = root/'releases'
        candidates = [releases/state['release'], *sorted(releases.iterdir(), key=lambda p:p.stat().st_mtime, reverse=True)]
        for directory in candidates:
            engine = directory/'renderer/player-engine.mjs'
            if (re.fullmatch('[a-f0-9]{20}', directory.name) and engine.is_file()
                    and (engine.parent/'miacode-preview.wasm').is_file()
                    and (engine.parent/'index.html').is_file()
                    and b'miaHandNoteColors' in engine.read_bytes()):
                state.update(tinted=True, renderer_url=f'/miaview/{directory.name}/renderer/index.html')
                break
    return state


def handler_for(root):
    cache = ReviewCache()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data):
            content = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(content)

        def do_POST(self):
            if self.headers.get('X-ChroMAI-Preview') != '1':
                return self.reply(403, {'error': '请从手顺预览页操作。'})
            route = urlsplit(self.path).path
            try:
                if route == '/api/renderer/install':
                    preview_assets.start_install(root)
                    return self.reply(202, player_status(root))
                action = re.fullmatch(r'/api/compute/([1-7])/(continuous|relay|labels)', route)
                if route != '/api/chart' and not action:
                    return self.reply(404, {'error': '未知的预览操作。'})
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= MAX_CHART_BYTES:
                    return self.reply(400, {'error': '请选择非空的 Simai 文本谱面（上限 8 MB）。'})
                text, _ = decode_text_with_encoding(self.rfile.read(size))
                metadata = parse_maidata_metadata(text, difficulty_range=SLOTS)
                slots = [dict(slot=d.index, level=d.level or '', designer=d.designer or metadata.fields.get('des', ''))
                         for d in metadata.difficulties if d.inote and d.inote.strip()]
                if not slots:
                    raise ValueError('文件中没有可读取的 &inote 难度段。')
                if action:
                    slot = int(action[1])
                    if slot not in {d['slot'] for d in slots}:
                        raise ValueError('这张谱面没有所选难度。')
                    return self.reply(200, cache.compute(text, metadata, slot, action[2]))
                return self.reply(200, dict(title=metadata.title, artist=metadata.artist, slots=slots, text=text))
            except (ValueError, ArithmeticError) as exc:
                self.reply(400, {'error': str(exc)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as exc:
                self.reply(500, {'error': f'手顺计算失败：{exc}'})

        def do_GET(self):
            route = unquote(urlsplit(self.path).path)
            try:
                if route == '/api/identity':
                    return self.reply(200, {'service': SERVICE})
                if route == '/api/renderer':
                    return self.reply(200, player_status(root))
                if route == '/':
                    asset = STATIC/'hand_preview.html'
                elif route[1:] in ASSETS:
                    asset = STATIC/route[1:]
                elif route.startswith('/miaview/'):
                    asset = preview_assets.asset_path(root, route[len('/miaview/'):])
                else:
                    return self.send_error(404)
                if not asset.is_file():
                    return self.send_error(404)
                chart_preview.serve_file(self, asset, head=self.command=='HEAD', immutable=route.startswith('/miaview/'))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except (OSError, ValueError):
                self.send_error(404)

        do_HEAD = do_GET

        def log_message(self, *args):
            pass
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description='ChroMAI 手顺预览')
    parser.add_argument('--port', type=int, default=17868)
    parser.add_argument('--runtime-root', type=Path, default=Path('webui_workspace/preview_runtime/miaview'))
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args(argv)
    address = ('127.0.0.1', args.port)
    try:
        server = PreviewServer(address, handler_for(args.runtime_root.resolve()))
    except OSError:
        url = f'http://127.0.0.1:{args.port}/'
        try:
            with urlopen(url+'api/identity', timeout=2) as response:
                existing = json.load(response).get('service') == SERVICE
        except (OSError, ValueError):
            existing = False
        if not existing:
            parser.exit(1, f'无法监听端口 {args.port}，可用 --port 另选端口。\n')
        print(f'手顺预览已运行：{url}', flush=True)
        if args.open_browser:
            webbrowser.open(url)
        return
    with server:
        url = f'http://127.0.0.1:{server.server_port}/'
        print(f'手顺预览：{url}  ·  Ctrl+C 关闭', flush=True)
        if args.open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
