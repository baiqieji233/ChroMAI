"""CPU-only, throttled progress. Positions always describe committed events."""
from contextlib import contextmanager
import json
import time

PREFIX = 'composer_v2_generation_progress'
INTERVAL = 2.
STAGES = dict(queued='等待开始', starting='正在启动', loading_checkpoint='读取检查点', loading_model='载入模型',
    freezing_checkpoint='准备检查点副本', detecting_timing='识别 BPM 和偏移', preparing_audio='准备音频特征',
    encoding_audio='读取歌曲上下文', generating='正在写谱', comparing='比较候选写法', retrying='调整冲突位置',
    exporting='导出谱面', completed='已完成', failed='生成失败', cancelled='已中止')


def readable_line(line):
    """Compact live logs; the task's disk log retains the full JSON record."""
    prefix, _, raw = line.partition(' ')
    if prefix not in {'composer_v2_batch', PREFIX} or len(raw) > 64 * 1024:
        return line
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):return line
        if prefix == PREFIX:return '生成进度 · ' + _chart_text(value)
        failures = int(value.get('failed', 0))
        result = f"批量生成 · 已结束 {int(value.get('completed',0))+failures}/{value['count']} 张（成功 {value.get('completed',0)}"
        result += (f'，失败 {failures}' if failures else '') + '）'
        if value.get('status') == 'preparing':result += ' · ' + STAGES.get(value.get('stage'), '准备音频和模型')
        else:result += f" · 正在进行 {len(value.get('active',[]))} 张 · 等待 {value.get('queued',0)} 张"
        if 'elapsed_sec' in value:result += ' · 已用 ' + _clock(value['elapsed_sec'])
        rows = [row for row in value.get('chart_progress', []) if row.get('status') == 'running']
        if rows:result += ' ｜ ' + ' ｜ '.join(_chart_text(row) for row in rows[:4])
        if len(rows) > 4:result += f' ｜ 另 {len(rows)-4} 张详见逐张进度'
        return result
    except (ValueError, TypeError, KeyError, OverflowError):
        return line


def _clock(seconds):
    seconds=max(0,int(float(seconds)))
    return f'{seconds//60:02}:{seconds%60:02}'


def _chart_text(value):
    text=f"第 {value.get('index',1)} 张 · seed {value.get('seed','—')} · {STAGES.get(value.get('stage'),'正在生成')}"
    if value.get('audio_duration_sec'):
        percent=100 if value.get('status') in {'completed','failed'} else min(99,int(100*float(value.get('progress',0))))
        text+=f" · {_clock(value.get('audio_sec',0))}/{_clock(value['audio_duration_sec'])}（{percent}%）"
    if 'events' in value:text+=f" · {value['events']} 个事件"
    return text


class GenerationProgress:
    def __init__(self, seed=0, *, index=1, count=1, emit=None, interval=INTERVAL):
        self.started = time.monotonic()
        self.last_emit = float('-inf')
        self.interval = interval
        self.emit = emit
        self.state = dict(index=index, count=count, seed=int(seed), status='running',
                          stage='starting', events=0, progress=0., started_unix=time.time())

    def update(self, *, force=False, **values):
        self.state.update(values)
        now = time.monotonic()
        if not force and now - self.last_emit < self.interval:
            return
        self.last_emit = now
        data = dict(self.state, elapsed_sec=max(0., now - self.started), updated_at=time.time())
        if self.emit is not None:
            self.emit(data)
        else:
            print(PREFIX + ' ' + json.dumps(data, ensure_ascii=False), flush=True)

    def pulse(self):
        # Called during token decoding too, so a long candidate comparison does
        # not look frozen. This reads no tensor and never advances song progress.
        self.update()

    def position(self, clock, beat, duration, *, events, stage='generating', force=False):
        seconds = min(float(duration), max(0., float(clock.at(beat)))) if events else 0.
        self.update(force=force, stage=stage, events=int(events), beat=str(beat),
                    audio_sec=seconds, audio_duration_sec=float(duration),
                    progress=min(.995, seconds / duration) if duration > 0 else 0.)


def get_progress(args):
    value = getattr(args, '_generation_progress', None)
    if value is None or value.state.get('status') in {'completed','failed'}:
        value = GenerationProgress(getattr(args,'seed',0), count=getattr(args, '_batch_count', 1))
        args._generation_progress = value
    return value


@contextmanager
def progress_scope(args, progress):
    old = getattr(args, '_generation_progress', None)
    args._generation_progress = progress
    try:
        yield progress
    finally:
        if old is None:
            del args._generation_progress
        else:
            args._generation_progress = old


class ProgressLogReader:
    """Read only appended worker bytes, retaining incomplete UTF-8/JSON lines."""
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending = b''

    def read(self):
        try:
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                data = stream.read(64 * 1024)
                self.offset += len(data)
        except OSError:
            return []
        lines = (self.pending + data).split(b'\n')
        self.pending = lines.pop()
        if len(self.pending) > 128 * 1024:
            self.pending = b''
        records = []
        for line in lines:
            if not line.startswith((PREFIX + ' ').encode()):
                continue
            try:
                value = json.loads(line.split(b' ', 1)[1].decode('utf-8'))
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict):
                records.append(value)
        return records


def batch_fraction(report):
    rows = report.get('chart_progress', [])
    return sum(1. if row['status'] in {'completed', 'failed'} else min(.995, max(0., row.get('progress', 0.)))
               for row in rows) / max(1, report['count'])
