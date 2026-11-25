"""Export the playable audio beside a Composer v2 chart."""
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile


def track_path(audio, folder):
    suffix = Path(audio).suffix.lower()
    if suffix not in {'.mp3', '.wav', '.ogg'}:
        if shutil.which('ffmpeg') is None:
            raise ValueError('导出谱面包需要 FFmpeg 转换该音频，请安装 FFmpeg 或输入 MP3、WAV、OGG。')
        suffix = '.wav'
    return Path(folder) / ('track' + suffix)


def export_track(audio, folder):
    audio = Path(audio)
    target = track_path(audio, folder)
    if audio.resolve() == target.resolve():
        return target
    with NamedTemporaryFile(dir=target.parent, suffix=target.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if audio.suffix.lower() == target.suffix:
            shutil.copyfile(audio, temporary)
        else:
            subprocess.run(
                ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio),
                 '-map', '0:a:0', '-c:a', 'pcm_s16le', str(temporary)],
                check=True, capture_output=True,
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
