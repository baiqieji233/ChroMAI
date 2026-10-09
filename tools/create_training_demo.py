"""Create the small, real-audio training ZIP shipped with the project."""

from __future__ import annotations

import math
import wave
import zipfile
from pathlib import Path


def create_demo(root: str | Path = "examples/training_zip_demo") -> Path:
    base = Path(root).resolve()
    raw = base / "raw_dataset"
    for index, frequency in enumerate((220.0, 330.0), start=1):
        song_dir = raw / f"song_{index:03d}"
        song_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(song_dir / "track.wav", frequency=frequency)

    archive = base.parent / "maimai_training_demo.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(raw).as_posix())
    return archive


def _write_wav(path: Path, *, frequency: float) -> None:
    sample_rate = 22050
    duration = 6.0
    total = int(sample_rate * duration)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total):
            time_sec = index / sample_rate
            value = 0.08 * math.sin(2.0 * math.pi * frequency * time_sec)
            for beat in range(12):
                delta = abs(time_sec - beat * 0.5)
                if delta < 0.012:
                    value += 0.65 * (1.0 - delta / 0.012)
            sample = max(-1.0, min(1.0, value))
            frames.extend(int(sample * 32767).to_bytes(2, "little", signed=True))
        handle.writeframes(frames)


if __name__ == "__main__":
    print(create_demo())
