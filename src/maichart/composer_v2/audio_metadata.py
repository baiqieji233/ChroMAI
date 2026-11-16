"""Zero-dependency audio tag reading and chart export metadata resolution."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
import zlib

DEFAULT_CHART_TITLE = "Generated maimai chart"
DEFAULT_LEVEL_LABEL = "13"
MAX_ID3V2_READ_BYTES = 32 * 1024 * 1024
FFPROBE_TIMEOUT_SEC = 2.0
_ID3V1_SUFFIXES = frozenset({".mp3", ".mp2", ".mp1", ""})


def _clean_single_line(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "")
    return re.sub(r"[\r\n]+", " ", text).strip()


def _sanitize_tag_text(value: Any) -> str:
    cleaned = _clean_single_line(value)
    if not cleaned or cleaned == "None":
        return ""
    return cleaned


def _decode_synchsafe(data: bytes) -> int:
    value = 0
    for byte in data:
        value = (value << 7) | (byte & 0x7F)
    return value


def _deunsync(data: bytes) -> bytes:
    return data.replace(b"\xff\x00", b"\xff")


def _decompress_text_frame(payload: bytes) -> bytes:
    decoder = zlib.decompressobj()
    try:
        text = decoder.decompress(payload, MAX_ID3V2_READ_BYTES + 1)
    except zlib.error:
        return b""
    return text if decoder.eof and len(text) <= MAX_ID3V2_READ_BYTES else b""


def _is_valid_frame_id(frame_id: bytes) -> bool:
    return bool(frame_id) and all(65 <= byte <= 90 or 48 <= byte <= 57 for byte in frame_id)


def _decode_legacy_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    if raw.isascii():
        return raw.decode("ascii")
    for encoding in ("utf-8", "gb2312"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    # Prefer GBK over Latin-1 only when bytes contain C1 controls or paired high
    # bytes, avoiding misdecoding Western accents like "Pokémon" or "Übersetzung".
    has_c1 = any(0x80 <= byte <= 0x9F for byte in raw)
    has_consec_high = any(raw[i] >= 0x80 and raw[i + 1] >= 0x80 for i in range(len(raw) - 1))
    if has_c1 or has_consec_high:
        try:
            return raw.decode("gbk")
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1", errors="replace")


def _trim_utf16_nul(raw: bytes) -> bytes:
    even_len = len(raw) & ~1
    raw = raw[:even_len]
    for idx in range(0, even_len, 2):
        if raw[idx : idx + 2] == b"\x00\x00":
            return raw[:idx]
    return raw


def _decode_id3v2_text(payload: bytes) -> str:
    if not payload:
        return ""
    encoding = payload[0]
    raw = payload[1:]
    if not raw:
        return ""
    if encoding == 0:
        return _decode_legacy_bytes(raw.split(b"\x00", 1)[0])
    if encoding == 1:
        raw = _trim_utf16_nul(raw)
        if not raw:
            return ""
        if raw[:2] == b"\xff\xfe":
            return raw[2:].decode("utf-16-le", errors="replace").lstrip("\ufeff")
        if raw[:2] == b"\xfe\xff":
            return raw[2:].decode("utf-16-be", errors="replace").lstrip("\ufeff")
        try:
            return raw.decode("utf-16").lstrip("\ufeff")
        except UnicodeError:
            return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    if encoding == 2:
        raw = _trim_utf16_nul(raw)
        if not raw:
            return ""
        return raw.decode("utf-16-be", errors="replace").lstrip("\ufeff")
    if encoding == 3:
        raw = raw.split(b"\x00", 1)[0]
        try:
            return raw.decode("utf-8").lstrip("\ufeff")
        except UnicodeDecodeError:
            return _decode_legacy_bytes(raw).lstrip("\ufeff")
    return ""


def _parse_id3v2(path: Path, file_size: int) -> dict[str, str]:
    result = {"title": "", "artist": ""}
    if file_size < 10:
        return result
    try:
        with path.open("rb") as handle:
            header = handle.read(10)
            if len(header) != 10 or header[:3] != b"ID3":
                return result
            version = header[3]
            revision = header[4]
            flags = header[5]
            size_bytes = header[6:10]
            if version not in (2, 3, 4) or revision == 0xFF or any(byte >= 0x80 for byte in size_bytes):
                return result
            tag_size = _decode_synchsafe(size_bytes)
            if tag_size <= 0:
                return result
            read_len = min(tag_size, max(0, file_size - 10), MAX_ID3V2_READ_BYTES)
            body = handle.read(read_len)
    except OSError:
        return result

    if not body:
        return result

    offset = 0
    if version == 2:
        if flags & 0x40:
            # ID3v2.2 compression bit had no finalized framing spec.
            return result
        if flags & 0x80:
            body = _deunsync(body)
        while offset + 6 <= len(body):
            frame_id = body[offset : offset + 3]
            if frame_id[0] == 0 or not _is_valid_frame_id(frame_id):
                break
            frame_size = (body[offset + 3] << 16) | (body[offset + 4] << 8) | body[offset + 5]
            offset += 6
            if frame_size == 0:
                continue
            if offset + frame_size > len(body):
                break
            payload = body[offset : offset + frame_size]
            offset += frame_size
            target = "title" if frame_id == b"TT2" else "artist" if frame_id == b"TP1" else None
            if target and not result[target]:
                text = _sanitize_tag_text(_decode_id3v2_text(payload))
                if text:
                    result[target] = text
            if result["title"] and result["artist"]:
                break
        return result

    if version == 3:
        if flags & 0x80:
            body = _deunsync(body)
        if flags & 0x40:
            if len(body) < 4:
                return result
            ext_size = int.from_bytes(body[:4], "big")
            if 4 + ext_size > len(body):
                return result
            offset = 4 + ext_size
    else:
        if flags & 0x40:
            if len(body) < 4 or any(byte >= 0x80 for byte in body[:4]):
                return result
            ext_size = _decode_synchsafe(body[:4])
            if ext_size < 6 or ext_size > len(body):
                return result
            offset = ext_size

    while offset + 10 <= len(body):
        frame_id = body[offset : offset + 4]
        if frame_id[0] == 0 or not _is_valid_frame_id(frame_id):
            break
        size_bytes = body[offset + 4 : offset + 8]
        frame_flags2 = body[offset + 9]
        if version == 3:
            frame_size = int.from_bytes(size_bytes, "big")
        elif all(byte < 0x80 for byte in size_bytes):
            frame_size = _decode_synchsafe(size_bytes)
        else:
            # Tolerate non-synchsafe sizes written by legacy v2.4 taggers only
            # when the raw integer stays within the tag bounds.
            frame_size = int.from_bytes(size_bytes, "big")
        offset += 10
        if frame_size == 0:
            continue
        if offset + frame_size > len(body):
            break
        payload = body[offset : offset + frame_size]
        offset += frame_size

        target = "title" if frame_id == b"TIT2" else "artist" if frame_id == b"TPE1" else None
        if target is None or result[target]:
            continue

        if version == 3:
            if frame_flags2 & 0x40:
                continue  # Encrypted frame
            compressed = bool(frame_flags2 & 0x80)
            grouped = bool(frame_flags2 & 0x20)
            if compressed:
                if len(payload) < 4:
                    continue
                payload = payload[4:]
                if grouped:
                    if len(payload) < 1:
                        continue
                    payload = payload[1:]
                payload = _decompress_text_frame(payload)
            elif grouped:
                if len(payload) < 1:
                    continue
                payload = payload[1:]
        else:
            if frame_flags2 & 0x04:
                continue  # Encrypted frame
            if (flags & 0x80) or (frame_flags2 & 0x02):
                payload = _deunsync(payload)
            if frame_flags2 & 0x40:
                if len(payload) < 1:
                    continue
                payload = payload[1:]
            if frame_flags2 & 0x01:
                if len(payload) < 4:
                    continue
                payload = payload[4:]
            if frame_flags2 & 0x08:
                payload = _decompress_text_frame(payload)

        text = _sanitize_tag_text(_decode_id3v2_text(payload))
        if text:
            result[target] = text
        if result["title"] and result["artist"]:
            break

    return result


def _parse_id3v1(path: Path, file_size: int) -> dict[str, str]:
    result = {"title": "", "artist": ""}
    if file_size < 128:
        return result
    try:
        with path.open("rb") as handle:
            handle.seek(file_size - 128)
            tail = handle.read(128)
    except OSError:
        return result
    if len(tail) != 128 or tail[:3] != b"TAG":
        return result
    result["title"] = _sanitize_tag_text(_decode_legacy_bytes(tail[3:33].split(b"\x00", 1)[0]))
    result["artist"] = _sanitize_tag_text(_decode_legacy_bytes(tail[33:63].split(b"\x00", 1)[0]))
    return result


def _extract_ffprobe_tags(tag_dict: Any, result: dict[str, str]) -> None:
    if not isinstance(tag_dict, dict):
        return
    lowered = {str(key).strip().lower(): value for key, value in tag_dict.items()}
    for key in ("title", "artist"):
        if not result[key] and key in lowered:
            cleaned = _sanitize_tag_text(lowered[key])
            if cleaned:
                result[key] = cleaned


def _probe_ffprobe_metadata(path: Path) -> dict[str, str]:
    result = {"title": "", "artist": ""}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return result
    try:
        kwargs: dict[str, Any] = {}
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format_tags=title,artist:stream_tags=title,artist",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFPROBE_TIMEOUT_SEC,
            check=False,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return result
    if completed.returncode != 0 or not completed.stdout:
        return result
    try:
        payload = json.loads(completed.stdout)
    except (ValueError, TypeError):
        return result
    if not isinstance(payload, dict):
        return result
    fmt = payload.get("format")
    if isinstance(fmt, dict):
        _extract_ffprobe_tags(fmt.get("tags"), result)
    streams = payload.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict):
                _extract_ffprobe_tags(stream.get("tags"), result)
            if result["title"] and result["artist"]:
                break
    return result


def read_audio_metadata(audio_path: str | os.PathLike[str] | None) -> dict[str, str]:
    """Read sanitized ``title`` and ``artist`` tags without raising on bad files."""
    empty = {"title": "", "artist": ""}
    if audio_path is None:
        return empty
    if isinstance(audio_path, str) and not audio_path.strip():
        return empty
    try:
        path = Path(audio_path)
        if not path.is_file():
            return empty
        file_size = path.stat().st_size
    except (OSError, TypeError, ValueError):
        return empty
    if file_size <= 0:
        return empty

    result = _parse_id3v2(path, file_size)
    if (not result["title"] or not result["artist"]) and path.suffix.lower() in _ID3V1_SUFFIXES:
        v1 = _parse_id3v1(path, file_size)
        if not result["title"] and v1["title"]:
            result["title"] = v1["title"]
        if not result["artist"] and v1["artist"]:
            result["artist"] = v1["artist"]

    if not result["title"] or not result["artist"]:
        probed = _probe_ffprobe_metadata(path)
        if not result["title"] and probed["title"]:
            result["title"] = probed["title"]
        if not result["artist"] and probed["artist"]:
            result["artist"] = probed["artist"]

    return result


def format_level_label(level: float) -> str:
    """Convert numeric chart constant (e.g. 13.6) to maimai display label ('13+')."""
    if isinstance(level, bool):
        return DEFAULT_LEVEL_LABEL
    try:
        numeric = float(level)
    except (TypeError, ValueError):
        return DEFAULT_LEVEL_LABEL
    if not math.isfinite(numeric) or numeric <= 0:
        return DEFAULT_LEVEL_LABEL
    tenths = int(round(numeric * 10))
    whole, frac = divmod(tenths, 10)
    if whole >= 7 and frac >= 6:
        return f"{whole}+"
    return f"{whole}"


def validate_difficulty(difficulty: Any) -> int:
    """Validate Maidata export slot (1..6, rejecting booleans)."""
    if isinstance(difficulty, bool) or not isinstance(difficulty, int) or difficulty not in range(1, 7):
        raise ValueError("difficulty must be an integer from 1 to 6.")
    return difficulty


def resolve_chart_metadata(
    audio_path: str | os.PathLike[str] | None,
    *,
    level: float,
    title: str | None = None,
    artist: str | None = None,
    designer: str | None = None,
    difficulty: int = 5,
    default_designer: str = "Composer v2",
) -> dict[str, Any]:
    """Resolve safe single-line chart export metadata from overrides and audio tags."""
    validated_difficulty = validate_difficulty(difficulty)
    level_label = format_level_label(level)

    resolved_title = _clean_single_line(title) if title is not None else ""
    audio_meta: dict[str, str] | None = None

    def get_audio_meta() -> dict[str, str]:
        nonlocal audio_meta
        if audio_meta is None:
            audio_meta = read_audio_metadata(audio_path)
        return audio_meta

    if not resolved_title:
        resolved_title = get_audio_meta()["title"]
    if not resolved_title:
        stem = ""
        if audio_path is not None and (not isinstance(audio_path, str) or audio_path.strip()):
            try:
                stem = _clean_single_line(Path(audio_path).stem)
            except (TypeError, ValueError):
                stem = ""
        resolved_title = stem or DEFAULT_CHART_TITLE

    if artist is not None:
        resolved_artist = _clean_single_line(artist)
    else:
        resolved_artist = get_audio_meta()["artist"]

    if designer is not None:
        resolved_designer = _clean_single_line(designer)
    else:
        resolved_designer = _clean_single_line(default_designer) or "Composer v2"

    return {
        "title": resolved_title,
        "artist": resolved_artist,
        "designer": resolved_designer,
        "difficulty": validated_difficulty,
        "level_label": level_label,
    }
