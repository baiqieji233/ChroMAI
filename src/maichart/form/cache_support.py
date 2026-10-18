"""Shared cache discovery and persistent source fingerprints."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import threading


@lru_cache(maxsize=32)
def cache_roots(cache_dir):
    current = Path(cache_dir).resolve()
    roots = [current]
    # Earlier WebUI versions accidentally appended /audio to the base root.
    base = current.parent if current.name == 'audio' else current
    if base not in roots:
        roots.append(base)
    nested = base / 'audio'
    if nested.is_dir() and nested not in roots:
        roots.append(nested)
    # The stream UI originally defaulted to cache/chart_stream_v1. Moving
    # to a dataset cache must still discover those exact content identities.
    # Consumers validate hashes/settings; this is discovery, not a relaxed key.
    project=Path(__file__).resolve().parents[3]
    for candidate in (project/'cache'/'chart_stream_v1',project/'webui_workspace'/'cache'/'song_form_shared'):
        if candidate.is_dir() and candidate not in roots:roots.append(candidate)
    if base.name.lower() != 'cache':
        return tuple(roots)
    try:
        siblings = sorted(base.parent.parent.iterdir())
    except OSError:
        siblings = []
    for sibling in siblings:
        for candidate in (sibling / base.name,):
            try:
                if candidate.is_dir() and candidate not in roots:
                    roots.append(candidate)
            except OSError:
                continue
    shared = Path(__file__).resolve().parents[3] / 'webui_workspace' / 'cache' / 'song_form_shared'
    if shared.is_dir() and shared not in roots:
        roots.append(shared)
    return tuple(roots)


class SourceDigests:
    """Reuse SHA256 when file identity, timestamps and sampled bytes agree.

    This is a local filesystem cache, not tamper detection. Changed files
    are fully hashed; a stat check after reading avoids caching torn writes.
    SQLite transactions keep concurrent training processes from replacing
    each other's fingerprint inventory.
    """

    def __init__(self, cache_dir):
        self._lock = threading.RLock()
        self._files = {}
        self._memory = {}
        self._counts = dict(hits=0, full_reads=0, bytes_hashed=0)
        self._db = None
        try:
            root = Path(cache_dir)
            root.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(root / 'source_digests.sqlite3', timeout=2,
                                       check_same_thread=False, isolation_level=None)
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('CREATE TABLE IF NOT EXISTS digests '
                             '(path TEXT PRIMARY KEY, signature TEXT NOT NULL, sha256 TEXT NOT NULL)')
        except (OSError, sqlite3.Error):
            if self._db is not None:
                self._db.close()
            self._db = None

    @staticmethod
    def _signature(path):
        stat = path.stat()
        probe = hashlib.sha256()
        with path.open('rb') as stream:
            for offset in sorted({0, max(0, stat.st_size // 2 - 2048), max(0, stat.st_size - 4096)}):
                stream.seek(offset)
                probe.update(stream.read(4096))
        return json.dumps([stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                           stat.st_ino, probe.hexdigest()], separators=(',', ':'))

    def digest(self, path):
        source = Path(path).resolve()
        name = str(source)
        with self._lock:
            file_lock = self._files.setdefault(name, threading.Lock())
        with file_lock:
            for _attempt in range(2):
                signature = self._signature(source)
                with self._lock:
                    row = self._memory.get(name)
                    if row is None and self._db is not None:
                        try:
                            row = self._db.execute('SELECT signature, sha256 FROM digests WHERE path=?',
                                                   (name,)).fetchone()
                        except sqlite3.Error:
                            pass
                    if row is not None and row[0] == signature:
                        self._memory[name] = row
                        self._counts['hits'] += 1
                        return row[1]
                checksum = hashlib.sha256()
                count = 0
                with source.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(block)
                        count += len(block)
                if self._signature(source) != signature:
                    continue
                value = checksum.hexdigest()
                with self._lock:
                    self._memory[name] = (signature, value)
                    self._counts['full_reads'] += 1
                    self._counts['bytes_hashed'] += count
                    if self._db is not None:
                        try:
                            self._db.execute('INSERT OR REPLACE INTO digests VALUES (?, ?, ?)',
                                             (name, signature, value))
                        except sqlite3.Error:
                            pass
                return value
        raise ValueError(f'Source changed while fingerprinting: {source}')

    def stats(self):
        with self._lock:
            return dict(self._counts)


@lru_cache(maxsize=16)
def source_digests(cache_dir):
    return SourceDigests(cache_dir)
