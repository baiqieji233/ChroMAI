"""Process-lifetime exclusion for a Composer training output directory."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path


class TrainingOutputBusyError(RuntimeError):
    """Another process owns the requested training output directory."""


def canonical_output_directory(directory) -> Path:
    """Resolve aliases and normalize case on Windows before creating the lock."""
    return Path(os.path.normcase(str(Path(directory).expanduser().resolve())))


@contextmanager
def training_output_lock(directory):
    """Fail immediately if another trainer owns this canonical output directory.

    Keep the context around preparation, training and every checkpoint save.
    The OS releases the lock when the descriptor closes or its process exits.
    The file deliberately survives release: unlinking it could let contenders
    lock different inodes for the same output directory.
    """
    destination = canonical_output_directory(directory)
    destination.mkdir(parents=True, exist_ok=True)
    stream = (destination / ".composer-training.lock").open("a+b")
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                # Windows permits locking a byte beyond EOF, so initialization
                # never writes through another process's already-held lock.
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise TrainingOutputBusyError(
                    f"Composer training output is already in use: {destination}") from error
            raise
        yield destination
    finally:
        try:
            if acquired:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
