"""Terminate one WebUI task together with its data-loader descendants."""
from __future__ import annotations

import os
import signal
import subprocess


def terminate_task_process(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        result = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                capture_output=True, timeout=10,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode == 0 or process.poll() is not None:
            return
        raise RuntimeError(f'Could not terminate task process tree (PID {process.pid})')
    # New task subprocesses have a separate session. Older running tasks may
    # share the GUI's group, which must never be killed as a side effect.
    if os.getpgid(process.pid) == process.pid:
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
