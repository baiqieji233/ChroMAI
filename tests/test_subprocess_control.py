from types import SimpleNamespace
import os
import subprocess
import sys

import pytest

from maichart.webui import subprocess_control


def test_windows_cancel_terminates_descendants_too(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess_control.os, 'name', 'nt')
    monkeypatch.setattr(subprocess_control.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    monkeypatch.setattr(subprocess_control.subprocess, 'run',
                        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0))
    process = SimpleNamespace(pid=12345, poll=lambda:None)
    subprocess_control.terminate_task_process(process)
    assert calls == [['taskkill','/PID','12345','/T','/F']]


def test_cancel_does_not_target_an_already_finished_pid(monkeypatch):
    monkeypatch.setattr(subprocess_control.subprocess, 'run',
                        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError('Already finished')))
    subprocess_control.terminate_task_process(SimpleNamespace(pid=12345,poll=lambda:0))


@pytest.mark.skipif(os.name != 'nt', reason='Windows process tree integration')
def test_windows_cancel_stops_a_real_child_process():
    import ctypes
    from ctypes import wintypes
    source = ('import subprocess,sys,time; '
              'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
              'print(child.pid,flush=True); time.sleep(30)')
    process = subprocess.Popen([sys.executable,'-c',source],stdout=subprocess.PIPE,text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = None
    try:
        child_pid = int(process.stdout.readline())
        handle = kernel.OpenProcess(0x00100000,False,child_pid)
        assert handle
        subprocess_control.terminate_task_process(process)
        process.wait(timeout=5)
        assert kernel.WaitForSingleObject(handle,5000) == 0
    finally:
        if process.poll() is None:
            subprocess_control.terminate_task_process(process)
            process.wait(timeout=5)
        if handle:
            kernel.CloseHandle(handle)
        process.stdout.close()
