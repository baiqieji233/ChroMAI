import os
from pathlib import Path
import subprocess
import sys

import pytest

from maichart.composer.training_lock import (
    canonical_output_directory, training_output_lock, training_request_lock,
)


CHILD = """
import sys
from maichart.composer.training_lock import training_output_lock, TrainingOutputBusyError
try:
    with training_output_lock(sys.argv[1]):
        print('acquired', flush=True)
        if sys.argv[2] == 'hold':
            sys.stdin.read(1)
except TrainingOutputBusyError as error:
    print(str(error), flush=True)
    sys.exit(17)
"""


def child_environment():
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    return env


def contender(directory):
    return subprocess.run([sys.executable, "-c", CHILD, str(directory), "once"],
        capture_output=True, text=True, env=child_environment(), timeout=15)


@pytest.mark.parametrize("termination", ["exit", "kill"])
def test_real_process_contention_and_release(tmp_path, termination):
    directory = tmp_path / "output"
    owner = subprocess.Popen([sys.executable, "-c", CHILD, str(directory), "hold"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=child_environment())
    try:
        assert owner.stdout.readline().strip() == "acquired"
        # An equivalent path must contend with the same owner.
        alias = directory / "nested" / ".."
        if os.name == "nt":
            alias = Path(str(alias).swapcase())
        denied = contender(alias)
        assert denied.returncode == 17, denied.stderr
        assert "already in use" in denied.stdout
        independent = contender(tmp_path / "other")
        assert independent.returncode == 0, independent.stderr
        if termination == "kill":
            owner.kill()
            owner.communicate(timeout=15)
        else:
            owner.communicate("x", timeout=15)
            assert owner.returncode == 0
        acquired = contender(directory)
        assert acquired.returncode == 0, acquired.stderr
        assert acquired.stdout.strip() == "acquired"
        assert (directory / ".composer-training.lock").exists()
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.communicate(timeout=15)


def test_exception_releases_lock_and_yields_canonical_path(tmp_path):
    directory = tmp_path / "output"
    with pytest.raises(ValueError, match="preparation failed"):
        with training_output_lock(directory / ".." / "output") as destination:
            assert destination == canonical_output_directory(directory)
            raise ValueError("preparation failed")
    result = contender(directory)
    assert result.returncode == 0, result.stderr


def test_request_reservation_excludes_another_webui_process(tmp_path):
    child = CHILD.replace('training_output_lock', 'training_request_lock')
    directory = tmp_path / 'output'
    with training_request_lock(directory):
        denied = subprocess.run([sys.executable, '-c', child, str(directory), 'once'],
            capture_output=True, text=True, env=child_environment(), timeout=15)
        assert denied.returncode == 17, denied.stderr
        assert 'already in use' in denied.stdout
    acquired = subprocess.run([sys.executable, '-c', child, str(directory), 'once'],
        capture_output=True, text=True, env=child_environment(), timeout=15)
    assert acquired.returncode == 0, acquired.stderr


def test_cli_owns_directory_through_the_training_body(tmp_path,monkeypatch):
    from maichart.composer import train
    directory=tmp_path/'cli'
    def body(args,flags):
        denied=contender(args.checkpoint_dir)
        assert denied.returncode==17,denied.stderr
        assert '--checkpoint-dir' in flags
        return 42
    monkeypatch.setattr(train,'_train',body)
    assert train.main(['--manifest','unused','--checkpoint-dir',str(directory),'--device','cpu'])==42
    assert contender(directory).returncode==0
