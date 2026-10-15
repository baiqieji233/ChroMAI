import pytest
from maichart.form.output_guard import training_output_guard


def test_same_output_cannot_have_two_writers_and_exit_releases_lock(tmp_path):
    with training_output_guard(tmp_path/'model'):
        with pytest.raises(ValueError,match='已有训练任务'):
            with training_output_guard(tmp_path/'model'):
                pytest.fail('Concurrent checkpoint writer acquired the same directory')
        with training_output_guard(tmp_path/'other'):
            pass
    with training_output_guard(tmp_path/'model'):
        pass


def test_exception_releases_training_output_lock(tmp_path):
    with pytest.raises(RuntimeError):
        with training_output_guard(tmp_path):
            raise RuntimeError('training failed')
    with training_output_guard(tmp_path):
        pass
