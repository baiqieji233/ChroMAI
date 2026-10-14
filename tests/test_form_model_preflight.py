from pathlib import Path

import pytest

from maichart.form.train import _validate, parser
from maichart.webui import form_api


def test_invalid_derived_planner_width_is_rejected_before_data_preparation():
    args = parser().parse_args(['--model-family','song_form_v1','--manifest', 'unopened.json', '--checkpoint-dir', 'unused',
                                '--d-model', '180', '--nhead', '6'])
    with pytest.raises(ValueError, match='width must be divisible by form_heads'):
        _validate(args)


@pytest.mark.parametrize(('width', 'heads'), [(512, 8), (65, 1), (192, 6), (16, 1)])
def test_preflight_accepts_existing_valid_model_and_planner_dimensions(width, heads):
    args = parser().parse_args(['--manifest', 'unopened.json', '--checkpoint-dir', 'unused',
                                '--d-model', str(width), '--nhead', str(heads)])
    _validate(args)


def test_api_rejects_invalid_fresh_planner_width_before_registering_task(tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind): raise AssertionError('task should not be created')
    with pytest.raises(ValueError, match='width must be divisible by form_heads'):
        form_api.start_form_task(object(), 'train', {'manifest': str(manifest),
            'checkpoint_dir': str(tmp_path/'out'), 'd_model': 180, 'nhead': 6,
            'mode': 'fresh', 'device': 'cpu'}, runtime=Runtime())
