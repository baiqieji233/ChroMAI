from dataclasses import replace

import pytest
import torch

from maichart.form.checkpoint import (
    RATIONAL_GEOMETRY_INPUT_CONTRACT,
    RATIONAL_INPUT_CONTRACT,
    load,
    load_finetune_model,
    payload,
)
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.rational_planner import RationalPlanner, RationalPlannerConfig
from maichart.form.train import parser, resolve_rhythm_geometry_settings


def _models(profile='none'):
    codec = RationalCodec()
    model = FormTransformer(FormConfig(vocab_size=len(codec), d_model=16, nhead=1,
        layers=1, dim_feedforward=32, max_tokens=32, max_memory_tokens=16,
        rhythm_geometry_profile=profile))
    planner = RationalPlanner(RationalPlannerConfig(width=16, form_heads=1, form_layers=1))
    return model, planner, codec


def test_rational_geometry_contract_is_checkpointed_separately(tmp_path):
    for profile, expected in (('none', RATIONAL_INPUT_CONTRACT),
                              ('exact_fraction_v1', RATIONAL_GEOMETRY_INPUT_CONTRACT)):
        model, planner, codec = _models(profile)
        path = tmp_path / f'{profile}.pt'
        torch.save(payload(model, planner, codec), path)
        loaded, _planner, _codec, saved = load(path)
        assert loaded.config.rhythm_geometry_profile == profile
        assert saved['input_contract'] == expected


def test_none_to_exact_finetune_only_allows_new_geometry_parameters(tmp_path):
    model, planner, codec = _models('none')
    source = tmp_path / 'none.pt'
    torch.save(payload(model, planner, codec), source)
    target = load_finetune_model(source, replace(model.config, rhythm_geometry_profile='exact_fraction_v1'))
    torch.testing.assert_close(target.output_head.weight, model.output_head.weight)
    assert target.interval_head.weight.shape == (3, 16)
    with pytest.raises(ValueError, match='Only none'):
        load_finetune_model(source, model.config)


def test_profile_resolution_fresh_resume_and_finetune():
    fresh = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o'])
    profile, weight = resolve_rhythm_geometry_settings(fresh)
    assert (profile, weight) == ('exact_fraction_v1', .1)
    config, _ = __import__('maichart.form.train', fromlist=['fresh_model_configs']).fresh_model_configs(fresh)
    assert config.rhythm_geometry_profile == 'exact_fraction_v1'

    old_model, _planner, _codec = _models('none')
    resume = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--resume', 'old.pt'])
    profile, weight = resolve_rhythm_geometry_settings(resume, {'args': {}}, old_model)
    assert (profile, weight) == ('none', 0.)

    incompatible = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--resume', 'old.pt',
                                        '--rhythm-geometry-profile', 'exact_fraction_v1'])
    with pytest.raises(ValueError, match='resume must inherit'):
        resolve_rhythm_geometry_settings(incompatible, {'args': {}}, old_model)

    tuned = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--init-checkpoint', 'old.pt',
                                 '--rhythm-geometry-profile', 'exact_fraction_v1'])
    profile, weight = resolve_rhythm_geometry_settings(tuned, {'args': {}}, old_model)
    assert (profile, weight) == ('exact_fraction_v1', .1)
