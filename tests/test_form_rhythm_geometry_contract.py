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
    # Migrating to a profile the source already declares is not a migration.
    with pytest.raises(ValueError, match='different rhythm geometry profile'):
        load_finetune_model(source, model.config)


def test_finetune_migration_rejects_narrowing_a_geometry_contract(tmp_path):
    """A checkpoint that carries geometry cannot be finetuned back to none."""
    model, planner, codec = _models('exact_fraction_v1')
    source = tmp_path / 'v1.pt'
    torch.save(payload(model, planner, codec), source)
    with pytest.raises(ValueError, match='targets exact_fraction_v1 or exact_fraction_v2'):
        load_finetune_model(source, replace(model.config, rhythm_geometry_profile='none'))


def test_v1_to_v2_finetune_carries_matching_geometry_channels(tmp_path):
    """A v1 checkpoint keeps its learned sub-beat channels when moving to v2."""
    from maichart.form.rhythm_geometry import GEOMETRY_DIM, GEOMETRY_DIM_V2
    from maichart.form.checkpoint import V1_TO_V2_CHANNELS

    model, planner, codec = _models('exact_fraction_v1')
    with torch.no_grad():
        # Distinct values per channel make a wrong channel mapping visible.
        model.decoder_geometry_projection.weight.copy_(
            torch.arange(1, 1 + model.decoder_geometry_projection.weight.numel(),
                         dtype=torch.float32).reshape(model.decoder_geometry_projection.weight.shape))
    source = tmp_path / 'v1.pt'
    torch.save(payload(model, planner, codec), source)
    target = load_finetune_model(source, replace(model.config, rhythm_geometry_profile='exact_fraction_v2'))
    source_weight = model.decoder_geometry_projection.weight
    target_weight = target.decoder_geometry_projection.weight
    assert target_weight.shape == (16, GEOMETRY_DIM_V2)
    for origin, destination in V1_TO_V2_CHANNELS:
        torch.testing.assert_close(target_weight[:, destination], source_weight[:, origin])
    carried = {destination for _origin, destination in V1_TO_V2_CHANNELS}
    for channel in range(GEOMETRY_DIM_V2):
        if channel not in carried:
            torch.testing.assert_close(target_weight[:, channel], torch.zeros(16))
    # The transferable decoder body is preserved exactly.
    torch.testing.assert_close(target.output_head.weight, model.output_head.weight)


def test_profile_resolution_fresh_resume_and_finetune():
    fresh = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o'])
    profile, weight = resolve_rhythm_geometry_settings(fresh)
    assert (profile, weight) == ('exact_fraction_v2', .1)
    config, _ = __import__('maichart.form.train', fromlist=['fresh_model_configs']).fresh_model_configs(fresh)
    assert config.rhythm_geometry_profile == 'exact_fraction_v2'
    # The bar-phase contract is wider than the span-relative one it replaces.
    assert config.geometry_dim == 13

    old_model, _planner, _codec = _models('exact_fraction_v1')
    resume = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--resume', 'old.pt'])
    profile, weight = resolve_rhythm_geometry_settings(resume, {'args': {}}, old_model)
    assert (profile, weight) == ('exact_fraction_v1', 0.)

    incompatible = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--resume', 'old.pt',
                                        '--rhythm-geometry-profile', 'exact_fraction_v2'])
    with pytest.raises(ValueError, match='resume must inherit'):
        resolve_rhythm_geometry_settings(incompatible, {'args': {}}, old_model)

    tuned = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--init-checkpoint', 'old.pt',
                                 '--rhythm-geometry-profile', 'exact_fraction_v2'])
    profile, weight = resolve_rhythm_geometry_settings(tuned, {'args': {}}, old_model)
    assert (profile, weight) == ('exact_fraction_v2', .1)

    downgrade = parser().parse_args(['--manifest', 'm', '--checkpoint-dir', 'o', '--init-checkpoint', 'old.pt',
                                     '--rhythm-geometry-profile', 'none'])
    with pytest.raises(ValueError, match='cannot remove'):
        resolve_rhythm_geometry_settings(downgrade, {'args': {}}, old_model)
