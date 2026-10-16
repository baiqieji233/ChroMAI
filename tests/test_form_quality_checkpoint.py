from dataclasses import replace

import pytest
import torch

from maichart.form import checkpoint
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.rational_planner import RationalPlanner, RationalPlannerConfig
from maichart.generation.model_compatibility import checkpoint_capabilities


def models(*, quality="legacy", geometry="none", curve_bin=1., context=1.):
    codec = RationalCodec()
    enhanced = quality != "legacy"
    model = FormTransformer(FormConfig(vocab_size=len(codec), d_model=16, nhead=1,
        layers=1, dim_feedforward=32, dropout=0., max_tokens=32, max_memory_tokens=16,
        descriptor_dim=512 if enhanced else 504, rhythm_geometry_profile=geometry,
        quality_profile=quality, curve_bin_seconds=curve_bin if enhanced else 2.))
    planner = RationalPlanner(RationalPlannerConfig(width=16, form_heads=1, form_layers=1,
        dropout=0., descriptor_dim=512 if enhanced else 504, quality_profile=quality,
        context_seconds=context if enhanced else 0.))
    return model, planner, codec


def save(tmp_path, model, planner, codec, name="source.pt", **metadata):
    path = tmp_path / name
    torch.save(checkpoint.payload(model, planner, codec, **metadata), path)
    return path


def test_absent_quality_metadata_keeps_the_original_checkpoint_contract(tmp_path):
    model, planner, codec = models()
    raw = checkpoint.payload(model, planner, codec)
    for key in ("quality_profile", "curve_bin_seconds", "position_alignment_tolerance_ms"):
        raw["model_config"].pop(key)
    for key in ("quality_profile", "context_seconds"):
        raw["blueprint_config"].pop(key)
    path = tmp_path / "historical.pt"; torch.save(raw, path)
    loaded, planned, _, stored = checkpoint.load(path)
    assert stored["input_contract"] == checkpoint.RATIONAL_INPUT_CONTRACT
    assert loaded.config.quality_profile == planned.config.quality_profile == "legacy"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)


def test_enhanced_roundtrip_versions_every_input_semantic_and_advertises_real_supervision(tmp_path):
    model, planner, codec = models(quality="musical_evidence_v1", geometry="exact_fraction_v2", curve_bin=1.25, context=1.5)
    path = save(tmp_path, model, planner, codec, capabilities={"position_alignment": {
        "contract": "authored_audio_queries_v1", "steps": 3}})
    loaded, planned, _, stored = checkpoint.load(path)
    contract = stored["input_contract"]
    assert contract["version"] == "song_form_inputs_v5_musical_evidence"
    assert contract["descriptor_dim"] == 512
    assert contract["audio_feature_policy"] == "native_measured_v2"
    assert contract["audio_evidence_profile"] == "balanced_evidence_v1"
    assert contract["load_curve_density"] == "musical_log_v1"
    assert contract["duration_semantics"] == "miacode_head_bpm_v1"
    assert contract["curve_bin_seconds"] == 1.25
    assert contract["planner_context_seconds"] == 1.5
    assert loaded.config.quality_profile == planned.config.quality_profile == "musical_evidence_v1"
    capabilities = checkpoint_capabilities(stored)
    assert capabilities["valid"]
    assert capabilities["position_alignment_supervision"]
    assert capabilities["position_alignment_trained"]
    assert capabilities["curve_bin_seconds"] == 1.25
    assert capabilities["planner_context_seconds"] == 1.5
    stored["capabilities"] = {}
    assert not checkpoint_capabilities(stored)["position_alignment_trained"]


@pytest.mark.parametrize(("owner", "field", "value"), [
    ("blueprint_config", "quality_profile", "legacy"),
    ("blueprint_config", "descriptor_dim", 504),
    ("model_config", "curve_bin_seconds", 1.5),
    ("model_config", "position_alignment_tolerance_ms", 25.),
    ("blueprint_config", "context_seconds", 2.),
])
def test_metadata_cannot_silently_change_an_enhanced_input_contract(tmp_path, owner, field, value):
    model, planner, codec = models(quality="musical_evidence_v1")
    raw = checkpoint.payload(model, planner, codec)
    raw[owner][field] = value
    path = tmp_path / "mismatch.pt"; torch.save(raw, path)
    with pytest.raises(ValueError, match="contract|profile|descriptor"):
        checkpoint.load(path)
    assert not checkpoint_capabilities(raw)["valid"]


@pytest.mark.parametrize(("owner", "field"), [("model_config", "curve_bin_seconds"),
                                             ("blueprint_config", "context_seconds")])
def test_enhanced_semantic_metadata_cannot_fall_back_to_legacy_constructor_defaults(tmp_path, owner, field):
    model, planner, codec = models(quality="musical_evidence_v1")
    raw = checkpoint.payload(model, planner, codec)
    raw[owner].pop(field)
    path = tmp_path / "missing-metadata.pt"; torch.save(raw, path)
    with pytest.raises(ValueError, match="lacks metadata"):
        checkpoint.load(path)
    assert not checkpoint_capabilities(raw)["valid"]


def test_webui_inspection_exposes_saved_evidence_capabilities_without_cuda(tmp_path):
    from maichart.webui.form_api import inspect_form_checkpoint
    model, planner, codec = models(quality="musical_evidence_v1", context=1.5)
    model.config.position_alignment_tolerance_ms = 25.
    path = save(tmp_path, model, planner, codec,
                capabilities={"position_alignment": {"contract": "authored_audio_queries_v1", "steps": 2}})
    checked = inspect_form_checkpoint(path)
    assert checked["valid"]
    for values in (checked, checked["generation_capabilities"]):
        assert values["quality_profile"] == "musical_evidence_v1"
        assert values["curve_bin_seconds"] == 1.
        assert values["planner_context_seconds"] == 1.5
        assert values["position_alignment_supervision"]
        assert values["position_alignment_trained"]
    assert checked["resume_settings"]["position_alignment_tolerance_ms"] == 25.


def test_quality_upgrade_preserves_descriptor_coordinates_branches_and_shared_weights(tmp_path):
    model, planner, codec = models()
    with torch.no_grad():
        model.form_projection.weight.copy_(torch.arange(model.form_projection.weight.numel()).reshape_as(model.form_projection.weight))
        planner.form_projection.weight.copy_(torch.arange(planner.form_projection.weight.numel()).reshape_as(planner.form_projection.weight))
    path = save(tmp_path, model, planner, codec)
    target = checkpoint.load_finetune_model(path, replace(model.config, quality_profile="musical_evidence_v1",
        descriptor_dim=512, curve_bin_seconds=1.))
    planned = checkpoint.load_finetune_planner(path, replace(planner.config,
        quality_profile="musical_evidence_v1", descriptor_dim=512, context_seconds=1.))
    for previous, updated in ((model, target), (planner, planned)):
        torch.testing.assert_close(updated.form_projection.weight[:, :504], previous.form_projection.weight[:, :504], rtol=0, atol=0)
        torch.testing.assert_close(updated.form_projection.weight[:, -3:], previous.form_projection.weight[:, -3:], rtol=0, atol=0)
        assert not updated.form_projection.weight[:, 504:512].count_nonzero()
    torch.testing.assert_close(target.branch_embedding.weight[:11], model.branch_embedding.weight, rtol=0, atol=0)
    torch.testing.assert_close(target.output_head.weight, model.output_head.weight, rtol=0, atol=0)
    assert target.branch_embedding.weight.shape == (13, 16)
    assert not planned.temporal_projection.weight.count_nonzero()
    roundtrip = save(tmp_path, target, planned, codec, name="upgraded.pt")
    checkpoint.load(roundtrip)


def test_new_descriptor_and_zero_temporal_projection_keep_transferred_planner_function(tmp_path):
    model, planner, codec = models()
    path = save(tmp_path, model, planner, codec)
    target = checkpoint.load_finetune_planner(path, replace(planner.config,
        quality_profile="musical_evidence_v1", descriptor_dim=512, context_seconds=1.))
    planner.eval(); target.eval()
    audio = torch.randn(1, 5, 120)
    form = torch.randn(1, 2, 507)
    expanded = torch.cat((form[..., :504], torch.randn(1, 2, 8), form[..., -3:]), dim=-1)
    condition = torch.randn(1, 16)
    with torch.inference_mode():
        old = planner(audio, form, condition)["load_curve"]
        new = target(audio, expanded, condition, audio_times_sec=torch.arange(5.)[None])["load_curve"]
    torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)


def test_geometry_and_quality_can_be_upgraded_in_one_explicit_finetune(tmp_path):
    model, planner, codec = models(geometry="exact_fraction_v1")
    with torch.no_grad():
        model.decoder_geometry_projection.weight.fill_(.2)
    path = save(tmp_path, model, planner, codec)
    target = checkpoint.load_finetune_model(path, replace(model.config, quality_profile="musical_evidence_v1",
        descriptor_dim=512, curve_bin_seconds=1., rhythm_geometry_profile="exact_fraction_v2"))
    assert target.decoder_geometry_projection.weight.shape == (16, 13)
    carried = {destination for _, destination in checkpoint.V1_TO_V2_CHANNELS}
    for channel in range(13):
        expected = .2 if channel in carried else 0.
        torch.testing.assert_close(target.decoder_geometry_projection.weight[:, channel], torch.full((16,), expected))


def test_same_quality_finetune_may_change_bin_tolerance_and_context_explicitly(tmp_path):
    model, planner, codec = models(quality="musical_evidence_v1")
    path = save(tmp_path, model, planner, codec)
    target = checkpoint.load_finetune_model(path, replace(model.config, curve_bin_seconds=.5,
                                                         position_alignment_tolerance_ms=20.))
    planned = checkpoint.load_finetune_planner(path, replace(planner.config, context_seconds=2.))
    stored = checkpoint.payload(target, planned, codec)
    assert stored["input_contract"]["curve_bin_seconds"] == .5
    assert stored["input_contract"]["position_alignment_tolerance_ms"] == 20.
    assert stored["input_contract"]["planner_context_seconds"] == 2.
    torch.testing.assert_close(target.output_head.weight, model.output_head.weight, rtol=0, atol=0)


def test_quality_downgrades_and_missing_new_weights_are_rejected(tmp_path):
    model, planner, codec = models(quality="musical_evidence_v1")
    path = save(tmp_path, model, planner, codec)
    with pytest.raises(ValueError, match="downgrade"):
        checkpoint.load_finetune_model(path, replace(model.config, quality_profile="legacy", descriptor_dim=504))
    with pytest.raises(ValueError, match="downgrade"):
        checkpoint.load_finetune_planner(path, replace(planner.config, quality_profile="legacy", descriptor_dim=504))
    raw = checkpoint.payload(model, planner, codec)
    raw["model_state_dict"].pop("alignment_query.weight")
    path = tmp_path / "missing.pt"; torch.save(raw, path)
    with pytest.raises(ValueError, match="parameter contract mismatch"):
        checkpoint.load(path)


def test_enhanced_readiness_does_not_inherit_layout_capability_from_historical_counts():
    legacy = {"optimizer_steps": 100, "blueprint_steps": 20, "planner_effective_steps": 1,
              "task_training": {"draft": 1000, "revision": 900},
              "input_contract": checkpoint.RATIONAL_INPUT_CONTRACT}
    assert checkpoint.task_is_ready(legacy)
    upgraded = {**legacy, "model_config": {"quality_profile": "musical_evidence_v1"},
                "input_contract": checkpoint._geometry_contract("none", "musical_evidence_v1")}
    assert not checkpoint.task_is_ready(upgraded)
    for evidence in ({"steps": 3}, {"steps": 0, "contract": "form_condition_v2"},
                     {"steps": 3, "contract": "song_form_inputs_v4"}):
        assert not checkpoint.task_is_ready({**upgraded, "capabilities": {"draft": evidence}})
    trained = {**upgraded, "capabilities": {"draft": {"steps": 1, "contract": "form_condition_v2"}}}
    assert checkpoint.task_is_ready(trained)
    assert not checkpoint.task_is_ready(trained, "revision")


def test_blueprint_only_migrated_checkpoint_cannot_generate_until_layout_is_trained(tmp_path):
    model, planner, codec = models(quality="musical_evidence_v1")
    path = save(tmp_path, model, planner, codec, optimizer_steps=10, blueprint_steps=10,
                planner_effective_steps=1, task_training={"draft": 500, "revision": 400},
                capabilities={"load_curve": {"steps": 1, "contract": "load_curve_v1"}})
    checkpoint.load(path)
    with pytest.raises(ValueError, match="draft and whole-song blueprint"):
        checkpoint.load(path, inference=True)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["capabilities"]["draft"] = {"steps": 1, "contract": "form_condition_v2"}
    torch.save(raw, path)
    checkpoint.load(path, inference=True)
