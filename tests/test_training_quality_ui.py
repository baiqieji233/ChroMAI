"""Static contract checks for the quality-weighted training workspace."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "src/maichart/webui/static/index.html").read_text(encoding="utf-8")
APP = (ROOT / "src/maichart/webui/static/app.js").read_text(encoding="utf-8")
CSS = (ROOT / "src/maichart/webui/static/styles.css").read_text(encoding="utf-8")


def test_quality_workspace_is_visible_and_uses_current_manifest():
    assert 'id="qualityModeTab"' in INDEX
    assert 'data-mode="quality"' in INDEX
    assert 'class="panel quality-panel"' in INDEX
    assert 'id="openQualityWorkspaceButton"' in INDEX
    assert 'qualityManifest()' in APP
    assert '"/api/quality/catalog"' in APP
    assert '"/api/quality/save"' in APP


def test_training_payload_contains_saved_quality_contract():
    for field in (
        "quality_enabled",
        "quality_annotations",
        "preference_enabled",
        "preference_weight",
        "preference_max_pairs",
    ):
        assert field in APP
    assert "annotationPath" in APP
    assert "quality_training_capability" in APP


def test_quality_ui_explains_saved_annotations_and_negative_limit():
    assert "只对已保存的素材标注生效" in INDEX
    assert "不会宣称模型已经学会负向偏好" in INDEX
    assert "验证集不参与加权" in INDEX
    assert "finetune" in INDEX and "resume" in INDEX
    assert "反例" in INDEX
    assert "corrected_path" in APP
    assert "corrected_difficulty" in APP


def test_quality_workspace_has_paging_search_and_region_controls():
    for field in (
        "qualitySearch",
        "qualityLabelFilter",
        "qualityPrevButton",
        "qualityNextButton",
        "qualitySelectAll",
        "qualityAddRegionButton",
        "qualitySaveRegionsButton",
        "qualityCorrectedFile",
        "plannerProfile",
        "planTeacherMix",
        "intensityCurveSource",
        "intensityCurveStrength",
        "intensityCurveMode",
    ):
        assert f'id="{field}"' in INDEX
    assert "start_sec" in APP and "end_sec" in APP
    assert "max=\"4\"" in INDEX
    assert 'body[data-workspace="quality"]' in CSS
    for mode in ("follow", "hard", "soft", "strict"):
        assert f'value="{mode}"' in INDEX
    assert "柔性跟随" in INDEX and "严苛跟随" in INDEX
    assert "仅规划" in INDEX and "生成后上限" in INDEX
    assert "music_structure_v2" in INDEX and "fresh" in INDEX
    assert '"music_structure_v2"' in APP


def test_training_materials_support_multiple_sources_and_catalog_navigation():
    for field in (
        "datasetCards",
        "datasetComposeButton",
        "datasetWorkspaceRefreshButton",
        "registerDatasetButton",
        "datasetBackButton",
        "datasetCatalogPanel",
        "qualitySortBy",
        "qualitySortDirection",
        "qualityPageInput",
        "qualityPageJumpButton",
    ):
        assert f'id="{field}"' in INDEX
    assert "selectedTrainingManifests" in APP
    assert "manifests:" in APP
    assert "/api/datasets/catalog" in APP
    assert "/api/datasets/compose" in APP
    assert "sort_by" in APP and "sort_order" in APP
    assert "maidata_title" in APP and "folder_name" in APP
    assert "dataset-workspace" in CSS and "dataset-card" in CSS
    assert "assertTrainingSelection" in APP
    assert "manifests: selectedTrainingManifests()" in APP


def test_browsing_dataset_does_not_require_replacing_training_manifest():
    start = APP.index("async function openDataset")
    end = APP.index("function selectDataset", start)
    body = APP[start:end]
    assert "selectDataset(item)" not in body
    assert "activePath" in body and "loadQualityCatalog" in body

