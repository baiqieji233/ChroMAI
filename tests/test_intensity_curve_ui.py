from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "maichart" / "webui" / "static"


def test_intensity_workspace_contract_is_present():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'data-mode="intensity"' in html
    for element_id in (
        "intensityCurveEnabled", "openIntensityCurveButton", "generateIntensityCurveButton",
        "intensityCurveCanvas", "intensityAudio", "intensitySaveButton",
        "applyIntensitySectionButton",
        "intensityZoom", "intensityPan", "intensityFitSelection", "intensityEditMode",
        "intensitySelectionSet", "intensitySelectionUp", "intensitySelectionDown",
        "markIntensityRestButton", "clearIntensityRestButton", "intensityRestCount",
    ):
        assert f'id="{element_id}"' in html


def test_intensity_ui_uses_separate_analysis_poll_and_curve_payload():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '"/api/intensity/analyze"' in js
    assert "pollIntensityTask" in js
    assert "intensity_curve_enabled" in js
    assert 'schema: "maichart-intensity-curve-v1"' in js
    assert 'audio_sha256' in js
    assert 'intensity_mode' in js
    assert 'intensity_curve_mode: valueOrDefault("#intensityCurveMode", "follow")' in js
    assert "rest_regions" in js
    assert "normalizeRestRegions" in js
    assert "invalidateLearnedCurveForCondition" in js
    assert "等级、Radar 或 Style 已变化" in js
    assert "normalizeGenerationLevel(previousLevel)" in js
    assert "contextStale" in js
    assert "请重新生成模型来源曲线后再保存" in js
    assert "density_feedback_capability" in js
    assert "follow / hard 密度反馈能力" in js
    # The large spectrogram is for the editor only and is deliberately omitted
    # from the generation request.
    assert "spectrogram: undefined" in js


def test_playability_ui_matches_backend_retry_and_timeout_contract():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="runtimeReferenceExecutable"' not in html
    assert 'id="runtimeReferenceTimeoutSec"' not in html
    assert "内嵌 MiaCode" in html
    assert "确认问题减少才保留修改" in html
    assert 'href="https://github.com/Team-MiaCode/MiaCode"' in html
    assert "感谢" in html and "团队与贡献者" in html
    assert "软参考 ${softCount}" in js
    assert "窗口重试 ${retryCount}" in js


def test_app_javascript_parses():
    result = subprocess.run(
        ["node", "--check", str(STATIC / "app.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
