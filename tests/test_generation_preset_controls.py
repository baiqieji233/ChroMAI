"""Check numeric requests and curve editing against the actual browser functions."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "src/maichart/webui/static/app.js").read_text(encoding="utf-8")
HTML = (ROOT / "src/maichart/webui/static/index.html").read_text(encoding="utf-8")


def _function(name):
    start = APP.index(f"function {name}(")
    end = APP.index("\n}", start) + 2
    return APP[start:end]


def _run_js(script):
    result = subprocess.run(["node"], input=script, capture_output=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_continuous_levels_and_preferences_are_composable():
    profile = json.loads((ROOT / "src/maichart/webui/static/generation_presets.json").read_text(encoding="utf-8"))
    definitions = APP[APP.index("const preferenceDefinitions ="):APP.index("const datasetStorageKey")]
    result = _run_js(definitions + "\n" + _function("interpolateLevelPreset")
                     + "\n" + _function("composeLevelCondition")
                     + "\nconst profile = " + json.dumps(profile) + ";" + r'''
const rows = [], combinations = [];
for (let i = 0; i < 31; i++) {
  const level = Number((12 + i / 10).toFixed(1));
  rows.push(composeLevelCondition(profile, level, [], 0));
  for (const preferences of [["keyboard"], ["stars"], ["technical"], ["stars", "technical"],
                             ["keyboard", "stars"], Object.keys(preferenceDefinitions)]) {
    const composed = composeLevelCondition(profile, level, preferences, .8);
    combinations.push({ level, preferences, condition: composed });
  }
}
const forward = composeLevelCondition(profile, 13.4, ["keyboard", "stars", "technical"], .7);
const reverse = composeLevelCondition(profile, 13.4, ["technical", "stars", "keyboard"], .7);
const noPreference = composeLevelCondition(profile, 13.4);
const zeroStrength = composeLevelCondition(profile, 13.4, ["stars", "touch"], 0);
const repeat = composeLevelCondition(profile, 13.4, ["keyboard", "stars", "technical", "stars"], .7);
const intermediate = composeLevelCondition(profile, 13.45);
let rejected = 0;
for (const level of [11.9, 15.1, NaN, Infinity]) {
  try { composeLevelCondition(profile, level); } catch (_) { rejected++; }
}
console.log(JSON.stringify({ rows, combinations, forward, reverse, repeat, noPreference, zeroStrength, intermediate, rejected }));
''')
    levels = [round(12 + index / 10, 1) for index in range(31)]
    assert [row["level"] for row in result["rows"]] == levels
    for key in ("note_density", "peak"):
        values = [row["radar"][key] for row in result["rows"]]
        assert values == sorted(values)
    for row in result["rows"] + [item["condition"] for item in result["combinations"]]:
        assert row["radar"]["chord"] == row["style"]["chord_bias"]
        assert row["style"]["melody_following"] == .9
        assert row["radar"]["slide_tricky"] <= row["style"]["slide_bias"]
        assert row["radar"]["slide_sequence"] <= row["style"]["slide_bias"]
        assert all(0 <= value <= 1 for section in ("radar", "style") for value in row[section].values())
    for item in result["combinations"]:
        base = result["rows"][round((item["level"] - 12) * 10)]
        for key in ("note_density", "peak"):
            assert item["condition"]["radar"][key] == base["radar"][key]
    assert result["forward"] == result["reverse"] == result["repeat"]
    assert result["zeroStrength"] == result["noPreference"]
    assert result["intermediate"]["level"] == 13.45
    assert result["intermediate"]["interpolated"] is True
    assert result["rejected"] == 4


def test_contrast_enhancement_preserves_timing_and_supports_undo_and_persistence():
    result = _run_js("\n".join(_function(name) for name in (
        "intensityProfileValid", "cloneIntensityProfile", "pushIntensityHistory",
        "enhanceIntensityContrast", "undoIntensityCurve", "saveIntensityCurveState",
    )) + r'''
const intensityCurveStorageKey = "curve";
let stored = null;
const localStorage = { setItem: (_, value) => { stored = JSON.parse(value); } };
const original = {schema: "maichart-intensity-curve-v1", duration_sec: 12, audio_sha256: "a".repeat(64),
  points: [0, .1, .3, .5, .7, .9, 1].map((intensity, index) => ({time_sec: index * 2, intensity}))};
let intensityCurveState = {profile: cloneIntensityProfile(original), automatic: cloneIntensityProfile(original), history: [], saved: true};
const intensityView = {selected: new Set([2])};
function drawIntensityCurve() {}
function updateIntensityButtons() {}
function setIntensityStatus() {}
enhanceIntensityContrast();
const enhanced = cloneIntensityProfile(intensityCurveState.profile);
const savedAfterEdit = stored.saved;
enhanceIntensityContrast();
const repeated = cloneIntensityProfile(intensityCurveState.profile);
undoIntensityCurve();
const firstUndo = cloneIntensityProfile(intensityCurveState.profile);
undoIntensityCurve();
const secondUndo = cloneIntensityProfile(intensityCurveState.profile);
intensityCurveState.profile.points.forEach(point => { point.intensity = .5; });
enhanceIntensityContrast();
console.log(JSON.stringify({enhanced, repeated, firstUndo, secondUndo, savedAfterEdit,
  automatic: intensityCurveState.automatic, persistedAfterUndo: stored.profile,
  noopHistory: intensityCurveState.history.length}));
''')
    values = lambda profile: [point["intensity"] for point in profile["points"]]
    original = [0, .1, .3, .5, .7, .9, 1]
    assert values(result["enhanced"]) == [0, 0, .25, .5, .75, 1, 1]
    assert values(result["repeated"]) == [0, 0, .1875, .5, .8125, 1, 1]
    assert [point["time_sec"] for point in result["enhanced"]["points"]] == list(range(0, 13, 2))
    assert result["enhanced"]["edited"] is True
    assert result["savedAfterEdit"] is False
    assert values(result["firstUndo"]) == values(result["enhanced"])
    assert values(result["secondUndo"]) == original
    assert values(result["automatic"]) == original
    assert values(result["persistedAfterUndo"]) == original
    assert result["noopHistory"] == 0


def test_intensity_help_describes_current_backend_limits_and_reset_fields():
    assert 'id="intensityContrastButton"' in HTML
    assert 'on("#intensityContrastButton", "click", enhanceIntensityContrast)' in APP
    assert "取消此开关也不会关闭模型内部 planner" in HTML
    assert "strict：不使用跟随幅度限制" in HTML
    assert "0.675 / 1 / 1.325" in HTML
    assert "等价于关掉上一项" not in HTML
    fields = APP[APP.index("const GENERATION_PARAM_FIELDS"):APP.index("let activeTaskId")]
    for field in ("intensityCurveStrength", "intensityCurveMode", "intensityCurveSource"):
        assert f'"{field}"' in fields
    assert 'type="range" min="0" max="1" step="0.001"' in APP
    assert "旧 checkpoint 的结构不能通过 resume 或 finetune 切换" in HTML
