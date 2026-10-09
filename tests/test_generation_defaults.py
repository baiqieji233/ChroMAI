"""Generation defaults must come from the corpus, and must not drift apart.

The panel, the CLI and the server each carry a copy of the defaults, and the
corpus profile carries the measurement they are supposed to come from.  These
tests pin all four together: a hardcoded slider value that silently diverges from
the measured median is exactly the bug that made the UI ask for a 0.5-style chart
(8 nps, 50% chords) while the corpus sits at 4.9 nps and 28% chords.
"""

import json
import re
from pathlib import Path

import pytest

from maichart.generation.conditions import (
    CONDITION_KEYS,
    GENERATION_DEFAULTS,
    GENERATION_SAMPLING_DEFAULTS,
    RADAR_KEYS,
    STYLE_KEYS,
)

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "src/maichart/webui/static/app.js"
INDEX_HTML = ROOT / "src/maichart/webui/static/index.html"
PROFILE_CANDIDATES = [
    ROOT / "webui_workspace/datasets/full训练集/condition_profile.json",
    ROOT / "webui_workspace/datasets/seafoodcup2/condition_profile.json",
]

# melody_following is the one control the profile cannot measure: it runs without
# audio and stores 0.000.  Its default comes from the training-side measurement
# instead (24 songs: min 0.803, p50 0.895, max 0.975).
AUDIO_MEASURED = {"melody_following": 0.90}


def _slider_defaults() -> dict[str, str]:
    """`[key, label, help, default]` rows out of app.js, keyed by control name."""

    source = APP_JS.read_text(encoding="utf-8")
    rows = dict(re.findall(r'\["([a-z_]+)",\s*"[^"]*",\s*(?:"[^"]*"|"[^"]*"\s*\+[^,]*),\s*"([\d.]+)"\]', source))
    return rows


def test_every_control_has_a_default():
    for key in CONDITION_KEYS:
        assert key in GENERATION_DEFAULTS, f"{key} has no generation default"
    assert set(GENERATION_DEFAULTS) == set(CONDITION_KEYS)


def test_defaults_are_inside_the_measured_corpus_range():
    """Every default must sit in the corpus p10-p90 band of the median level.

    Measured by `.scratch/bpmfix/generation_defaults_survey.py` over 1,150 charts
    in the 12.8-13.8 band; a default outside the band would ask for a chart the
    corpus does not contain.
    """

    bands = {
        "level": (12.0, 15.0),
        "note_density": (0.238, 0.368), "peak": (0.250, 0.562), "sweep": (0.056, 0.181),
        "slide_tricky": (0.048, 0.186), "slide_sequence": (0.023, 0.133),
        "jack": (0.106, 0.282), "chord": (0.160, 0.405),
        "tap_bias": (0.590, 0.832), "slide_bias": (0.071, 0.228), "hold_bias": (0.031, 0.137),
        "touch_bias": (0.000, 0.140), "chord_bias": (0.160, 0.405),
        "melody_following": (0.803, 0.975), "pattern_reuse": (0.000, 0.107),
        "directionality": (0.007, 0.143),
    }
    for key, (low, high) in bands.items():
        value = GENERATION_DEFAULTS[key]
        assert low - 1e-9 <= value <= high + 1e-9, f"{key}={value} outside corpus band {low}-{high}"


def test_defaults_match_the_stored_corpus_profile():
    profile_path = next((path for path in PROFILE_CANDIDATES if path.is_file()), None)
    if profile_path is None:
        pytest.skip("no condition_profile.json in this workspace")
    controls = json.loads(profile_path.read_text(encoding="utf-8"))["controls"]
    for key, entry in controls.items():
        if key in AUDIO_MEASURED:
            continue
        median = float(entry["median"])
        assert abs(GENERATION_DEFAULTS[key] - median) <= 0.02, \
            f"{key}: default {GENERATION_DEFAULTS[key]} vs profile median {median}"
    assert controls["melody_following"]["median"] == 0.0, \
        "if the profile ever measures melody_following, AUDIO_MEASURED should be dropped"


def test_panel_sliders_use_the_same_numbers():
    sliders = _slider_defaults()
    assert sliders, "no slider defaults were found in app.js"
    for key in RADAR_KEYS + STYLE_KEYS:
        assert key in sliders, f"{key} is missing from the panel"
        assert abs(float(sliders[key]) - GENERATION_DEFAULTS[key]) <= 1e-9, (
            f"{key}: panel says {sliders[key]}, defaults say {GENERATION_DEFAULTS[key]}")


def test_panel_level_and_sampling_defaults_match_the_constants():
    html = INDEX_HTML.read_text(encoding="utf-8")

    def field_value(element_id: str) -> str:
        found = re.search(rf'id="{element_id}"[^>]*value="([^"]*)"', html)
        assert found, f"#{element_id} has no value attribute"
        return found.group(1)

    assert float(field_value("level")) == GENERATION_DEFAULTS["level"]
    assert float(field_value("repetitionPenalty")) == GENERATION_SAMPLING_DEFAULTS["repetition_penalty"]
    assert int(field_value("repetitionWindow")) == GENERATION_SAMPLING_DEFAULTS["repetition_window"]
    assert float(field_value("temperature")) == GENERATION_SAMPLING_DEFAULTS["temperature"]
    assert float(field_value("topP")) == GENERATION_SAMPLING_DEFAULTS["top_p"]
    assert float(field_value("guidanceScale")) == GENERATION_SAMPLING_DEFAULTS["guidance_scale"]
    assert int(field_value("candidates")) == GENERATION_SAMPLING_DEFAULTS["candidates"]
    # The panel's decode budget follows the backend default and is distinct from
    # the training target cap. Users may raise it for larger event_ctx prefixes.
    assert int(field_value("maxTokens")) == GENERATION_SAMPLING_DEFAULTS["max_tokens"]


def test_cli_defaults_match_the_panel():
    from maichart.generation.event_cli import build_parser

    required = ["--checkpoint", "c.pt", "--audio", "a.mp3", "--output", "o.txt"]
    args = build_parser().parse_args(required)
    assert args.level == GENERATION_DEFAULTS["level"]
    assert args.temperature == GENERATION_SAMPLING_DEFAULTS["temperature"]
    assert args.top_p == GENERATION_SAMPLING_DEFAULTS["top_p"]
    assert args.repetition_penalty == GENERATION_SAMPLING_DEFAULTS["repetition_penalty"]
    assert args.repetition_window == GENERATION_SAMPLING_DEFAULTS["repetition_window"]
    assert args.guidance_scale == GENERATION_SAMPLING_DEFAULTS["guidance_scale"]
    assert args.candidates == GENERATION_SAMPLING_DEFAULTS["candidates"]
    assert args.max_tokens == GENERATION_SAMPLING_DEFAULTS["max_tokens"]
    assert args.sample is True, "greedy decoding must be opt-in (--greedy)"
    assert args.tempo_mode == "constant"
    # --greedy flips it back for reproducibility checks, and --sample still works.
    assert build_parser().parse_args(required + ["--greedy"]).sample is False
    assert build_parser().parse_args(required + ["--sample"]).sample is True


def test_server_fallbacks_match_the_constants():
    source = (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    for needle in (
        'payload.get("level", GENERATION_DEFAULTS["level"])',
        'payload.get("temperature", GENERATION_SAMPLING_DEFAULTS["temperature"])',
        'payload.get("top_p", GENERATION_SAMPLING_DEFAULTS["top_p"])',
        'payload.get("repetition_penalty",',
        'payload.get("repetition_window",',
        'payload.get("sample", GENERATION_SAMPLING_DEFAULTS["sample"])',
        'payload.get("max_tokens", GENERATION_SAMPLING_DEFAULTS["max_tokens"])',
    ):
        assert needle in source, f"server fallback drifted: {needle}"
    # No stale literal may survive next to a payload lookup.
    assert 'payload.get("temperature", 0.65)' not in source
    assert 'payload.get("repetition_window", 32)' not in source
    assert 'payload.get("level", 13.0)' not in source
