"""Every model-capacity knob the trainer accepts must be reachable in the WebUI.

The feed-forward width was supported end to end -- the CLI took
``--dim-feedforward``, the server forwarded it, and the page sent it -- but its
input was marked as belonging to the legacy family and hidden, so an event-family
run silently used the default.  A knob nobody can see is a knob that does not
exist, so this pins the contract rather than the one field.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "src/maichart/webui/static/index.html").read_text(encoding="utf-8")
APP_JS = (ROOT / "src/maichart/webui/static/app.js").read_text(encoding="utf-8")

# Model-capacity inputs the event trainer actually consumes.
CAPACITY_FIELDS = ("dModel", "nhead", "numLayers", "dimFeedforward", "dropout")


def label_block(field_id: str) -> str:
    """Return the ``<label>`` element that owns an input id."""

    position = INDEX.find(f'id="{field_id}"')
    assert position != -1, f"#{field_id} is missing from index.html"
    start = INDEX.rfind("<label", 0, position)
    end = INDEX.find("</label>", position)
    return INDEX[start:end]




def test_training_filter_controls_and_workspace_tabs_exist():
    for field_id in ("minDifficultyIndex", "maxDifficultyIndex", "minLevel", "maxLevel",
                      "maxAudioTokens", "conditionDefault", "planLossWeight", "maxSteps", "validationMaxSteps",
                      "cudaMemoryBudgetGb"):
        assert f'id="{field_id}"' in INDEX
    for key in ("min_difficulty_index", "max_difficulty_index", "min_level", "max_level",
                "max_audio_tokens", "condition_default", "plan_loss_weight", "max_steps", "validation_max_steps"):
        assert key in APP_JS
    assert "cuda_memory_budget_gb" in APP_JS
    assert 'valueOrDefault("#cudaMemoryBudgetGb", 0)' in APP_JS
    cuda_budget = label_block("cudaMemoryBudgetGb")
    assert 'value="0"' in cuda_budget
    assert "60%" in cuda_budget and "不保证加速" in cuda_budget
    assert 'id="trainingModeTab"' in INDEX and 'id="generationModeTab"' in INDEX
    assert "setWorkspaceMode" in APP_JS
    assert 'condition_dropout: Number(valueOrDefault("#conditionDropout", 0))' in APP_JS
    assert "manifest: datasetPaths().manifest" in APP_JS
    server = (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    from maichart.webui.server import _event_condition_parameters
    assert _event_condition_parameters({})["dropout"] == 0.0
    assert '"--plan-loss-weight"' in server
    assert '"--max-steps"' in server and '"--validation-max-steps"' in server

    for field_id in CAPACITY_FIELDS:
        assert f'id="{field_id}"' in INDEX, f"#{field_id} missing from the page"
    for key in ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout"):
        assert re.search(rf"\b{key}:", APP_JS), f"{key} is never sent by the page"


def test_feedforward_width_is_visible_for_event_families():
    """The whole point: an event run must be able to set FFN."""

    block = label_block("dimFeedforward")
    families = re.search(r'data-training-family="([^"]+)"', block)
    assert families, "the feed-forward field must declare its families"
    declared = families.group(1).split()
    assert "event_v1" in declared, "event runs need to reach the feed-forward width"
    assert "legacy_frame_v2" in declared, "the legacy baseline keeps its own width"
    assert " hidden" not in block.split(">")[0], "it must not be hidden for the default family"


def test_family_visibility_accepts_several_families():
    """One control can serve more than one pipeline; the helper must allow it."""

    body = APP_JS[APP_JS.index("function familyFieldVisible"):]
    body = body[:body.index("\n}")]
    assert ".split(" in body, "familyFieldVisible must accept a space separated list"
    assert 'name === "event_v1" && current === "event_ctx_v1"' in body, \
        "the cross-window family still inherits every event_v1 field"


def test_global_frames_stays_cli_only_by_decision():
    """Global memory frames has no UI control: its default is the推荐 value.

    Pinned so the omission stays deliberate -- if someone wires it up, this test
    should be updated together with the page, not silently.
    """

    from maichart.generation.event_audio import GLOBAL_MEMORY_FRAMES

    assert GLOBAL_MEMORY_FRAMES == 256
    assert "globalFrames" not in INDEX
    assert "global_frames" not in APP_JS
    # ...but the CLI does expose it for advanced use.
    cli = (ROOT / "src/maichart/training/train_events.py").read_text(encoding="utf-8")
    assert "--global-frames" in cli


def test_documented_recommendation_matches_defaults():
    """The page defaults must not contradict the documented safe configuration."""

    def default_of(field_id: str) -> str:
        return re.search(r'id="' + field_id + r'"[^>]*value="([^"]*)"', INDEX).group(1)

    assert default_of("segmentBeats") == "16"
    assert default_of("trainMaxTokens") == "768"
    # 0.5 keeps every radar/style channel learnable while 1.0 shuts them all.
    assert default_of("conditionMaskProb") == "0.5"
    assert default_of("slideOversample") == "2"
    assert default_of("touchOversample") == "1"
    assert default_of("contextBeats") == "8"
    assert default_of("audioMarginBeats") == "4"
    assert default_of("contextTokens") == "192"
    assert default_of("overlapBeats") == "0"
    assert default_of("conditionDropout") == "0"
    assert default_of("planLossWeight") == "0.1"
    assert default_of("contextDropout") == "0.15"
    assert default_of("dModel") == "256"
    assert default_of("dimFeedforward") == "1024"
    assert default_of("dropout") == "0.1"
    assert default_of("weightDecay") == "0.01"


def test_checkpoint_inspection_rehydrates_event_context_contract():
    for key in ("context_beats", "audio_margin_beats", "context_tokens", "overlap_beats", "context_dropout"):
        assert f'familyParams.{key}' in APP_JS
    assert 'conditionParams.dropout' in APP_JS
    assert 'currentTrainingMode() === "resume"' in APP_JS
    assert 'training_capabilities?.independent_touch_oversample' in APP_JS
    assert '后端未加载独立Touch采样' in APP_JS
    assert 'independent_touch_oversample' in (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    server = (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    assert 'independent_note_types_touch_priority_v1' in server
    assert '不能普通 resume；请改用 finetune' in server


def test_local_onset_training_and_generation_controls_are_real_and_gated():
    assert 'id="onsetProfile"' in INDEX
    assert 'id="onsetLossWeight"' in INDEX
    assert 'id="localOnsetBiasStrength"' in INDEX
    assert 'value="0" disabled' in label_block("localOnsetBiasStrength")
    assert 'data-training-family="event_v1 event_ctx_v1"' in label_block("onsetProfile")
    assert 'onset_profile: valueOrDefault("#onsetProfile", "auto")' in APP_JS
    assert 'onset_loss_weight: Number(valueOrDefault("#onsetLossWeight", 0.2))' in APP_JS
    # Actual disabled/enabled serialization is exercised by the Node behavior
    # test; keep this markup test independent from the expression spelling.
    assert 'local_onset_bias_strength:' in APP_JS
    assert 'caps.local_onset_guidance === true' in APP_JS
    assert 'training_capabilities?.local_onset_training !== true' in APP_JS
    assert 'local_onset_guidance' in (ROOT / "src/maichart/webui/server.py").read_text(encoding="utf-8")
    assert "0 关闭" in INDEX
    assert "0.3–0.6" in INDEX
    assert "不替代强度曲线或密度限制" in INDEX
    assert "每个音乐峰" in INDEX


def _style_keys():
    """The radar/style slide definitions from app.js (app.js list literals)."""

    return re.findall(r'\["(\w+)",\s*"([^"]+)",\s*"((?:[^"\\]|\\.)*)",\s*"([\d.]+)"\]', APP_JS)


def test_generation_style_sliders_carry_measured_ranges_and_defaults():
    """Every radar/style control documents semantics + corpus range and starts at its median.

    Defaults are the level-conditioned corpus medians (see generation_defaults_survey); the
    training-side 0.5 mask/default is a null-conditioning mechanism, not a generation-page
    "no preference" value.
    """

    names = [entry[0] for entry in _style_keys()]
    assert names[:7] == ["note_density", "peak", "sweep", "slide_tricky",
                         "slide_sequence", "jack", "chord"]
    assert "melody_following" in names
    for key, label, help_text, fallback in _style_keys():
        assert help_text.strip(), f"{key} must document what it controls"
        # melody_following is measured from audio onsets (the profile cannot),
        # so its help text cites the measured song range rather than 语料.
        assert "语料" in help_text or "中位等级带" in help_text or "训练侧实测" in help_text, \
            f"{key} must cite a measured range"
        assert 0.0 < float(fallback) <= 1.0, f"{key} default must be within [0,1]"
    rhythm = [entry for entry in _style_keys() if entry[0] == "melody_following"][0]
    assert "非旋律" in rhythm[1], "the misleading name must be corrected in the label"
    # The density slider starts at the corpus median (≈4.9 nps), not the 8 nps
    # the old 0.5 silently asked for.
    density = [entry for entry in _style_keys() if entry[0] == "note_density"][0]
    assert density[3] == "0.305"


def test_generation_condition_explanations_and_presets_are_exposed():
    """The page explains backend condition names and offers one-click presets."""

    assert 'class="condition-guide"' in INDEX
    for term in ("note_density", "peak", "sweep", "slide_tricky", "slide_sequence",
                 "jack", "chord", "tap_bias", "slide_bias", "hold_bias", "touch_bias",
                 "chord_bias", "melody_following", "pattern_reuse", "directionality"):
        assert term in APP_JS or term in INDEX, f"missing condition terminology: {term}"
    assert 'id="levelRange"' in INDEX
    assert 'id="conditionPreferenceStrength"' in INDEX
    assert 'id="applyConditionPreset"' in INDEX
    assert 'on("#applyConditionPreset", "click"' in APP_JS
    for preference in ("keyboard", "stars", "technical", "chord", "hold", "touch", "movement"):
        assert f"{preference}:" in APP_JS, f"missing preference: {preference}"
    assert "composeLevelCondition" in APP_JS
    assert '<details class="condition-advanced">' in INDEX
    assert "condition_mask_prob" in INDEX
    assert "0.5 永远有效" not in INDEX
    for backend_term in ("Classifier-free guidance scale", "null 分支", "非结构 token",
                         "notes/sec", "slide_tricky", "slide_sequence"):
        assert backend_term in INDEX or backend_term in APP_JS
    assert 'id="generationDevice"' in INDEX
    assert 'device: valueOrDefault("#generationDevice", "auto")' in APP_JS
    generation_payload = APP_JS[APP_JS.index("function generationPayload"):]
    generation_payload = generation_payload[:generation_payload.index("\n}")]
    assert "...contextPayload()" not in generation_payload, \
        "event_ctx layout is selected from the checkpoint contract, not training controls"


def test_saved_settings_round_trip_covers_the_new_field():
    """Settings are collected from the DOM, so an unhidden field is persisted too.

    Nothing is hardcoded per field: every ``input[id]`` is saved and restored, so
    simply making the feed-forward width visible also makes it stick across
    reloads.  Pinned here because a hardcoded list would silently drop it.
    """

    settings = APP_JS[APP_JS.index("function settingsFields"):]
    settings = settings[:settings.index("\n}")]
    assert 'querySelectorAll("input[id], select[id], textarea[id]")' in settings
    assert 'element.type !== "file"' in settings


def test_generation_neutral_defaults():
    """The generation panel ships the measured neutral set, not taste knobs."""

    def default_of(field_id: str) -> str:
        found = re.search(r'id="' + field_id + r'"[^>]*value="([^"]*)"', INDEX)
        assert found, f"#{field_id} missing"
        return found.group(1)

    assert default_of("temperature") == "0.65"
    assert default_of("topP") == "0.9"
    assert default_of("repetitionPenalty") == "1.05", "mild penalty: the corpus repeats little"
    assert default_of("repetitionWindow") == "56"
    assert default_of("guidanceScale") == "1"
    assert default_of("candidates") == "1"
    assert default_of("level") == "13.3"
    sample = re.search(r'id="sampleEvents"[^>]*', INDEX).group(0)
    assert "checked" in sample, "sampling is the measured-neutral decode mode"


def test_restore_defaults_button_is_wired():
    """恢复默认设置 must clear both storages and reload over a fresh DOM."""

    assert 'id="resetSettingsButton"' in INDEX
    handler = APP_JS[APP_JS.index('$("#resetSettingsButton")'):]
    handler = handler[:handler.index("});") + 3]
    assert "discardSettings()" in handler
    assert "location.reload()" in handler
    clear = APP_JS[APP_JS.index("function discardSettings"):]
    clear = clear[:clear.index("\n}") + 2]
    assert "settingsStorageKey" in clear and "datasetStorageKey" in clear
    # The debounced saver must not resurrect the storage right after a reset.
    assert "clearTimeout(settingsSaveTimer)" in clear
    # The destructive scope must be stated on the button itself, so nobody
    # presses it thinking it only touches the generation knobs.
    start = INDEX.rindex("<button", 0, INDEX.index('id="resetSettingsButton"'))
    element = INDEX[start:INDEX.index("</button>", start)]
    assert "训练参数" in element, "the global reset must name its training-side scope"


def test_generation_reset_is_scoped_to_the_generation_panel():
    """A reset button next to the generation parameters that cannot touch training.

    The whole point of the split: training parameters are expensive to re-enter
    and users are wary of losing them, so the generation-panel reset must be
    scoped to an explicit allow-list of generation fields.
    """

    # It lives in the generation panel, on the sampling group header.
    generation_start = INDEX.index("生成谱面")
    button_at = INDEX.index('id="resetGenerationButton"')
    assert button_at > generation_start, "the button belongs to the generation panel"
    assert "采样与随机性" in INDEX[button_at - 600:button_at]

    handler = APP_JS[APP_JS.index('$("#resetGenerationButton")'):]
    handler = handler[:handler.index("});") + 3]
    assert "resetGenerationDefaults()" in handler
    assert "location.reload()" not in handler, "a scoped reset must not reload"

    fields = APP_JS[APP_JS.index("const GENERATION_PARAM_FIELDS"):]
    fields = fields[:fields.index("];")]
    for generation_id in ("temperature", "repetitionPenalty", "topP", "level", "seed",
                          "guidanceScale", "sampleEvents", "maxTokens", "generationDevice",
                          "conditionPreferenceStrength", "preferKeyboard", "preferStars", "preferTechnical"):
        assert f'"{generation_id}"' in fields, f"#{generation_id} must be reset"
    for training_id in ("epochs", "learningRate", "batchSize", "dModel", "numLayers",
                        "manifestPath", "checkpointDir", "rawRoot", "conditionMaskProb",
                        "weightDecay", "patience", "contextBeats", "slideOversample", "touchOversample"):
        assert f'"{training_id}"' not in fields, f"#{training_id} must never be touched"
    # Song inputs, paths and metadata stay out of the reset too.
    for other in ("bpm", "first", "tempoMap", "audioPath", "checkpointPath", "outputPath",
                  "title", "artist", "designer", "chartLevel"):
        assert f'"{other}"' not in fields, f"#{other} must never be touched"


def test_generation_reset_snapshots_the_dom_defaults_at_load():
    """The snapshot must happen after the sliders exist and before restore.

    The radar/style sliders are built by controlRows(), so capturing defaults
    earlier would silently drop all 15 of them from the reset.
    """

    build = APP_JS[APP_JS.index('controlRows("radarControls"'):]
    build = build[:build.index("updateTrainingControls();")]
    assert 'controlRows("styleControls", styleKeys);' in build
    assert "generationDefaults = captureGenerationDefaults();" in build
    assert build.index("captureGenerationDefaults()") < build.index("restoreSettings()"), \
        "defaults must be captured before saved settings overwrite them"


def test_every_button_is_wired_and_every_reference_exists():
    """No dead buttons, and no references to ids the page does not have.

    A partial front-end rewrite once left most buttons without handlers, so the
    page looked alive but did nothing.  Both directions are pinned here: every
    ``<button id=...>`` must be referenced by a binding, and every id the script
    looks up must exist in the markup.
    """

    html_ids = set(re.findall(r'id="([^"]+)"', INDEX))
    buttons = re.findall(r"<button[^>]*id=\"([^\"]+)\"", INDEX)
    assert buttons, "the page must expose buttons"

    referenced = set(re.findall(r'\$\("#([A-Za-z0-9_]+)"\)', APP_JS))
    referenced |= set(re.findall(r'\bon\("#([A-Za-z0-9_]+)"', APP_JS))
    referenced |= set(re.findall(r'\belement\("#([A-Za-z0-9_]+)"\)', APP_JS))

    dead = [button for button in buttons if button not in referenced]
    assert not dead, f"buttons without any handler: {dead}"

    unknown = sorted(name for name in referenced if name not in html_ids)
    assert not unknown, f"app.js references ids missing from index.html: {unknown}"


def test_handlers_are_registered_for_every_interactive_surface():
    """Key workflows must have a click handler, not just a disabled-state rule."""

    for button in ("preprocessButton", "inspectManifestButton", "trainButton",
                   "autoPipelineButton", "goldScanButton", "goldBuildButton",
                   "refreshLogsButton", "generateButton", "acceptanceButton",
                   "cancelButton", "trainingModeTab", "generationModeTab"):
        assert f'on("#{button}"' in APP_JS or f'$("#{button}")' in APP_JS, \
            f"#{button} has no click handler"
