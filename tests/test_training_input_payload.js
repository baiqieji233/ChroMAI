"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/index.html"), "utf8");
const begin = source.indexOf("function trainingPayload(options = {}) {");
const end = source.indexOf("function collectValues(", begin);
assert.ok(begin >= 0 && end > begin);

function payloadFor(family, mode, values) {
  return new Function("currentTrainingMode", "trainingSourcePaths", "selectedTrainingManifests",
    "selectedDatasetItems", "valueOrDefault", "currentModelFamily", "checkedOrDefault",
    "element", "contextPayload", "numberOrDefault", "qualityState", "qualityManifest",
    source.slice(begin, end) + "; return trainingPayload();")(
      () => mode, () => ({}), () => [], () => [],
      (selector, fallback) => values[selector] ?? fallback, () => family,
      (_selector, fallback) => fallback, () => null, () => ({}),
      (_selector, fallback) => fallback, {}, () => "",
    );
}

test("visible independent pooling and sustained-action choices reach the training request", () => {
  const payload = payloadFor("event_ctx_v1", "finetune", {
    "#actionStateProfile": "sustain_state_v1", "#localPoolingPolicy": "per_sample_v2",
    "#onsetProfile": "none", "#trainingCheckpointPath": "old.pt",
  });
  assert.equal(payload.action_state_profile, "sustain_state_v1");
  assert.equal(payload.local_pooling_policy, "per_sample_v2");
  assert.equal(payload.onset_profile, "none");
  assert.equal(payload.context_budget_policy, "auto");
  assert.equal(payload.validation_manifest, null);
});

test("auto inherits in the backend, and legacy frame requests omit event-only inputs", () => {
  const defaults = payloadFor("event_ctx_v1", "resume", {});
  assert.equal(defaults.action_state_profile, "auto");
  assert.equal(defaults.local_pooling_policy, "auto");
  const legacy = payloadFor("legacy_frame_v2", "finetune", {"#actionStateProfile": "sustain_state_v1"});
  assert.equal(legacy.action_state_profile, undefined);
  assert.equal(legacy.local_pooling_policy, undefined);
});

test("song form submits every selected source for backend composition", () => {
  const formBegin = source.indexOf("function songFormTrainingPayload(options = {}) {");
  const formEnd = source.indexOf('// Shared by', formBegin);
  const items = [{manifest: "a.json", path: "dataset-a"}, {manifest: "b.json", path: "dataset-b"}];
  const build = new Function("currentTrainingMode", "trainingSourcePaths", "selectedTrainingManifests",
    "selectedDatasetItems", "valueOrDefault", "formDevice", "checkedOrDefault",
    source.slice(formBegin, formEnd) + "; return songFormTrainingPayload();");
  for (const qualityEnabled of [false, true]) {
    const payload = build(() => "fresh", () => ({manifest: "a.json"}),
      () => items.map((item) => item.manifest), () => items,
      (selector, fallback) => ({"#formPrecision": "bf16", "#formLoaderWorkers": 2,
        "#formPrefetchFactor": 3, "#formLogInterval": 4, "#formCheckpointIntervalSteps": 80,
        "#formAudioCacheGb": 1.5, "#formAudioCacheMode": "resident",
        "#formSampleCacheMb": 512}[selector] ?? fallback),
      () => "cpu", () => qualityEnabled);
    assert.deepEqual(payload.manifests, ["a.json", "b.json"]);
    assert.deepEqual(payload.dataset_paths, ["dataset-a", "dataset-b"]);
    assert.equal(payload.model_family, "song_form_v1");
    assert.equal(payload.quality_enabled, qualityEnabled);
    assert.equal(payload.segment_beats, undefined);
    assert.deepEqual([payload.precision, payload.loader_workers, payload.prefetch_factor,
      payload.log_interval, payload.checkpoint_interval_steps, payload.audio_cache_gb,
      payload.audio_cache_mode, payload.sample_cache_mb], ["bf16", 2, 3, 4, 80, 1.5, "resident", 512]);
  }
});

test("main song-form generation payload forwards the onset merge range", () => {
  const begin = source.indexOf("function songFormGenerationPayload() {");
  const end = source.indexOf("function generationPayload()", begin);
  assert.ok(begin >= 0 && end > begin);
  const build = new Function("valueOrDefault", "checkedOrDefault", "collectCondition",
    "formDevice", "parseTempoMap", "intensityCurvePayload", "songFormRhythmSnapDivisions",
    source.slice(begin, end) + "; return songFormGenerationPayload();");
  const payload = build((selector, fallback) => ({"#formOnsetMergeMs": "0", "#temperature": "0.2", "#seed": ""}[selector] ?? fallback),
    (_selector, fallback) => fallback, () => ({}), () => "auto", () => null, () => null, () => [4, 8, 12, 16, 24, 32]);
  assert.equal(payload.onset_merge_ms, 0);
  assert.equal(payload.temperature, 0.2);
  assert.equal(payload.seed, null);
  assert.equal(payload.device, "auto");
});

test("main song-form generation payload forwards both type guidance strengths", () => {
  const begin = source.indexOf("function songFormGenerationPayload() {");
  const end = source.indexOf("function generationPayload()", begin);
  const build = new Function("valueOrDefault", "checkedOrDefault", "collectCondition",
    "formDevice", "parseTempoMap", "intensityCurvePayload", "songFormRhythmSnapDivisions",
    source.slice(begin, end) + "; return songFormGenerationPayload();");
  const payload = build((selector, fallback) => ({
    "#formPlannerTypeStrength": "2.4", "#formStyleBalanceStrength": "0",
  }[selector] ?? fallback), (_selector, fallback) => fallback, () => ({}), () => "auto",
  () => null, () => null, () => [4, 8, 12, 16, 24, 32]);
  assert.equal(payload.planner_type_strength, 2.4);
  assert.equal(payload.style_balance_strength, 0);

  const defaults = build((_selector, fallback) => fallback, (_selector, fallback) => fallback,
    () => ({}), () => "auto", () => null, () => null, () => [4, 8, 12, 16, 24, 32]);
  assert.equal(defaults.planner_type_strength, 1.0);
  assert.equal(defaults.style_balance_strength, 1.0);
});

test("legacy event generation payload does not include song-form type guidance", () => {
  const begin = source.indexOf("function generationPayload() {");
  const end = source.indexOf("function ", begin + 12);
  const build = new Function("currentModelFamily", "songFormGenerationPayload", "collectCondition",
    "valueOrDefault", "datasetPaths", "element", "checkedOrDefault", "intensityCurvePayload", "parseTempoMap",
    source.slice(begin, end) + "; return generationPayload();");
  const payload = build(() => "event_v1", () => ({}), () => ({level: 13.3}),
    (_selector, fallback) => fallback, () => ({manifest: "manifest.json"}), () => ({value: ""}),
    (_selector, fallback) => fallback, () => null, () => null);
  assert.equal(payload.planner_type_strength, undefined);
  assert.equal(payload.style_balance_strength, undefined);
});

test("shared formDevice leaves auto for backend hardware detection", () => {
  const begin = source.indexOf("function formDevice(selector) {");
  const end = source.indexOf("function songFormRhythmSnapDivisions", begin);
  const formDevice = new Function("valueOrDefault", source.slice(begin, end) + "; return formDevice;")(
    (_selector, fallback) => fallback);
  assert.equal(formDevice("#generationDevice"), "auto");
});

test("candidate count is hidden outside event and legacy generation", () => {
  assert.match(html, /class="field" data-generation-family="event_v1 legacy_frame_v2">候选数量/);
});

test("song-form curve restrictions reset unsupported saved values and restore event options", () => {
  const begin = source.indexOf("function updateSongFormGenerationOptions() {");
  const end = source.indexOf("function updateFamilyFields()", begin);
  const curveMode = {options: ["follow", "hard", "soft", "strict"].map(value => ({value, textContent: value})), value: "soft",
    querySelector: selector => selector.includes("hard") ? curveMode.options.find(option => option.value === "hard") : null};
  const audioOption = {disabled: false};
  const curveSource = {value: "audio", querySelector: () => audioOption};
  let family = "song_form_v1", saves = 0;
  const apply = new Function("currentGenerationFamily", "element", "saveSettings",
    "songFormCurveSourceBeforeRestriction", "songFormCurveModeBeforeRestriction", "songFormCurveRestrictionsActive",
    source.slice(begin, end) + "; return updateSongFormGenerationOptions;")(
      () => family, selector => selector === "#intensityCurveMode" ? curveMode
        : selector === "#intensityCurveSource" ? curveSource : null,
      () => { saves += 1; }, null, null, false);
  apply();
  assert.equal(curveMode.value, "follow");
  assert.equal(curveMode.options.find(option => option.value === "soft").disabled, true);
  assert.equal(curveMode.options.find(option => option.value === "strict").disabled, true);
  assert.equal(curveMode.options.find(option => option.value === "hard").textContent, "hard（放宽幅度裁剪）");
  assert.equal(curveSource.value, "model");
  assert.equal(audioOption.disabled, true);
  family = "event_v1";
  apply();
  assert.equal(curveMode.value, "soft");
  assert.equal(curveMode.options.find(option => option.value === "soft").disabled, false);
  assert.equal(curveMode.options.find(option => option.value === "hard").textContent, "hard（严苛跟随）");
  assert.equal(curveSource.value, "audio");
  assert.equal(audioOption.disabled, false);
  assert.equal(saves, 3);
});

test("main song-form rhythm snapping defaults off and forwards its tolerance", () => {
  const begin = source.indexOf("function songFormGenerationPayload() {");
  const end = source.indexOf("function generationPayload()", begin);
  const build = new Function("valueOrDefault", "checkedOrDefault", "collectCondition",
    "formDevice", "parseTempoMap", "intensityCurvePayload", "songFormRhythmSnapDivisions",
    source.slice(begin, end) + "; return songFormGenerationPayload();");
  const payload = build((selector, fallback) => ({"#formRhythmDivisionCustom": "20,40"}[selector] ?? fallback),
    (selector, fallback) => selector === "#formRhythmSnapEnabled" || selector === "#formRhythmDivision48" ? true : fallback,
    () => ({}), () => "cpu", () => null, () => null,
    (enabled) => enabled ? [4, 8, 12, 16, 24, 32, 48, 20, 40] : [4, 8, 12, 16, 24, 32]);
  assert.equal(payload.rhythm_snap_enabled, true);
  assert.equal(payload.rhythm_snap_ms, 15);
  assert.deepEqual(payload.rhythm_snap_divisions, [4, 8, 12, 16, 24, 32, 48, 20, 40]);
  assert.equal(payload.onset_merge_ms, 40);
});

test("main song-form division choices merge custom values and reject an empty enabled grid", () => {
  const begin = source.indexOf("function songFormRhythmSnapDivisions(enabled) {");
  const end = source.indexOf("function songFormTrainingPayload(options = {})", begin);
  const parse = new Function("valueOrDefault", "checkedOrDefault",
    source.slice(begin, end) + "; return songFormRhythmSnapDivisions;")(
      (selector, fallback) => selector === "#formRhythmDivisionCustom" ? "20,20,48" : fallback,
      (selector, fallback) => selector === "#formRhythmDivision64" ? true : fallback);
  assert.deepEqual(parse(false), [4, 8, 12, 16, 24, 32, 64, 20, 48]);
  const empty = new Function("valueOrDefault", "checkedOrDefault",
    source.slice(begin, end) + "; return songFormRhythmSnapDivisions;")(
      (_selector, fallback) => fallback,
      () => false);
  assert.throws(() => empty(true), /至少选择一种候选分音/);
});
