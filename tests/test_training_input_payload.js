"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
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
    "formDevice", "parseTempoMap", "intensityCurvePayload",
    source.slice(begin, end) + "; return songFormGenerationPayload();");
  const payload = build((selector, fallback) => selector === "#formOnsetMergeMs" ? "0" : fallback,
    (_selector, fallback) => fallback, () => ({}), () => "cpu", () => null, () => null);
  assert.equal(payload.onset_merge_ms, 0);
});
