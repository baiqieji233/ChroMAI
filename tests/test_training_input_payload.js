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
