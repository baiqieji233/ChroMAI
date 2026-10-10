"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const begin = source.indexOf("function generationPayload() {");
const end = source.indexOf("function intensityCurvePayload()", begin);
assert.ok(begin >= 0 && end > begin);

test("a stored onset strength is suppressed for an unsupported checkpoint", () => {
  const control = {disabled: true, value: "0.35"};
  const element = selector => selector === "#localOnsetBiasStrength" ? control : null;
  const value = (selector, fallback) => selector === "#localOnsetBiasStrength" ? control.value : fallback;
  const payload = new Function("element", "valueOrDefault", "checkedOrDefault", "collectCondition",
    "currentModelFamily", "datasetPaths", "intensityCurvePayload", "parseTempoMap",
    source.slice(begin, end) + "; return generationPayload;")(
      element, value, (_selector, fallback) => fallback, () => ({level:13.3}),
      () => "event_ctx_v1", () => ({manifest:""}), () => null, () => null,
    );
  assert.equal(payload().local_onset_bias_strength, 0);
  control.disabled = false;
  assert.equal(payload().local_onset_bias_strength, 0.35);
  control.disabled = true;
  assert.equal(payload().local_onset_bias_strength, 0);
});
