"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");

function payloadFor(controls) {
  const begin = source.indexOf("function generationPayload() {");
  const end = source.indexOf("function intensityCurvePayload()", begin);
  const getter = (key, fallback) => Object.hasOwn(controls, key) ? controls[key] : fallback;
  return new Function("element", "valueOrDefault", "checkedOrDefault", "collectCondition",
    "currentModelFamily", "datasetPaths", "intensityCurvePayload", "parseTempoMap",
    source.slice(begin, end) + ";return generationPayload();")(
      () => null, getter, getter, () => ({ level: 13.3 }),
      () => "event_ctx_v1", () => ({ manifest: "" }), () => null, () => null);
}

test("turning off detection suppresses stale local-repair controls", () => {
  const payload = payloadFor({ "#conflictDetectionEnabled": false, "#conflictRepairMode": "local", "#conflictRetries": "5" });
  assert.equal(payload.conflict_repair_mode, "none");
  assert.equal(payload.conflict_retries, 0);
});

test("local mode reaches backend and old external executable is never sent", () => {
  const payload = payloadFor({ "#conflictDetectionEnabled": true, "#conflictRepairMode": "local", "#conflictRetries": "3",
    "#runtimeReferenceExecutable": "D:/MiaCode.exe", "#runtimeReferenceMode": "reject" });
  assert.equal(payload.conflict_repair_mode, "local");
  assert.equal(payload.conflict_retries, 3);
  assert.ok(!Object.hasOwn(payload, "runtime_reference_executable"));
  assert.ok(!Object.hasOwn(payload, "runtime_reference_mode"));
});

test("detect-only suppresses a previously selected retry budget", () => {
  const payload = payloadFor({ "#conflictDetectionEnabled": true, "#conflictRepairMode": "none", "#conflictRetries": "5" });
  assert.equal(payload.conflict_retries, 0);
});
