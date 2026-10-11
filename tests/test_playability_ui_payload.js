"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");

function payloadFor(controls) {
  const begin = source.indexOf("function generationPayload() {");
  const end = source.indexOf("function intensityCurvePayload()", begin);
  const helperStart = source.indexOf("const CONFLICT_KIND_FIELDS = [");
  const helperEnd = source.indexOf("let activeTaskId", helperStart);
  assert.ok(helperStart >= 0 && helperEnd > helperStart);
  const getter = (key, fallback) => Object.hasOwn(controls, key) ? controls[key] : fallback;
  return new Function("element", "valueOrDefault", "checkedOrDefault", "collectCondition", "saveSettings",
    "currentModelFamily", "datasetPaths", "intensityCurvePayload", "parseTempoMap",
    source.slice(helperStart, helperEnd) + source.slice(begin, end) + ";return generationPayload();")(
      () => null, getter, getter, () => ({ level: 13.3 }),
      () => {}, () => "event_ctx_v1", () => ({ manifest: "" }), () => null, () => null);
}

function controlsWithPreset(name) {
  const values = {
    "#conflictDetectionEnabled": true,
    "#conflictRepairMode": "local",
    "#conflictRetries": "2",
    "#conflictBacktrackWindows": "-1",
    "#conflictBacktrackBudget": "4",
    "#sustainRepairMode": "off",
  };
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) {
      const key = selector.startsWith("#") ? selector : selector;
      const control = { value: values[key] ?? "", checked: values[key] === true };
      elements.set(selector, control);
    }
    return elements.get(selector);
  };
  Object.entries(values).forEach(([selector, value]) => {
    const target = element(selector);
    target.value = String(value);
    target.checked = value === true;
  });
  const helperStart = source.indexOf("const CONFLICT_KIND_FIELDS = [");
  const helperEnd = source.indexOf("let activeTaskId", helperStart);
  assert.ok(helperStart >= 0 && helperEnd > helperStart);
  new Function("element", "valueOrDefault", "saveSettings",
    source.slice(helperStart, helperEnd) + `; applyConflictPreset(${JSON.stringify(name)});`)(
      element,
      (selector, fallback) => element(selector)?.value ?? fallback,
      () => {},
    );
  return {
    ...values,
    ...Object.fromEntries([
      "conflictMultiTouchMode", "conflictOverlapMode", "conflictTapOnSlideMode",
      "conflictSlideHeadTapMode", "conflictSlideTooFastMode", "tapOnSlideThresholdMs",
    ].map(id => [`#${id}`, element(`#${id}`).value])),
  };
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

test("presets and advanced controls reach the generation payload", () => {
  const preserve = payloadFor(controlsWithPreset("preserve"));
  assert.deepEqual(preserve.conflict_kind_modes, {
    MultiTouch: "report", Overlap: "enforce", TapOnSlide: "enforce",
    SlideHeadTap: "enforce", SlideTooFast: "enforce",
  });
  assert.equal(preserve.tap_on_slide_threshold_ms, 150);

  const cautiousControls = controlsWithPreset("cautious");
  cautiousControls["#conflictTapOnSlideMode"] = "off";
  cautiousControls["#conflictBacktrackWindows"] = "2";
  cautiousControls["#conflictBacktrackBudget"] = "8";
  cautiousControls["#sustainRepairMode"] = "hold_slide";
  const cautious = payloadFor(cautiousControls);
  assert.equal(cautious.conflict_kind_modes.TapOnSlide, "off");
  assert.equal(cautious.tap_on_slide_threshold_ms, 250);
  assert.equal(cautious.conflict_backtrack_windows, 2);
  assert.equal(cautious.conflict_backtrack_budget, 8);
  assert.equal(cautious.sustain_repair_mode, "hold_slide");
});

test("note repair policy is local-only and preserves the three backend modes", () => {
  const edit = payloadFor({
    "#conflictDetectionEnabled": true, "#conflictRepairMode": "local",
    "#noteRepairMode": "edit", "#conflictRetries": "1",
  });
  assert.equal(edit.note_repair_mode, "edit");
  const fallback = payloadFor({
    "#conflictDetectionEnabled": true, "#conflictRepairMode": "reject",
    "#noteRepairMode": "edit_then_delete",
  });
  assert.equal(fallback.note_repair_mode, "delete");
});
