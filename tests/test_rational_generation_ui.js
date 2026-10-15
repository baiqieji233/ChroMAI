"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");

test("main-page rational controls reach payload and running-server condition transport", () => {
  const values = {"#formPositionComplexity": "0.4", "#formPositionBeamSize": "4", "#formEarlyEosPenalty": "2"};
  const context = {currentGenerationFamily: () => "song_form_v2", collectCondition: () => ({level:13.3}),
    valueOrDefault: (key, fallback) => Object.hasOwn(values, key) ? values[key] : fallback,
    checkedOrDefault: (_key, fallback) => fallback,
    parseTempoMap: () => null, formDevice: () => "cpu", intensityCurvePayload: () => null,
    songFormRhythmSnapDivisions: () => [4,8,12,16,24,32],
    songFormRationalGridDivisions: () => [4,6,8,12,16,24,32,48,64,96]};
  vm.createContext(context);
  const start = source.indexOf("function songFormGenerationPayload() {");
  const end = source.indexOf("function uniqueSongFormOutputPath(", start);
  assert.ok(end > start);
  vm.runInContext(source.slice(start, end), context);
  const payload = context.songFormGenerationPayload();
  assert.equal(payload.position_complexity, 0.4);
  assert.equal(payload.position_beam_size, 4);
  assert.equal(payload.early_eos_penalty, 2);
  assert.equal(payload.condition.generation_options.early_eos_penalty, 2);
  assert.equal(payload.rhythm_snap_enabled, false);
});

test("rational diagnostics survive both task fields and streamed log records", () => {
  const context = {};
  vm.createContext(context);
  const start = source.indexOf("function taskOnsetDiagnostics(task) {");
  const end = source.indexOf("function renderTaskOnsetDiagnostics(task) {", start);
  vm.runInContext(source.slice(start, end), context);
  const evidence = {schema:"song_form_rational_diagnostics_v1",empty_eos_spans:16};
  assert.equal(context.taskOnsetDiagnostics({result:{rational_diagnostics:evidence}}).empty_eos_spans, 16);
  assert.equal(context.taskOnsetDiagnostics({logs:["form_rational_diagnostics "+JSON.stringify(evidence)]}).empty_eos_spans, 16);
});
