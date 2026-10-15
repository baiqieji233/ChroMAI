"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.join(__dirname, "..");
const app = fs.readFileSync(path.join(root, "src/maichart/webui/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(root, "src/maichart/webui/static/index.html"), "utf8");

test("rhythm geometry UI exposes auto, legacy, and exact profiles without a grid selector", () => {
  assert.match(html, /id="formRhythmGeometryProfileSelect"[\s\S]*?value="auto" selected[\s\S]*?value="none"[\s\S]*?value="exact_fraction_v1"/);
  assert.match(html, /id="formRhythmLossWeight"[^>]*placeholder="自动"/);
  assert.match(html, /精确拍位/);
  assert.match(app, /不吸附固定分音，也不使用固定帧率/);
});

test("geometry profile state preserves resume and requires explicit finetune migration", () => {
  const begin = app.indexOf("function songFormRhythmGeometryUiState(");
  const end = app.indexOf("function updateSongFormRhythmGeometryControls()", begin);
  assert.ok(begin >= 0 && end > begin);
  const resolve = new Function(`${app.slice(begin, end)}; return songFormRhythmGeometryUiState;`)();

  assert.deepEqual(resolve("fresh", "auto"), {effective:"exact_fraction_v1", locked:false, upgradeNeedsExplicitChoice:false});
  assert.deepEqual(resolve("resume", "exact_fraction_v1", "none"), {effective:"none", locked:true, upgradeNeedsExplicitChoice:false});
  assert.deepEqual(resolve("finetune", "auto", "none"), {effective:"none", locked:false, upgradeNeedsExplicitChoice:true});
  assert.deepEqual(resolve("finetune", "exact_fraction_v1", "none"), {effective:"exact_fraction_v1", locked:false, upgradeNeedsExplicitChoice:false});
  assert.deepEqual(resolve("finetune", "none", "exact_fraction_v1"), {effective:"exact_fraction_v1", locked:true, upgradeNeedsExplicitChoice:false});
});

test("song-form training payload forwards profile intent and optional loss", () => {
  const begin = app.indexOf("function songFormTrainingPayload(options = {}) {");
  const end = app.indexOf("// Shared by", begin);
  assert.ok(begin >= 0 && end > begin);
  const values = {"#formRhythmGeometryProfileSelect":"exact_fraction_v1", "#formRhythmLossWeight":"0.2"};
  const get = (selector, fallback) => Object.hasOwn(values, selector) ? values[selector] : fallback;
  const build = new Function("currentTrainingMode", "trainingSourcePaths", "selectedTrainingManifests", "selectedDatasetItems", "valueOrDefault", "currentModelFamily", "formDevice", "checkedOrDefault", `${app.slice(begin, end)}; return songFormTrainingPayload;`)(
    () => "fresh", () => ({manifest:"dataset.json"}), () => [], () => [], get, () => "song_form_v2", () => "cpu", (_selector, fallback) => fallback);
  assert.deepEqual({rhythm_geometry_profile:build().rhythm_geometry_profile, rhythm_loss_weight:build().rhythm_loss_weight},
    {rhythm_geometry_profile:"exact_fraction_v1", rhythm_loss_weight:0.2});
  values["#formRhythmLossWeight"] = "";
  assert.equal(build().rhythm_loss_weight, null);
});
