"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/index.html"), "utf8");

function block(begin, end) {
  const start = source.indexOf(begin);
  const stop = source.indexOf(end, start);
  assert.ok(start >= 0 && stop > start, `missing ${begin}`);
  return source.slice(start, stop);
}

const qualityFields = [...block("const qualityFields = [", "];").matchAll(
  /\["(\w+)", "#(\w+)", ([\d.]+), ([\d.]+), "/g)].map(([, key, id, fallback]) => ({key, id, fallback: Number(fallback)}));

function htmlInput(id) {
  const match = html.match(new RegExp(`<input id="${id}" type="number"[^>]*>`));
  assert.ok(match, `${id} must be a real numeric input with ASCII-quoted attributes`);
  return match[0];
}

test("every submitted quality control exists as a numeric input", () => {
  assert.equal(qualityFields.length, 9);
  for (const {id} of qualityFields) htmlInput(id);
  // Typographic quotes as attribute delimiters silently break ids and styles.
  assert.doesNotMatch(html, /=[“”]|[“”]>/);
});

test("html defaults, submit fallbacks and the recommended recipe agree", () => {
  const recipeSource = block("function applySongFormQualityRecipe() {", "function songFormTrainingPayload(");
  const recommended = recipeSource.match(/history: \[([^\]]+)\]/)[1].split(",").map(Number);
  const ids = [...recipeSource.match(/const ids = \[([^\]]+)\]/)[1].matchAll(/"(\w+)"/g)].map((match) => match[1]);
  assert.equal(ids.length, recommended.length);
  for (const {id, fallback} of qualityFields) {
    const value = Number(htmlInput(id).match(/value="([^"]*)"/)[1]);
    assert.equal(value, fallback, `${id} html default`);
    if (id !== "formRepetitionPenaltyWeight") assert.equal(recommended[ids.indexOf(id)], value, `${id} recipe`);
  }
  assert.match(html, /<select id="formQualityRecipe"><option value="history" selected>/);
  assert.match(htmlInput("formRepetitionPenaltyWeight"), /disabled/);
});

test("saved legacy all-off values migrate once, customized values stay", () => {
  const migrate = new Function(block("function migrateSongFormQualityDefaults(values) {", "function saveSettings(")
    + "; return migrateSongFormQualityDefaults;")();
  const legacy = () => ({formQualityRecipe: "off", formPerturbApplyProb: "1", formPerturbCurveNoiseStd: "0",
    formDiversityLossWeight: "0", learningRate: "0.0002"});

  const values = legacy();
  assert.equal(migrate(values), true);
  assert.deepEqual(values, {learningRate: "0.0002"});

  assert.equal(migrate({...legacy(), formQualityRecipe: "custom"}), false);
  const changed = {...legacy(), formPerturbCurveNoiseStd: "0.05"};
  assert.equal(migrate(changed), false);
  assert.equal(changed.formPerturbCurveNoiseStd, "0.05");
  assert.equal(migrate({...legacy(), __songFormQualityDefaults: 2}), false);
});
