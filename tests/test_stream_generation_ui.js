"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/index.html"), "utf8");

function install(context, name, next) {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf(`function ${next}(`, start);
  assert.ok(start >= 0 && end > start);
  vm.runInContext(source.slice(start, end), context);
}

test("visible device selector has no family restriction or collapsed ancestor", () => {
  const stack = [];
  let selector = null;
  for (const match of html.matchAll(/<\/?([a-z][a-z0-9]*)\b[^>]*>/gi)) {
    const tag = match[1].toLowerCase();
    if (match[0].startsWith("</")) {
      const index = stack.findLastIndex(row => row.tag === tag);
      if (index >= 0) stack.splice(index);
    } else {
      if (/id="generationDevice"/.test(match[0])) selector = [...stack, {tag, text: match[0]}];
      if (!["input", "br", "img", "meta", "link", "hr", "source", "wbr"].includes(tag)) {
        stack.push({tag, text: match[0]});
      }
    }
  }
  assert.ok(selector);
  assert.equal([...html.matchAll(/id="generationDevice"/g)].length, 1);
  for (const ancestor of selector) {
    assert.doesNotMatch(ancestor.text, /data-generation-family=|\bhidden\b/);
    assert.notEqual(ancestor.tag, "details");
  }
  const select = html.slice(html.indexOf('id="generationDevice"'));
  const values = [...select.slice(0, select.indexOf("</select>")).matchAll(/value="([^"]+)"/g)].map(m => m[1]);
  assert.deepEqual(values, ["auto", "cuda", "cpu"]);
});

for (const device of ["auto", "cuda", "cpu"]) {
  test(`stream payload retains ${device}, seed and temperature zero`, () => {
    const values = {"#generationDevice": device, "#seed": "42", "#temperature": "0", "#bpm": "130"};
    const context = vm.createContext({
      valueOrDefault: (key, fallback) => values[key] ?? fallback,
      collectCondition: () => ({level: 13.6}), parseTempoMap: () => null,
    });
    install(context, "formDevice", "songFormRhythmSnapDivisions");
    install(context, "chartStreamGenerationPayload", "intensityCurvePayload");
    const result = context.chartStreamGenerationPayload();
    assert.equal(result.device, device);
    assert.equal(result.temperature, 0);
    assert.equal(result.seed, 42);
    assert.equal(result.bpm, 130);
    assert.equal(result.position_beam_size, 4);
    assert.equal(result.position_complexity, .35);
    assert.equal(result.position_interval_strength, 1);
    assert.equal(result.early_eos_penalty, 0);
  });
}

test("empty seed and BPM delegate to server random seed and audio tempo", () => {
  const context = vm.createContext({valueOrDefault: (_key, fallback) => fallback,
    collectCondition: () => ({}), parseTempoMap: () => null, formDevice: () => "auto"});
  install(context, "chartStreamGenerationPayload", "intensityCurvePayload");
  const result = context.chartStreamGenerationPayload();
  assert.equal(result.seed, null);
  assert.equal(result.bpm, null);
});

test('stream timing controls are visible in its family panel and forwarded', () => {
  const values = {'#streamPositionBeamSize':'8', '#streamPositionTemperature':'0',
    '#streamPositionComplexity':'1.2', '#streamMaxDenominator':'96',
    '#streamMaxIntervalDenominator':'24', '#streamMinSpacingMs':'30', '#streamEarlyEosPenalty':'0.5'};
  const context = vm.createContext({valueOrDefault: (key, fallback) => values[key] ?? fallback,
    collectCondition: () => ({}), parseTempoMap: () => null, formDevice: () => 'cpu'});
  install(context, 'chartStreamGenerationPayload', 'intensityCurvePayload');
  const result = context.chartStreamGenerationPayload();
  for (const id of Object.keys(values)) assert.ok(html.includes(`id="${id.slice(1)}"`));
  assert.equal(result.position_beam_size, 8);
  assert.equal(result.position_temperature, 0);
  assert.equal(result.position_complexity, 1.2);
  assert.equal(result.max_denominator, 96);
  assert.equal(result.max_interval_denominator, 24);
  assert.equal(result.min_spacing_ms, 30);
  assert.equal(result.early_eos_penalty, .5);
});

for (const family of ['chart_stream_v1', 'song_form_v2']) {
  test(`checkpoint inspection detects ${family} independently of training selection`, async () => {
    const calls = [], messages = [];
    const context = vm.createContext({
      valueOrDefault: () => 'stream.pt', generationInspectRequest: 0,
      currentModelFamily: () => 'song_form_v2', element: () => null,
      updateFamilyFields() {}, updateGenerationCapabilityControls() {},
      setGenerationStatus: text => messages.push(text),
      api: async (url, payload) => {
        calls.push({url, payload});
        return {valid: true, can_generate: true, model_family: family,
          generation_capabilities: {continuous_stream: true, rational_event_positions: true}};
      },
    });
    const begin = source.indexOf('async function inspectGenerationCheckpoint()');
    const end = source.indexOf('function updateGoldFields()', begin);
    vm.runInContext(source.slice(begin, end), context);
    const result = await context.inspectGenerationCheckpoint();
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/checkpoint/inspect');
    assert.equal(result.model_family, family);
    assert.equal(context.generationInspection.data.model_family, family);
    assert.ok(messages.at(-1).includes(family));
    if (family === 'chart_stream_v1') assert.ok(messages.at(-1).includes('连续整曲生成可用'));
  });
}
