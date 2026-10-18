"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/index.html"), "utf8");

function install(context, name, next) {
  if (!context.__streamGroundingInstalled) {
    const a=source.indexOf('const STREAM_GROUNDING_FIELDS =');
    const b=source.indexOf('function chartStreamTrainingPayload(',a);
    vm.runInContext(source.slice(a,b),context);
    context.__streamGroundingInstalled=true;
  }
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
    assert.equal(result.position_complexity_floor,8);
    assert.equal(result.learned_ending_strength,1);
    assert.equal(result.constraint_min_probability,.05);
    assert.equal(result.event_reconsider_attempts,4);
    assert.equal(result.early_eos_penalty, 0);
    assert.equal(result.max_button_heads, 2);
    assert.equal(result.hold_protection, 'on');
    assert.equal(result.reference_check_attempts, 8);
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

test('stream training forwards visible dropout and seed including zero', () => {
  const values = {'#streamDropout':'0', '#streamSeed':'173', '#streamWeightDecay':'0.02'};
  const context = vm.createContext({valueOrDefault:(key, fallback)=>values[key] ?? fallback,
    checkedOrDefault:()=>true,
    currentTrainingMode:()=> 'fresh', trainingSourcePaths:()=>({}),
    selectedTrainingManifests:()=>[], selectedDatasetItems:()=>[], formDevice:()=> 'cpu'});
  install(context, 'chartStreamTrainingPayload', 'collectValues');
  const result=context.chartStreamTrainingPayload();
  assert.equal(result.dropout,0); assert.equal(result.seed,173);
  assert.equal(result.weight_decay,.02);
  assert.equal(result.quality_enabled,true);
  assert.equal(result.grounding_profile,'auto');
  assert.equal(result.detail_audio_nodes,8192);
  assert.equal(result.prefetch_batches,2);
  for (const id of ['streamDropout','streamSeed']) assert.ok(html.includes(`id="${id}"`));
});

test('finetune reinspection preserves explicit grounding upgrade and training settings', () => {
  const nodes={};
  for (const match of html.matchAll(/id="(stream[^"]+)"/g)) nodes['#'+match[1]]={value:'',disabled:false};
  nodes['#qualityEnabled']={checked:false};
  nodes['#streamGroundingProfile'].value='causal_audio_v1';
  nodes['#streamAlignmentLossWeight'].value='0.24';
  nodes['#streamPrefetchBatches'].value='4';
  const context=vm.createContext({element:id=>nodes[id]});
  install(context,'applyChartStreamCheckpointConfig','renderChartStreamCheckpointInspection');
  const data={model_config:{d_model:256,grounding_profile:'none'},
    args:{alignment_loss_weight:0,prefetch_batches:0,quality_enabled:true}};
  context.applyChartStreamCheckpointConfig(data,'finetune',true);
  assert.equal(nodes['#streamGroundingProfile'].value,'causal_audio_v1');
  assert.equal(nodes['#streamAlignmentLossWeight'].value,'0.24');
  assert.equal(nodes['#streamPrefetchBatches'].value,'4');
  assert.equal(nodes['#streamDModel'].value,'256');
  assert.equal(nodes['#qualityEnabled'].checked,false);
  context.applyChartStreamCheckpointConfig(data,'resume',true);
  assert.equal(nodes['#streamGroundingProfile'].value,'none');
  assert.equal(nodes['#streamGroundingProfile'].disabled,true);
});

test('stream checkpoint inspection resolves its manifest before sending the request', async () => {
  const calls=[];
  const context=vm.createContext({currentTrainingMode:()=> 'finetune',currentModelFamily:()=> 'chart_stream_v1',
    element:()=>null,assertTrainingSelection(){},assertDatasetBackend:async()=>{},
    valueOrDefault:()=> 'old.pt',trainingSourcePaths:()=>({manifest:'source.json'}),
    checkpointInspectRequest:0,currentCheckpointKey:()=> 'same',setCheckpointStatus(){},
    selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],
    api:async(url,payload)=>{calls.push(payload);return {valid:true};},
    applyChartStreamCheckpointConfig(){},renderChartStreamCheckpointInspection(){},updateTrainingControls(){}});
  const start=source.indexOf('async function inspectCheckpoint('),end=source.indexOf('async function inspectGenerationCheckpoint(',start);
  vm.runInContext(source.slice(start,end),context);
  await context.inspectCheckpoint();
  assert.equal(calls[0].manifest,'source.json');
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
