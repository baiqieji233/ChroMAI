"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");

function functionSource(start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `expected ${start} source`);
  return source.slice(from, to);
}

function installGeometryControls(context) {
  context.checkpointInspection = {data:null};
  context.currentTrainingMode ||= () => context.mode || 'fresh';
  vm.runInContext(functionSource('function songFormRhythmGeometryUiState(',
    'function songFormTrainingPayload('),context);
}

test("migration controls are visible only for song-form resume", () => {
  const group = {hidden: true};
  const context = {
    updateEventSetControls() {}, updateFamilyFields() {}, updateDatasetButtons() {}, updateTrainingModeHelp() {},
    currentTrainingMode: () => "resume", currentModelFamily: () => "song_form_v1",
    element: selector => selector === "#formResumeDataPolicyGroup" ? group : null,
    window: {},
  };
  vm.createContext(context);
  installGeometryControls(context);
  const begin = source.indexOf("function updateTrainingControls() {");
  const end = source.indexOf("// ---------------------------------------------------------------------------", begin);
  vm.runInContext(source.slice(begin, end), context);
  context.updateTrainingControls();
  assert.equal(group.hidden, false);
  context.currentTrainingMode = () => "fresh";
  context.updateTrainingControls();
  assert.equal(group.hidden, true);
  context.currentTrainingMode = () => "resume";
  context.currentModelFamily = () => "event_ctx_v1";
  context.updateTrainingControls();
  assert.equal(group.hidden, true);
});

test("song-form resume restores and locks inherited settings while finetune keeps them editable", () => {
  const elements = new Map();
  const context = {currentModelFamily: () => "song_form_v2", element(selector) {
    if (!elements.has(selector)) elements.set(selector, {value: "manual", checked: false, disabled: false, type: selector === "#qualityEnabled" ? "checkbox" : "number"});
    return elements.get(selector);
  }};
  vm.createContext(context);
  installGeometryControls(context);
  vm.runInContext(`${functionSource("function applySongFormCheckpointConfig(data, mode) {", "async function inspectCheckpoint(")}\nglobalThis.apply = applySongFormCheckpointConfig;`, context);
  const settings = {lr: 0.0003, batch_size: 2, grad_accumulation: 4, seed: 7, weight_decay: 0.02,
    condition_drop_prob: 0.1, blueprint_loss_weight: 1.2, load_curve_loss_weight: 0.3,
    relation_loss_weight: 0.2, quality_enabled: true, max_steps: 12, limit_charts: 16,
    precision: "bf16", max_region_seconds: 12, blueprint_batch_size: 4,
    perturb_curve_noise_std: .03, perturb_apply_prob: .5, diversity_loss_weight: .02};
  context.apply({valid: true, resume_settings: settings,
    model_config: {d_model: 128, layers: 3, nhead: 4, max_tokens: 1024, max_memory_tokens: 256, dropout: 0.2},
    epoch: 5, blueprint_steps: 10, optimizer_steps: 20, can_generate: true, can_revise: false}, "resume");
  for (const selector of ["#learningRate", "#batchSize", "#gradAccumulation", "#trainingSeed", "#weightDecay",
    "#formConditionDropout", "#formBlueprintLossWeight", "#formCurveLossWeight", "#formRelationLossWeight",
    "#qualityEnabled", "#maxSteps", "#formLimitCharts", "#formPrecision", "#formMaxRegionSeconds", "#formBlueprintBatchSize"]) {
    assert.equal(elements.get(selector).disabled, true, `${selector} should be locked for resume`);
  }
  assert.equal(elements.get("#formLimitCharts").value, "16");
  assert.equal(elements.get("#qualityEnabled").checked, true);
  assert.equal(elements.get("#formDModel").disabled, true);
  assert.equal(elements.get("#formDropout").value, "0.2");
  assert.equal(elements.get("#formBlueprintBatchSize").value, "4");
  assert.equal(elements.get("#formPerturbCurveNoiseStd").value, "0.03");
  assert.equal(elements.get("#formPerturbCurveNoiseStd").disabled, true);
  assert.equal(elements.get("#formPerturbHistoryCorruptionProb").value, "0");
  assert.equal(elements.get("#formPerturbHistoryCorruptionProb").disabled, true);
  assert.equal(elements.get("#formDiversityLossWeight").value, "0.02");
  assert.equal(elements.get("#formQualityApplyRecipe").disabled, true);
  assert.match(elements.get("#checkpointInspectStatus").textContent, /limit_charts=16/);

  context.apply({valid: true, model_config: {d_model: 256, layers: 4, nhead: 8}}, "finetune");
  assert.equal(elements.get("#formDModel").value, "256");
  assert.equal(elements.get("#formDModel").disabled, true);
  assert.equal(elements.get("#learningRate").disabled, false);
  assert.equal(elements.get("#formLimitCharts").disabled, false);
  assert.equal(elements.get("#formBlueprintBatchSize").disabled, false);
  assert.equal(elements.get("#formDiversityLossWeight").disabled, false);
  assert.equal(elements.get("#formRepetitionPenaltyWeight").disabled, true);
  assert.equal(elements.get("#formPerturbOnsetDropProb").disabled, true);
});

test("song-form output nonce is a directory component and preserves the requested filename", () => {
  const context = {};
  vm.createContext(context);
  vm.runInContext(`${functionSource("function uniqueSongFormOutputPath(requested) {", "function generationPayload() {")}\nglobalThis.unique = uniqueSongFormOutputPath;`, context);
  const first = context.unique("webui_workspace/generated/maidata.txt");
  const second = context.unique("webui_workspace/generated/maidata.txt");
  assert.match(first, /^webui_workspace\/generated\/\.task-[^/]+\/maidata\.txt$/);
  assert.notEqual(first, second);
  assert.match(context.unique("D:\\charts\\maidata.txt"), /^D:\\charts\\\.task-[^\\]+\\maidata\.txt$/);
});

test("rational resume without saved planner batch displays the trainer fallback", () => {
  const elements = new Map();
  const context = {currentModelFamily: () => "song_form_v2", element(selector) {
    if (!elements.has(selector)) elements.set(selector, {value: "", disabled: false, type: "number"});
    return elements.get(selector);
  }};
  vm.createContext(context);
  installGeometryControls(context);
  vm.runInContext(`${functionSource("function applySongFormCheckpointConfig(data, mode) {", "async function inspectCheckpoint(")}\nglobalThis.apply = applySongFormCheckpointConfig;`, context);
  context.apply({valid: true, model_family: "song_form_v2", args: {}}, "resume");
  assert.equal(elements.get("#formBlueprintBatchSize").value, "1");
  assert.equal(elements.get("#formBlueprintBatchSize").disabled, true);
  assert.equal(elements.get("#formEventDecisionLossWeight").value, "0");
  assert.equal(elements.get("#formEventDecisionLossWeight").disabled, true);
  context.apply({valid: true, model_family: "song_form_v2", args: {}}, "finetune");
  assert.equal(elements.get("#formBlueprintBatchSize").disabled, false);
  assert.equal(elements.get("#formEventDecisionLossWeight").disabled, false);
});

test("completed song-form task retries artifact collection at most three times", async () => {
  const timers = [];
  const taskWatchers = new Map();
  let reads = 0;
  let finished;
  const context = {
    activeTaskId: "task-1",
    taskWatchers,
    api: async () => { reads += 1; return {task_id: "task-1", kind: "form_generate", status: "completed", result: {}}; },
    window: {MaichartWorkflow: null},
    renderTask() {},
    setTimeout(callback, delay) { timers.push({callback, delay}); },
  };
  vm.createContext(context);
  const pollSource = functionSource("async function pollTask(taskId) {", "function hasTaskArtifacts(task)")
    + functionSource("function hasTaskArtifacts(task) {", "function renderTask(task) {");
  vm.runInContext(`${pollSource}\nglobalThis.pollTask = pollTask;`, context);
  taskWatchers.set("task-1", {callbacks: new Set([task => { finished = task; }]), task: null});
  await context.pollTask("task-1");
  for (let index = 0; timers.length; index += 1) {
    assert.ok(index < 4, "artifact polling must terminate");
    timers.shift().callback();
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.equal(reads, 4);
  assert.equal(finished.artifacts_missing, true);
  assert.equal(taskWatchers.has("task-1"), false);
});

test("repair summary calls out exhausted budget and partial coverage when reported", () => {
  const container = {children: [], replaceChildren(...children) { this.children = children; }, appendChild(child) { this.children.push(child); }};
  const context = {
    element: () => container,
    document: {createElement: () => ({textContent: "", className: "", children: [], appendChild(child) { this.children.push(child); }})},
    renderTaskOnsetDiagnostics() {},
  };
  vm.createContext(context);
  vm.runInContext(`${functionSource("function composerGenerationSummary(task) {", "function renderGenerationResult(task) {")}\nglobalThis.render = renderTaskResults;`, context);
  context.render({kind: "form_generate", result: {repair_summary: {remaining_actionable_hard: 3,
    budget_exhausted: true, coverage_status: "partial"}}});
  assert.match(container.children[0].textContent, /修订预算已耗尽/);
  assert.match(container.children[0].textContent, /部分语法或路径未覆盖/);
  context.render({kind: "form_generate", result: {repair_summary: {remaining_actionable_hard: 3}}});
  assert.doesNotMatch(container.children[0].textContent, /预算已耗尽|未覆盖/);
});

test("local load notice preserves separate peak units and reads existing worker logs", () => {
  const container = {children: [], replaceChildren(...children) { this.children = children; }, appendChild(child) { this.children.push(child); }};
  const context = {
    element: () => container,
    document: {createElement: () => ({textContent: "", className: "", children: [], appendChild(child) { this.children.push(child); }})},
    renderTaskOnsetDiagnostics() {},
  };
  vm.createContext(context);
  vm.runInContext(`${functionSource("function composerGenerationSummary(task) {", "function renderGenerationResult(task) {")}\nglobalThis.render = renderTaskResults;`, context);
  context.render({kind:"form_generate", result:{}, logs:[
    '[12:00:00] form_local_load {"peak_all_head_nps":20,"peak_occupied_events_nps":14,"burst_interval_count":1,"burst_intervals":[]}'
  ]});
  assert.match(container.children[0].textContent, /20 个音符头 \/ 14 个起音/);
  assert.match(container.children[0].textContent, /集中段 1 处/);
});
