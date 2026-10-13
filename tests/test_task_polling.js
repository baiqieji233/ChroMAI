"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const appSource = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const apiStart = appSource.indexOf("async function api(path, body) {");
const apiEnd = appSource.indexOf("\nfunction guard(handler)", apiStart);
const epochStart = appSource.indexOf("function trainingEpochStatus(task) {");
const epochEnd = appSource.indexOf("\n// ---------------------------------------------------------------------------", epochStart);
const taskStart = appSource.indexOf("const TASK_KIND_LABEL = {");
const taskEnd = appSource.indexOf("\nfunction renderTaskResults(task)", taskStart);
assert.ok(apiStart >= 0 && apiEnd > apiStart, "api() must exist in app.js");
assert.ok(taskStart >= 0 && taskEnd > taskStart, "task polling functions must exist in app.js");

function createHarness(fetchImpl) {
  const timers = [];
  const elements = new Map();
  const context = {
    fetch: fetchImpl,
    encodeURIComponent,
    setTimeout(callback, delay) { timers.push({callback, delay}); return timers.length; },
    window: {MaichartWorkflow: null, MaichartLogs: null},
    trainingEpochStatus: () => null,
    renderTaskResults() {},
    renderGenerationResult() {},
    renderAcceptanceFromTask() {},
    renderGoldResult() {},
    console,
  };
  context.element = selector => {
    if (!elements.has(selector)) {
      elements.set(selector, {textContent: "", hidden: false, disabled: false, style: {}, parentElement: {hidden: false}});
    }
    return elements.get(selector);
  };
  vm.createContext(context);
  vm.runInContext(`${appSource.slice(epochStart, epochEnd)}\n${appSource.slice(apiStart, apiEnd)}\n${appSource.slice(taskStart, taskEnd)}\nglobalThis.harness = {api, watchTask, pollTask, taskWatchers, trainingEpochStatus};`, context);
  return {context, elements, timers, harness: context.harness};
}

function jsonResponse(data, ok = true, status = 200) {
  return {ok, status, text: async () => JSON.stringify(data)};
}

test("failed task renders its error and logs, stops polling, and releases callbacks", async () => {
  const task = {task_id: "t1", kind: "form_generate", status: "failed", error: "子进程退出码 1", logs: ["生成开始", "Traceback: bad BPM"]};
  const {harness, elements, timers} = createHarness(async () => jsonResponse(task));
  let callbackCount = 0;
  harness.watchTask("t1", async result => { callbackCount += 1; assert.equal(result.task_id, task.task_id); });
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(elements.get("#taskMessage").textContent, task.error);
  assert.equal(elements.get("#taskLogs").textContent, task.logs.join("\n"));
  assert.equal(harness.taskWatchers.has("t1"), false);
  assert.equal(callbackCount, 1);
  assert.equal(timers.length, 0);
});

test("queued and running tasks keep polling at the normal interval", async () => {
  for (const status of ["queued", "running"]) {
    const {harness, timers} = createHarness(async () => jsonResponse({task_id: "t2", kind: "form_generate", status}));
    harness.watchTask("t2");
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(harness.taskWatchers.has("t2"), true);
    assert.deepEqual(timers.map(timer => timer.delay), [800]);
  }
});

test("song-form task summaries distinguish batches from optimizer steps and report phase ETA", () => {
  const {harness} = createHarness(async () => jsonResponse({}));
  const status = harness.trainingEpochStatus({kind: "form_train", status: "running", progress: 0.2,
    result: {epoch_target: 4, progress_detail: {epoch: 2, phase: "layout_revision", unit: "batches",
      completed: 3, total: 10, overall_fraction: 0.3, phase_fraction: 0.3,
      optimizer_steps: 2, blueprint_steps: 4, eta_phase_sec: 65,
      items_per_sec: 0.5, samples_per_sec: 1.5, tokens_per_sec: 120,
      accumulation_pending: 1, grad_accumulation: 4, lr: 0.0002,
      elapsed_sec: 100, phase_elapsed_sec: 65,
      losses: {loss: 0.25}, precision: "bf16",
      timing: {data_wait_sec: 1, h2d_sec: 2, compute_sec: 50, checkpoint_sec: 3},
      memory: {allocated_gib: 2, reserved_gib: 3, peak_allocated_gib: 4},
      cache_scope: "main_process_only",
      cache: {audio: {hits: 8, misses: 2, evictions: 1, entries: 4, bytes: 1024, budget_bytes: 2048},
        samples: {hits: 5, misses: 3, evictions: 0, entries: 2, bytes: 512, budget_bytes: 1024}}}}});
  assert.match(status.text, /epoch 2\/4/);
  assert.match(status.text, /本轮任务数进度（规划谱面数＋配置 batch 数）30%/);
  assert.match(status.text, /batch 3\/10/);
  assert.match(status.text, /参数更新 2 步/);
  assert.match(status.text, /累计计时（非 GPU 计时）/);
  assert.match(status.text, /主进程本地，worker 缓存未汇总/);
  assert.match(status.text, /CUDA 显存：/);
  assert.match(status.text, /累积 batch 1\/4/);
  assert.match(status.text, /本阶段 ETA 1分5秒/);
  assert.doesNotMatch(status.text, /全程 ETA/);
  const preparing = harness.trainingEpochStatus({kind: "form_train", status: "running", progress: 0,
    result: {phase: "prepare", progress_detail: {stage: "dataset"}}});
  assert.match(preparing.text, /准备数据 · dataset/);
});

test("network errors retain the watcher and schedule reconnect", async () => {
  const {harness, elements, timers} = createHarness(async () => { throw new Error("offline"); });
  harness.watchTask("t3");
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(harness.taskWatchers.has("t3"), true);
  assert.match(elements.get("#taskMessage").textContent, /正在重连：offline/);
  assert.deepEqual(timers.map(timer => timer.delay), [3000]);
});

test("successful task reads may contain error, while failed POST responses still throw", async () => {
  const task = {task_id: "t4", status: "failed", error: "子进程退出码 1"};
  const {harness} = createHarness(async () => jsonResponse(task));
  assert.equal(JSON.stringify(await harness.api("/api/task?id=t4")), JSON.stringify(task));

  const postHarness = createHarness(async () => jsonResponse({error: "invalid request"}));
  await assert.rejects(postHarness.harness.api("/api/form/generate", {audio: "song.mp3"}), /invalid request/);
});

test("HTTP and application errors without a task record still reject GET task reads", async () => {
  for (const [data, ok, status] of [
    [{error: "task missing"}, false, 404],
    [{error: "backend error"}, true, 200],
    [{task_id: "t5", status: "failed", error: "service unavailable"}, false, 503],
  ]) {
    const {harness} = createHarness(async () => jsonResponse(data, ok, status));
    await assert.rejects(harness.api("/api/task?id=t5"), new RegExp(data.error));
  }
});
