"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const appSource = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const apiStart = appSource.indexOf("async function api(path, body) {");
const apiEnd = appSource.indexOf("\nfunction guard(handler)", apiStart);
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
  vm.runInContext(`${appSource.slice(apiStart, apiEnd)}\n${appSource.slice(taskStart, taskEnd)}\nglobalThis.harness = {api, watchTask, pollTask, taskWatchers};`, context);
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
