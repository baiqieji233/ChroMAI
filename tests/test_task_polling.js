"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const appSource = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const apiStart = appSource.indexOf("async function api(path, body) {");
const apiEnd = appSource.indexOf("\nfunction guard(handler)", apiStart);
const epochStart = appSource.indexOf("function trainingMetricNumber(values, key) {");
const epochEnd = appSource.indexOf("\n// ---------------------------------------------------------------------------", epochStart);
const taskStart = appSource.indexOf("const TASK_KIND_LABEL = {");
const taskEnd = appSource.indexOf("\nfunction renderTaskResults(task)", taskStart);
const resultsStart = appSource.indexOf("function renderTaskResults(task) {");
const resultsEnd = appSource.indexOf("\nfunction renderGenerationResult(task)", resultsStart);
assert.ok(epochStart >= 0 && epochEnd > epochStart, "training metrics and task summary helpers must exist in app.js");
assert.ok(apiStart >= 0 && apiEnd > apiStart, "api() must exist in app.js");
assert.ok(taskStart >= 0 && taskEnd > taskStart, "task polling functions must exist in app.js");
assert.ok(resultsStart >= 0 && resultsEnd > resultsStart, "task result renderer must exist in app.js");

function createHarness(fetchImpl) {
  const timers = [];
  const elements = new Map();
  const context = {
    fetch: fetchImpl,
    encodeURIComponent,
    setTimeout(callback, delay) { timers.push({callback, delay}); return timers.length; },
    window: {MaichartWorkflow: null, MaichartLogs: null},
    trainingEpochStatus: () => null,
    renderGenerationResult() {},
    renderAcceptanceFromTask() {},
    renderGoldResult() {},
    console,
  };
  context.element = selector => {
    if (!elements.has(selector)) {
      elements.set(selector, {textContent: "", hidden: false, disabled: false, style: {}, parentElement: {hidden: false},
        children: [], replaceChildren(...children) { this.children = children; },
        appendChild(child) { this.children.push(child); return child; },
        append(...children) { this.children.push(...children); },
        prepend(child) { this.children.unshift(child); return child; }});
    }
    return elements.get(selector);
  };
  context.document = {createElement: tag => ({tagName: tag, textContent: "", className: "", children: [],
    appendChild(child) { this.children.push(child); return child; },
    append(...children) { this.children.push(...children); },
    prepend(child) { this.children.unshift(child); return child; }})};
  vm.createContext(context);
  vm.runInContext(`${appSource.slice(epochStart, epochEnd)}\n${appSource.slice(apiStart, apiEnd)}\n${appSource.slice(taskStart, taskEnd)}\n${appSource.slice(resultsStart, resultsEnd)}\nglobalThis.harness = {api, watchTask, pollTask, taskWatchers, trainingEpochStatus, trainingMetricEvents, renderTaskTrainingMetrics, taskOnsetDiagnostics, renderTaskOnsetDiagnostics, renderTaskResults};`, context);
  return {context, elements, timers, harness: context.harness};
}

function jsonResponse(data, ok = true, status = 200) {
  return {ok, status, text: async () => JSON.stringify(data)};
}

function descendantText(node) {
  return [node?.textContent || "", ...(node?.children || []).map(descendantText)].join(" ");
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
  const caching = harness.trainingEpochStatus({kind: "form_train", status: "running", progress: 0,
    result: {phase: "audio_cache", progress_detail: {mode: "resident", phase: "audio",
      completed: 2, total: 4, bytes: 1024 ** 2, required_bytes: 2 * 1024 ** 3,
      budget_bytes: 3 * 1024 ** 3, unique_audio: 4, preload_loads: 3}}});
  assert.match(caching.text, /音频全量驻留/);
  assert.match(caching.text, /音频 2\/4/);
  assert.match(caching.text, /预计需求 2.00 GB/);
  assert.match(caching.text, /预加载读取 3 次/);
  assert.match(status.text, /本阶段 ETA 1分5秒/);
  assert.doesNotMatch(status.text, /全程 ETA/);
  const preparing = harness.trainingEpochStatus({kind: "form_train", status: "running", progress: 0,
    result: {phase: "prepare", progress_detail: {stage: "dataset"}}});
  assert.match(preparing.text, /准备数据 · 读取数据集/);
  const cached = harness.trainingEpochStatus({kind: "form_train", status: "running", progress: 0,
    result: {phase: "prepare", progress_detail: {stage: "chart", charts: 84, completed_songs: 67,
      total_songs: 1922, cache: "hit", preparation_cache: {hits: 84, rebuilt: 0, invalid: 0}}}});
  assert.match(cached.text, /歌曲 67\/1922/);
  assert.match(cached.text, /结构缓存：命中，直接复用/);
  assert.match(cached.text, /结构缓存累计：命中 84 \/ 重建 0 \/ 损坏 0/);
});

test("song-form task summaries keep metric detail out of the phase line", () => {
  const {harness} = createHarness(async () => jsonResponse({}));
  const progress = harness.trainingEpochStatus({kind: "form_train", status: "running", progress: 0.4,
    result: {epoch_target: 1, progress_detail: {epoch: 1, phase: "layout_revision", unit: "batches",
      completed: 2, total: 4, overall_fraction: 0.4, phase_fraction: 0.5,
      losses: {draft_note_configuration_loss: 0.42, draft_note_configuration_accuracy: 0.71},
      metric_counts: {draft_note_configuration_loss: 120, draft_note_configuration_accuracy: 120}}}});
  assert.match(progress.text, /epoch 1\/1/);
  assert.match(progress.text, /batch 2\/4/);
  assert.doesNotMatch(progress.text, /0\.4200|音符字符准确率/);
});

function renderedMetricText(groups) {
  return groups.flatMap(group => group.children).flatMap(child => child.children || [])
    .map(node => node.textContent).join(" ");
}

test("draft panel displays audio-position accuracy and coverage with their own denominators",()=>{
  const {harness,elements}=createHarness(async()=>jsonResponse({}));
  harness.renderTaskTrainingMetrics({kind:"form_train",status:"running",logs:[],result:{
    phase:"layout_revision",training_metrics:{layout_revision:{epoch:1,phase:"layout_revision",completed:3,total:10,
      losses:{position_alignment_loss:.3,position_alignment_event_accuracy:.75,position_alignment_coverage:.5},
      metric_counts:{position_alignment_loss:8,position_alignment_event_accuracy:6,position_alignment_coverage:8}}}}});
  const groups=elements.get("#taskTrainingMetricsGroups").children;
  const draft=groups.find(group=>group.className.includes("training-metric-draft"));
  const text=descendantText(draft);
  assert.match(text,/音频—拍位辅助损失/);
  assert.match(text,/音频位置匹配准确率/);
  assert.match(text,/官谱起音的音频查询覆盖率/);
  assert.match(text,/75\.0% · n=6/);
  assert.match(text,/50\.0% · n=8/);
});

test("song-form metric panel shows live draft/revision plus same-epoch planner values", () => {
  const {harness, elements} = createHarness(async () => jsonResponse({}));
  const task = {kind: "form_train", status: "running", logs: [
    `[06:37:21] form_progress ${JSON.stringify({epoch: 9, phase: "blueprint", completed: 1859, total: 1859, unit: "charts", losses: {loss: 1.31, loss_onset: 0.68}})}`,
    `[06:38:02] form_progress ${JSON.stringify({epoch: 9, phase: "layout_revision", completed: 517, total: 65392, unit: "batches", losses: {draft_note_configuration_loss: 0.62, draft_note_configuration_accuracy: 0.78, draft_position_header_accuracy: 0.99, draft_first_head_accuracy: 0.55, revision_note_configuration_loss: 0.17, revision_note_configuration_accuracy: 0.94, revision_position_header_accuracy: 0.99, revision_first_head_accuracy: 0.88}, metric_counts: {draft_note_configuration_loss: 39640, draft_note_configuration_accuracy: 39640, draft_position_header_accuracy: 38049, draft_first_head_accuracy: 8513, revision_note_configuration_loss: 36216, revision_note_configuration_accuracy: 36216, revision_position_header_accuracy: 35412, revision_first_head_accuracy: 7966}})}`,
  ], result: {phase: "checkpoint", progress_detail: {phase: "checkpoint"}, training_metrics: {
    blueprint: {epoch: 9, phase: "blueprint", completed: 1859, total: 1859, unit: "charts", losses: {loss: 1.31, loss_onset: 0.68}},
    layout_revision: {epoch: 9, phase: "layout_revision", completed: 517, total: 65392, unit: "batches", losses: {draft_note_configuration_loss: 0.62, draft_note_configuration_accuracy: 0.78, draft_position_header_accuracy: 0.99, draft_first_head_accuracy: 0.55, revision_note_configuration_loss: 0.17, revision_note_configuration_accuracy: 0.94, revision_position_header_accuracy: 0.99, revision_first_head_accuracy: 0.88}, metric_counts: {draft_note_configuration_loss: 39640, draft_note_configuration_accuracy: 39640, draft_position_header_accuracy: 38049, draft_first_head_accuracy: 8513, revision_note_configuration_loss: 36216, revision_note_configuration_accuracy: 36216, revision_position_header_accuracy: 35412, revision_first_head_accuracy: 7966}},
  }, metrics: {epoch: 8, draft_revision_loss: 0.22, whole_song_plan_loss: 1.44, draft_note_configuration_accuracy: 0.77}}};
  harness.renderTaskTrainingMetrics(task);
  const groups = elements.get("#taskTrainingMetricsGroups").children;
  const text = groups.map(descendantText).join(" ");
  assert.equal(elements.get("#taskTrainingMetrics").hidden, false);
  assert.match(elements.get("#taskTrainingMetricsSource").textContent, /最近阶段记录：谱面配置 · epoch 9/);
  assert.match(text, /草稿配置/);
  assert.match(text, /修订配置/);
  assert.match(text, /本 epoch 已完成的全曲规划/);
  assert.match(text, /0\.6200/);
  assert.match(text, /n=4万/);
  assert.match(text, /1\.3100/);
  assert.doesNotMatch(text, /0\.7700/);
});

test("song-form metric panel recovers the latest epoch when phase logs have rolled off", () => {
  const {harness, elements} = createHarness(async () => jsonResponse({}));
  harness.renderTaskTrainingMetrics({kind: "form_train", status: "running", logs: [], result: {
    phase: "checkpoint", progress_detail: {phase: "checkpoint"}, training_metrics: {},
    metric_history: [{epoch: 8, draft_revision_loss: 0.22, whole_song_plan_loss: 1.44,
      draft_note_configuration_loss: 0.65, draft_note_configuration_accuracy: 0.77,
      draft_position_header_accuracy: 0.99, draft_first_head_accuracy: 0.53,
      revision_note_configuration_loss: 0.17, revision_note_configuration_accuracy: 0.94,
      revision_position_header_accuracy: 0.99, revision_first_head_accuracy: 0.88,
      metric_counts: {draft_note_configuration_loss: 4726494, draft_note_configuration_accuracy: 4726494,
        draft_position_header_accuracy: 4415130, draft_first_head_accuracy: 997733,
        revision_note_configuration_loss: 4726494, revision_note_configuration_accuracy: 4726494,
        revision_position_header_accuracy: 4415130, revision_first_head_accuracy: 997733}}],
  }});
  assert.equal(elements.get("#taskTrainingMetrics").hidden, false);
  assert.match(elements.get("#taskTrainingMetricsSource").textContent, /最近记录：epoch 8/);
  const text = elements.get("#taskTrainingMetricsGroups").children.map(descendantText).join(" ");
  assert.match(text, /n=472\.6万/);
  assert.match(text, /0\.2200/);
});

test("song-form metric panel omits task groups with zero selected targets", () => {
  const {harness, elements} = createHarness(async () => jsonResponse({}));
  harness.renderTaskTrainingMetrics({kind: "form_train", status: "running", result: {
    training_metrics: {layout_revision: {epoch: 2, phase: "layout_revision",
      losses: {draft_note_configuration_loss: 0.4, revision_note_configuration_loss: 0.0},
      metric_counts: {draft_note_configuration_loss: 20, revision_note_configuration_loss: 0}}},
  }});
  const text = elements.get("#taskTrainingMetricsGroups").children.map(descendantText).join(" ");
  assert.match(text, /草稿配置/);
  assert.doesNotMatch(text, /修订配置/);
  assert.match(text, /0\.4000/);
  assert.doesNotMatch(text, /0\.0000/);
});

test("song-form metric events deduplicate epochs and prefer complete summaries", () => {
  const {harness} = createHarness(async () => jsonResponse({}));
  const events = harness.trainingMetricEvents({kind: "form_train", logs: [], result: {
    metric_history: [
      {epoch: 2, partial_epoch_metrics: true, draft_note_configuration_loss: 0.8},
      {epoch: 1, draft_note_configuration_loss: 0.9},
      {epoch: 2, draft_note_configuration_loss: 0.6},
    ],
  }});
  assert.deepEqual(Array.from(events.history, item => item.epoch), [1, 2]);
  assert.equal(events.history[1]?.partial_epoch_metrics, false);
  assert.equal(events.history[1].draft_note_configuration_loss, 0.6);
});

test("archived progress is labeled incomplete and partial planner loss is omitted", () => {
  const {harness, elements} = createHarness(async () => jsonResponse({}));
  harness.renderTaskTrainingMetrics({kind: "form_train", status: "archived", logs: [], result: {
    phase: "layout_revision", progress_detail: {phase: "layout_revision", epoch: 12, completed: 5, total: 10,
      unit: "batches", losses: {draft_note_configuration_loss: 0.57}, metric_counts: {draft_note_configuration_loss: 20}},
    metric_history: [{epoch: 11, draft_note_configuration_loss: 0.59},
      {epoch: 12, partial_epoch_metrics: true, whole_song_plan_loss: 0, draft_note_configuration_loss: 0.57}],
  }});
  assert.match(elements.get("#taskTrainingMetricsSource").textContent, /历史末条进度/);
  const text = elements.get("#taskTrainingMetricsGroups").children.map(descendantText).join(" ");
  assert.match(text, /本轮未完成/);
  assert.doesNotMatch(text, /Planner 损失/);
});

test("a later partial duplicate cannot fill missing metrics in a complete epoch",()=>{
  const {harness}=createHarness(async()=>jsonResponse({}));
  const events=harness.trainingMetricEvents({kind:"form_train",result:{metric_history:[
    {epoch:4,partial_epoch_metrics:false,draft_note_configuration_loss:.6},
    {epoch:4,partial_epoch_metrics:true,draft_note_configuration_loss:.5,whole_song_plan_loss:0},
  ]}});
  assert.equal(events.epoch.draft_note_configuration_loss,.6);
  assert.equal(events.epoch.whole_song_plan_loss,undefined);
  assert.equal(events.history[0].partial_epoch_metrics,false);
});

test("completed archived epoch is not labeled as an unfinished current epoch",()=>{
  const {harness,elements}=createHarness(async()=>jsonResponse({}));
  harness.renderTaskTrainingMetrics({kind:"form_train",status:"archived",result:{
    phase:"layout_revision",progress_detail:{phase:"layout_revision",epoch:11,losses:{draft_note_configuration_loss:.6}},
    metric_history:[{epoch:11,partial_epoch_metrics:false,draft_note_configuration_loss:.59}],
  }});
  assert.doesNotMatch(elements.get("#taskTrainingMetricsSource").textContent,/本轮未完成/);
  assert.match(elements.get("#taskTrainingMetricsSource").textContent,/epoch 11/);
});

test("song-form onset diagnostics render from payload or retained log", () => {
  const {harness, elements} = createHarness(async () => jsonResponse({}));
  const diagnostics = {schema: "song_form_onset_diagnostics_v1", gap_threshold_sec: 2,
    gap_count: 1, totals: {raw_local_peaks: 120, audible_peaks: 30},
    gaps: [{start_sec: 10, end_sec: 18.8, duration_sec: 8.8, raw_peak_count: 0,
      audible_peak_count: 0, classification: "low_audio_activity"}]};
  harness.renderTaskResults({kind: "form_generate", result: {}, logs: [
    `[12:00:00] form_onset_diagnostics ${JSON.stringify(diagnostics)}`,
  ]});
  const notice = elements.get("#taskResults").children[0];
  assert.match(descendantText(notice), /起音空白诊断/);
  assert.match(descendantText(notice), /10\.0–18\.8 秒/);
  assert.match(descendantText(notice), /音频活动较低/);
  assert.equal(harness.taskOnsetDiagnostics({kind: "form_generate", result: {
    onset_summary: {selection: {onset_diagnostics: diagnostics}}, logs: []}}).schema,
  "song_form_onset_diagnostics_v1");
});

test("song-form repair results surface remaining hard conflicts without claiming playability", () => {
  const {harness, elements} = createHarness(async () => jsonResponse({}));
  harness.renderTaskResults({kind: "form_generate", result: {repair_summary: {
    remaining_actionable_hard: 19, status: "partial_budget_exhausted"}}});
  const notice = elements.get("#taskResults").children[0];
  assert.match(notice.textContent, /剩余可处理硬冲突：19 项/);
  assert.match(notice.textContent, /不构成可玩性认证/);
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
