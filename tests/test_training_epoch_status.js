"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const appSource = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
const functionStart = appSource.indexOf("function trainingEpochStatus(task) {");
const functionEnd = appSource.indexOf("// ---------------------------------------------------------------------------", functionStart);
assert.ok(functionStart >= 0 && functionEnd > functionStart, "epoch formatter must exist in app.js");
const trainingEpochStatus = new Function(`${appSource.slice(functionStart, functionEnd)}; return trainingEpochStatus;`)();

test("resume uses the absolute target epoch and additional epoch count", () => {
  const result = trainingEpochStatus({
    kind: "train",
    status: "running",
    result: {
      epoch_start: 4,
      epochs_requested: 11,
      epoch_target: 15,
      epoch_semantics: "additional_from_checkpoint",
      last_epoch: 5,
    },
  });
  assert.equal(result.knownTarget, true);
  assert.equal(result.progress, 5 / 15);
  assert.match(result.text, /epoch 5 \/ 15/);
  assert.match(result.text, /追加 11 轮/);
});

test("completed early-stopped training fills the task bar but keeps actual epoch count", () => {
  const result = trainingEpochStatus({
    kind: "train", status: "completed",
    result: { epoch_start: 4, epochs_requested: 11, epoch_target: 15,
      epoch_semantics: "additional_from_checkpoint", last_epoch: 7 },
  });
  assert.equal(result.progress, 1);
  assert.match(result.text, /epoch 7 \/ 15/);
});

test("legacy running task with no target never claims a percentage or zero epochs", () => {
  const result = trainingEpochStatus({
    kind: "train", status: "running", result: { last_epoch: 5 },
  });
  assert.equal(result.knownTarget, false);
  assert.equal(result.progress, null);
  assert.match(result.text, /已完成 epoch 5/);
  assert.match(result.text, /目标未知，请查看任务日志/);
  assert.doesNotMatch(result.text, /0 轮/);
});

test("legacy running task before its first epoch still reports an unknown target", () => {
  const result = trainingEpochStatus({ kind: "train", status: "running", result: {} });
  assert.equal(result.knownTarget, false);
  assert.match(result.text, /目标未知/);
});

test("song form separates live tensors, idle cache and device headroom", () => {
  const result = trainingEpochStatus({kind: "form_train", status: "running", progress: .1,
    result: {epoch_target: 30, phase: "layout_revision", progress_detail: {
      epoch: 1, phase: "layout_revision", completed: 4000, total: 14069,
      memory: {allocated_gib: .4, reserved_gib: 6.5, peak_allocated_gib: 6.,
        inactive_cached_gib: 6.1, device_free_gib: 3.8},
      cuda_cache_guard: {enabled: true, reserved_high_water_gib: 7.2, trim_count: 2},
    }}});
  assert.match(result.text, /闲置缓存 6\.10 GiB/);
  assert.match(result.text, /设备可用 3\.80 GiB/);
  assert.match(result.text, /显存缓存保护：阈值 7\.20 GiB · 已回收 2 次/);
});
