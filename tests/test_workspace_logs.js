"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const logs = require("../src/maichart/webui/static/workspace_logs.js");

test("log helpers classify severity and key events for novice filtering", () => {
  const lines = [
    "[12:00] INFO loading dataset",
    "[12:01] WARNING validation set is small",
    "[12:02] ERROR failed to save checkpoint",
    "[12:03] epoch 2 validation loss=0.42",
    'form_progress {"phase":"draft","batch":3}',
    'form_checkpoint {"path":"form_last.pt","resumable":true}',
  ];
  assert.deepEqual(logs.filterLines(lines, { severity: "error" }), [lines[2]]);
  assert.deepEqual(logs.filterLines(lines, { severity: "key" }), [lines[1], lines[2], lines[3], lines[4], lines[5]]);
  assert.deepEqual(logs.filterLines(lines, { query: "LOSS" }), [lines[3]]);
  assert.equal(logs.downloadUrl("logs/run a.log"), "/api/download?path=logs%2Frun%20a.log");
});
