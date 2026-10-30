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

test("display logs stay within UTF-8 budgets and retain the latest error", () => {
  const input = Array.from({length: 300}, (_, index) => `[${index}] composer_progress {"epoch":2,"batch":${index},"data":"${"中".repeat(9000)}"}`);
  input.push("[latest] ERROR checkpoint write failed");
  const displayed = logs.boundedDisplayLines(input);
  assert.ok(Buffer.byteLength(displayed.join("\n"), "utf8") <= 64 * 1024);
  assert.ok(displayed.every(line => Buffer.byteLength(line, "utf8") <= 2 * 1024));
  assert.match(displayed[0], /较早日志已省略/);
  assert.equal(displayed.at(-1), input.at(-1));
  assert.match(displayed.at(-2), /过长行已截断/);
  assert.deepEqual(logs.filterLines(displayed, {severity: "error"}), [input.at(-1)]);
  assert.ok(logs.filterLines(displayed, {query: "batch"}).length > 0);
  assert.equal(input[0].length > 2000, true, "original log data is unchanged");
});

test("ordinary logs and empty logs preserve their display and copy content", () => {
  const lines = ["info ready", "WARNING low disk", "ERROR save failed"];
  assert.deepEqual(logs.boundedDisplayLines(lines), lines);
  assert.deepEqual(logs.boundedDisplayLines([]), []);
  assert.equal(logs.boundedDisplayLines(lines).join("\n"), lines.join("\n"));
});

test("oversized single-line JSON is truncated without parsing its payload", () => {
  const line = 'composer_generation_validation {"status":"failed","invalid":' + "x".repeat(270698);
  const displayed = logs.boundedDisplayLines([line]);
  assert.equal(displayed.length, 1);
  assert.ok(displayed[0].startsWith('composer_generation_validation {"status":"failed"'));
  assert.match(displayed[0], /过长行已截断/);
  assert.ok(Buffer.byteLength(displayed[0], "utf8") <= 2048);
});

test("UTF-8 truncation preserves emoji boundaries", () => {
  for (const prefix of ["", "a", "ab", "abc"]) {
    const displayed = logs.boundedDisplayLines([prefix + "😀".repeat(3000)])[0];
    assert.ok(Buffer.byteLength(displayed, "utf8") <= 2048);
    assert.ok(!displayed.includes("\uFFFD"));
    const retained = displayed.split(" …［")[0].slice(prefix.length);
    assert.equal(retained, "😀".repeat([...retained].length));
  }
});
