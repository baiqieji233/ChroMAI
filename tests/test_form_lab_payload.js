"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  buildAnalysisPayload,
  buildGenerationPayload,
  mapRelationsToRegions,
  buildRevisionPayload,
  buildTrainingPayload,
  resolvePreparedDataset,
} = require("../src/maichart/webui/static/form_lab.js");

test("analysis payload keeps unknown meter explicit and leaves Simai grid inference out", () => {
  const payload = buildAnalysisPayload({audio: "song.mp3", bpm: "123.5", first: "-0.25", max_region_seconds: "14"});
  assert.deepEqual(payload, {
    audio: "song.mp3", bpm: 123.5, first: -0.25, meter_map: null, tempo_map: null, max_region_seconds: 14,
  });
});

test("analysis payload serializes explicit compound meter and ordered meter changes", () => {
  const payload = buildAnalysisPayload({
    audio: "song.mp3", bpm: "120", first: "0", meter_known: true,
    meter_numerator: "6", meter_denominator: "8", meter_grouping: "3,3",
  }, [{beat: "32/3", numerator: "7", denominator: "8", grouping: "2,2,3"}]);

  assert.deepEqual(payload.meter_map.map(({beat, numerator, denominator, grouping}) =>
    ({beat, numerator, denominator, grouping})), [
    {beat: "0/1", numerator: 6, denominator: 8, grouping: [3, 3]},
    {beat: "32/3", numerator: 7, denominator: 8, grouping: [2, 2, 3]},
  ]);
  assert.throws(() => buildAnalysisPayload({
    audio: "song.mp3", bpm: 120, meter_known: true,
    meter_numerator: 6, meter_denominator: 8, meter_grouping: "2,2",
  }), /分组之和/);
  assert.throws(() => buildAnalysisPayload({
    audio: "song.mp3", bpm: 120, meter_known: true,
    meter_numerator: 4, meter_denominator: 4,
  }, [{beat: "8", numerator: "3", denominator: "4"}, {beat: "6", numerator: "7", denominator: "8"}]), /严格递增/);
  const clock = buildAnalysisPayload({
    audio: "song.mp3", bpm: "120", first: "-0.2", tempo_map_json: '[{"beat":"0","bpm":120},{"beat":"32/3","bpm":150}]',
  });
  assert.deepEqual(clock.tempo_map, [{beat: "0/1", bpm: 120}, {beat: "32/3", bpm: 150}]);
  assert.throws(() => buildAnalysisPayload({
    audio: "song.mp3", bpm: 120, tempo_map_json: '[{"beat":"1","bpm":120}]',
  }), /必须从第 0 拍开始/);
});

test("training requires output paths and validates the separate model configuration", () => {
  assert.throws(() => buildTrainingPayload({manifest: "train.json"}), /输出目录不能为空/);
  assert.throws(() => buildTrainingPayload({manifest: "train.json", checkpoint_dir: "run", mode: "resume"}), /初始化 checkpoint/);
  assert.throws(() => buildTrainingPayload({manifest: "train.json", checkpoint_dir: "run", mode: "finetune"}), /初始化 checkpoint/);
  assert.throws(() => buildTrainingPayload({manifest: "train.json", checkpoint_dir: "run", d_model: "250", nhead: "8"}), /整除/);
  const payload = buildTrainingPayload({
    manifest: "merged.json", checkpoint_dir: "run", mode: "resume", init_checkpoint: "base.pt",
    epochs: "5", batch_size: "2", d_model: "256", layers: "4", nhead: "8",
    max_tokens: "2048", memory_tokens: "512",
  });
  assert.equal(payload.mode, "resume");
  assert.equal(payload.init_checkpoint, "base.pt");
  assert.equal(payload.max_memory_tokens, 512);
  assert.equal(payload.meter_map, undefined);
  const experimental = buildTrainingPayload({
    manifest: "merged.json", checkpoint_dir: "run", mode: "finetune", init_checkpoint: "base.pt",
    device: "cuda", max_steps: "300", limit_charts: "8", use_analysis_meter: true,
  }, [{beat: "0", numerator: 3, denominator: 4, grouping: [3]}]);
  assert.equal(experimental.mode, "finetune");
  assert.equal(experimental.max_steps, 300);
  assert.equal(experimental.limit_charts, 8);
  assert.equal(experimental.device, "cuda");
  assert.equal(experimental.meter_map[0].numerator, 3);
});

test("generation carries bounded revision controls and memory strength", () => {
  const payload = buildGenerationPayload({
    checkpoint: "run/event_best.pt", audio: "song.mp3", output: "out.txt", level: "13.3",
    memory_strength: "0", onset_threshold: "0.4", repair: true,
    max_checks: "24", max_edits: "4", deadline_seconds: "10",
  }, {
    bpm: "128", first: "-0.1", meter_known: true, meter_numerator: 6,
    meter_denominator: 8, meter_grouping: "3,3",
    tempo_map_json: '[{"beat":"0","bpm":128},{"beat":"24","bpm":150}]',
  });
  assert.equal(payload.level, 13.3);
  assert.equal(payload.checkpoint, "run/event_best.pt");
  assert.equal(payload.checkpoint_dir, undefined);
  assert.equal(payload.bpm, 128);
  assert.equal(payload.first, -0.1);
  assert.equal(payload.meter_map[0].numerator, 6);
  assert.deepEqual(payload.tempo_map, [{beat: "0/1", bpm: 128}, {beat: "24/1", bpm: 150}]);
  assert.equal(payload.memory_strength, 0);
  assert.equal(payload.repair, true);
  assert.equal(payload.max_checks, 24);
  assert.equal(payload.max_edits, 4);
  assert.equal(payload.deadline_seconds, 10);
  assert.throws(() => buildGenerationPayload({
    checkpoint: "run/event_best.pt", audio: "song.mp3", output: "out.txt", level: "13.3", onset_threshold: "1",
  }), /起音贴合阈值/);
  assert.throws(() => buildGenerationPayload({
    checkpoint: "run", audio: "song.mp3", output: "out.txt", level: "13.3",
  }), /\.pt/);
});

test("revision keeps exact rational edit boundaries and validates their order", () => {
  const payload = buildRevisionPayload({
    checkpoint: "run/event_best.pt", audio: "song.mp3", maidata_path: "maidata.txt", output: "out.txt",
    difficulty_index: "5", editable_start: "32/3", editable_end: "40/3", memory_strength: "0.7",
  });
  assert.equal(payload.checkpoint, "run/event_best.pt");
  assert.equal(payload.output, "out.txt");
  assert.equal(payload.editable_start, "32/3");
  assert.equal(payload.editable_end, "40/3");
  assert.equal(payload.memory_strength, 0.7);
  assert.throws(() => buildRevisionPayload({
    checkpoint: "run/event_best.pt", audio: "song.mp3", maidata_path: "maidata.txt", output: "out.txt",
    difficulty_index: "5", editable_start: "4", editable_end: "4",
  }), /晚于开始拍/);
  assert.throws(() => buildRevisionPayload({
    checkpoint: "run/event_best.pt", audio: "song.mp3", maidata_path: "maidata.txt", output: "out.txt",
    difficulty_index: "5", editable_start: "4.5", editable_end: "8",
  }), /有理数/);
  assert.throws(() => buildRevisionPayload({
    checkpoint: "run/event_best.pt", audio: "song.mp3", maidata_path: "maidata.txt",
    difficulty_index: "5", editable_start: "4", editable_end: "8",
  }), /修订输出路径/);
});

test("analysis recurrence links resolve musical region ids through region spans", () => {
  const report = {
    spans: [
      {index: 0, start: "0", end: "8"},
      {index: 1, start: "8", end: "16"},
      {index: 2, start: "16", end: "20"},
      {index: 3, start: "20", end: "28"},
    ],
    segmentation: {
      span_region_indices: [0, 1, 1, 2],
      region_spans: [
        {index: 0, start: "0", end: "8"},
        {index: 1, start: "8", end: "20"},
        {index: 2, start: "20", end: "28"},
      ],
    },
    recurrence: {links: [{source: 1, target: 2, similarity: 0.92}]},
  };
  const [link] = mapRelationsToRegions(report);
  assert.equal(link.sourceRegion.start, "8");
  assert.equal(link.sourceRegion.end, "20");
  assert.equal(link.targetRegion.start, "20");
  assert.equal(link.targetRegion.end, "28");
});

test("saved training source restores only when one prepared catalog entry matches exactly", () => {
  const prepared = [
    {name: "Full Set", path: "D:/data/full", manifest: "D:/data/full/training_manifest.json", prepared: true},
    {name: "Raw Set", path: "D:/data/raw", manifest: "D:/data/raw/training_manifest.json", prepared: false},
  ];
  const savedSingle = {
    manifest: "D:/data/full/training_manifest.json",
    datasets: ["D:/data/full"],
  };
  assert.equal(resolvePreparedDataset(prepared, savedSingle), prepared[0]);
  assert.equal(resolvePreparedDataset(prepared, {
    ...savedSingle,
    datasets: ["D:/data/full", "D:/data/other"],
  }), null);
  assert.equal(resolvePreparedDataset(prepared, {
    manifest: "D:/data/raw/training_manifest.json", datasets: ["D:/data/raw"],
  }), null);
  assert.equal(resolvePreparedDataset(prepared, {
    manifest: "D:/data/other/training_manifest.json", datasets: ["D:/data/other"],
  }), null);
});
