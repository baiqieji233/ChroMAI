"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");

test("training preflight rejects duplicate clicks and releases controls on errors", async () => {
  const begin = source.indexOf("async function startTrain(");
  const end = source.indexOf("async function submitTrain(", begin);
  let rejectSubmission;
  let submissions = 0;
  const start = new Function("submitTrain", "updateDatasetButtons", "let trainingStarting=false,trainingPreflight=false; const liveComposerTrainingTasks=new Set();" + source.slice(begin,end) + ";return startTrain;")(
    () => { submissions++; return new Promise((_resolve,reject) => { rejectSubmission=reject; }); }, () => {},
  );
  const first = start();
  await start();
  assert.equal(submissions,1);
  rejectSubmission(new Error("inspection rejected"));
  await assert.rejects(first,/inspection rejected/);
  const retry = start();
  assert.equal(submissions,2);
  rejectSubmission(new Error("inspection rejected"));
  await assert.rejects(retry,/inspection rejected/);
});

function functionSource(name, nextName) {
  const begin = source.indexOf(`function ${name}(`);
  const end = source.indexOf(`function ${nextName}(`, begin);
  assert.ok(begin >= 0 && end > begin, `${name} must exist before ${nextName}`);
  return source.slice(begin, end);
}

function makeElement(values) {
  return selector => values[selector] || null;
}

test("training mode changes update only untouched learning-rate defaults", () => {
  const learningRate = {value: "0.0007", dataset: {touched: "1"}};
  const pooling = {value: "per_sample_v2"};
  const actionState = {value: "sustain_state_v1"};
  const elements = {
    "#learningRate": learningRate,
    "#localPoolingPolicy": pooling,
    "#actionStateProfile": actionState,
  };
  let mode = "finetune";
  const setDefaults = new Function("element", "currentTrainingMode",
    functionSource("setTrainingModeDefaults", "updateDatasetButtons") + "; return setTrainingModeDefaults;")(
      makeElement(elements), () => mode,
    );

  setDefaults();
  assert.equal(learningRate.value, "0.0007");
  assert.equal(pooling.value, "per_sample_v2");
  assert.equal(actionState.value, "sustain_state_v1");

  learningRate.dataset.touched = "";
  mode = "finetune";
  setDefaults();
  assert.equal(learningRate.value, "0.00002");
  mode = "resume";
  setDefaults();
  assert.equal(learningRate.value, "0.0002");
});

test("song-form resume help describes selected migration without changing learning rate", () => {
  const values = {
    "#modelFamily": {value:"song_form_v1"},
    "#formResumeDataPolicy": {value:"restart_epoch"},
    "#trainingModeHelp": {textContent:""}, "#epochsHelp": {textContent:""},
    "#learningRate": {value:"0.00007"},
  };
  const update = new Function("element", "currentTrainingMode", "currentModelFamily",
    functionSource("updateTrainingModeHelp", "updateDatasetButtons") + "; return updateTrainingModeHelp;")(
      makeElement(values), () => "resume", () => values['#modelFamily'].value);
  update();
  assert.match(values["#trainingModeHelp"].textContent,/迁移续训/);
  assert.match(values["#epochsHelp"].textContent,/重跑未完成轮次/);
  assert.equal(values["#learningRate"].value,"0.00007");
  values["#formResumeDataPolicy"].value="strict";
  update();
  assert.match(values["#trainingModeHelp"].textContent,/严格从断点/);
});

test("checkpoint inspection preserves a manual finetune context-token limit but resume adopts the contract", () => {
  const contextTokens = {value: "512", dataset: {touched: "1"}};
  const dModel = {value: "256"};
  const elements = {"#contextTokens": contextTokens, "#dModel": dModel};
  let mode = "finetune";
  let saves = 0;
  const apply = new Function("element", "currentTrainingMode", "saveSettings",
    functionSource("applyCheckpointModelConfig", "renderCheckpointInspection") + "; return applyCheckpointModelConfig;")(
      makeElement(elements), () => mode, () => { saves += 1; },
    );
  const checkpoint = {
    model_config: {d_model: 320},
    data_contract: {family_params: {context_tokens: 320}},
  };

  apply(checkpoint);
  assert.equal(dModel.value, "320", "checkpoint model shape is inherited");
  assert.equal(contextTokens.value, "512", "explicit finetune budget is retained");
  assert.equal(saves, 1);

  mode = "resume";
  apply(checkpoint);
  assert.equal(contextTokens.value, "320", "resume follows the checkpoint contract");
  assert.equal(saves, 2);
});

test("unsupported onset guidance disables the control without erasing its saved value", () => {
  const onset = {value: "0.35", disabled: false};
  const onsetValue = {textContent: ""};
  const onsetStatus = {textContent: ""};
  const elements = {
    "#localOnsetBiasStrength": onset,
    "#localOnsetBiasStrengthValue": onsetValue,
    "#localOnsetGuidanceStatus": onsetStatus,
  };
  const buttons = [];
  const update = new Function("element", "document", "generationInspection", "valueOrDefault",
    "checkedOrDefault", "CONFLICT_KIND_FIELDS",
    functionSource("updateGenerationCapabilityControls", "familyFieldVisible") + "; return updateGenerationCapabilityControls;")(
      makeElement(elements), {querySelectorAll: () => buttons},
      {path: "model.pt", data: {valid: true, generation_capabilities: {local_onset_guidance: false}}},
      selector => selector === "#checkpointPath" ? "model.pt" : "",
      (_selector, fallback) => fallback,
      [],
    );

  update();
  assert.equal(onset.disabled, true);
  assert.equal(onset.value, "0.35");
  assert.equal(onsetValue.textContent, "0.35");
  assert.match(onsetStatus.textContent, /不支持此指导/);
});

test("manual edit metadata is retained with saved settings", () => {
  const learningRate = {dataset: {touched: "1"}};
  const contextTokens = {dataset: {touched: "1"}};
  const collect = new Function("settingsFields", "element",
    functionSource("collectSettings", "saveSettings") + "; return collectSettings;")(
      () => [],
      selector => selector === "#learningRate" ? learningRate : contextTokens,
    );
  assert.deepEqual(collect().__manualTouched, {learningRate: true, contextTokens: true});
});
