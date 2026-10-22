"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
function fn(name) { const begin = source.indexOf(`function ${name}(`); assert.ok(begin>=0,name); return source.slice(begin,source.indexOf("\n}",begin)+2); }
function setup() {
  const values = {};
  const context = vm.createContext({currentTrainingMode:()=>"fresh",currentModelFamily:()=>"song_form_v2",
    currentGenerationFamily:()=>"song_form_v2",trainingSourcePaths:()=>({}),selectedTrainingManifests:()=>[],
    selectedDatasetItems:()=>[],valueOrDefault:(id,fallback)=>values[id] ?? fallback,
    checkedOrDefault:(_id,fallback)=>fallback,formDevice:()=>"cpu",collectCondition:()=>({level:13.3}),
    parseTempoMap:()=>null,intensityCurvePayload:()=>null,songFormRationalGridDivisions:()=>[4,8,12,16,24,32],
    songFormRhythmSnapDivisions:()=>[4,8,12,16,24,32]});
  vm.runInContext(["parseMeterMap","songFormTrainingPayload","songFormGenerationPayload"].map(fn).join("\n"),context);
  return {context,values};
}
test("optional meter maps keep unknown meter and accept exact rational beat strings",()=>{
  const {context,values} = setup();
  assert.equal(context.songFormTrainingPayload().meter_map,null);
  assert.equal(context.songFormGenerationPayload().meter_map,null);
  const meters = [{beat:"0",numerator:3,denominator:4,grouping:[3]},
    {beat:"9/2",numerator:7,denominator:8,grouping:[2,2,3]}];
  values["#formTrainingMeterMap"] = values["#formGenerationMeterMap"] = JSON.stringify(meters);
  assert.deepEqual(JSON.parse(JSON.stringify(context.songFormTrainingPayload().meter_map)),meters);
  assert.deepEqual(JSON.parse(JSON.stringify(context.songFormGenerationPayload().meter_map)),meters);
  for (const invalid of ['{}','[{"beat":0.5,"numerator":3,"denominator":4}]',
    '[{"beat":"0","numerator":7,"denominator":8,"grouping":[3,3]}]']) {
    values["#formTrainingMeterMap"] = invalid;
    assert.throws(()=>context.songFormTrainingPayload(),/拍号|拍组/);
  }
});
test("evidence profile and optional numeric overrides are explicit in training payload",()=>{
  const {context,values} = setup();
  const defaults = context.songFormTrainingPayload();
  assert.equal(defaults.quality_profile,"auto");
  for (const key of ["position_alignment_loss_weight","planner_context_seconds","curve_bin_seconds"]) assert.equal(defaults[key],null);
  Object.assign(values,{"#formQualityProfile":"musical_evidence_v1","#formPositionAlignmentLossWeight":".15",
    "#formPositionAlignmentToleranceMs":"20", "#formPlannerContextSeconds":"1","#formCurveBinSeconds":"1"});
  const enhanced = context.songFormTrainingPayload();
  assert.equal(enhanced.quality_profile,"musical_evidence_v1");
  assert.equal(enhanced.position_alignment_loss_weight,.15);
  assert.equal(enhanced.position_alignment_tolerance_ms,20);
  assert.equal(enhanced.planner_context_seconds,1);
  assert.equal(enhanced.curve_bin_seconds,1);
  values["#formCurveBinSeconds"]="0";
  assert.throws(()=>context.songFormTrainingPayload(),/负荷标签/);
  values["#formQualityProfile"]="legacy";
  assert.equal(context.songFormTrainingPayload().position_alignment_loss_weight,null);
  assert.equal(context.songFormTrainingPayload().curve_bin_seconds,null);
});
test("curve analysis forwards exactly the generation meter map",async()=>{
  const {context,values} = setup();
  const meters = [{beat:"0",numerator:7,denominator:8,grouping:[2,2,3]}];
  values["#formGenerationMeterMap"] = JSON.stringify(meters);
  let analyzed;
  Object.assign(context,{intensityAudioPath:()=>"song.mp3",setWorkspaceMode(){},setIntensitySong(){},setIntensityStatus(){},
    generationPayload:()=>context.songFormGenerationPayload(),intensityCurveState:{},pollIntensityTask:async()=>{},
    api:async(url,payload)=>url==="/api/status" ? {intensity_curve_capability:true} : (analyzed=payload,{task_id:"curve"})});
  vm.runInContext("async "+fn("generateIntensityCurve"),context);
  await context.generateIntensityCurve();
  assert.deepEqual(JSON.parse(JSON.stringify(analyzed.meter_map)),meters);
});
test("resume restores and locks meter and evidence contract while finetune defaults to inheritance",()=>{
  const nodes = new Map();
  for (const id of ["formTrainingMeterMap","formQualityProfile","formQualityProfileHelp","formPositionAlignmentLossWeight",
    "formPlannerContextSeconds","formCurveBinSeconds"]) nodes.set("#"+id,{value:"auto",disabled:false});
  let mode="resume";
  const inspection = {valid:true,model_family:"song_form_v2",model_config:{quality_profile:"musical_evidence_v1",curve_bin_seconds:1},
    blueprint_config:{context_seconds:1.5},resume_settings:{position_alignment_loss_weight:.2},
    dataset_contract:{meter_map:[{beat:"0",numerator:3,denominator:4}]}};
  const context = vm.createContext({element:id=>nodes.get(id),currentTrainingMode:()=>mode,currentModelFamily:()=>"song_form_v2",
    checkpointInspection:{data:inspection},updateSongFormRhythmGeometryControls(){}});
  vm.runInContext(["updateSongFormQualityProfileControls","applySongFormCheckpointConfig"].map(fn).join("\n"),context);
  context.applySongFormCheckpointConfig(inspection,mode);
  assert.equal(nodes.get("#formTrainingMeterMap").disabled,true);
  assert.equal(JSON.parse(nodes.get("#formTrainingMeterMap").value)[0].numerator,3);
  assert.equal(nodes.get("#formQualityProfile").value,"musical_evidence_v1");
  assert.equal(nodes.get("#formPositionAlignmentLossWeight").value,"0.2");
  assert.equal(nodes.get("#formPlannerContextSeconds").value,"1.5");
  assert.equal(nodes.get("#formCurveBinSeconds").value,"1");
  for (const id of ["formQualityProfile","formPositionAlignmentLossWeight","formPlannerContextSeconds","formCurveBinSeconds"]) assert.equal(nodes.get("#"+id).disabled,true);
  mode="finetune";
  context.applySongFormCheckpointConfig(inspection,mode);
  assert.equal(nodes.get("#formQualityProfile").value,"auto");
  assert.equal(nodes.get("#formTrainingMeterMap").disabled,false);
  assert.equal(nodes.get("#formPositionAlignmentLossWeight").disabled,false);
});
test("finetune preflight keeps an explicit upgrade, meter override and geometry choice",async()=>{
  const nodes = new Map();
  const choices = {formTrainingMeterMap:'[{"beat":"0","numerator":3,"denominator":4}]',
    formQualityProfile:"musical_evidence_v1",formPositionAlignmentLossWeight:".25",formPlannerContextSeconds:"1.5",
    formCurveBinSeconds:"1",formRhythmGeometryProfileSelect:"exact_fraction_v2",formRhythmLossWeight:".2"};
  for (const [id,value] of Object.entries(choices)) nodes.set("#"+id,{value});
  const inspection = {valid:true,can_train:true,model_family:"song_form_v2",model_config:{quality_profile:"legacy"},
    dataset_contract:{meter_map:[]}};
  const context = vm.createContext({element:id=>nodes.get(id),currentTrainingMode:()=>"finetune",currentModelFamily:()=>"song_form_v2",
    valueOrDefault:(id,fallback)=>id==="#trainingCheckpointPath"?"old.pt":nodes.get(id)?.value??fallback,
    checkpointInspection:{data:null},checkpointInspectRequest:0,currentCheckpointKey:()=>"key",
    assertTrainingSelection(){},assertDatasetBackend:async()=>{},setCheckpointStatus(){},updateTrainingControls(){},
    trainingSourcePaths:()=>({manifest:'training_manifest.json'}),
    updateSongFormRhythmGeometryControls(){},api:async()=>inspection});
  vm.runInContext(["applySongFormCheckpointConfig"].map(fn).join("\n")+"\nasync "+fn("inspectCheckpoint"),context);
  await context.inspectCheckpoint({preserveEditableConfig:true});
  for (const [id,value] of Object.entries(choices)) assert.equal(nodes.get("#"+id).value,value,id);
});

function restoredEvidenceHarness() {
  const nodes = new Map();
  for (const id of ["formQualityProfile","formQualityProfileHelp","formPositionAlignmentLossWeight",
    "formPositionAlignmentToleranceMs","formPlannerContextSeconds","formCurveBinSeconds"]) {
    nodes.set("#"+id,{value:"auto",disabled:false,dataset:{}});
  }
  let mode="resume";
  const old = {valid:true,model_family:"song_form_v2",model_config:{quality_profile:"legacy",curve_bin_seconds:2},
    blueprint_config:{context_seconds:0},resume_settings:{position_alignment_loss_weight:0,position_alignment_tolerance_ms:15}};
  const context = vm.createContext({element:id=>nodes.get(id),currentModelFamily:()=>"song_form_v2",currentTrainingMode:()=>mode,
    valueOrDefault:(id,fallback)=>nodes.get(id)?.value??fallback,
    checkpointInspection:{data:old},updateSongFormRhythmGeometryControls(){},
    trainingSourcePaths:()=>({}),selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],
    checkedOrDefault:(_id,fallback)=>fallback,formDevice:()=>"cpu"});
  vm.runInContext(["updateSongFormQualityProfileControls","markSongFormEvidenceInput","applySongFormCheckpointConfig",
    "songFormTrainingPayload"].map(fn).join("\n"),context);
  context.applySongFormCheckpointConfig(old,"resume");
  return {context,nodes,setMode:value=>{mode=value;}};
}
test("explicit upgrade discards restored legacy overrides but keeps manually entered values",()=>{
  const {context,nodes,setMode}=restoredEvidenceHarness();
  assert.equal(nodes.get("#formCurveBinSeconds").value,"2");
  setMode("finetune");
  nodes.get("#formQualityProfile").value="musical_evidence_v1";
  context.markSongFormEvidenceInput(nodes.get("#formQualityProfile"));
  context.updateSongFormQualityProfileControls();
  const automatic = context.songFormTrainingPayload();
  assert.equal(automatic.quality_profile,"musical_evidence_v1");
  // Null explicitly delegates to the backend's upgrade defaults .15 / 15 / 1 / 1.
  for (const key of ["position_alignment_loss_weight","position_alignment_tolerance_ms","planner_context_seconds","curve_bin_seconds"]) assert.equal(automatic[key],null);
  nodes.get("#formPositionAlignmentLossWeight").value="0";
  context.markSongFormEvidenceInput(nodes.get("#formPositionAlignmentLossWeight"));
  context.updateSongFormQualityProfileControls();
  assert.equal(context.songFormTrainingPayload().position_alignment_loss_weight,0);
});
test("switching a fresh enhanced configuration to legacy never submits disabled stale overrides",()=>{
  const {context,nodes,setMode}=restoredEvidenceHarness();
  setMode("fresh");
  for (const [id,value] of [["formPositionAlignmentLossWeight",".4"],["formPlannerContextSeconds","2"],["formCurveBinSeconds",".5"]]) {
    nodes.get("#"+id).value=value;
    context.markSongFormEvidenceInput(nodes.get("#"+id));
  }
  nodes.get("#formQualityProfile").value="legacy";
  context.markSongFormEvidenceInput(nodes.get("#formQualityProfile"));
  context.updateSongFormQualityProfileControls();
  const payload=context.songFormTrainingPayload();
  for (const key of ["position_alignment_loss_weight","position_alignment_tolerance_ms","planner_context_seconds","curve_bin_seconds"]) assert.equal(payload[key],null);
  assert.equal(nodes.get("#formPositionAlignmentLossWeight").value,".4");
});
