"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const assert = require("node:assert/strict");
const test = require("node:test");
const source = fs.readFileSync(path.join(__dirname, "../src/maichart/webui/static/app.js"), "utf8");
function fn(name) {
  const start = source.indexOf(`function ${name}(`);
  return source.slice(start, source.indexOf("\n}", start) + 2);
}
function setup() {
  const nodes = { "#intensitySaveButton": {}, "#intensityCurveEnabled": { checked: true },
    "#intensityCurveConditionNotice": {}, "#intensityCurveStatus": {} };
  const profile = {schema:"maichart-intensity-curve-v1",method:"learned_music_structure_v3",audio_sha256:"a".repeat(64),
    duration_sec:10,points:[{time_sec:0,intensity:.5},{time_sec:10,intensity:.3}],
    time_axis:"audio_absolute_v2",curve_resolution_sec:2.5,
    components:{density:[.4,.6],onset:[.2,.8]},
    condition:{level:13.3,radar:{note_density:.3},style:{slide_bias:.1}},rest_regions:[{start_sec:2,end_sec:4}]};
  let stored = null;
  const context = vm.createContext({
    intensityCurveState:{profile,automatic:profile,saved:true,contextStale:false,conditionChanged:false,history:[],path:"song.mp3"},
    intensityCurveStorageKey:"curve",localStorage:{setItem:(_,v)=>stored=JSON.parse(v),getItem:()=>JSON.stringify(stored)},
    element:key=>nodes[key] || null,checkedOrDefault:()=>true,updateIntensityViewControls(){},
    setWorkspaceMode(){},saveSettings(){},normalizeRestRegions:r=>r,
    normalizeIntensityProfile:p=>p,
  });
  vm.runInContext(["intensityProfileValid","updateIntensityButtons","updateCurveConditionNotice","setIntensityStatus",
    "saveIntensityCurveState","loadIntensityCurveState","intensityCurvePayload",
    "invalidateLearnedCurveForCondition","invalidateLearnedCurveForTiming","saveIntensityCurveAndReturn"].map(fn).join("\n"), context);
  return {context,nodes,profile,stored:()=>stored};
}

test("changing controls keeps saved curve and original prediction metadata usable across reload",()=>{
  const {context,nodes,profile,stored}=setup();
  const original=JSON.stringify(profile);
  vm.runInContext("invalidateLearnedCurveForCondition(); loadIntensityCurveState(); updateIntensityButtons();",context);
  assert.equal(context.intensityCurveState.saved,true);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(nodes["#intensitySaveButton"].disabled,false);
  assert.equal(nodes["#intensityCurveConditionNotice"].hidden,false);
  const payload=vm.runInContext("intensityCurvePayload()",context);
  assert.deepEqual(JSON.parse(JSON.stringify(payload.points)),profile.points);
  assert.deepEqual(JSON.parse(JSON.stringify(payload.rest_regions)),profile.rest_regions);
  assert.deepEqual(JSON.parse(JSON.stringify(payload.condition)),profile.condition);
  assert.equal(payload.time_axis,"audio_absolute_v2");
  assert.equal(payload.curve_resolution_sec,2.5);
  assert.deepEqual(JSON.parse(JSON.stringify(payload.components)),profile.components);
  assert.equal(JSON.stringify(profile),original);
  assert.equal(stored().conditionChanged,true);
});

test("control warning neither hides timing invalidation nor auto-saves edited points",()=>{
  const {context,nodes}=setup();
  vm.runInContext("invalidateLearnedCurveForTiming(); invalidateLearnedCurveForCondition(); loadIntensityCurveState(); updateIntensityButtons();",context);
  assert.equal(nodes["#intensitySaveButton"].disabled,true);
  assert.equal(vm.runInContext("intensityCurvePayload()",context),null);
  assert.throws(()=>vm.runInContext("saveIntensityCurveAndReturn()",context),/节拍/);
  context.intensityCurveState.contextStale=false;
  context.intensityCurveState.saved=false;
  vm.runInContext("invalidateLearnedCurveForCondition()",context);
  assert.equal(vm.runInContext("intensityCurvePayload()",context),null);
  vm.runInContext("saveIntensityCurveAndReturn()",context);
  assert.ok(vm.runInContext("intensityCurvePayload()",context));
});

test("legacy mixed stale flag no longer requires regenerating just to save",()=>{
  const {context}=setup();
  const old=JSON.stringify({profile:context.intensityCurveState.profile,contextStale:true,saved:false});
  context.localStorage.getItem=()=>old;
  vm.runInContext("loadIntensityCurveState(); saveIntensityCurveAndReturn();",context);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.ok(vm.runInContext("intensityCurvePayload()",context));
});

test("song-form curve editing and reload preserve null optional tempo identity",()=>{
  const {context,profile}=setup();
  Object.assign(profile,{method:"learned_song_form_v1",bpm_input:null,
    explicit_tempo_map:null,tempo_map_sha256:null,first:0,
    quality_profile:"musical_evidence_v1",curve_bin_seconds:1,
    tempo_changes:[{beat:"0",bpm:184}],checkpoint:"model.pt",
    checkpoint_identity:{size:100,sha256:"b".repeat(64)}});
  profile.points[0].intensity=.7;
  context.intensityCurveState.saved=false;
  vm.runInContext("saveIntensityCurveAndReturn(); loadIntensityCurveState();",context);
  const payload=JSON.parse(JSON.stringify(vm.runInContext("intensityCurvePayload()",context)));
  assert.ok(Object.hasOwn(payload,"tempo_map_sha256"));
  assert.ok(Object.hasOwn(payload,"explicit_tempo_map"));
  assert.equal(payload.tempo_map_sha256,null);
  assert.equal(payload.explicit_tempo_map,null);
  assert.equal(payload.bpm_input,null);
  assert.equal(payload.quality_profile,"musical_evidence_v1");
  assert.equal(payload.curve_bin_seconds,1);
  assert.equal(payload.points[0].intensity,.7);
  assert.deepEqual(payload.tempo_changes,profile.tempo_changes);
  assert.deepEqual(payload.checkpoint_identity,profile.checkpoint_identity);
});
