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
function setup(family = "event_v1") {
  const nodes = { "#intensitySaveButton": {}, "#intensityCurveEnabled": { checked: true },
    "#intensityCurveConditionNotice": {}, "#intensityCurveStatus": {}, "#checkpointPath": {value:"model.pt"} };
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
    valueOrDefault:(key,fallback)=>nodes[key]?.value??fallback,
    setWorkspaceMode(){},saveSettings(){},normalizeRestRegions:r=>r,
    currentGenerationFamily:()=>family,
    normalizeIntensityProfile:p=>p,
  });
  vm.runInContext(["intensityProfileValid","updateIntensityButtons","updateCurveConditionNotice","setIntensityStatus",
    "saveIntensityCurveState","loadIntensityCurveState","intensityCurvePayload",
    "invalidateLearnedCurveForCondition","invalidateLearnedCurveForTiming","normalizeAutomaticCurveEditFlag","invalidateLearnedCurveForCheckpoint","saveIntensityCurveAndReturn"].map(fn).join("\n"), context);
  return {context,nodes,profile,stored:()=>stored};
}

test("Composer saves an editable curve and enables curve generation",()=>{
  const {context,nodes}=setup('chart_composer_v1');
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  assert.equal(context.intensityCurveState.saved,true);
  assert.equal(nodes['#intensityCurveEnabled'].checked,true);
  assert.match(nodes['#intensityCurveStatus'].textContent,/已保存/);
});

test("saving an untouched automatic curve never converts it into manual control",()=>{
  const {context,profile}=setup('chart_composer_v1');
  profile.method='learned_chart_composer_event_set_stream_v1';profile.edited=false;
  const before=JSON.stringify(profile);
  vm.runInContext('saveIntensityCurveAndReturn(); saveIntensityCurveAndReturn();',context);
  assert.equal(JSON.stringify(profile),before);
  assert.equal(vm.runInContext('intensityCurvePayload().edited',context),false);
  profile.edited=true;profile.points[0].intensity=.8;
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  assert.equal(vm.runInContext('intensityCurvePayload().edited',context),true);
  assert.equal(profile.points[0].intensity,.8);
});

test("checkpoint changes invalidate automatic predictions but retain deliberate edits",()=>{
  const {context,nodes,profile}=setup('chart_composer_v1');
  profile.checkpoint='MODEL.pt';profile.edited=false;
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  nodes['#checkpointPath'].value='new.pt';
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.saved,false);
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.equal(vm.runInContext('intensityCurvePayload()',context),null);
  assert.throws(()=>vm.runInContext('saveIntensityCurveAndReturn()',context),/检查点/);
  context.intensityCurveState.contextStale=false;context.intensityCurveState.saved=true;
  profile.edited=true;profile.points[0].intensity=.85;
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.saved,true);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(profile.points[0].intensity,.85);
});

test("untrained model audio preview keeps its automatic provenance in generation payload",()=>{
  const {context,profile}=setup('chart_composer_v1');
  Object.assign(profile,{method:'audio_reference_untrained_composer_preview_v1',checkpoint:'model.pt',
    model_prediction_available:false,preview_only:true,readout_policy:'audio_reference',edited:false});
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  const payload=vm.runInContext('intensityCurvePayload()',context);
  assert.equal(payload.edited,false);assert.equal(payload.preview_only,true);
  assert.equal(payload.model_prediction_available,false);assert.equal(payload.readout_policy,'audio_reference');
});

test("same checkpoint path with a newer saved revision invalidates only the automatic preview",()=>{
  const {context,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'model.pt',edited:false,checkpoint_identity:{epoch:13,optimizer_steps:7592}});
  context.identity={epoch:13,optimizer_steps:7592};
  vm.runInContext('invalidateLearnedCurveForCheckpoint(identity)',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  context.identity.optimizer_steps=8000;
  vm.runInContext('invalidateLearnedCurveForCheckpoint(identity)',context);
  assert.equal(context.intensityCurveState.contextStale,true);
});

test("backend absolute checkpoint and project-relative input identify the same model",()=>{
  const {context,nodes,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'E:\\simai_transformer\\checkpoints\\v50\\composer_last.pt',edited:false,
    checkpoint_identity:{epoch:14,optimizer_steps:8000}});
  context.intensityCurveState.projectRoot='E:\\simai_transformer';
  nodes['#checkpointPath'].value='./checkpoints/tmp/../v50/COMPOSER_LAST.pt';
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  assert.equal(context.intensityCurveState.saved,true);
  nodes['#checkpointPath'].value='checkpoints/v51/composer_last.pt';
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,true); // Same basename never proves identity.
});

test("unresolved absolute-relative paths wait for project root and recover an old false warning",()=>{
  const {context,nodes,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'E:/simai_transformer/checkpoints/v50/model.pt',edited:false});
  nodes['#checkpointPath'].value='checkpoints/v50/model.pt';
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  Object.assign(context.intensityCurveState,{saved:false,contextStale:true,contextStaleReason:'checkpoint',
    projectRoot:'e:/simai_transformer/'});
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(context.intensityCurveState.contextStaleReason,null);
  assert.equal(nodes['#intensitySaveButton'].disabled,false);
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  assert.equal(context.intensityCurveState.saved,true);
});

test("a known revision mismatch survives repeated save checks and reload",()=>{
  const {context,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'model.pt',edited:false,checkpoint_identity:{epoch:13,optimizer_steps:7592}});
  context.identity={epoch:14,optimizer_steps:8000};
  vm.runInContext('invalidateLearnedCurveForCheckpoint(identity); invalidateLearnedCurveForCheckpoint();',context);
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.throws(()=>vm.runInContext('saveIntensityCurveAndReturn()',context),/检查点/);
  vm.runInContext('loadIntensityCurveState();',context);
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.throws(()=>vm.runInContext('saveIntensityCurveAndReturn()',context),/检查点/);
});

test("empty revision metadata does not masquerade as epoch or step zero",()=>{
  const {context,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'model.pt',edited:false,checkpoint_identity:{epoch:13,optimizer_steps:7592}});
  context.identity={epoch:'',optimizer_steps:null};
  vm.runInContext('invalidateLearnedCurveForCheckpoint(identity)',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(context.intensityCurveState.checkpointRevision,undefined);
  context.identity={epoch:' ',optimizer_steps:'\t'};
  vm.runInContext('invalidateLearnedCurveForCheckpoint(identity)',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(context.intensityCurveState.checkpointRevision,undefined);
});

test("checkpoint recovery never clears an independent timing warning",()=>{
  const {context,nodes,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'model.pt',edited:false});
  vm.runInContext('invalidateLearnedCurveForTiming()',context);
  nodes['#checkpointPath'].value='new.pt';
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  nodes['#checkpointPath'].value='model.pt';
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.equal(context.intensityCurveState.contextStaleReason,'timing');
  assert.throws(()=>vm.runInContext('saveIntensityCurveAndReturn()',context),/节拍/);
});

test("a fresh backend result replaces older inspection identity before saving",async()=>{
  const {context,nodes,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'E:/simai_transformer/checkpoints/v50/model.pt',edited:false,
    checkpoint_identity:{epoch:14,optimizer_steps:8000}});
  nodes['#checkpointPath'].value='checkpoints/v50/model.pt';
  Object.assign(context.intensityCurveState,{taskId:'curve',projectRoot:'E:/simai_transformer',
    checkpointRevision:{path:'checkpoints/v50/model.pt',epoch:13,optimizer_steps:7592},
    contextStale:true,contextStaleReason:'checkpoint'});
  context.intensityAudioPath=()=> 'song.mp3';
  context.api=async()=>({status:'completed',result:{profile}});
  context.showIntensityProfile=value=>Object.assign(context.intensityCurveState,{profile:value,
    saved:false,contextStale:false,contextStaleReason:null});
  vm.runInContext('async '+fn('pollIntensityTask'),context);
  await context.pollIntensityTask('curve','song.mp3');
  assert.equal(context.intensityCurveState.contextStale,false);
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  assert.equal(context.intensityCurveState.saved,true);
  assert.equal(context.intensityCurveState.checkpointRevision.optimizer_steps,8000);
});

test("legacy save-only edited flag is normalized on reload and makes an old automatic checkpoint stale",()=>{
  const {context,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'old.pt',edited:true,rest_regions:[],
    reference_points:JSON.parse(JSON.stringify(profile.points))});
  const saved=JSON.stringify({profile,contextVersion:2,saved:true});
  context.localStorage.getItem=()=>saved;
  vm.runInContext('loadIntensityCurveState()',context);
  assert.equal(context.intensityCurveState.profile.edited,false);
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.equal(context.intensityCurveState.saved,false);
  assert.throws(()=>vm.runInContext('saveIntensityCurveAndReturn()',context),/检查点/);
});

test("bootstrap checks curve identity after restoring the selected checkpoint and chord control",()=>{
  const {context,profile,nodes}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'old.pt',edited:true,rest_regions:[],
    reference_points:JSON.parse(JSON.stringify(profile.points))});
  const saved=JSON.stringify({profile,contextVersion:2,saved:true});
  context.localStorage.getItem=()=>saved;nodes['#checkpointPath'].value='';
  vm.runInContext('loadIntensityCurveState(false)',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(context.intensityCurveState.profile.edited,true);
  nodes['#checkpointPath'].value='old.pt';nodes['#composerChordMultiplier']={value:'.8'};
  vm.runInContext('invalidateLearnedCurveForCheckpoint()',context);
  assert.equal(context.intensityCurveState.contextStale,false);
  assert.equal(context.intensityCurveState.profile.edited,true);
  assert.ok(source.indexOf('loadIntensityCurveState(false);')<source.indexOf('const restoredSettings = restoreSettings();'));
  assert.match(source.slice(source.indexOf('const restoredSettings = restoreSettings();'),
    source.indexOf('if (restoredSettingNotice)')),/invalidateLearnedCurveForCheckpoint\(\)/);
});

test("legacy flag migration preserves every real point, rest or chord edit",()=>{
  const {context,profile,nodes}=setup('chart_composer_v1');
  const base={...JSON.parse(JSON.stringify(profile)),rest_regions:[],edited:true,
    reference_points:JSON.parse(JSON.stringify(profile.points))};
  for(const edit of ['point','time','rest','chord','controlChord','noReference','manualSource']) {
    const changed=JSON.parse(JSON.stringify(base));
    if(edit==='point')changed.points[0].intensity+=.000001;
    if(edit==='time')changed.points[0].time_sec+=.000001;
    if(edit==='rest')changed.rest_regions=[{start_sec:2,end_sec:4}];
    if(edit==='chord')changed.chord_multiplier=.8;
    if(edit==='noReference')delete changed.reference_points;
    if(edit==='manualSource')changed.method='manual_curve';
    context.profile=changed;context.multiplier=edit==='controlChord'?.8:1;
    vm.runInContext('normalizeAutomaticCurveEditFlag(profile,multiplier)',context);
    assert.equal(changed.edited,true,edit);
  }
  Object.assign(profile,JSON.parse(JSON.stringify(base)),{checkpoint:'model.pt'});
  context.intensityCurveState.profile=profile;
  vm.runInContext('saveIntensityCurveAndReturn()',context);
  assert.equal(profile.edited,false);
});

test("an in-flight automatic result cannot silently bind to a newly selected checkpoint",async()=>{
  const {context,nodes,profile}=setup('chart_composer_v1');
  Object.assign(profile,{checkpoint:'old.pt',edited:false});nodes['#checkpointPath'].value='new.pt';
  context.intensityCurveState.taskId='curve';
  context.intensityAudioPath=()=> 'song.mp3';
  context.api=async()=>({status:'completed',result:{profile}});
  context.showIntensityProfile=value=>{context.intensityCurveState.profile=value;context.intensityCurveState.contextStale=false;};
  vm.runInContext('async '+fn('pollIntensityTask'),context);
  await context.pollIntensityTask('curve','song.mp3');
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.match(nodes['#intensityCurveStatus'].textContent,/检查点/);
});

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
