"use strict";
const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
function fn(name){
  const start=source.indexOf(`function ${name}(`);
  return source.slice(start,source.indexOf('\n}',start)+2);
}

function setup(){
  const values={'#level':'14.6','#composerConditionFile':'style.json','#composerCurveStrength':'1.2',
    '#intensityCurveStrength':'0','#composerIntensityCurveSpan':'trim','#composerChordMultiplier':'.75',
    '#checkpointPath':'composer.pt','#bpm':'180','#first':'.3','#formGenerationMeterMap':'meter',
    '#intensityCurveSource':'model'};
  const profile={schema:'maichart-intensity-curve-v1',method:'learned_audio_grounded_v3',duration_sec:10,
    points:[{time_sec:0,intensity:.8},{time_sec:10,intensity:.4}],
    reference_points:[{time_sec:0,intensity:.5},{time_sec:10,intensity:.5}],
    reference_event_rates:[1,1],reference_head_rates:[4,4],baseline_head_rate:4,baseline_p95_rate:8,
    intensity_mapping_policy:'bounded_odds_v1',intensity_rate_cap:64,
    model_family:'chart_composer_v1',architecture_profile:'audio_grounded_v3'};
  const context=vm.createContext({isEventSetGeneration:()=>false,updateEventSetControls:()=>{},valueOrDefault:(s,d)=>values[s]??d,checkedOrDefault:()=>true,
    collectCondition:()=>({level:14.6,style:{tap_bias:.2}}),parseTempoMap:()=>[{beat:'0',bpm:180}],
    parseMeterMap:()=>[{beat:'0',numerator:3,denominator:4}],formDevice:()=> 'cpu',
    normalizeRestRegions:r=>r??[],currentGenerationFamily:()=> 'chart_composer_v1',
    intensityCurveState:{profile,saved:true,contextStale:false},
    saveIntensityCurveState(){},updateIntensityButtons(){},setIntensityStatus(){},
  });
  vm.runInContext(['intensityCurvePayload','chartComposerGenerationPayload',
    'invalidateLearnedCurveForTiming','invalidateLearnedCurveForCondition'].map(fn).join('\n'),context);
  return {context,profile};
}

test('Composer sends independent curve strength, enum span and chord multiplier',()=>{
  const {context,profile}=setup();
  const payload=context.chartComposerGenerationPayload();
  assert.equal(payload.curve_strength,1.2);
  assert.equal(payload.intensity_curve_strength,0);
  assert.equal(payload.intensity_curve_span,'trim');
  assert.equal(payload.intensity_curve_chord_multiplier,.75);
  assert.equal(payload.condition_file,'style.json');
  assert.equal(payload.meter_map[0].numerator,3);
  assert.deepEqual(payload.intensity_curve.reference_points,profile.reference_points);
  assert.equal(payload.intensity_curve.baseline_head_rate,4);
  assert.equal(payload.intensity_curve.intensity_mapping_policy,'bounded_odds_v1');
  assert.deepEqual(payload.intensity_curve.reference_head_rates,[4,4]);
});

test('Composer prediction sends actual generation condition and timing',async()=>{
  const {context}=setup();let request;
  Object.assign(context,{generationPayload:()=>context.chartComposerGenerationPayload(),
    intensityAudioPath:()=> 'song.wav',setWorkspaceMode(){},setIntensitySong(){},pollIntensityTask:async()=>{},
    api:async(url,payload)=>url==='/api/status'?{intensity_curve_capability:true}:(request=payload,{task_id:'curve'})});
  vm.runInContext('async '+fn('generateIntensityCurve'),context);
  await context.generateIntensityCurve();
  assert.equal(request.level,14.6);
  assert.equal(request.condition_file,'style.json');
  assert.equal(request.condition.style.tap_bias,.2);
  assert.equal(request.bpm,180);
  assert.equal(request.first,.3);
  assert.equal(request.meter_map[0].numerator,3);
});

test('current Composer method participates in condition notices and timing invalidation',()=>{
  const {context}=setup();
  context.invalidateLearnedCurveForCondition();
  assert.equal(context.intensityCurveState.conditionChanged,true);
  assert.equal(context.intensityCurveState.saved,true);
  context.invalidateLearnedCurveForTiming();
  assert.equal(context.intensityCurveState.contextStale,true);
  assert.equal(context.intensityCurveState.saved,false);
});
