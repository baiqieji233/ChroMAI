const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
function fn(name){
  const start=source.indexOf(`function ${name}(`);
  return source.slice(start,source.indexOf('\n}',start)+2);
}

test('Composer analysis supports model curve source and passes checkpoint',async()=>{
  let request;
  const context=vm.createContext({
    intensityAudioPath:()=> 'song.mp3',setWorkspaceMode(){},setIntensitySong(){},setIntensityStatus(){},
    generationPayload:()=>({model_family:'chart_composer_v1',checkpoint:'composer.pt',bpm:190,first:.25}),
    currentGenerationFamily:()=> 'chart_composer_v1',valueOrDefault:()=> 'model',
    intensityCurveState:{},pollIntensityTask:async()=>{},
    api:async(url,payload)=>url==='/api/status'?{intensity_curve_capability:true}:(request=payload,{task_id:'curve'})
  });
  vm.runInContext('async '+fn('generateIntensityCurve'),context);
  await context.generateIntensityCurve();
  assert.equal(request.curve_source,'model');
  assert.equal(request.checkpoint,'composer.pt');
  assert.equal(request.model_family,'chart_composer_v1');
  assert.equal(request.bpm,190);
});

test('switching song-form to Composer keeps model source enabled and enables curve save',()=>{
  let family='song_form_v2';
  const model={disabled:false},audio={disabled:false},save={};
  const sourceSelect={value:'model',querySelector:s=>s.includes('"audio"')?audio:model};
  const context=vm.createContext({currentGenerationFamily:()=>family,
    element:s=>({'#intensityCurveSource':sourceSelect,'#intensitySaveButton':save}[s]||null),
    saveSettings(){},songFormCurveSourceBeforeRestriction:null,songFormCurveModeBeforeRestriction:null,
    songFormCurveRestrictionsActive:false});
  vm.runInContext(fn('updateSongFormGenerationOptions'),context);
  context.updateSongFormGenerationOptions();
  family='chart_composer_v1';context.updateSongFormGenerationOptions();
  assert.equal(sourceSelect.value,'model');assert.equal(model.disabled,false);
  assert.equal(audio.disabled,false);assert.match(save.textContent,/保存/);
  family='event_v1';context.updateSongFormGenerationOptions();assert.equal(model.disabled,false);
});
