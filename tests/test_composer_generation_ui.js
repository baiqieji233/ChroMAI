const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
function fn(name){let start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}

test('Composer payload forwards Radar/Style, tempo, temperature and blank seed',()=>{
  const values={'#level':'14.2','#temperature':'.91','#seed':'','#checkpointPath':'cp.pt'};
  const condition={level:14.2,radar:{note_density:.42},style:{slide_bias:.7}};
  const tempo=[{beat:'0',bpm:150},{beat:'8',bpm:180}];
  const context=vm.createContext({valueOrDefault:(id,fallback)=>values[id]??fallback,
    formDevice:()=> 'cpu',collectCondition:()=>condition,parseTempoMap:()=>tempo});
  vm.runInContext(fn('chartComposerGenerationPayload'),context);
  const payload=context.chartComposerGenerationPayload();
  assert.equal(payload.condition,condition);assert.equal(payload.tempo_map,tempo);
  assert.equal(payload.seed,null);assert.equal(payload.temperature,.91);
  values['#seed']='0';assert.equal(context.chartComposerGenerationPayload().seed,0);
});

test('Composer temperature and tempo controls are visible to its family',()=>{
  assert.match(html,/<label[^>]*data-generation-family="[^"]*chart_composer_v1[^"]*">Temperature/);
  assert.match(html,/<label[^>]*data-generation-family="[^"]*chart_composer_v1[^"]*">手动 tempo map/);
  assert.match(source,/status.composer_generation_contract !== "composer_generation_controls_v3"/);
});

test('Composer result explains actual controls and greedy invariance',()=>{
  const context=vm.createContext({});vm.runInContext(fn('composerGenerationSummary'),context);
  const summary=context.composerGenerationSummary({kind:'composer_generate',result:{seed:0,
    effective_settings:{temperature:0,condition_source:'ui+file'},composer_report:{weights_sha256:'a'.repeat(64),
      events_sha256:'b'.repeat(64),optimizer_steps:3500,condition:{level:14.2,style:{slide_bias:.7}}}}});
  assert.match(summary,/seed：0/);assert.match(summary,/贪心/);assert.match(summary,/3500/);
  assert.match(summary,/已覆盖/);assert.match(summary,/音符序列 bbbbbbbbbbbb/);
});
