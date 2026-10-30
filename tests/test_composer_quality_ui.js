const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
function fn(name){let start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}

test('quality controls submit real zero and retain their main-page reset registration',()=>{
  const values={'#composerQualityContextStrength':'0','#composerQualityPlanStrength':'1.4','#composerQualityLookaheadStrength':'.5'};
  const c=vm.createContext({isEventSetGeneration:()=>false,updateEventSetControls:()=>{},valueOrDefault:(id,fallback)=>values[id]??fallback,checkedOrDefault:(_id,fallback)=>fallback,
    formDevice:()=> 'cpu',collectCondition:()=>({}),parseTempoMap:()=>null,intensityCurvePayload:()=>null});
  vm.runInContext(fn('chartComposerGenerationPayload'),c);
  const payload=c.chartComposerGenerationPayload();
  assert.equal(payload.quality_context_strength,0);
  assert.equal(payload.quality_plan_strength,1.4);
  assert.equal(payload.quality_lookahead_strength,.5);
  for(const id of ['composerQualityContextStrength','composerQualityPlanStrength','composerQualityLookaheadStrength']) {
    assert.ok(source.includes(`"${id}"`));assert.ok(html.includes(`id="${id}"`));
  }
  assert.ok(html.includes('id="composerConfigurationQualityPolicy"'));
});

test('quality result displays effective off instead of claiming the requested strength applied',()=>{
  const c=vm.createContext({isEventSetGeneration:()=>false,updateEventSetControls:()=>{},});vm.runInContext(fn('composerGenerationSummary'),c);
  const text=c.composerGenerationSummary({kind:'composer_generate',result:{composer_report:{configuration_quality_policy:'off'},effective_settings:{quality_lookahead_strength:1}}});
  assert.match(text,/新增强度不生效/);assert.ok(!text.includes('微前瞻强度 1'));
});
