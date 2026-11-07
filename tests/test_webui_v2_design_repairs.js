const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
function fn(name){const i=source.indexOf(`function ${name}(`);assert.ok(i>=0,name);return source.slice(i,source.indexOf('\n}',i)+2);}

// The actual row factory, event handlers, persistence and payload functions run
// together. Only transport and DOM primitives are replaced; no browser storage.
function page(saved=null){
 const ids=new Map(),mirrors=new Map();let saves=0;
 class Node{
  constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.dataset={};this.value='';this.type='';this.id='';this.textContent='';this.events={};this.classList={toggle(){}};}
  append(...nodes){for(const n of nodes){n.parentElement=this;this.children.push(n);}}
  appendChild(n){this.append(n);return n;}
  replaceChildren(...nodes){this.children=[];this.append(...nodes);}
  addEventListener(name,f){(this.events[name]??=[]).push(f);}
  fire(name){for(const f of this.events[name]||[])f({target:this});}
  closest(selector){let n=this;while(n){if(selector==='.control-row' && n.className==='control-row')return n;n=n.parentElement;}return null;}
  querySelectorAll(selector){
   const nodes=this.children.flatMap(n=>[n,...n.querySelectorAll(selector)]);
   return nodes.filter(n=>selector==='input[type=range],input[type=number]'?['range','number'].includes(n.type)
    :selector==='input[type=range]'?n.type==='range':selector==='input[type=number]'?n.type==='number'
    :selector==='input'?n.tagName==='INPUT':selector==='small'?n.tagName==='SMALL'
    :selector==='span'?n.tagName==='SPAN':selector==='[data-v2-override]'?n.overrideKey!==undefined
    :selector.startsWith('label[for=')?n.labelFor===selector.match(/"([^"]+)"/)[1]:false);
  }
  querySelector(s){return this.querySelectorAll(s)[0]||null;}
  set innerHTML(text){
   for(const m of text.matchAll(/<input\b([^>]*)>/g)){
    const n=new Node('input'),attrs=Object.fromEntries([...m[1].matchAll(/([\w-]+)="([^"]*)"/g)].map(x=>[x[1],x[2]]));
    Object.assign(n,{id:attrs.id||'',type:attrs.type,value:attrs.value||''});
    if(n.id)ids.set(n.id,n);
    if(attrs['data-number-for']){n.dataset.numberFor=attrs['data-number-for'];mirrors.set(n.dataset.numberFor,n);}
    if(attrs['data-v2-override'])n.overrideKey=attrs['data-v2-override'];
    this.append(n);
   }
   for(const tag of ['small','span'])if(text.includes('<'+tag)){const n=new Node(tag);n.textContent=text.match(new RegExp('<'+tag+'[^>]*>(.*?)</'+tag+'>'))?.[1]||'';this.append(n);}
   const label=text.match(/<label for="([^"]+)"/);if(label){const n=new Node('label');n.labelFor=label[1];this.append(n);}
  }
 }
 const add=(id,value='',type='text')=>{const n=new Node('input');Object.assign(n,{id,value,type});ids.set(id,n);return n;};
 for(const id of ['radarControls','styleControls']){const n=new Node();n.id=id;ids.set(id,n);}
 for(const [id,value] of Object.entries({level:'13.3',levelRange:'13.3',checkpointPath:'model.pt',seed:'17',first:'0',v2GenerateTemperature:'.9',v2GenerateTopP:'.95'}))add(id,value);
 const doc={createElement:t=>new Node(t),getElementById:id=>ids.get(id)||null,
  querySelector:s=>mirrors.get(s.match(/data-number-for="([^"]+)"/)?.[1])||null,
  querySelectorAll:s=>s==='[data-number-for]'?[...mirrors.values()]:[...ids.values()].filter(n=>n.tagName==='INPUT' && n.id)};
 const ctx=vm.createContext({document:doc,Math,Intl,Map,Set,console,
  element:s=>ids.get(s.slice(1))||null,valueOrDefault:(s,f)=>ids.get(s.slice(1))?.value??String(f),
  currentFamily:'chart_composer_v2',currentGenerationFamily:()=>ctx.currentFamily,
  generationInspection:{path:'model.pt',data:{valid:true,condition_contract:{policy:'radar_style_v1',supported_keys:[]}}},
  levelPresetProfile:null,GENERATION_LEVEL_DEFAULT:13.3,GENERATION_LEVEL_MIN:12,GENERATION_LEVEL_MAX:15,
  saveSettings:()=>saves++,invalidateLearnedCurveForCondition:()=>{},updateGenerationCapabilityControls:()=>{},
  updateConditionRecipeDisplay:()=>{},parseTempoMap:()=>null,parseMeterMap:()=>null,
  settingsStorageKey:'settings',datasetStorageKey:'datasets',datasetWorkspaceState:{selected:new Set()},
  migrateSongFormQualityDefaults:()=>false,migrateEventSetLearningDefaults:()=>false,restoreLegacyDatasetPaths:()=>false,
  localStorage:{getItem:k=>k==='settings'?JSON.stringify(saved):null,setItem(){}},
  GENERATION_PARAM_FIELDS:['level','v2GenerateTemperature','v2GenerateTopP']});
 const block=source.slice(source.indexOf('const radarKeys ='),source.indexOf('// Difficulty controls are calibrated'));
 vm.runInContext(block,ctx);
 for(const name of ['normalizeGenerationLevel','interpolateLevelPreset','controlRows','settingsFields','collectSettings','restoreSettings',
  'captureGenerationDefaults','generationFieldIds','resetGenerationDefaults','setConditionValue','v2NumericValue','composerV2GenerationPayload'])vm.runInContext(fn(name),ctx);
 ctx.controlRows('radarControls',vm.runInContext('radarKeys',ctx));ctx.controlRows('styleControls',vm.runInContext('styleKeys',ctx));
 ctx.generationDefaults=ctx.captureGenerationDefaults();
 ctx.generationInspection.data.condition_contract.supported_keys=[...ids.keys()];
 if(saved)ctx.restoreSettings();
 ctx.updateV2ConditionControls();
 return {ctx,ids,mirrors,Node,state:vm.runInContext('v2ConditionState',ctx),saves:()=>saves,
  toggle:key=>ids.get(key).closest('.control-row').querySelector('[data-v2-override]')};
}

test('first manual condition uses field reference and disabled conditions stay unknown',()=>{
 const p=page();assert.equal(p.ctx.composerV2GenerationPayload().condition.note_density,undefined);
 assert.equal(p.state.note_density.value,.305);assert.equal(p.state.tap_bias.value,.720);assert.equal(p.state.touch_bias.value,.039);
 const toggle=p.toggle('note_density');toggle.checked=true;toggle.fire('change');
 assert.equal(p.ctx.composerV2GenerationPayload().condition.note_density,.305);
 assert.ok(p.saves()>0);
});

test('loaded level reference initializes an override without later overwriting the manual choice',()=>{
 const p=page();
 p.ctx.levelPresetProfile=JSON.parse(fs.readFileSync('src/maichart/webui/static/generation_presets.json','utf8'));
 p.ids.get('level').value='14.2';p.ctx.updateV2ConditionControls();
 const expected=Number(p.ctx.interpolateLevelPreset(p.ctx.levelPresetProfile,14.2).radar.note_density.toFixed(3));
 assert.equal(p.state.note_density.value,expected);
 const toggle=p.toggle('note_density');toggle.checked=true;toggle.fire('change');
 p.ids.get('level').value='13';p.ctx.updateV2ConditionControls();
 assert.equal(p.ctx.composerV2GenerationPayload().condition.note_density,expected);
});

test('cleared radar and style overrides follow difficulty again through level controls and reload',()=>{
 const p=page(),handlers=new Map();
 p.ctx.levelPresetProfile=JSON.parse(fs.readFileSync('src/maichart/webui/static/generation_presets.json','utf8'));
 vm.runInContext(fn('applyConditionPreset'),p.ctx);
 p.ctx.on=(id,event,handler)=>handlers.set(id,handler);
 for(const id of ['level','levelRange'])
  vm.runInContext(source.split('\n').find(line=>line.startsWith(`on("#${id}",`)),p.ctx);
 p.ctx.updateV2ConditionControls();
 const keys=['note_density','tap_bias'];
 for(const key of keys){
  const toggle=p.toggle(key);toggle.checked=true;toggle.fire('change');
  p.mirrors.get(key).value='.87';p.mirrors.get(key).fire('input');
  toggle.checked=false;toggle.fire('change');
 }
 function check(level){
  const reference=p.ctx.interpolateLevelPreset(p.ctx.levelPresetProfile,level);
  for(const key of keys){
   const expected=Number((reference.radar[key]??reference.style[key]).toFixed(3));
   assert.equal(p.state[key].value,expected);
   assert.equal(Number(p.ids.get(key).value),expected);
   assert.equal(Number(p.mirrors.get(key).value),expected);
   assert.equal(p.ids.get(key).disabled,true);
   assert.equal(p.ctx.composerV2GenerationPayload().condition[key],undefined);
  }
 }
 check(13.3);
 p.ids.get('levelRange').value='14.2';handlers.get('#levelRange')({target:p.ids.get('levelRange')});check(14.2);
 p.ids.get('level').value='13';handlers.get('#level')();check(13);
 const reloaded=page({version:2,values:p.ctx.collectSettings()});
 reloaded.ctx.levelPresetProfile=p.ctx.levelPresetProfile;
 reloaded.ids.get('level').value='14.2';reloaded.ctx.updateV2ConditionControls();
 for(const key of keys){
  assert.equal(reloaded.state[key].value,reloaded.ctx.v2ConditionReference(key).value);
  reloaded.toggle(key).checked=true;reloaded.toggle(key).fire('change');
  assert.equal(reloaded.ctx.composerV2GenerationPayload().condition[key],reloaded.state[key].value);
 }
});

test('manual condition edits, refresh, and family switching preserve separate values',()=>{
 const p=page();const t=p.toggle('note_density');t.checked=true;t.fire('change');
 p.mirrors.get('note_density').value='.42';p.mirrors.get('note_density').fire('input');
 assert.equal(p.ctx.composerV2GenerationPayload().condition.note_density,.42);
 const values=p.ctx.collectSettings();assert.equal(values.note_density,'0.305');
 const reload=page({version:2,values});assert.equal(reload.ctx.composerV2GenerationPayload().condition.note_density,.42);
 assert.equal(reload.toggle('note_density').checked,true);
 reload.ctx.currentFamily='event_ctx_v1';reload.ctx.updateV2ConditionControls();
 assert.equal(reload.ids.get('note_density').value,'0.305');
 reload.ctx.currentFamily='chart_composer_v2';reload.ctx.updateV2ConditionControls();
 assert.equal(reload.ids.get('note_density').value,'0.42');
});

test('reset clears manual overrides and resets the values actually sent',()=>{
 const p=page();const t=p.toggle('note_density');t.checked=true;t.fire('change');
 p.mirrors.get('note_density').value='.8';p.mirrors.get('note_density').fire('input');
 p.ctx.resetGenerationDefaults();
 assert.equal(p.toggle('note_density').checked,false);assert.equal(p.state.note_density.override,false);
 assert.deepEqual({...p.ctx.composerV2GenerationPayload().condition},{});
 assert.equal(Number(p.ids.get('note_density').value),p.state.note_density.value);
});

test('v2 level slider handler changes the value in the actual request',()=>{
 const p=page();let handler;
 p.ctx.on=(id,event,f)=>{handler=f;};
 vm.runInContext(source.split('\n').find(line=>line.startsWith('on("#levelRange", "input"')),p.ctx);
 p.ids.get('levelRange').value='14.2';handler({target:p.ids.get('levelRange')});
 assert.equal(p.ctx.composerV2GenerationPayload().level,14.2);
});

test('empty, out-of-range and fractional generation numbers fail before submission',()=>{
 const p=page();
 for(const [id,value] of [['v2GenerateTemperature',''],['v2GenerateTopP','0'],['v2GenerateTopP','1.01']]){
  const n=p.ids.get(id),before=n.value;n.value=value;assert.throws(()=>p.ctx.composerV2GenerationPayload());n.value=before;
 }
 p.ids.get('v2GenerateTemperature').value='0';assert.equal(p.ctx.composerV2GenerationPayload().temperature,0);
 p.ids.get('seed').value='1.5';assert.throws(()=>p.ctx.composerV2GenerationPayload());
});

test('v2 numeric fields specify useful steps, required values and bounds',()=>{
 const inputs=[...html.matchAll(/<input\b[^>]*\bid="v2[^"]*"[^>]*>/g)].map(m=>m[0]).filter(t=>t.includes('type="number"'));
 const optional=new Set(['v2GenerateActionTemperature','v2GenerateActionTopP','v2GenerateOnsetStrength']);
 for(const tag of inputs){
  assert.ok(!tag.includes('step="any"'),tag);assert.ok(tag.includes(' min='),tag);
  if(!optional.has(tag.match(/\bid="([^"]+)"/)[1]))assert.ok(tag.includes(' required'),tag);
 }
 assert.match(inputs.find(t=>t.includes('id="v2GenerateTemperature"')),/step="0\.05"/);
 assert.match(inputs.find(t=>t.includes('id="v2TrainLr"')),/step="0\.000001"/);
});

test('manual validation payload omits disabled automatic split controls for older servers',()=>{
 const ctx=vm.createContext({currentTrainingMode:()=> 'fresh',trainingSourcePaths:()=>({manifest:'train.json'}),
  selectedTrainingManifests:()=>[],valueOrDefault:(key,fallback)=>({'#v2TrainValidationSource':'manual','#v2TrainValidationManifest':'dev.json','#v2TrainHoldoutFraction':'0'}[key]??String(fallback))});
 vm.runInContext(source.slice(0,source.indexOf('function trainingMetricNumber(')),ctx);
 const payload=ctx.composerV2TrainingPayload();assert.equal(payload.validation_source,'manual');
 assert.equal(payload.validation_manifest,'dev.json');assert.equal('holdout_fraction' in payload,false);
});

test('switching v1 to v2 and back rebuilds the correct metric cards',()=>{
 const p=page(),ctx=p.ctx;
 for(const id of ['taskTrainingMetrics','taskTrainingMetricsSource','taskTrainingMetricsGroups'])p.ids.set(id,new p.Node());
 ctx.composerTrainingView=()=>({cards:[['旧模型卡片',[['指标','1']]]],note:'旧模型指标'});
 for(const name of ['trainingMetricNumber','composerV2TrainingView','renderComposerV2TrainingMetrics','renderTaskTrainingMetrics'])vm.runInContext(fn(name),ctx);
 const v1={task_id:'v1',kind:'composer_train',result:{}};
 ctx.renderTaskTrainingMetrics(v1);
 ctx.renderTaskTrainingMetrics({task_id:'v2',kind:'composer_v2_train',result:{epoch_target:50}});
 ctx.renderTaskTrainingMetrics(v1);
 assert.equal(p.ids.get('taskTrainingMetricsGroups').children[0].children[0].textContent,'旧模型卡片');
});

test('checkpoint initialization restores refresh interval and clears null optional paths',()=>{
 const values={'#v2TrainRefreshEvents':{value:'16'},'#v2TrainMertModel':{value:'stale'}};
 const ctx=vm.createContext({COMPOSER_V2_FIELDS:['refresh_events','mert_model'],currentTrainingMode:()=> 'finetune',element:k=>values[k]});
 vm.runInContext(fn('v2FieldId')+'\n'+fn('applyComposerV2Inspection'),ctx);
 ctx.applyComposerV2Inspection({valid:true,model_config:{refresh_events:32,mert_model:null}});
 assert.equal(values['#v2TrainRefreshEvents'].value,'32');assert.equal(values['#v2TrainMertModel'].value,'');
});

test('old running-server records display distinct training and validation series and total progress',()=>{
 const ctx=vm.createContext({});vm.runInContext(fn('trainingMetricNumber')+'\n'+fn('composerV2TrainingView'),ctx);
 const runtime={device:'cuda',precision:'bf16',windows:200,model_config:{d_model:384,layers:8}};
 const train={stage:'training',epoch:2,batch:50,total:100,optimizer_steps:10,loss:.4,time_loss:.5};
 const complete={stage:'epoch_complete',epoch:1,total:50,loss:.9,validation_loss:.9};
 const task={kind:'composer_v2_train',status:'running',progress:.5,message:'composer_v2_checkpoint',
  result:{epoch_target:50,composer_v2_runtime:runtime,composer_v2_progress:train},
  logs:['composer_v2_progress '+JSON.stringify(complete),'composer_v2_progress '+JSON.stringify(train),'composer_v2_checkpoint {"epoch":1}']};
 const view=ctx.composerV2TrainingView(task);
 assert.equal(view.progress,.03);assert.match(view.status,/第 2\/50 轮/);assert.match(view.status,/保存检查点/);
 assert.equal(view.cards[0][1][0][1],'0.4000');assert.ok(view.cards[0][1].some(([k,v])=>k.startsWith('验证 loss')&&v==='0.9000'));
 assert.ok(view.cards[0][1].some(([k,v])=>k.startsWith('完整轮训练')&&v==='未记录'));
 assert.ok(view.cards[2][1].some(([k,v])=>k==='模型宽度 / 层数'&&v==='384 / 8'));
 assert.equal(view.history.length,1);
});

test('new checkpoint step count and generation repair status supersede older progress records',()=>{
 const ctx=vm.createContext({});
 vm.runInContext(fn('trainingMetricNumber')+'\n'+fn('composerV2TrainingView')+'\n'+fn('composerV2GenerationMessage'),ctx);
 const view=ctx.composerV2TrainingView({status:'running',result:{epoch_target:50,optimizer_steps:500,
  composer_v2_checkpoint:{optimizer_steps:500},composer_v2_progress:{stage:'training',epoch:1,batch:20,total:100,optimizer_steps:480}}});
 assert.ok(view.cards[1][1].some(([key,value])=>key==='参数更新步'&&value==='500'));
 const task={status:'running',message:'正在修复 Hold 尾部冲突',result:{composer_v2_event:{events:32,beat:'17/2'}}};
 assert.equal(ctx.composerV2GenerationMessage(task),'正在修复 Hold 尾部冲突');
 task.message='composer_v2_event';assert.match(ctx.composerV2GenerationMessage(task),/32 个事件.*17\/2/);
 task.message='composer_v2_rollback';assert.match(ctx.composerV2GenerationMessage(task),/正在回退/);
});
