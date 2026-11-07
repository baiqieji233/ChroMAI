const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
test('v2 train and generation require conditions and timing backend contract',()=>{
 assert.equal((source.match(/composer_v2_continuous_events_v3/g)||[]).length,2);
 assert.equal(source.includes('composer_v2_main_workbench_v1'),false);
 assert.match(source,/请重启为更新后的 WebUI 后台/);
});
function fn(name){let start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}
function context(values={}){const ctx=vm.createContext({Math,valueOrDefault:(key,fallback)=>values[key]??String(fallback),element:key=>({checked:values[key]===true}),
  currentTrainingMode:()=> 'fresh',trainingSourcePaths:()=>({manifest:'source.json'}),selectedTrainingManifests:()=>[],
  v2ConditionState:{note_density:{value:.5,override:false}},v2ConditionCapability:()=>({enabled:true}),
  parseTempoMap:()=>null,parseMeterMap:()=>null});vm.runInContext(fn('v2NumericValue'),ctx);return ctx;}

test('optional native tail guard and milliseconds reach main generation payload',()=>{
 const ctx=context({'#v2GenerateInterceptTapOnSlide':true,'#v2GenerateTapOnSlideThresholdMs':'85'});
 vm.runInContext(fn('composerV2GenerationPayload'),ctx);
 const p=ctx.composerV2GenerationPayload();
 assert.equal(p.intercept_tap_on_slide,true);assert.equal(p.tap_on_slide_threshold_ms,85);
 assert.match(html,/id="v2GenerateInterceptTapOnSlide" type="checkbox">/);
 vm.runInContext(fn('requireV2SplitSamplingSupport'),ctx);
 assert.throws(()=>ctx.requireV2SplitSamplingSupport({},p),/撞尾门控/);
});

test('onset reference uses the main controls and cannot silently hit an older backend',()=>{
 const ctx=context({'#v2GenerateOnsetStrength':'0'});
 vm.runInContext(fn('composerV2GenerationPayload'),ctx);
 assert.equal(ctx.composerV2GenerationPayload().onset_strength,0);
 ctx.element=()=>({disabled:true});
 assert.equal('onset_strength' in ctx.composerV2GenerationPayload(),false);
 vm.runInContext(fn('requireV2SplitSamplingSupport'),ctx);
 assert.throws(()=>ctx.requireV2SplitSamplingSupport({}, {onset_strength:1}),/起音参考强度/);
 assert.doesNotThrow(()=>ctx.requireV2SplitSamplingSupport({training_capabilities:{composer_v2_onset_readout:true}}, {onset_strength:1}));
 assert.match(html,/id="v2TrainOnsetReadout"/);
 assert.match(html,/data-v2-onset-readout hidden/);
 assert.match(html,/id="v2GenerateOnsetStrength"[^>]*value="1"[^>]*disabled/);
 assert.match(source,/generation_capabilities\?\.onset_readout/);
 const tr=context({'#v2TrainOnsetReadout':'phase_audio_v1'});
 vm.runInContext('const COMPOSER_V2_FIELDS=["onset_readout"];'+fn('v2FieldId')+fn('composerV2TrainingPayload'),tr);
 assert.equal(tr.composerV2TrainingPayload().onset_readout,'phase_audio_v1');
});
test('v2 payload includes rollback and excludes legacy curve/condition controls',()=>{
 const ctx=context({'#v2GenerateRollbackWindows':'5','#seed':'17','#bpm':'180'});
 vm.runInContext(fn('composerV2GenerationPayload'),ctx);
 const p=ctx.composerV2GenerationPayload();
 assert.equal(p.model_family,'chart_composer_v2');assert.equal(p.rollback_windows,5);assert.equal(p.candidates,1);assert.equal(p.cpu_threads,2);
 assert.equal(p.exhaustion_policy,'auto');
 assert.equal(Object.keys(p.condition).length,0);
 for(const key of ['curve_strength','intensity_curve','action_slots'])assert.equal(key in p,false);
});
test('continuous configuration sampling values are optional and independent of timing',()=>{
 const ctx=context({'#v2GenerateTemperature':'.65','#v2GenerateTopP':'.93',
   '#v2GenerateActionTemperature':'0','#v2GenerateActionTopP':'.98'});
 vm.runInContext(fn('composerV2GenerationPayload'),ctx);
 const result=ctx.composerV2GenerationPayload();
 assert.equal(result.temperature,.65);assert.equal(result.top_p,.93);
 assert.equal(result.action_temperature,0);assert.equal(result.action_top_p,.98);
 ctx.element=key=>({disabled:key.startsWith('#v2GenerateAction')});
 const legacy=ctx.composerV2GenerationPayload();
 assert.equal('action_temperature' in legacy,false);
 assert.equal('action_top_p' in legacy,false);
 for(const id of ['v2GenerateActionTemperature','v2GenerateActionTopP']) {
   assert.match(html,new RegExp('id="'+id+'"[^>]*disabled'));
   assert.ok(source.includes('"'+id+'",'));
 }
});
test('old backend cannot silently ignore a split configuration sampling request',()=>{
 const ctx=vm.createContext({});vm.runInContext(fn('requireV2SplitSamplingSupport'),ctx);
 assert.doesNotThrow(()=>ctx.requireV2SplitSamplingSupport({}, {}));
 assert.throws(()=>ctx.requireV2SplitSamplingSupport({}, {action_temperature:0}),/需要更新/);
 assert.doesNotThrow(()=>ctx.requireV2SplitSamplingSupport({training_capabilities:{
   composer_v2_split_sampling:true}}, {action_top_p:.98}));
});
test('v2 timing auto is explicit and sends only enabled overrides',()=>{
 const ctx=context();vm.runInContext(fn('composerV2GenerationPayload'),ctx);
 assert.equal(ctx.composerV2GenerationPayload().bpm,null);
 assert.equal(ctx.composerV2GenerationPayload().first,0);
 const auto=context({'#v2GenerateAutoFirst':true});vm.runInContext(fn('composerV2GenerationPayload'),auto);
 assert.equal(auto.composerV2GenerationPayload().first,null);
 const ready=context({'#bpm':'180'});ready.v2ConditionState.note_density.override=true;
 vm.runInContext(fn('composerV2GenerationPayload'),ready);
 assert.equal(ready.composerV2GenerationPayload().condition.note_density,.5);
});
test('v2 manual tempo map retains automatic first as null through the main-page payload',()=>{
 const ctx=context({'#v2GenerateAutoFirst':true});
 const tempoMap=[{beat:'0',bpm:112},{beat:'36',bpm:180}];
 ctx.parseTempoMap=()=>tempoMap;
 vm.runInContext(fn('composerV2GenerationPayload'),ctx);
 const payload=ctx.composerV2GenerationPayload();
 assert.equal(payload.first,null);assert.equal(payload.bpm,null);
 assert.equal(payload.tempo_map,tempoMap);
 assert.ok(source.includes('不会自动写入变速表'));
});
test('v2 training forwards auto/manual/off contract and arbitrary integral target window',()=>{
 const ctx=context({'#v2TrainValidationSource':'auto','#v2TrainHoldoutFraction':'.05','#v2TrainWindowBeats':'24'});
 vm.runInContext(source.slice(0,source.indexOf('function trainingMetricNumber(')),ctx);
 const p=ctx.composerV2TrainingPayload();
 assert.equal(p.validation_source,'auto');assert.equal(p.holdout_fraction,.05);assert.equal(p.window_beats,24);
});
test('v2 common training settings submit batch and both memory budgets',()=>{
 const ctx=context({'#v2TrainBatchSize':'2','#v2TrainGradAccumulation':'32','#v2TrainCudaMemoryBudgetGb':'7.5','#v2TrainAudioCacheGb':'0'});
 vm.runInContext(source.slice(0,source.indexOf('function trainingMetricNumber(')),ctx);
 const payload=ctx.composerV2TrainingPayload();
 assert.equal(payload.batch_size,2);
 assert.equal(payload.grad_accumulation,32);
 assert.equal(payload.cuda_memory_budget_gb,7.5);
 assert.equal(payload.audio_cache_gb,0);
});
test('same-page validation and rollback controls expose documented defaults',()=>{
 for(const id of ['v2TrainValidationSource','v2TrainHoldoutFraction','v2GenerateRollbackWindows'])assert.ok(html.includes(`id="${id}"`));
 assert.match(html,/id="v2GenerateRollbackWindows"[^>]*min="0"[^>]*max="8"[^>]*value="2"/);
 assert.ok(html.includes('冲突恢复次数'));
 assert.ok(html.includes('两者共用此预算，0关闭'));
 assert.match(html,/id="v2TrainValidationSource"><option value="auto" selected>/);
 assert.match(html,/id="v2TrainWindowBeats"[^>]*min="1"[^>]*step="1"/);
 assert.ok(html.includes('本地MERT目录（留空自动下载）'));
 assert.ok(source.includes('"v2GenerateRollbackWindows",'));
});

test('advanced v2 training details fold while common input stays open',()=>{
 assert.match(html,/<details class="settings-section" data-advanced><summary>模型与上下文<\/summary>/);
 assert.match(html,/<details class="settings-section" data-advanced><summary>验证、性能与保存<\/summary>/);
 assert.match(html,/<details class="settings-section" open><summary>常用训练设置<\/summary>/);
 assert.match(html,/<details class="settings-section" open><summary>数据与音频特征<\/summary>/);
 assert.match(html,/id="v2GenerateCpuThreads"[^>]*min="1"[^>]*value="2"/);
 assert.ok(source.includes('"v2GenerateCpuThreads",'));
});

test('completion presents the clock actually used including explicit zero',()=>{
 const ctx=vm.createContext({});vm.runInContext(fn('composerV2CompletionMessage'),ctx);
 const message=ctx.composerV2CompletionMessage({result:{composer_v2_report:{timing:{effective_bpm:95,effective_first:0,bpm_auto:true,first_auto:false}}}});
 assert.match(message,/95\.0000（自动识别）/);assert.match(message,/0\.00000 秒（手动）/);
 assert.match(message,/半速／倍速/);
});
test('completion surfaces variable-tempo and weak-phase warnings',()=>{
 const ctx=vm.createContext({});vm.runInContext(fn('composerV2CompletionMessage'),ctx);
 const report={timing:{effective_bpm:180,effective_first:.01,bpm_auto:true,first_auto:true,variable_tempo_suspected:true}};
 assert.match(ctx.composerV2CompletionMessage({result:{composer_v2_report:report}}),/当前仍采用单一 BPM/);
 report.timing.variable_tempo_suspected=false;report.timing.warnings=['weak pulse'];
 assert.match(ctx.composerV2CompletionMessage({result:{composer_v2_report:report}}),/检测一致性偏低/);
});

test('recovery summary distinguishes budget from actual source edits and rollbacks',()=>{
 const ctx=vm.createContext({});vm.runInContext(fn('composerGenerationSummary'),ctx);
 const message=ctx.composerGenerationSummary({kind:'composer_v2_generate',result:{composer_v2_report:{
   recovery_budget:2,rollbacks:[],duration_repairs:[{source_beat:'541/4'}]}}});
 assert.match(message,/冲突恢复预算 2 次/);
 assert.match(message,/实际撤回 0 次，Hold时值修复 1 次/);
 const local=ctx.composerGenerationSummary({kind:'composer_v2_generate',result:{composer_v2_report:{
   recovery_budget:2,recovery_budget_scope:'forward_progress_v2',rollbacks:[],duration_repairs:[{},{},{}]}}});
 assert.match(local,/单处冲突恢复预算 2 次/);
 assert.match(local,/全曲实际撤回 0 次，Hold时值修复 3 次/);
});

test('report mode is exposed and completion identifies bypassed events',()=>{
 assert.match(html,/id="v2GenerateExhaustionPolicy"><option value="auto" selected>/);
 assert.ok(source.includes('"v2GenerateExhaustionPolicy",'));
 const ctx=vm.createContext({});vm.runInContext(fn('composerV2CompletionMessage'),ctx);
 const result=ctx.composerV2CompletionMessage({result:{composer_v2_report:{accepted_with_conflicts_count:3}}});
 assert.match(result,/放行 3 个冲突事件/);
 assert.ok(source.includes('下载未解决冲突报告'));
});

test('old backend cannot silently ignore strict stop while automatic mode works',()=>{
 const ctx=vm.createContext({});vm.runInContext(fn('requireV2ExhaustionSupport'),ctx);
 assert.doesNotThrow(()=>ctx.requireV2ExhaustionSupport({}, {exhaustion_policy:'auto'}));
 assert.throws(()=>ctx.requireV2ExhaustionSupport({}, {exhaustion_policy:'stop'}),/需要更新后的后台/);
 assert.doesNotThrow(()=>ctx.requireV2ExhaustionSupport({training_capabilities:{composer_v2_exhaustion_policy:true}}, {exhaustion_policy:'stop'}));
});
