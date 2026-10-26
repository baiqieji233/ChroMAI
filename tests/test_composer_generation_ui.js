const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
function fn(name){let start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}

test('Composer payload forwards Radar/Style, tempo, temperature and blank seed',()=>{
  const values={'#level':'14.2','#temperature':'.91','#seed':'','#checkpointPath':'cp.pt',
    '#composerCurveStrength':'1.4','#composerMinSameHeadMs':'65','#composerPointerHistoryStrength':'.4',
    '#composerActionHistoryStrength':'.7','#composerActionTypeTemperature':'.8','#composerActionTypeMinP':'.06'};
  const condition={level:14.2,radar:{note_density:.42},style:{slide_bias:.7}};
  const tempo=[{beat:'0',bpm:150},{beat:'8',bpm:180}];
  const context=vm.createContext({valueOrDefault:(id,fallback)=>values[id]??fallback,
    checkedOrDefault:(id,fallback)=>Boolean(values[id]??fallback),
    formDevice:()=> 'cpu',collectCondition:()=>condition,parseTempoMap:()=>tempo});
  vm.runInContext(fn('chartComposerGenerationPayload'),context);
  const payload=context.chartComposerGenerationPayload();
  assert.equal(payload.condition,condition);assert.equal(payload.tempo_map,tempo);
  assert.equal(payload.seed,null);assert.equal(payload.temperature,.91);
  assert.equal(payload.curve_strength,1.4);assert.equal(payload.device,'cpu');
  assert.equal(payload.min_same_head_ms,65);
  assert.equal(payload.pointer_history_strength,.4);
  assert.equal(payload.action_history_strength,.7);
  assert.equal(payload.action_type_temperature,.8);
  assert.equal(payload.action_type_min_p,.06);
  assert.equal(payload.intercept_tap_on_slide,false);
  assert.equal(payload.touch_history_guard,true);assert.equal(payload.touch_history_bound,1);
  values['#composerTouchHistoryGuard']=false;values['#composerTouchHistoryBound']='0';
  assert.equal(context.chartComposerGenerationPayload().touch_history_guard,false);
  assert.equal(context.chartComposerGenerationPayload().touch_history_bound,0);
  values['#composerInterceptTapOnSlide']=true;
  assert.equal(context.chartComposerGenerationPayload().intercept_tap_on_slide,true);
  values['#seed']='0';assert.equal(context.chartComposerGenerationPayload().seed,0);
});

test('Main UI resets Touch controls and intercept with the generation defaults',()=>{
  const fields=source.slice(source.indexOf('const GENERATION_PARAM_FIELDS ='),source.indexOf('const CONFLICT_KIND_FIELDS ='));
  for (const id of ['composerTouchHistoryGuard','composerTouchHistoryBound','composerInterceptTapOnSlide']) {
    assert.ok(fields.includes(`"${id}"`),id);
  }
  assert.match(html,/id="composerTouchHistoryGuard"[^>]*checked/);
  assert.match(html,/id="composerTouchHistoryBound"[^>]*min="0"[^>]*max="4"[^>]*value="1"/);
});

test('New Composer controls persist across reset and submit their documented defaults',()=>{
  const fields=source.slice(source.indexOf('const GENERATION_PARAM_FIELDS ='),source.indexOf('const CONFLICT_KIND_FIELDS ='));
  for(const id of ['composerActionHistoryStrength','composerActionTypeTemperature','composerActionTypeMinP'])
    assert.ok(fields.includes(`"${id}"`),id);
  const c=vm.createContext({valueOrDefault:(_id,fallback)=>fallback,checkedOrDefault:(_id,fallback)=>fallback,
    formDevice:()=> 'cpu',collectCondition:()=>({}),parseTempoMap:()=>null});
  vm.runInContext(fn('chartComposerGenerationPayload'),c);
  const payload=c.chartComposerGenerationPayload();
  assert.equal(payload.action_history_strength,1);
  assert.equal(payload.action_type_temperature,1);
  assert.equal(payload.action_type_min_p,0);
});

test('Touch summary distinguishes effective, requested and disabled protection, including zero bound',()=>{
  const c=vm.createContext({});vm.runInContext(fn('composerGenerationSummary'),c);
  function summary(effective,requested,bound) {return c.composerGenerationSummary({kind:'composer_generate',result:{composer_report:{
    touch_history_guard:effective,touch_history_guard_requested:requested,touch_history_bound_requested:bound,
    touch_history_log_odds_bound:effective?bound:null}}});}
  assert.match(summary(true,true,0),/已生效；放大上限 0/);
  assert.match(summary(false,true,1),/已请求，当前模型或动作温度不适用/);
  assert.match(summary(false,false,2),/Touch 历史保护：关闭；放大上限 2/);
});

test('Composer uses dedicated temperatures and shared tempo controls',()=>{
  assert.doesNotMatch(html,/<label[^>]*data-generation-family="[^"]*chart_composer_v1[^"]*">Temperature/);
  assert.match(html,/<label[^>]*data-generation-family="[^"]*chart_composer_v1[^"]*">手动 tempo map/);
  assert.ok(source.includes('status.composer_generation_contract !== "composer_generation_controls_v11_history_type_sampling"'));
  assert.match(html,/id="composerCurveStrength"[^>]*min="0"[^>]*max="2"[^>]*value="1"/);
  assert.match(source,/"composerCurveStrength"/);
  assert.match(source,/"composerPointerTemperature"/);
  assert.match(source,/"composerActionTemperature"/);
  assert.match(html,/id="composerMinSameHeadMs"[^>]*min="0"[^>]*max="200"[^>]*value="50"/);
  assert.match(html,/id="composerPointerHistoryStrength"[^>]*min="0"[^>]*max="1"[^>]*value="1"/);
  assert.match(html,/id="composerActionHistoryStrength"[^>]*min="0"[^>]*max="1"[^>]*value="1"/);
  assert.match(html,/id="composerActionTypeTemperature"[^>]*min="0"[^>]*max="2"[^>]*value="1"/);
  assert.match(html,/id="composerActionTypeMinP"[^>]*min="0"[^>]*max="1"[^>]*value="0"/);
  assert.match(html,/value="selected_prefix_v3"/);
  assert.match(source,/"composerPointerHistoryStrength"/);
  for (const id of ['composerActionHistoryStrength','composerActionTypeTemperature','composerActionTypeMinP']) assert.ok(source.includes(`"${id}"`));
});

test('Composer result explains actual controls and greedy invariance',()=>{
  const context=vm.createContext({});vm.runInContext(fn('composerGenerationSummary'),context);
  const summary=context.composerGenerationSummary({kind:'composer_generate',result:{seed:0,
    effective_settings:{temperature:0,pointer_temperature:0.2,action_temperature:0.3,pointer_history_strength:.3,
      action_history_strength:.8,action_type_temperature:1.2,action_type_min_p:.1,condition_source:'ui+file'},composer_report:{weights_sha256:'a'.repeat(64),
      events_sha256:'b'.repeat(64),optimizer_steps:3500,condition:{level:14.2,style:{slide_bias:.7}}}}});
  assert.match(summary,/seed：0/);assert.ok(!summary.includes('贪心'));assert.match(summary,/候选温度 0.2/);assert.match(summary,/动作温度 0.3/);assert.match(summary,/3500/);
  assert.match(summary,/已覆盖/);assert.match(summary,/音符序列 bbbbbbbbbbbb/);
  assert.match(summary,/选时历史权重 0.3/);
  assert.match(summary,/动作历史权重 0.8/);assert.match(summary,/动作类别温度 1.2；min-p 0.1/);
});

test('Composer summary distinguishes learned workload and all unresolved hard Muri',()=>{
  const context=vm.createContext({});vm.runInContext(fn('composerGenerationSummary'),context);
  const summary=context.composerGenerationSummary({kind:'composer_generate',status:'completed',result:{
    composer_report:{curve_strength:0,learned_workload_curve:{times_sec:[0,2,4]},
      guard:{checks:18,rejections:4,skipped_candidates:2},
      playability:{remaining_hard:3,remaining_actionable_hard:0},stop_reason:'audio_end'}}});
  assert.match(summary,/已预测学习负荷曲线：3/);assert.match(summary,/引导强度 0/);
  assert.match(summary,/拒绝 4 次/);assert.match(summary,/跳过候选 2/);
  assert.match(summary,/未解决硬冲突 3 项（可处理 0 项）/);
  assert.match(summary,/结束原因：audio_end/);
  assert.ok(!summary.includes('无冲突'));
});

test('Composer summary reports missing workload capability without inventing a curve',()=>{
  const context=vm.createContext({});vm.runInContext(fn('composerGenerationSummary'),context);
  const summary=context.composerGenerationSummary({kind:'composer_generate',status:'completed',
    result:{composer_report:{schema:'legacy',playability:{remaining_hard:0}}}});
  assert.match(summary,/未提供学习负荷曲线/);
  assert.ok(!summary.includes('已预测学习负荷曲线'));
});


test('Touch layout and motif generation inputs forward independently and explain disabled memory',()=>{
  const values={'#composerTouchLayoutPolicy':'legacy_two_heads','#composerMotifStrength':'1.8'};
  const c=vm.createContext({valueOrDefault:(id,fallback)=>values[id]??fallback,
    checkedOrDefault:(id,fallback)=>fallback,formDevice:()=> 'cpu',collectCondition:()=>({}),parseTempoMap:()=>null});
  vm.runInContext(fn('chartComposerGenerationPayload'),c);
  vm.runInContext(fn('composerGenerationSummary'),c);
  const p=c.chartComposerGenerationPayload();
  assert.equal(p.touch_layout_policy,'legacy_two_heads');assert.equal(p.motif_strength,1.8);
  const summary=c.composerGenerationSummary({kind:'composer_generate',result:{
    effective_settings:{motif_strength:1.8,touch_layout_policy:'open'},composer_report:{motif_memory_policy:'off'}}});
  assert.match(summary,/模型关闭，强度调整无效/);assert.match(summary,/Touch布局：开放/);
  const fields=source.slice(source.indexOf('const GENERATION_PARAM_FIELDS ='),source.indexOf('const CONFLICT_KIND_FIELDS ='));
  for(const id of ['composerTouchLayoutPolicy','composerMotifStrength'])assert.ok(fields.includes(`"${id}"`));
});
