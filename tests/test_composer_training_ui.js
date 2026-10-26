const fs=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
const helper=source.slice(source.indexOf('function composerTrainingView('),source.indexOf('function renderTaskTrainingMetrics('));
const context=vm.createContext({trainingMetricNumber:(object,key)=>{
  const raw=object?.[key];return raw===undefined||raw===null?null:Number.isFinite(Number(raw))?Number(raw):null;
}});
vm.runInContext(helper,context);
test('Composer dashboard presents recall, ETA, loss and timing without inventing missing values',()=>{
  const view=context.composerTrainingView({result:{epoch_target:12,composer_stage:'training',
    composer_runtime:{samples:947683,parameters:85500000,action_vocab_size:18454,
      candidate_recall:.9947,device:'cuda',precision:'bf16',batch_size:16,grad_accumulation:2},
    composer_metrics:{epoch:2,batch:50,total:100,samples_per_sec:80,eta_sec:120,
      pointer_accuracy:.65,loss:2.1,timing:{data_sec:4},cache:{hits:90,misses:10}}}});
  const text=JSON.stringify(view);
  assert.match(view.status,/epoch 2\/12/);assert.match(view.status,/80.*事件\/秒/);
  assert.match(text,/99\.47%/);assert.match(text,/65\.00%/);assert.match(text,/等待数据/);
  assert.match(view.note,/不能当作 GPU 内核耗时/);
  assert.ok(!text.includes('NaN'));assert.ok(!text.includes('Infinity'));
});
test('Composer preparation is visible before the first training batch',()=>{
  const view=context.composerTrainingView({result:{composer_stage:'preparation',
    composer_preparation:{completed:500,total:1787,samples:275000}}});
  assert.match(view.status,/准备数据/);assert.match(view.status,/500\/1787/);
  assert.match(JSON.stringify(view),/—/);
});

test('Recent Composer loss is derived from cumulative sums rather than plotting their tiny differences',()=>{
  const before={epoch:2,metric_batches:100,resume_skipped_batches:0,loss:10,action_loss:8};
  const now={epoch:2,metric_batches:200,resume_skipped_batches:0,loss:6,action_loss:5};
  const recent=context.composerRecentMetrics([before,now],now,100);
  assert.equal(recent.batches,100);
  assert.equal(recent.values.loss,2);
  assert.equal(recent.values.action_loss,2);
  assert.equal(recent.values.pointer_loss,null);
});

test('Recent Composer metrics never combine epochs or separate resume phases',()=>{
  const now={epoch:3,metric_batches:500,resume_skipped_batches:0,loss:4};
  const wrong=[{epoch:2,metric_batches:250,loss:10},
    {epoch:3,metric_batches:250,resume_skipped_batches:9600,loss:10}];
  assert.equal(context.composerRecentMetrics(wrong,now),null);
  const recent=context.composerRecentMetrics([...wrong,{epoch:3,metric_batches:250,loss:5}],now);
  assert.equal(recent.values.loss,3);
});

test('Composer dashboard labels cumulative means and displays actual recent window size',()=>{
  const before={epoch:3,metric_batches:250,loss:5};
  const now={epoch:3,metric_batches:500,loss:4};
  const view=context.composerTrainingView({result:{composer_metrics:now,composer_metric_history:[before,now]}});
  assert.equal(Object.fromEntries(view.cards.find(([title])=>title==='学习情况')[1])['累计总 loss'],'4');
  const card=view.cards.find(([title])=>title==='最近 250 个有效 batch');
  assert.equal(Object.fromEntries(card[1])['窗口总 loss'],'3');
});

test('Composer dashboard shows runtime capacity after automatic expansion',()=>{
  const view=context.composerTrainingView({result:{effective_settings:{action_slots:8},
    composer_runtime:{action_slots_requested:8,max_required_action_slots:9,action_slots:9}}});
  const values=Object.fromEntries(view.cards.find(([title])=>title==='模型与数据')[1]);
  assert.equal(values['动作槽位（请求下限 / 数据需求 / 实际）'],'8 / 9 / 9');
});

test('Recovery diagnostics report actual acceptance and recent weighted loss',()=>{
  const view=context.composerTrainingView({result:{composer_stage:'training',
    composer_recovery:{loss:.0325,outcomes:{accepted:1,unknown_reference_span:1}}}});
  const byLabel=Object.fromEntries(view.cards.find(([title])=>title==='学习情况')[1]);
  assert.equal(byLabel['最近恢复 loss（含权重）'],'0.0325');
  assert.equal(byLabel['最近恢复接受 / 尝试'],'1 / 2');
});

test('Finetune checkpoint inspection preserves the chosen destination architecture',()=>{
  const start=source.indexOf('function applyChartComposerCheckpointConfig(');
  const fn=source.slice(start,source.indexOf('\nfunction renderChartComposerCheckpointInspection',start));
  const elements={'#composerArchitectureProfile':{value:'audio_grounded_v3'},'#composerLensWidth':{value:'96'},
    '#composerHistoryResidualBound':{value:'.8'},'#composerHistoryMemoryRmsBound':{value:'8'}};
  const c=vm.createContext({element:id=>elements[id]});
  vm.runInContext(fn,c);
  c.applyChartComposerCheckpointConfig({model_config:{architecture_profile:'music_context_v2',lens_width:64,history_residual_bound:1}},'finetune');
  assert.equal(elements['#composerArchitectureProfile'].value,'audio_grounded_v3');
  assert.equal(elements['#composerLensWidth'].value,'96');
  assert.equal(elements['#composerHistoryResidualBound'].value,'.8');
  c.applyChartComposerCheckpointConfig({model_config:{architecture_profile:'music_context_v2',lens_width:64,history_residual_bound:1}},'resume');
  assert.equal(elements['#composerArchitectureProfile'].value,'music_context_v2');
  assert.equal(elements['#composerArchitectureProfile'].disabled,true);
  assert.equal(elements['#composerHistoryMemoryRmsBound'].value,'8');
  assert.equal(elements['#composerHistoryMemoryRmsBound'].disabled,true);
});
test('Composer workload diagnostics show losses and log1p density error as numbers',()=>{
  const view=context.composerTrainingView({result:{composer_stage:'training',composer_metrics:{
    workload_loss:.1234,curve_slope_loss:.0089,workload_rate_mae:.25}}});
  const values=view.cards.find(([title])=>title==='学习情况')[1];
  const byLabel=Object.fromEntries(values);
  assert.equal(byLabel['负荷曲线监督 loss'],'0.1234');
  assert.equal(byLabel['负荷变化斜率 loss'],'0.0089');
  assert.equal(byLabel['密度 log1p 平均绝对误差'],'0.25');
  assert.ok(!byLabel['密度 log1p 平均绝对误差'].includes('%'));
});
test('Composer runtime controls reach its training payload',()=>{
  const chunk=source.slice(source.indexOf('function chartComposerTrainingPayload('),source.indexOf('function collectValues('));
  const values={'#composerPrefetchBatches':'4','#composerSampleWorkers':'6','#composerSampleLocality':'3','#composerLogInterval':'10','#composerMaxSteps':'32',
    '#gradAccumulation':'8','#composerOptimizerImplementation':'standard',
    '#composerSymbolicDropout':'.2','#composerAudioPointerWeight':'.25','#composerProfileSteps':'8',
    '#composerArchitectureProfile':'audio_grounded_v3','#composerLensWidth':'96','#composerLensChunkSize':'16',
    '#composerHistoryResidualBound':'.8','#composerHistoryMemoryRmsBound':'12','#composerHistoryMemoryPolicy':'rms_cap_v1',
    '#composerHistoryAttentionPolicy':'math_v1','#composerHistoryDropout':'.4','#composerLedgerDropout':'.6',
    '#composerPlacementWeight':'.25','#composerPlannerWeight':'.3','#composerRecoveryInterval':'80',
    '#composerRecoverySamples':'3','#composerRecoverySteps':'4','#composerRecoveryWeight':'.2'};
  const c=vm.createContext({currentTrainingMode:()=> 'fresh',trainingSourcePaths:()=>({manifest:'data.json'}),
    selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],valueOrDefault:(id,fallback)=>values[id]??fallback,
    formDevice:()=> 'cpu',checkedOrDefault:()=>false});
  vm.runInContext(chunk,c);const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.grad_accumulation,8);assert.equal(payload.prefetch_batches,4);
  assert.equal(payload.sample_workers,6);
  assert.equal(payload.sample_locality,3);
  assert.equal(payload.symbolic_dropout,.2);assert.equal(payload.audio_pointer_weight,.25);
  assert.equal(payload.profile_steps,8);
  assert.equal(payload.log_interval,10);assert.equal(payload.max_steps,32);
  assert.equal(payload.optimizer_implementation,'standard');assert.equal(payload.export_inference,false);
  assert.equal(payload.architecture_profile,'audio_grounded_v3');assert.equal(payload.lens_width,96);
  assert.equal(payload.lens_chunk_size,16);assert.equal(payload.history_residual_bound,.8);
  assert.equal(payload.history_memory_rms_bound,12);assert.equal(payload.history_memory_policy,'rms_cap_v1');
  assert.equal(payload.history_attention_policy,'math_v1');
  assert.equal(payload.history_dropout,.4);assert.equal(payload.ledger_dropout,.6);
  assert.equal(payload.placement_weight,.25);assert.equal(payload.planner_weight,.3);
  assert.equal(payload.recovery_interval,80);assert.equal(payload.recovery_samples,3);
  assert.equal(payload.recovery_steps,4);assert.equal(payload.recovery_weight,.2);
});

test('Both Composer interfaces expose persistent grounded fields and v2 baseline',()=>{
  const main=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
  const standalone=fs.readFileSync('src/maichart/webui/static/composer.html','utf8');
  const script=fs.readFileSync('src/maichart/webui/static/composer.js','utf8');
  for(const suffix of ['ArchitectureProfile','LensWidth','LensChunkSize','HistoryResidualBound','HistoryDropout',
    'LedgerDropout','PlacementWeight','PlannerWeight','RecoveryInterval','RecoverySamples','RecoverySteps','RecoveryWeight']) {
    assert.ok(main.includes(`id="composer${suffix}"`),suffix);
    assert.ok(standalone.includes(`id="composerLab${suffix}"`),suffix);
  }
  assert.match(main,/value="music_context_v2"/);assert.match(standalone,/value="music_context_v2"/);
  assert.match(standalone,/name="pointer_history_strength"[^>]*min="0"[^>]*max="1"/);
  assert.match(script,/localStorage\.setItem\(settingsKey/);
  assert.match(main,/需要重新训练/);assert.match(standalone,/需要重新训练/);
});

test('Composer dashboard displays effective grounded and inherited settings',()=>{
  const view=context.composerTrainingView({result:{settings_policy:'resume_inherited',effective_settings:{
    architecture_profile:'audio_grounded_v3',lens_width:64,lens_chunk_size:32,history_residual_bound:1,
    history_dropout:.3,ledger_dropout:.5,placement_weight:.15,planner_weight:.2,recovery_interval:100,
    recovery_samples:2,recovery_steps:2,recovery_weight:.1}}});
  const text=JSON.stringify(view.cards);
  assert.match(text,/audio_grounded_v3/);assert.match(text,/继承 checkpoint/);assert.match(text,/100 \/ 2 \/ 2/);
});
