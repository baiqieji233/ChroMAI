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
test('Composer runtime controls reach its training payload',()=>{
  const chunk=source.slice(source.indexOf('function chartComposerTrainingPayload('),source.indexOf('function collectValues('));
  const values={'#composerPrefetchBatches':'4','#composerLogInterval':'10','#composerMaxSteps':'32',
    '#gradAccumulation':'8','#composerOptimizerImplementation':'standard'};
  const c=vm.createContext({currentTrainingMode:()=> 'fresh',trainingSourcePaths:()=>({manifest:'data.json'}),
    selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],valueOrDefault:(id,fallback)=>values[id]??fallback,
    formDevice:()=> 'cpu',checkedOrDefault:()=>false});
  vm.runInContext(chunk,c);const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.grad_accumulation,8);assert.equal(payload.prefetch_batches,4);
  assert.equal(payload.log_interval,10);assert.equal(payload.max_steps,32);
  assert.equal(payload.optimizer_implementation,'standard');assert.equal(payload.export_inference,false);
});
