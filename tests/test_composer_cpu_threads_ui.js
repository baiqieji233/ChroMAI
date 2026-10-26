const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const existing=fs.readFileSync('tests/test_composer_main_controls_ui.js','utf8');
const prefix=existing.slice(0,existing.indexOf("test('"))
  .replace("const {test}=require('node:test');",'');
const context=vm.createContext({require,structuredClone});
vm.runInContext(prefix+'\nglobalThis.getHarness=harness;',context);

test('resume CPU threads inherits but remains editable through submit inspection',async()=>{
  const {c,el,parent,submitted}=context.getHarness();
  parent.args.cpu_threads=8;
  parent.args.sample_workers=1;
  await c.inspectCheckpoint();
  assert.equal(el('#composerCpuThreads').value,'8');
  assert.equal(el('#composerCpuThreads').disabled,false);
  assert.equal(el('#composerSampleWorkers').value,'1');
  el('#composerCpuThreads').value='2';
  await c.submitTrain();
  assert.equal(submitted().cpu_threads,2);
  assert.equal(submitted().sample_workers,1);
});

test('resume keeps an explicit sample worker runtime edit',async()=>{
  const {c,el,parent,submitted}=context.getHarness();
  parent.args.sample_workers=1;
  await c.inspectCheckpoint();
  el('#composerSampleWorkers').value='3';
  await c.submitTrain();
  assert.equal(submitted().sample_workers,3);
});

test('old checkpoint CPU threads defaults to zero and UI declares persistence/reset controls',async()=>{
  const {c,el}=context.getHarness();
  await c.inspectCheckpoint();
  assert.equal(el('#composerCpuThreads').value,'0');
  const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
  assert.match(html,/id="composerCpuThreads" type="number" min="0" max="256" step="1" value="0"/);
  const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
  assert.ok(source.includes('composerCpuThreads:0'));
  assert.ok(source.includes('causal_motif_runtime_controls_v5_selected_state'));
  assert.ok(source.includes('input[id], select[id], textarea[id]'));
});

test('finetune starts from zero and preserves an explicit runtime edit',async()=>{
  const {c,el,parent,setMode}=context.getHarness();
  parent.args.cpu_threads=8;
  await c.inspectCheckpoint();
  setMode('finetune');
  await c.inspectCheckpoint();
  assert.equal(el('#composerCpuThreads').value,'0');
  assert.equal(el('#composerCpuThreads').disabled,false);
  el('#composerCpuThreads').value='4';
  await c.inspectCheckpoint({preserveEditableConfig:true});
  assert.equal(c.chartComposerTrainingPayload().cpu_threads,4);
});
