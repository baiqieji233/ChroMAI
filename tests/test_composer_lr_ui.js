const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const existing=fs.readFileSync('tests/test_composer_main_controls_ui.js','utf8');
const prefix=existing.slice(0,existing.indexOf("test('")).replace("const {test}=require('node:test');",'');
const context=vm.createContext({require,structuredClone});
vm.runInContext(prefix+'\nglobalThis.getHarness=harness;',context);

test('Scheduled resume keeps base rate and inherited schedule instead of last decayed rate',async()=>{
  const {c,el,parent}=context.getHarness();
  Object.assign(parent.args,{lr_schedule:'cosine',warmup_steps:500,min_lr_ratio:.1,lr:.0002});
  parent.optimizer_lr=.00005;
  await c.inspectCheckpoint();
  const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.lr,.0002);assert.equal(payload.lr_schedule,'cosine');
  assert.equal(payload.warmup_steps,500);assert.equal(payload.min_lr_ratio,.1);
  assert.equal(el('#composerLrSchedule').disabled,true);
});

test('Fresh optional schedule defaults remain absent until selected',()=>{
  const {c,el,setMode}=context.getHarness();setMode('fresh');
  let payload=c.chartComposerTrainingPayload();
  assert.equal(payload.lr_schedule,'auto');assert.equal(payload.warmup_steps,null);
  el('#composerLrSchedule').value='cosine';el('#composerWarmupSteps').value='10';el('#composerMinLrRatio').value='0.2';
  payload=c.chartComposerTrainingPayload();
  assert.equal(payload.lr_schedule,'cosine');assert.equal(payload.warmup_steps,10);assert.equal(payload.min_lr_ratio,.2);
});

test('Old backend cannot silently ignore a requested cosine schedule',async()=>{
  const {c,el,setMode}=context.getHarness();setMode('fresh');
  el('#composerLrSchedule').value='cosine';
  await assert.rejects(c.submitTrain(),/预热与退火/);
});
