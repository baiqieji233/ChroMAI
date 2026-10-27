const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('tests/test_composer_main_controls_ui.js','utf8');
const prefix=source.slice(0,source.indexOf("test('")).replace("const {test}=require('node:test');",'');
const context=vm.createContext({require,structuredClone});
vm.runInContext(prefix+'\nglobalThis.getHarness=harness;',context);

test('Composer condition masking keeps a distinct default, payload and resume contract',async()=>{
  const {c,el,parent,setMode}=context.getHarness();
  parent.model_config.condition_mask_prob=.35;
  await c.inspectCheckpoint();
  assert.equal(el('#composerConditionMaskProb').value,'0.35');
  assert.equal(el('#composerConditionMaskProb').disabled,true);
  assert.equal(c.chartComposerTrainingPayload().condition_mask_prob,.35);
  setMode('finetune');c.updateTrainingControls();
  el('#composerConditionMaskProb').value='.1';
  assert.equal(el('#composerConditionMaskProb').disabled,false);
  assert.equal(c.chartComposerTrainingPayload().condition_mask_prob,.1);
  const app=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
  vm.runInContext(app.slice(app.indexOf('const COMPOSER_FRESH_TRAINING_DEFAULTS ='),
    app.indexOf('function setTrainingModeDefaults(')),c);
  setMode('fresh');c.applyComposerFreshTrainingDefaults();
  assert.equal(c.chartComposerTrainingPayload().condition_mask_prob,0);
  assert.match(fs.readFileSync('src/maichart/webui/static/index.html','utf8'),/条件遮挡概率（保留等级）/);
});

test('An old backend cannot silently ignore even explicit off/flat controls',async()=>{
  const {c,el,setMode}=context.getHarness();setMode('fresh');
  el('#composerFutureHorizonPolicy').value='off';
  el('#composerActionHeadPolicy').value='flat_v1';
  el('#composerStructurePolicy').value='off';
  c.api=async path=>{
    if(path==='/api/status')return {training_capabilities:{composer_training_runtime:'music_context_shared_slots_v2',
      composer_training_controls:'causal_motif_runtime_controls_v11_fresh_training_defaults',composer_structure_context:'metrical_context_v2'}};
    throw Error('An old service must not receive a training request');
  };
  await assert.rejects(c.submitTrain(),/未来音频视野、动作分类头及条件遮挡控件/);
});

test('The updated backend receives explicit off/flat selections and condition masking',async()=>{
  const {c,el,setMode,submitted}=context.getHarness();setMode('fresh');
  el('#composerFutureHorizonPolicy').value='off';el('#composerActionHeadPolicy').value='flat_v1';
  el('#composerConditionMaskProb').value='.25';
  await c.submitTrain();
  assert.equal(submitted().future_horizon_policy,'off');
  assert.equal(submitted().action_head_policy,'flat_v1');
  assert.equal(submitted().condition_mask_prob,.25);
});
