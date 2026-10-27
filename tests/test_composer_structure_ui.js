const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('tests/test_composer_main_controls_ui.js','utf8');
const prefix=source.slice(0,source.indexOf("test('")).replace("const {test}=require('node:test');",'');
const context=vm.createContext({require,structuredClone});
vm.runInContext(prefix+'\nglobalThis.getHarness=harness;',context);

test('Structure policy, frozen backend and contact policy inherit and lock on resume',async()=>{
  const {c,el,parent}=context.getHarness();
  Object.assign(parent.model_config,{structure_policy:'metrical_context_v1',structure_backend:'beat_this',
    structure_dropout:.2,contact_policy:'diagnostic_v1'});
  await c.inspectCheckpoint();
  const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.structure_policy,'metrical_context_v1');
  assert.equal(payload.structure_backend,'beat_this');assert.equal(payload.structure_dropout,.2);
  assert.equal(payload.contact_policy,'diagnostic_v1');
  for(const id of ['composerStructurePolicy','composerStructureBackend','composerStructureDropout','composerContactPolicy'])
    assert.equal(el('#'+id).disabled,true);
});

test('Both workspaces expose actual structure and contact controls',()=>{
  const main=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
  const lab=fs.readFileSync('src/maichart/webui/static/composer.html','utf8');
  for(const key of ['structure_policy','structure_backend','structure_dropout','contact_policy'])
    assert.match(lab,new RegExp('name="'+key+'"'));
  assert.match(main,/composerGenerationContactPolicy/);
  assert.match(lab,/不强制分段或换手/);
});
