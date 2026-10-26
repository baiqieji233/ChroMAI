const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
function fn(name){const start=source.indexOf(`function ${name}(`);return source.slice(source.slice(start-6,start)==='async '?start-6:start,source.indexOf('\n}',start)+2);}
function harness(){
  const elements=new Map();
  const el=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,disabled:false});return elements.get(id);};
  let mode='resume',submitted;
  const parent={valid:true,can_resume:true,family:'chart_composer_v1',model_config:{dropout:.25},
    args:{batch_size:96,grad_accumulation:4,lr:.0002,precision:'bf16',seed:0,max_steps:0},optimizer_lr:.00005};
  const c=vm.createContext({element:el,currentTrainingMode:()=>mode,currentModelFamily:()=> 'chart_composer_v1',
    currentCheckpointKey:()=> 'parent',checkpointInspection:{key:'',data:null},checkpointInspectRequest:0,
    currentGenerationFamily:()=> 'chart_composer_v1',datasetUploading:false,trainingStarting:false,
    window:{},updateFamilyFields:()=>{},updateDatasetButtons:()=>{},updateTrainingModeHelp:()=>{},
    updateSongFormRhythmGeometryControls:()=>{},updateSongFormQualityControls:()=>{},setCheckpointStatus:()=>{},
    renderChartComposerCheckpointInspection:()=>{},setWorkspaceMode:()=>{},assertTrainingSelection:()=>{},
    assertDatasetBackend:async()=>{},formatList:()=>'',watchTask:()=>{},
    trainingSourcePaths:()=>({manifest:'data.json'}),selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],
    valueOrDefault:(id,fallback)=>el(id).value===''?fallback:el(id).value,
    checkedOrDefault:(id)=>el(id).checked,formDevice:()=> 'cpu',
    api:async(path,payload)=>{
      if(path==='/api/composer/inspect')return structuredClone(parent);
      if(path==='/api/status')return {training_capabilities:{composer_training_runtime:'music_context_shared_slots_v2',composer_training_controls:'causal_motif_runtime_controls_v6_history_stability'}};
      if(path==='/api/composer/train'){submitted=payload;return {task_id:'fake'};}
      throw Error(path);
    }});
  for(const name of ['chartComposerTrainingPayload','applyChartComposerCheckpointConfig','updateTrainingControls','inspectCheckpoint','submitTrain'])vm.runInContext(fn(name),c);
  c.trainingPayload=c.chartComposerTrainingPayload;
  el('#trainingCheckpointPath').value='parent.pt';
  return {c,el,parent,setMode:value=>{mode=value;},submitted:()=>submitted};
}

test('Real submit inspection preserves rebatch edits, current optimizer lr and immutable dropout',async()=>{
  const {c,el,submitted}=harness();
  await c.inspectCheckpoint();
  assert.equal(el('#learningRate').value,'0.00005');
  assert.equal(el('#composerDropout').value,'0.25');assert.equal(el('#composerDropout').disabled,true);
  el('#composerRebatchResume').checked=true;c.updateTrainingControls();
  el('#batchSize').value='64';el('#gradAccumulation').value='6';el('#composerPrecision').value='fp32';
  el('#learningRate').value='0.00003';
  await c.submitTrain();
  const p=submitted();assert.equal(p.batch_size,64);assert.equal(p.grad_accumulation,6);
  assert.equal(p.rebatch_resume,true);assert.equal(p.lr,.00003);assert.equal(p.precision,'fp32');assert.equal(p.dropout,.25);
});

test('Cancelling rebatch restores saved batches; fresh and finetune cannot submit persisted rebatch',async()=>{
  const {c,el,setMode,submitted}=harness();await c.inspectCheckpoint();
  el('#composerRebatchResume').checked=true;c.updateTrainingControls();
  el('#batchSize').value='64';el('#gradAccumulation').value='6';
  el('#composerRebatchResume').checked=false;c.updateTrainingControls();
  assert.equal(el('#batchSize').value,'96');assert.equal(el('#gradAccumulation').value,'4');
  assert.equal(el('#batchSize').disabled,true);
  await c.submitTrain();assert.equal(submitted().rebatch_resume,false);assert.equal(submitted().batch_size,96);
  for(const mode of ['fresh','finetune']){
    setMode(mode);el('#composerRebatchResume').checked=true;
    assert.equal(c.chartComposerTrainingPayload().rebatch_resume,false);
    c.updateTrainingControls();assert.equal(el('#composerRebatchResume').checked,false);
    assert.equal(el('#composerDropout').disabled,mode!=='fresh');
  }
});

test('Resume inspection reports contract migration instead of claiming missing optimizer',async()=>{
  const {c,parent}=harness();parent.can_resume=false;
  parent.training_migration='Training contract changed; use finetune with a new directory';
  const result=await c.inspectCheckpoint();
  assert.equal(result.valid,false);assert.equal(result.error,parent.training_migration);
});

test('Anchor policy is inherited and locked on resume, editable for explicit finetune',async()=>{
  const {c,el,parent,setMode}=harness();
  parent.model_config.history_anchor_policy='legacy_digit_v1';
  await c.inspectCheckpoint();
  assert.equal(el('#composerHistoryAnchorPolicy').value,'legacy_digit_v1');
  assert.equal(el('#composerHistoryAnchorPolicy').disabled,true);
  assert.equal(c.chartComposerTrainingPayload().history_anchor_policy,'legacy_digit_v1');
  setMode('finetune');c.updateTrainingControls();
  assert.equal(el('#composerHistoryAnchorPolicy').disabled,false);
  el('#composerHistoryAnchorPolicy').value='parsed_heads_v2';
  await c.inspectCheckpoint({preserveEditableConfig:true});
  assert.equal(c.chartComposerTrainingPayload().history_anchor_policy,'parsed_heads_v2');
});


test('Causal and motif policies inherit and lock on resume, preserve explicit finetune edits',async()=>{
  const {c,el,parent,setMode}=harness();
  parent.model_config.action_state_policy='committed_legacy_v1';
  await c.inspectCheckpoint();
  for(const [id,want] of [['composerActionStatePolicy','committed_legacy_v1'],
      ['composerMotifMemoryPolicy','off'],['composerSlideFactorPolicy','marginal_v1'],
      ['composerRecoveryTouchLayoutPolicy','legacy_two_heads'],['composerMotifMaxEvents','16']]){
    assert.equal(el('#'+id).value,want);assert.equal(el('#'+id).disabled,true);
  }
  setMode('finetune');c.updateTrainingControls();
  el('#composerActionStatePolicy').value='candidate_prefix_v2';
  el('#composerMotifMemoryPolicy').value='audio_recurrence_archive_v1';
  el('#composerSlideFactorPolicy').value='branch_joint_v2';
  el('#composerRecoveryTouchLayoutPolicy').value='open';el('#composerMotifMaxEvents').value='24';
  await c.inspectCheckpoint({preserveEditableConfig:true});
  const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.action_state_policy,'candidate_prefix_v2');
  assert.equal(payload.motif_memory_policy,'audio_recurrence_archive_v1');
  assert.equal(payload.slide_factor_policy,'branch_joint_v2');
  assert.equal(payload.touch_layout_policy,'open');assert.equal(payload.motif_max_events,24);
});

test('History policies inherit on resume and RMS parent default is loaded only on first finetune inspection',()=>{
  const {c,el}=harness();
  const parent={model_config:{history_memory_rms_bound:12,history_memory_policy:'unbounded_v1',history_attention_policy:'auto_v1'}};
  c.applyChartComposerCheckpointConfig(parent,'finetune');
  assert.equal(el('#composerHistoryMemoryRmsBound').value,'12');
  el('#composerHistoryMemoryRmsBound').value='10';
  c.applyChartComposerCheckpointConfig(parent,'finetune',true);
  assert.equal(el('#composerHistoryMemoryRmsBound').value,'10');
  const legacy={model_config:{}};
  c.applyChartComposerCheckpointConfig(legacy,'resume');
  assert.equal(el('#composerHistoryMemoryPolicy').value,'unbounded_v1');
  assert.equal(el('#composerHistoryMemoryPolicy').disabled,true);
  assert.equal(el('#composerHistoryAttentionPolicy').value,'auto_v1');
  assert.equal(el('#composerHistoryAttentionPolicy').disabled,true);
  assert.equal(el('#composerHistoryMemoryRmsBound').value,'8');
  assert.equal(el('#composerHistoryMemoryRmsBound').disabled,true);
});

test('Policy UI has complete geometry choice and fresh auto reset without generation-reset coupling',()=>{
  const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
  assert.ok(html.includes('value="parsed_geometry_v3"'));
  for(const id of ['composerActionStatePolicy','composerMotifMemoryPolicy','composerSlideFactorPolicy','composerRecoveryTouchLayoutPolicy']){
    assert.ok(html.includes(`id="${id}"`));assert.ok(source.includes(`${id}:'auto'`));
    const gen=source.slice(source.indexOf('const GENERATION_PARAM_FIELDS ='),source.indexOf('const CONFLICT_KIND_FIELDS ='));
    assert.ok(!gen.includes(`"${id}"`));
  }
  for(const id of ['composerHistoryMemoryPolicy','composerHistoryAttentionPolicy']) {
    assert.ok(html.includes(`id="${id}"`));assert.ok(source.includes(`${id}:'auto'`));
    const gen=source.slice(source.indexOf('const GENERATION_PARAM_FIELDS ='),source.indexOf('const CONFLICT_KIND_FIELDS ='));
    assert.ok(!gen.includes(`"${id}"`));
  }
  assert.match(html,/id="composerHistoryMemoryRmsBound"[^>]+min="1"[^>]+max="64"/);
});
