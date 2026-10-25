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
      if(path==='/api/status')return {training_capabilities:{composer_training_runtime:'music_context_shared_slots_v2',composer_training_controls:'rebatch_dropout_controls_v1'}};
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
