const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
function fn(name){const start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}
function harness(){
  const nodes=new Map();
  const element=selector=>{
    if(!nodes.has(selector)){const field={hidden:false};nodes.set(selector,{id:selector.slice(1),value:'',disabled:false,dataset:{},closest:()=>field,field});}
    return nodes.get(selector);
  };
  const context=vm.createContext({generationInspection:null,element,
    document:{querySelectorAll:selector=>selector.includes('chartComposerTrainingControls')?[element('#composerArchitectureProfile'),element('#composerCandidateCount'),element('#composerRolloutSteps')]:[]},
    valueOrDefault:(id,value)=>element(id).value||value,currentTrainingMode:()=> 'fresh',
    trainingSourcePaths:()=>({manifest:'train.json'}),selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],
    checkedOrDefault:(_id,fallback)=>fallback,formDevice:()=> 'cpu',collectCondition:()=>({level:13}),parseTempoMap:()=>null,intensityCurvePayload:()=>null});
  for(const name of ['isEventSetGeneration','updateEventSetControls','applyEventSetFreshDefaults','chartComposerTrainingPayload','chartComposerGenerationPayload','applyChartComposerCheckpointConfig'])vm.runInContext(fn(name),context);
  return {context,element};
}
test('event-set actual UI helpers apply defaults and hide unsupported fields',()=>{
  const {context:c,element:e}=harness();
  e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.applyEventSetFreshDefaults();c.updateEventSetControls();
  assert.equal(e('#batchSize').value,'16');assert.equal(e('#gradAccumulation').value,'1');
  assert.equal(e('#composerDModel').value,'256');assert.equal(e('#composerPrecision').value,'bf16');
  assert.equal(e('#composerCandidateCount').field.hidden,true);assert.equal(e('#composerCandidateCount').disabled,true);
  const payload=c.chartComposerTrainingPayload();assert.equal(payload.batch_size,16);assert.equal(payload.rollout_steps,16);
  assert.equal(payload.audio_grid_seconds,.01);assert.equal(e('#composerAudioGridSeconds').value,'0.01');
  assert.equal(payload.plan_weight,.1);assert.equal(e('#composerPlanWeight').value,'0.1');
  assert.equal(payload.validation_batch_size,4);assert.equal(payload.validation_max_batches,32);
  assert.equal(payload.warmup_steps,100);assert.equal(payload.prefetch_batches,2);assert.equal(payload.audio_cache_gb,2);
  assert.equal(payload.lr_schedule,'cosine');assert.equal(payload.plan_blank_window_seconds,.5);
  assert.equal(e('#composerSampleWorkers').disabled,true);
  assert.equal(payload.architecture_profile,'event_set_stream_v1');assert.ok(!('candidate_count' in payload));
  e('#composerArchitectureProfile').value='audio_grounded_v3';c.updateEventSetControls();
  assert.equal(e('#composerCandidateCount').field.hidden,false);assert.equal(e('#composerCandidateCount').disabled,false);
});
test('new architecture submissions reject an old backend before POST',async()=>{
  const {context:c,element:e}=harness();const calls=[];
  Object.assign(c,{setWorkspaceMode(){},datasetUploading:false,assertTrainingSelection(){},assertDatasetBackend:async()=>{},
    trainingPayload:()=>({model_family:'chart_composer_v1',architecture_profile:'event_set_stream_v1',manifest:'train.json'}),
    generationPayload:()=>({model_family:'chart_composer_v1',checkpoint:'new.pt',audio:'audio.wav'}),
    api:async path=>{calls.push(path);return {};}});
  e('#checkpointPath').value='new.pt';
  c.generationInspection={path:'new.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1'}};
  vm.runInContext('async '+fn('submitTrain')+'\nasync '+fn('submitGenerate'),c);
  await assert.rejects(c.submitTrain(),/当前后台尚未加载整拍集合架构/);
  await assert.rejects(c.submitGenerate(),/当前后台尚未加载整拍集合生成/);
  assert.deepEqual(calls,['/api/status','/api/status']);
});
test('generation schema controls real filtering and restores old disabled state',()=>{
  const {context:c,element:e}=harness();e('#composerActionHistoryStrength').disabled=true;
  e('#checkpointPath').value='new.pt';
  c.generationInspection={path:'new.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1'}};
  c.updateEventSetControls();assert.equal(e('#composerActionHistoryStrength').field.hidden,true);
  const payload=c.chartComposerGenerationPayload();assert.ok(!('action_history_strength' in payload));
  assert.ok('pointer_temperature' in payload);assert.ok('intensity_curve_mode' in payload);
  c.generationInspection={data:{model_config:{architecture_profile:'audio_grounded_v3'}}};
  c.updateEventSetControls();assert.equal(e('#composerActionHistoryStrength').field.hidden,false);
  assert.equal(e('#composerActionHistoryStrength').disabled,true);
});

test('generation controls require a valid inspection of the current checkpoint',()=>{
  const {context:c,element:e}=harness();e('#checkpointPath').value='current.pt';
  c.generationInspection={path:'previous.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1'}};
  assert.equal(c.isEventSetGeneration(),false);
  c.generationInspection.path='current.pt';assert.equal(c.isEventSetGeneration(),true);
  c.generationInspection.data.valid=false;assert.equal(c.isEventSetGeneration(),false);
});

test('a late inspection cannot apply the previous checkpoint architecture',async()=>{
  const {context:c,element:e}=harness();e('#checkpointPath').value='previous.pt';
  let finish;
  const pending=new Promise(resolve=>{finish=resolve;});
  Object.assign(c,{generationInspectRequest:0,currentModelFamily:()=> 'chart_composer_v1',
    setGenerationStatus(){},updateFamilyFields(){},api:()=>pending});
  vm.runInContext('async '+fn('inspectGenerationCheckpoint'),c);
  const inspection=c.inspectGenerationCheckpoint();
  e('#checkpointPath').value='current.pt';
  finish({valid:true,family:'chart_composer_v1',schema:'maichart-composer-event-set-checkpoint-v1'});
  assert.equal(await inspection,null);assert.equal(c.generationInspection,null);
});

for(const architecture of ['event_set_stream_v1','audio_grounded_v3']) {
  test(`checkpoint status explicitly identifies ${architecture}`,async()=>{
    const {context:c,element:e}=harness();e('#checkpointPath').value='model.pt';
    const messages=[];
    Object.assign(c,{generationInspectRequest:0,currentModelFamily:()=> 'chart_composer_v1',
      element:()=>null,updateFamilyFields(){},updateEventSetControls(){},
      setGenerationStatus:text=>messages.push(text),api:async()=>({valid:true,family:'chart_composer_v1',
        epoch:3,optimizer_steps:500,model_config:{architecture_profile:architecture}})});
    vm.runInContext('async '+fn('inspectGenerationCheckpoint'),c);
    await c.inspectGenerationCheckpoint();
    assert.ok(messages.at(-1).includes(architecture));assert.match(messages.at(-1),/更新 500 步/);
  });
}

test('learning repair payload maps actual policies and checkpoint resume is locked',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.applyEventSetFreshDefaults();
  let payload=c.chartComposerTrainingPayload();assert.equal(payload.continuation_policy,'auto');assert.equal(payload.lane_load_policy,'auto');assert.equal(payload.recovery_policy,'auto');assert.equal(payload.prefix_augmentation_rate,.15);assert.equal(payload.generation_eval_temperature,.65);assert.equal(payload.transition_exit_weight,.5);
  const legacyPolicies={decoder_policy:'postnorm_v1',window_balance_policy:'equal_window_v1',
    timing_group_policy:'phase_only_v1',literal_output_policy:'flat_family_v1',
    history_window_policy:'rolling_legacy_v1',continuation_policy:'off',sensor_load_policy:'button_only_v1',
    gap_continuation_policy:'additive_v1',motion_continuation_policy:'motion_continuation_v1',lane_load_policy:'off',
    recovery_policy:'persistent_v1',timing_output_policy:'relative_delta_v1',event_family_policy:'legacy_v1',
    prefix_augmentation_input_policy:'causal_window_config_time_first_decision_v3',timing_grid_audio_policy:'off'};
  for(const key of Object.keys(legacyPolicies))assert.equal(payload[key],'auto',key);
  e('#composerLearningPolicy').value='legacy';payload=c.chartComposerTrainingPayload();assert.equal(payload.history_window_policy,'rolling_legacy_v1');assert.equal(payload.motion_continuation_policy,'motion_continuation_v1');assert.equal(payload.lane_load_policy,'off');
  for(const [key,value] of Object.entries(legacyPolicies))assert.equal(payload[key],value,key);
  c.currentTrainingMode=()=> 'resume';
  c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{motion_continuation_policy:'motion_continuation_v1'},args:{}},'resume');
  assert.equal(e('#composerLearningPolicy').disabled,true);assert.equal(e('#composerTransitionExitWeight').disabled,true);
  payload=c.chartComposerTrainingPayload();assert.equal(payload.history_window_policy,'rolling_legacy_v1');assert.equal(payload.continuation_policy,'off');assert.equal(payload.lane_load_policy,'off');assert.equal(payload.transition_exit_weight,0);assert.equal(payload.prefix_augmentation_rate,0);assert.equal(payload.recovery_policy,'persistent_v1');assert.equal(e('#composerPrefixAugmentationRate').disabled,true);assert.equal(e('#composerGenerationEvalTemperature').disabled,true);
  for(const [key,value] of Object.entries(legacyPolicies))assert.equal(payload[key],value,key);
});

test('resume preserves explicit architecture policies instead of replacing them with new defaults',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.currentTrainingMode=()=> 'resume';
  const policies={decoder_policy:'prenorm_independent_v2',window_balance_policy:'sqrt_window_v2',
    timing_group_policy:'phase_gap_v2',literal_output_policy:'type_blocks_v1',
    timing_output_policy:'absolute_grid_v1',timing_grid_audio_policy:'candidate_pool_v1',event_family_policy:'audio_prior_transition_v3'};
  c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',
    model_config:{architecture_profile:'event_set_stream_v1',...policies},args:{}},'resume');
  const payload=c.chartComposerTrainingPayload();
  for(const [key,value] of Object.entries(policies))assert.equal(payload[key],value,key);
});

test('music type budget is submitted, inherited on resume and preserved on finetune reinspection',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.applyEventSetFreshDefaults();assert.equal(c.chartComposerTrainingPayload().event_family_audio_weight,.25);
  const data={schema:'maichart-composer-event-set-checkpoint-v1',model_config:{architecture_profile:'event_set_stream_v1',event_family_policy:'audio_semantic_v2',event_family_audio_weight:.4},args:{}};
  c.applyChartComposerCheckpointConfig(data,'finetune');
  e('#composerEventFamilyAudioWeight').value='.6';e('#composerPrefixAugmentationRate').value='.2';
  c.applyChartComposerCheckpointConfig(data,'finetune',{preserveEditable:true});
  assert.equal(c.chartComposerTrainingPayload().event_family_audio_weight,.6);
  assert.equal(e('#composerPrefixAugmentationRate').value,'.2');
  c.currentTrainingMode=()=> 'resume';c.applyChartComposerCheckpointConfig(data,'resume');
  const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.event_family_audio_weight,.4);assert.equal(payload.event_family_policy,'audio_semantic_v2');
  assert.equal(e('#composerEventFamilyAudioWeight').disabled,true);
});
