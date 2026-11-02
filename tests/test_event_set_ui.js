const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const source=fs.readFileSync('src/maichart/webui/static/app.js','utf8');
function fn(name){const start=source.indexOf(`function ${name}(`);return source.slice(start,source.indexOf('\n}',start)+2);}
function harness(){
  const nodes=new Map();
  const element=selector=>{
    if(!nodes.has(selector)){const field={hidden:false};nodes.set(selector,{id:selector.slice(1),value:'',disabled:false,dataset:{},hasAttribute:()=>false,closest:()=>field,field});}
    return nodes.get(selector);
  };
  const context=vm.createContext({generationInspection:null,element,initializeEventSetParameterGroups(){},
    document:{querySelectorAll:selector=>selector.includes('chartComposerTrainingControls')?[element('#composerArchitectureProfile'),element('#composerCandidateCount'),element('#composerRolloutSteps')]:[]},
    valueOrDefault:(id,value)=>element(id).value||value,currentTrainingMode:()=> 'fresh',
    trainingSourcePaths:()=>({manifest:'train.json'}),selectedTrainingManifests:()=>[],selectedDatasetItems:()=>[],
    checkedOrDefault:(_id,fallback)=>fallback,formDevice:()=> 'cpu',collectCondition:()=>({level:13}),parseTempoMap:()=>null,intensityCurvePayload:()=>null});
  vm.runInContext(source.slice(source.indexOf('const EVENT_SET_LEARNING_POLICIES ='),source.indexOf('function chartComposerTrainingPayload(')),context);
  for(const name of ['isEventSetGeneration','updateEventSetControls','applyEventSetFreshDefaults','chartComposerTrainingPayload','chartComposerGenerationPayload','applyChartComposerCheckpointConfig'])vm.runInContext(fn(name),context);
  return {context,element};
}
test('event-set actual UI helpers apply defaults and hide unsupported fields',()=>{
  const {context:c,element:e}=harness();
  e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.applyEventSetFreshDefaults();c.updateEventSetControls();
  assert.equal(e('#batchSize').value,'16');assert.equal(e('#gradAccumulation').value,'4');
  assert.equal(e('#composerDModel').value,'256');assert.equal(e('#composerPrecision').value,'bf16');
  assert.equal(e('#composerCandidateCount').field.hidden,true);assert.equal(e('#composerCandidateCount').disabled,true);
  const payload=c.chartComposerTrainingPayload();assert.equal(payload.batch_size,16);assert.equal(payload.rollout_steps,16);
  assert.equal(payload.audio_grid_seconds,.01);assert.equal(e('#composerAudioGridSeconds').value,'0.01');
  assert.equal(payload.plan_weight,.1);assert.equal(e('#composerPlanWeight').value,'0.1');
  assert.equal(payload.validation_batch_size,4);assert.equal(payload.validation_max_batches,32);
  assert.equal(payload.warmup_steps,0);assert.equal(payload.prefetch_batches,2);assert.equal(payload.audio_cache_gb,1);
  assert.equal(payload.lr_schedule,'constant');assert.equal(payload.plan_blank_window_seconds,.5);
  assert.equal(payload.epochs,50);assert.equal(payload.lr,.0002);assert.equal(payload.training_window_policy,'auto');
  assert.equal(payload.max_phrase_events,64);assert.equal(payload.phrase_seam_events,2);assert.equal(payload.phrase_execution_policy,'auto');
  assert.equal(payload.phrase_modes,4);assert.equal(payload.phrase_relation_weight,.1);assert.equal(payload.detail_representation_policy,'auto');assert.equal(payload.duration_feature_policy,'auto');
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
  assert.ok(!('pointer_temperature' in payload));assert.ok('intensity_curve_mode' in payload);
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
  let payload=c.chartComposerTrainingPayload();assert.equal(payload.continuation_policy,'auto');assert.equal(payload.lane_load_policy,'auto');assert.equal(payload.recovery_policy,'auto');assert.equal(payload.prefix_augmentation_rate,.15);assert.equal(payload.generation_eval_temperature,.65);assert.equal(payload.transition_exit_weight,0);
  const legacyPolicies={transition_policy:'off',decoder_policy:'postnorm_v1',window_balance_policy:'equal_window_v1',
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
  const policies={transition_policy:'music_exit_v1',decoder_policy:'prenorm_independent_v2',window_balance_policy:'sqrt_window_v2',
    timing_group_policy:'phase_gap_v2',literal_output_policy:'type_blocks_v1',
    timing_output_policy:'absolute_grid_v1',timing_grid_audio_policy:'query_pool_v2',event_family_policy:'audio_prior_transition_v3',
    prefix_augmentation_input_policy:'musical_loop_recovery_v5',recovery_policy:'greedy_mixed_v3'};
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

test('resume retains the residual musical exit version recorded by the checkpoint',()=>{
  for(const transitionPolicy of ['music_exit_v2','music_exit_v3']){
  for(const prefixPolicy of ['musical_loop_recovery_v6','musical_loop_recovery_v7','musical_loop_recovery_v8']){
    const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
    c.currentTrainingMode=()=> 'resume';
    c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',
      model_config:{architecture_profile:'event_set_stream_v1',transition_policy:transitionPolicy,
        prefix_augmentation_input_policy:prefixPolicy,recovery_policy:'greedy_mixed_v4',
        timing_grid_audio_policy:'query_pool_v3'},args:{}},'resume');
    const payload=c.chartComposerTrainingPayload();
    assert.equal(payload.transition_policy,transitionPolicy);
    assert.equal(payload.prefix_augmentation_input_policy,prefixPolicy);
    assert.equal(payload.recovery_policy,'greedy_mixed_v4');
    assert.equal(payload.timing_grid_audio_policy,'query_pool_v3');
    assert.equal(e('#composerLearningPolicy').disabled,true);
  }
  }
});


test('advanced strategy overrides survive preset selection and weights reach payload',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  e('#eventSetPolicy_prefix_augmentation_input_policy').value='musical_loop_recovery_v8';
  e('#composerPhraseContextPolicy').value='ordered_phrase_v2';
  e('#composerLearningPolicy').value='legacy';
  e('#composerEventFamilyAudioWeight').value='.45';
  const payload=c.chartComposerTrainingPayload();
  assert.equal(payload.prefix_augmentation_input_policy,'musical_loop_recovery_v8');
  assert.equal(payload.phrase_context_policy,'ordered_phrase_v2');
  assert.equal(payload.event_family_audio_weight,.45);
});

test('blank stage temperatures inherit total and explicit zero overrides it',()=>{
  const {context:c,element:e}=harness();e('#checkpointPath').value='new.pt';
  c.generationInspection={path:'new.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1'}};
  e('#temperature').value='.4';
  let payload=c.chartComposerGenerationPayload();
  assert.equal(payload.temperature,.4);assert.ok(!('pointer_temperature' in payload));assert.ok(!('action_temperature' in payload));
  e('#composerPointerTemperature').value='0';payload=c.chartComposerGenerationPayload();
  assert.equal(payload.pointer_temperature,0);assert.ok(!('action_temperature' in payload));
});

test('finetune locks inherited backbone while resume payload preserves objective values',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  const data={schema:'maichart-composer-event-set-checkpoint-v1',model_config:{architecture_profile:'event_set_stream_v1',d_model:128,nhead:4,history_events:96,rollout_steps:24,dropout:.2,audio_grid_seconds:.02,plan_blank_window_seconds:.7},args:{seed:9,max_steps:17,recovery_interval:11,recovery_samples:3,recovery_steps:4,recovery_weight:.3}};
  c.currentTrainingMode=()=> 'finetune';c.applyChartComposerCheckpointConfig(data,'finetune');
  for(const id of ['composerDModel','composerHistoryEvents','composerRolloutSteps','composerAudioGridSeconds','composerPlanBlankWindowSeconds','composerDropout'])assert.equal(e('#'+id).disabled,true,id);
  c.currentTrainingMode=()=> 'resume';c.applyChartComposerCheckpointConfig(data,'resume');
  const payload=c.chartComposerTrainingPayload();assert.equal(payload.seed,9);assert.equal(payload.max_steps,17);assert.equal(payload.recovery_interval,11);assert.equal(payload.recovery_weight,.3);assert.equal(payload.history_events,96);
  for(const id of ['composerSeed','composerMaxSteps','composerRecoveryInterval','composerRecoveryWeight'])assert.equal(e('#'+id).disabled,true,id);
});


test('grouping moves controls with their labels and restores their original homes',()=>{
  const nodes=new Map();
  class Node {
    constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.dataset={};this.style={};this.value='';this.hidden=false;}
    set id(value){this._id=value;nodes.set('#'+value,this);} get id(){return this._id;}
    set innerHTML(html){for(const match of html.matchAll(/id="([^"]+)"/g)){const child=new Node();child.id=match[1];this.append(child);}}
    append(child){if(child.parent){child.parent.children=child.parent.children.filter(n=>n!==child);}this.children.push(child);child.parent=this;}
    before(child){this.parent.append(child);}
    after(child){this.parent.append(child);}
    closest(){return this.tagName==='LABEL'?this:this.parent;}
    addEventListener(type,listener){this.listener=listener;}
  }
  const host=new Node();host.id='chartComposerTrainingControls';
  for(const id of ['composerTrainingCommon','composerTrainingEvaluation','composerTrainingAdvanced']){const node=new Node();node.id=id;host.append(node);}
  const ids=vm.runInNewContext('('+fn('initializeEventSetParameterGroups').match(/const ids=(\{.*\});/)[1]+')');
  const homes=new Map();
  for(const id of Object.values(ids).flat()){const label=new Node('label'),control=new Node('input');control.id=id;label.append(control);nodes.get('#composerTrainingAdvanced').append(label);homes.set(id,label.parent);}
  nodes.get('#composerArchitectureProfile').value='event_set_stream_v1';
  const c=vm.createContext({element:id=>nodes.get(id)||null,valueOrDefault:(id,fallback)=>nodes.get(id)?.value||fallback,
    document:{createElement:tag=>new Node(tag),createComment:()=>new Node('comment')}});
  vm.runInContext(source.slice(source.indexOf('const EVENT_SET_LEARNING_POLICIES ='),source.indexOf('function chartComposerTrainingPayload(')),c);
  vm.runInContext(fn('initializeEventSetParameterGroups'),c);c.initializeEventSetParameterGroups();
  for(const [group,list] of Object.entries(ids))for(const id of list)assert.equal(nodes.get('#'+id).parent.parent,nodes.get(id==='composerResetLearningRate'?'#eventSetBasicAdvancedGrid':'#eventSetGroup_'+group),id);
  assert.equal(nodes.has('#eventSetPolicy_phrase_context_policy'),false);assert.equal(nodes.has('#eventSetPolicy_duration_audio_policy'),false);
  assert.ok(nodes.get('#composerPhraseContextPolicy'));assert.ok(nodes.get('#composerDurationAudioPolicy'));
  assert.equal(nodes.get('#eventSetValidationGeneration').style.gridColumn,'1 / -1');
  assert.equal(nodes.get('#eventSetGroup_validation').children.at(-1),nodes.get('#eventSetValidationGeneration'));
  assert.ok(nodes.get('#eventSetValidationGenerationGrid'));
  const advanced=nodes.get('#eventSetPolicy_prefix_augmentation_input_policy');advanced.value='musical_loop_recovery_v8';
  nodes.get('#composerLearningPolicy').value='legacy';nodes.get('#composerLearningPolicy').listener();
  assert.equal(advanced.value,'musical_loop_recovery_v8');
  nodes.get('#composerArchitectureProfile').value='audio_grounded_v3';c.initializeEventSetParameterGroups();
  for(const [id,home] of homes)assert.equal(nodes.get('#'+id).parent.parent,home,id);
});


test('advanced policy choices explain key experiments in Chinese while retaining raw contracts',()=>{
  const c=vm.createContext({});
  vm.runInContext(source.slice(source.indexOf('const EVENT_SET_LEARNING_POLICIES ='),source.indexOf('function chartComposerTrainingPayload(')),c);
  for(const key of ['event_family_policy','prefix_augmentation_input_policy','recovery_policy']){
    const options=vm.runInContext('EVENT_SET_LEARNING_POLICIES['+JSON.stringify(key)+']',c);
    for(const value of options){const label=vm.runInContext('EVENT_SET_POLICY_LABELS['+JSON.stringify(value)+']',c);assert.match(label,/[\u3400-\u9fff]/,value);}
  }
});


test('workspace event-set summary reports active inputs without old candidate slots',()=>{
  const workspace=fs.readFileSync('src/maichart/webui/static/workspace.js','utf8');
  const start=workspace.indexOf('function refresh() {'),end=workspace.indexOf('    const hint =',start);
  const values={modelFamily:'chart_composer_v1',composerArchitectureProfile:'event_set_stream_v1',epochs:'20',
    composerHistoryEvents:'96',composerRolloutSteps:'24',composerDModel:'256',composerLayers:'4',
    composerDimFeedforward:'768',composerPhraseContextPolicy:'auto',eventSetPolicy_event_family_policy:'auto',
    composerLearningPolicy:'auto',composerHoldoutFraction:'.05'};
  let summary;
  const c=vm.createContext({selectedMode:()=> 'resume',selectedDatasetItems:()=>[],
    $:()=>null,value:(id,fallback='')=>values[id]??fallback,nameOf:id=>values[id]??'',checked:()=>false,
    list:(_id,rows)=>summary=rows});
  vm.runInContext(workspace.slice(start,end)+'}',c);c.refresh();
  const rendered=summary.join('\n');assert.match(rendered,/总目标 20/);assert.match(rendered,/历史 96/);
  assert.match(rendered,/短段展开 24/);assert.match(rendered,/有序配置短句/);assert.match(rendered,/共享续写状态/);
  assert.match(rendered,/5%（开发＋最终留出合计）/);assert.doesNotMatch(rendered,/动作槽位|候选 96|额外推理权重关闭/);
});


test('fresh defaults replace known output directories while preserving explicit custom paths',()=>{
  const {context:c,element:e}=harness();
  for(const directory of ['', 'checkpoints/song_form_v2','checkpoints/chart_stream_v1',
      'checkpoints/chart_composer_v1','checkpoints/prism_composer_v16_signal',
      'checkpoints\\song_form_v2\\']){
    e('#checkpointDir').value=directory;c.applyEventSetFreshDefaults();
    assert.equal(e('#checkpointDir').value,'checkpoints/event_set_stream_v1',directory);
  }
  for(const directory of ['checkpoints/prism_composer_v37','E:/experiments/custom-run']){
    e('#checkpointDir').value=directory;c.applyEventSetFreshDefaults();
    assert.equal(e('#checkpointDir').value,directory);
  }
  c.currentTrainingMode=()=> 'resume';e('#checkpointDir').value='checkpoints/song_form_v2';
  c.applyEventSetFreshDefaults();assert.equal(e('#checkpointDir').value,'checkpoints/song_form_v2');
});


test('shared continuation and recovery target follow fresh and checkpoint policies',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.applyEventSetFreshDefaults();let payload=c.chartComposerTrainingPayload();
  assert.equal(payload.event_family_policy,'auto');assert.equal(payload.recovery_target_policy,'auto');assert.equal(payload.gap_continuation_policy,'auto');
  assert.equal(payload.transition_exit_weight,0);assert.equal(payload.event_family_audio_weight,.25);
  e('#eventSetPolicy_event_family_policy').value='shared_continuation_v1';
  e('#eventSetPolicy_recovery_target_policy').value='causal_correspondence_v1';
  payload=c.chartComposerTrainingPayload();assert.equal(payload.event_family_policy,'shared_continuation_v1');
  assert.equal(payload.recovery_target_policy,'causal_correspondence_v1');
  c.currentTrainingMode=()=> 'resume';
  c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{architecture_profile:'event_set_stream_v1',event_family_policy:'audio_state_phrase_v4'},args:{}},'resume');
  payload=c.chartComposerTrainingPayload();assert.equal(payload.event_family_policy,'audio_state_phrase_v4');assert.equal(payload.recovery_target_policy,'authored_next_v1');
});

test('gap continuation offers off and retains historical choices',()=>{
  const c=vm.createContext({});
  vm.runInContext(source.slice(source.indexOf('const EVENT_SET_LEARNING_POLICIES ='),source.indexOf('function chartComposerTrainingPayload(')),c);
  assert.deepEqual(Array.from(vm.runInContext('EVENT_SET_LEARNING_POLICIES.gap_continuation_policy',c)),['off','probability_v2','additive_v1']);
});

test('resume learning-rate reset is explicit and survives checkpoint reinspection',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.currentTrainingMode=()=> 'resume';
  const data={schema:'maichart-composer-event-set-checkpoint-v1',model_config:{architecture_profile:'event_set_stream_v1'},args:{lr:.0002,lr_schedule:'cosine',warmup_steps:100,min_lr_ratio:.1}};
  c.applyChartComposerCheckpointConfig(data,'resume');
  assert.equal(c.chartComposerTrainingPayload().reset_learning_rate,undefined);
  e('#composerResetLearningRate').checked=true;e('#learningRate').value='.0005';e('#composerLrSchedule').value='constant';e('#composerWarmupSteps').value='0';e('#composerMinLrRatio').value='1';e('#composerTimingCompute').value='dense';
  c.applyChartComposerCheckpointConfig(data,'resume',true);
  const p=c.chartComposerTrainingPayload();assert.equal(p.reset_learning_rate,true);assert.equal(p.lr,.0005);assert.equal(p.lr_schedule,'constant');assert.equal(p.warmup_steps,0);assert.equal(p.min_lr_ratio,1);assert.equal(p.timing_compute,'dense');
  c.currentTrainingMode=()=> 'fresh';assert.equal(c.chartComposerTrainingPayload().reset_learning_rate,undefined);
});


test('resume learning-rate reset requires backend capability while normal runs remain compatible',()=>{
  const c=vm.createContext({});vm.runInContext(fn('assertEventSetResumeLearningRateCapability'),c);
  const payload={architecture_profile:'event_set_stream_v1',mode:'resume',reset_learning_rate:true};
  assert.throws(()=>c.assertEventSetResumeLearningRateCapability(payload,{capabilities:{}}),/重载更新后的 WebUI/);
  assert.doesNotThrow(()=>c.assertEventSetResumeLearningRateCapability(payload,{capabilities:{event_set_resume_lr_reset:true}}));
  assert.doesNotThrow(()=>c.assertEventSetResumeLearningRateCapability({...payload,reset_learning_rate:false},{capabilities:{}}));
});

test('phrase mode temperature is visible only for trained phrase modes',()=>{
  const {context:c,element:e}=harness();
  e('#composerArchitectureProfile').value='event_set_stream_v1';
  e('#checkpointPath').value='model.pt';
  c.generationInspection={path:'model.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1',generation_capabilities:{persistent_phrase_modes:false}}};
  c.updateEventSetControls();assert.equal(e('#composerPhraseModeTemperature').field.hidden,true);
  c.generationInspection.data.generation_capabilities.persistent_phrase_modes=true;
  c.updateEventSetControls();assert.equal(e('#composerPhraseModeTemperature').field.hidden,false);
  e('#composerPhraseModeTemperature').value='0';assert.equal(c.chartComposerGenerationPayload().phrase_mode_temperature,0);
});


test('whole-song controls follow checkpoint capabilities and retain explicit locks',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  let training=c.chartComposerTrainingPayload();assert.equal(training.song_profile_policy,'auto');assert.equal(training.song_response_policy,'auto');assert.equal(training.song_profile_weight,.1);
  e('#checkpointPath').value='song.pt';c.generationInspection={path:'song.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1',generation_capabilities:{}}};
  let payload=c.chartComposerGenerationPayload();assert.ok(!('song_style_source' in payload));
  c.generationInspection.data.generation_capabilities={song_profile:true,song_response:true};
  e('#composerSongStyleSource').value='manual';e('#composerSongStyleLocks').value='{"tap_bias":0.2}';e('#composerSongResponseMode').value='2';
  payload=c.chartComposerGenerationPayload();assert.equal(payload.song_style_source,'manual');assert.equal(payload.song_style_locks.tap_bias,.2);assert.equal(payload.song_response_mode,2);
  e('#composerSongStyleLocks').value='[]';assert.throws(()=>c.chartComposerGenerationPayload(),/JSON 对象/);
});


test('whole-song controls persist and expose bounded training inputs',()=>{
  const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
  const generationFields=source.slice(source.indexOf('const GENERATION_PARAM_FIELDS ='),source.indexOf('];',source.indexOf('const GENERATION_PARAM_FIELDS =')));
  for(const id of ['composerSongStyleSource','composerSongStyleSeed','composerSongStyleLocks','composerSongResponseMode'])assert.ok(generationFields.includes(`"${id}"`));
  for(const [id,min,max] of [['composerSongProfileComponents',1,16],['composerSongResponseModes',1,8],['composerSongResponseGroupSize',2,4]])
    assert.match(html,new RegExp(`id="${id}"[^>]*min="${min}"[^>]*max="${max}"`));
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  c.currentTrainingMode=()=> 'resume';c.updateEventSetControls();assert.equal(e('#composerSongProfilePolicy').disabled,true);assert.equal(e('#composerSongProfileWeight').disabled,true);
  c.currentTrainingMode=()=> 'fresh';c.updateEventSetControls();assert.equal(e('#composerSongProfilePolicy').disabled,false);assert.equal(e('#composerSongProfileWeight').disabled,false);
});


test('response-only checkpoints expose and submit response mode without profile settings',()=>{
  const {context:c,element:e}=harness();e('#checkpointPath').value='response.pt';
  c.generationInspection={path:'response.pt',data:{valid:true,schema:'maichart-composer-event-set-checkpoint-v1',generation_capabilities:{song_profile:false,song_response:true}}};
  e('#composerSongResponseMode').value='2';c.updateEventSetControls();
  assert.equal(e('#composerSongStyleGeneration').hidden,false);
  assert.equal(e('#composerSongStyleSource').disabled,true);assert.equal(e('#composerSongResponseMode').disabled,false);
  const payload=c.chartComposerGenerationPayload();assert.equal(payload.song_response_mode,2);assert.ok(!('song_style_source' in payload));
});


test('validation source prevents stale checkpoint split from being submitted for new stages',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  e('#composerValidationManifest').value='old/data_split/dev.json';
  let payload=c.chartComposerTrainingPayload();assert.equal(payload.validation_manifest,null);assert.equal(payload.holdout_fraction,.05);
  e('#composerValidationSource').value='manual';payload=c.chartComposerTrainingPayload();assert.equal(payload.validation_manifest,'old/data_split/dev.json');
  e('#composerValidationManifest').value='';assert.throws(()=>c.chartComposerTrainingPayload(),/填写验证集清单/);
  e('#composerValidationManifest').value='custom.json';e('#composerValidationSource').value='off';payload=c.chartComposerTrainingPayload();assert.equal(payload.validation_manifest,null);assert.equal(payload.holdout_fraction,0);assert.equal(e('#composerValidationManifest').value,'custom.json');
  c.currentTrainingMode=()=> 'finetune';e('#composerValidationSource').value='manual';
  c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{},args:{validation_manifest:'parent/dev.json'}},'finetune');
  assert.equal(e('#composerValidationManifest').value,'custom.json');assert.equal(c.chartComposerTrainingPayload().validation_manifest,'custom.json');
  c.currentTrainingMode=()=> 'resume';c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{},args:{validation_manifest:'parent/dev.json'}},'resume');
  assert.equal(e('#composerValidationManifest').value,'parent/dev.json');assert.equal(c.chartComposerTrainingPayload().validation_manifest,null);
  c.currentTrainingMode=()=> 'fresh';c.applyEventSetFreshDefaults();assert.equal(e('#composerValidationSource').value,'auto');assert.equal(c.chartComposerTrainingPayload().validation_manifest,null);
});


test('validation inputs follow the selected source without changing selection on refresh',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  e('#composerValidationSource').value='auto';c.updateEventSetControls();assert.equal(e('#composerValidationManifest').disabled,true);assert.equal(e('#composerHoldoutFraction').disabled,false);
  e('#composerValidationSource').value='manual';e('#composerValidationManifest').value='explicit.json';c.updateEventSetControls();c.updateEventSetControls();
  assert.equal(e('#composerValidationSource').value,'manual');assert.equal(e('#composerValidationManifest').disabled,false);assert.equal(e('#composerHoldoutFraction').disabled,true);assert.equal(c.chartComposerTrainingPayload().validation_source,'manual');
  e('#composerValidationSource').value='off';c.updateEventSetControls();assert.equal(e('#composerValidationManifest').disabled,true);assert.equal(e('#composerHoldoutFraction').disabled,true);
  c.currentTrainingMode=()=> 'resume';c.updateEventSetControls();assert.equal(e('#composerValidationSource').disabled,true);assert.equal(e('#composerValidationManifest').disabled,true);
});


test('data worker process is an editable runtime setting on resume',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  assert.equal(c.chartComposerTrainingPayload().data_workers,0);
  e('#composerDataWorkers').value='1';c.currentTrainingMode=()=> 'resume';c.updateEventSetControls();
  assert.equal(e('#composerDataWorkers').disabled,false);assert.equal(c.chartComposerTrainingPayload().data_workers,1);
});


test('independent data process requires backend capability before submission',async()=>{
  const {context:c}=harness();const calls=[];
  Object.assign(c,{setWorkspaceMode(){},datasetUploading:false,assertTrainingSelection(){},assertDatasetBackend:async()=>{},
    trainingPayload:()=>({model_family:'chart_composer_v1',architecture_profile:'event_set_stream_v1',manifest:'train.json',data_workers:1}),
    api:async path=>{calls.push(path);return {capabilities:{event_set_stream:true},training_capabilities:{composer_detail_compute:'packed_v1'}};}});
  vm.runInContext('async '+fn('submitTrain'),c);
  await assert.rejects(c.submitTrain(),/尚未加载独立数据准备进程/);assert.deepEqual(calls,['/api/status']);
});


test('event-set rebatch is explicit and survives checkpoint reinspection',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';c.currentTrainingMode=()=> 'resume';
  e('#composerRebatchResume').checked=true;e('#batchSize').value='16';e('#gradAccumulation').value='4';
  c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{},args:{batch_size:32,grad_accumulation:2}},'resume',true);
  const payload=c.chartComposerTrainingPayload();assert.equal(payload.rebatch_resume,true);assert.equal(payload.batch_size,16);assert.equal(payload.grad_accumulation,4);
  e('#composerRebatchResume').checked=false;assert.equal(c.chartComposerTrainingPayload().rebatch_resume,undefined);
});


test('detail compute is editable during resume and passed to trainer',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  assert.equal(c.chartComposerTrainingPayload().detail_compute,'serial');
  e('#composerDetailCompute').value='packed';c.currentTrainingMode=()=> 'resume';c.updateEventSetControls();
  assert.equal(e('#composerDetailCompute').disabled,false);assert.equal(c.chartComposerTrainingPayload().detail_compute,'packed');
});


test('new composition policies are selectable and old batch order survives resume',()=>{
  const html=fs.readFileSync('src/maichart/webui/static/index.html','utf8');
  for(const policy of ['masked_deployment_joint_v3','unified_profile_response_v2','shared_ordered_v4','component_modes_v2','typed_components_v2','interleaved_microbatches_v2','song_local_v1'])assert.ok(html.includes(`value="${policy}"`),policy);
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  assert.equal(c.chartComposerTrainingPayload().batch_order_policy,'auto');
  c.currentTrainingMode=()=> 'resume';c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{},args:{}},'resume');
  assert.equal(e('#composerBatchOrderPolicy').value,'song_local_v1');assert.equal(e('#composerBatchOrderPolicy').disabled,true);
  assert.equal(c.chartComposerTrainingPayload().batch_order_policy,'song_local_v1');
});


test('unified whole-song choice binds one count and preserves legacy resume controls',()=>{
  const {context:c,element:e}=harness();e('#composerArchitectureProfile').value='event_set_stream_v1';
  e('#composerSongProfileComponents').value='6';e('#composerSongResponseModes').value='3';c.updateEventSetControls();
  assert.equal(e('#composerSongResponseModes').field.hidden,true);assert.equal(e('#composerSongResponseModes').disabled,true);
  assert.equal(c.chartComposerTrainingPayload().song_response_modes,6);
  e('#composerSongResponsePolicy').value='music_response_v1';c.updateEventSetControls();
  assert.equal(e('#composerSongResponseModes').field.hidden,false);assert.equal(c.chartComposerTrainingPayload().song_response_modes,3);
  c.currentTrainingMode=()=> 'resume';c.applyChartComposerCheckpointConfig({schema:'maichart-composer-event-set-checkpoint-v1',model_config:{song_response_policy:'music_response_v1',song_response_modes:3,song_profile_components:6},args:{}},'resume');
  assert.equal(e('#composerSongResponseModes').field.hidden,false);assert.equal(e('#composerSongResponseModes').value,'3');
});
