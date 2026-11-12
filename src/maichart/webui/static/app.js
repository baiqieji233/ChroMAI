"use strict";
const COMPOSER_V2_FIELDS = ["audio_layers", "song_layers", "music_context", "activation_checkpointing", "epochs", "batch_size", "grad_accumulation", "cuda_memory_budget_gb", "audio_cache_gb", "lr", "lr_schedule", "precision", "cache_dir", "audio_backend", "mert_model", "validation_manifest", "prepare_device", "d_model", "layers", "nhead", "dim_feedforward", "dropout", "window_beats", "history_beats", "future_beats", "ticks_per_beat", "max_tokens", "device", "warmup_steps", "min_lr_ratio", "validation_max_batches", "checkpoint_interval_steps", "save_every", "log_interval", "seed", "max_steps", "limit_charts", "cpu_threads", "validation_source", "holdout_fraction", "refresh_events", "timing_recovery", "onset_readout", "rhythm_structure", "intensity_curve", "early_stopping_patience", "reviewed_reference_policy", "auto_quality_policy", "difficulty_source", "min_level", "max_level", "open_mdp_dir", "chartgate_dir"];
function v2FieldId(key,prefix="v2Train") { return prefix+key.split("_").map(v=>v[0].toUpperCase()+v.slice(1)).join(""); }
function v2NumericValue(raw, label, {min=-Infinity, max=Infinity, integer=false,exclusiveMin=false}={}) {
  if(raw===null || raw===undefined || String(raw).trim()==="")throw new Error(`请填写${label}。`);
  const value=Number(raw);
  if(!Number.isFinite(value) || (exclusiveMin?value<=min:value<min) || value>max || (integer && !Number.isInteger(value)))
    throw new Error(`${label}须为${integer?'整数':'有限数字'}${Number.isFinite(min)?`，${exclusiveMin?'大于':'不小于'} ${min}`:''}${Number.isFinite(max)?`，不大于 ${max}`:''}。`);
  return value;
}
function validateV2NumericInputs(scope) {
  for(const node of document.querySelectorAll(`${scope} input[type=number]`)) {
    if(node.disabled || node.checkValidity())continue;
    node.reportValidity();
    throw new Error(`请检查${node.closest('label')?.firstChild?.textContent?.trim() || '数值参数'}的取值和步长。`);
  }
}
function composerV2TrainingPayload(options={}) {
  const payload={model_family:"chart_composer_v2", ...trainingSourcePaths(options),
    manifests:selectedTrainingManifests(),mode:currentTrainingMode(),
    quality_enabled:checkedOrDefault('#v2TrainQualityEnabled',true),
    init_checkpoint:valueOrDefault("#trainingCheckpointPath","").trim()||null,
    checkpoint_dir:valueOrDefault("#checkpointDir","checkpoints/chart_composer_v2").trim()};
  const strings=new Set(["activation_checkpointing","music_context","lr_schedule","precision","cache_dir","audio_backend","mert_model","validation_manifest","prepare_device","device","validation_source","timing_recovery","onset_readout","rhythm_structure","intensity_curve","reviewed_reference_policy","auto_quality_policy","difficulty_source","open_mdp_dir","chartgate_dir"]);
  for(const key of COMPOSER_V2_FIELDS) {
    const raw=valueOrDefault("#"+v2FieldId(key),"").trim();
    if(raw!=="")payload[key]=strings.has(key)?raw:Number(raw);
  }
  // Disabled automatic-split controls must not override an explicit manual
  // validation source, including when talking to an older running server.
  if(payload.validation_source && payload.validation_source!=='auto')delete payload.holdout_fraction;
  if (payload.mode==='resume') {
    delete payload.difficulty_source;
    delete payload.min_level;
    delete payload.max_level;
    delete payload.open_mdp_dir;
    delete payload.chartgate_dir;
  }
  if (payload.difficulty_source!=='fitted') delete payload.open_mdp_dir;
  if (payload.difficulty_source!=='chartgate') delete payload.chartgate_dir;
  if (payload.mode==='resume' || !payload.quality_enabled) delete payload.auto_quality_policy;
  return payload;
}
function composerV2GenerationPayload(options={}) {
  const seed=valueOrDefault("#seed","").trim();
  const number=(selector,fallback,label,bounds)=>v2NumericValue(valueOrDefault(selector,fallback),label,bounds);
  const count=number('#v2GenerateCount',1,'生成张数',{min:1,max:32,integer:true});
  const tempoMap=parseTempoMap();
  const condition={};
  for(const [key,state] of Object.entries(v2ConditionState))if(state.override)
    condition[key]=v2NumericValue(state.value,key,{min:0,max:1});
  if(Object.keys(condition).length && !v2ConditionCapability().enabled)throw new Error("当前 checkpoint 未训练 Radar / Style 条件；请先检查支持 radar_style_v1 的 checkpoint。");
  const payload={model_family:"chart_composer_v2",checkpoint:valueOrDefault("#checkpointPath","").trim(),
    audio:valueOrDefault("#audioPath","").trim(),output:valueOrDefault("#outputPath","").trim()||null,
    bpm:valueOrDefault("#bpm","").trim()===""?null:number('#bpm',120,'BPM',{min:.001}),
    first:element('#v2GenerateAutoFirst')?.checked?null:number('#first',0,'首拍偏移'),level:number('#level',13.3,'目标等级',{min:12,max:15}),
    seed:seed===""?Math.floor(Math.random()*2147483647):v2NumericValue(seed,'随机种子',{min:0,integer:true}),
    temperature:number('#v2GenerateTemperature',.9,'采音温度',{min:0}),top_p:number('#v2GenerateTopP',.95,'采音 Top-p',{min:0,max:1,exclusiveMin:true}),
    count,parallel:count>1?number('#v2GenerateParallel',1,'同时生成数',{min:1,max:32,integer:true}):1,
    window_retries:number('#v2GenerateWindowRetries',16,'每事件提议重试次数',{min:1,max:128,integer:true}),
    phrase_search:element('#v2GeneratePhraseSearch')?.disabled?'event':valueOrDefault('#v2GeneratePhraseSearch','layout'),
    phrase_beats:number('#v2GeneratePhraseBeats',4,'提前试写长度',{min:1,max:16}),
    phrase_commit_beats:number('#v2GeneratePhraseCommitBeats',2,'每次保留长度',{min:0,exclusiveMin:true,max:16}),
    candidates:number('#v2GenerateCandidates',2,'每次比较几种写法',{min:1,integer:true}),device:valueOrDefault("#v2GenerateDevice","auto"),
    audio_device:valueOrDefault("#v2GenerateAudioDevice","auto"),inference_backend:valueOrDefault("#v2GenerateInferenceBackend","auto"),
    rollback_windows:number('#v2GenerateRollbackWindows',2,'冲突恢复次数',{min:0,max:8,integer:true}),
    exhaustion_policy:valueOrDefault('#v2GenerateExhaustionPolicy','auto'),
    forward_feasibility:element('#v2GenerateForwardFeasibility')?.disabled?'off':valueOrDefault('#v2GenerateForwardFeasibility','balanced_v1'),
    touch_reachability_policy:valueOrDefault('#v2GenerateTouchReachabilityPolicy','palm_span_v1'),
    intercept_slide_head_tap:!!element('#v2GenerateInterceptSlideHeadTap')?.checked,
    self_refine:element('#v2GenerateSelfRefine')?.disabled?'off':valueOrDefault('#v2GenerateSelfRefine','off'),
    tempo_detection:valueOrDefault('#v2GenerateTempoDetection','auto'),
    intercept_tap_on_slide:!!element('#v2GenerateInterceptTapOnSlide')?.checked,
    record_candidate_probabilities:!!element('#v2GenerateRecordCandidateProbabilities')?.checked,
    cpu_threads:number('#v2GenerateCpuThreads',0,'CPU 计算线程',{min:0,integer:true}),
    cache_dir:valueOrDefault("#v2GenerateCacheDir","cache/chart_composer_v2"),max_events:number('#v2GenerateMaxEvents',0,'最大事件数',{min:0,integer:true}),
    condition,tempo_map:tempoMap,meter_map:valueOrDefault("#formGenerationMeterMap","").trim()?parseMeterMap("#formGenerationMeterMap"):null};
  if(payload.phrase_commit_beats>payload.phrase_beats)throw new Error("每次保留长度不能超过提前试写长度。");
  if (element('#v2GenerateForwardFeasibility')?.disabled) delete payload.forward_feasibility;
  for (const [key,selector,label,bounds] of [
    ['action_temperature','#v2GenerateActionTemperature','配置温度',{min:0}],
    ['action_top_p','#v2GenerateActionTopP','配置 Top-p',{min:0,max:1,exclusiveMin:true}],
    ['onset_strength','#v2GenerateOnsetStrength','起音参考强度',{min:0,max:2}],
    ['tap_on_slide_threshold_ms','#v2GenerateTapOnSlideThresholdMs','撞尾检测阈值（ms）',{min:0,max:250}],
    ['forward_tap_on_slide_ms','#v2GenerateForwardTapOnSlideMs','采样前撞尾范围（ms）',{min:0,max:250}],
    ['touch_cohold_radius','#v2GenerateTouchCoholdRadius','Touch 同手兼顾距离',{min:60,max:260}],
  ]) {
    const raw=valueOrDefault(selector,'').trim();
    if (raw!=='' && !element(selector)?.disabled) payload[key]=v2NumericValue(raw,label,bounds);
    else if(raw==='' && key.startsWith('action_') && !element(selector)?.disabled)payload[key]=null;
  }
  const referenceSelection=element('#v2GenerateReferenceSelection');
  if(referenceSelection && !referenceSelection.disabled && referenceSelection.value==='aligned')payload.reference_selection='aligned';
  const phraseRank=element('#v2GeneratePhraseRankPolicy');
  const phraseRankPolicy=phraseRank?.disabled?'comfort_v1':valueOrDefault('#v2GeneratePhraseRankPolicy','comfort_v1');
  if(phraseRankPolicy && phraseRankPolicy!=='comfort_v1')payload.phrase_rank_policy=phraseRankPolicy;
  const tempoOctavePolicy=valueOrDefault('#v2GenerateTempoOctavePolicy','off');
  if(tempoOctavePolicy && tempoOctavePolicy!=='off')payload.tempo_octave_policy=tempoOctavePolicy;
  const downbeatAlignment=valueOrDefault('#v2GenerateDownbeatAlignment','off');
  if(downbeatAlignment && downbeatAlignment!=='off')payload.downbeat_alignment=downbeatAlignment;
  if(!options.omitCurve && typeof v2CurvePayload==='function')Object.assign(payload,v2CurvePayload());
  return payload;
}
function composerV2EvaluationPayload() {
  const checkpoint=valueOrDefault('#checkpointPath','').trim();
  if(!checkpoint)throw new Error('请先填写当前模型 checkpoint 路径。');
  const integer=(selector,fallback,label,min=0)=>v2NumericValue(valueOrDefault(selector,fallback),label,{min,integer:true});
  const windowSampling=valueOrDefault('#v2EvalWindowSampling','spread');
  if(!['spread','prefix'].includes(windowSampling))throw new Error('请选择有效的抽查位置。');
  return {model_family:'chart_composer_v2',checkpoint,manifest:valueOrDefault('#v2EvalManifest','').trim()||null,
    cache_dir:valueOrDefault('#v2GenerateCacheDir','cache/chart_composer_v2').trim(),
    limit_charts:integer('#v2EvalLimitCharts',2,'评估谱面数'),
    max_batches:integer('#v2EvalMaxBatches',8,'抽查片段数'),
    timing_rollouts:integer('#v2EvalTimingRollouts',2,'独立选时检查数'),
    cpu_threads:integer('#v2EvalCpuThreads',2,'CPU计算线程',1),window_sampling:windowSampling,
    check_training_overlap:!!element('#v2EvalCheckTrainingOverlap')?.checked};
}

async function startV2AutoQualityAudit(overrideOptions={}) {
  const button=element('#v2AutoQualityAuditButton');
  const panel=overrideOptions.panel || 'training';
  const source=autoAuditSource(panel);
  const setAuditStatus=text=>{
    if(autoAuditSource(panel).key===source.key) setAutoAuditStatus(panel, text);
  };
  const setAuditBusy=busy=>{
    if(button)button.disabled=busy;
    if(typeof document!=='undefined') {
      for(const node of document.querySelectorAll('[data-run-auto-audit]'))node.disabled=busy;
    }
  };
  if(button?.disabled)return;
  try {
    setAuditBusy(true);
    if(source.readOnly)throw new Error('请返回素材列表，再选择要检查的来源。');
    if(!source.catalog)assertTrainingSelection();
    if(!source.key)throw new Error('请先打开已预处理的数据集或勾选训练来源。');
    const limit=overrideOptions && Number.isInteger(overrideOptions.limitCharts)
      ?overrideOptions.limitCharts
      :v2NumericValue(valueOrDefault('#v2AutoQualityLimit','0'),'质检张数',{min:0,integer:true});
    const highlightNode=element('#v2AutoQualityHighlightMode');
    const handFlowNode=element('#v2AutoQualityHandFlowQuantile');
    const handFlowQuantile=handFlowNode && String(handFlowNode.value??'').trim()!==''
      ?v2NumericValue(handFlowNode.value,'手顺异常软降权比例',{min:0,max:0.25}):null;
    const extraNumeric=[
      ['tap_on_slide_ms','#v2AutoQualityTailMs','撞尾判定窗口',{min:0,max:250}],
      ['touch_cohold_radius','#v2AutoQualityTouchRadius','Touch 同手兼顾半径',{min:60,max:260}],
      ['good_quantile','#v2AutoQualityGoodQuantile','优选好段分位门槛',{min:0.5,max:0.98}],
      ['max_problem_fraction','#v2AutoQualityMaxProblemFraction','整谱排除问题占比上限',{min:0.05,max:1}],
      ['workers','#v2AutoQualityWorkers','后台体检并行进程数',{min:0,max:32,integer:true}],
    ];
    const extraSelects=[
      ['inner_muri_mode','#v2AutoQualityInnerMuriMode'],
      ['touch_reachability_policy','#v2AutoQualityTouchPolicy'],
      ['domain_pathology_mode','#v2AutoQualityDomainMode'],
      ['acoustic_audit_mode','#v2AutoQualityAcousticMode'],
    ];
    const extraPayload={};
    for(const [key,selector,label,bounds] of extraNumeric) {
      const node=element(selector);
      if(node && String(node.value??'').trim()!=='')extraPayload[key]=v2NumericValue(node.value,label,bounds);
    }
    for(const [key,selector] of extraSelects) {
      const node=element(selector);
      if(node && String(node.value??'').trim()!=='')extraPayload[key]=String(node.value).trim();
    }
    const payload={model_family:'chart_composer_v2',...source.payload,limit_charts:limit,
      tap_on_slide_mode:valueOrDefault('#v2AutoQualityTailMode','mask_hard'),
      ...extraPayload};
    if(highlightNode)payload.highlight_good_mode=highlightNode.value||'auto';
    if(handFlowQuantile!==null)payload.experimental_hand_flow_quantile=handFlowQuantile;
    const backend=await api('/api/status');
    if(!backend.training_capabilities?.composer_v2_auto_quality_audit)throw new Error('自动质检任务需要新版后台，请在当前任务结束后重启 WebUI。');
    setAuditStatus(`正在提交「${source.label}」的体检…`);
    const started=await api('/api/composer-v2/auto_quality_audit',payload);
    watchTask(started.task_id,task=>{
      setAuditBusy(false);
      const report=task.result?.composer_v2_auto_quality_audit;
      const summaryText=report
        ?`「${source.label}」${limit?'抽查预览':'自动质检'}完成，检查 ${report.audited_charts} 张。报告 ${report.report_path}`
        :(task.error||task.message||'任务已结束');
      setAuditStatus(summaryText);
      if(report) finishAutoAudit(panel, source, report, limit?'preview':'formal');
    },task=>{
      const progress=task.result?.composer_v2_auto_quality_progress;
      if(progress)setAuditStatus(`「${source.label}」正在体检 ${progress.completed}/${progress.total} 张`);
    });
  } catch(error) {
    setAuditBusy(false);
    setAuditStatus(error.message);
  }
}
async function startV2QualityEvaluation(kind='evaluate') {
  const audit=kind==='candidate_audit';
  const button=element(audit?'#v2AuditButton':'#v2EvaluateButton');
  if(button?.disabled)return;
  const input=valueOrDefault('#v2AuditInput','').trim();
  if(audit && !input)throw new Error('请填写谱面旁的 .candidate_audit.jsonl 路径。');
  const payload=audit?{input}:composerV2EvaluationPayload();
  if(button)button.disabled=true;
  const status=element('#v2EvaluationStatus');
  try {
    const backend=await api('/api/status');
    if(backend.training_capabilities?.composer_v2_quality_evaluation!==true)
      throw new Error('采音与选优评估需要更新后的后台；请在当前任务结束后重启 WebUI。');
    saveSettings();
    const data=await api('/api/composer_v2/'+kind,payload);
    if(status)status.textContent='已提交评估；进度、日志和输出报告显示在当前任务区。';
    watchTask(data.task_id,task=>{
      if(status)status.textContent=task.status==='completed'?v2EvaluationSummary(task):'评估未完成；请在当前任务区查看日志。';
    });
  } catch(error) {
    if(status)status.textContent=error.message;
    throw error;
  } finally {if(button)button.disabled=false;}
}
function v2EvaluationSummary(task) {
  const report=task.result?.composer_v2_evaluation;
  if(report) {
    const selections=report.sample_selection||[],charts=new Set(selections.map(row=>`${row.source}:${row.difficulty}`));
    const phase=report.metrics?.phase,checks=(report.joint_timing?.samples||[]).filter(row=>row.status==='measured');
    const songs=report.coverage?.selected_audio??report.per_song?.length;
    return `评估完成：${charts.size}张谱面、${selections.length}段${songs!=null?'、'+songs+'首独立音频':''}；真实前文下拍内落点正确率${phase?.accuracy!=null?(phase.accuracy*100).toFixed(1)+'%':'未测'}；独立选时${checks.length?checks.filter(row=>row.joint_exact_beat).length+'/'+checks.length+'次匹配参考':'未测'}。歌曲等权与token均值见任务报告。`;
  }
  const audit=task.result?.composer_v2_candidate_audit;
  return audit?`选优分析完成：${audit.pools?.length||0}个候选池${audit.status==='unavailable'?'，记录不足':''}。请在当前任务区查看或下载报告。`:'评估完成；请在当前任务区查看或下载输出报告。';
}
function composerV2CompletionMessage(task) {
  const batch=task.result?.composer_v2_batch;
  if(batch && Array.isArray(batch.charts)) {
    const conflicts=batch.charts.filter(x=>x.status==='completed').reduce((sum,x)=>sum+Number(x.accepted_with_conflicts_count||0),0);
    return `Composer v2 批量生成结束：成功 ${batch.completed}/${batch.count} 张${batch.failed?`，失败 ${batch.failed} 张（详见批量报告）`:''}${conflicts?`，共放行 ${conflicts} 个冲突事件`:''}。顶部预览可逐张切换，输出位于各自的 seed 目录；仍需试听与游玩检查。`;
  }
  const report=task.result?.composer_v2_report||{},timing=report.timing;
  const count=Number(report.accepted_with_conflicts_count||0);
  const references=report.rhythm_structure;
  const referenceNote=references && Number.isFinite(Number(references.context_refreshes))
    ?`，对齐参考 ${references.refreshes_with_reference}/${references.context_refreshes} 次${references.mean_reference_score!=null?`（平均可信度 ${Number(references.mean_reference_score).toFixed(2)}）`:''}`:'';
  const base=(count>0?`Composer v2 生成完成（放行 ${count} 个冲突事件）`:"Composer v2 生成完成")+referenceNote;
  if(timing?.automatic_tempo_map_applied) {
    const map=timing.effective_tempo_map||[];
    return `${base}；已应用自动变速表（${map.length}段）：${map.map(x=>`${Number(x.bpm).toFixed(3)} BPM @ ${x.beat}拍`).join(' → ')}。请试听核对变速边界与半速／倍速；详情见生成报告。`;
  }
  if(!timing||!Number.isFinite(timing.effective_bpm)||!Number.isFinite(timing.effective_first))return base+"；查看谱面与 .composer_v2.json，仍需试听与游玩检查。";
  const source=timing.tempo_source==="manual_tempo_map"?"变速表":timing.bpm_auto?"自动识别":"手动";
  const warning=timing.variable_tempo_suspected?"检测到局部变速，当前仍采用单一 BPM；请核对并填写手动 tempo map。":Array.isArray(timing.warnings)&&timing.warnings.length?"检测一致性偏低，请查看日志中的时钟警告。":"";
  return `${base}；BPM ${timing.effective_bpm.toFixed(4)}（${source}），起始偏移 ${timing.effective_first.toFixed(5)} 秒（${timing.first_auto?"自动相位":"手动"}）。${warning}${timing.bpm_auto?"请复核半速／倍速计数；":""}仍需试听与游玩检查。`;
}
function requireV2ExhaustionSupport(status,payload) {
  if(payload.exhaustion_policy==='stop' && !status.training_capabilities?.composer_v2_exhaustion_policy)
    throw new Error('严格停止选项需要更新后的后台；请在当前训练结束后重启 WebUI。默认继续生成并报告可直接使用。');
}
function requireV2SplitSamplingSupport(status,payload) {
  const customFeasibility = payload.phrase_search==='hybrid'
    || payload.phrase_rank_policy==='comfort_v2'
    || payload.exhaustion_policy==='downgrade_or_skip'
    || payload.intercept_slide_head_tap
    || (payload.forward_feasibility && payload.forward_feasibility!=='balanced_v1')
    || (payload.touch_reachability_policy && payload.touch_reachability_policy!=='palm_span_v1')
    || (payload.self_refine && payload.self_refine!=='off')
    || (payload.forward_tap_on_slide_ms!=null && payload.forward_tap_on_slide_ms!==150)
    || (payload.touch_cohold_radius!=null && payload.touch_cohold_radius!==180);
  if (customFeasibility && !status.training_capabilities?.composer_v2_feasibility_options) {
    throw new Error('这些生成选项需要新版后台，请在运行任务结束后重启 WebUI。');
  }
  if(payload.record_candidate_probabilities===true && status.training_capabilities?.composer_v2_probability_audit!==true)
    throw new Error('记录候选评估数据需要更新后的后台；请在当前任务结束后重启 WebUI。');
  if(payload.count>1 && payload.parallel>1 && !status.training_capabilities?.composer_v2_parallel_generation)
    throw new Error('同时生成多张需要新版后台，请重启 WebUI 后提交。');
  if(payload.count>1 && !status.training_capabilities?.composer_v2_batch_generation)
    throw new Error('批量生成需要更新后的后台，请在当前任务结束后重启 WebUI。');
  if (payload.tempo_detection && !status.training_capabilities?.composer_v2_variable_tempo)
    throw new Error('自动变速与全曲校准需要更新后的后台，请在当前任务结束后重启后台。');
  if (payload.intercept_tap_on_slide && !status.training_capabilities?.composer_v2_tail_guard)
    throw new Error('可选撞尾门控需要更新后的后台，请在当前任务结束后重启后台。');
  if (payload.onset_strength!=null && !status.training_capabilities?.composer_v2_onset_readout) {
    throw new Error('起音参考强度需要更新后的 WebUI 后台，请重启当前后台后提交。');
  }
  if (payload.reference_selection && !status.training_capabilities?.composer_v2_reference_selection) {
    throw new Error('对齐参考检索需要更新后的 WebUI 后台，请重启当前后台后提交。');
  }
  if ((payload.action_temperature!=null || payload.action_top_p!=null)
      && !status.training_capabilities?.composer_v2_split_sampling) {
    throw new Error('独立配置采样需要更新后的 WebUI 后台。请重启当前使用的后台后再提交。');
  }
  if(payload.phrase_search && payload.phrase_search!=='event' && !status.training_capabilities?.composer_v2_phrase_search)
    throw new Error('比较多种写法需要更新后的后台；请在当前任务结束后重启 WebUI，或选择“逐步生成（速度优先）”。');
  if(payload.phrase_rank_policy && payload.phrase_rank_policy!=='comfort_v1' && !status.training_capabilities?.composer_v2_phrase_rank_policy)
    throw new Error('候选段落似然重排需要更新后的 WebUI 后台；请在当前任务结束后重启后台，或先使用默认「手感稳定优先」。');
  if((payload.tempo_octave_policy || payload.downbeat_alignment) && !status.training_capabilities?.composer_v2_timing_calibration)
    throw new Error('可选半／倍速与强拍对齐校准需要更新后的 WebUI 后台；请在当前任务结束后重启后台，或先保持关闭。');
  if(payload.cpu_threads===0 && !status.training_capabilities?.composer_v2_cpu_auto_threads)
    throw new Error('自动分配CPU线程需要新版后台；请在当前任务结束后重启 WebUI，或先手动填写线程数。');
}
function composerV2GenerationMessage(task) {
  if(task.status!=='running')return task.message||'';
  const detailed=composerV2ProgressState(task);
  if(detailed.batch) {
    const b=detailed.batch;
    const failed=Number(b.failed)||0;
    return `${generationStageLabel(b.status==='preparing'?b.stage||'preparing_audio':'generating')} · 已结束 ${(Number(b.completed)||0)+failed}/${b.count} 张（成功 ${b.completed||0}${failed?`，失败 ${failed}`:''}） · ${Math.min(99,Math.floor(detailed.progress*100))}%`;
  }
  if(detailed.rows.length) {
    const row=detailed.rows[0];
    return `${generationStageLabel(row.stage)}${row.audio_duration_sec?` · ${generationTime(row.audio_sec)} / ${generationTime(row.audio_duration_sec)}`:''}${row.events?` · ${row.events} 个事件`:''}`;
  }
  const batch=task.result?.composer_v2_batch_progress;
  if(batch?.count>1) {
    if(batch.status==='preparing')return '正在准备音频和模型';
    if(batch.parallel>1)return `并行 ${batch.active?.length||0}/${batch.parallel} 张 · 已完成 ${batch.completed+batch.failed}/${batch.count}（成功 ${batch.completed}、失败 ${batch.failed}）`;
    return `第 ${batch.index}/${batch.count} 张 · 已成功 ${batch.completed} 张、失败 ${batch.failed} 张 · ${task.message||'正在生成'}`;
  }
  const labels={composer_v2_rollback:'正在回退并重试冲突位置',composer_v2_hold_release_repair:'正在修复 Hold 尾部冲突',
    composer_v2_sampling:'生成采样设置已确定',composer_v2_window:'已完成一个生成窗口，继续续写',composer_v2_done:'谱面生成结束，正在收集输出'};
  if(labels[task.message])return labels[task.message];
  const event=task.result?.composer_v2_event;
  if(task.message==='composer_v2_event' && event)return `已生成 ${event.events} 个事件 · 当前拍位 ${event.beat}`;
  return task.message||'等待生成进度';
}
function updateV2DifficultyControls() {
  const source=valueOrDefault('#v2TrainDifficultySource','declared');
  const fitted=source==='fitted'||source==='chartgate';
  const directory=element('#v2TrainOpenMdpDirectoryField');
  if(directory) directory.hidden=source!=='fitted';
  const path=element('#v2TrainOpenMdpDir');
  if(path) path.disabled=source!=='fitted' || currentTrainingMode()==='resume';
  const gate=element('#v2TrainChartgateOptions');
  if(gate) gate.hidden=source!=='chartgate';
  const gatePath=element('#v2TrainChartgateDir');
  if(gatePath) gatePath.disabled=source!=='chartgate' || currentTrainingMode()==='resume';
  const high=element('#v2TrainMaxLevel');
  if(high) high.placeholder=fitted?'16':'15';
  for(const id of ['#v2TrainMinLevel','#v2TrainMaxLevel']) {
    const bound=element(id);
    if(bound) bound.max=fitted?'20':'15';
  }
}

async function initializeV2Chartgate() {
  const button=element('#v2TrainChartgateSetup');
  const status=element('#v2TrainChartgateStatus');
  if(!button||!status||button.disabled) return;
  button.disabled=true;
  status.textContent='正在准备 Chart;Gate…';
  try {
    const payload={
      cache_dir:valueOrDefault('#v2TrainCacheDir','').trim(),
      chartgate_dir:valueOrDefault('#v2TrainChartgateDir','').trim()
    };
    const started=await api('/api/composer-v2/chartgate_setup',payload);
    watchTask(started.task_id, task=>{
      button.disabled=false;
      status.textContent=task.status==='completed'
        ? 'Chart;Gate 已就绪，可以开始训练。'
        : '初始化未完成。请查看任务日志。';
    });
  } catch(error) {
    button.disabled=false;
    status.textContent=error.message;
  }
}

function applyComposerV2Inspection(data) {
  if(!data?.valid)return;
  const saved={...(data.args||{}),...(data.model_config||{})};
  saved.onset_readout=data.model_config?.onset_readout_policy||'off';
  saved.rhythm_structure=data.model_config?.structure_policy||'off';
  saved.intensity_curve=data.model_config?.intensity_policy||'off';
  saved.music_context=data.model_config?.music_context_policy||'off';
  saved.reviewed_reference_policy=data.training_contract?.reviewed_reference_policy||data.args?.reviewed_reference_policy||(currentTrainingMode()==='resume'?'off':'auto');
  saved.audio_layers=data.model_config?.audio_layers??2;
  saved.song_layers=data.model_config?.song_layers??2;
  if(currentTrainingMode()==='resume') {
    const difficulty=data.training_contract?.difficulty_selection;
    saved.difficulty_source=difficulty?.source??'declared';
    saved.min_level=difficulty?.min_level??12;
    saved.max_level=difficulty?.max_level??15;
    saved.open_mdp_dir=data.args?.open_mdp_dir??'';
    saved.chartgate_dir=data.args?.chartgate_dir??'';
  }
  if(currentTrainingMode()==="resume" && !saved.validation_source)saved.validation_source=saved.validation_manifest?"manual":"off";
  if(currentTrainingMode()==='resume' && element('#v2TrainQualityEnabled'))element('#v2TrainQualityEnabled').checked=saved.quality_enabled!==false;
  for(const key of COMPOSER_V2_FIELDS) {
    if (['cuda_memory_budget_gb', 'audio_cache_gb', 'activation_checkpointing'].includes(key)) continue;
    const node=element("#"+v2FieldId(key));
    if(node && Object.prototype.hasOwnProperty.call(saved,key) && (currentTrainingMode()==="resume" || ["d_model","layers","nhead","dim_feedforward","dropout","window_beats","history_beats","future_beats","ticks_per_beat","max_tokens","audio_backend","mert_model","refresh_events"].includes(key)))node.value=saved[key]==null?'':String(saved[key]);
  }
  updateV2DifficultyControls();
}

function trainingMetricNumber(values, key) {
  const raw = values?.[key];
  if (raw === undefined || raw === null || raw === "" || typeof raw === "boolean") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function formatTrainingMetricCount(count) {
  const value = trainingMetricNumber({value: count}, "value");
  if (value === null) return "";
  return new Intl.NumberFormat("zh-CN", {
    notation: value >= 10000 ? "compact" : "standard",
    maximumFractionDigits: value >= 10000 ? 1 : 0,
  }).format(value);
}

function trainingMetricEvents(task) {
  const phaseMetrics = {};
  const epochByNumber = new Map();
  let latestProgress = null;
  const addPhase = (snapshot) => {
    if (!snapshot || !["blueprint", "layout_revision"].includes(snapshot.phase)
        || !snapshot.losses || typeof snapshot.losses !== "object") return;
    const previous = phaseMetrics[snapshot.phase];
    const nextEpoch = trainingMetricNumber(snapshot, "epoch");
    const previousEpoch = trainingMetricNumber(previous, "epoch");
    const nextCompleted = trainingMetricNumber(snapshot, "completed");
    const previousCompleted = trainingMetricNumber(previous, "completed");
    if (!previous || (nextEpoch !== null && (previousEpoch === null || nextEpoch > previousEpoch))
        || (nextEpoch === previousEpoch && nextCompleted !== null
            && (previousCompleted === null || nextCompleted >= previousCompleted))) {
      phaseMetrics[snapshot.phase] = snapshot;
    }
  };
  const addEpoch = (snapshot) => {
    if (snapshot && typeof snapshot === "object" && trainingMetricNumber(snapshot, "epoch") !== null) {
      const epochNumber = trainingMetricNumber(snapshot, "epoch");
      const previous = epochByNumber.get(epochNumber) || {};
      const previousPartial = previous.partial_epoch_metrics === true;
      const nextPartial = snapshot.partial_epoch_metrics === true;
      epochByNumber.set(epochNumber, !Object.keys(previous).length ? {...snapshot}
        : previousPartial && !nextPartial
          ? {...snapshot, partial_epoch_metrics: false}
          : !previousPartial && nextPartial ? {...previous, partial_epoch_metrics: false}
            : {...previous, ...snapshot});
    }
  };
  for (const line of Array.isArray(task?.logs) ? task.logs : []) {
    const progressMatch = String(line).match(/(?:^|\s)form_progress\s+(\{.*\})\s*$/);
    if (progressMatch) {
      try {
        latestProgress = JSON.parse(progressMatch[1]);
        addPhase(latestProgress);
      } catch (_) { /* ignore malformed log lines */ }
    }
    const epochMatch = String(line).match(/(?:^|\s)form_epoch\s+(\{.*\})\s*$/);
    if (epochMatch) {
      try { addEpoch(JSON.parse(epochMatch[1])); } catch (_) { /* ignore malformed log lines */ }
    }
  }
  const result = task?.result || {};
  const detail = result.progress_detail;
  if (detail?.losses && typeof detail.losses === "object") {
    const snapshot = {phase: detail.phase || result.phase, ...detail};
    addPhase(snapshot);
    if (!latestProgress) latestProgress = snapshot;
  }
  const savedPhases = result.training_metrics && typeof result.training_metrics === "object"
    ? result.training_metrics : {};
  for (const [phase, value] of Object.entries(savedPhases)) addPhase({phase, ...value});
  if (Array.isArray(result.metric_history)) result.metric_history.forEach(addEpoch);
  if (Array.isArray(result.training_history)) result.training_history.forEach(addEpoch);
  addEpoch(result.metrics);
  const history = [...epochByNumber.entries()]
    .sort(([epochA], [epochB]) => epochA - epochB)
    .map(entry => entry[1]).filter(value => value !== null && value !== undefined);
  const epoch = history.at(-1) || null;
  const reportedPhase = ["blueprint", "layout_revision"].includes(result.phase)
    ? result.phase : latestProgress?.phase;
  const currentEpoch = trainingMetricNumber(epoch, "epoch");
  let phase = reportedPhase && phaseMetrics[reportedPhase]
    && (currentEpoch === null || trainingMetricNumber(phaseMetrics[reportedPhase], "epoch") >= currentEpoch)
    ? reportedPhase : null;
  let phaseFromCurrentProgress = phase !== null
    && (result.phase === phase || result.progress_detail?.phase === phase);
  if (!phase && ["queued", "running", "cancelling"].includes(task?.status)) {
    const candidates = Object.entries(phaseMetrics).filter(([, value]) => value);
    candidates.sort(([phaseA, valueA], [phaseB, valueB]) => {
      const epochA = trainingMetricNumber(valueA, "epoch") ?? -1;
      const epochB = trainingMetricNumber(valueB, "epoch") ?? -1;
      if (epochA !== epochB) return epochA - epochB;
      if (phaseA === phaseB) return (trainingMetricNumber(valueA, "completed") ?? -1)
        - (trainingMetricNumber(valueB, "completed") ?? -1);
      return phaseA === "blueprint" ? -1 : 1;
    });
    phase = candidates.at(-1)?.[0] || null;
  }
  return {phaseMetrics, epoch, phase, phaseFromCurrentProgress, history, epochs: history};
}

function taskOnsetDiagnostics(task) {
  const rational = task?.result?.rational_diagnostics;
  if (rational?.schema === "song_form_rational_diagnostics_v1") return rational;
  const summary = task?.result?.onset_summary;
  if (summary?.schema === "song_form_onset_diagnostics_v1") return summary;
  if (summary?.onset_diagnostics?.schema === "song_form_onset_diagnostics_v1") return summary.onset_diagnostics;
  if (summary?.selection?.onset_diagnostics?.schema === "song_form_onset_diagnostics_v1") {
    return summary.selection.onset_diagnostics;
  }
  for (const line of Array.isArray(task?.logs) ? task.logs : []) {
    for (const eventName of ["form_onset_diagnostics", "form_onsets", "form_rational_diagnostics"]) {
      const match = String(line).match(new RegExp("(?:^|\\s)" + eventName + "\\s+(\\{.*\\})\\s*$"));
      if (!match) continue;
      try {
        const event = JSON.parse(match[1]);
        const diagnostics = eventName === "form_onsets" ? event.onset_diagnostics : event;
        if (["song_form_onset_diagnostics_v1", "song_form_rational_diagnostics_v1"].includes(diagnostics?.schema)) return diagnostics;
      } catch (_) { /* Ignore truncated or malformed log records. */ }
    }
  }
  return null;
}

function renderTaskOnsetDiagnostics(task) {
  const container = element("#taskResults");
  if (!container || task?.kind !== "form_generate") return;
  const diagnostics = taskOnsetDiagnostics(task);
  if (!diagnostics) return;

  if (diagnostics.schema === "song_form_rational_diagnostics_v1") {
    const section = document.createElement("section");
    section.className = "task-onset-diagnostics";
    const heading = document.createElement("h3"); heading.textContent = "有理数事件诊断";
    const summary = document.createElement("p");
    summary.textContent = `生成 ${diagnostics.event_count ?? 0} 个采音点；${diagnostics.empty_eos_spans ?? 0} 个计算段由模型直接结束（其中 ${diagnostics.forced_empty_eos_spans?.length ?? 0} 段当时仍有合法位置）；${diagnostics.incomplete_spans?.length ?? 0} 段搜索未完成。最长起音间隔 ${Number(diagnostics.max_gap_sec || 0).toFixed(1)} 秒；较大分母事件 ${diagnostics.large_denominator_events ?? 0} 个。大分母不自动判错，空白也可能是合理休止，需结合音乐检查。`;
    const guidance = document.createElement("p");
    guidance.textContent = "v2 联合生成拍位与配置，旧起音阈值不参与。拍位简洁偏好只作软惩罚；过早结束抑制默认关闭，可小幅上调观察，但可能增加密度，不能代替训练。";
    section.append(heading, summary, guidance); container.prepend(section); return;
  }

  const section = document.createElement("section");
  section.className = "task-onset-diagnostics";
  const heading = document.createElement("h3");
  heading.textContent = "起音空白诊断";
  section.appendChild(heading);
  const totals = diagnostics.totals || {};
  const threshold = trainingMetricNumber(diagnostics, "gap_threshold_sec");
  const gapCount = trainingMetricNumber(diagnostics, "gap_count") ?? diagnostics.gaps?.length ?? 0;
  const summary = document.createElement("p");
  summary.textContent = `记录 ${gapCount} 段${threshold !== null ? `，单段至少 ${threshold.toFixed(1)} 秒` : ""}。总候选峰 ${trainingMetricNumber(totals, "raw_local_peaks") ?? "未知"}，可听峰 ${trainingMetricNumber(totals, "audible_peaks") ?? "未知"}。这只是空白区证据摘要，不是音乐质量评分。`;
  section.appendChild(summary);

  const labels = {
    declared_rest: "标记的休止区", low_audio_activity: "音频活动较低",
    curve_suppressed: "曲线压低候选", planner_below_threshold: "规划器低于阈值",
    post_threshold_suppressed: "阈值后被抑制", unclassified: "原因未分类",
  };
  const gaps = Array.isArray(diagnostics.gaps) ? diagnostics.gaps.slice(0, 4) : [];
  if (gaps.length) {
    const list = document.createElement("ul");
    list.className = "task-onset-gap-list";
    for (const gap of gaps) {
      const item = document.createElement("li");
      const start = trainingMetricNumber(gap, "start_sec");
      const end = trainingMetricNumber(gap, "end_sec");
      const duration = trainingMetricNumber(gap, "duration_sec");
      const time = start !== null && end !== null ? `${start.toFixed(1)}–${end.toFixed(1)} 秒` : "时间未知";
      const length = duration !== null ? `（${duration.toFixed(1)} 秒）` : "";
      const rawPeaks = trainingMetricNumber(gap, "raw_peak_count");
      const audiblePeaks = trainingMetricNumber(gap, "audible_peak_count");
      item.textContent = `${time}${length} · ${labels[gap.classification] || "原因未分类"}`
        + (rawPeaks !== null || audiblePeaks !== null ? ` · 候选峰 ${rawPeaks ?? "?"} / 可听 ${audiblePeaks ?? "?"}` : "");
      list.appendChild(item);
    }
    section.appendChild(list);
  } else {
    const empty = document.createElement("p");
    empty.textContent = "没有达到诊断时长的长空白。";
    section.appendChild(empty);
  }
  if (diagnostics.gaps_truncated) {
    const truncated = document.createElement("p");
    truncated.className = "muted";
    truncated.textContent = `仅显示最长的 ${gaps.length} 段；共 ${gapCount} 段。`;
    section.appendChild(truncated);
  }
  container.prepend(section);
}

function composerTrainingView(task) {
  const result=task?.result||{},runtime=result.composer_runtime||{},metrics=result.composer_metrics||{};
  const objective=runtime.action_objective_policy??result.effective_settings?.action_objective_policy??'event_balanced_v1';
  const jointObjective=['joint_event_nll_v2','joint_event_recovery_v3'].includes(objective);
  const number=(object,key)=>trainingMetricNumber(object,key);
  const show=(value,digits=2)=>value===null||value===undefined?'—':value.toLocaleString('zh-CN',{maximumFractionDigits:digits});
  const seconds=value=>value===null?'—':`${Math.floor(value/3600)}时${Math.floor(value%3600/60)}分${Math.floor(value%60)}秒`;
  const percent=value=>value===null?'—':`${(value*100).toFixed(2)}%`;
  const conditionalCE=(object,key)=>number(object,key+'_weight')>0?show(number(object,key),4):'—';
  if((runtime.architecture_profile??result.effective_settings?.architecture_profile)==='event_set_stream_v1'||result.schema==='maichart-composer-event-set-checkpoint-v1') {
    const effective=result.effective_settings||{},coverage=runtime.codec_coverage||{},timing=metrics.timing||{},prefetch=metrics.prefetch||{};
    const phase=metrics.stage??result.composer_training_stage?.stage;
    const phaseName=phase==='symbolic'?'谱面序列预训练':phase==='audio_conditioned'?'音频条件训练':'等待训练阶段';
    const epoch=number(metrics,'epoch')??number(result.composer_training_stage||{},'epoch');
    const target=result.epoch_target??runtime.epoch_target??runtime.end_epoch??'—';
    const updates=number(metrics,'optimizer_steps')??number(result.composer_checkpoint||{},'optimizer_steps');
    let status=`整拍集合 · ${phaseName}`;
    if(epoch!==null)status+=` · epoch ${epoch}/${target} · batch ${metrics.batch??'—'}/${metrics.total??'—'}`;
    if(number(metrics,'supervised_steps_per_sec')!==null)status+=` · ${show(number(metrics,'supervised_steps_per_sec'))} 监督步骤/秒`;
    if(number(metrics,'samples_per_sec')!==null)status+=` · ${show(number(metrics,'samples_per_sec'))} 窗口序列/秒`;
    const cards=[['整拍集合训练',[
      ['当前阶段',phaseName],['当前 / 总轮数',`${epoch??'—'} / ${target}`],['参数更新步',show(updates,0)],
      ['训练窗口序列',show(number(runtime,'samples'),0)],['历史 / 展开事件数',`${runtime.history_events??effective.history_events??'—'} / ${runtime.rollout_steps??effective.rollout_steps??'—'}`],
      ['时间窗口（秒）',show(number(effective,'window_seconds'))],['batch 窗口序列 × 梯度累积',`${runtime.batch_size??effective.batch_size??'—'} × ${runtime.grad_accumulation??effective.grad_accumulation??'—'}`],
      ['设备 / 精度',`${runtime.device??effective.device??'—'} / ${runtime.precision??effective.precision??'—'}`],
      ['模型参数',number(runtime,'parameters')===null?'—':`${show(Number(runtime.parameters)/1e6)} M`]]],
      ['时间窗加权损失（本轮累计批均值）',[
        ['总 loss',show(number(metrics,'loss'),4)],['时间 loss',show(number(metrics,'time_loss'),4)],
        ['NOTE / WAIT / EOS loss',show(number(metrics,'kind_loss'),4)],['拍距 loss',show(number(metrics,'delta_loss'),4)],
        ['出音误差（分项）',show(number(metrics,'occurrence_note_nll'),4)],
        ['正常休止误差（分项）',show(number(metrics,'occurrence_wait_nll'),4)],
        ['真实前文休止误差（分项）',show(number(metrics,'occurrence_authored_rest_nll'),4)],
        ['短句事件率倍率（加权）',show(number(metrics,'occurrence_arrangement_gain'),3)],
        ['曲尾无事件误差（分项）',show(number(metrics,'occurrence_eos_nll'),4)],
        ['整拍配置 loss',show(number(metrics,'configuration_loss'),4)],['形状 loss',show(number(metrics,'shape_loss'),4)],
        ['等价相位 loss',show(number(metrics,'phase_loss'),4)],['细节 loss',show(number(metrics,'detail_loss'),4)],
        ['主预测损失（采音与配置）',show(number(metrics,'main_prediction_loss'),4)],
        ['物量计划损失（已乘权重）',show(number(metrics,'plan_weighted_loss'),4)],
        ['物量计划的固定下界（原始）',show(number(metrics,'plan_rate_floor'),4)],
        ['物量计划超额误差（原始）',show(number(metrics,'plan_rate_excess'),4)],
        ['整曲风格连续分布误差（CRPS）',show(number(metrics,'song_profile_crps'),4)],
        ['整曲风格补全 loss',show(number(metrics,'song_profile_loss'),4)],
        ['整曲风格加权 loss',show(number(metrics,'song_profile_weighted_loss'),4)],
        ['同谱远距短句组',show(number(metrics,'song_response_groups'),2)],
        ['整曲响应选择熵',show(number(metrics,'song_response_prior_entropy'),4)],
        ['联合事件目标',show(number(metrics,'unified_event_nll'),4)],
        ['平衡短句与整曲目标',show(number(metrics,'unified_balanced_variational_objective'),4)],
        ['短句选择校准',show(number(metrics,'unified_phrase_mode_prior_kl'),4)],
        ['整曲选择校准',show(number(metrics,'unified_prior_kl'),4)],
        ['整曲先验不确定性',show(number(metrics,'unified_prior_entropy'),4)],
        ['整曲后验不确定性',show(number(metrics,'unified_posterior_entropy'),4)],
        ['音频规划 loss（未乘权重）',show(number(metrics,'plan_loss'),4)],
        ['转折 loss（已乘权重）',show(number(metrics,'transition_exit_loss'),4)],
        ['自产历史恢复 loss（已乘权重）',show(number(metrics,'recovery_loss'),4)],
        ['转折目标数',show(number(metrics,'exit_count'),0)],
        ['循环历史训练 / 尝试',[show(number(metrics?.augmentation,'applied'),0),show(number(metrics?.augmentation,'attempted'),0)].join(' / ')],
        ['转折目标 NOTE / WAIT / EOS',[show(number(metrics,'exit_note_targets'),0),show(number(metrics,'exit_wait_targets'),0),show(number(metrics,'exit_eos_targets'),0)].join(' / ')],
        ['转折目标 短段 / 长段',[show(number(metrics,'exit_short_run_targets'),0),show(number(metrics,'exit_long_run_targets'),0)].join(' / ')],
        ['每秒速率 NLL（头数率 / 事件率）',show(number(metrics,'plan_rate_loss'),4)],
        ['局部休止概率 BCE',show(number(metrics,'plan_blank_loss'),4)],['当前学习率',show(number(metrics,'learning_rate'),7)],
        ['梯度范数',show(number(metrics,'last_gradient_norm'),4)],['跳过的梯度累积窗口',show(number(metrics,'skipped_windows'),0)]]],
      ['速度与主机等待',[
        ['监督步骤 / 秒',show(number(metrics,'supervised_steps_per_sec'))],['窗口序列 / 秒',show(number(metrics,'samples_per_sec'))],
        ['已消费监督步骤',show(number(metrics,'supervised_steps'),0)],['已计入均值的 batch',show(number(metrics,'metric_batches'),0)],
        ['等待数据（累计）',seconds(number(timing,'data_sec'))],['传输提交（累计）',seconds(number(timing,'transfer_sec'))],
        ['计算提交（累计）',seconds(number(timing,'compute_sec'))],['预取等待（累计）',seconds(number(prefetch,'consumer_wait_sec'))]]]];
    if(Object.keys(coverage).length)cards.push(['训练集编码覆盖率',[
      ['拟合事件',show(number(coverage,'events'),0)],['可旋转事件',show(number(coverage,'rotatable_events'),0)],
      ['精确原子回退事件',show(number(coverage,'fallback_events'),0)],['旋转分解覆盖率',percent(number(coverage,'rotation_fraction'))]]]);
    if(result.composer_validation){const values=result.composer_validation.metrics||result.composer_validation;const rows=Object.entries(values).filter(([,value])=>typeof value==='number'&&Number.isFinite(value)).map(([key,value])=>[key,show(value,4)]);if(rows.length)cards.push(['验证集（教师强制）',rows]);}
    if(result.composer_generation_validation)cards.push(['固定歌曲闭环验收',[['状态',String(result.composer_generation_validation.status??'—')],['结果',String(result.composer_generation_validation.report_path??'—')]]]);
    return {status,cards,note:'时间与配置损失保留固定时间窗权重，以 batch 窗口数 × rollout 步数归一化，再累计有效批均值；并非逐事件等权平均。规划损失单列为未乘权重的速率与休止分项之和；新版按每秒口径归一化，总 loss 不能与旧按格均值口径直接比较。总 loss 包含自产历史恢复项，单列的恢复 loss 已乘监督权重。监督步骤包含 NOTE、WAIT、EOS，窗口序列/秒不等于音符或事件/秒。计时为主机提交与等待时间，不能当作 GPU 内核耗时；教师强制损失不能证明自由生成质量。'};
  }
  const epoch=number(metrics,'epoch'),batch=number(metrics,'batch'),total=number(metrics,'total');
  const stages={preparation:'准备数据',training:'训练事件',checkpoint:'保存检查点',epoch_complete:'本轮完成'};
  let status=`Composer · ${stages[result.composer_stage]||'等待数据'}`;
  if (epoch!==null) status+=` · epoch ${epoch}/${result.epoch_target??'?'} · batch ${batch??'?'}/${total??'?'}`;
  const speed=number(metrics,'samples_per_sec'),eta=number(metrics,'eta_sec');
  if (speed!==null) status+=` · ${show(speed)} 事件/秒`;
  if (eta!==null) status+=` · 本轮 ETA ${seconds(eta)}`;
  const timing=metrics.timing||{},cache=metrics.cache||{},memory=metrics.cuda_memory||{};
  const gpu=metrics.gpu_timing||{},profiled=number(gpu,'samples')||0;
  const gpuMean=key=>profiled?show(number(gpu,key)/profiled,2)+' ms':'未计时';
  const cards=[['模型与数据',[
    ['有效训练事件',show(number(runtime,'samples'),0)],['模型参数',`${show(number(runtime,'parameters')===null?null:Number(runtime.parameters)/1e6)} M`],
    ['动作词表（含 STOP）',show(number(runtime,'action_vocab_size'),0)],['候选召回率',percent(number(runtime,'candidate_recall'))],
    ['动作槽位（请求下限 / 数据需求 / 实际）',`${show(number(runtime,'action_slots_requested'),0)} / ${show(number(runtime,'max_required_action_slots'),0)} / ${show(number(runtime,'action_slots'),0)}`],
    ['设备 / 精度',`${runtime.device||'—'} / ${runtime.precision||'—'}`],
    ['实际 CPU 算子线程',`${runtime.torch_cpu_threads??'—'}`],
    ['batch × 梯度累积',`${runtime.batch_size??'—'} × ${runtime.grad_accumulation??1}`]]],
    ['学习情况',[
      ['累计总 loss',show(number(metrics,'loss'),4)],['选位 loss',show(number(metrics,'pointer_loss'),4)],
      [jointObjective?'配置联合 NLL':'动作 loss',show(number(metrics,'action_loss'),4)],['结束 loss',show(number(metrics,'eos_loss'),4)],
      ['配置首槽 CE（共同标尺）',show(number(metrics,'action_first_slot_ce'),4)],
      ['配置旧标尺 CE',show(number(metrics,'action_legacy_ce'),4)],
      ['后续真音符 CE（不含 STOP）',conditionalCE(metrics,'action_later_nonstop_ce')],
      ['后续真音符有效权重',show(number(metrics,'action_later_nonstop_ce_weight'),2)],
      ['含恢复的参数更新目标',number(metrics,'optimized_updates')>0?show(number(metrics,'optimized_loss'),4):'—'],
      ['当前学习率',show(number(metrics,'learning_rate'),7)],
      ['音频配置首槽 CE',show(number(metrics,'audio_action_first_slot_ce'),4)],
      ['未来意图 loss',show(number(metrics,'future_loss'),4)],['候选选位准确率',percent(number(metrics,'pointer_accuracy'))],
      ['负荷曲线监督 loss',show(number(metrics,'workload_loss'),4)],
      ['负荷变化斜率 loss',show(number(metrics,'curve_slope_loss'),4)],
      ['独立音频选位 loss',show(number(metrics,'audio_pointer_loss'),4)],
      [jointObjective?'音频配置联合 NLL':'独立音频动作 loss',show(number(metrics,'audio_action_loss'),4)],
      ['短窗占用监督 loss',show(number(metrics,'placement_loss'),4)],
      ['四拍块规划 loss',show(number(metrics,'planner_loss'),4)],
      ['最近恢复 loss（含权重）',show(number(result.composer_recovery||{},'loss'),4)],
      ['恢复配置 loss（未乘权重）',show(number(result.composer_recovery?.components_unweighted||{},'action_loss'),4)],
      ['最近恢复接受 / 尝试',`${result.composer_recovery?.outcomes?.accepted??'—'} / ${result.composer_recovery?Object.values(result.composer_recovery.outcomes||{}).reduce((a,b)=>a+Number(b),0):'—'}`],
      ['恢复自产历史长度',(result.composer_recovery?.episode?.generated_history_events??[]).join(' / ')||'—'],
      ['历史窗口负荷 loss',show(number(metrics,'trailing_workload_loss'),4)],
      ['密度 log1p 平均绝对误差',show(number(metrics,'workload_rate_mae'),4)]]],
    ['速度与等待',[
      ['batch / 秒',show(number(metrics,'batches_per_sec'))],['事件 / 秒',show(speed)],['本轮剩余',seconds(eta)],
      ['等待数据（累计）',seconds(number(timing,'data_sec'))],['传输提交（累计）',seconds(number(timing,'transfer_sec'))],
      ['计算提交（累计）',seconds(number(timing,'compute_sec'))],['保存（累计）',seconds(number(timing,'save_sec'))],
      ['音频取数（后台累计）',seconds(number(cache.cpu_timing||{},'audio_lookup_sec'))],
      ['标签表构建（后台累计）',seconds(number(cache.cpu_timing||{},'song_table_sec'))],
      ['持续动作（后台累计）',seconds(number(cache.cpu_timing||{},'action_state_sec'))]]],
    ['缓存与显存',[
      ['驻留缓存命中 / 未命中',`${show(number(cache,'hits'),0)} / ${show(number(cache,'misses'),0)}`],
      ['映射库复用 / 新建',`${show(number(cache,'mapped_bank_hits'),0)} / ${show(number(cache,'bank_builds'),0)}`],
      ['映射库写入失败',show(number(cache,'bank_write_errors'),0)],
      ['逻辑特征 MiB / 有效预算',`${show(number(cache,'bytes')===null?null:Number(cache.bytes)/1024**2,0)} / ${show((number(cache,'effective_budget_bytes')??number(cache,'budget_bytes'))===null?null:Number(cache.effective_budget_bytes??cache.budget_bytes)/1024**2,0)}`],
      ['已分配 / 已保留 GiB',`${show(number(memory,'allocated_gib'))} / ${show(number(memory,'reserved_gib'))}`],
      ['峰值已分配 GiB',show(number(memory,'peak_allocated_gib'))]]],
    ['CUDA 分段计时',[
      ['计时 batch',show(profiled,0)],['传输均值',gpuMean('h2d_ms')],
      ['前向均值',gpuMean('forward_ms')],['反向均值',gpuMean('backward_ms')],['更新均值',gpuMean('update_ms')]]]];
  if (result.composer_validation && typeof result.composer_validation==='object') {
    const rows=Object.entries(result.composer_validation.metrics||result.composer_validation)
      .filter(([,value])=>typeof value==='number' && Number.isFinite(value))
      .map(([key,value])=>[key,show(value,4)]);
    if (rows.length) cards.push(['验证集指标（教师强制）',rows]);
  }
  if (result.composer_generation_validation) {
    const validation=result.composer_generation_validation;
    cards.push(['固定歌曲闭环验收',[
      ['状态',String(validation.status??'—')],
      ['结果',String(validation.report_path??'—')]]]);
  }
  const preparation=result.composer_preparation||{};
  const recent=composerRecentMetrics(result.composer_metric_history||[],metrics);
  if (recent) cards.splice(2,0,[`最近 ${recent.batches} 个有效 batch`,[
    ['窗口总 loss',show(recent.values.loss,4)],['窗口选位 loss',show(recent.values.pointer_loss,4)],
    [jointObjective?'窗口配置联合 NLL':'窗口动作 loss',show(recent.values.action_loss,4)],[jointObjective?'窗口音频配置联合 NLL':'窗口音频动作 loss',show(recent.values.audio_action_loss,4)],
    ['窗口配置首槽 CE',show(recent.values.action_first_slot_ce,4)],['窗口配置旧标尺 CE',show(recent.values.action_legacy_ce,4)],
    ['窗口后续真音符 CE',conditionalCE(recent.values,'action_later_nonstop_ce')]]]);
  const effective=result.effective_settings||runtime;
  if (effective.architecture_profile) {
    cards.push(['音频锚定训练参数',[
      ['架构',effective.architecture_profile],['短窗音频宽度 / 分块',`${effective.lens_width??'—'} / ${effective.lens_chunk_size??'—'}`],
      ['配置学习目标',objective],
      ['普通 dropout',`${runtime.dropout??effective.dropout??'—'}`],
      ['历史键位锚定',runtime.history_anchor_policy??effective.history_anchor_policy??'旧模型兼容'],
      ['历史记忆幅度策略',runtime.history_memory_policy??effective.history_memory_policy??'旧模型兼容'],
      ['历史记忆 RMS 上限',runtime.history_memory_rms_bound??effective.history_memory_rms_bound??8],
      ['历史跨注意力后端',runtime.history_attention_policy??effective.history_attention_policy??'旧模型兼容'],
      ['按键计时通道',runtime.head_timing_policy??effective.head_timing_policy??'merged_pads_v1'],
      ['跨拍相对键位',runtime.cross_event_relation_policy??effective.cross_event_relation_policy??'off'],
      ['未来音频视野',runtime.future_horizon_policy??effective.future_horizon_policy??'off'],
      ['相位呼应先验',runtime.echo_prior_policy??effective.echo_prior_policy??'off'],
      ['选时分布',runtime.timing_distribution_policy??effective.timing_distribution_policy??'hierarchical_v1'],
      ['逐按钮历史负担',runtime.button_load_policy??effective.button_load_policy??'off'],
      ['动作分类头结构',runtime.action_head_policy??effective.action_head_policy??'flat_v1'],
      ['配置质量架构',runtime.configuration_quality_policy??effective.configuration_quality_policy??'off'],
      ['训练样本组织',runtime.sample_sequence_policy??effective.sample_sequence_policy??'independent_events_v1'],
      ['条件遮挡概率（保留等级）',runtime.condition_mask_prob??effective.condition_mask_prob??0],
      ['恢复采样策略',runtime.recovery_sampling_policy??effective.recovery_sampling_policy??'calibrated_kind_touch_rings_v2'],
      ['恢复选位 / 动作温度',`${runtime.recovery_pointer_temperature??effective.recovery_pointer_temperature??1} / ${runtime.recovery_action_temperature??effective.recovery_action_temperature??.85}`],
      ['持续动作状态',runtime.action_state_policy??effective.action_state_policy??'committed_legacy_v1'],
      ['动机记忆 / 事件上限',`${runtime.motif_memory_policy??effective.motif_memory_policy??'off'} / ${runtime.motif_max_events??effective.motif_max_events??16}`],
      ['Slide分支因子',runtime.slide_factor_policy??effective.slide_factor_policy??'marginal_v1'],
      ['结构条件 / 后端',`${runtime.structure_policy??effective.structure_policy??'off'} / ${runtime.structure_backend??effective.structure_backend??'native'}`],
      ['接触预算',runtime.contact_policy??effective.contact_policy??'off'],
      ['恢复Touch布局',runtime.touch_layout_policy??effective.touch_layout_policy??'legacy_two_heads'],
      ['轮中重分 batch',runtime.rebatch_resume===undefined?(result.rebatch_resume_requested===undefined?'—':result.rebatch_resume_requested?'已请求，待运行确认':'未请求'):(runtime.rebatch_resume?'开启':'关闭')],
      ['历史残差上限',`${effective.history_residual_bound??'—'}`],
      ['历史 / 台账遮挡',`${effective.history_dropout??'—'} / ${effective.ledger_dropout??'—'}`],
      ['占用 / 规划监督权重',`${effective.placement_weight??'—'} / ${effective.planner_weight??'—'}`],
      ['恢复间隔 / 样本 / 步数',`${effective.recovery_interval??'—'} / ${effective.recovery_samples??'—'} / ${effective.recovery_steps??'—'}`],
      ['恢复监督权重',`${effective.recovery_weight??'—'}`],
      ['参数来源',result.settings_policy==='resume_inherited'?'继承 checkpoint':'当前请求']]]);
  }
  if (result.composer_stage==='preparation') {
    status+=` · ${preparation.completed??preparation.completed_songs??0}/${preparation.total??preparation.total_songs??'?'} · 已收集 ${preparation.samples??preparation.charts??0} 项`;
  }
  return {status,cards,note:'学习情况中的 loss 与准确率为本轮、本次运行的累计均值；最近窗口单独列出，歌曲切换时可以波动。不同配置学习目标的总loss不可直接比较，请同时看配置首槽CE与旧标尺CE。计时为主机累计计时，异步 CUDA 可把等待计入后续步骤，不能当作 GPU 内核耗时。后台 CPU 细分与等待时间有重叠，不能相加；映射库文件字节不是实际 RAM 占用。'};
}

function composerRecentMetrics(history,current,width=250) {
  if(current?.recent_metrics && Number(current.recent_metric_batches)>0)
    return {batches:Number(current.recent_metric_batches),values:current.recent_metrics};
  const count=trainingMetricNumber(current,'metric_batches');
  if (count===null||count<=0) return null;
  let base=null,baseCount=0;
  if (count>width) {
    for (const row of history) {
      const n=trainingMetricNumber(row,'metric_batches');
      if (n===null||n<0||n>=count||Number(row.epoch)!==Number(current.epoch)
          ||Number(row.resume_skipped_batches??0)!==Number(current.resume_skipped_batches??0)) continue;
      if (base===null||Math.abs(n-(count-width))<Math.abs(baseCount-(count-width))) {base=row;baseCount=n;}
    }
    if (base===null) return null;
  }
  const batches=count-baseCount,values={};
  for (const key of ['loss','pointer_loss','action_loss','audio_action_loss','action_first_slot_ce','action_legacy_ce']) {
    const now=trainingMetricNumber(current,key),before=base?trainingMetricNumber(base,key):0;
    const value=now===null||before===null?null:(now*count-before*baseCount)/batches;
    values[key]=value!==null&&Number.isFinite(value)&&value>=-1e-3?Math.max(0,value):null;
  }
  return {batches,values};
}

function composerV2TrainingView(task) {
  const result=task.result||{},runtime=result.composer_v2_runtime||{},prepared=result.composer_v2_prepare||{};
  const progress=result.composer_v2_progress||{};
  let metrics=result.composer_v2_training_metrics||(progress.stage==='training'?progress:null);
  let complete=result.composer_v2_epoch_metrics||(progress.stage==='epoch_complete'?progress:null);
  let epochStart=result.composer_v2_epoch_start,validation=result.composer_v2_validation;
  let position=result.composer_v2_position;
  const timingSamples=[],saveSamples=[];
  const epochs=new Map();
  const addEpoch=row=>{if(trainingMetricNumber(row,'epoch')!==null)epochs.set(Number(row.epoch),row);};
  if(Array.isArray(result.composer_v2_epoch_history))result.composer_v2_epoch_history.forEach(addEpoch);
  // Older running servers keep only the last record per prefix. Their bounded
  // log ring still supplies recent completed epochs and the last training mean.
  let lastPrefix='',lastStage='';
  for(const line of task.logs||[]) {
    const match=String(line).match(/(?:^|\s)(composer_v2_progress|composer_v2_prepare|composer_v2_checkpoint|composer_v2_runtime|composer_v2_position)\s+(\{.*\})\s*$/);
    if(!match)continue;
    let row;try{row=JSON.parse(match[2]);}catch(_){continue;}
    if(!row || typeof row!=='object' || Array.isArray(row))continue;
    lastPrefix=match[1];lastStage=row.stage||'';
    if(match[1]==='composer_v2_position')position=row;
    if(match[1]==='composer_v2_checkpoint'&&trainingMetricNumber(row,'save_sec')!==null)saveSamples.push(Number(row.save_sec));
    if(match[1]!=='composer_v2_progress')continue;
    if(row.stage==='epoch_start')epochStart=row;
    if(row.stage==='validation')validation=row;
    if(row.stage==='training')timingSamples.push(row);
    if(row.stage==='training' && (!metrics || Number(row.optimizer_steps)>=Number(metrics.optimizer_steps||0)))metrics=row;
    if(row.stage==='epoch_complete'){addEpoch(row);if(!complete || Number(row.epoch)>=Number(complete.epoch))complete=row;}
  }
  if(complete)addEpoch(complete);
  const history=[...epochs.values()].sort((a,b)=>Number(a.epoch)-Number(b.epoch));
  const stage=result.composer_v2_stage||(lastPrefix==='composer_v2_checkpoint'?'checkpoint':lastStage)||progress.stage||runtime.stage||'queued';
  const target=trainingMetricNumber(result,'epoch_target')??trainingMetricNumber(runtime,'epoch_target')??trainingMetricNumber(result.effective_settings||{},'epochs');
  const finished=trainingMetricNumber(complete,'epoch')||0;
  const activeEpoch=Math.max(trainingMetricNumber(metrics,'epoch')||0,trainingMetricNumber(epochStart,'epoch')||0,trainingMetricNumber(position,'epoch')||0);
  const activeMetrics=[position,metrics,epochStart].filter(row=>Number(row?.epoch)===activeEpoch).sort((a,b)=>Number(b.batch||0)-Number(a.batch||0))[0];
  const batch=trainingMetricNumber(activeMetrics,'batch'),total=trainingMetricNumber(activeMetrics,'total')??trainingMetricNumber(runtime,'batches_per_epoch');
  const fraction=total>0 && batch!==null?Math.max(0,Math.min(1,batch/total)):0;
  const done=Math.max(finished,activeEpoch>0?activeEpoch-1+fraction:0);
  const amount=target>0?Math.max(0,Math.min(1,done/target)):0;
  const epoch=Math.max(finished,activeEpoch);
  const preparing=stage==='preparation' || (!metrics && !complete && !epochStart && !position);
  const phaseNames={queued:'等待启动',preparation:'准备谱面与音频',epoch_start:'训练中',training:'训练中',validation:'验证中',checkpoint:'保存检查点',epoch_complete:'本轮完成',failed:'运行失败',nonfinite_update_skipped:'已跳过非有限更新'};
  let status=phaseNames[stage]||'准备训练';
  if (preparing && prepared.total) {
    const label = {charts:'检查谱面源', vocabulary:'建立词表与事件索引', audio:'音频'}[prepared.stage] || '音频';
    status += ` · ${label} ${prepared.completed ?? 0}/${prepared.total}（准备阶段）`;
    if (prepared.stage === 'audio' && prepared.cache_builds !== undefined) {
      status += ` · 复用 ${prepared.cache_hits ?? 0}，新建 ${prepared.cache_builds}`;
    }
  }
  if(epoch>0)status+=` · 第 ${epoch}/${target||'?'} 轮`;
  if(activeMetrics && activeEpoch>finished && total>0)status+=` · 本轮 batch ${batch}/${total}（${(fraction*100).toFixed(1)}%）`;
  if(target>0 && !preparing)status+=` · 总计划 ${(amount*100).toFixed(1)}%`;
  if(['completed','cancelled','failed'].includes(task.status))status=`${({completed:'本次任务已结束',cancelled:'任务已中断',failed:'任务失败'})[task.status]} · ${status}${task.status==='completed' && amount<1?'；原定轮数尚未全部跑完':''}`;
  const show=(value,digits=4)=>value===null||value===undefined?'未记录':Number(value).toFixed(digits);
  const n=(object,key)=>trainingMetricNumber(object,key);
  const latestEpoch=complete?`第 ${complete.epoch} 轮`:'尚无完整轮次';
  const trainEpoch=n(complete,'epoch_train_loss') ?? (complete?.validation_loss==null?n(complete,'loss'):null);
  const settings=result.effective_settings||{},config=runtime.model_config||{};
  const duration=value=>{if(value==null||!Number.isFinite(value))return '正在估算';const sec=Math.max(0,Math.ceil(value)),minutes=Math.ceil(sec/60);return sec>=3600?`${Math.floor(minutes/60)} 小时 ${minutes%60} 分钟`:sec>=60?`${minutes} 分钟`:`${sec} 秒`;};
  const preparationEta=n(prepared,'eta_sec')??(prepared.completed>=3&&prepared.total>prepared.completed&&prepared.elapsed_sec>=3?
    prepared.elapsed_sec*(prepared.total-prepared.completed)/prepared.completed:null);
  const mean=values=>values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
  const terminal=['completed','cancelled','failed','archived'].includes(task.status);
  let batchSeconds=null,epochEta=null,taskEta=null;
  const processed=n(activeMetrics,'epoch_processed_batches'),elapsed=n(activeMetrics,'epoch_elapsed_sec');
  if(processed>=3&&elapsed>0)batchSeconds=elapsed/processed;
  // Old tasks can estimate from two real log samples. The elapsed difference
  // excludes earlier epochs and never counts skipped batches on resume.
  if(batchSeconds===null&&metrics) {
    const samples=timingSamples.filter(row=>Number(row.epoch)===Number(metrics.epoch)&&Number(row.batch)<Number(metrics.batch)&&
      Number(row.elapsed_sec)<Number(metrics.elapsed_sec)).slice(-6);
    const first=samples[0];
    if(first&&metrics.batch-first.batch>=3)batchSeconds=(metrics.elapsed_sec-first.elapsed_sec)/(metrics.batch-first.batch);
  }
  if(batchSeconds===null)batchSeconds=mean(history.slice(-3).filter(row=>row.epoch_processed_batches>0&&row.epoch_train_sec>0)
    .map(row=>row.epoch_train_sec/row.epoch_processed_batches));
  const validationMean=mean(history.slice(-3).map(row=>n(row,'validation_sec')).filter(value=>value!==null&&value>=0));
  const validationKnown=validationMean!==null||runtime.validation_windows===0||settings.validation_source==='off';
  const saveMean=mean(saveSamples.slice(-5))||0;
  if(batchSeconds!==null&&total>0) {
    const currentDone=activeEpoch<=finished;
    const trainingLeft=['validation','epoch_complete'].includes(stage)||currentDone?0:Math.max(0,total-(batch||0))*batchSeconds;
    const validationElapsed=stage==='validation'&&Number.isFinite(validation?.started_unix)?Math.max(0,Date.now()/1000-validation.started_unix):0;
    const validationLeft=currentDone?0:Math.max(0,(validationMean||0)-validationElapsed);
    epochEta=trainingLeft+validationLeft;
    if(target>0)taskEta=epochEta+Math.max(0,target-Math.max(activeEpoch,finished))*(total*batchSeconds+(validationMean||0)+saveMean)+saveMean;
    const attempts=n(activeMetrics,'update_attempts');
    if(settings.max_steps>0&&attempts!==null)taskEta=Math.min(taskEta??Infinity,Math.max(0,settings.max_steps-attempts)*Number(settings.grad_accumulation||1)*batchSeconds+saveMean);
  }
  if(terminal){epochEta=null;taskEta=null;}
  const etaPending=terminal?'本次任务已结束':stage==='validation'&&!validationKnown?'首次验证，正在估算':'正在估算';
  const epochEtaText=epochEta===null||stage==='validation'&&!validationKnown?etaPending:`约 ${duration(epochEta)}`;
  const taskEtaText=taskEta===null?etaPending:`约 ${duration(taskEta)}${!validationKnown?'（验证用时待首轮记录）':''}`;
  if(!terminal&&!preparing&&epochEta!==null&&!(stage==='validation'&&!validationKnown))status+=` · 本轮预计剩余 ${duration(epochEta)}`;
  if(!terminal&&preparing&&preparationEta!==null)status+=` · 本阶段预计剩余 ${duration(preparationEta)}`;
  const coverage=complete?.validation_coverage||runtime.validation_coverage;
  const stepRecords=[n(result,'optimizer_steps'),n(metrics,'optimizer_steps'),n(position,'optimizer_steps'),n(complete,'optimizer_steps'),n(result.composer_v2_checkpoint,'optimizer_steps')].filter(value=>value!==null);
  const bestBasis=complete?.best_basis||runtime.best_basis||result.composer_v2_checkpoint?.best_basis||'';
  const cards=[['学习指标',[
    ['累计训练 loss（本次运行，加权 token 均值）',show(n(metrics,'loss'))],
    ['时间 token loss（同一累计范围）',show(n(metrics,'time_loss'))],
    ['拍内落点 loss',show(n(metrics,'phase_loss'))],
    ['配置 token loss（同一累计范围）',show(n(metrics,'component_loss'))],
    [`完整轮训练 loss（${latestEpoch}）`,show(trainEpoch)],
    [`验证 loss（${latestEpoch}）`,show(n(complete,'validation_loss'))],
    [`验证拍内落点 loss（${latestEpoch}）`,show(n(complete,'validation_phase_loss'))],
    ['best 选择依据',bestBasis.startsWith('validation')?'独立开发集验证 loss':bestBasis.startsWith('training')?'训练 loss（未配置验证集）':'等待记录'],
      ['每轮验证预算',settings.validation_max_batches===0?'完整开发集':settings.validation_max_batches?`最多 ${settings.validation_max_batches} batch`:'以训练日志为准'],
      ['实际验证覆盖',coverage?`${coverage.selected_windows}/${coverage.total_windows} 片段 · ${coverage.selected_audio}/${coverage.total_audio} 首音频`:'旧进程未记录'],
      ['验证抽查方式',coverage?.policy==='audio_balanced_spread_v1'?'歌曲均衡、曲内分散':coverage?.policy==='manifest_prefix_v1'?'旧版清单前缀':'以训练日志为准']]],
    ['训练进度',[
      ['本轮预计剩余',preparing?'准备数据中':epochEtaText],
      ['整个任务预计剩余',preparing?'准备完成后估算':taskEtaText],
      ['准备阶段预计剩余',preparing?duration(preparationEta):'已完成'],
      ['计划轮数 / 已完成轮数',`${target??'未知'} / ${finished}`],
      ['参数更新步',show(stepRecords.length?Math.max(...stepRecords):null,0)],
      ['当前学习率',n(metrics,'lr')===null?'未记录':Number(metrics.lr).toExponential(3)],
      ['累计 token',show(n(metrics,'token_count'),0)],
      ['数据准备',prepared.total?`${prepared.completed??0}/${prepared.total} ${{charts:'份谱面源',vocabulary:'项索引',audio:'份音频'}[prepared.stage]||'项'}`:'等待记录']]],
    ['实际运行配置',[
      ['设备 / 精度',`${runtime.device||'未记录'} / ${runtime.precision||'未记录'}`],
      ['模型宽度 / 层数',`${config.d_model??'未记录'} / ${config.layers??'未记录'}`],
      ['可训练参数',show(n(runtime,'trainable_parameters'),0)],
      ['有效 batch（片段 / 更新）',show(n(runtime,'effective_batch'),0)],
      ['CUDA 显存预算',runtime.cuda_memory_budget_gb === 0 ? '默认分配' : n(runtime,'cuda_memory_budget_gb') === null ? '未记录' : `${runtime.cuda_memory_budget_gb} GiB`],
      ['音频缓存内存预算',n(runtime,'audio_cache_gb') === null ? '未记录' : `${runtime.audio_cache_gb} GiB`],
      ['训练 / 验证谱面数',`${runtime.train_charts??'旧日志未记录'} / ${runtime.validation_charts??'旧日志未记录'}`],
      ['训练 / 验证计算片段',`${runtime.windows??'未记录'} / ${runtime.validation_windows??'未记录'}`],
      ['取批顺序',({audio_local_length_bucket_v2:'按音频分组后打散',length_bucket_seed_epoch_v1:'全库打散'})[runtime.batch_order]||'未记录'],
      ['节奏前文纠错',runtime.timing_recovery?.policy==='causal_rhythm_v1'?`已启用 · ${runtime.timing_recovery.views} 个附加片段`:'关闭'],
      ['音频特征',runtime.audio_backend||'未记录']]],
    ['吞吐与计时（本次运行累计）',[
      ['token / 秒',show(n(metrics,'tokens_per_sec'),1)],['片段 / 秒',show(n(metrics,'samples_per_sec'),1)],
      ['已运行 / 秒',show(n(metrics,'elapsed_sec'),1)],['等待数据 / 秒',show(n(metrics,'data_sec'),1)],
      ['主机提交与边界等待 / 秒',show(n(metrics,'compute_sec'),1)],
      ['音频特征映射打开次数',show(n(metrics?.audio_banks,'bank_opens'),0)],
      ['保留音频映射（份 / GiB）',n(metrics?.audio_banks,'mapped_bytes')===null?'未记录':`${metrics.audio_banks.open_banks} / ${(metrics.audio_banks.mapped_bytes/1024**3).toFixed(2)}`],
      ['峰值张量显存 / GiB',n(metrics,'gpu_peak_bytes')===null?'未记录':(metrics.gpu_peak_bytes/1024**3).toFixed(2)]]]];
  if(config.intensity_policy==='rate_curve_v1' || metrics?.curve_loss!=null || complete?.validation_curve_loss!=null) {
    const errors=complete?.validation_curve_mae||[];
    cards.push(['疏密曲线学习',[
      ['曲线训练 loss',show(n(metrics,'curve_loss'))],['反向传播总目标',show(n(metrics,'objective_loss'))],
      ['曲线验证 loss',show(n(complete,'validation_curve_loss'))],
      ['出音频率平均误差（次/秒）',show(errors[0])],['同拍额外物量平均误差（个/秒）',show(errors[1])]]]);
  }
  return {status,progress:amount,knownTarget:target>0,indeterminate:preparing,cards,history,
    note:'训练累计均值与完整轮训练、验证 loss 分开显示。验证预算为0时检查完整开发集；正数按记录的方式抽查。曲线误差单独显示，谱面质量仍需试听。'};
}

function renderComposerV2TrainingMetrics(task,panel,source,groups) {
  const view=composerV2TrainingView(task);
  const signature=JSON.stringify([task.task_id,view]);
  if(panel._v2MetricSignature===signature)return;
  panel._v2MetricSignature=signature;panel.hidden=false;groups.replaceChildren();
  if(source)source.textContent=view.note;
  for(const [title,rows] of view.cards) {
    const card=document.createElement('section');card.className='training-metric-group training-metric-card';
    const heading=document.createElement('h4');heading.textContent=title;
    const list=document.createElement('dl');list.className='training-metric-list';
    for(const [label,value] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;list.append(dt,dd);}
    card.append(heading,list);groups.appendChild(card);
  }
  if(view.history.length) {
    const card=document.createElement('section');card.className='training-metric-trends';
    const heading=document.createElement('h4');heading.textContent='最近完整轮次（训练与验证分列）';card.appendChild(heading);
    const table=document.createElement('table');table.className='v2-epoch-table';
    const header=document.createElement('tr');
    for(const label of ['轮次','本轮训练 loss','开发集验证 loss']){const th=document.createElement('th');th.textContent=label;header.appendChild(th);}
    const thead=document.createElement('thead');thead.appendChild(header);table.appendChild(thead);
    const body=document.createElement('tbody');
    for(const row of view.history.slice(-10)) {
      const tr=document.createElement('tr');
      const train=trainingMetricNumber(row,'epoch_train_loss') ?? (row.validation_loss==null?trainingMetricNumber(row,'loss'):null);
      const validation=trainingMetricNumber(row,'validation_loss');
      for(const value of [row.epoch,train===null?'未记录':train.toFixed(4),validation===null?'未记录':validation.toFixed(4)]){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td);}
      body.appendChild(tr);
    }
    table.appendChild(body);card.appendChild(table);groups.appendChild(card);
  }
}

function renderTaskTrainingMetrics(task) {
  const panel = element("#taskTrainingMetrics");
  const source = element("#taskTrainingMetricsSource");
  const groups = element("#taskTrainingMetricsGroups");
  if (!panel || !groups) return;
  const sharedNote=panel.querySelector?.('.task-training-metrics-note');
  if(sharedNote)sharedNote.hidden=task?.kind==='composer_v2_train';
  if(task?.kind==="composer_v2_train") {
    panel._composerMetricSignature=null;
    renderComposerV2TrainingMetrics(task,panel,source,groups);return;
  }
  panel._v2MetricSignature=null;
  if (task?.kind==='composer_train') {
    const view=composerTrainingView(task);
    const history=(task.result?.composer_metric_history||[]).filter(row=>Number.isFinite(Number(row.loss)));
    // Polls usually repeat identical metrics until the next trainer log line.
    // Compare the small displayed view, never stringify the multi-MB task.
    const signature=JSON.stringify([task.task_id,view,history.map(row=>[row.epoch,row.batch,row.loss])]);
    if (panel._composerMetricSignature===signature) return;
    panel._composerMetricSignature=signature;
    groups.replaceChildren();
    panel.hidden=false;
    if (source) source.textContent=view.note;
    for (const [title,rows] of view.cards) {
      const card=document.createElement('section');card.className='training-metric-group training-metric-card';
      const heading=document.createElement('h4');heading.textContent=title;card.appendChild(heading);
      const list=document.createElement('dl');list.className='training-metric-list';
      for (const [label,value] of rows) {
        const term=document.createElement('dt'),description=document.createElement('dd');
        term.textContent=label;description.textContent=value;list.append(term,description);
      }
      card.appendChild(list);groups.appendChild(card);
    }
    if (history.length>1) {
      const card=document.createElement('section');card.className='training-metric-trends';
      const heading=document.createElement('h4');heading.textContent='累计总 loss 走势（每轮重新计均值，不代表自由生成质量）';card.appendChild(heading);
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
      svg.setAttribute('viewBox','0 0 300 64');svg.setAttribute('class','training-metric-sparkline');
      svg.setAttribute('role','img');svg.setAttribute('aria-label','最近训练总损失走势');
      const values=history.map(row=>Number(row.loss));const low=Math.min(...values),span=Math.max(...values)-low||1;
      const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');
      line.setAttribute('points',values.map((v,i)=>`${i*300/(values.length-1)},${60-(v-low)/span*56}`).join(' '));
      line.setAttribute('fill','none');line.setAttribute('stroke','currentColor');line.setAttribute('stroke-width','2');
      svg.appendChild(line);card.appendChild(svg);groups.appendChild(card);
    }
    return;
  }
  panel._composerMetricSignature=null;
  groups.replaceChildren();
  if (task?.kind === "chart_stream_train") {
    const metrics = task.result?.stream_metrics;
    panel.hidden = !metrics;
    if (source) source.textContent = metrics ? `连续谱面训练 · epoch ${metrics.epoch} · 本轮累计 batch 均值；不代表自由生成质量` : "等待训练指标";
    if (!metrics) return;
    const list = document.createElement("dl"); list.className = "training-metric-list";
    for (const [key, label] of [["loss","总 loss"],["token_loss","事件 token loss"],
      ["token_accuracy","token 准确率"],["plan_loss","连续写作计划 loss"],
      ["position_accuracy","拍位字符准确率"],["note_accuracy","配置字符准确率"],
      ["whole_position_accuracy","完整拍位准确率"],["whole_configuration_accuracy","完整配置准确率"],
      ["whole_event_accuracy","完整事件准确率"],["event_balance_loss","事件均衡 loss"],
      ["alignment_loss","谱师选音关联 loss"],["alignment_accuracy","音频关联节点准确率"],
      ["alignment_coverage","选音监督覆盖率"],
      ["eos_accuracy","整曲结束准确率"],["eos_targets","已见真实结束目标"],["eos_token_loss","整曲结束 token loss"],
      ["cycle_exit_accuracy","原谱循环退出准确率"],["cycle_exit_targets","已见循环退出目标"],["cycle_exit_loss","循环退出 loss"],
      ["event_decision_loss","续写与结束 loss"],["curve_loss","强度曲线 loss"],
      ["batches_per_sec","本轮 batch / 秒"],["recent_batches_per_sec","近期 batch / 秒"],
      ["data_wait_sec","等待数据 / 秒"],["transfer_host_sec","传输提交 / 秒"],
      ["compute_host_sec","主机计算计时 / 秒"],["save_sec","保存 / 秒"],
      ["optimizer_steps","参数更新步"]]) {
      const value = trainingMetricNumber(metrics,key);
      if (value === null) continue;
      const term=document.createElement("dt"), definition=document.createElement("dd");
      term.textContent=label;
      definition.textContent=(key.endsWith("accuracy") || key.endsWith("coverage")) ? `${(value*100).toFixed(2)}%`
        : key === "optimizer_steps" || key.endsWith("targets") ? String(value) : value.toFixed(4);
      list.append(term,definition);
    }
    groups.append(list);
    return;
  }
  if (task?.kind !== "form_train") {
    panel.hidden = true;
    return;
  }

  const events = trainingMetricEvents(task);
  const livePhase = events.phase;
  const live = ["queued", "running", "cancelling"].includes(task.status) && livePhase !== null;
  const latestPhaseEpoch = trainingMetricNumber(events.phaseMetrics[livePhase], "epoch");
  const latestSummaryEpoch = trainingMetricNumber(events.epoch, "epoch");
  const archivedProgress = task.status === "archived" && livePhase !== null
    && (latestSummaryEpoch === null || (latestPhaseEpoch !== null && latestPhaseEpoch > latestSummaryEpoch)
        || events.epoch?.partial_epoch_metrics === true);
  const phaseSnapshot = (live || archivedProgress) ? events.phaseMetrics[livePhase] : null;
  const liveValues = phaseSnapshot?.losses || {};
  const liveCounts = phaseSnapshot?.metric_counts || {};
  const epochValues = events.epoch || {};
  const epochCounts = epochValues.metric_counts || {};
  const epochNumber = trainingMetricNumber(epochValues, "epoch");
  const phaseNames = {blueprint: "全曲规划", layout_revision: "谱面配置"};
  const phasePrefix = events.phaseFromCurrentProgress ? "当前阶段" : "最近阶段记录";
  const dataLabel = live
    ? `${phasePrefix}：${phaseNames[livePhase]}${trainingMetricNumber(phaseSnapshot, "epoch") !== null ? ` · epoch ${trainingMetricNumber(phaseSnapshot, "epoch")}` : ""}`
    : archivedProgress && phaseSnapshot
      ? `历史末条进度：${phaseNames[livePhase] || "训练"} · epoch ${trainingMetricNumber(phaseSnapshot, "epoch") ?? "?"}（本轮未完成）`
    : epochNumber !== null
      ? `最近记录：epoch ${epochNumber}${epochValues.partial_epoch_metrics ? "（部分 epoch）" : ""}`
      : "等待训练指标日志";
  if (source) source.textContent = dataLabel;
  const formatValue = (value, kind) => value === null ? "—" : kind === "text" ? String(value) : kind === "params"
    ? `${(value / 1e6).toFixed(2)}M` : kind === "accuracy"
    ? `${(value * 100).toFixed(1)}%` : kind === "count"
      ? formatTrainingMetricCount(value) : value.toFixed(4);
  const hasValue = (values, key, counts) => trainingMetricNumber(counts, key) !== 0
    && (Object.prototype.hasOwnProperty.call(values || {}, key) || trainingMetricNumber(counts, key) !== null);
  const appendMetric = (list, label, value, kind, count) => {
    if (count === 0 || (value === null && count === null)) return;
    const term = document.createElement("dt"); term.textContent = label;
    const definition = document.createElement("dd");
    definition.textContent = `${formatValue(value, kind)}${count !== null ? ` · n=${formatTrainingMetricCount(count)}` : ""}`;
    if (count !== null) definition.title = `目标位置数：${count}`;
    list.append(term, definition);
  };
  const appendCard = (title, sourceText, rows, values, counts, className = "", target = groups) => {
    if (!rows.some(([key]) => hasValue(values, key, counts))) return null;
    const card = document.createElement("section");
    card.className = `training-metric-group training-metric-card${className ? ` ${className}` : ""}`;
    const heading = document.createElement("h4"); heading.textContent = title; card.appendChild(heading);
    const sourceLabel = document.createElement("p"); sourceLabel.className = "training-metric-source";
    sourceLabel.textContent = sourceText; card.appendChild(sourceLabel);
    const list = document.createElement("dl"); list.className = "training-metric-list";
    for (const [key, label, kind] of rows) appendMetric(list, label, kind === "text" ? (values?.[key] ?? null)
      : trainingMetricNumber(values, key), kind,
      trainingMetricNumber(counts, key));
    if (!list.children.length) return null;
    card.appendChild(list); target.appendChild(card); return card;
  };
  const epochSource = epochNumber === null ? "epoch 汇总" : `epoch ${epochNumber}${epochValues.partial_epoch_metrics ? "（部分 epoch）" : ""}`;
  const phaseSource = `${events.phaseFromCurrentProgress ? "当前" : "最近"}${phaseNames[livePhase] || "训练"}进度 · epoch ${trainingMetricNumber(phaseSnapshot, "epoch") ?? "?"}`
    + (trainingMetricNumber(phaseSnapshot, "completed") !== null && trainingMetricNumber(phaseSnapshot, "total") !== null
      ? ` · ${phaseSnapshot.unit === "charts" ? "谱面" : "batch"} ${phaseSnapshot.completed}/${phaseSnapshot.total}` : "");
  const layoutValues = livePhase === "layout_revision" && (live || archivedProgress) ? liveValues : epochValues;
  const layoutCounts = livePhase === "layout_revision" && (live || archivedProgress) ? liveCounts : epochCounts;
  const layoutSource = livePhase === "layout_revision" && archivedProgress
    ? `${phaseSource} · 历史末条进度（本轮未完成）`
    : livePhase === "layout_revision" && live ? phaseSource : epochSource;
  const draftRows = [["draft_note_configuration_loss", "音符字符交叉熵（越低越好）", "loss"],
    ["draft_note_configuration_accuracy", "音符字符准确率", "accuracy"],
    ["draft_first_head_accuracy", "首个音符选择", "accuracy"],
    ["draft_configuration_exact_accuracy", "完整配置匹配（给定正确前缀）", "accuracy"],
    ["draft_position_fraction_exact_accuracy", "完整分数拍位匹配（给定正确前缀）", "accuracy"],
    ["position_alignment_loss", "音频—拍位辅助损失（草稿）", "loss"],
    ["position_alignment_event_accuracy", "音频位置匹配准确率（给定正确前缀）", "accuracy"],
    ["position_alignment_coverage", "官谱起音的音频查询覆盖率", "accuracy"],
    ["draft_event_continue_accuracy", "该续写时选择续写", "accuracy"],
    ["draft_event_terminate_accuracy", "该结束时选择结束", "accuracy"]];
  const revisionRows = [["revision_note_configuration_loss", "音符字符交叉熵", "loss"],
    ["revision_note_configuration_accuracy", "音符字符准确率", "accuracy"],
    ["revision_configuration_exact_accuracy", "完整配置匹配（给定正确前缀）", "accuracy"],
    ["revision_first_head_accuracy", "首个音符选择", "accuracy"]];
  const totalRows = [["loss", "阶段累计配置损失", "loss"], ["draft_revision_loss", "草稿 + 修订累计损失", "loss"]];
  const totalDetails = document.createElement("details"); totalDetails.className = "training-metric-advanced";
  const totalSummary = document.createElement("summary"); totalSummary.textContent = "配置总体（高级）";
  totalDetails.appendChild(totalSummary);
  const totalCard = appendCard("配置总体", layoutSource, totalRows, layoutValues, layoutCounts, "training-metric-total", totalDetails);
  if (totalCard) groups.appendChild(totalDetails);
  appendCard("草稿配置", layoutSource, draftRows, layoutValues, layoutCounts, "training-metric-draft");
  const revisionDetails = document.createElement("details"); revisionDetails.className = "training-metric-advanced";
  const revisionSummary = document.createElement("summary"); revisionSummary.textContent = "修订配置（高级）";
  revisionDetails.appendChild(revisionSummary);
  const revisionCard = appendCard("修订配置", layoutSource, revisionRows, layoutValues, layoutCounts, "training-metric-revision", revisionDetails);
  if (revisionCard) {
    groups.appendChild(revisionDetails);
  }
  const currentBlueprint = livePhase === "blueprint" && live;
  const blueprintSnapshot = currentBlueprint ? phaseSnapshot : events.phaseMetrics.blueprint;
  const sameEpochBlueprint = (live || archivedProgress) && blueprintSnapshot
    && trainingMetricNumber(blueprintSnapshot, "epoch") === trainingMetricNumber(phaseSnapshot, "epoch");
  const plannerValues = sameEpochBlueprint ? blueprintSnapshot.losses : epochValues;
  const plannerCounts = sameEpochBlueprint ? {} : epochCounts;
  const plannerRows = sameEpochBlueprint
    ? [["loss", "规划总损失", "loss"], ["loss_onset", "起音损失", "loss"], ["loss_type", "类型损失", "loss"],
      ["loss_duration", "时值损失", "loss"], ["loss_load_curve", "局部负荷损失", "loss"],
      ["load_curve_loss", "负荷曲线损失", "loss"],
      ["planner_density_prediction", "预测负荷均值", "loss"], ["planner_density_target", "目标负荷均值", "loss"]]
    : [["whole_song_plan_loss", "全曲规划损失", "loss"]];
  const plannerRowsAvailable = plannerRows.filter(([key]) => !(epochValues.partial_epoch_metrics === true
    && !sameEpochBlueprint && key === "whole_song_plan_loss"
    && trainingMetricNumber(plannerValues, key) === 0));
  appendCard("全曲规划", sameEpochBlueprint ? `本 epoch 已完成的全曲规划 · epoch ${trainingMetricNumber(blueprintSnapshot, "epoch") ?? "?"}` : epochSource,
    plannerRowsAvailable, plannerValues, plannerCounts, "training-metric-planner");
  const formatRows = [["draft_position_header_accuracy", "拍位与起始格式准确率", "accuracy"],
    ["revision_position_header_accuracy", "修订拍位与起始格式准确率", "accuracy"]];
  const formatDetails = document.createElement("details"); formatDetails.className = "training-metric-advanced";
  const formatSummary = document.createElement("summary"); formatSummary.textContent = "格式准确率（高级）";
  formatDetails.appendChild(formatSummary);
  const formatCard = appendCard("格式准确率", layoutSource, formatRows, layoutValues, layoutCounts, "training-metric-format", formatDetails);
  if (formatCard) {
    groups.appendChild(formatDetails);
  }

  const rawModelSummary = task?.result?.model_summary || {};
  const modelConfig = rawModelSummary.model_config || {};
  const modelSummary = {...rawModelSummary,
    config_width: rawModelSummary.config_width ?? modelConfig.d_model,
    config_heads: rawModelSummary.config_heads ?? modelConfig.nhead,
    config_layers: rawModelSummary.config_layers ?? (modelConfig.layers != null ? Number(modelConfig.layers) * 2 : null),
    config_layer_label: rawModelSummary.config_layer_label ?? (modelConfig.layers != null ? `${modelConfig.layers} 编码 + ${modelConfig.layers} 解码` : null),
    audio_dim: rawModelSummary.audio_dim ?? modelConfig.audio_dim,
    condition_count: rawModelSummary.condition_count ?? modelConfig.condition_dim,
    planner_width: rawModelSummary.planner_width ?? rawModelSummary.blueprint_config?.width};
  const modelRows = [["parameter_count", "总参数量", "params"], ["layout_parameter_count", "配置参数量", "params"],
    ["planner_parameter_count", "Planner 参数量", "params"], ["condition_count", "条件数量", "count"], ["audio_dim", "音频维度", "count"],
    ["config_width", "配置宽度", "count"], ["config_heads", "注意力头数", "count"], ["config_layer_label", "配置层数", "text"],
    ["planner_width", "规划器宽度", "count"]];
  appendCard("模型与数据", "当前任务摘要", modelRows, modelSummary, {}, "training-metric-model");

  const completeHistory = events.history.filter(item => item && item.partial_epoch_metrics !== true);
  const trendRows = [["draft_note_configuration_loss", "草稿 CE", "loss"],
    ["revision_note_configuration_loss", "修订 CE", "loss"], ["whole_song_plan_loss", "Planner 损失", "loss"]];
  const trend = document.createElement("section"); trend.className = "training-metric-trends";
  const trendHeading = document.createElement("h4"); trendHeading.textContent = "完整 epoch 趋势"; trend.appendChild(trendHeading);
  const trendNote = document.createElement("p"); trendNote.className = "training-metric-source";
  trendNote.textContent = completeHistory.length ? `显示最近 ${Math.min(completeHistory.length, 20)} 个完整 epoch；每条曲线独立缩放，只比较自身走势。` : "等待完整 epoch 汇总；部分 epoch 不计入完成趋势。";
  trend.appendChild(trendNote);
  const svgCreate = (tag) => typeof document.createElementNS === "function"
    ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
  const setAttr = (node, key, value) => { if (typeof node.setAttribute === "function") node.setAttribute(key, String(value)); else node[key] = String(value); };
  const trendHistory = completeHistory.slice(-20);
  for (const [key, label, kind] of trendRows) {
    const pointPairs = trendHistory.map(item => ({epoch: item.epoch, value: trainingMetricNumber(item, key)}))
      .filter(point => point.value !== null);
    const points = pointPairs.map(point => point.value);
    if (!points.length) continue;
    const chart = document.createElement("div"); chart.className = "training-metric-trend";
    const labelNode = document.createElement("span"); labelNode.className = "training-metric-trend-label"; labelNode.textContent = label; chart.appendChild(labelNode);
    const svg = svgCreate("svg"); setAttr(svg, "class", "training-metric-sparkline"); setAttr(svg, "viewBox", "0 0 240 48"); setAttr(svg, "role", "img");
    const min = Math.min(...points), max = Math.max(...points), span = max - min || 1;
    setAttr(svg, "aria-label", `${label}：epoch ${pointPairs[0].epoch}到${pointPairs.at(-1).epoch}，范围${min.toFixed(4)}到${max.toFixed(4)}`);
    const polyline = svgCreate("polyline"); setAttr(polyline, "fill", "none"); setAttr(polyline, "stroke", "currentColor"); setAttr(polyline, "stroke-width", "2");
    const firstEpoch=Number(pointPairs[0].epoch), epochSpan=Number(pointPairs.at(-1).epoch)-firstEpoch;
    setAttr(polyline, "points", pointPairs.map(point => `${epochSpan > 0 ? (Number(point.epoch)-firstEpoch)*240/epochSpan : 120},${44 - (point.value - min) / span * 36}`).join(" ")); svg.appendChild(polyline); chart.appendChild(svg);
    const first = points[0], last = points[points.length - 1];
    const footer = document.createElement("span"); footer.className = "training-metric-trend-meta";
    const previous = points.length > 1 ? points[points.length - 2] : null;
    const previousDelta = previous === null ? null : last - previous;
    footer.textContent = `epoch ${pointPairs[0].epoch}–${pointPairs[pointPairs.length - 1].epoch} · 起 ${formatValue(first, kind)} · 终 ${formatValue(last, kind)}`
      + (previousDelta === null ? "" : ` · 较前 ${previousDelta > 0 ? "+" : ""}${formatValue(previousDelta, kind)}`);
    chart.appendChild(footer); trend.appendChild(chart);
  }
  if (trend.children.length > 1) groups.appendChild(trend);
  panel.hidden = false;
  if (!groups.children.length && source) source.textContent = "尚无训练指标日志";
}

function trainingEpochStatus(task) {
  if(task?.kind==="composer_v2_train"){const view=composerV2TrainingView(task);return {...view,text:view.status};}
  if (!task) return null;
  if (task.kind==='composer_train') {
    const view=composerTrainingView(task);
    return {knownTarget:!!task.result?.epoch_target,progress:task.progress||0,text:view.status};
  }
  if (task.kind === "chart_stream_train") {
    return {knownTarget:!!task.result?.epoch_target,progress:task.progress||0,text:task.message||"等待连续流训练进度"};
  }
  if (task.kind === "form_train") {
    const result = task.result || {};
    const detail = result.progress_detail || {};
    const phaseKey = result.phase === "audio_cache" || result.phase === "saved"
      ? result.phase : detail.phase || result.phase;
    if (!phaseKey) return null;
    const phases = {prepare: "准备数据", audio_cache: "音频全量驻留", blueprint: "全曲规划", layout_revision: "谱面配置训练", draft: "谱面草稿", revision: "谱面修订", saved: "保存检查点", checkpoint: "保存检查点", arrangement: "整理输出"};
    const stages = {dataset: "读取数据集", preparation: "构建训练索引", chart: "准备谱面", ready: "数据准备完成"};
    const phase = `${phases[phaseKey] || String(phaseKey)}${detail.stage ? ` · ${stages[detail.stage] || detail.stage}` : ""}`;
    const epoch = Number(detail.epoch), epochTarget = Number(result.epoch_target);
    const completed = Number(detail.completed), total = Number(detail.total);
    const phaseFraction = Number(detail.phase_fraction), runFraction = Number(detail.overall_fraction);
    const unit = detail.unit === "charts" ? "首" : detail.unit === "batches" ? "batch"
      : detail.unit === "optimizer_step" ? "优化器步" : detail.unit === "blueprint_step" ? "规划步" : "项";
    const parts = [];
    if (phaseKey === "audio_cache") {
      const bytes = Number(detail.bytes), required = Number(detail.required_bytes), budget = Number(detail.budget_bytes);
      const formatBytes = (value) => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GB` : `${(value / 1024 ** 2).toFixed(0)} MB`;
      if (Number.isFinite(completed) && Number.isFinite(total)) parts.push(`音频 ${completed}/${total}`);
      if (Number.isFinite(bytes)) parts.push(`已驻留 ${formatBytes(bytes)}`);
      if (Number.isFinite(required)) parts.push(`预计需求 ${formatBytes(required)}`);
      if (Number.isFinite(budget)) parts.push(`硬预算 ${formatBytes(budget)}`);
      if (detail.unique_audio != null) parts.push(`唯一音频 ${detail.unique_audio}`);
      if (detail.preload_loads != null) parts.push(`预加载读取 ${detail.preload_loads} 次`);
      if (detail.saved_reads != null) parts.push(`避免重复读取 ${detail.saved_reads} 次`);
    }
    if (phaseKey === "prepare") {
      if (Number.isFinite(Number(detail.completed_songs)) && Number(detail.total_songs) > 0) {
        parts.push(`歌曲 ${detail.completed_songs}/${detail.total_songs}`);
      }
      if (Number.isFinite(Number(detail.charts))) parts.push(`谱面 ${detail.charts} 张`);
      if (detail.cache) parts.push(`结构缓存：${({hit: "命中，直接复用", rebuilt: "已重新构建", disabled: "未启用"})[detail.cache] || detail.cache}`);
      const prepared = detail.preparation_cache;
      if (prepared && prepared.enabled !== false) {
        parts.push(`结构缓存累计：命中 ${prepared.hits ?? 0} / 重建 ${prepared.rebuilt ?? 0} / 损坏 ${prepared.invalid ?? 0}`);
      }
    }
    if (Number.isFinite(epoch) && epoch > 0) {
      const local = Number.isFinite(completed) && Number.isFinite(total) && total > 0
        ? ` · 当前阶段 ${unit} ${completed}/${total}` : "";
      parts.push(`epoch ${epoch}${epochTarget > 0 ? `/${epochTarget}` : ""}${local}`);
    } else if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
      parts.push(`${unit} ${completed}/${total}`);
    }
    if (Number.isFinite(runFraction)) parts.push(`本轮任务数进度（规划谱面数＋配置 batch 数）${(runFraction * 100).toFixed(0)}%`);
    if (Number.isFinite(phaseFraction)) parts.push(`本阶段进度 ${(phaseFraction * 100).toFixed(0)}%`);
    const optimizerSteps = Number(detail.optimizer_steps);
    const blueprintSteps = Number(detail.blueprint_steps);
    if (Number.isFinite(optimizerSteps)) parts.push(`参数更新 ${optimizerSteps} 步`);
    if (Number.isFinite(blueprintSteps)) parts.push(`全曲规划 ${blueprintSteps} 步`);
    const pending = Number(detail.accumulation_pending), accumulationTarget = Number(detail.grad_accumulation);
    if (Number.isFinite(pending) && Number.isFinite(accumulationTarget)) parts.push(`累积 batch ${pending}/${accumulationTarget}`);
    const phaseEta = detail.eta_phase_sec == null ? NaN : Number(detail.eta_phase_sec);
    if (Number.isFinite(phaseEta) && phaseEta >= 0) {
      const seconds = Math.floor(phaseEta);
      parts.push(`本阶段 ETA ${seconds >= 3600 ? `${Math.floor(seconds / 3600)}小时${Math.floor(seconds % 3600 / 60)}分` : seconds >= 60 ? `${Math.floor(seconds / 60)}分${seconds % 60}秒` : `${seconds}秒`}`);
    }
    const speed = Number(detail.items_per_sec);
    if (Number.isFinite(speed) && speed >= 0) parts.push(`${speed.toFixed(2)} ${unit}/秒`);
    const samplesSpeed = Number(detail.samples_per_sec), tokensSpeed = Number(detail.tokens_per_sec);
    if (Number.isFinite(samplesSpeed) && samplesSpeed >= 0) parts.push(`${samplesSpeed.toFixed(1)} 样本/秒`);
    const memoryPacking = detail.attention_memory;
    if (memoryPacking?.dense_tokens > 0 && memoryPacking?.packed_tokens > 0) {
      parts.push(`注意力空位压紧：${memoryPacking.dense_tokens} → ${memoryPacking.packed_tokens} token（当前 batch）`);
    }
    if (Number.isFinite(tokensSpeed) && tokensSpeed >= 0) parts.push(`${tokensSpeed.toFixed(0)} token/秒`);
    const elapsed = Number(detail.elapsed_sec), phaseElapsed = Number(detail.phase_elapsed_sec);
    const duration = (seconds) => seconds >= 3600 ? `${Math.floor(seconds / 3600)}小时${Math.floor(seconds % 3600 / 60)}分` : seconds >= 60 ? `${Math.floor(seconds / 60)}分${Math.floor(seconds % 60)}秒` : `${Math.floor(seconds)}秒`;
    if (Number.isFinite(elapsed) && elapsed >= 0) parts.push(`已用时 ${duration(elapsed)}`);
    if (Number.isFinite(phaseElapsed) && phaseElapsed >= 0) parts.push(`阶段已用时 ${duration(phaseElapsed)}`);
    const lr = Number(detail.lr);
    if (Number.isFinite(lr)) parts.push(`lr ${lr.toExponential(2)}`);
    if (detail.precision) parts.push(String(detail.precision));
    const timing = detail.timing && typeof detail.timing === "object" ? ["data_wait_sec", "h2d_sec", "compute_sec", "checkpoint_sec"]
      .filter((key) => Number.isFinite(Number(detail.timing[key])))
      .map((key) => `${({data_wait_sec: "数据", h2d_sec: "传输", compute_sec: "计算", checkpoint_sec: "保存"})[key]} ${Number(detail.timing[key]).toFixed(1)}s`) : [];
    if (timing.length) parts.push(`主机累计计时（非 GPU 计时）：${timing.join(" / ")}`);
    const memory = detail.memory && typeof detail.memory === "object" ? ["allocated_gib", "reserved_gib", "peak_allocated_gib", "inactive_cached_gib", "device_free_gib"]
      .filter((key) => Number.isFinite(Number(detail.memory[key])))
      .map((key) => `${({allocated_gib: "已分配", reserved_gib: "已保留", peak_allocated_gib: "峰值已分配", inactive_cached_gib: "闲置缓存", device_free_gib: "设备可用"})[key]} ${Number(detail.memory[key]).toFixed(2)} GiB`) : [];
    if (memory.length) parts.push(`CUDA 显存：${memory.join(" / ")}`);
    if (detail.cuda_cache_guard?.enabled) parts.push(`显存缓存保护：阈值 ${Number(detail.cuda_cache_guard.reserved_high_water_gib).toFixed(2)} GiB · 已回收 ${detail.cuda_cache_guard.trim_count ?? 0} 次`);
    const cache = detail.cache && typeof detail.cache === "object" ? ["audio", "samples"].flatMap((name) => {
      const stat = detail.cache[name];
      if (!stat || typeof stat !== "object") return [];
      const size = Number(stat.bytes), budget = Number(stat.budget_bytes);
      const byteLabel = (value) => value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(0)} MB` : `${(value / 1024).toFixed(0)} KB`;
      return [`${name === "audio" ? "音频" : "样本"}命中 ${stat.hits ?? 0} / 未命中 ${stat.misses ?? 0}${Number.isFinite(size) ? ` · ${byteLabel(size)}${Number.isFinite(budget) ? `/${byteLabel(budget)}` : ""}` : ""}`];
    }) : [];
    if (cache.length) parts.push(`缓存${detail.cache_scope === "main_process_only" ? "（主进程本地，worker 缓存未汇总）" : "（训练进程本地）"}：${cache.join("；")}`);
    return {knownTarget: Number.isFinite(epochTarget) && epochTarget > 0,
      progress: Number.isFinite(Number(task.progress)) ? Number(task.progress) : 0,
      text: `${phase}${parts.length ? ` · ${parts.join(" · ")}` : ""}`};
  }
  if (task.kind !== "train") return null;
  const result = task.result || {};
  const target = Number(result.epoch_target);
  const hasTarget = Number.isFinite(target) && target > 0;
  const start = Number(result.epoch_start || 0);
  const current = Number(result.last_epoch ?? start);
  if (hasTarget) {
    const semantics = result.epoch_semantics === "additional_from_checkpoint"
      ? `resume：从 epoch ${start} 追加 ${result.epochs_requested || Math.max(0, target - start)} 轮`
      : `本阶段共 ${result.epochs_requested || target} 轮`;
    return {
      knownTarget: true,
      progress: task.status === "completed" ? 1 : Math.max(0, Math.min(1, current / target)),
      text: `训练进度：已完成 epoch ${current} / ${target}；${semantics}`,
    };
  }
  if (["queued", "running", "cancelling"].includes(task.status) || result.last_epoch !== undefined) {
    const last = result.last_epoch !== undefined ? `已完成 epoch ${current}；` : "";
    return {
      knownTarget: false,
      progress: null,
      text: `训练进度：${last}目标未知，请查看任务日志。`,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Condition sliders.  Each entry is [key, label, help, default]; the help text
// Tooltips give the current 1,859-chart global P10-P90. The level preset
// replaces these fallback defaults with its per-level calibrated baseline.
// ---------------------------------------------------------------------------
const radarKeys = [
  ["note_density", "整体密度", "音符头总数/音频秒数/16 的归一化值；默认 0.305 约 4.88 heads/s。不是响度。当前 1,859 张全局 P10–P90：0.226–0.409；等级预设会改用该等级区间。;训练侧实测", "0.305"],
  ["peak", "峰值密度", "一秒窗口内最大起音事件组数/32 的归一化值；同刻多押算一个事件组。当前 1,859 张全局 P10–P90：0.219–0.656；等级预设会改用该等级区间。;训练侧实测", "0.375"],
  ["sweep", "滑条量（sweep）", "Slide 头比例×平均路径端点环距/4；不是完整描线长度或体感运动量。当前 1,859 张全局 P10–P90：0.053–0.170。;训练侧实测", "0.109"],
  ["slide_tricky", "滑条复杂度", "Slide 头比例×程序判作复杂的 Slide 比例；复杂度含多部件、短时值和部分路径符号，不等于主观手法难度。当前全局 P10–P90：0.046–0.182。;训练侧实测", "0.114"],
  ["slide_sequence", "连续滑条", "Slide 头比例×相邻 Slide 间隔满足 0<间隔≤2 拍的比例；同刻 Slide 不计入。当前全局 P10–P90：0.020–0.125。;训练侧实测", "0.066"],
  ["jack", "同键连打（jack）", "分子是间隔≤0.5 秒的相邻起音组同键重复按钮头；分母是所有可比相邻组的后继按钮头（不限时间），长间隔也会稀释。当前全局 P10–P90：0.105–0.275。;训练侧实测", "0.188"],
  ["chord", "和弦比例", "含 Touch 等类型的同刻多 note 起音组/有音起音组；preset 会与 Style 的 chord_bias 设为相同值。当前全局 P10–P90：0.162–0.403。;训练侧实测", "0.280"],
];
const styleKeys = [
  ["tap_bias", "Tap 倾向", "Tap 头比例，含 break 修饰的 Tap。当前 1,859 张全局 P10–P90：0.602–0.827。;训练侧实测", "0.720"],
  ["slide_bias", "Slide 倾向", "Slide 头占全部 note 头比例；建议配合 Radar 的 sweep、complexity 与 sequence 理解。当前全局 P10–P90：0.068–0.218。;训练侧实测", "0.140"],
  ["hold_bias", "Hold 倾向", "Hold+TouchHold 头比例。当前全局 P10–P90：0.032–0.139；TouchHold 同时也计入 touch_bias，两项不互斥。;训练侧实测", "0.077"],
  ["touch_bias", "Touch 倾向", "Touch+TouchHold 头比例。当前全局 P10–P90：0.000–0.137；TouchHold 同时也计入 hold_bias。;训练侧实测", "0.039"],
  ["chord_bias", "和弦倾向", "与 Radar 的 chord 使用完全相同的同刻多 note 事件组比例；preset 会联动，手动分开设置会偏离训练中的成对值。当前全局 P10–P90：0.162–0.403。;训练侧实测", "0.280"],
  ["melody_following", "节奏贴合（非旋律，melody_following）", "谱面起音落在音频起音±80ms 内的匹配比例代理；没有声部/音高标签，不代表识别旋律。当前全局 P10–P90：0.830–0.967，中位0.912；fallback预设固定0.9。;训练侧实测", "0.900"],
  ["pattern_reuse", "模式复用", "完全相同的非恒定四拍配置签名重复率；不是主歌/副歌级段落呼应。当前全局 P10–P90：0.000–0.100。;训练侧实测", "0.028"],
  ["directionality", "方向性", "|顺时针−逆时针|/有效单键转移数，只看净旋向一致度，不指定方向或所有双手动作。当前全局 P10–P90：0.008–0.138。;训练侧实测", "0.043"],
];
// Optional conditions remain unknown until explicitly enabled. Reference values
// are shared chart statistics, not a claim of calibration for this checkpoint.
const v2ConditionState=Object.fromEntries([...radarKeys,...styleKeys].map(([key,,,fallback])=>[key,{value:Number(fallback),override:false,edited:false}]));
let v2ConditionActive=false, otherFamilyConditionValues={};
function v2ConditionReference(key) {
  const fallback=Number([...radarKeys,...styleKeys].find(entry=>entry[0]===key)?.[3] ?? 0);
  const level=normalizeGenerationLevel(valueOrDefault('#level',13.3));
  const preset=levelPresetProfile?interpolateLevelPreset(levelPresetProfile,level):null;
  const value=preset?.radar?.[key] ?? preset?.style?.[key] ?? fallback;
  const bounds=preset?.controls?.[key];
  return {value:Number(Number(value).toFixed(3)),text:preset
    ? `${level.toFixed(2)} 等级历史参考 ${Number(value).toFixed(3)}${bounds?`，P10–P90 ${bounds.p10.toFixed(3)}–${bounds.p90.toFixed(3)}`:''}`
    : `共享谱面历史参考 ${fallback.toFixed(3)}`};
}
function restoreV2ConditionSettings(saved) {
  if(saved?.version!==1 || !saved.fields)return;
  for(const [key,state] of Object.entries(v2ConditionState)) {
    const row=saved.fields[key];
    if(!row || typeof row.value!=='number' || !Number.isFinite(row.value) || row.value<0 || row.value>1)continue;
    Object.assign(state,{value:row.value,override:row.override===true,edited:row.edited===true || row.override===true});
  }
}
function resetV2ConditionState() {
  for(const [key,state] of Object.entries(v2ConditionState))
    Object.assign(state,{value:v2ConditionReference(key).value,override:false,edited:false});
  updateV2ConditionControls();
}
function v2ConditionCapability() {
  const data=generationInspection.data;
  const checked=generationInspection.path===valueOrDefault('#checkpointPath','').trim() && data?.valid;
  return {checked:!!checked,enabled:!checked || data.condition_contract?.policy==='radar_style_v1',
    keys:checked?(data.condition_contract?.supported_keys||[]):[...radarKeys,...styleKeys].map(x=>x[0])};
}
function updateV2ConditionControls() {
  const active=currentGenerationFamily()==='chart_composer_v2', caps=v2ConditionCapability();
  if(active!==v2ConditionActive) {
    for(const [key] of [...radarKeys,...styleKeys]) {
      if(active)otherFamilyConditionValues[key]=element('#'+key)?.value;
      const value=active?v2ConditionState[key].value:otherFamilyConditionValues[key];
      if(value!==undefined) {
        const range=element('#'+key),number=document.querySelector(`[data-number-for="${key}"]`);
        if(range)range.value=String(value);
        if(number)number.value=String(value);
      }
    }
    v2ConditionActive=active;
  }
  for(const [key] of [...radarKeys,...styleKeys]) {
    const range=element('#'+key), row=range?.closest('.control-row');if(!row)continue;
    const state=v2ConditionState[key],reference=v2ConditionReference(key);
    if(active) {
      if(!state.override)state.value=reference.value;
      range.value=String(state.value);
      const mirror=row.querySelector('input[type=number]');if(mirror)mirror.value=String(state.value);
    }
    row.classList.toggle('v2-condition-row',active);
    const help=row.querySelector('small'), name=row.querySelector(`label[for="${key}"]`);
    if(help) {
      help.dataset.originalHelp??=help.textContent;
      help.textContent=active?((key==='melody_following'
        ? '节奏贴合起音代理：谱面起音与现有 logmel 正通量提取的起音在 ±80ms 内的匹配比例；无有效通量时训练为未知。不是旋律或音高识别。'
        : help.dataset.originalHelp.split('当前')[0].replace(/；默认[^。]*。/g,'。').split(';训练侧实测')[0])
        + ` 手动起始值：${reference.text}；不是当前 checkpoint 的校准结果。`)
        :help.dataset.originalHelp;
    }
    if(name && key==='melody_following')name.textContent=active?'节奏贴合（非旋律）':styleKeys.find(entry=>entry[0]===key)[1];
    let toggle=row.querySelector('[data-v2-override]');
    if(!toggle) {
      const label=document.createElement('label');label.className='v2-condition-mode';
      label.innerHTML=`<input type="checkbox" data-v2-override="${key}"> 手动指定 <span></span>`;
      row.append(label);toggle=label.querySelector('input');
      toggle.addEventListener('change',()=>{
        if(toggle.checked && !state.edited)state.value=v2ConditionReference(key).value;
        state.override=toggle.checked;if(toggle.checked)state.edited=true;
        updateV2ConditionControls();saveSettings();
      });
      for(const input of row.querySelectorAll('input[type=range],input[type=number]'))input.addEventListener('input',()=>{
        if(v2ConditionActive){state.value=input.value.trim()===''?NaN:Number(input.value);state.edited=true;saveSettings();}
      });
    }
    const supported=caps.enabled && caps.keys.includes(key);
    toggle.parentElement.hidden=!active;toggle.checked=v2ConditionState[key].override;
    // A previously enabled override can always be cleared after switching checkpoints.
    toggle.disabled=!supported && !toggle.checked;
    toggle.parentElement.querySelector('span').textContent=!supported?'（此 checkpoint 不支持）':toggle.checked?'（发送当前值）':'（自动由音乐决定）';
    toggle.parentElement.title=key==='melody_following'?'logmel 正通量起音 ±80ms 匹配比例代理，不是旋律识别。':'';
    for(const input of row.querySelectorAll('input[type=range],input[type=number]'))input.disabled=active && (!supported || !toggle.checked);
  }
  const hint=element('#v2ConditionStatus');
  if(hint) {hint.hidden=!active;hint.textContent=caps.enabled
    ? '默认全部保持「自动由音乐决定」，让模型顺应音乐段落安排手法；只需对想专门干预的项（如减少 Touch、增加 Slide 或多押）勾选「手动指定」并拖动滑块作为软目标，其余项继续自动跟歌。关闭手动指定后显示值恢复为该等级的语料参考。'
    : '当前 checkpoint 尚未确认 Radar / Style 能力，或仅训练了等级（level_only_v1）；这些滑块不会生效。请检查 checkpoint；旧权重需在新目录 finetune 才能学习新条件。';}
}

// Difficulty controls are calibrated from original charts. Preferences move
// within that level's corpus range; no new model coordinate or training contract.
let levelPresetProfile = null;
let lastConditionRecipe = null;
let conditionEditedBeforeProfile = false;
const preferenceDefinitions = {
  keyboard: { field: "preferKeyboard", label: "键盘", axes: { tap_bias: 1, chord: 0.7, jack: 0.25, slide_bias: -0.6, sweep: -0.4, touch_bias: -0.5 } },
  stars: { field: "preferStars", label: "星星", axes: { slide_bias: 1, sweep: 1, slide_sequence: 0.7, tap_bias: -0.5, jack: -0.3 } },
  technical: { field: "preferTechnical", label: "技巧", axes: { slide_tricky: 1, slide_sequence: 0.5, directionality: 0.5 } },
  chord: { field: "preferChord", label: "和弦", axes: { chord: 1, tap_bias: 0.3 } },
  hold: { field: "preferHold", label: "Hold", axes: { hold_bias: 1, tap_bias: -0.3 } },
  touch: { field: "preferTouch", label: "Touch", axes: { touch_bias: 1, tap_bias: -0.3 } },
  movement: { field: "preferMovement", label: "移动", axes: { directionality: 1, sweep: 0.4 } },
};

const datasetStorageKey = "maichart.training.dataset.v1";
const settingsStorageKey = "maichart.webui.settings.v1";
const workspaceStorageKey = "maichart.webui.workspace";
const intensityCurveStorageKey = "maichart.webui.intensity-curve.v1";
const GENERATION_LEVEL_MIN = 12;
const GENERATION_LEVEL_MAX = 15;
const GENERATION_LEVEL_DEFAULT = 13.3;
// A previous WebUI version could persist an unrelated difficulty value (for
// example 1.0) under #level.  Keep the repair in one small, testable helper so
// restoring settings cannot silently create an out-of-contract curve condition.
function normalizeGenerationLevel(value, fallback = GENERATION_LEVEL_DEFAULT) {
  const level = Number(value);
  return Number.isFinite(level) && level >= GENERATION_LEVEL_MIN && level <= GENERATION_LEVEL_MAX
    ? level : fallback;
}
let restoredSettingNotice = "";

// Generation knobs only.  Song data (bpm/first/tempoMap), file paths, metadata
// and every training control are deliberately absent: resetting the generation
// panel must never endanger a training configuration the user settled on.
const GENERATION_PARAM_FIELDS = [
  "v2EvalManifest", "v2EvalLimitCharts", "v2EvalMaxBatches", "v2EvalTimingRollouts", "v2EvalCpuThreads", "v2EvalWindowSampling", "v2EvalCheckTrainingOverlap", "v2AuditInput",
  "v2GenerateRecordCandidateProbabilities", "v2GenerateTemperature", "v2GenerateTopP", "v2GenerateActionTemperature", "v2GenerateActionTopP", "v2GenerateOnsetStrength", "v2GenerateInterceptTapOnSlide", "v2GenerateTapOnSlideThresholdMs", "v2GenerateInterceptSlideHeadTap", "v2GenerateForwardFeasibility", "v2GenerateForwardTapOnSlideMs", "v2GenerateTouchReachabilityPolicy", "v2GenerateTouchCoholdRadius", "v2GenerateSelfRefine", "v2GenerateCandidates", "v2GenerateCount", "v2GenerateParallel", "v2GenerateWindowRetries", "v2GenerateDevice", "v2GenerateMaxEvents", "v2GenerateRollbackWindows", "v2GenerateExhaustionPolicy", "v2GenerateCpuThreads", "v2GenerateAutoFirst", "v2GenerateTempoDetection", "v2GeneratePhraseSearch", "v2GeneratePhraseBeats", "v2GeneratePhraseCommitBeats", "v2GeneratePhraseRankPolicy", "v2GenerateTempoOctavePolicy", "v2GenerateDownbeatAlignment",
  "tempoMode", "division",                                  // 节拍与时间轴
  "level", "conditionPreferenceStrength",                   // 等级基准与叠加偏好
  "preferKeyboard", "preferStars", "preferTechnical", "preferChord", "preferHold", "preferTouch", "preferMovement",
  "generationDevice",                                       // 生成运行设备
  "formPositionComplexity", "formPositionBeamSize", "formEarlyEosPenalty",
  "streamPositionTemperature", "streamPositionBeamSize", "streamPositionComplexity", "streamEarlyEosPenalty",
  "streamEosTemperature", "streamPositionAudioStrength", "streamRepetitionStrength",
  "streamPositionDensityStrength", "streamRepetitionStructureStrength", "streamConfigurationExplorationTemperature", "streamRepetitionEscapeAttempts", "streamSustainEndSec",
  "streamRepetitionMinSeconds", "streamRepetitionMinBeats",
  "streamPositionIntervalStrength", "streamUnplannedGapPenaltyStrength", "streamRhythmDiversityStrength",
  "streamPositionComplexityFloor", "streamLearnedEndingStrength", "streamConstraintMinProbability", "streamEventReconsiderAttempts",
  "streamMaxButtonHeads", "streamHoldProtection",
  "streamReferenceChecks",
  "streamTrimExactSilence",
  "streamMaxDenominator", "streamMaxIntervalDenominator", "streamMinSpacingMs",
  "streamStyleBalanceStrength", "streamSectionTypeStrength", "streamPlanIntentStrength", "streamLocalChordStrength", "streamPhysicalStrength",
  "streamConfigurationCandidates",
  "composerSongStyleSource", "composerSongStyleSeed", "composerSongStyleLocks", "composerSongResponseMode",
  "composerMaxEvents", "composerConditionFile", "composerCurveStrength", "composerPointerTemperature", "composerActionTemperature", "composerPhraseModeTemperature", "composerActionHistoryStrength", "composerActionTypeTemperature", "composerActionTypeMinP", "composerMinSameHeadMs", "composerPointerHistoryStrength", "composerActionRetryPolicy",
  "composerPointerGridPolicy", "composerRepetitionGuard", "composerSurvivalCalibration", "composerDensityFeedbackOverride",
  "composerInterceptTapOnSlide", "composerTouchHistoryGuard", "composerTouchHistoryBound",
  "composerTouchLayoutPolicy", "composerGenerationContactPolicy", "composerMotifStrength",
  "composerQualityContextStrength", "composerQualityPlanStrength", "composerQualityLookaheadStrength",
  "seed", "candidates", "temperature", "sampleEvents",       // 采样与随机性
  "topP", "repetitionPenalty", "repetitionWindow", "maxTokens",
  "intensityFollow", "intensityFollowSpan",                  // 逐窗口强度跟随
  "intensityMode", "intensityCurveEnabled",
  "intensityCurveStrength", "intensityCurveMode", "intensityCurveSource", "composerIntensityCurveSpan", "composerChordMultiplier",
  "tailDamping", "tailDampingWindows", "tailDampingStrength", // 收尾衰减
  "audioBoundsTrim", "audioBoundsLead", "silenceThreshold", "silenceWindowShare", // 音频边界
  "guidanceScale", "presenceThreshold", "buttonThreshold",
  "fineNoteStrength", "localOnsetBiasStrength", "conflictDetectionEnabled", "detectHardConflicts", "detectSoftConflicts",
  "conflictRepairMode", "conflictRetries", "conflictPreset", "conflictMultiTouchMode", "conflictOverlapMode",
  "conflictTapOnSlideMode", "conflictSlideHeadTapMode", "conflictSlideTooFastMode", "tapOnSlideThresholdMs",
  "conflictBacktrackWindows", "conflictBacktrackBudget", "conflictBacktrackUnlimited", "sustainRepairMode", "noteRepairMode",
];

const CONFLICT_KIND_FIELDS = [
  ["MultiTouch", "conflictMultiTouchMode"],
  ["Overlap", "conflictOverlapMode"],
  ["TapOnSlide", "conflictTapOnSlideMode"],
  ["SlideHeadTap", "conflictSlideHeadTapMode"],
  ["SlideTooFast", "conflictSlideTooFastMode"],
];
const CONFLICT_PRESETS = {
  preserve: { MultiTouch: "report", Overlap: "enforce", TapOnSlide: "enforce", SlideHeadTap: "enforce", SlideTooFast: "enforce", threshold: 150 },
  balanced: { MultiTouch: "report", Overlap: "enforce", TapOnSlide: "enforce", SlideHeadTap: "enforce", SlideTooFast: "enforce", threshold: 200 },
  cautious: { MultiTouch: "enforce", Overlap: "enforce", TapOnSlide: "enforce", SlideHeadTap: "enforce", SlideTooFast: "enforce", threshold: 250 },
};

function conflictKindModesFromUI() {
  const modes = {};
  CONFLICT_KIND_FIELDS.forEach(([kind, id]) => { modes[kind] = valueOrDefault(`#${id}`, "enforce"); });
  return modes;
}

function applyConflictPreset(name) {
  const preset = CONFLICT_PRESETS[name];
  if (!preset) return;
  CONFLICT_KIND_FIELDS.forEach(([kind, id]) => {
    const target = element(`#${id}`);
    if (target) target.value = preset[kind];
  });
  const threshold = element("#tapOnSlideThresholdMs");
  if (threshold) threshold.value = String(preset.threshold);
  const custom = element("#conflictPreset");
  if (custom) custom.value = name;
  saveSettings();
}

let activeTaskId = null;
let datasetUploading = false;
let preprocessStarting = false;
let trainingStarting = false;
let trainingPreflight = false;
const liveComposerTrainingTasks = new Set();
const pendingTaskCancellations = new Set();
let generationSubmitting = false;
let datasetVersion = 0;
let manifestRequest = 0;
let checkpointInspectRequest = 0;
let generationInspectRequest = 0;
let uploadedDatasets = [];
let checkpointInspection = { key: "", data: null };
let generationInspection = { path: "", data: null };
let generationDefaults = {};
let songFormCurveSourceBeforeRestriction = null;
let songFormCurveModeBeforeRestriction = null;
let songFormCurveRestrictionsActive = false;
let intensityCurveState = { path: "", hash: "", profile: null, automatic: null, history: [], taskId: null, timer: null, dragging: -1, saved: false, contextStale: false, conditionChanged: false };
const qualityState = { items: [], page: 1, pageSize: 25, pages: 1, total: 0, folderTotal: 0, query: "", label: "", autoFilter: "", annotationPath: "", annotationManifest: "", summary: null, selected: new Set(), editedWeights: new Map(), selectionRequest: 0, selectionLoading: false, saving: false, detail: null, regions: [], correctedPath: "", catalogRequest: 0, detailRequest: 0 };
// Dataset selection is intentionally separate from qualityState: selected
// sources feed training, while activeDataset is the source currently being
// browsed/annotated. This prevents a merged training manifest from hiding the
// origin of a quality label.
const datasetWorkspaceState = { items: [], selected: new Set(), activePath: "", catalog: null, catalogRequest: 0, selectionTouched: false, view: "cards", prepared: null, applying: false };
const intensityView = { start: 0, end: null, selected: new Set(), gesture: null, spectrogram: null, specSource: null, sectionSelectionKey: "" };
let settingsSaveTimer = null;
const preprocessTasks = new Map();

const $ = (selector) => document.querySelector(selector);

function element(selector) {
  return document.querySelector(selector);
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { error: text || `HTTP ${response.status}` }; }
  const taskStatusResponse = body === undefined && path.split("?", 1)[0] === "/api/task"
    && typeof data.task_id === "string"
    && ["queued", "running", "cancelling", "completed", "failed", "cancelled"].includes(data.status);
  if (!response.ok || (!taskStatusResponse && data.error)) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function guard(handler) {
  return async (...args) => {
    try {
      await handler(...args);
    } catch (error) {
      setStatus(`操作失败：${error.message || error}`, false);
      const message = element("#taskMessage");
      if (message) message.textContent = `错误：${error.message || error}`;
      window.MaichartWorkflow?.reportError(`操作失败：${error.message || error}`);
      console.error(error);
    }
  };
}

function on(selector, eventName, handler) {
  const target = element(selector);
  if (target) target.addEventListener(eventName, guard(handler));
}

// ---------------------------------------------------------------------------
// Workspace switching: training and generation are separate pages instead of
// one very long scroll of unrelated controls.
// ---------------------------------------------------------------------------
function setWorkspaceMode(mode) {
  const value = mode === "intensity" ? "generation" : ["generation", "quality"].includes(mode) ? mode : "training";
  document.body.dataset.workspace = value;
  element("#trainingModeTab")?.classList.toggle("active", value === "training");
  element("#qualityModeTab")?.classList.toggle("active", value === "quality");
  element("#generationModeTab")?.classList.toggle("active", value === "generation");
  element("#intensityModeTab")?.classList.toggle("active", value === "intensity");
  if (value === "intensity") requestAnimationFrame(drawIntensityCurve);
  window.MaichartWorkflow?.setLogsExpanded(false);
  window.MaichartWorkflow?.refresh();
  window.dispatchEvent(new CustomEvent('workspace-mode-changed',{detail:{mode:value}}));
  try { localStorage.setItem(workspaceStorageKey, value); } catch (_) { /* storage disabled */ }
}

function initialWorkspace() {
  try { return localStorage.getItem(workspaceStorageKey) || "generation"; } catch (_) { return "generation"; }
}

function loadIntensityCurveState(validateContext = true) {
  try {
    const saved = JSON.parse(localStorage.getItem(intensityCurveStorageKey));
    if (saved && typeof saved === "object") intensityCurveState = { ...intensityCurveState, ...saved,
      profile: saved.profile ? normalizeIntensityProfile(saved.profile) : null,
      automatic: saved.automatic ? normalizeIntensityProfile(saved.automatic) : null,
      history: [], saved: saved.saved === true,
      // Older UI versions mixed condition and timing invalidation. Let their
      // curves be saved again; backend identity/timing preflight stays strict.
      contextStale: saved.contextVersion === 2 && saved.contextStale === true,
      conditionChanged: saved.conditionChanged === true || (saved.contextVersion !== 2 && saved.contextStale === true) };
  } catch (_) { /* storage disabled or stale */ }
  if (validateContext) invalidateLearnedCurveForCheckpoint();
}

function saveIntensityCurveState() {
  const profile = intensityCurveState.profile;
  try {
    localStorage.setItem(intensityCurveStorageKey, JSON.stringify({
      path: intensityCurveState.path, hash: intensityCurveState.hash,
      profile: profile ? { ...profile, spectrogram: undefined } : null,
      automatic: intensityCurveState.automatic ? { ...intensityCurveState.automatic, spectrogram: undefined } : null,
      saved: intensityCurveState.saved === true,
      contextStale: intensityCurveState.contextStale === true,
      contextStaleReason: intensityCurveState.contextStaleReason || null,
      contextVersion: 2,
      conditionChanged: intensityCurveState.conditionChanged === true,
      checkpointRevision: intensityCurveState.checkpointRevision || null,
    }));
  } catch (_) { /* curve is still usable for this page */ }
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------
function settingsFields() {
  return Array.from(document.querySelectorAll("input[id], select[id], textarea[id]"))
    .filter((element) => element.type !== "file" && element.id !== "datasetSelect"
      && !["taskSelector", "trainingRunRecipe", "trainingBatchRecipe", "composerResetLearningRate", "generationPreviewChart"].includes(element.id)
      && !element.id.endsWith("ParameterSearch")
      && !element.id.startsWith("intensitySection"));
}

function collectSettings() {
  const values = {};
  settingsFields().forEach((target) => {
    values[target.id] = target.type === "checkbox" ? target.checked : target.value;
  });
  values.__composerV2Conditions={version:1,fields:Object.fromEntries(Object.entries(v2ConditionState).map(([key,state])=>[key,{...state}]))};
  // Shared slider IDs persist the other families' values, independently of v2.
  if(v2ConditionActive)Object.assign(values,otherFamilyConditionValues);
  values.__manualTouched = {
    learningRate: element("#learningRate")?.dataset.touched === "1",
    contextTokens: element("#contextTokens")?.dataset.touched === "1",
  };
  values.__songFormEvidenceProvenance = Object.fromEntries([
    "formQualityProfile", "formPositionAlignmentLossWeight", "formPositionAlignmentToleranceMs",
    "formPlannerContextSeconds", "formCurveBinSeconds",
  ].map((id) => {
    const data = element(`#${id}`)?.dataset || {};
    return [id, {manual: data.evidenceManual === "1", inherited: data.evidenceInheritedProfile,
      inheritedValue: data.evidenceInheritedValue}];
  }));
  values.__songFormQualityDefaults = 2;
  values.__eventSetSharedContinuationDefaults = 1;
  values.__eventSetLearningDefaults = 2;
  values.__composerV2ReleaseDefaults = 1;
  values.__composerV2PhraseMusicDefaults = 1;
  if (v2TrainPlanBeforeSmoke) values.__v2TrainPlanBeforeSmoke = v2TrainPlanBeforeSmoke;
  return values;
}

// Settings saved before the recommended robustness defaults hold the former
// all-off values. Drop them once so index.html defaults apply; any value the
// user changed (or a recipe other than the old default) is kept.
function migrateSongFormQualityDefaults(values) {
  if (!values || values.__songFormQualityDefaults === 2) return false;
  const legacy = {formPerturbApplyProb: 1, formPerturbCurveNoiseStd: 0, formPerturbCurveDropoutProb: 0,
    formPerturbHistoryCorruptionProb: 0, formPerturbOnsetDropProb: 0, formPerturbOnsetAddProb: 0,
    formPerturbOnsetShiftMs: 0, formDiversityLossWeight: 0};
  const untouched = (values.formQualityRecipe ?? "off") === "off"
    && Object.entries(legacy).every(([id, value]) => values[id] == null || Number(values[id]) === value);
  if (!untouched) return false;
  for (const id of [...Object.keys(legacy), "formQualityRecipe"]) delete values[id];
  return true;
}

function saveSettings() {
  if (settingsSaveTimer) clearTimeout(settingsSaveTimer);
  settingsSaveTimer = setTimeout(flushSettings, 200);
  window.MaichartWorkflow?.refresh();
}

function flushSettings() {
  if (settingsSaveTimer) { clearTimeout(settingsSaveTimer); settingsSaveTimer = null; }
  try {
    localStorage.setItem(settingsStorageKey, JSON.stringify({ version: 2, values: collectSettings() }));
  } catch (_) { /* storage disabled or full */ }
}

function migrateEventSetLearningDefaults(values) {
  if (!values || values.__eventSetLearningDefaults === 2
      || values.modelFamily !== 'chart_composer_v1'
      || values.composerArchitectureProfile !== 'event_set_stream_v1'
      || values.trainingMode === 'resume'
      || !['auto', 'repaired'].includes(values.composerLearningPolicy ?? 'auto')) return false;
  // Auto is resolved by the current trainer. Explicit fields override the
  // preset and must survive imports: old settings have no edit provenance.
  for (const id of Object.keys(values)) {
    if (id.startsWith('eventSetPolicy_') && (values[id] == null || values[id] === '')) values[id] = 'auto';
  }
  for (const id of ['composerDraftObjectivePolicy', 'composerTrainingWindowPolicy',
      'composerPhraseExecutionPolicy', 'composerDetailRepresentationPolicy',
      'composerDurationFeaturePolicy', 'composerDurationObjectivePolicy',
      'composerSongProfilePolicy', 'composerSongResponsePolicy',
      'composerPhraseContextPolicy', 'composerDurationAudioPolicy', 'composerCurvePreviewPolicy']) {
    if (values[id] == null || values[id] === '') values[id] = 'auto';
  }
  values.__eventSetLearningDefaults = 2;
  return true;
}

function migrateComposerV2ReleaseDefaults(values) {
  if(!values)return false;
  let changed=false;
  if(values.__composerV2ReleaseDefaults!==1) {
  // One release migration; subsequent deliberate overrides are persisted.
  values.v2GenerateAutoFirst=true;
  values.v2GenerateInterceptTapOnSlide=true;
  for(const [key,value] of Object.entries({v2GenerateActionTemperature:'0.9',v2GenerateActionTopP:'0.95'}))
    if(values[key]==null || values[key]==='')values[key]=value;
  if(values.v2GenerateCandidates==null || String(values.v2GenerateCandidates)==='1')values.v2GenerateCandidates='2';
  if(values.trainingMode!=='resume' && [undefined,'off','auto'].includes(values.v2TrainOnsetReadout))values.v2TrainOnsetReadout='auto';
  values.__composerV2ReleaseDefaults=1;
  changed=true;
  }
  if(values.__composerV2PhraseMusicDefaults!==1) {
    Object.assign(values,{v2GeneratePhraseSearch:'layout',v2GeneratePhraseBeats:'4',v2GeneratePhraseCommitBeats:'2'});
    if(values.trainingMode==='fresh' || !values.trainingMode)Object.assign(values,{v2TrainAudioLayers:'4',v2TrainSongLayers:'2',v2TrainMusicContext:'auto'});
    values.__composerV2PhraseMusicDefaults=1;changed=true;
  }
  return changed;
}

function restoreSettings() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(settingsStorageKey)); } catch (_) { saved = null; }
  // Dataset selection has its own storage so resetting generation/training
  // fields does not silently discard a carefully curated source list.
  try {
    const datasetSaved = JSON.parse(localStorage.getItem(datasetStorageKey));
    if (Array.isArray(datasetSaved?.datasets)) datasetWorkspaceState.selected = new Set(datasetSaved.datasets.map((item) => String(item.manifest || item.path || item)).filter(Boolean));
    datasetWorkspaceState.selectionTouched = datasetSaved?.selection_touched === true || datasetWorkspaceState.selected.size > 0;
    if (datasetSaved?.active_dataset) datasetWorkspaceState.activePath = String(datasetSaved.active_dataset);
  } catch (_) { /* optional */ }
  const values = saved?.values;
  if (!values || typeof values !== "object") return restoreLegacyDatasetPaths();
  if (saved.version !== 2 && values.modelFamily === "song_form_v1" && values.trainingMode === "fresh") {
    Object.assign(values, {modelFamily:"song_form_v2",formDModel:"256",formLayers:"4",formNhead:"8",formDimFeedforward:"0",formPlannerWidth:"128"});
    if (values.checkpointDir === "checkpoints/song_form_v1") values.checkpointDir="checkpoints/song_form_v2";
    restoredSettingNotice = "从头训练入口已更新为精确有理数事件模型；默认采用 256 维紧凑配置。旧模型仍可通过模型族选择生成与续训。";
  }
  if (Number(values.conflictBacktrackBudget) === -1) {
    values.conflictBacktrackUnlimited = true;
    values.conflictBacktrackBudget = "4";
  }
  // Migrate the former external-reference switch. An old executable path is
  // never used by the embedded detector, including after a browser reload.
  if (["report", "reject"].includes(values.runtimeReferenceMode) && values.conflictDetectionEnabled !== true) {
    values.conflictDetectionEnabled = true;
    values.conflictRepairMode = values.runtimeReferenceMode === "reject" ? "reject" : "none";
  }
  if(typeof migrateComposerV2ReleaseDefaults==="function" && migrateComposerV2ReleaseDefaults(values)) {
    try { localStorage.setItem(settingsStorageKey,JSON.stringify(saved)); } catch (_) {}
    restoredSettingNotice='Composer v2 已更新默认：先定节奏，再比较配置；提前试写4拍，每次保留前2拍。新训练采用4层音频编码及全曲音乐上下文，续训沿用检查点设置。';
  }
  migrateSongFormQualityDefaults(values);
  migrateEventSetLearningDefaults(values);
  if (!values.__eventSetSharedContinuationDefaults && (values.composerLearningPolicy ?? 'auto') === 'auto' && Number(values.composerTransitionExitWeight) === .5) delete values.composerTransitionExitWeight;
  let restored = 0;
  settingsFields().forEach((target) => {
    if (!Object.prototype.hasOwnProperty.call(values, target.id)) return;
    const value = values[target.id];
    if (target.type === "checkbox") target.checked = value === true;
    else if (value !== undefined && value !== null) target.value = String(value);
    restored += 1;
  });
  const beforeSmoke = values.__v2TrainPlanBeforeSmoke;
  const smokeFields = Object.keys(V2_TRAIN_SMOKE_PLAN);
  if (beforeSmoke && smokeFields.every(id => typeof beforeSmoke[id] === 'string')) {
    v2TrainPlanBeforeSmoke = Object.fromEntries(smokeFields.map(id => [id, beforeSmoke[id]]));
  }
  restoreV2ConditionSettings(values.__composerV2Conditions);
  const manualTouched = values.__manualTouched || {};
  const learningRate = element("#learningRate");
  const contextTokens = element("#contextTokens");
  if (learningRate) {
    const legacyDefault = values.trainingMode === "finetune" ? 0.00002 : 0.0002;
    const inferLegacyTouched = Object.prototype.hasOwnProperty.call(values, "learningRate")
      && Number.isFinite(Number(values.learningRate))
      && Number(values.learningRate) !== legacyDefault;
    learningRate.dataset.touched = manualTouched.learningRate === true
      || (manualTouched.learningRate === undefined && inferLegacyTouched) ? "1" : "";
  }
  if (contextTokens) contextTokens.dataset.touched = manualTouched.contextTokens === true ? "1" : "";
  for (const [id, provenance] of Object.entries(values.__songFormEvidenceProvenance || {})) {
    if (!["formQualityProfile", "formPositionAlignmentLossWeight", "formPositionAlignmentToleranceMs",
      "formPlannerContextSeconds", "formCurveBinSeconds"].includes(id)) continue;
    const node = element(`#${id}`);
    if (!node || !provenance || typeof provenance !== "object") continue;
    node.dataset.evidenceManual = provenance.manual === true ? "1" : "";
    if (typeof provenance.inherited === "string") node.dataset.evidenceInheritedProfile = provenance.inherited;
    if (typeof provenance.inheritedValue === "string") node.dataset.evidenceInheritedValue = provenance.inheritedValue;
  }
  const levelInput = element("#level");
  if (levelInput) {
    const previousLevel = levelInput.value;
    const normalizedLevel = normalizeGenerationLevel(previousLevel);
    const levelWasRepaired = Number(previousLevel) !== normalizedLevel;
    levelInput.value = String(normalizedLevel);
    // The range slider is a view of the same condition field.  Always sync it
    // from the validated number input, including when an old browser saved a
    // separate, stale slider value.
    const levelRange = element("#levelRange");
    if (levelRange) levelRange.value = String(normalizedLevel);
    if (levelWasRepaired) {
      restoredSettingNotice = `检测到旧设置中的目标等级 ${previousLevel || "空值"} 超出当前支持范围 12.00–15.00，已恢复为 13.30；原强度曲线仍可使用，也可按需重新生成。`;
      if (String(intensityCurveState.profile?.method || "").startsWith("learned_music_structure_")) {
        intensityCurveState.conditionChanged = true;
        saveIntensityCurveState();
      }
      // Canonicalize the saved copy as well, so a reload before any further
      // user input does not repeat the repair indefinitely.
      try {
        saved.values.level = String(normalizedLevel);
        if (Object.prototype.hasOwnProperty.call(saved.values, "levelRange")) saved.values.levelRange = String(normalizedLevel);
        localStorage.setItem(settingsStorageKey, JSON.stringify(saved));
      } catch (_) { /* storage disabled or stale */ }
    }
  }
  document.querySelectorAll("[data-number-for]").forEach((mirror) => {
    const source = document.getElementById(mirror.dataset.numberFor);
    if (source) mirror.value = source.value;
  });
  return restored > 0;
}

function restoreLegacyDatasetPaths() {
  try {
    const saved = JSON.parse(localStorage.getItem(datasetStorageKey));
    if (!saved?.manifest) return false;
    if (element("#rawRoot")) element("#rawRoot").value = saved.raw_root || "";
    if (element("#cacheDir")) element("#cacheDir").value = saved.cache_dir || "cache";
    if (element("#manifestPath")) element("#manifestPath").value = saved.manifest;
    if (Array.isArray(saved.datasets)) {
      datasetWorkspaceState.selected = new Set(saved.datasets.map((item) => String(item.manifest || item.path || item)).filter(Boolean));
    }
    datasetWorkspaceState.selectionTouched = saved.selection_touched === true || datasetWorkspaceState.selected.size > 0;
    if (saved.active_dataset) datasetWorkspaceState.activePath = String(saved.active_dataset);
    return true;
  } catch (_) { return false; }
}

function saveDatasetWorkspace() {
  try {
    const current = JSON.parse(localStorage.getItem(datasetStorageKey) || "{}") || {};
    current.version = 2;
    current.datasets = [...datasetWorkspaceState.selected];
    current.selection_touched = datasetWorkspaceState.selectionTouched;
    current.active_dataset = datasetWorkspaceState.activePath || "";
    current.raw_root = valueOrDefault("#rawRoot", "").trim();
    current.cache_dir = valueOrDefault("#cacheDir", "").trim();
    current.manifest = valueOrDefault("#manifestPath", "").trim();
    localStorage.setItem(datasetStorageKey, JSON.stringify(current));
  } catch (_) { /* localStorage is optional */ }
}

function discardSettings() {
  if (settingsSaveTimer) clearTimeout(settingsSaveTimer);
  settingsSaveTimer = null;
  try {
    localStorage.removeItem(settingsStorageKey);
    localStorage.removeItem(datasetStorageKey);
  } catch (_) { /* ignore */ }
}

function generationFieldIds() {
  return [...GENERATION_PARAM_FIELDS, ...radarKeys.map((entry) => entry[0]),
    ...styleKeys.map((entry) => entry[0])];
}

function captureGenerationDefaults() {
  const snapshot = {};
  generationFieldIds().forEach((id) => {
    const target = document.getElementById(id);
    if (!target) return;
    snapshot[id] = target.type === "checkbox" ? target.checked : target.value;
  });
  return snapshot;
}

function resetGenerationDefaults() {
  invalidateLearnedCurveForCondition();
  Object.entries(generationDefaults).forEach(([id, value]) => {
    const target = document.getElementById(id);
    if (!target) return;
    if (target.type === "checkbox") target.checked = value === true;
    else target.value = String(value);
    const mirror = document.querySelector(`[data-number-for="${id}"]`);
    if (mirror) mirror.value = target.value;
  });
  if(v2ConditionActive)for(const key of Object.keys(v2ConditionState))otherFamilyConditionValues[key]=generationDefaults[key]??otherFamilyConditionValues[key];
  if (levelPresetProfile) applyConditionPreset();
  resetV2ConditionState();
  updateGenerationCapabilityControls();
  saveSettings();
  const status = element("#resetGenerationStatus");
  if (status) status.textContent = "已恢复生成面板默认值；训练参数未改动。";
}

function setConditionValue(key, value) {
  if(v2ConditionActive && key!=='level')return;
  const target = document.getElementById(key);
  if (!target) return;
  target.value = String(value);
  const mirror = document.querySelector(`[data-number-for="${key}"]`);
  if (mirror) mirror.value = String(value);
}

function applyConditionPreset() {
  if(v2ConditionActive){updateV2ConditionControls();updateConditionRecipeDisplay();saveSettings();return;}
  if (!levelPresetProfile) throw new Error("等级参考正在载入；稍后重试。载入失败时仍可使用高级微调。");
  const level = Number(valueOrDefault("#level", 13.3));
  if (!Number.isFinite(level) || level < 12 || level > 15) throw new Error("等级预设支持 12.00–15.00，请填写范围内的数字。");
  const selected = selectedConditionPreferences();
  const strength = Number(valueOrDefault("#conditionPreferenceStrength", 0.5));
  const preset = composeLevelCondition(levelPresetProfile, level, selected, strength);
  Object.entries(preset.radar).forEach(([name, value]) => setConditionValue(name, value));
  Object.entries(preset.style).forEach(([name, value]) => setConditionValue(name, value));
  lastConditionRecipe = { level, preferences: selected, strength,
    calibration_schema: levelPresetProfile.schema, source: levelPresetProfile.source };
  updateConditionRecipeDisplay(preset);
  saveSettings();
}

function selectedConditionPreferences() {
  return Object.keys(preferenceDefinitions).filter(key => element(`#${preferenceDefinitions[key].field}`)?.checked);
}

function interpolateLevelPreset(profile, level) {
  if (!Number.isFinite(level) || level < 12 || level > 15) throw new Error("等级须为 12–15 的有限数字");
  const rows = profile.levels;
  const highIndex = rows.findIndex(row => Number(row.level) >= level - 1e-9);
  const high = rows[highIndex < 0 ? rows.length - 1 : highIndex];
  const low = rows[Math.max(0, (highIndex < 0 ? rows.length - 1 : highIndex) - 1)];
  const mix = high.level === low.level ? 0 : Math.max(0, Math.min(1, (level - low.level) / (high.level - low.level)));
  const result = { level, radar: {}, style: {}, controls: {}, support: high.support,
    raw_count: high.raw_count, exact_count: high.exact_count, fallback: high.fallback,
    interpolated: mix > 1e-8 && mix < 1 - 1e-8 };
  for (const section of ["radar", "style"]) {
    for (const key of Object.keys(high[section])) {
      result[section][key] = Number(low[section][key]) * (1 - mix) + Number(high[section][key]) * mix;
      const lo = low.controls[key], hi = high.controls[key];
      result.controls[key] = { p10: lo.p10 * (1 - mix) + hi.p10 * mix, p90: lo.p90 * (1 - mix) + hi.p90 * mix };
    }
  }
  return result;
}

function composeLevelCondition(profile, level, preferences = [], strength = 0.5) {
  if (!Number.isFinite(strength) || strength < 0 || strength > 1) throw new Error("偏好力度须在 0–1 内");
  const result = interpolateLevelPreset(profile, level);
  const selected = [...new Set(preferences)].sort().map(key => preferenceDefinitions[key]).filter(Boolean);
  const base = { ...result.radar, ...result.style }, values = { ...base };
  if (selected.length && strength > 0) {
    // Average requested deviations: order-independent and repeated application
    // starts from the same level, so multi-select cannot compound the controls.
    for (const key of Object.keys(base)) {
      if (["note_density", "peak", "melody_following", "chord_bias", "pattern_reuse"].includes(key)) continue;
      const bounds = result.controls[key];
      let delta = 0;
      for (const preference of selected) {
        const axis = preference.axes[key] || 0;
        const target = axis > 0 ? Math.max(base[key], bounds.p90) : Math.min(base[key], bounds.p10);
        delta += Math.abs(axis) * (target - base[key]);
      }
      values[key] = Math.max(0, Math.min(1, base[key] + strength * delta / selected.length));
    }
    // Preserve the overall action-ratio mass rather than independently inflating
    // Tap, Slide, Hold and Touch. Touch Hold overlap already exists in the corpus.
    const ratios = ["tap_bias", "slide_bias", "hold_bias", "touch_bias"];
    const total = ratios.reduce((sum, key) => sum + values[key], 0);
    const baseTotal = ratios.reduce((sum, key) => sum + base[key], 0);
    if (total > 1e-9) ratios.forEach(key => { values[key] *= baseTotal / total; });
  }
  values.chord_bias = values.chord;
  // Complex/continuous slides are subsets of Slide notes in training statistics.
  values.slide_tricky = Math.min(values.slide_tricky, values.slide_bias);
  values.slide_sequence = Math.min(values.slide_sequence, values.slide_bias);
  for (const section of ["radar", "style"]) {
    for (const key of Object.keys(result[section])) result[section][key] = Number(Math.max(0, Math.min(1, values[key])).toFixed(3));
  }
  return result;
}

function updateConditionRecipeDisplay(preset = null) {
  const level = Number(valueOrDefault("#level", 13.3));
  const range = element("#levelRange");
  if (range && Number.isFinite(level)) range.value = String(level);
  const strength = Number(valueOrDefault("#conditionPreferenceStrength", 0.5));
  const label = element("#conditionPreferenceStrengthLabel");
  if (label) label.textContent = `${strength <= 0.3 ? "轻度" : strength <= 0.7 ? "明显" : "强烈"} · ${Math.round(strength * 100)}%`;
  for (const definition of Object.values(preferenceDefinitions)) {
    const input = element(`#${definition.field}`);
    input?.closest(".preference-card")?.classList.toggle("selected", input.checked);
  }
  const status = element("#conditionPresetStatus");
  if (!status) return;
  if (!preset) {
    status.textContent = "保留当前 Radar / Style 微调值。改变目标等级或偏好会重新计算；点击「重新应用」可回到当前组合。";
    return;
  }
  const names = selectedConditionPreferences().map(key => preferenceDefinitions[key].label);
  status.textContent = `${level.toFixed(2)} 等级基准${names.length ? ` ＋ ${names.join("、")}` : " · 常规写法"}已应用。整体密度目标约 ${(preset.radar.note_density * 16).toFixed(1)} 次起音/秒；偏好不额外提高密度与峰值。`;
  const evidence = element("#conditionPresetEvidence");
  if (evidence) evidence.textContent = `${preset.interpolated ? "该小数等级由相邻基准插值。" : ""}参考来自 full 训练集原谱；附近有效参考约 ${Number(preset.support || 0).toFixed(0)} 份，本档明确小数标签 ${Number(preset.exact_count || 0)} 份${Number(preset.raw_count || 0) < 20 ? "，直接样本较少，借用邻近等级平滑" : ""}。原谱整数 / 加号标签的参考权重降低；提供的小数标签也未经官方核验。这是生成起点，不是定数保证。`;
}

async function loadLevelPresetProfile() {
  const status = element("#conditionPresetStatus");
  try {
    const profile = await api("/api/download?path=src%2Fmaichart%2Fwebui%2Fstatic%2Fgeneration_presets.json");
    if (profile.schema !== "maichart-level-presets-v1" || !Array.isArray(profile.levels) || profile.levels.length !== 31) throw new Error("等级参考文件不完整");
    profile.levels.forEach((row, index) => {
      if (Math.abs(Number(row.level) - (12 + index * 0.1)) > 1e-6) throw new Error("等级参考排序无效");
      for (const [section, entries] of [["radar", radarKeys], ["style", styleKeys]]) {
        for (const [key] of entries) {
          const value = row[section]?.[key], bounds = row.controls?.[key];
          if (!Number.isFinite(value) || value < 0 || value > 1 || !Number.isFinite(bounds?.p10) || !Number.isFinite(bounds?.p90)
            || bounds.p10 < 0 || bounds.p90 > 1 || bounds.p10 > value || bounds.p90 < value) throw new Error(`等级参考 ${key} 无效`);
        }
      }
    });
    levelPresetProfile = profile;
    if(v2ConditionActive)updateV2ConditionControls();
    else if (!restoredSettings && !conditionEditedBeforeProfile) applyConditionPreset();
    else updateConditionRecipeDisplay();
    const button = element("#applyConditionPreset"); if (button) button.disabled = false;
  } catch (error) {
    if (status) status.textContent = `等级参考未载入：${error.message}。保留已有参数，请展开高级微调；可点击「重新应用」重试。`;
    const button = element("#applyConditionPreset"); if (button) button.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
function controlRows(containerId, entries) {
  const container = document.getElementById(containerId);
  if (!container) return;
  entries.forEach((entry) => {
    const [key, label, help = "", fallback = "0.5"] = entry;
    const row = document.createElement("div");
    row.className = "control-row";
    row.innerHTML = `<label for="${key}">${label}</label>`
      + `<input id="${key}" type="range" min="0" max="1" step="0.001" value="${fallback}">`
      + `<input data-number-for="${key}" type="number" min="0" max="1" step="0.001" value="${fallback}">`
      + (help ? `<small>${help}</small>` : "");
    const range = row.querySelector("input[type=range]");
    const number = row.querySelector("input[type=number]");
    range.addEventListener("input", () => { number.value = range.value; });
    number.addEventListener("input", () => { range.value = number.value; });
    container.appendChild(row);
  });
}

function setStatus(text, good = false) {
  const target = element("#serverStatus");
  if (!target) return;
  target.textContent = text;
  target.classList.toggle("good", good);
}

function setManifestStatus(text, kind = "") {
  const target = element("#manifestCheckStatus");
  if (!target) return;
  target.className = `inline-status ${kind}`.trim();
  target.textContent = text;
}

function setCheckpointStatus(text, kind = "") {
  const target = element("#checkpointInspectStatus");
  if (!target) return;
  target.className = `inline-status ${kind}`.trim();
  target.textContent = text;
}

function setGenerationStatus(text, kind = "") {
  const target = element("#generationCheckpointStatus");
  if (!target) return;
  target.className = `inline-status ${kind}`.trim();
  target.textContent = text;
}

function setGoldStatus(text, kind = "") {
  const target = element("#goldStatus");
  if (!target) return;
  target.className = `inline-status ${kind}`.trim();
  target.textContent = text;
}

function setAcceptanceStatus(text, kind = "") {
  const target = element("#acceptanceStatus");
  if (!target) return;
  target.className = `inline-status ${kind}`.trim();
  target.textContent = text;
}

function numberOrDefault(selector, fallback) {
  const target = element(selector);
  const raw = target ? String(target.value).trim() : "";
  if (raw === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function checkedOrDefault(selector, fallback) {
  const target = element(selector);
  return target ? target.checked : fallback;
}

function valueOrDefault(selector, fallback) {
  const target = element(selector);
  return target ? target.value : fallback;
}

function formatList(items) {
  return Array.isArray(items) && items.length
    ? items.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join("\n")
    : "无";
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function currentTrainingMode() {
  return valueOrDefault("#trainingMode", "fresh");
}

function currentModelFamily() {
  return valueOrDefault("#modelFamily", "chart_composer_v2");
}

function currentGenerationFamily() {
  const selected = currentModelFamily();
  const inspected = generationInspection.data?.model_family;
  return inspected && generationInspection.path === valueOrDefault("#checkpointPath", "").trim() ? inspected : selected;
}

function updateGenerationCapabilityControls() {
  updateV2BatchControls();
  const data = generationInspection.data;
  const caps = data?.generation_capabilities || {};
  const checked = generationInspection.path === valueOrDefault("#checkpointPath", "").trim()
    && data?.valid === true;
  const fine = element("#fineNoteStrength");
  if (fine) fine.disabled = !(checked && caps.fine_note_strength === true);
  document.querySelectorAll("[data-fine-note-value]").forEach((button) => {
    button.disabled = !fine || fine.disabled;
  });
  const fineValue = element("#fineNoteStrengthValue");
  if (fineValue && fine) fineValue.textContent = Number(fine.value).toFixed(2);
  const onset = element("#localOnsetBiasStrength");
  if (onset) {
    onset.disabled = !(checked && caps.local_onset_guidance === true);
  }
  const onsetValue = element("#localOnsetBiasStrengthValue");
  if (onsetValue && onset) onsetValue.textContent = Number(onset.value).toFixed(2);
  const onsetStatus = element("#localOnsetGuidanceStatus");
  if (onsetStatus) {
    onsetStatus.textContent = !checked
      ? "检查当前生成 checkpoint 后确认能力。"
      : caps.local_onset_guidance === true
        ? "当前 checkpoint 已训练局部起音辅助头；0 关闭，建议先试 0.3–0.6。"
        : "当前 checkpoint 不支持此指导；使用 chart_onsets_v1 训练并完成能力检查后可用。";
  }
  const conflictEnabled = checked && caps.conflict_detection === true
    && checkedOrDefault("#conflictDetectionEnabled", false);
  ["#detectHardConflicts", "#detectSoftConflicts", "#conflictRepairMode", "#conflictPreset",
    "#tapOnSlideThresholdMs", "#conflictBacktrackWindows", "#conflictBacktrackBudget", "#conflictBacktrackUnlimited", "#sustainRepairMode",
    ...CONFLICT_KIND_FIELDS.map(([, id]) => `#${id}`)].forEach((selector) => {
    const target = element(selector);
    if (target) target.disabled = !conflictEnabled;
  });
  const noteRepair = element("#noteRepairMode");
  const backtrackBudget = element("#conflictBacktrackBudget");
  if (backtrackBudget) backtrackBudget.disabled = !conflictEnabled
    || checkedOrDefault("#conflictBacktrackUnlimited", false);
  if (noteRepair) noteRepair.disabled = !(conflictEnabled && valueOrDefault("#conflictRepairMode", "none") === "local");
  const retries = element("#conflictRetries");
  if (retries) retries.disabled = !(conflictEnabled && valueOrDefault("#conflictRepairMode", "none") !== "none");

}

function updateV2BatchControls() {
  const rankSelect=element('#v2GeneratePhraseRankPolicy');
  if(rankSelect) {
    const searchDisabled=element('#v2GeneratePhraseSearch')?.disabled;
    const searchMode=valueOrDefault('#v2GeneratePhraseSearch','layout');
    const candidateCount=Number(valueOrDefault('#v2GenerateCandidates',2));
    rankSelect.disabled=Boolean(searchDisabled || searchMode==='event' || candidateCount<=1);
  }
  const field=element('#v2GenerateParallel');if(!field)return;
  const count=Number(valueOrDefault('#v2GenerateCount',1));field.disabled=count<=1;
  const active=Math.min(count,Number(field.value)||1),help=element('#v2GenerateParallelHelp');
  if(help)help.textContent=count<=1?'生成多张时，可设置同时运行的张数。':`最多同时运行 ${active} 张，其余排队；并行数越高，显存占用越大。`;
}

// A field marked for one family applies to its descendants too: the
// cross-window model is the event pipeline plus context, so every event_v1
// setting still applies to it.  A field may declare several families at once.
function familyFieldVisible(declared, current) {
  return String(declared).split(/\s+/).filter(Boolean).some((name) => (
    name === current || (name === "event_v1" && current === "event_ctx_v1")
  ));
}

function updateSongFormGenerationOptions() {
  const isSongForm = ["song_form_v1", "song_form_v2"].includes(currentGenerationFamily());
  const isComposer = currentGenerationFamily() === "chart_composer_v1";
  const curveMode = element("#intensityCurveMode");
  const legacyCurveModes = curveMode ? [...curveMode.options].filter((option) => ["soft", "strict"].includes(option.value)) : [];
  legacyCurveModes.forEach((option) => { option.disabled = isSongForm; });
  const hardMode = curveMode?.querySelector('option[value="hard"]');
  if (hardMode) hardMode.textContent = isSongForm ? "hard（放宽幅度裁剪）" : "hard（严苛跟随）";
  legacyCurveModes.forEach((option) => {
    option.textContent = option.value === "soft"
      ? (isComposer ? "soft（温和指导）" : "soft（软指导 / 仅规划，兼容旧版）")
      : (isComposer ? "strict（跟随并限制单拍物量）" : "strict（密度上限 / 生成后上限，兼容旧版）");
  });
  const curveSource = element("#intensityCurveSource");
  const audioCurveSource = curveSource?.querySelector('option[value="audio"]');
  if (isSongForm && ["soft", "strict"].includes(curveMode?.value)) {
    if (songFormCurveModeBeforeRestriction === null) songFormCurveModeBeforeRestriction = curveMode.value;
    curveMode.value = "follow";
    saveSettings();
  }
  if (audioCurveSource) audioCurveSource.disabled = isSongForm;
  if (isSongForm && !songFormCurveRestrictionsActive) {
    songFormCurveRestrictionsActive = true;
    if (curveSource) songFormCurveSourceBeforeRestriction = curveSource.value;
  }
  if (isSongForm && ["audio", "auto"].includes(curveSource?.value)) {
    curveSource.value = "model";
    saveSettings();
  } else if (!isSongForm && songFormCurveRestrictionsActive) {
    if (curveSource && songFormCurveSourceBeforeRestriction) curveSource.value = songFormCurveSourceBeforeRestriction;
    if (curveMode && songFormCurveModeBeforeRestriction) curveMode.value = songFormCurveModeBeforeRestriction;
    songFormCurveRestrictionsActive = false;
    songFormCurveSourceBeforeRestriction = null;
    songFormCurveModeBeforeRestriction = null;
    saveSettings();
  }
  const modelCurveSource = curveSource?.querySelector('option[value="model"]');
  if (modelCurveSource) modelCurveSource.disabled = false;
  const curveSave = element("#intensitySaveButton");
  if (curveSave) curveSave.textContent = "保存并返回生成页";
}

function updateFamilyFields() {
  const v2Generation=currentGenerationFamily()==="chart_composer_v2";
  const checked=generationInspection.path===valueOrDefault("#checkpointPath","").trim()
    && generationInspection.data?.valid;
  // Start with the newest route. Inspection can disable unsupported old
  // capabilities, but never removes their controls from the v2 workbench.
  for(const field of document.querySelectorAll('[data-v2-probability-audit]'))field.hidden=!v2Generation;
  const splitSampling=v2Generation && (!checked || generationInspection.data?.generation_capabilities?.continuous_time);
  for (const field of document.querySelectorAll('[data-v2-action-sampling]')) {
    field.hidden=!v2Generation;
    field.querySelector('input').disabled=!splitSampling;
  }
  for(const field of document.querySelectorAll('[data-v2-phrase-search]')) {
    field.hidden=!v2Generation;
    field.querySelector('input,select').disabled=!splitSampling;
  }
  updateV2BatchControls();
  const phraseHelp=element('#v2PhraseSearchHelp');
  if(phraseHelp)phraseHelp.textContent=checked && !splitSampling?'这个旧模型不支持比较整段写法，将使用“逐步生成”。':'默认先确定一小段音符出现的时间和数量，再比较不同键位与动作组合；“节奏与配置一起选”还会尝试不同的出音时间。增加比较会更慢。';
  const onsetSupported=v2Generation && (!checked || generationInspection.data?.generation_capabilities?.onset_readout);
  for (const field of document.querySelectorAll('[data-v2-onset-readout]')) {
    field.hidden=!v2Generation; field.querySelector('input').disabled=!onsetSupported;
  }
  const capabilities=checked?generationInspection.data?.generation_capabilities:null;
  const musicInfo=element('#v2GenerationMusicInfo');
  if(musicInfo)musicInfo.textContent=!checked?'音乐理解：新版模型支持成对参考与前方采音；生成时使用检查点已学到的方案。':capabilities?.paired_music
    ?'音乐理解：已启用成对音乐参考与前方采音，随检查点自动使用。'
    :'音乐理解：使用检查点原有方案；新能力可在新目录微调启用。';
  const builtInReference=!!capabilities?.rhythm_structure;
  const referenceChoice=v2Generation && (!checked || capabilities?.continuous_time) && !builtInReference;
  for (const field of document.querySelectorAll('[data-v2-reference-selection]')) {
    field.hidden=!v2Generation; field.querySelector('select').disabled=!referenceChoice;
  }
  const referenceHelp=element('#v2ReferenceSelectionHelp');
  if(referenceHelp && v2Generation)referenceHelp.textContent=builtInReference
    ?'这个模型已经学过如何参考前面相似音乐段的写法，会自动使用对应方式，无需在这里调整。'
    :(checked && !capabilities?.continuous_time)
      ?'这个旧模型不支持按相似音乐寻找前文。'
      :'让模型参考前面相似音乐段的写法，帮助前后呼应。通常保持推荐项；实验项会改变旧模型寻找参考的方式，效果可能变好也可能变差。';
  updateV2ConditionControls();
  document.querySelectorAll("#generationTargetGroup .condition-presets > :not(.level-recipe-row)").forEach(target=>{target.hidden=v2Generation;});
  document.querySelectorAll('#generationTargetGroup .condition-advanced').forEach(target=>{target.hidden=v2Generation && !target.querySelector('#v2ConditionStatus');});
  const legacyConditionHint=element('#v2ConditionStatus')?.closest('details')?.querySelector('p:not(#v2ConditionStatus)');
  if(legacyConditionHint)legacyConditionHint.hidden=v2Generation;
  const bpmInput=element('#bpm');if(bpmInput)bpmInput.placeholder='留空则自动估计';const genDeviceField=element("#generationDevice")?.closest(".field");if(genDeviceField)genDeviceField.hidden=v2Generation;const firstInput=element("#first");if(firstInput)firstInput.disabled=v2Generation && checkedOrDefault("#v2GenerateAutoFirst",true);
  const targetHeading=element("#generationTargetGroup .workflow-group-heading h3");
  if(targetHeading)targetHeading.textContent=v2Generation?"选择目标等级":"选择等级与偏好";
  const targetHint=element("#generationTargetGroup .workflow-group-heading p");
  if(targetHint)targetHint.textContent=v2Generation?"等级作为模型软条件，实际难度仍需试玩确认。":"先使用等级基准，再按需要微调。";
  if(!generationInspection.data || generationInspection.path!==valueOrDefault("#checkpointPath","").trim()) {
    const status=element("#generationCheckpointStatus");
    if(status)status.textContent=v2Generation?"已使用 Composer v2 最新默认；提交时自动检查 checkpoint，旧模型不支持的项会停用。":"尚未检查。选择 checkpoint 并检查后显示对应模型族参数。";
  }
  document.querySelectorAll("[data-training-family]").forEach((target) => {
    target.hidden = !familyFieldVisible(target.dataset.trainingFamily, currentModelFamily());
  });
  document.querySelectorAll("[data-generation-family]").forEach((target) => {
    target.hidden = !familyFieldVisible(target.dataset.generationFamily, currentGenerationFamily());
  });
  const songTraining = ["song_form_v1", "song_form_v2"].includes(currentModelFamily());
  const streamTraining = currentModelFamily() === "chart_stream_v1";
  const composerTraining = ["chart_composer_v1","chart_composer_v2"].includes(currentModelFamily());
  for(const id of ["epochs","learningRate","batchSize","gradAccumulation","device","qualityEnabled","resetOptimizer"]) {
    const field=element("#"+id)?.closest(".field, .checkbox-field");if(field)field.hidden=currentModelFamily()==="chart_composer_v2";
  }
  if(element('#trainingQuality'))element('#trainingQuality').hidden=currentModelFamily()==='chart_composer_v2';
  const quickSettings=element("#trainingRunGroup")?.querySelector?.(".quick-settings");if(quickSettings)quickSettings.hidden=currentModelFamily()==="chart_composer_v2";
  ["#dModel", "#nhead", "#numLayers", "#dimFeedforward", "#trainMaxTokens", "#segmentBeats", "#maxAudioTokens",
   "#conditionMaskProb", "#conditionDefault", "#conditionDropout", "#contextDropout", "#dropout", "#weightDecay",
   "#plannerProfile", "#planLossWeight", "#planTeacherMix", "#onsetProfile", "#onsetLossWeight", "#actionStateProfile",
   "#localPoolingPolicy", "#noAmp", "#audioCacheGb", "#cudaMemoryBudgetGb", "#prewarmWorkers", "#maxSteps", "#validationMaxSteps",
   "#minDifficultyIndex", "#maxDifficultyIndex", "#minLevel", "#maxLevel", "#slideOversample", "#touchOversample", "#fineRhythmOversample",
   "#fineRhythmRatioThreshold", "#contextBeats", "#audioMarginBeats", "#contextTokens", "#overlapBeats",
   "#preferenceEnabled", "#preferenceWeight", "#preferenceMaxPairs"].forEach((selector) => {
    const node = element(selector); const field = node?.closest(".field, .checkbox-field");
    if (field && (songTraining || streamTraining || composerTraining) && !field.dataset.trainingFamily) field.hidden = true;
    else if (field && !songTraining && !streamTraining && !composerTraining && !field.dataset.trainingFamily) field.hidden = false;
  });
  const songHint = element("#songFormTrainingControls"); if (songHint) songHint.hidden = !songTraining;
  const composerHint = element("#chartComposerTrainingControls"); if (composerHint) composerHint.hidden = currentModelFamily()!=="chart_composer_v1";
  const songGenerationHint = element("#songFormGenerationControls"); if (songGenerationHint) songGenerationHint.hidden = !["song_form_v1", "song_form_v2"].includes(currentGenerationFamily());
  for (const id of ['trainingArchitecture','trainingLearning']) {
    const section=element('#'+id);if(section)section.hidden=composerTraining;
  }
  if (songTraining || streamTraining) {
    element("#trainingArchitecture")?.setAttribute("open", "");
    element("#trainingLearning")?.setAttribute("open", "");
  }
  if (["song_form_v1", "song_form_v2"].includes(currentGenerationFamily())) {
    document.querySelectorAll(".condition-advanced").forEach((details) => details.setAttribute("open", ""));
    element("#generationCurveGroup")?.setAttribute("open", "");
  }
  updateSongFormGenerationOptions();
  updateGenerationCapabilityControls();
  const family = currentModelFamily();
  const familyHelp = element("#modelFamilyHelp");
  if (familyHelp) {
    familyHelp.textContent = family === "chart_composer_v2"
      ? "chart_composer_v2（当前主路线）：连续生成整曲事件序列，默认保留16拍历史、每16个事件整理上下文、局部音乐额外前瞻4拍。MERT 在准备阶段自动加载。"
      : family === "chart_stream_v1"
      ? "chart_stream_v1：整曲连续事件流；计算块只管理显存，不产生音乐边界。音频、近期谱面、持续动作和近期历史摘要共同参与。"
      : family === "chart_composer_v1"
      ? "chart_composer_v1：独立事件候选与动作槽位模型，按 Composer manifest 训练；checkpoint 只与同族 Composer 训练/生成兼容。"
      : family === "song_form_v2"
      ? "song_form_v2：直接联合生成精确有理数拍位与配置，音频按节拍/音乐事件查询。保留 Planner、段落关联、Radar / Style、负荷曲线和修订；旧 checkpoint 自动使用原生成方式。"
      : ["song_form_v1", "song_form_v2"].includes(family)
      ? "song_form_v1：全曲 blueprint、段落关联与草稿修订模型。Radar / Style 由 event_derived_radar_style_v1 条件监督提供；强度曲线需完成独立训练接入后才会成为真实生成约束。"
      : family === "event_v1"
      ? "事件模型使用 112 维 event_audio_v2_rhythm 和显式 null embedding；Planner profile 选择 music_structure_v2/v3 时使用音乐结构规划器，选择 legacy_v1 才是旧 planner。"
      : family === "event_ctx_v1"
        ? "跨窗口事件模型：使用 112 维 event_audio_v2_rhythm 和显式 null embedding；每个窗口额外看到「前面已写的音符」和「前后的音乐」。Planner profile 选择 music_structure_v2/v3 时学习局部音乐结构；checkpoint 与 event_v1 互不通用，必须单独训练。"
        : "旧固定帧基线，仅用于对照；不具备 event 模型的表示能力。";
  }
  const dirHelp = element("#checkpointDirHelp");
  if (dirHelp) {
    dirHelp.textContent = family === "chart_composer_v2"
      ? "composer_v2_last.pt / composer_v2_best.pt / composer_v2_inference.pt；与其他模型族分开保存。"
      : family === "chart_stream_v1"
      ? "chart_stream_v1 产物为 stream_last.pt；与 event / song_form checkpoint 不兼容。"
      : family === "chart_composer_v1"
      ? "chart_composer_v1 产物为 composer_last.pt；与其他模型族 checkpoint 不兼容。"
      : ["song_form_v1", "song_form_v2"].includes(family)
      ? "song-form 产物为 form_last.pt；resume 严格继承合同，finetune 请使用新的输出目录。"
      : family === "legacy_frame_v2"
        ? "旧基线输出 generator_last.pt / generator_epoch_NNN.pt，不能作为 event checkpoint 使用。"
        : "event_last.pt / event_best.pt / event_epoch_NNN.pt。建议使用独立输出目录。";
  }
  const bpmHelp = element("#bpmHelp");
  if (bpmHelp) {
    bpmHelp.textContent = v2Generation ? "留空时按上方「自动BPM分析」设置识别节拍：选「识别变速并校准全曲」时采用通过连续性检查的变速段，选恒定模式时不会自动写入变速表。填写 BPM 则使用手动恒定值，手动 tempo map 整表优先。勾选“自动识别首拍偏移”后会单独分析节拍相位（勾选时右侧手动首拍偏移停用）。" : currentGenerationFamily() === "legacy_frame_v2"
      ? "旧基线：留空自动分析；填写则覆盖为单一 BPM，tempo map 可进一步覆盖。"
      : "未填 tempo map 时，留空使用自动估计 BPM，填写则采用该值；auto 会额外使用高置信度变速点。填写 tempo map 后，其 beat 0 BPM 与变速点优先。";
  }
  const preprocessHelp = element("#preprocessFamilyStatus");
  if (preprocessHelp) preprocessHelp.textContent = family === "chart_composer_v2"
    ? "Composer v2 读取训练来源中的音频和谱面；MERT 特征存放在独立缓存中，不复用 v1 模型和训练状态。请先准备 manifest，再开始训练。"
    : family === "chart_stream_v1"
    ? "chart_stream_v1 复用 event source manifest 和有理音频缓存，但训练目标是整曲连续事件流；不生成分段 EOS 或 fixed-frame labels。"
    : family === "chart_composer_v1"
    ? "chart_composer_v1 直接读取 Composer manifest 与独立音频缓存；不会复用其他模型族的 checkpoint 合同。"
    : ["song_form_v1", "song_form_v2"].includes(family)
    ? "song-form 使用 event source manifest：解析 maidata、音频特征和精确时间缓存；不生成旧 fixed-frame labels/alignment。已有 event manifest 可直接复用。"
    : family === "legacy_frame_v2"
      ? "旧固定帧模型才使用 legacy preprocess build、division、frame labels 和 alignment。"
      : "event 模型使用 prepare_events：解析 maidata、音频特征和精确时间缓存，不生成 fixed-frame labels/alignment。";
}

function datasetPaths() {
  return {
    raw_root: valueOrDefault("#rawRoot", "").trim(),
    cache_dir: valueOrDefault("#cacheDir", "").trim(),
    manifest: valueOrDefault("#manifestPath", "").trim(),
  };
}

function selectedTrainingManifests() {
  return selectedDatasetItems().map((item) => item.manifest).filter(Boolean);
}

function trainingSourcePaths({ autoPreprocess = false } = {}) {
  const manifests = selectedTrainingManifests();
  // Preprocessing always uses its current single directory. Normal training
  // identifies its selected sources; no unrelated preprocessing path is sent.
  return manifests.length && !autoPreprocess
    ? { manifest: manifests[0], raw_root: null, cache_dir: null }
    : datasetPaths();
}

function assertTrainingSelection({ autoPreprocess = false } = {}) {
  if (!datasetWorkspaceState.selectionTouched && !datasetWorkspaceState.selected.size) return;
  const items = selectedDatasetItems();
  if (items.length !== datasetWorkspaceState.selected.size) throw new Error("数据集列表仍在刷新，请等待来源加载完成后再检查或训练。");
  if (!items.length) throw new Error("当前已清空训练数据集选择。请至少勾选一个数据集，或返回训练页重新确认旧 manifest。");
  const unprepared = items.filter((item) => !item.manifest || item.prepared === false);
  const current = datasetPaths();
  const canPrepareSingle = autoPreprocess && items.length === 1 && items[0].path === current.raw_root && items[0].manifest === current.manifest;
  if (unprepared.length && !canPrepareSingle) throw new Error(`选中的数据集尚未预处理：${unprepared.map((item) => item.name || item.path).join("、")}。请先进入训练页预处理。`);
}

async function assertDatasetBackend() {
  if (!selectedTrainingManifests().length) return;
  const status = await api("/api/status");
  if (status.training_capabilities?.multi_dataset_training !== true) {
    throw new Error("当前后端尚未加载多训练集功能，请重启 E 盘 WebUI 服务后再检查或训练。勾选的来源尚未提交训练。");
  }
}

function contextPayload() {
  return {
    context_beats: numberOrDefault("#contextBeats", 8),
    audio_margin_beats: numberOrDefault("#audioMarginBeats", 4),
    context_tokens: numberOrDefault("#contextTokens", 768),
    overlap_beats: numberOrDefault("#overlapBeats", 0),
    context_dropout: numberOrDefault("#contextDropout", 0.15),
  };
}

function preprocessPayload() {
  return {
    ...datasetPaths(),
    model_family: currentModelFamily(),
    sample_rate: Number(valueOrDefault("#prepSampleRate", 22050)),
    division: Number(valueOrDefault("#prepDivision", 16)),
    feature_profile: valueOrDefault("#prepFeatureProfile", "basic_v1"),
    limit: numberOrDefault("#prepLimit", null),
    force: checkedOrDefault("#prepForce", false),
  };
}

function formDevice(selector) {
  return valueOrDefault(selector, "auto");
}
function songFormRhythmSnapDivisions(enabled) {
  const divisions = [4, 8, 12, 16, 24, 32, 48, 64]
    .filter((division) => checkedOrDefault(`#formRhythmDivision${division}`, division <= 32));
  const custom = valueOrDefault("#formRhythmDivisionCustom", "").trim();
  if (custom) {
    for (const token of custom.split(",")) {
      const value = token.trim();
      if (!/^\d+$/.test(value)) throw new Error("其他分音请用逗号分隔的整数填写。");
      const division = Number(value);
      if (division < 1 || division > 768) throw new Error("分音范围为 1–768。");
      divisions.push(division);
    }
  }
  const unique = [...new Set(divisions)];
  if (unique.length > 32) throw new Error("候选分音去重后最多 32 种。");
  if (enabled && !unique.length) throw new Error("启用节奏近似时至少选择一种候选分音。");
  return unique;
}
function songFormRationalGridDivisions() {
  const raw = valueOrDefault("#formRationalGridDivisions", "4,6,8,12,16,24,32,48,64,96").trim();
  const values = raw ? raw.split(",").map((token) => token.trim()) : [];
  if (!values.length) throw new Error("有理拍近似至少需要一种候选分音；填 0 可关闭近似，而不是清空候选。");
  if (values.some((token) => !/^\d+$/.test(token))) throw new Error("有理拍近似候选请用逗号分隔的整数填写。");
  const unique = [...new Set(values.map(Number))];
  if (unique.some((value) => value < 1 || value > 768)) throw new Error("有理拍近似候选范围为 1–768。");
  if (unique.length > 32) throw new Error("有理拍近似候选去重后最多 32 种。");
  return unique;
}

function songFormRhythmGeometryUiState(mode, requested, checkpointProfile = "none") {
  const rank = {none: 0, exact_fraction_v1: 1, exact_fraction_v2: 2};
  const known = Object.keys(rank);
  const profile = known.includes(checkpointProfile) ? checkpointProfile : "none";
  if (mode === "resume") return { effective: profile, locked: true, upgradeNeedsExplicitChoice: false };
  const explicit = known.includes(requested) && requested !== "none";
  if (mode === "finetune" && profile !== "none") {
    // A checkpoint that already carries geometry keeps it unless the user
    // explicitly asks for a wider contract (v1 -> v2). Narrowing would discard
    // learned channels, so it stays on the stored profile.
    const migrate = explicit && rank[requested] > rank[profile];
    return { effective: migrate ? requested : profile, locked: false, upgradeNeedsExplicitChoice: false };
  }
  if (mode === "finetune") {
    // No stored contract: auto and none both keep the old path, so an explicit
    // choice is required before any geometry training happens.
    return { effective: explicit ? requested : "none", locked: false,
      upgradeNeedsExplicitChoice: !explicit };
  }
  return { effective: requested === "auto" ? "exact_fraction_v2" : requested,
    locked: false, upgradeNeedsExplicitChoice: false };
}

function updateSongFormRhythmGeometryControls() {
  const selector = element("#formRhythmGeometryProfileSelect");
  const help = element("#formRhythmGeometryProfileHelp");
  const loss = element("#formRhythmLossWeight");
  if (!selector) return;
  const inspection = checkpointInspection.data;
  const checkpointProfile = inspection?.valid === true
    ? inspection.model_config?.rhythm_geometry_profile : "none";
  const state = songFormRhythmGeometryUiState(currentTrainingMode(), selector.value, checkpointProfile);
  selector.disabled = currentModelFamily() !== "song_form_v2" || state.locked;
  if (loss) loss.disabled = currentModelFamily() !== "song_form_v2"
    || currentTrainingMode() === "resume" || state.effective === "none";
  if (!help) return;
  if (currentModelFamily() !== "song_form_v2") {
    help.textContent = "仅 song_form_v2 支持精确节奏几何训练。";
  } else if (currentTrainingMode() === "resume") {
    help.textContent = `resume 严格继承 checkpoint 的 ${state.effective} 几何合同与辅助损失设置。`;
  } else if (state.upgradeNeedsExplicitChoice) {
    help.textContent = "当前 checkpoint 没有数值几何合同；选择 exact_fraction_v1 或 exact_fraction_v2 才会迁移。自动与 none 都保持旧路径。";
  } else if (state.effective === "exact_fraction_v2") {
    help.textContent = "拍位坐标以小节与分组相位为基准，同一音乐位置在任何分段内取值一致；24 分在拍 1 与拍 2 因此是两个类别。小节相位来自拍号表，未提供拍号时退化为绝对拍位的四拍平滑基并标记为未知。";
  } else if (state.effective === "exact_fraction_v1") {
    help.textContent = "使用分段内相对拍位的数值坐标与相邻间隔辅助监督；不吸附固定分音，也不使用固定帧率。";
  } else {
    help.textContent = "关闭数值几何辅助，保持旧输入合同。";
  }
}

function updateSongFormQualityProfileControls() {
  const selector = element("#formQualityProfile");
  if (!selector) return;
  const mode = currentTrainingMode();
  const config = checkpointInspection.data?.valid ? checkpointInspection.data.model_config || {} : {};
  const saved = config.quality_profile || "legacy";
  const effective = mode === "resume" || (mode === "finetune" && selector.value === "auto")
    ? saved : selector.value === "auto" ? "musical_evidence_v1" : selector.value;
  const locked = mode === "resume";
  selector.disabled = locked || currentModelFamily() !== "song_form_v2";
  for (const id of ["formPositionAlignmentLossWeight", "formPositionAlignmentToleranceMs", "formPlannerContextSeconds", "formCurveBinSeconds"]) {
    const node = element(`#${id}`);
    if (node && !locked && effective === "musical_evidence_v1"
        && node.dataset?.evidenceInheritedProfile === "legacy" && node.dataset.evidenceManual !== "1"
        && node.value === node.dataset.evidenceInheritedValue) {
      // Values restored for resume describe the old contract. They are not
      // deliberate overrides of the explicitly selected upgrade defaults.
      node.value = "";
      delete node.dataset.evidenceInheritedProfile;
      delete node.dataset.evidenceInheritedValue;
    }
    if (node) node.disabled = locked || currentModelFamily() !== "song_form_v2" || effective === "legacy";
  }
  const help = element("#formQualityProfileHelp");
  if (help) help.textContent = locked
    ? `resume 继承 ${effective} 合同与相关权重。修改这些目标请使用 finetune 和新输出目录。`
    : effective === "musical_evidence_v1"
      ? "增强音乐证据：均衡音频时间覆盖、Planner 局部时序、1 秒负荷标签和音频—谱面位置辅助监督。官谱拍位只作为监督答案，解码仍可输出任意有理数拍。已有模型升级需 finetune，并使用新输出目录；新增目标需训练后才生效。"
      : "保持已有模型的输入与训练合同。finetune 的自动模式继承 checkpoint；显式选择增强音乐证据才会升级，现有权重不会自动获得新能力。";
}

function markSongFormEvidenceInput(node) {
  if (!node?.dataset) return;
  node.dataset.evidenceManual = "1";
  delete node.dataset.evidenceInheritedProfile;
  delete node.dataset.evidenceInheritedValue;
}

function updateSongFormQualityControls(mode = currentTrainingMode()) {
  const family = currentModelFamily();
  const resume = mode === "resume";
  const supported = ["song_form_v1", "song_form_v2"].includes(family);
  const common = ["#formPerturbApplyProb", "#formPerturbCurveNoiseStd", "#formPerturbCurveDropoutProb",
    "#formPerturbHistoryCorruptionProb", "#formDiversityLossWeight", "#formQualityRecipe", "#formQualityApplyRecipe"];
  for (const selector of common) {
    const node = element(selector); if (node) node.disabled = !supported || resume;
  }
  for (const selector of ["#formPerturbOnsetDropProb", "#formPerturbOnsetAddProb", "#formPerturbOnsetShiftMs"]) {
    const node = element(selector); if (node) node.disabled = family !== "song_form_v1" || resume;
  }
  const repetition = element("#formRepetitionPenaltyWeight"); if (repetition) repetition.disabled = true;
  const status = element("#formQualityTrainingStatus");
  if (status) status.textContent = resume
    ? "resume 继承并锁定 checkpoint 的扰动与辅助损失；旧模型未记录时按关闭值继承。调整请使用 finetune 和新输出目录。"
    : `${family === "song_form_v2" ? "v2 联合生成拍位，不使用 v1 起音骨架扰动。" : "v1 可扰动输入起音骨架，监督答案保持原谱。"}默认使用推荐的轻度曲线与历史扰动；多样性损失默认关闭；重复惩罚与 Planner Guidance 尚未接入。`;
}

function applySongFormQualityRecipe() {
  if (currentTrainingMode() === "resume") return;
  const selected = element("#formQualityRecipe")?.value || "history";
  // Field order: apply, curve noise, curve dropout, history, v1 onset drop/add/shift, diversity.
  // "history" is the recommended recipe and must match the index.html defaults.
  const recipes = {
    off: [1, 0, 0, 0, 0, 0, 0, 0], light: [.3, .02, 0, 0, 0, 0, 0, 0], history: [.5, .03, 0, .02, .05, 0, 0, 0],
  };
  if (!Object.hasOwn(recipes, selected)) return;
  const ids = ["formPerturbApplyProb", "formPerturbCurveNoiseStd", "formPerturbCurveDropoutProb", "formPerturbHistoryCorruptionProb",
    "formPerturbOnsetDropProb", "formPerturbOnsetAddProb", "formPerturbOnsetShiftMs", "formDiversityLossWeight"];
  ids.forEach((id, index) => { const node = element(`#${id}`); if (node) node.value = String(recipes[selected][index]); });
  const repetition = element("#formRepetitionPenaltyWeight"); if (repetition) repetition.value = "0";
  updateSongFormQualityControls(); saveSettings();
}

function songFormTrainingPayload(options = {}) {
  const mode = currentTrainingMode(); const source = trainingSourcePaths(options); const manifests = selectedTrainingManifests();
  const selectedFamily = typeof currentModelFamily === "function" ? currentModelFamily() : "song_form_v1";
  const rhythmGeometryProfile = valueOrDefault("#formRhythmGeometryProfileSelect", "auto");
  if (!["auto", "none", "exact_fraction_v1", "exact_fraction_v2"].includes(rhythmGeometryProfile)) throw new Error("节奏几何训练方案无效。");
  const rawRhythmLossWeight = valueOrDefault("#formRhythmLossWeight", "").trim();
  const rhythmLossWeight = rawRhythmLossWeight === "" ? null : Number(rawRhythmLossWeight);
  if (rhythmLossWeight !== null && (!Number.isFinite(rhythmLossWeight) || rhythmLossWeight < 0 || rhythmLossWeight > 10)) {
    throw new Error("节奏几何辅助损失权重必须在 0–10 之间。");
  }
  const positionEventLossWeight = Number(valueOrDefault("#formPositionEventLossWeight", .25));
  if (!Number.isFinite(positionEventLossWeight) || positionEventLossWeight < 0 || positionEventLossWeight > 10) {
    throw new Error("拍位均衡损失权重必须在 0–10 之间。");
  }
  const qualityProfile = valueOrDefault("#formQualityProfile", "auto");
  if (!["auto", "legacy", "musical_evidence_v1"].includes(qualityProfile)) throw new Error("音乐证据训练方案无效。");
  const evidenceOptions = {};
  for (const [key, selector, maximum, positive, label] of [
    ["position_alignment_loss_weight", "#formPositionAlignmentLossWeight", 10, false, "音频—谱面位置辅助损失权重"],
    ["position_alignment_tolerance_ms", "#formPositionAlignmentToleranceMs", 100, true, "音频—谱面位置监督容差"],
    ["planner_context_seconds", "#formPlannerContextSeconds", 30, false, "Planner 局部时序范围"],
    ["curve_bin_seconds", "#formCurveBinSeconds", 30, true, "负荷标签时间粒度"],
  ]) {
    const inactive = mode !== "resume" && (qualityProfile === "legacy"
      || (typeof element === "function" && element(selector)?.disabled === true));
    const raw = inactive ? "" : valueOrDefault(selector, "").trim();
    const value = raw === "" ? null : Number(raw);
    if (value !== null && (!Number.isFinite(value) || value < 0 || (positive && value === 0) || value > maximum
        || (key === "position_alignment_tolerance_ms" && value < 1))) {
      throw new Error(`${label}须${positive ? "大于 0 且" : "在 0–"}${positive ? `不超过 ${maximum}` : maximum}，或留空自动。`);
    }
    evidenceOptions[key] = value;
  }
  const qualityFields = [
    ["perturb_apply_prob", "#formPerturbApplyProb", .5, 1, "扰动应用概率"],
    ["perturb_curve_noise_std", "#formPerturbCurveNoiseStd", .03, .3, "曲线噪声标准差"],
    ["perturb_curve_dropout_prob", "#formPerturbCurveDropoutProb", 0, .5, "曲线整点置零概率"],
    ["perturb_history_corruption_prob", "#formPerturbHistoryCorruptionProb", .02, .3, "历史配置扰动概率"],
    ["perturb_onset_drop_prob", "#formPerturbOnsetDropProb", .05, .5, "起音骨架删除概率"],
    ["perturb_onset_add_prob", "#formPerturbOnsetAddProb", 0, .5, "起音骨架添加强度"],
    ["perturb_onset_shift_ms", "#formPerturbOnsetShiftMs", 0, 50, "起音骨架偏移标准差"],
    ["diversity_loss_weight", "#formDiversityLossWeight", 0, .3, "配置表示多样性损失权重"],
    ["repetition_penalty_weight", "#formRepetitionPenaltyWeight", 0, .5, "重复惩罚损失权重"],
  ];
  const quality = {};
  for (const [key, selector, fallback, maximum, label] of qualityFields) {
    let value = Number(valueOrDefault(selector, fallback));
    if (mode !== "resume" && (key === "repetition_penalty_weight"
      || (selectedFamily !== "song_form_v1" && key.startsWith("perturb_onset_")))) value = 0;
    if (!Number.isFinite(value) || value < 0 || value > maximum) throw new Error(`${label}必须在 0–${maximum} 之间。`);
    quality[key] = value;
  }
  return { model_family: selectedFamily, manifest: source.manifest || manifests[0] || "", checkpoint_dir: valueOrDefault("#checkpointDir", "").trim(),
    ...(selectedFamily === "song_form_v2" ? {quality_profile: qualityProfile, ...evidenceOptions} : {}),
    meter_map: valueOrDefault("#formTrainingMeterMap", "").trim() ? parseMeterMap("#formTrainingMeterMap") : null,
    ...quality,
    manifests, dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
    quality_enabled: checkedOrDefault("#qualityEnabled", false),
    resume_data_policy: mode === "resume" ? valueOrDefault("#formResumeDataPolicy", "strict") : "strict",
    mode, init_checkpoint: mode === "fresh" ? null : valueOrDefault("#trainingCheckpointPath", "").trim(), epochs: Number(valueOrDefault("#epochs", 20)), batch_size: Number(valueOrDefault("#batchSize", 4)),
    device: formDevice("#device"), lr: Number(valueOrDefault("#learningRate", .0002)), weight_decay: Number(valueOrDefault("#weightDecay", .01)),
    cuda_memory_budget_gb: Number(valueOrDefault("#formCudaMemoryBudgetGb", 0)),
    grad_accumulation: Number(valueOrDefault("#gradAccumulation", 1)), save_every: Number(valueOrDefault("#saveEvery", 0)),
    seed: valueOrDefault("#trainingSeed", "").trim() === "" ? 0 : Number(valueOrDefault("#trainingSeed", 0)),
    max_steps: Number(valueOrDefault("#maxSteps", 0)), limit_charts: Number(valueOrDefault("#formLimitCharts", 0)),
    prepare_workers: Number(valueOrDefault("#formPrepareWorkers", 1)),
    blueprint_batch_size: valueOrDefault("#formBlueprintBatchSize", "").trim() === "" ? null : Number(valueOrDefault("#formBlueprintBatchSize", "")),
    blueprint_workers: Number(valueOrDefault("#formBlueprintWorkers", 0)),
    blueprint_prefetch: Number(valueOrDefault("#formBlueprintPrefetch", 8)),
    precision: valueOrDefault("#formPrecision", "auto"),
    loader_workers: Number(valueOrDefault("#formLoaderWorkers", 0)),
    prefetch_factor: Number(valueOrDefault("#formPrefetchFactor", 2)),
    layout_prepare_threads: Number(valueOrDefault("#formLayoutPrepareThreads", 4)),
    attention_padding: valueOrDefault("#formAttentionPadding", "compact"),
    log_interval: Number(valueOrDefault("#formLogInterval", 10)),
    checkpoint_interval_steps: Number(valueOrDefault("#formCheckpointIntervalSteps", 500)),
    audio_cache_gb: Number(valueOrDefault("#formAudioCacheGb", 1)),
    audio_cache_mode: valueOrDefault("#formAudioCacheMode", "lru"),
    sample_cache_mb: Number(valueOrDefault("#formSampleCacheMb", 256)),
    control_profile: valueOrDefault("#formControlProfile", "auto"), condition_drop_prob: Number(valueOrDefault("#formConditionDropout", .15)),
    event_decision_loss_weight: Number(valueOrDefault("#formEventDecisionLossWeight", .25)),
    position_event_loss_weight: positionEventLossWeight,
    rhythm_geometry_profile: rhythmGeometryProfile, rhythm_loss_weight: rhythmLossWeight,
    dropout: Number(valueOrDefault("#formDropout", .1)), blueprint_loss_weight: Number(valueOrDefault("#formBlueprintLossWeight", 1)),
    load_curve_loss_weight: Number(valueOrDefault("#formCurveLossWeight", .2)), relation_loss_weight: Number(valueOrDefault("#formRelationLossWeight", .1)),
    max_region_seconds: Number(valueOrDefault("#formMaxRegionSeconds", 14)), d_model: Number(valueOrDefault("#formDModel", 256)), layers: Number(valueOrDefault("#formLayers", 4)), nhead: Number(valueOrDefault("#formNhead", 8)),
    dim_feedforward: selectedFamily === "song_form_v2" ? Number(valueOrDefault("#formDimFeedforward", 0)) : 0,
    planner_width: Number(valueOrDefault("#formPlannerWidth", 128)),
    max_tokens: Number(valueOrDefault("#formMaxTokens", 2048)), max_memory_tokens: Number(valueOrDefault("#formMaxMemoryTokens", 512)), dropout: Number(valueOrDefault("#formDropout", .1)) };
}
// Shared by "开始训练" and the one-click button, so both send the same settings.
function trainingPayload(options = {}) {
  if (currentModelFamily() === "chart_composer_v2") return composerV2TrainingPayload(options);
  if (currentModelFamily() === "chart_stream_v1") return chartStreamTrainingPayload(options);
  if (currentModelFamily() === "chart_composer_v1") return chartComposerTrainingPayload(options);
  if (["song_form_v1", "song_form_v2"].includes(currentModelFamily())) return songFormTrainingPayload(options);
  const mode = currentTrainingMode();
  // planner_profile values: "auto", "music_structure_v3", "music_structure_v2",
  // "legacy_v1".  The generation panel only reads it to describe the checkpoint
  // it inspected; training chooses the profile.
  return {
    ...trainingSourcePaths(options),
    manifests: selectedTrainingManifests(),
    dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
    validation_manifest: valueOrDefault("#validationManifestPath", "").trim() || null,
    model_family: currentModelFamily(),
    checkpoint_dir: valueOrDefault("#checkpointDir", "").trim(),
    epochs: Number(valueOrDefault("#epochs", 20)),
    max_steps: Number(valueOrDefault("#maxSteps", 0)),
    validation_max_steps: Number(valueOrDefault("#validationMaxSteps", 0)),
    batch_size: Number(valueOrDefault("#batchSize", 16)),
    segment_beats: Number(valueOrDefault("#segmentBeats", 16)),
    segment_frames: Number(valueOrDefault("#segmentFrames", 512)),
    max_len: Number(valueOrDefault("#maxLen", 4096)),
    grad_accumulation: Number(valueOrDefault("#gradAccumulation", 2)),
    max_tokens: Number(valueOrDefault("#trainMaxTokens", 768)),
    max_audio_tokens: Number(valueOrDefault("#maxAudioTokens", 1024)),
    min_difficulty_index: Number(valueOrDefault("#minDifficultyIndex", 4)),
    max_difficulty_index: Number(valueOrDefault("#maxDifficultyIndex", 6)),
    min_level: Number(valueOrDefault("#minLevel", 12)),
    max_level: Number(valueOrDefault("#maxLevel", 15)),
    condition_default: Number(valueOrDefault("#conditionDefault", 0.5)),
    plan_loss_weight: Number(valueOrDefault("#planLossWeight", 0.1)),
    no_amp: checkedOrDefault("#noAmp", false),
    d_model: Number(valueOrDefault("#dModel", 256)),
    nhead: Number(valueOrDefault("#nhead", 4)),
    num_layers: Number(valueOrDefault("#numLayers", 4)),
    dim_feedforward: Number(valueOrDefault("#dimFeedforward", 1024)),
    dropout: Number(valueOrDefault("#dropout", 0.1)),
    device: valueOrDefault("#device", "auto"),
    seed: valueOrDefault("#trainingSeed", "").trim() === "" ? null : Number(element("#trainingSeed").value),
    weight_decay: Number(valueOrDefault("#weightDecay", 0.01)),
    audio_cache_gb: Number(valueOrDefault("#audioCacheGb", 4)),
    cuda_memory_budget_gb: Number(valueOrDefault("#cudaMemoryBudgetGb", 0)),
    prewarm_workers: valueOrDefault("#prewarmWorkers", "auto"),
    condition_mask_prob: Number(valueOrDefault("#conditionMaskProb", 0.5)),
    slide_oversample: Number(valueOrDefault("#slideOversample", 2)),
    touch_oversample: Number(valueOrDefault("#touchOversample", 1)),
    fine_rhythm_oversample: Number(valueOrDefault("#fineRhythmOversample", 1)),
    fine_rhythm_ratio_threshold: Number(valueOrDefault("#fineRhythmRatioThreshold", 0.25)),
    ...contextPayload(),
    include_unusable: checkedOrDefault("#includeUnusable", false),
    training_mode: mode,
    training_checkpoint: mode === "fresh" ? null : valueOrDefault("#trainingCheckpointPath", "").trim(),
    lr: Number(valueOrDefault("#learningRate", 0.0002)),
    save_every: Number(valueOrDefault("#saveEvery", 5)),
    patience: Number(valueOrDefault("#patience", 15)),
    reset_optimizer: checkedOrDefault("#resetOptimizer", false),
    feature_profile: valueOrDefault("#featureProfile", "basic_v1"),
    condition_dropout: Number(valueOrDefault("#conditionDropout", 0)),
    limit: numberOrDefault("#prepLimit", null),
    force: checkedOrDefault("#prepForce", false),
    quality_enabled: checkedOrDefault("#qualityEnabled", false),
    quality_annotations: qualityState.annotationManifest === qualityManifest() ? (qualityState.annotationPath || null) : null,
    preference_enabled: checkedOrDefault("#preferenceEnabled", false),
    preference_weight: Number(valueOrDefault("#preferenceWeight", 0.05)),
    preference_max_pairs: Number(valueOrDefault("#preferenceMaxPairs", 32)),
    planner_profile: valueOrDefault("#plannerProfile", "auto"),
    plan_teacher_mix: Number(valueOrDefault("#planTeacherMix", 0.5)),
    ...(currentModelFamily() === "legacy_frame_v2" ? {} : {
      onset_profile: valueOrDefault("#onsetProfile", "auto"),
      onset_loss_weight: Number(valueOrDefault("#onsetLossWeight", 0.2)),
      local_pooling_policy: valueOrDefault("#localPoolingPolicy", "auto"),
      action_state_profile: valueOrDefault("#actionStateProfile", "auto"),
      context_budget_policy: "auto",
      chart_filter_policy: "auto",
    }),
  };
}

const STREAM_GROUNDING_FIELDS = [
  ['streamContinuationProfile','continuation_profile','auto'],
  ['streamContextReplayTokens','context_replay_tokens',256],
  ['streamPrefixMaskSpanEvents','prefix_mask_span_events',8],
  ['streamCycleExitLossWeight','cycle_exit_loss_weight',.25],
  ['streamSymmetryAugmentation','symmetry_augmentation','mirror_lr'],
  ['streamGroundingProfile','grounding_profile','auto'],
  ['streamDetailAudioNodes','detail_audio_nodes',8192], ['streamLocalAudioNeighbors','local_audio_neighbors',32],
  ['streamLocalAudioSeconds','local_audio_seconds',2], ['streamArchiveEvents','archive_events',4096],
  ['streamArchiveRetrievals','archive_retrievals',8], ['streamArchiveMinSimilarity','archive_min_similarity',.5],
  ['streamHistoryNoteDropout','history_note_dropout',.15], ['streamPrefixNoteDropout','prefix_note_dropout',.1],
  ['streamHistoryPositionDropout','history_position_dropout',.1],
  ['streamHistoryPositionSpanEvents','history_position_span_events',4],
  ['streamHistoryPositionKeepEvents','history_position_keep_events',2],
  ['streamConditionDropout','condition_dropout',.1],
  ['streamAlignmentLossWeight','alignment_loss_weight',.15], ['streamAlignmentToleranceMs','alignment_tolerance_ms',15],
  ['streamCurveSlopeLossWeight','curve_slope_loss_weight',.05],
  ['streamEventBalanceLossWeight','event_balance_loss_weight',.2],
];

function chartStreamTrainingPayload(options = {}) {
  const mode = currentTrainingMode();
  return {
    ...trainingSourcePaths(options), manifests: selectedTrainingManifests(),
    dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
    model_family: "chart_stream_v1", mode,
    ...Object.fromEntries(STREAM_GROUNDING_FIELDS.map(([id,key,fallback]) =>
      [key, typeof fallback === 'string' ? valueOrDefault('#'+id,fallback) : Number(valueOrDefault('#'+id,fallback))])),
    manifest: valueOrDefault("#manifestPath", "").trim(),
    checkpoint_dir: valueOrDefault("#checkpointDir", "").trim(),
    init_checkpoint: mode === "fresh" ? null : valueOrDefault("#trainingCheckpointPath", "").trim(),
    epochs: Number(valueOrDefault("#epochs", 10)), batch_size: Number(valueOrDefault("#batchSize", 1)),
    grad_accumulation: Number(valueOrDefault("#gradAccumulation", 1)),
    d_model: Number(valueOrDefault("#streamDModel", 256)), layers: Number(valueOrDefault("#streamLayers", 4)),
    nhead: Number(valueOrDefault("#streamNhead", 8)), dim_feedforward: Number(valueOrDefault("#streamDimFeedforward", 768)),
    max_tokens: Number(valueOrDefault("#streamMaxTokens", 1024)), max_history_tokens: Number(valueOrDefault("#streamMaxHistoryTokens", 256)),
    max_stream_memory: Number(valueOrDefault("#streamMemoryTokens", 32)),
    dropout: Number(valueOrDefault("#streamDropout", .1)), seed: Number(valueOrDefault("#streamSeed", 0)),
    quality_enabled: checkedOrDefault("#qualityEnabled", false),
    plan_loss_weight: Number(valueOrDefault("#streamPlanLossWeight", .15)),
    event_decision_loss_weight: Number(valueOrDefault("#streamEventDecisionLossWeight",.25)),
    lr: Number(valueOrDefault("#learningRate", .0002)), weight_decay: Number(valueOrDefault("#streamWeightDecay", .01)),
    device: formDevice("#device"), precision: valueOrDefault("#streamPrecision", "auto"),
    attention_backend: valueOrDefault("#streamAttentionBackend", "auto"),
    hand_backend: valueOrDefault("#streamHandBackend", "auto"),
    execution_mode: valueOrDefault("#streamExecutionMode", "auto"),
    cuda_graph_budget_gb: Number(valueOrDefault("#streamCudaGraphBudgetGb", 2)),
    audio_cache_gb: Number(valueOrDefault("#streamAudioCacheGb", 8)),
    prepare_workers: Number(valueOrDefault("#streamPrepareWorkers", 4)),
    prefetch_batches: Number(valueOrDefault("#streamPrefetchBatches", 2)),
    data_workers: Number(valueOrDefault("#streamDataWorkers", 4)),
    cpu_threads: Number(valueOrDefault("#streamCpuThreads", 0)),
    input_cache_mode: valueOrDefault("#streamInputCacheMode", 'mapped'),
    structure_cache_mb: Number(valueOrDefault("#streamStructureCacheMb",256)),
    optimizer_implementation: valueOrDefault("#streamOptimizerImplementation",'auto'),
    input_residency: valueOrDefault("#streamInputResidency",'auto'),
    resident_cache_gb: Number(valueOrDefault("#streamResidentCacheGb",0)),
    resident_memory_reserve_gb: Number(valueOrDefault("#streamResidentMemoryReserveGb",0)),
    cache_validation_records: Number(valueOrDefault("#streamCacheValidationRecords",0)),
    max_region_seconds: Number(valueOrDefault("#streamMaxRegionSeconds", 15)),
    limit_charts: Number(valueOrDefault("#streamLimitCharts", 0)),
    log_interval: Number(valueOrDefault("#streamLogInterval", 25)),
    checkpoint_interval_steps: Number(valueOrDefault("#streamCheckpointIntervalSteps", 500)),
    cache_dir: valueOrDefault("#streamCacheDir", "cache/chart_stream_v1"),
  };
}

const EVENT_SET_LEARNING_POLICIES = {"phrase_context_policy": ["ordered_phrase_v2", "causal_phrase_v1", "off"], "duration_audio_policy": ["interval_v1", "off"], "physical_policy": ["runtime_hard_v2", "legacy_report_v1"], "transition_policy": ["music_exit_v3", "music_exit_v2", "music_exit_v1", "off"], "decoder_policy": ["prenorm_independent_v2", "postnorm_v1"], "window_balance_policy": ["sqrt_window_v2", "equal_window_v1"], "timing_output_policy": ["absolute_grid_v1", "relative_delta_v1"], "timing_grid_audio_policy": ["query_pool_v3", "query_pool_v2", "candidate_pool_v1", "off"], "timing_group_policy": ["phase_gap_v2", "phase_only_v1"], "literal_output_policy": ["type_blocks_v1", "flat_family_v1"], "event_family_policy": ["audio_state_phrase_v4", "audio_prior_transition_v3", "audio_semantic_v2", "semantic_mass_v1", "legacy_v1"], "prefix_augmentation_input_policy": ["musical_loop_recovery_v7", "musical_loop_recovery_v8", "musical_loop_recovery_v6", "musical_loop_recovery_v5", "phase_recovery_v4", "causal_window_config_time_first_decision_v3"], "history_window_policy": ["block_aligned_v2", "rolling_legacy_v1"], "continuation_policy": ["pattern_runs_v2", "typed_runs_v1", "off"], "sensor_load_policy": ["separate_touch_v1", "button_only_v1"], "gap_continuation_policy": ["off", "probability_v2", "additive_v1"], "motion_continuation_policy": ["context_motion_v3", "motion_distinct_v2", "motion_continuation_v1", "off"], "lane_load_policy": ["direct_buttons_v1", "off"], "recovery_policy": ["greedy_mixed_v4", "greedy_mixed_v3", "mixed_exit_v2", "persistent_v1"]};
EVENT_SET_LEARNING_POLICIES.event_family_policy.unshift('shared_continuation_v1');
EVENT_SET_LEARNING_POLICIES.recovery_target_policy=['causal_correspondence_v1','authored_next_v1'];
EVENT_SET_LEARNING_POLICIES.phrase_context_policy.unshift('shared_ordered_v4','aligned_phrase_v3');
EVENT_SET_LEARNING_POLICIES.template_policy=['family_preserving_v1','cross_family_v1'];
EVENT_SET_LEARNING_POLICIES.candidate_state_policy=['selected_native_v1','cursor_v1'];
EVENT_SET_LEARNING_POLICIES.plan_kind_policy=['unified_first_arrival_v6','unified_first_arrival_v5','phrase_arranged_hazard_v4','audio_hazard_v3','rate_integral_v2','off'];
EVENT_SET_LEARNING_POLICIES.prefix_augmentation_input_policy.unshift('continuous_draft_recovery_v11','musical_arrangement_recovery_v10','musical_entry_recovery_v9');
EVENT_SET_LEARNING_POLICIES.draft_policy=['beat_ordered_draft_v1','off'];
EVENT_SET_LEARNING_POLICIES.event_family_policy.unshift('shared_music_mixture_v2');
EVENT_SET_LEARNING_POLICIES.literal_set_policy=['permuted_unique_v1','canonical_v1'];
EVENT_SET_LEARNING_POLICIES.song_choice_objective_policy=['event_credit_v1','profile_only_v1'];
EVENT_SET_LEARNING_POLICIES.recovery_policy.unshift('first_step_phrase_v6','certified_phrase_v5');
const EVENT_SET_POLICY_LABELS = {"phrase_arranged_hazard_v4":"音乐事件率＋整句倍率（旧版）","musical_arrangement_recovery_v10":"单步入口与主动休止（旧版）","audio_hazard_v3":"音乐事件率直接判断（对照）","rate_integral_v2":"历史解码器叠加事件率（旧版对照）","musical_entry_recovery_v9":"漏写入口恢复＋真实休止对照（v9）","audio_state_phrase_v4": "音频先验＋持续状态与短句共同决定类型（v4）", "audio_prior_transition_v3": "音频先验与类型转折（v3）", "audio_semantic_v2": "音频语义校准（v2）", "semantic_mass_v1": "语义类型概率（v1）", "legacy_v1": "旧版基线（v1）", "musical_loop_recovery_v8": "长循环出口训练＋正常历史对照（v8）", "musical_loop_recovery_v7": "长循环错误历史恢复（v7）", "musical_loop_recovery_v6": "音乐循环与错拍恢复（v6）", "musical_loop_recovery_v5": "音乐循环历史恢复（v5）", "phase_recovery_v4": "错拍拍位恢复（v4）", "causal_window_config_time_first_decision_v3": "旧版短段首步恢复（v3）", "greedy_mixed_v4": "确定性生成＋构造出口混合恢复（v4）", "greedy_mixed_v3": "确定性与随机混合恢复（v3）", "mixed_exit_v2": "重复出口混合恢复（v2）", "persistent_v1": "持续保存错误历史恢复（v1）", "music_exit_v3": "比较当前光标之前音乐的转折学习（v3）", "music_exit_v2": "音乐转折残差学习（v2）", "music_exit_v1": "音乐转折联合学习（v1）", "off": "关闭（对照）", "prenorm_independent_v2": "逐层独立的前置归一化（v2）", "postnorm_v1": "旧版后置归一化（v1）", "sqrt_window_v2": "按短段长度平方根平衡（v2）", "equal_window_v1": "每个短段等权（v1）", "absolute_grid_v1": "绝对拍位网格（v1）", "relative_delta_v1": "相对拍距（v1）", "query_pool_v3": "候选音频与时序逐项匹配（v3）", "query_pool_v2": "候选音频与时序标量匹配（v2）", "candidate_pool_v1": "候选时刻音频池化（v1）", "phase_gap_v2": "拍内位置与间隔联合分组（v2）", "phase_only_v1": "仅拍内位置分组（v1）", "type_blocks_v1": "按音符类型分块输出（v1）", "flat_family_v1": "旧版平铺类型输出（v1）", "block_aligned_v2": "历史窗口按训练块对齐（v2）", "rolling_legacy_v1": "旧版滚动窗口（v1）", "pattern_runs_v2": "跟踪重复写法的连续状态（v2）", "typed_runs_v1": "按类型跟踪连续状态（v1）", "separate_touch_v1": "分别计算按钮与Touch负荷（v1）", "button_only_v1": "仅计算按钮负荷（v1）", "probability_v2": "以概率表达间隔延续（v2）", "additive_v1": "旧版加法间隔延续（v1）", "context_motion_v3": "结合上下文的动作衔接（v3）", "motion_distinct_v2": "区分动作写法的衔接（v2）", "motion_continuation_v1": "基础动作延续（v1）", "direct_buttons_v1": "显式按钮键位负荷（v1）", "runtime_hard_v2": "依据运行证据判定硬冲突（v2）", "legacy_report_v1": "旧版物理报告（v1）"};
Object.assign(EVENT_SET_POLICY_LABELS,{unified_first_arrival_v6:'统一落点与无事件尾概率（新版）',unified_first_arrival_v5:'统一落点与无事件尾概率（旧版）',continuous_draft_recovery_v11:'连续留白与入口恢复片段（推荐）',beat_ordered_draft_v1:'按拍位组织的可调整草稿（推荐）',shared_continuation_v1:'共享续写状态（推荐）',causal_correspondence_v1:'时间恢复与可信配置分开（推荐）',authored_next_v1:'原谱下一事件（旧策略）',
  aligned_phrase_v3:'拍位对齐的有序短句（v3）',
  family_preserving_v1:'先选类型，类内保持模板（v1）',
  cross_family_v1:'旧版跨类型保持模板（v1）',
  shared_music_mixture_v2:'音乐与历史共同决定类型（新版）',
  event_credit_v1:'按实际续写效果学习整曲选择（新版）',profile_only_v1:'仅学习整曲统计偏好（旧版）',
  first_step_phrase_v6:'首步与短句连续配置恢复（新版）',
  certified_phrase_v5:'可信接缝与连续配置恢复（旧版）',
  permuted_unique_v1:'同拍集合无序续写（新版）',canonical_v1:'旧字典序',
  selected_native_v1:'选定落点原生持续状态（v1）',
  cursor_v1:'旧版游标持续状态（v1）'
});
function assertEventSetResumeLearningRateCapability(payload, backend) {
  if(payload.architecture_profile==='event_set_stream_v1' && payload.mode==='resume' && payload.reset_learning_rate===true && backend.capabilities?.event_set_resume_lr_reset!==true)
    throw new Error('当前服务尚不支持调整续训学习率，请在不影响活动任务的前提下重载更新后的 WebUI 服务。');
}
function chartComposerTrainingPayload(options = {}) {
  const mode = currentTrainingMode();
  const source = trainingSourcePaths(options);
  const manifests = selectedTrainingManifests();
  const validationSource=mode==='resume'?'resume':valueOrDefault('#composerValidationSource','auto');
  if(!['resume','auto','manual','off'].includes(validationSource))throw new Error('未知的验证数据来源');
  const validationPath=valueOrDefault('#composerValidationManifest','').trim();
  if(validationSource==='manual'&&!validationPath)throw new Error('使用指定独立验证集时，请填写验证集清单路径');
  const validationManifest=validationSource==='manual'?validationPath:null;
  const holdoutFraction=validationSource==='off'?0:Number(valueOrDefault('#composerHoldoutFraction',.05));
  if(valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1') {
    const payload={model_family:'chart_composer_v1',architecture_profile:'event_set_stream_v1',mode,
      manifest:source.manifest||manifests[0]||'',manifests,dataset_paths:selectedDatasetItems().map(item=>item.path).filter(Boolean),
      checkpoint_dir:valueOrDefault('#checkpointDir','').trim(),init_checkpoint:mode==='fresh'?null:valueOrDefault('#trainingCheckpointPath','').trim(),
      cache_dir:valueOrDefault('#composerCacheDir','cache/event_set_stream_v1'),device:formDevice('#device'),precision:valueOrDefault('#composerPrecision','bf16'),
      holdout_fraction:holdoutFraction,validation_manifest:validationManifest,validation_source:validationSource,generation_eval_seeds:valueOrDefault('#composerGenerationEvalSeeds','0,1')};
    for(const [key,id,fallback] of [['epochs','epochs',50],['batch_size','batchSize',16],['grad_accumulation','gradAccumulation',4],['lr','learningRate',.0005],
      ['d_model','composerDModel',384],['layers','composerLayers',8],['nhead','composerNhead',8],['dim_feedforward','composerDimFeedforward',1536],
      ['history_events','composerHistoryEvents',128],['rollout_steps','composerRolloutSteps',32],['window_seconds','composerWindowSeconds',8],
      ['prefix_augmentation_rate','composerPrefixAugmentationRate',.15],['generation_eval_temperature','composerGenerationEvalTemperature',.65],['audio_grid_seconds','composerAudioGridSeconds',.01],['plan_weight','composerPlanWeight',.1],['plan_blank_window_seconds','composerPlanBlankWindowSeconds',.5],
      ['validation_max_batches','composerValidationMaxBatches',32],['validation_batch_size','composerValidationBatchSize',4],['generation_eval_interval_epochs','composerGenerationEvalIntervalEpochs',1],
      ['warmup_steps','composerWarmupSteps',0],['data_workers','composerDataWorkers',0],['prefetch_batches','composerPrefetchBatches',2],['sample_workers','composerSampleWorkers',1],['audio_cache_gb','composerAudioCacheGb',1],['min_lr_ratio','composerMinLrRatio',.1],
      ['dropout','composerDropout',.1],['checkpoint_interval_steps','composerCheckpointIntervalSteps',500],['log_interval','composerLogInterval',25],
      ['max_steps','composerMaxSteps',0],['limit_charts','composerLimitCharts',0],['prepare_workers','composerPrepareWorkers',1],['cpu_threads','composerCpuThreads',2],
      ['seed','composerSeed',0],['recovery_interval','composerRecoveryInterval',25],['recovery_samples','composerRecoverySamples',2],['recovery_steps','composerRecoverySteps',2],
      ['configuration_recovery_steps','composerConfigurationRecoverySteps',8],['recovery_bridge_steps','composerRecoveryBridgeSteps',8],
      ['recovery_weight','composerRecoveryWeight',.1],['symbolic_pretrain_epochs','composerSymbolicPretrainEpochs',0],['generation_eval_songs','composerGenerationEvalSongs',0]])
      payload[key]=Number(valueOrDefault('#'+id,fallback));
    if(mode==='resume' && element('#composerResetLearningRate')?.checked)payload.reset_learning_rate=true;
    if(mode==='resume' && element('#composerRebatchResume')?.checked)payload.rebatch_resume=true;
    for(const [key,id,fallback] of [['training_window_policy','composerTrainingWindowPolicy','auto'],['phrase_execution_policy','composerPhraseExecutionPolicy','auto'],['detail_representation_policy','composerDetailRepresentationPolicy','auto'],['duration_feature_policy','composerDurationFeaturePolicy','auto']]) payload[key]=valueOrDefault('#'+id,fallback);
    payload.duration_objective_policy=valueOrDefault('#composerDurationObjectivePolicy','auto');
    payload.duration_calibration_weight=Number(valueOrDefault('#composerDurationCalibrationWeight',.1));
    for(const [key,id,fallback] of [['max_phrase_events','composerMaxPhraseEvents',64],['phrase_seam_events','composerPhraseSeamEvents',2],['phrase_modes','composerPhraseModes',4],['phrase_relation_weight','composerPhraseRelationWeight',.1],['draft_loss_weight','composerDraftLossWeight',.1],['draft_responsibility_temperature','composerDraftResponsibilityTemperature',1]]) payload[key]=Number(valueOrDefault('#'+id,fallback));
    for(const [key,id] of [['song_profile_policy','composerSongProfilePolicy'],['song_response_policy','composerSongResponsePolicy']])payload[key]=valueOrDefault('#'+id,'auto');
    payload.curve_preview_policy=valueOrDefault('#composerCurvePreviewPolicy','auto');
    payload.curve_preview_weight=Number(valueOrDefault('#composerCurvePreviewWeight',.05));
    payload.deployment_condition_fraction=Number(valueOrDefault('#composerDeploymentConditionFraction',.25));
    for(const [key,id,fallback] of [['song_profile_components','composerSongProfileComponents',4],['song_profile_weight','composerSongProfileWeight',.1],['song_response_modes','composerSongResponseModes',4],['song_response_group_size','composerSongResponseGroupSize',4]])payload[key]=Number(valueOrDefault('#'+id,fallback));
    payload.batch_order_policy=valueOrDefault('#composerBatchOrderPolicy','auto');
    payload.timing_compute=valueOrDefault('#composerTimingCompute','supervised');
    payload.detail_compute=valueOrDefault('#composerDetailCompute','serial');
    payload.learning_policy=valueOrDefault('#composerLearningPolicy','auto');
    if(mode!=='resume' && payload.learning_policy==='legacy'){payload.song_profile_policy='off';payload.song_response_policy='off';payload.curve_preview_policy='off';}
    if(mode!=='resume' && !['legacy','checkpoint'].includes(payload.learning_policy) && ['auto','unified_profile_response_v2'].includes(payload.song_response_policy))payload.song_response_modes=payload.song_profile_components;
    payload.transition_exit_weight=Number(valueOrDefault('#composerTransitionExitWeight',0));
    payload.event_family_audio_weight=Number(valueOrDefault('#composerEventFamilyAudioWeight',.25));
    payload.phrase_loss_weight=Number(valueOrDefault('#composerPhraseLossWeight',.05));
    const learningKeys=['draft_policy','plan_kind_policy','transition_policy','template_policy','candidate_state_policy','decoder_policy','window_balance_policy','timing_group_policy','literal_output_policy','history_window_policy','continuation_policy','sensor_load_policy','gap_continuation_policy','motion_continuation_policy','lane_load_policy','recovery_policy','timing_output_policy','event_family_policy','recovery_target_policy','prefix_augmentation_input_policy','timing_grid_audio_policy'];
    const legacyValues={draft_policy:'off',plan_kind_policy:'off',transition_policy:'off',decoder_policy:'postnorm_v1',window_balance_policy:'equal_window_v1',timing_group_policy:'phase_only_v1',literal_output_policy:'flat_family_v1',history_window_policy:'rolling_legacy_v1',continuation_policy:'off',sensor_load_policy:'button_only_v1',gap_continuation_policy:'additive_v1',motion_continuation_policy:'motion_continuation_v1',lane_load_policy:'off',recovery_policy:'persistent_v1',timing_output_policy:'relative_delta_v1',event_family_policy:'legacy_v1',recovery_target_policy:'authored_next_v1',prefix_augmentation_input_policy:'causal_window_config_time_first_decision_v3',timing_grid_audio_policy:'off'};
    Object.assign(legacyValues,{template_policy:'cross_family_v1',candidate_state_policy:'cursor_v1',literal_set_policy:'canonical_v1'});
    learningKeys.push('literal_set_policy','song_choice_objective_policy');
    legacyValues.song_choice_objective_policy='profile_only_v1';
    const savedPolicies=element('#composerLearningPolicy')?.dataset.savedPolicies;
    for(const key of learningKeys)payload[key]=mode==='resume'&&savedPolicies?JSON.parse(savedPolicies)[key]:(payload.learning_policy==='legacy'?legacyValues[key]:'auto');
    payload.draft_objective_policy=mode==='resume'&&savedPolicies
      ? (JSON.parse(savedPolicies).draft_objective_policy??'balanced_mean_v1')
      : payload.learning_policy==='legacy'?'balanced_mean_v1':valueOrDefault('#composerDraftObjectivePolicy','auto');
    for(const [key,id,legacy] of [['phrase_context_policy','composerPhraseContextPolicy','off'],['duration_audio_policy','composerDurationAudioPolicy','off'],['physical_policy',null,'legacy_report_v1']]) {
      payload[key]=mode==='resume'&&savedPolicies?(JSON.parse(savedPolicies)[key]??legacy):(payload.learning_policy==='legacy'?legacy:(id?valueOrDefault('#'+id,'auto'):'auto'));
    }
    for(const key of Object.keys(EVENT_SET_LEARNING_POLICIES)){const node=element('#eventSetPolicy_'+key);if(node&&mode!=='resume'&&node.value&&node.value!=='auto')payload[key]=node.value;}
    for(const node of document.querySelectorAll('[data-generation-eval-key]'))payload['generation_eval_'+node.dataset.generationEvalKey]=node.tagName==='SELECT'?node.value:(node.value.trim()===''?null:Number(node.value));
    if(mode!=='resume')for(const [key,id] of [['phrase_context_policy','composerPhraseContextPolicy'],['duration_audio_policy','composerDurationAudioPolicy']]){const value=valueOrDefault('#'+id,'auto');if(value&&value!=='auto')payload[key]=value;}
    payload.lr_schedule=valueOrDefault('#composerLrSchedule','cosine');
    const newStage=!['legacy','checkpoint'].includes(payload.learning_policy)&&mode!=='resume';
    const orderedDraft=payload.draft_policy==='beat_ordered_draft_v1'||(newStage&&payload.draft_policy==='auto');
    const unifiedTime=['unified_first_arrival_v5','unified_first_arrival_v6'].includes(payload.plan_kind_policy)||(newStage&&payload.plan_kind_policy==='auto');
    if(unifiedTime)payload.plan_weight=0;
    if(orderedDraft){payload.phrase_loss_weight=0;payload.phrase_relation_weight=0;}
    return payload;
  }
  return {
    model_family: "chart_composer_v1", mode,
    architecture_profile: valueOrDefault('#composerArchitectureProfile','audio_grounded_v3'),
    action_objective_policy: valueOrDefault('#composerActionObjectivePolicy','auto'),
    history_anchor_policy: valueOrDefault('#composerHistoryAnchorPolicy','auto'),
    history_memory_policy: valueOrDefault('#composerHistoryMemoryPolicy','auto'),
    history_attention_policy: valueOrDefault('#composerHistoryAttentionPolicy','auto'),
    head_timing_policy: valueOrDefault('#composerHeadTimingPolicy','auto'),
    cross_event_relation_policy: valueOrDefault('#composerCrossEventRelationPolicy','auto'),
    future_horizon_policy: valueOrDefault('#composerFutureHorizonPolicy','auto'),
    echo_prior_policy: valueOrDefault('#composerEchoPriorPolicy','auto'),
    timing_distribution_policy: valueOrDefault('#composerTimingDistributionPolicy','auto'),
    button_load_policy: valueOrDefault('#composerButtonLoadPolicy','auto'),
    fine_timing_policy: valueOrDefault('#composerFineTimingPolicy','auto'),
    density_feedback_policy: valueOrDefault('#composerDensityFeedbackPolicy','auto'),
    action_head_policy: valueOrDefault('#composerActionHeadPolicy','auto'),
    configuration_quality_policy: valueOrDefault('#composerConfigurationQualityPolicy','auto'),
    condition_mask_prob: Number(valueOrDefault('#composerConditionMaskProb',0)),
    action_state_policy: valueOrDefault('#composerActionStatePolicy','auto'),
    motif_memory_policy: valueOrDefault('#composerMotifMemoryPolicy','auto'),
    slide_factor_policy: valueOrDefault('#composerSlideFactorPolicy','auto'),
    structure_policy: valueOrDefault('#composerStructurePolicy','auto'),
    structure_backend: valueOrDefault('#composerStructureBackend','auto'),
    structure_dropout: Number(valueOrDefault('#composerStructureDropout',.15)),
    contact_policy: valueOrDefault('#composerContactPolicy','auto'),
    motif_max_events: Number(valueOrDefault('#composerMotifMaxEvents',16)),
    touch_layout_policy: valueOrDefault('#composerRecoveryTouchLayoutPolicy','auto'),
    manifest: source.manifest || manifests[0] || "",
    manifests, dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
    checkpoint_dir: valueOrDefault("#checkpointDir", "").trim(),
    init_checkpoint: mode === "fresh" ? null : valueOrDefault("#trainingCheckpointPath", "").trim(),
    epochs: Number(valueOrDefault("#epochs", 20)),
    batch_size: Number(valueOrDefault("#batchSize", 64)),
    lr: Number(valueOrDefault("#learningRate", .0002)),
    lr_schedule: valueOrDefault('#composerLrSchedule','auto'),
    warmup_steps: valueOrDefault('#composerWarmupSteps','')===''?null:Number(valueOrDefault('#composerWarmupSteps','')),
    min_lr_ratio: valueOrDefault('#composerMinLrRatio','')===''?null:Number(valueOrDefault('#composerMinLrRatio','')),
    grad_accumulation: Number(valueOrDefault('#gradAccumulation',4)),
    rebatch_resume: mode === 'resume' && checkedOrDefault('#composerRebatchResume',false),
    dropout: Number(valueOrDefault('#composerDropout',.1)),
    prefetch_batches: Number(valueOrDefault('#composerPrefetchBatches',2)),
    sample_workers: Number(valueOrDefault('#composerSampleWorkers',1)),
    cpu_threads: Number(valueOrDefault('#composerCpuThreads',2)),
    sample_locality: Number(valueOrDefault('#composerSampleLocality',4)),
    sample_sequence_policy: valueOrDefault('#composerSampleSequencePolicy','auto'),
    symbolic_dropout: Number(valueOrDefault('#composerSymbolicDropout',.15)),
    audio_pointer_weight: Number(valueOrDefault('#composerAudioPointerWeight',.15)),
    lens_width: Number(valueOrDefault('#composerLensWidth',64)),
    lens_chunk_size: Number(valueOrDefault('#composerLensChunkSize',32)),
    history_residual_bound: Number(valueOrDefault('#composerHistoryResidualBound',1)),
    history_memory_rms_bound: Number(valueOrDefault('#composerHistoryMemoryRmsBound',8)),
    history_dropout: Number(valueOrDefault('#composerHistoryDropout',.3)),
    ledger_dropout: Number(valueOrDefault('#composerLedgerDropout',.5)),
    placement_weight: Number(valueOrDefault('#composerPlacementWeight',.15)),
    planner_weight: Number(valueOrDefault('#composerPlannerWeight',.2)),
    recovery_interval: Number(valueOrDefault('#composerRecoveryInterval',25)),
    recovery_samples: Number(valueOrDefault('#composerRecoverySamples',2)),
    recovery_steps: Number(valueOrDefault('#composerRecoverySteps',2)),
    recovery_weight: Number(valueOrDefault('#composerRecoveryWeight',.1)),
    recovery_sampling_policy: valueOrDefault('#composerRecoverySamplingPolicy','auto'),
    recovery_pointer_temperature: Number(valueOrDefault('#composerRecoveryPointerTemperature',1)),
    recovery_action_temperature: Number(valueOrDefault('#composerRecoveryActionTemperature',.85)),
    profile_steps: Number(valueOrDefault('#composerProfileSteps',0)),
    log_interval: Number(valueOrDefault('#composerLogInterval',25)),
    max_steps: Number(valueOrDefault('#composerMaxSteps',0)),
    generation_eval_songs: Number(valueOrDefault("#composerGenerationEvalSongs", 2)),
    generation_eval_interval_epochs: Number(valueOrDefault("#composerGenerationEvalIntervalEpochs", 1)),
    validation_manifest: validationManifest,
    validation_source: validationSource,
    holdout_fraction: holdoutFraction,
    validation_max_batches: Number(valueOrDefault('#composerValidationMaxBatches',32)),
    validation_batch_size: Number(valueOrDefault('#composerValidationBatchSize',8)),
    generation_eval_seeds: valueOrDefault('#composerGenerationEvalSeeds','0,1'),
    archive_checkpoints: valueOrDefault('#composerArchiveCheckpoints','auto'),
    optimizer_implementation: valueOrDefault('#composerOptimizerImplementation','auto'),
    export_inference: checkedOrDefault('#composerExportInference',true),
    d_model: Number(valueOrDefault("#composerDModel", 384)),
    dim_feedforward: Number(valueOrDefault("#composerDimFeedforward", 1536)),
    layers: Number(valueOrDefault("#composerLayers", 6)),
    nhead: Number(valueOrDefault("#composerNhead", 8)),
    history_events: Number(valueOrDefault("#composerHistoryEvents", 128)),
    candidate_count: Number(valueOrDefault("#composerCandidateCount", 96)),
    action_slots: Number(valueOrDefault("#composerActionSlots", 8)),
    device: formDevice("#device"),
    precision: valueOrDefault("#composerPrecision", "auto"),
    seed: Number(valueOrDefault("#composerSeed", 0)),
    limit_charts: Number(valueOrDefault("#composerLimitCharts", 0)),
    prepare_workers: Number(valueOrDefault("#composerPrepareWorkers", 4)),
    audio_cache_gb: Number(valueOrDefault("#composerAudioCacheGb", 4)),
    checkpoint_interval_steps: Number(valueOrDefault("#composerCheckpointIntervalSteps", 500)),
    cache_dir: valueOrDefault("#composerCacheDir", "cache/chart_composer_v1").trim(),
  };
}

function collectValues(entries) {
  const values = {};
  entries.forEach(([key]) => {
    const target = document.getElementById(key);
    values[key] = target ? Number(target.value) : 0.5;
  });
  return values;
}

function collectCondition() {
  return {
    level: Number(valueOrDefault("#level", 13.3)),
    difficulty: Number(valueOrDefault("#acceptanceDifficulty", 5)),
    radar: collectValues(radarKeys),
    style: collectValues(styleKeys),
    ...(lastConditionRecipe ? { preset_recipe: lastConditionRecipe } : {}),
  };
}

function songFormGenerationPayload() {
  const rational = currentGenerationFamily() === "song_form_v2";
  const localChordStrength = Number(valueOrDefault("#formLocalChordStrength", 1.0));
  const rhythmSourceMs = Number(valueOrDefault("#formRhythmSnapSourceMs", 30));
  const condition = {...collectCondition(), generation_options: {
    local_chord_strength: localChordStrength, rhythm_snap_source_ms: rhythmSourceMs,
    position_complexity: Number(valueOrDefault("#formPositionComplexity", .2)),
    position_beam_size: Number(valueOrDefault("#formPositionBeamSize", 1)),
    early_eos_penalty: Number(valueOrDefault("#formEarlyEosPenalty", 0)),
    position_max_denominator: Number(valueOrDefault("#formPositionMaxDenominator", 96)),
    interval_max_denominator: Number(valueOrDefault("#formIntervalMaxDenominator", 96)),
    rational_min_spacing_ms: Number(valueOrDefault("#formRationalMinSpacingMs", 20)),
    rational_grid_snap_ms: Number(valueOrDefault("#formRationalGridSnapMs", 8)),
    rational_grid_divisions: songFormRationalGridDivisions(),
    physical_feasibility_strength: Number(valueOrDefault("#formPhysicalFeasibilityStrength", 0))}};
  return { model_family: currentGenerationFamily(), checkpoint: valueOrDefault("#checkpointPath", "").trim(), audio: valueOrDefault("#audioPath", "").trim(), output: valueOrDefault("#outputPath", "").trim() || null,
    level: Number(valueOrDefault("#level", 13.3)), condition, bpm: valueOrDefault("#bpm", "").trim() === "" ? null : Number(valueOrDefault("#bpm", 120)), first: Number(valueOrDefault("#first", 0)), tempo_map: parseTempoMap(),
    meter_map: valueOrDefault("#formGenerationMeterMap", "").trim() ? parseMeterMap("#formGenerationMeterMap") : null,
    device: formDevice("#generationDevice"),
    max_region_seconds: Number(valueOrDefault("#formGenerationMaxRegionSeconds", 14)), intensity_curve_enabled: checkedOrDefault("#intensityCurveEnabled", false), intensity_curve: intensityCurvePayload(),
    intensity_curve_strength: Number(valueOrDefault("#intensityCurveStrength", .65)), intensity_curve_span: Number(valueOrDefault("#intensityFollowSpan", .5)), intensity_curve_mode: valueOrDefault("#intensityCurveMode", "follow"),
    memory_strength: Number(valueOrDefault("#formMemoryStrength", .6)), onset_threshold: Number(valueOrDefault("#formOnsetThreshold", .5)),
    onset_merge_ms: Number(valueOrDefault("#formOnsetMergeMs", 40)),
    rhythm_snap_enabled: !rational && checkedOrDefault("#formRhythmSnapEnabled", false),
    rhythm_snap_ms: Number(valueOrDefault("#formRhythmSnapMs", 15)),
    rhythm_snap_source_ms: rhythmSourceMs,
    rhythm_snap_divisions: songFormRhythmSnapDivisions(checkedOrDefault("#formRhythmSnapEnabled", false)),
    rhythm_snap_mode: valueOrDefault("#formRhythmSnapMode", "sequence"),
    planner_type_strength: rational ? 0 : Number(valueOrDefault("#formPlannerTypeStrength", 1.0)),
    position_temperature: Number(valueOrDefault("#formPositionTemperature", .2)),
    position_characters: Number(valueOrDefault("#formPositionCharacters", 24)),
    position_max_denominator: Number(valueOrDefault("#formPositionMaxDenominator", 96)),
    interval_max_denominator: Number(valueOrDefault("#formIntervalMaxDenominator", 96)),
    position_complexity: Number(valueOrDefault("#formPositionComplexity", .2)),
    position_beam_size: Number(valueOrDefault("#formPositionBeamSize", 1)),
    early_eos_penalty: Number(valueOrDefault("#formEarlyEosPenalty", 0)),
     max_events_per_span: Number(valueOrDefault("#formMaxEventsPerSpan", 0)),
     rational_min_spacing_ms: Number(valueOrDefault("#formRationalMinSpacingMs", 20)),
    rational_grid_snap_ms: Number(valueOrDefault("#formRationalGridSnapMs", 8)),
    rational_grid_divisions: songFormRationalGridDivisions(),
    physical_feasibility_strength: Number(valueOrDefault("#formPhysicalFeasibilityStrength", 0)),
    style_balance_strength: Number(valueOrDefault("#formStyleBalanceStrength", 1.0)),
    local_chord_strength: localChordStrength,
    temperature: Number(valueOrDefault("#temperature", .65)), seed: valueOrDefault("#seed", "").trim() === "" ? null : Number(valueOrDefault("#seed", 0)), repair: checkedOrDefault("#formRepair", false), max_checks: Number(valueOrDefault("#formMaxChecks", 24)), max_edits: Number(valueOrDefault("#formMaxEdits", 4)), deadline_seconds: Number(valueOrDefault("#formDeadlineSeconds", 10)) };
}
function uniqueSongFormOutputPath(requested) {
  const path = String(requested || "").trim();
  if (!path) return path;
  const separatorIndex = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  const separator = path.includes("\\") ? "\\" : "/";
  const directory = separatorIndex >= 0 ? path.slice(0, separatorIndex + 1) : "";
  const filename = path.slice(separatorIndex + 1);
  const nonce = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
  return `${directory}.task-${nonce}${separator}${filename}`;
}
function chartComposerGenerationPayload() {
  const payload = {
    model_family: "chart_composer_v1",
    checkpoint: valueOrDefault("#checkpointPath", "").trim(),
    audio: valueOrDefault("#audioPath", "").trim(),
    output: valueOrDefault("#outputPath", "").trim() || null,
    bpm: valueOrDefault("#bpm", "").trim() === "" ? null : Number(valueOrDefault("#bpm", 120)),
    first: Number(valueOrDefault("#first", 0)),
    level: Number(valueOrDefault("#level", 13.3)),
    condition: collectCondition(),
    tempo_map: parseTempoMap(),
    meter_map: valueOrDefault("#formGenerationMeterMap", "").trim() ? parseMeterMap("#formGenerationMeterMap") : null,
    temperature: Number(valueOrDefault("#temperature", .65)),
    pointer_temperature: (valueOrDefault("#composerPointerTemperature", "").trim()===""?null:Number(valueOrDefault("#composerPointerTemperature", .65))),
    action_temperature: (valueOrDefault("#composerActionTemperature", "").trim()===""?null:Number(valueOrDefault("#composerActionTemperature", .65))),
    action_retry_policy: valueOrDefault('#composerActionRetryPolicy','auto'),
    pointer_grid_policy: valueOrDefault('#composerPointerGridPolicy','grid_family_coherent_v3'),
    survival_calibration: valueOrDefault('#composerSurvivalCalibration','auto'),
    density_feedback_policy: valueOrDefault('#composerDensityFeedbackOverride','auto'),
    repetition_guard: valueOrDefault('#composerRepetitionGuard','official_tail_v2'),
    action_history_strength: Number(valueOrDefault('#composerActionHistoryStrength',1)),
    action_type_temperature: Number(valueOrDefault('#composerActionTypeTemperature',1)),
    action_type_min_p: Number(valueOrDefault('#composerActionTypeMinP',0)),
    curve_strength: Number(valueOrDefault("#composerCurveStrength", 1)),
    pointer_history_strength: Number(valueOrDefault('#composerPointerHistoryStrength',1)),
    min_same_head_ms: Number(valueOrDefault("#composerMinSameHeadMs", 50)),
    intercept_tap_on_slide: checkedOrDefault("#composerInterceptTapOnSlide", false),
    contact_policy: valueOrDefault('#composerGenerationContactPolicy','auto'),
    touch_layout_policy: valueOrDefault('#composerTouchLayoutPolicy','open'),
    motif_strength: Number(valueOrDefault('#composerMotifStrength',1)),
    quality_context_strength: Number(valueOrDefault('#composerQualityContextStrength',1)),
    quality_plan_strength: Number(valueOrDefault('#composerQualityPlanStrength',1)),
    quality_lookahead_strength: Number(valueOrDefault('#composerQualityLookaheadStrength',0)),
    touch_history_guard: checkedOrDefault('#composerTouchHistoryGuard',true),
    touch_history_bound: Number(valueOrDefault('#composerTouchHistoryBound',1)),
    seed: valueOrDefault("#seed", "").trim() === "" ? null : Number(valueOrDefault("#seed", 0)),
    max_events: Number(valueOrDefault("#composerMaxEvents", 0)),
    device: formDevice("#generationDevice"),
    condition_file: valueOrDefault("#composerConditionFile", "").trim() || null,
    intensity_curve_enabled: checkedOrDefault("#intensityCurveEnabled", false),
    intensity_curve: intensityCurvePayload(),
    intensity_curve_strength: Number(valueOrDefault("#intensityCurveStrength", .65)),
    intensity_curve_span: valueOrDefault("#composerIntensityCurveSpan", "full"),
    intensity_curve_mode: valueOrDefault("#intensityCurveMode", "follow"),
    intensity_curve_chord_multiplier: Number(valueOrDefault("#composerChordMultiplier", 1)),
  };
  for(const key of ['pointer_temperature','action_temperature'])if(payload[key]===null)delete payload[key];
  if(isEventSetGeneration()) {
    for(const [key,id,fallback] of [['phrase_candidates','composerPhraseCandidates',1],['audio_tracking_strength','composerAudioTrackingStrength',0],['softcomfort_strength','composerSoftcomfortStrength',1],['phrase_write_beats','composerPhraseWriteBeats',8],['phrase_commit_beats','composerPhraseCommitBeats',4],['phrase_step_budget','composerPhraseStepBudget',256],['phrase_repair_budget','composerPhraseRepairBudget',8],['phrase_rollback_budget','composerPhraseRollbackBudget',4],['phrase_time_budget_sec','composerPhraseTimeBudgetSec',30]])payload[key]=Number(valueOrDefault('#'+id,fallback));
    const songCaps=generationInspection?.data?.generation_capabilities||{};
    if(songCaps.song_profile===true){
      payload.song_style_source=valueOrDefault('#composerSongStyleSource','auto');
      payload.song_style_seed=valueOrDefault('#composerSongStyleSeed','').trim()===''?null:Number(valueOrDefault('#composerSongStyleSeed',0));
      try{payload.song_style_locks=JSON.parse(valueOrDefault('#composerSongStyleLocks','{}'));}catch{throw new Error('整曲风格手动锁定必须是有效的 JSON 对象');}
      if(!payload.song_style_locks || Array.isArray(payload.song_style_locks) || typeof payload.song_style_locks!=='object')throw new Error('整曲风格手动锁定必须是 JSON 对象');
    }
    if(songCaps.song_response===true)payload.song_response_mode=Number(valueOrDefault('#composerSongResponseMode',-1));
    payload.phrase_mode_temperature=Number(valueOrDefault('#composerPhraseModeTemperature',1));
    payload.search_policy=valueOrDefault('#composerSearchPolicy','phrase_v1');
    payload.physical_policy=valueOrDefault('#composerPhysicalPolicy','runtime_hard_v2');
    if(generationInspection?.data?.generation_capabilities?.optional_audio_tracking!==true)payload.audio_tracking_strength=0;
    const allowed=new Set(['model_family','checkpoint','audio','output','bpm','first','level','condition','tempo_map','meter_map','temperature','pointer_temperature','action_temperature','seed','max_events','device','min_same_head_ms','touch_layout_policy','condition_file','intensity_curve_enabled','intensity_curve','intensity_curve_file','intensity_curve_strength','intensity_curve_mode']);
    for(const key of ['phrase_mode_temperature','phrase_candidates','audio_tracking_strength','softcomfort_strength','phrase_write_beats','phrase_commit_beats','phrase_step_budget','phrase_repair_budget','phrase_rollback_budget','phrase_time_budget_sec','search_policy','physical_policy','song_style_source','song_style_seed','song_style_locks','song_response_mode'])allowed.add(key);
    return Object.fromEntries(Object.entries(payload).filter(([key])=>allowed.has(key)));
  }
  return payload;
}
function isEventSetGeneration() {
  const data=generationInspection?.data;
  return generationInspection?.path===valueOrDefault('#checkpointPath','').trim() && data?.valid===true
    && (data?.model_config?.architecture_profile==='event_set_stream_v1'||data?.schema==='maichart-composer-event-set-checkpoint-v1');
}
function initializeEventSetParameterGroups() {
  const host=element('#chartComposerTrainingControls');if(!host)return;
  let groups=element('#eventSetParameterGroups');
  if(!groups){
    groups=document.createElement('div');groups.id='eventSetParameterGroups';host.append(groups);
    const titles={basic:'基础运行',learning:'出音与配置学习',planning:'短段规划与历史呼应',recovery:'连续恢复与错误前文',validation:'验证与检查点',model:'模型骨架',performance:'性能与诊断'};
    for(const [key,title] of Object.entries(titles)){const details=document.createElement('details');details.className='settings-section';details.open=['basic','learning','planning','validation'].includes(key);details.innerHTML=`<summary>${title}</summary><div class="settings-content form-grid" id="eventSetGroup_${key}"></div>`;groups.append(details);}
    const basicAdvanced=document.createElement('details');basicAdvanced.id='eventSetBasicAdvanced';basicAdvanced.className='settings-section';basicAdvanced.style.gridColumn='1 / -1';basicAdvanced.innerHTML='<summary>基础运行高级设置</summary><div class="settings-content form-grid" id="eventSetBasicAdvancedGrid"></div>';element('#eventSetGroup_basic').append(basicAdvanced);
    const names={draft_policy:'短段草稿',plan_kind_policy:'出音与休止判断',transition_policy:'重复段转折学习',template_policy:'类型与模板保持',candidate_state_policy:'候选落点持续状态',decoder_policy:'解码器归一化',window_balance_policy:'短段样本权重',timing_output_policy:'拍位输出方式',timing_grid_audio_policy:'候选拍位与音乐匹配',timing_group_policy:'节奏分组',literal_output_policy:'配置字面输出',event_family_policy:'事件类型学习',recovery_target_policy:'恢复目标',prefix_augmentation_input_policy:'错误历史构造',history_window_policy:'历史窗口对齐',continuation_policy:'持续配置状态',sensor_load_policy:'Touch与按钮负荷',gap_continuation_policy:'间隔持续状态',motion_continuation_policy:'动作衔接状态',lane_load_policy:'键位负荷状态',recovery_policy:'自主生成恢复',physical_policy:'训练物理判定'};
    Object.assign(names,{literal_set_policy:'同拍集合续写',song_choice_objective_policy:'整曲写法选择学习'});
    const advancedPolicies=document.createElement('details');advancedPolicies.className='settings-section';advancedPolicies.style.gridColumn='1 / -1';advancedPolicies.innerHTML='<summary>高级学习策略</summary><div class="settings-content form-grid" id="eventSetAdvancedPolicies"></div>';element('#eventSetGroup_learning').append(advancedPolicies);
    for(const [key,choices] of Object.entries(EVENT_SET_LEARNING_POLICIES)){
      if(['phrase_context_policy','duration_audio_policy'].includes(key))continue;
      const field=document.createElement('label');field.className='field';field.innerHTML=`${names[key]||key}<select id="eventSetPolicy_${key}"><option value="auto">${({event_family_policy:"自动（新训练：音乐与历史共同决定类型）",recovery_target_policy:"自动（新训练：时间恢复与可信配置分开）",transition_policy:"自动（新训练：关闭）",gap_continuation_policy:"自动（新训练：关闭）"})[key]||"随预设（自动）"}</option>${choices.map(value=>`<option value="${value}" title="${value}">${key==='gap_continuation_policy'&&value==='off'?'关闭（推荐）':EVENT_SET_POLICY_LABELS[value]||value}</option>`).join('')}</select><small>${key==='event_family_policy'?'音乐、历史与持续状态共同形成续写状态，供类型选择使用。':key==='recovery_target_policy'?'时间恢复可继续学习；只有与原谱存在可信对应关系的配置才参与配置监督。':key==='gap_continuation_policy'?'新训练使用绝对拍位时默认关闭旧间隔重复概率重分配；历史策略仍可选择，续训继承。':'单项选择优先于预设；续训继承。'}</small>`;
      element(key==='draft_policy'?'#eventSetGroup_planning':key==='prefix_augmentation_input_policy'?'#eventSetGroup_recovery':['event_family_policy','recovery_target_policy','plan_kind_policy'].includes(key)?'#eventSetGroup_learning':'#eventSetAdvancedPolicies').append(field);
      if(['draft_policy','plan_kind_policy'].includes(key))field.querySelector?.('select')?.addEventListener('change',updateEventSetControls);
    }
    const panel=[['phrase_mode_temperature','短句写法温度（0贪心）',1],['pointer_temperature','选时温度（留空继承验证总温度）',''],['action_temperature','配置温度（留空继承验证总温度）',''],['min_same_head_ms','同键最短间隔（毫秒）',50],['touch_layout_policy','Touch布局','open',['open','legacy_two_heads']],['max_events','诊断事件上限（0自动）',0],['search_policy','生成流程','phrase_v1',['phrase_v1','legacy']],['physical_policy','物理判定','runtime_hard_v2',['runtime_hard_v2','legacy']],['phrase_candidates','候选数',1,[1,2,4]],['phrase_write_beats','试写拍数',8],['phrase_commit_beats','提交拍数',4],['phrase_step_budget','每候选步数',256],['phrase_repair_budget','历史编辑预算',8],['phrase_rollback_budget','撤回预算',4],['phrase_time_budget_sec','每短句秒数',30],['softcomfort_strength','手感评分强度',1],['audio_tracking_strength','采音连续性',0]];
    const advanced=document.createElement('details');advanced.id='eventSetValidationGeneration';advanced.className='settings-section';advanced.style.gridColumn='1 / -1';advanced.innerHTML='<summary>验证自由生成设置</summary><p>参考谱条件下的固定歌曲抽检；下列采样与搜索设置独立于手动生成页，默认一致；采用各验证谱派生条件与原节拍，不使用外部强度曲线。</p><div class="form-grid" id="eventSetValidationGenerationGrid"></div>';element('#eventSetGroup_validation').append(advanced);
    for(const [key,name,value,choices] of panel){const field=document.createElement('label');field.className='field';field.innerHTML=name+(choices?`<select id="eventSetEval_${key}" data-generation-eval-key="${key}">${choices.map(v=>`<option value="${v}" ${v===value?'selected':''}>${({touch_layout_policy:{open:'开放Touch布局',legacy_two_heads:'旧双头限制'},search_policy:{phrase_v1:'短句事务生成',legacy:'旧点生成流程（对照）'},physical_policy:{runtime_hard_v2:'运行证据硬冲突（v2）',legacy:'旧物理判定（对照）'}})[key]?.[v]??v}</option>`).join('')}</select>`:`<input id="eventSetEval_${key}" data-generation-eval-key="${key}" type="number" min="0" step="any" value="${value}">`);element('#eventSetValidationGenerationGrid').append(field);}
    const preset=element('#composerLearningPolicy');preset?.addEventListener('change',()=>{for(const [id,value] of [['composerTransitionExitWeight',0],['composerPrefixAugmentationRate',.15],['composerEventFamilyAudioWeight',.25]]){const node=element('#'+id);if(node)node.value=String(preset.value==='legacy'?0:value);}});
  }
  const stream=valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1';groups.hidden=!stream;
  const sections=['composerTrainingCommon','composerTrainingEvaluation','composerTrainingAdvanced'];
  for(const id of sections){const node=element('#'+id);if(node)node.hidden=stream;}
  const ids={basic:['composerArchitectureProfile','composerPrecision','composerResetLearningRate','composerLrSchedule','composerWarmupSteps','composerMinLrRatio'],learning:['composerLearningPolicy','composerPhraseContextPolicy','composerDurationAudioPolicy','composerPhraseLossWeight','composerEventFamilyAudioWeight','composerPlanWeight'],recovery:['composerPrefixAugmentationRate','composerRecoveryInterval','composerRecoverySamples','composerRecoverySteps','composerConfigurationRecoverySteps','composerRecoveryBridgeSteps','composerRecoveryWeight','composerSymbolicPretrainEpochs'],validation:['composerValidationSource','composerValidationManifest','composerHoldoutFraction','composerValidationMaxBatches','composerValidationBatchSize','composerGenerationEvalSongs','composerGenerationEvalIntervalEpochs','composerGenerationEvalSeeds','composerGenerationEvalTemperature','composerCheckpointIntervalSteps'],model:['composerDModel','composerLayers','composerNhead','composerDimFeedforward','composerHistoryEvents','composerRolloutSteps','composerWindowSeconds','composerAudioGridSeconds','composerPlanBlankWindowSeconds','composerDropout'],performance:['composerBatchOrderPolicy','composerTimingCompute','composerDetailCompute','composerDataWorkers','composerPrepareWorkers','composerSampleWorkers','composerCpuThreads','composerPrefetchBatches','composerAudioCacheGb','composerSeed','composerMaxSteps','composerLimitCharts','composerLogInterval','composerCacheDir']};
  for(const [group,list] of Object.entries(ids))for(const id of list){const node=element('#'+id),field=node?.closest('label.field');if(!field)continue;if(stream){if(!field._eventSetHome){const anchor=document.createComment('event-set-home');field.before(anchor);field._eventSetHome=anchor;}element('#eventSetGroup_'+group).append(field);}else if(field._eventSetHome){field._eventSetHome.after(field);}}
  for(const id of ['composerDraftAdvanced','composerPhraseAdvanced','composerSongStyleAdvanced','composerDetailAdvanced']){const node=element('#'+id);if(!node)continue;if(stream){if(!node._eventSetHome){const anchor=document.createComment('event-set-home');node.before(anchor);node._eventSetHome=anchor;}element(['composerDraftAdvanced','composerPhraseAdvanced'].includes(id)?'#eventSetGroup_planning':'#eventSetGroup_learning').append(node);}else if(node._eventSetHome)node._eventSetHome.after(node);}
  if(stream){const resetField=element('#composerResetLearningRate')?.closest('label.field');if(resetField)element('#eventSetBasicAdvancedGrid').append(resetField);const rebatchField=element('#composerRebatchResume')?.closest('label.field');if(rebatchField){if(!rebatchField._eventSetHome){const anchor=document.createComment('event-set-home');rebatchField.before(anchor);rebatchField._eventSetHome=anchor;}element('#eventSetBasicAdvancedGrid').append(rebatchField);}element('#eventSetGroup_validation').append(element('#eventSetValidationGeneration'));const field=element('#composerTransitionExitWeight')?.closest('label.field');if(field){if(!field._eventSetHome){const anchor=document.createComment('event-set-home');field.before(anchor);field._eventSetHome=anchor;}element('#eventSetAdvancedPolicies').append(field);}}else{const rebatchField=element('#composerRebatchResume')?.closest('label.field');if(rebatchField?._eventSetHome)rebatchField._eventSetHome.after(rebatchField);const field=element('#composerTransitionExitWeight')?.closest('label.field');if(field?._eventSetHome)field._eventSetHome.after(field);}
}

function updateEventSetControls() {
  const stream=valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1';
  initializeEventSetParameterGroups();
  if(stream){
    const mode=currentTrainingMode();
    for(const id of ['composerDraftObjectivePolicy','composerConfigurationRecoverySteps','composerRecoveryBridgeSteps','composerDurationObjectivePolicy','composerDurationCalibrationWeight']){const node=element('#'+id);if(node)node.disabled=mode==='resume';}
    for(const id of ['composerCurvePreviewPolicy','composerCurvePreviewWeight']){const node=element('#'+id);if(node)node.disabled=mode==='resume';}
    for(const id of ['composerTrainingWindowPolicy','composerPhraseExecutionPolicy','composerDetailRepresentationPolicy','composerDurationFeaturePolicy','composerSongProfilePolicy','composerSongResponsePolicy','composerSongProfileComponents','composerSongProfileWeight','composerSongResponseModes','composerSongResponseGroupSize','composerDeploymentConditionFraction','composerBatchOrderPolicy','composerDraftLossWeight','composerDraftResponsibilityTemperature']){const node=element('#'+id);if(node)node.disabled=mode==='resume';}
    for(const id of ['composerMaxPhraseEvents','composerPhraseSeamEvents','composerPhraseModes','composerPhraseRelationWeight']){const node=element('#'+id);if(node)node.disabled=mode!=='fresh';}
    const reset=element('#composerResetLearningRate');
    if(reset){const field=reset.closest('label.field');if(field)field.hidden=mode!=='resume';if(mode!=='resume')reset.checked=false;}
    for(const id of ['learningRate','composerLrSchedule','composerWarmupSteps','composerMinLrRatio']){const node=element('#'+id);if(node)node.disabled=mode==='resume'&&!reset?.checked;}
    for(const id of ['composerArchitectureProfile','composerDModel','composerLayers','composerNhead','composerDimFeedforward','composerHistoryEvents','composerRolloutSteps','composerAudioGridSeconds','composerPlanBlankWindowSeconds','composerDropout']){const node=element('#'+id);if(node){node.disabled=mode!=='fresh';node.title=mode==='fresh'?'':'继承检查点；改变此项需要从头训练';}}
    for(const id of ['composerSeed','composerMaxSteps','composerLimitCharts','composerRecoveryInterval','composerRecoverySamples','composerRecoverySteps','composerRecoveryWeight','composerPlanWeight','composerWindowSeconds','composerSymbolicPretrainEpochs','composerGenerationEvalSeeds','composerValidationSource','composerValidationManifest','composerHoldoutFraction','composerEventFamilyAudioWeight']){const node=element('#'+id);if(node)node.disabled=mode==='resume';}
    for(const node of document.querySelectorAll('[id^="eventSetPolicy_"],[data-generation-eval-key]'))node.disabled=mode==='resume';
  }
  const fp16=element('#composerPrecision option[value=fp16]');if(fp16){fp16.hidden=stream;fp16.disabled=stream;if(stream&&element('#composerPrecision').value==='fp16')element('#composerPrecision').value='bf16';}
  const allowed=new Set(['composerTrainingWindowPolicy','composerMaxPhraseEvents','composerPhraseSeamEvents','composerPhraseExecutionPolicy','composerPhraseModes','composerPhraseRelationWeight','composerDetailRepresentationPolicy','composerDurationFeaturePolicy','composerValidationSource','composerHoldoutFraction','composerPhraseContextPolicy','composerDurationAudioPolicy','composerPhraseLossWeight','composerLearningPolicy','composerTransitionExitWeight','composerPrefixAugmentationRate','composerGenerationEvalTemperature','composerPlanBlankWindowSeconds','composerValidationMaxBatches','composerValidationBatchSize','composerGenerationEvalIntervalEpochs','composerWarmupSteps','composerPrefetchBatches','composerSampleWorkers','composerDataWorkers','composerAudioCacheGb','composerMinLrRatio','composerLrSchedule','composerPlanWeight','composerAudioGridSeconds','composerArchitectureProfile','composerPrecision','composerDModel','composerLayers','composerNhead','composerDimFeedforward','composerHistoryEvents','composerDropout','composerRolloutSteps','composerWindowSeconds','composerAudioGridSeconds','composerPlanWeight','composerPlanBlankWindowSeconds','composerSymbolicPretrainEpochs','composerRecoveryInterval','composerRecoverySamples','composerRecoverySteps','composerRecoveryWeight','composerPrepareWorkers','composerCpuThreads','composerSeed','composerMaxSteps','composerLimitCharts','composerLogInterval','composerCheckpointIntervalSteps','composerCacheDir','composerValidationManifest','composerGenerationEvalSongs','composerGenerationEvalSeeds']);
  for(const id of ['composerSongProfilePolicy','composerSongResponsePolicy','composerSongProfileComponents','composerSongProfileWeight','composerSongResponseModes','composerSongResponseGroupSize'])allowed.add(id);
  for(const id of ['composerDraftLossWeight','composerDraftResponsibilityTemperature','composerCurvePreviewPolicy','composerCurvePreviewWeight'])allowed.add(id);
  for(const id of ['composerDraftObjectivePolicy','composerConfigurationRecoverySteps','composerRecoveryBridgeSteps','composerDurationObjectivePolicy','composerDurationCalibrationWeight'])allowed.add(id);
  allowed.add('composerDeploymentConditionFraction');allowed.add('composerEventFamilyAudioWeight');allowed.add('composerRebatchResume');allowed.add('composerResetLearningRate');allowed.add('composerTimingCompute');allowed.add('composerDetailCompute');allowed.add('composerBatchOrderPolicy');
  for(const id of ['composerPhraseContextPolicy','composerDurationAudioPolicy','composerPhraseLossWeight'])allowed.add(id);
  document.querySelectorAll('#chartComposerTrainingControls input,#chartComposerTrainingControls select,#chartComposerTrainingControls textarea').forEach(control=>{
    const legacy=!allowed.has(control.id)&&!control.id.startsWith('eventSetPolicy_')&&!control.hasAttribute('data-generation-eval-key'),field=control.closest('label.field');
    if(legacy&&field){field.hidden=stream;if(stream){if(control.dataset.eventSetDisabled===undefined)control.dataset.eventSetDisabled=String(control.disabled);control.disabled=true;}else if(control.dataset.eventSetDisabled!==undefined){control.disabled=control.dataset.eventSetDisabled==='true';delete control.dataset.eventSetDisabled;}}
  });
  document.querySelectorAll('[data-event-set-only]').forEach(node=>{node.hidden=!stream || (node.querySelector('#composerResetLearningRate') && currentTrainingMode()!=='resume');});
  if(stream){
    const preset=valueOrDefault('#composerLearningPolicy','auto'),freshStage=currentTrainingMode()!=='resume'&&!['legacy','checkpoint'].includes(preset);
    const draft=valueOrDefault('#eventSetPolicy_draft_policy','auto'),time=valueOrDefault('#eventSetPolicy_plan_kind_policy','auto');
    const useDraft=draft==='beat_ordered_draft_v1'||(freshStage&&draft==='auto'),useTime=['unified_first_arrival_v5','unified_first_arrival_v6'].includes(time)||(freshStage&&time==='auto');
    for(const [id,replaced] of [['composerPlanWeight',useTime],['composerPhraseLossWeight',useDraft],['composerPhraseRelationWeight',useDraft]]){const node=element('#'+id),field=node?.closest('label.field');if(field)field.hidden=replaced;}
    for(const id of ['composerDraftLossWeight','composerDraftResponsibilityTemperature','composerDraftObjectivePolicy']){const node=element('#'+id),field=node?.closest('label.field');if(field)field.hidden=!useDraft;}
    const savedRecovery=element('#composerLearningPolicy')?.dataset.savedPolicies;
    const recovery=currentTrainingMode()==='resume'&&savedRecovery
      ? JSON.parse(savedRecovery).recovery_policy : valueOrDefault('#eventSetPolicy_recovery_policy','auto');
    const configurationRecovery=['certified_phrase_v5','first_step_phrase_v6'].includes(recovery)||(freshStage&&recovery==='auto');
    for(const id of ['composerConfigurationRecoverySteps','composerRecoveryBridgeSteps']){const field=element('#'+id)?.closest('label.field');if(field)field.hidden=!configurationRecovery;}
    const durationObjective=valueOrDefault('#composerDurationObjectivePolicy','auto');
    const durationCalibration=durationObjective==='hold_crps_v2'||(freshStage&&durationObjective==='auto');
    const durationWeightField=element('#composerDurationCalibrationWeight')?.closest('label.field');if(durationWeightField)durationWeightField.hidden=!durationCalibration;
  }
  const responsePolicy=valueOrDefault('#composerSongResponsePolicy','auto');
  const unifiedResponse=responsePolicy==='unified_profile_response_v2' || (currentTrainingMode()!=='resume' && responsePolicy==='auto' && !['legacy','checkpoint'].includes(valueOrDefault('#composerLearningPolicy','auto')));
  const profileComponents=element('#composerSongProfileComponents');if(profileComponents)profileComponents.max=unifiedResponse?'8':'16';
  const responseModes=element('#composerSongResponseModes');if(responseModes){const field=responseModes.closest('label.field');if(field)field.hidden=unifiedResponse;responseModes.disabled=currentTrainingMode()==='resume'||unifiedResponse;}
  const validationMode=currentTrainingMode()==='resume'?'resume':valueOrDefault('#composerValidationSource','auto');
  const validationSourceNode=element('#composerValidationSource'),validationPathNode=element('#composerValidationManifest'),validationFractionNode=element('#composerHoldoutFraction');
  if(validationSourceNode)validationSourceNode.disabled=validationMode==='resume';
  if(validationPathNode)validationPathNode.disabled=validationMode!=='manual';
  if(validationFractionNode)validationFractionNode.disabled=validationMode!=='auto';
  const sampleWorker=element('#composerSampleWorkers');if(sampleWorker){if(stream){if(sampleWorker.dataset.eventSetWorkerDisabled===undefined)sampleWorker.dataset.eventSetWorkerDisabled=String(sampleWorker.disabled);sampleWorker.value='1';sampleWorker.disabled=true;}else if(sampleWorker.dataset.eventSetWorkerDisabled!==undefined){sampleWorker.disabled=sampleWorker.dataset.eventSetWorkerDisabled==='true';delete sampleWorker.dataset.eventSetWorkerDisabled;}}
  const workerHint=element('#composerEventSetWorkerHint');if(workerHint)workerHint.hidden=!stream;
  const hint=element('#composerEventSetHint');if(hint)hint.hidden=!stream;
  document.querySelectorAll('[data-event-set-generation-only]').forEach(node=>{node.hidden=!isEventSetGeneration();});
  const songStyle=element('#composerSongStyleGeneration');
  if(songStyle){
    const caps=generationInspection?.data?.generation_capabilities||{},profile=caps.song_profile===true,response=caps.song_response===true;
    songStyle.hidden=!isEventSetGeneration()||(!profile&&!response);
    for(const id of ['composerSongStyleSource','composerSongStyleSeed','composerSongStyleLocks']){const node=element('#'+id);if(node){node.disabled=!profile;const field=node.closest('label.field');if(field)field.hidden=!profile;}}
    const mode=element('#composerSongResponseMode');if(mode){mode.disabled=!response;const field=mode.closest('label.field');if(field)field.hidden=!response;}
    const hint=element('#composerSongStyleHint');if(hint)hint.textContent=profile?'自动模式先选择整体写法，再使用该写法内的连续统计值，避免逐字段随机抽到极端密度；明确锁定优先。其他风格滑块的默认值不会锁定；沿用检查点默认和手动设置使用当前生成条件滑块。':'此检查点未训练自动风格补全，使用当前生成条件；音乐响应模式可单独设置。';
  }
  const phraseModeField=element('#composerPhraseModeTemperature')?.closest('label.field');
  if(phraseModeField)phraseModeField.hidden=!isEventSetGeneration() || generationInspection?.data?.generation_capabilities?.persistent_phrase_modes!==true;
  const phraseHint=element('#composerPhraseCapabilityHint');
  if(phraseHint&&isEventSetGeneration()){
    const caps=generationInspection?.data?.generation_capabilities||{};
    phraseHint.textContent=`短段组织：${caps.ordered_draft?'可调整的有序草稿（按拍位读取，实际占用优先）':'此检查点没有有序草稿，按原策略执行'}；时间判断：${caps.unified_first_arrival?'统一落点与无事件尾概率':'检查点原有出音策略'}；短句记忆：${caps.aligned_phrase_context?'拍位对齐有序短句':caps.ordered_phrase_memory?'有序配置短句':caps.learned_phrase_context?'旧版短句统计':'未启用'}；持续区间音频训练：${caps.duration_interval_audio?'已启用':'未启用'}。能力是否学好仍以生成验收为准；采音连续性默认关闭。`;
  }
  const trackingControl=element('#composerAudioTrackingStrength');
  if(trackingControl){
    const available=isEventSetGeneration()&&generationInspection?.data?.generation_capabilities?.optional_audio_tracking===true;
    trackingControl.disabled=!available;
    if(!available)trackingControl.value='0';
    trackingControl.title=available?'默认0关闭':'当前检查点未训练短句上下文，采音连续性不可用，提交值固定为0';
  }
  const oldGeneration=['composerActionRetryPolicy','composerPointerGridPolicy','composerSurvivalCalibration','composerDensityFeedbackOverride','composerRepetitionGuard','composerActionHistoryStrength','composerActionTypeTemperature','composerActionTypeMinP','composerCurveStrength','composerPointerHistoryStrength','composerInterceptTapOnSlide','composerGenerationContactPolicy','composerMotifStrength','composerQualityContextStrength','composerQualityPlanStrength','composerQualityLookaheadStrength','composerTouchHistoryGuard','composerTouchHistoryBound','composerIntensityCurveSpan','composerChordMultiplier'];
  for(const id of oldGeneration){const node=element('#'+id);if(node){const field=node.closest('label.field');if(field)field.hidden=isEventSetGeneration();if(isEventSetGeneration()){if(node.dataset.eventSetDisabled===undefined)node.dataset.eventSetDisabled=String(node.disabled);node.disabled=true;}else if(node.dataset.eventSetDisabled!==undefined){node.disabled=node.dataset.eventSetDisabled==='true';delete node.dataset.eventSetDisabled;}}}
}
function applyEventSetFreshDefaults() {
  if(currentTrainingMode()!=='fresh')return;
  const directory=element('#checkpointDir');
  if(directory){const current=directory.value.trim().replace(/\\/g,'/').replace(/\/+$/,'');if(!current||['checkpoints/song_form_v2','checkpoints/chart_stream_v1','checkpoints/chart_composer_v1','checkpoints/prism_composer_v16_signal'].includes(current))directory.value='checkpoints/event_set_stream_v1';}
  for(const node of document.querySelectorAll('[id^="eventSetPolicy_"]'))node.value='auto';
  const batchOrder=element('#composerBatchOrderPolicy');if(batchOrder)batchOrder.value='auto';
  const familyAudio=element('#composerEventFamilyAudioWeight');if(familyAudio)familyAudio.value='0.25';
  for(const [id,value] of [['composerCurvePreviewPolicy','auto'],['composerCurvePreviewWeight',.05]]){const node=element('#'+id);if(node)node.value=String(value);}
  for(const [id,value] of [['composerDraftObjectivePolicy','auto'],['composerConfigurationRecoverySteps',8],['composerRecoveryBridgeSteps',8]]){const node=element('#'+id);if(node)node.value=String(value);}
  for(const [id,value] of [['composerDurationObjectivePolicy','auto'],['composerDurationCalibrationWeight',.1]]){const node=element('#'+id);if(node)node.value=String(value);}
  const validationSource=element('#composerValidationSource');if(validationSource)validationSource.value='auto';
  const validationField=element('#composerValidationManifest');if(validationField?.dataset.validationInherited==='1'){validationField.value=validationField.dataset.beforeResumeValidation??'';delete validationField.dataset.validationInherited;delete validationField.dataset.beforeResumeValidation;}
  for(const [id,value] of [['composerTrainingWindowPolicy','auto'],['composerMaxPhraseEvents',64],['composerPhraseSeamEvents',2],['composerPhraseExecutionPolicy','auto'],['composerPhraseModes',4],['composerPhraseRelationWeight',0],['composerDetailRepresentationPolicy','auto'],['composerDurationFeaturePolicy','auto'],['composerPhraseContextPolicy','auto'],['composerDurationAudioPolicy','auto'],['composerPhraseLossWeight',0],['composerDraftLossWeight',.1],['composerDraftResponsibilityTemperature',1]]){const node=element('#'+id);if(node)node.value=String(value);}
  const deploymentFraction=element('#composerDeploymentConditionFraction');if(deploymentFraction)deploymentFraction.value='.25';
  for(const [id,value] of Object.entries({composerLearningPolicy:'auto',composerTransitionExitWeight:0,composerPrefixAugmentationRate:.15,composerGenerationEvalTemperature:.65,composerGenerationEvalSongs:0,epochs:50,batchSize:16,gradAccumulation:4,learningRate:.0005,composerDModel:384,composerLayers:8,composerNhead:8,composerDimFeedforward:1536,composerHistoryEvents:128,composerRolloutSteps:32,composerWindowSeconds:8,composerAudioGridSeconds:.01,composerPlanWeight:0,composerPlanBlankWindowSeconds:.5,composerValidationMaxBatches:32,composerValidationBatchSize:4,composerGenerationEvalIntervalEpochs:1,composerWarmupSteps:0,composerDetailCompute:'serial',composerDataWorkers:0,composerPrefetchBatches:2,composerSampleWorkers:1,composerAudioCacheGb:1,composerMinLrRatio:.1,composerLrSchedule:'constant',composerPrecision:'bf16',composerPrepareWorkers:1,composerCacheDir:'cache/event_set_stream_v1'})) {
    const node=element('#'+id);if(node)node.value=String(value);
  }
}
function generationPayload() {
  if(currentGenerationFamily()==="chart_composer_v2")return composerV2GenerationPayload();
  if (currentGenerationFamily() === "chart_stream_v1") return chartStreamGenerationPayload();
  if (currentGenerationFamily() === "chart_composer_v1") return chartComposerGenerationPayload();
  if (["song_form_v1", "song_form_v2"].includes(currentGenerationFamily())) return songFormGenerationPayload();
  const condition = collectCondition();
  return {
    checkpoint: valueOrDefault("#checkpointPath", "").trim(),
    audio: valueOrDefault("#audioPath", "").trim(),
    audio_features: valueOrDefault("#featurePath", "").trim() || null,
    output: valueOrDefault("#outputPath", "").trim() || null,
    model_family: currentModelFamily(),
    device: valueOrDefault("#generationDevice", "auto"),
    // Let the server discover the corpus condition profile next to the active
    // event manifest, so a bare WebUI generation request does not fall back to
    // an out-of-distribution all-0.5 control vector.
    manifest: datasetPaths().manifest,
    condition,
    level: condition.level,
    candidates: Number(valueOrDefault("#candidates", 1)),
    seed: valueOrDefault("#seed", "").trim() === "" ? null : Number(element("#seed").value),
    temperature: Number(valueOrDefault("#temperature", 0.65)),
    top_p: Number(valueOrDefault("#topP", 0.9)),
    guidance_scale: Number(valueOrDefault("#guidanceScale", 1)),
    sample: checkedOrDefault("#sampleEvents", true),
    repetition_penalty: Number(valueOrDefault("#repetitionPenalty", 1.05)),
    repetition_window: Number(valueOrDefault("#repetitionWindow", 56)),
    max_tokens: Number(valueOrDefault("#maxTokens", 768)),
    tempo_mode: valueOrDefault("#tempoMode", "constant"),
    division: Number(valueOrDefault("#division", 16)),
    presence_threshold: Number(valueOrDefault("#presenceThreshold", 0.45)),
    button_threshold: Number(valueOrDefault("#buttonThreshold", 0.45)),
    intensity_follow: checkedOrDefault("#intensityFollow", true),
    intensity_follow_span: Number(valueOrDefault("#intensityFollowSpan", 0.5)),
    intensity_mode: valueOrDefault("#intensityMode", "auto"),
    intensity_curve_strength: Number(valueOrDefault("#intensityCurveStrength", 0.65)),
    intensity_curve_mode: valueOrDefault("#intensityCurveMode", "follow"),
    intensity_curve_enabled: checkedOrDefault("#intensityCurveEnabled", false),
    intensity_curve: intensityCurvePayload(),
    tail_damping: checkedOrDefault("#tailDamping", false),
    tail_damping_windows: Number(valueOrDefault("#tailDampingWindows", 4)),
    tail_damping_strength: Number(valueOrDefault("#tailDampingStrength", 0.6)),
    audio_bounds_trim: checkedOrDefault("#audioBoundsTrim", true),
    audio_bounds_lead: Number(valueOrDefault("#audioBoundsLead", 0)),
    silence_threshold: Number(valueOrDefault("#silenceThreshold", 0.15)),
    silence_window_share: Number(valueOrDefault("#silenceWindowShare", 0.05)),
    bpm: valueOrDefault("#bpm", "").trim() === "" ? null : Number(element("#bpm").value),
    first: Number(valueOrDefault("#first", 0)),
    title: valueOrDefault("#title", "Generated maimai chart"),
    artist: valueOrDefault("#artist", ""),
    chart_level: valueOrDefault("#chartLevel", ""),
    designer: valueOrDefault("#designer", "AI"),
    gold_stats_path: valueOrDefault("#goldStatsPath", "").trim() || null,
    tempo_map: parseTempoMap(),
    // Playability constraints (added 2026-10-10).  Server-side these become
    // the --enable-conflict-detection / --conflict-repair-mode flags, and they
    // These are static post-process options. The server validates their ranges
    // and records that unsupported checkpoint capabilities are not advertised.
    conflict_detection_enabled: checkedOrDefault("#conflictDetectionEnabled", false),
    detect_hard_conflicts: checkedOrDefault("#detectHardConflicts", true),
    conflict_repair_mode: checkedOrDefault("#conflictDetectionEnabled", false)
      ? valueOrDefault("#conflictRepairMode", "none") : "none",
    conflict_retries: checkedOrDefault("#conflictDetectionEnabled", false)
      && valueOrDefault("#conflictRepairMode", "none") !== "none"
      ? Number(valueOrDefault("#conflictRetries", 0)) : 0,
    detect_soft_conflicts: checkedOrDefault("#detectSoftConflicts", false),
    conflict_kind_modes: checkedOrDefault("#conflictDetectionEnabled", false) ? conflictKindModesFromUI() : null,
    tap_on_slide_threshold_ms: Number(valueOrDefault("#tapOnSlideThresholdMs", 200)),
    conflict_backtrack_windows: Number(valueOrDefault("#conflictBacktrackWindows", 0)),
    conflict_backtrack_budget: checkedOrDefault("#conflictBacktrackUnlimited", false)
      ? -1 : Number(valueOrDefault("#conflictBacktrackBudget", 4)),
    sustain_repair_mode: valueOrDefault("#sustainRepairMode", "off"),
    note_repair_mode: valueOrDefault("#conflictRepairMode", "none") === "local"
      ? valueOrDefault("#noteRepairMode", "edit_then_delete") : "delete",
    fine_note_strength: Number(valueOrDefault("#fineNoteStrength", 0.5)),
    local_onset_bias_strength: element("#localOnsetBiasStrength")?.disabled
      ? 0 : Number(valueOrDefault("#localOnsetBiasStrength", 0)),
  };
}

function chartStreamGenerationPayload() {
  return { model_family: "chart_stream_v1", checkpoint: valueOrDefault("#checkpointPath", "").trim(),
    audio: valueOrDefault("#audioPath", "").trim(), output: valueOrDefault("#outputPath", "").trim() || null,
    bpm: valueOrDefault("#bpm", "").trim() === "" ? null : Number(valueOrDefault("#bpm", 120)),
    tempo_map: parseTempoMap(),
    seed: valueOrDefault("#seed", "").trim() === "" ? null : Number(valueOrDefault("#seed", 0)),
    first: Number(valueOrDefault("#first", 0)), level: Number(valueOrDefault("#level", 13.3)),
    temperature: Number(valueOrDefault("#temperature", .7)), device: formDevice("#generationDevice"),
    position_temperature: Number(valueOrDefault("#streamPositionTemperature", .15)),
    eos_temperature: Number(valueOrDefault("#streamEosTemperature", 1)),
    position_audio_strength: Number(valueOrDefault("#streamPositionAudioStrength", .25)),
    position_density_strength: Number(valueOrDefault("#streamPositionDensityStrength", .12)),
    repetition_strength: Number(valueOrDefault("#streamRepetitionStrength", .6)),
    repetition_structure_strength: Number(valueOrDefault("#streamRepetitionStructureStrength", .6)),
    configuration_exploration_temperature: Number(valueOrDefault("#streamConfigurationExplorationTemperature", 1)),
    repetition_escape_attempts: Number(valueOrDefault("#streamRepetitionEscapeAttempts", 1)),
    sustain_end_sec: valueOrDefault("#streamSustainEndSec","").trim() === "" ? null : Number(valueOrDefault("#streamSustainEndSec",0)),
    repetition_min_seconds: Number(valueOrDefault("#streamRepetitionMinSeconds", 4)),
    repetition_min_beats: Number(valueOrDefault("#streamRepetitionMinBeats", 8)),
    position_beam_size: Number(valueOrDefault("#streamPositionBeamSize", 4)),
    position_complexity: Number(valueOrDefault("#streamPositionComplexity", .35)),
    position_complexity_floor: Number(valueOrDefault("#streamPositionComplexityFloor", 8)),
    learned_ending_strength: Number(valueOrDefault("#streamLearnedEndingStrength", 1)),
    constraint_min_probability: Number(valueOrDefault("#streamConstraintMinProbability", .05)),
    event_reconsider_attempts: Number(valueOrDefault("#streamEventReconsiderAttempts", 4)),
    position_interval_strength: Number(valueOrDefault("#streamPositionIntervalStrength", 1)),
    unplanned_gap_penalty_strength: Number(valueOrDefault("#streamUnplannedGapPenaltyStrength", .5)),
    rhythm_diversity_strength: Number(valueOrDefault("#streamRhythmDiversityStrength", .6)),
    early_eos_penalty: Number(valueOrDefault("#streamEarlyEosPenalty", 0)),
    max_denominator: Number(valueOrDefault("#streamMaxDenominator", 384)),
    max_interval_denominator: Number(valueOrDefault("#streamMaxIntervalDenominator", 96)),
    min_spacing_ms: Number(valueOrDefault("#streamMinSpacingMs", 20)),
    style_balance_strength: Number(valueOrDefault("#streamStyleBalanceStrength", 0)),
    section_type_strength: Number(valueOrDefault("#streamSectionTypeStrength", 1)),
    plan_intent_strength: Number(valueOrDefault("#streamPlanIntentStrength", 1)),
    configuration_candidates: Number(valueOrDefault("#streamConfigurationCandidates",4)),
    local_chord_strength: Number(valueOrDefault("#streamLocalChordStrength", 0)),
    physical_feasibility_strength: Number(valueOrDefault("#streamPhysicalStrength", 0)),
    max_button_heads: Number(valueOrDefault("#streamMaxButtonHeads", 2)),
    hold_protection: valueOrDefault("#streamHoldProtection", "on"),
    reference_check_attempts: Number(valueOrDefault("#streamReferenceChecks",8)),
    generation_end_sec: valueOrDefault("#streamGenerationEndSec","").trim() === "" ? null : Number(valueOrDefault("#streamGenerationEndSec",0)),
    trim_exact_silence: valueOrDefault("#streamTrimExactSilence","on"),
    condition: collectCondition() };
}

function intensityCurvePayload() {
  if (!checkedOrDefault("#intensityCurveEnabled", false)) return null;
  const profile = intensityCurveState.profile;
  if (!intensityCurveState.saved || intensityCurveState.contextStale || !profile
      || !Array.isArray(profile.points) || !profile.points.length) {
    return null;
  }
  return {
    schema: "maichart-intensity-curve-v1",
    audio_sha256: profile.audio_sha256 || intensityCurveState.hash || null,
    duration_sec: Number(profile.duration_sec || 0),
    points: profile.points.map((point) => ({ time_sec: Number(point.time_sec), intensity: Math.max(0, Math.min(1, Number(point.intensity ?? 0.5))) })),
    rest_regions: normalizeRestRegions(profile.rest_regions, Number(profile.duration_sec || 0)),
    method: profile.method || "music-tension-proxy-v1",
    edited: profile.edited === true,
    ...(profile.checkpoint ? { checkpoint: profile.checkpoint } : {}),
    ...(profile.checkpoint_identity ? { checkpoint_identity: profile.checkpoint_identity } : {}),
    ...(profile.condition ? { condition: profile.condition } : {}),
    ...(profile.level !== undefined ? { level: profile.level } : {}),
    ...(profile.bpm !== undefined ? { bpm: profile.bpm } : {}),
    ...(profile.bpm_input !== undefined ? { bpm_input: profile.bpm_input } : {}),
    ...(profile.first !== undefined ? { first: profile.first } : {}),
    ...(profile.tempo_mode ? { tempo_mode: profile.tempo_mode } : {}),
    ...(profile.tempo_changes ? { tempo_changes: profile.tempo_changes } : {}),
    ...(profile.explicit_tempo_map !== undefined ? { explicit_tempo_map: profile.explicit_tempo_map } : {}),
    ...(profile.tempo_map_sha256 !== undefined ? { tempo_map_sha256: profile.tempo_map_sha256 } : {}),
    ...(profile.meter_map !== undefined ? { meter_map: profile.meter_map } : {}),
    ...(profile.quality_profile !== undefined ? { quality_profile: profile.quality_profile } : {}),
    ...(profile.curve_bin_seconds !== undefined ? { curve_bin_seconds: profile.curve_bin_seconds } : {}),
    ...(profile.time_axis ? { time_axis: profile.time_axis } : {}),
    ...(profile.components !== undefined ? { components: profile.components } : {}),
    ...(profile.curve_resolution_sec !== undefined ? { curve_resolution_sec: profile.curve_resolution_sec } : {}),
    ...Object.fromEntries(["baseline_head_rate", "baseline_p95_rate", "intensity_mapping_policy", "intensity_rate_cap", "model_prediction_available", "preview_only", "readout_policy",
      "reference_points", "reference_event_rates", "reference_head_rates", "model_family", "architecture_profile"]
      .filter(key => profile[key] !== undefined).map(key => [key, profile[key]])),
  };
}

function parseTempoMap() {
  const target = element("#tempoMap");
  const text = target ? target.value.trim() : "";
  if (!text) return null;
  const parsed = JSON.parse(text);
  if (!Array.isArray(parsed) && (!parsed || !Array.isArray(parsed.tempo_changes))) {
    throw new Error("Tempo map 必须是数组，或包含 tempo_changes 数组的对象");
  }
  return parsed;
}

function parseMeterMap(selector) {
  const text = valueOrDefault(selector, "").trim();
  if (!text) return null;
  let parsed;
  try { parsed = JSON.parse(text); }
  catch (_) { throw new Error("拍号表必须填写有效的 JSON 数组。"); }
  if (!Array.isArray(parsed)) throw new Error("拍号表必须是 JSON 数组；留空表示未知拍号。");
  parsed.forEach((meter) => {
    const beat = meter?.beat;
    const beatText = typeof beat === "number" && Number.isSafeInteger(beat) ? String(beat) : beat;
    if (typeof beatText !== "string" || !/^\+?\d+(?:\/[1-9]\d*)?$/.test(beatText.trim())) {
      throw new Error('拍号表的 beat 必须是非负整数或有理数字符串，例如 "0"、"9/2"。');
    }
    if (!Number.isSafeInteger(meter.numerator) || meter.numerator <= 0
        || !Number.isSafeInteger(meter.denominator) || meter.denominator <= 0) {
      throw new Error("拍号的 numerator 和 denominator 必须是正整数。");
    }
    if (meter.grouping !== undefined && (!Array.isArray(meter.grouping)
        || meter.grouping.some((value) => !Number.isSafeInteger(value) || value <= 0)
        || (meter.grouping.length && meter.grouping.reduce((sum, value) => sum + value, 0) !== meter.numerator))) {
      throw new Error("拍组 grouping 必须由正整数组成，非空时总和须等于 numerator。");
    }
  });
  return parsed;
}

// ---------------------------------------------------------------------------
// Training-mode dependent controls
// ---------------------------------------------------------------------------
const COMPOSER_FRESH_TRAINING_DEFAULTS = Object.freeze({
  epochs:20,batchSize:64,gradAccumulation:4,learningRate:.0002,device:'auto',
  checkpointDir:'checkpoints/prism_composer_v16_signal',
  composerDModel:384,composerDimFeedforward:1536,composerLayers:6,composerNhead:8,
  composerHistoryEvents:128,composerCandidateCount:96,composerActionSlots:8,
  composerArchitectureProfile:'audio_grounded_v3',composerActionObjectivePolicy:'auto',
  composerHistoryAnchorPolicy:'auto',composerHistoryMemoryPolicy:'auto',composerHistoryAttentionPolicy:'auto',
  composerHeadTimingPolicy:'auto',composerCrossEventRelationPolicy:'auto',composerFutureHorizonPolicy:'auto',composerEchoPriorPolicy:'auto',composerTimingDistributionPolicy:'auto',composerButtonLoadPolicy:'auto',composerFineTimingPolicy:'auto',composerDensityFeedbackPolicy:'auto',composerActionHeadPolicy:'auto',composerConfigurationQualityPolicy:'auto',composerConditionMaskProb:0,composerActionStatePolicy:'auto',
  composerMotifMemoryPolicy:'auto',composerSlideFactorPolicy:'auto',composerMotifMaxEvents:16,
  composerStructurePolicy:'auto',composerStructureBackend:'auto',composerStructureDropout:.15,composerContactPolicy:'auto',
  composerRecoveryTouchLayoutPolicy:'auto',composerDropout:.1,composerLensWidth:64,composerLensChunkSize:32,
  composerHistoryResidualBound:1,composerHistoryMemoryRmsBound:8,composerHistoryDropout:.3,composerLedgerDropout:.5,
  composerPlacementWeight:.15,composerPlannerWeight:.2,composerRecoveryInterval:25,
  composerRecoverySamples:2,composerRecoverySteps:2,composerRecoveryWeight:.1,
  composerRecoverySamplingPolicy:'auto',composerRecoveryPointerTemperature:1,composerRecoveryActionTemperature:.85,
  composerCpuThreads:2,composerSampleWorkers:1,composerPrefetchBatches:2,composerSampleLocality:4,composerSampleSequencePolicy:'auto',
  composerSymbolicDropout:.15,composerAudioPointerWeight:.15,
  composerLrSchedule:'auto',composerWarmupSteps:'',composerMinLrRatio:'',
  composerPrecision:'auto',composerSeed:0,composerLimitCharts:0,composerPrepareWorkers:4,
  composerAudioCacheGb:4,composerCacheDir:'cache/chart_composer_v1',composerCheckpointIntervalSteps:500,
  composerProfileSteps:0,composerLogInterval:25,composerMaxSteps:0,composerGenerationEvalSongs:2,composerGenerationEvalIntervalEpochs:1,
  composerOptimizerImplementation:'auto',composerExportInference:true,composerRebatchResume:false,
  composerValidationSource:'auto',composerValidationManifest:'',composerHoldoutFraction:.05,composerValidationMaxBatches:32,
  composerValidationBatchSize:8,composerGenerationEvalSeeds:'0,1',composerArchiveCheckpoints:'auto',
});

function applyComposerFreshTrainingDefaults() {
  if(currentTrainingMode()!=='fresh'||currentModelFamily()!=='chart_composer_v1')return false;
  if(valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1'){applyEventSetFreshDefaults();updateEventSetControls();return true;}
  for(const [id,value] of Object.entries(COMPOSER_FRESH_TRAINING_DEFAULTS)) {
    const target=element('#'+id);if(!target)continue;
    if(target.type==='checkbox')target.checked=value===true;
    else target.value=String(value);
  }
  const learningRate=element('#learningRate');if(learningRate)learningRate.dataset.touched='';
  return true;
}

function setTrainingModeDefaults({resetComposer=false}={}) {
  const mode = currentTrainingMode();
  if(resetComposer)applyComposerFreshTrainingDefaults();
  const eventSet=typeof currentModelFamily==='function' && currentModelFamily()==='chart_composer_v1' && valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1';
  if(resetComposer && eventSet && mode==='finetune')for(const [id,value] of [['batchSize',16],['gradAccumulation',4],['composerAudioCacheGb',1]]){const node=element('#'+id);if(node)node.value=String(value);}
  const learningRate = element("#learningRate");
  if (learningRate && learningRate.dataset.touched !== "1") {
    learningRate.value = eventSet ? "0.0005" : mode === "finetune" ? "0.00002" : "0.0002";
  }
  const help = element("#learningRateHelp");
  if (help) help.textContent = eventSet ? "整拍集合新阶段默认固定学习率 0.0005；续训继承检查点。" : mode === "finetune" ? "finetune 默认 2e-5。" : "fresh 默认 2e-4。";
  updateTrainingModeHelp(mode);
}

function updateTrainingModeHelp(mode = currentTrainingMode()) {
  const songForm = ["song_form_v1", "song_form_v2", "chart_stream_v1"].includes(element("#modelFamily")?.value);
  const restartEpoch = songForm && element("#formResumeDataPolicy")?.value === "restart_epoch";
  const modeHelp = element("#trainingModeHelp");
  if (modeHelp) {
    modeHelp.textContent = mode === "fresh"
      ? "从随机初始化开始；不需要已有 checkpoint。"
      : mode === "resume"
        ? restartEpoch
          ? "迁移续训：保留权重与优化器；原素材不变且解析结果变化时，清空旧游标并重跑当前轮。请使用新的输出目录。"
          : "严格从断点继续：需要与当前数据合同、模型族完全一致的 checkpoint。"
        : "已有模型加强训练：加载权重并新建优化器，可用小型精选集。允许迁移池化/过滤策略、上下文 token 上限和质量采样；模型族、词表、前置拍数与规划器保持兼容。";
  }
  const epochsHelp = element("#epochsHelp");
  if (epochsHelp) {
    epochsHelp.textContent = mode === "fresh"
      ? "本次从头训练的轮数。短跑可先限制 batch 数验证流程；正式训练轮数按数据量和实际生成效果选择。"
      : mode === "resume"
        ? currentModelFamily()==='chart_composer_v1'
          ? (valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1'?'整拍集合的 epochs 是总目标轮数；例如从第4轮恢复并希望训练到第20轮，请填20。':'Composer 的 epochs 是本次继续的轮数，包含尚未完成的当前轮。第4轮中途恢复并希望到第20轮，应填17；第4轮已完成则填16。')
          : songForm
          ? "本次继续训练的轮数，从 checkpoint 已完成轮次的下一轮开始；迁移时重跑未完成轮次，保留其中已学到的权重。"
          : "resume 的 epochs 是从断点额外训练的轮数；会从 checkpoint 的 epoch 接着计数，早停会按验证 loss 停止。"
        : "finetune 从新阶段 epoch 1 开始计数；epochs 是本次加强训练的轮数，早停会按验证 loss 停止。";
  }
}

function updateDatasetButtons() {
  const busy = datasetUploading || preprocessStarting || trainingStarting || trainingPreflight || liveComposerTrainingTasks.size > 0;
  const mapping = {
    "#datasetComposeButton": busy || datasetWorkspaceState.applying,
    "#trainingDatasetApplyButton": busy || datasetWorkspaceState.applying,
    "#preprocessButton": busy,
    "#inspectManifestButton": busy,
    "#trainButton": busy,
    "#autoPipelineButton": busy || (typeof currentModelFamily==="function" && currentModelFamily()==="chart_composer_v2") || selectedTrainingManifests().length > 1,
    "#goldScanButton": busy,
    "#goldBuildButton": busy,
    "#goldDesignerListButton": busy,
    "#datasetSelect": datasetUploading,
    "#datasetZip": datasetUploading,
  };
  Object.entries(mapping).forEach(([selector, disabled]) => {
    const target = element(selector);
    if (target) target.disabled = disabled;
  });
  const trainButton = element("#trainButton");
  if (trainButton) trainButton.title = liveComposerTrainingTasks.size
    ? "已有 Composer 训练正在运行或取消，等待任务退出后可继续提交。" : "";
  const needsCheckpoint = currentTrainingMode() !== "fresh";
  ["#trainingCheckpointFile", "#trainingCheckpointPath", "#checkpointInspectButton"].forEach((selector) => {
    const target = element(selector);
    if (target) target.disabled = !needsCheckpoint || busy;
  });
  const resetOptimizer = element("#resetOptimizer");
  if (resetOptimizer) resetOptimizer.disabled = currentTrainingMode() !== "finetune" || busy;
}

function updateTrainingControls() {
  const v2Resume=currentModelFamily()==="chart_composer_v2" && currentTrainingMode()==="resume";
  const smokePreset = element('#v2TrainSmokePreset');
  if (smokePreset) {
    smokePreset.disabled = v2Resume;
    smokePreset.title = v2Resume ? '严格续训沿用检查点训练计划。新建训练或微调时可选短跑。' : '';
    if (v2Resume && restoreV2TrainPlan()) {
      markActivePresetChip('data-v2-train-preset', '');
      element('#v2TrainPresetStatus').textContent = '已恢复短跑前的训练计划。严格续训沿用检查点计划。';
    }
  }
  if(element('#v2TrainQualityEnabled'))element('#v2TrainQualityEnabled').disabled=v2Resume;
  const v2Fixed=new Set(['song_layers','d_model','layers','nhead','dim_feedforward','dropout','window_beats','history_beats','future_beats','ticks_per_beat','max_tokens','audio_backend','mert_model','refresh_events']);
  for(const key of (typeof COMPOSER_V2_FIELDS!=="undefined"?COMPOSER_V2_FIELDS:[])){
    const node=element("#"+v2FieldId(key));if(node){
      const resourceBudget = ['cuda_memory_budget_gb','audio_cache_gb','save_every','early_stopping_patience','activation_checkpointing'].includes(key);
      node.disabled=(v2Resume && !resourceBudget) || (currentModelFamily()==='chart_composer_v2' && currentTrainingMode()==='finetune' && v2Fixed.has(key));
      node.title=node.disabled?'继承 checkpoint 的固定设置；模型合同变化需要从头训练。':'';
    }
  }
  const autoQuality=element('#v2TrainAutoQualityPolicy');
  if (autoQuality) autoQuality.disabled=v2Resume || !checkedOrDefault('#v2TrainQualityEnabled',true);
  updateV2DifficultyControls();
  const validationSource=element("#v2TrainValidationSource")?.value||"auto";
  const v2Validation=element("#v2TrainValidationManifest");if(v2Validation)v2Validation.disabled=v2Resume||validationSource!=="manual";
  const v2Fraction=element("#v2TrainHoldoutFraction");if(v2Fraction)v2Fraction.disabled=v2Resume||validationSource!=="auto";
  if(typeof updateZeroUnmarkedTrainingNotice==="function")updateZeroUnmarkedTrainingNotice();
  updateFamilyFields();
  updateDatasetButtons();
  updateTrainingModeHelp();
  const resume = currentTrainingMode() === "resume";
  (typeof STREAM_GROUNDING_FIELDS !== "undefined" ? STREAM_GROUNDING_FIELDS : []).forEach(([id])=>{ const target=element('#'+id); if(target) target.disabled=resume; });
  const resumePolicy = element("#formResumeDataPolicyGroup");
  if (resumePolicy) resumePolicy.hidden = !resume || !["song_form_v1", "song_form_v2", "chart_stream_v1"].includes(currentModelFamily());
  ["#onsetProfile", "#onsetLossWeight", "#localPoolingPolicy", "#actionStateProfile"].forEach((selector) => {
    const target = element(selector);
    if (target) target.disabled = resume;
  });
  if (currentModelFamily() === "chart_stream_v1" && currentTrainingMode() === "fresh") {
    ["#streamDModel", "#streamLayers", "#streamNhead", "#streamDimFeedforward",
      "#streamMaxTokens", "#streamMaxHistoryTokens", "#streamMemoryTokens", "#streamDropout"].forEach((selector) => {
      const target = element(selector); if (target) target.disabled = false;
    });
  }
  if (currentModelFamily()==='chart_composer_v1' && currentTrainingMode()==='fresh') {
    for (const id of ['composerDModel','composerDimFeedforward','composerLayers','composerNhead',
      'composerPhraseContextPolicy','composerDurationAudioPolicy','composerPhraseLossWeight','composerLearningPolicy','composerTransitionExitWeight','composerPrefixAugmentationRate','composerGenerationEvalTemperature','composerRolloutSteps','composerWindowSeconds','composerAudioGridSeconds','composerPlanWeight','composerPlanBlankWindowSeconds','composerSymbolicPretrainEpochs','composerHistoryEvents','composerCandidateCount','composerActionSlots','composerArchitectureProfile','composerActionObjectivePolicy','composerHistoryAnchorPolicy','composerHistoryMemoryPolicy','composerHistoryAttentionPolicy','composerHeadTimingPolicy','composerCrossEventRelationPolicy','composerFutureHorizonPolicy','composerEchoPriorPolicy','composerTimingDistributionPolicy','composerButtonLoadPolicy','composerFineTimingPolicy','composerDensityFeedbackPolicy','composerActionHeadPolicy','composerConfigurationQualityPolicy','composerConditionMaskProb',
      'composerActionStatePolicy','composerMotifMemoryPolicy','composerSlideFactorPolicy','composerMotifMaxEvents','composerRecoveryTouchLayoutPolicy','composerStructurePolicy','composerStructureBackend','composerStructureDropout','composerContactPolicy',
      'composerLensWidth','composerLensChunkSize','composerHistoryResidualBound','composerHistoryMemoryRmsBound','composerHistoryDropout',
      'composerLedgerDropout','composerPlacementWeight','composerPlannerWeight','composerRecoveryInterval',
      'composerRecoverySamples','composerRecoverySteps','composerRecoveryWeight','composerRecoverySamplingPolicy','composerRecoveryPointerTemperature','composerRecoveryActionTemperature','composerDropout']) {
      const target=element('#'+id);if(target)target.disabled=false;
    }
  }
  const composer = currentModelFamily() === 'chart_composer_v1';
  const freshDefaults=element("#composerApplyFreshDefaults");
  if(freshDefaults)freshDefaults.disabled=!composer||currentTrainingMode()!=='fresh';
  const rebatch = element('#composerRebatchResume');
  if (rebatch) {
    rebatch.disabled = !composer || !resume;
    if (!composer || !resume) rebatch.checked = false;
  }
  for (const id of ['batchSize','gradAccumulation']) {
    const target=element('#'+id);
    if (target) target.disabled=composer && resume && !rebatch?.checked;
    if (target && composer && resume && !rebatch?.checked && checkpointInspection.key===currentCheckpointKey() && checkpointInspection.data?.valid===true) {
      const key=id==='batchSize'?'batch_size':'grad_accumulation';
      const saved=checkpointInspection.data.args?.[key];
      if (saved!==undefined) target.value=String(saved);
    }
  }
  const composerDropout=element('#composerDropout');
  const composerAnchor=element('#composerHistoryAnchorPolicy');
  if (composerAnchor) composerAnchor.disabled=composer && resume;
  for (const id of ['composerActionObjectivePolicy','composerActionStatePolicy','composerMotifMemoryPolicy','composerSlideFactorPolicy','composerMotifMaxEvents','composerRecoveryTouchLayoutPolicy','composerStructurePolicy','composerStructureBackend','composerStructureDropout','composerContactPolicy','composerHistoryMemoryPolicy','composerHistoryAttentionPolicy','composerHeadTimingPolicy','composerCrossEventRelationPolicy','composerFutureHorizonPolicy','composerEchoPriorPolicy','composerTimingDistributionPolicy','composerButtonLoadPolicy','composerFineTimingPolicy','composerDensityFeedbackPolicy','composerActionHeadPolicy','composerConfigurationQualityPolicy','composerConditionMaskProb','composerLrSchedule','composerWarmupSteps','composerMinLrRatio']) {
    const target=element('#'+id); if(target)target.disabled=composer && resume;
  }
  const historyMemoryBound=element('#composerHistoryMemoryRmsBound');
  if(historyMemoryBound)historyMemoryBound.disabled=composer && resume;
  if (composerDropout) composerDropout.disabled=currentTrainingMode()!=='fresh';
  updateComposerRecoverySamplingControls();
  updateSongFormRhythmGeometryControls();
  updateSongFormQualityControls();
  if (typeof updateSongFormQualityProfileControls === "function") updateSongFormQualityProfileControls();
  updateEventSetControls();
  window.MaichartWorkflow?.refresh();
}

function updateComposerRecoverySamplingControls() {
  const policy=element('#composerRecoverySamplingPolicy')?.value||'auto';
  const legacy=policy==='calibrated_kind_touch_rings_v2';
  for (const [id,fallback] of [['composerRecoveryPointerTemperature','1'],['composerRecoveryActionTemperature','0.85']]) {
    const target=element('#'+id);
    if (!target) continue;
    if (legacy) target.value=fallback;
    target.disabled=currentTrainingMode()==='resume'||legacy;
  }
}

// ---------------------------------------------------------------------------
// Task handling
// ---------------------------------------------------------------------------
const TASK_KIND_LABEL = {
  review_bundle_export:'导出已确认片段',review_bundle_preview:'预览片段包',review_bundle_import:'导入片段包',
  candidate_budget: "训练来源学习份额估算",
  chart_stream_train: "连续谱面训练", chart_stream_generate: "连续谱面生成",
  composer_v2_train: "Composer v2 训练", composer_v2_generate: "Composer v2 生成",
  composer_train: "chart_composer_v1 训练", composer_generate: "chart_composer_v1 生成",
  preprocess: "预处理", train: "训练", generate: "生成", form_train: "song-form 训练", form_generate: "song-form 生成", form_analyze: "song-form 分析", form_revise: "song-form 修订", intensity: "强度曲线", gold: "金标集", acceptance: "验收",
};
const taskWatchers = new Map();
const trainingMetricSupportCache = new Map();

function generationTime(value) {
  const sec=Math.max(0,Math.floor(Number(value)||0));
  return `${Math.floor(sec/60).toString().padStart(2,'0')}:${(sec%60).toString().padStart(2,'0')}`;
}

function generationStageLabel(stage) {
  return ({queued:'等待开始',starting:'正在启动',loading_checkpoint:'读取检查点',loading_model:'载入模型',
    freezing_checkpoint:'准备检查点副本',detecting_timing:'识别 BPM 和偏移',preparing_audio:'准备音频特征',
    encoding_audio:'读取歌曲上下文',generating:'正在写谱',comparing:'比较候选写法',retrying:'调整冲突位置',
    exporting:'导出谱面',completed:'已完成',failed:'生成失败',cancelled:'已中止',unknown:'等待详细进度'})[stage]||'正在生成';
}

function composerV2ProgressState(task) {
  const result=task.result||{},settings=result.effective_settings||{};
  const batch=result.composer_v2_batch_progress||result.composer_v2_batch;
  const single=result.composer_v2_generation_progress;
  if(batch?.count>1) {
    let rows=batch.chart_progress||[];
    if(!rows.length) {
      const terminal=result.composer_v2_batch?.charts||[];
      const base=String(settings.batch_manifest||settings.output||'').replace(/[\\/][^\\/]*$/,'');
      const stem=String(settings.output||'maidata.txt').split(/[\\/]/).pop().replace(/\.[^.]+$/,'');
      rows=Array.from({length:Math.min(32,batch.count)},(_,i)=>{
        const index=i+1,done=terminal.find(row=>row.index===index),seed=Number(settings.seed||0)+i;
        if(done)return {...done,stage:done.status,progress:1};
        const active=batch.active?.includes(index);
        return {index,seed,status:active?'running':'unknown',stage:active?'generating':'unknown',
          log:base?`${base}/${String(index).padStart(2,'0')}_seed_${seed}/${stem}.generation.log`:null};
      });
    }
    const count=Number(batch.count)||1;
    const finished=Math.max((Number(batch.completed)||0)+(Number(batch.failed)||0),rows.filter(row=>['completed','failed'].includes(row.status)).length);
    const progress=Number.isFinite(batch.progress)?batch.progress:
      (finished+rows.filter(row=>row.status==='running').reduce((sum,row)=>sum+Math.min(.995,Math.max(0,Number(row.progress)||0)),0))/count;
    rows=rows.map(row=>({audio_sec:0,...row,log:row.log||null}));
    const running=rows.filter(row=>row.status==='running'),times=rows.filter(row=>row.status==='completed'&&row.elapsed_sec>0).map(row=>row.elapsed_sec);
    for(const row of running)if(Number.isFinite(row.eta_sec)&&row.elapsed_sec>0)times.push(row.elapsed_sec+row.eta_sec);
    const average=times.length?times.reduce((a,b)=>a+b,0)/times.length:null;
    const lanes=running.map(row=>Number.isFinite(row.eta_sec)?Math.max(0,row.eta_sec):average!==null?Math.max(0,average-(row.elapsed_sec||0)):null);
    let eta=null;
    if(average!==null&&!lanes.includes(null)&&batch.status!=='preparing') {
      while(lanes.length<Math.max(1,Number(batch.parallel)||1))lanes.push(0);
      for(let queued=Math.max(0,count-finished-running.length);queued>0;queued--){const lane=lanes.indexOf(Math.min(...lanes));lanes[lane]+=average;}
      eta=Math.max(0,...lanes);
    }
    if(!['queued','running'].includes(task.status))eta=null;
    return {batch,rows,progress:Math.min(1,Math.max(0,progress)),elapsed:batch.elapsed_sec,eta,
      indeterminate:batch.status==='preparing'};
  }
  return {batch:null,rows:single&&single.index!==0?[single]:[],progress:Math.min(1,Math.max(0,Number(single?.progress)||0)),
    elapsed:single?.elapsed_sec,eta:task.status==='running'&&Number.isFinite(single?.eta_sec)?single.eta_sec:null,
    indeterminate:!single?.audio_duration_sec&&single?.status!=='completed'};
}

function renderGenerationProgress(task) {
  const panel=element('#taskGenerationProgress'),list=element('#taskGenerationCharts'),summary=element('#taskGenerationSummary');
  if(!panel||!list||!summary)return;
  panel.hidden=task.kind!=='composer_v2_generate';
  if(panel.hidden){panel._signature=null;return;}
  const state=composerV2ProgressState(task),signature=JSON.stringify([task.task_id,task.status,state]);
  if(panel._signature===signature)return;
  panel._signature=signature;
  summary.textContent=`${state.batch?`已完成 ${state.batch.completed||0}/${state.batch.count} 张 · 并行 ${state.batch.active?.length||0}/${state.batch.parallel||1} 张 · `:''}${state.elapsed!=null?`已用 ${generationTime(state.elapsed)} · `:''}${state.eta!==null?`预计还需约 ${generationTime(state.eta)} · `:['queued','running'].includes(task.status)?'剩余时间正在估算 · ':''}按已写歌曲位置估算进度`;
  list.replaceChildren();
  const order={running:0,failed:1,queued:2,unknown:2,cancelled:3,completed:4};
  for(const row of [...state.rows].sort((a,b)=>(order[a.status]??2)-(order[b.status]??2)||(a.index||0)-(b.index||0))) {
    const card=document.createElement('div');card.className='generation-chart-progress';card.dataset.status=row.status||'running';
    const heading=document.createElement('div'),name=document.createElement('strong'),stage=document.createElement('span');
    heading.className='generation-chart-heading';name.textContent=`第 ${row.index||1} 张 · seed ${row.seed??'—'}`;
    stage.textContent=generationStageLabel(row.stage);heading.append(name,stage);card.append(heading);
    const detail=document.createElement('p');
    const percent=['completed','failed'].includes(row.status)?100:Math.min(99,Math.floor((row.progress||0)*100));
    detail.textContent=[row.audio_duration_sec?`${generationTime(row.audio_sec)} / ${generationTime(row.audio_duration_sec)}（${percent}%）`:null,
      row.events!=null?`${row.events} 个事件`:null,row.elapsed_sec!=null?`用时 ${generationTime(row.elapsed_sec)}`:null,
      row.status==='running'&&Number.isFinite(row.eta_sec)?`预计还需约 ${generationTime(row.eta_sec)}`:null].filter(Boolean).join(' · ');
    if(row.status==='running'&&row.updated_at&&Date.now()/1000-row.updated_at>10)detail.textContent+=` · 上次进度更新 ${Math.floor(Date.now()/1000-row.updated_at)} 秒前`;
    if(row.error)detail.textContent+=` · ${row.error}`;
    card.append(detail);
    const track=document.createElement('div'),bar=document.createElement('div');track.className='generation-chart-track';
    bar.style.width=`${percent}%`;track.append(bar);card.append(track);
    if(row.log&&row.status!=='queued'&&row.stage!=='starting') {
      const link=document.createElement('a');link.href=`/api/download?path=${encodeURIComponent(row.log)}`;
      link.textContent='详细日志';link.className='generation-chart-log';card.append(link);
    }
    list.append(card);
  }
  if(!state.rows.length)list.textContent='正在准备，收到进度后会显示每张谱面。';
}

function trainingSupportPaths(task) {
  if (task?.kind !== "form_train") return null;
  const result = task.result || {};
  const artifacts = Array.isArray(result.artifacts) ? result.artifacts.map(item => item?.path) : [];
  const checkpoint = [...artifacts, ...(result.paths || [])].find(path => /\.pt$/i.test(String(path || "")));
  const directory = result.checkpoint_dir || (checkpoint ? String(checkpoint).replace(/[\\/][^\\/]+$/, "") : "");
  const join = filename => directory ? `${String(directory).replace(/[\\/]+$/, "")}/${filename}` : "";
  const history = result.training_history_path || join("training_history.json");
  const model = result.training_model_summary_path || join("training_model_summary.json");
  return history || model ? {history, model} : null;
}

function applyTrainingMetricSupport(task, record) {
  if (Array.isArray(record.history)) {
    task.result = {...(task.result || {}), metric_history: record.history,
                   metric_history_source: "checkpoint_history"};
  }
  if (record.model?.schema === "song_form_training_model_summary_v1") {
    task.result = {...(task.result || {}), model_summary: record.model};
  }
}

function hydrateTrainingMetricHistory(task) {
  const paths = trainingSupportPaths(task);
  if (!paths) return;
  const key = JSON.stringify(paths);
  let record = trainingMetricSupportCache.get(task.task_id);
  if (!record || record.key !== key) {
    record = {key, fetchedAt: 0, pending: false};
    trainingMetricSupportCache.set(task.task_id, record);
    if (trainingMetricSupportCache.size > 20) trainingMetricSupportCache.delete(trainingMetricSupportCache.keys().next().value);
  }
  applyTrainingMetricSupport(task, record);
  if (record.pending || Date.now() - record.fetchedAt < 60000) return;
  record.pending = true;
  record.fetchedAt = Date.now();
  const read = async path => {
    if (!path) return null;
    const response = await fetch(`/api/download?path=${encodeURIComponent(path)}`, {cache: "no-store"});
    if (!response.ok) return null;
    return JSON.parse(await response.text());
  };
  Promise.allSettled([read(paths.history), read(paths.model)]).then(results => {
    if (trainingMetricSupportCache.get(task.task_id) !== record) return;
    const history = results[0].status === "fulfilled" ? results[0].value : null;
    const model = results[1].status === "fulfilled" ? results[1].value : null;
    if (Array.isArray(history)) record.history = history.slice(-100);
    if (model?.schema === "song_form_training_model_summary_v1") record.model = model;
    applyTrainingMetricSupport(task, record);
    if (activeTaskId === task.task_id) {
      const latest = taskWatchers.get(task.task_id)?.task || task;
      applyTrainingMetricSupport(latest, record);
      renderTaskTrainingMetrics(latest);
    }
  }).finally(() => {record.pending = false;});
}

function parseArchivedFormTrainingLog(text, entry) {
  const lines = String(text).split(/\r?\n/);
  const history = [];
  const phases = {};
  let latest = null, model = null;
  for (const line of lines) {
    const match = line.match(/(?:^|\s)(form_progress|form_epoch|form_model_summary)\s+(\{.*\})\s*$/);
    if (!match) continue;
    let value;
    try {value = JSON.parse(match[2]);} catch (_) {continue;}
    if (!value || typeof value !== "object" || Array.isArray(value)) continue;
    if (match[1] === "form_epoch") history.push(value);
    else if (match[1] === "form_model_summary") model = value;
    else {latest = value; if (value.phase && value.losses) phases[value.phase] = value;}
  }
  const argv = lines.find(line => line.startsWith("# argv=")) || "";
  const directory = argv.match(/--checkpoint-dir\s+(.+?)(?=\s+--[a-z]|$)/)?.[1]?.replace(/^"|"$/g, "") || "";
  const epochsRequested = Number(argv.match(/--epochs\s+(\d+)/)?.[1]);
  const headerId = lines.find(line => line.startsWith("# task_id="))?.slice(10);
  const result = {
    log_path: entry.log_path, checkpoint_dir: directory, metric_history: history,
    training_metrics: phases, metrics: history.at(-1), progress_detail: latest,
    phase: latest?.phase, model_summary: model, archived_status: entry.status,
    epoch_target: Number.isFinite(epochsRequested) ? epochsRequested : null,
  };
  return {
    task_id: entry.task_id || headerId || "archived-training", kind: "form_train", status: "archived",
    message: `历史训练日志 · 原记录状态：${entry.status || "未知"}；当前运行状态请查看任务列表。`,
    progress: 0, result, logs: lines.filter(line => line && !line.startsWith("#")).slice(-300),
  };
}

async function viewArchivedTrainingLog(entry, button) {
  button.disabled = true;
  button.textContent = "正在读取…";
  try {
    const response = await fetch(entry.download, {cache: "no-store"});
    if (!response.ok) throw new Error(`日志读取失败：HTTP ${response.status}`);
    const task = parseArchivedFormTrainingLog(await response.text(), entry);
    activeTaskId = task.task_id;
    renderTask(task);
  } finally {
    button.disabled = false;
    button.textContent = "查看训练走势";
  }
}

function watchTask(taskId, onFinished = null, onUpdate = null) {
  activeTaskId = taskId;
  const existing = taskWatchers.get(taskId);
  if (existing) {
    if (onFinished) existing.callbacks.add(onFinished);
    if (onUpdate) (existing.updates ||= new Set()).add(onUpdate);
    if (existing.task) renderTask(existing.task);
    return;
  }
  taskWatchers.set(taskId, { callbacks: new Set(onFinished ? [onFinished] : []), updates: new Set(onUpdate ? [onUpdate] : []), task: null });
  pollTask(taskId);
}

async function pollTask(taskId) {
  const watcher = taskWatchers.get(taskId);
  if (!watcher) return;
  try {
    const task = await api(`/api/task?id=${encodeURIComponent(taskId)}`);
    watcher.task = task;
    for (const update of watcher.updates || []) update(task);
    trackComposerTrainingTask(task);
    if (activeTaskId === taskId) renderTask(task);
    else window.MaichartWorkflow?.recordTask(task);
    if (["queued", "running", "cancelling"].includes(task.status)) {
      const hidden=typeof document!=="undefined" && document.visibilityState==="hidden";
      const training=/(?:^|_)train$/.test(task.kind||"");
      const delay=hidden ? 8000 : task.status==="cancelling" ? 800 : training ? 2000 : 800;
      setTimeout(() => pollTask(taskId), delay);
    } else {
      const expectsArtifacts = task.kind === "form_generate"
        || (Array.isArray(task.result?.artifacts) && task.result.artifacts.length > 0);
      if (task.status === "completed" && expectsArtifacts
          && !task.result?.missing_artifacts?.length && !hasTaskArtifacts(task)) {
        watcher.artifactPolls = (watcher.artifactPolls || 0) + 1;
        if (watcher.artifactPolls <= 3) {
          setTimeout(() => pollTask(taskId), 800);
          return;
        }
        task.artifacts_missing = true;
        if (activeTaskId === taskId) renderTask(task);
      }
      taskWatchers.delete(taskId);
      for (const callback of watcher.callbacks) await callback(task);
    }
  } catch (error) {
    if (activeTaskId === taskId) {
      const message = element("#taskMessage");
      if (message) message.textContent = `状态读取失败，正在重连：${error.message}`;
    }
    if (taskWatchers.has(taskId)) setTimeout(() => pollTask(taskId), 3000);
  }
}

function trackComposerTrainingTask(task) {
  if (!["composer_train","composer_v2_train"].includes(task.kind)) return;
  if (["queued", "running", "cancelling"].includes(task.status)) liveComposerTrainingTasks.add(task.task_id);
  else liveComposerTrainingTasks.delete(task.task_id);
  updateDatasetButtons();
}

function hasTaskArtifacts(task) {
  const result = task?.result || {};
  return [result.files, result.download_urls, result.paths].some((items) => Array.isArray(items) && items.length > 0);
}

function renderTask(task) {
  const kind = element("#taskKind");
  const status = element("#taskStatus");
  const progress = element("#progressBar");
  const message = element("#taskMessage");
  const epoch = element("#taskEpoch");
  const logs = element("#taskLogs");
  const epochStatus = trainingEpochStatus(task);
  if (kind) kind.textContent = `${TASK_KIND_LABEL[task.kind] || task.kind} · ${task.task_id}`;
  if (status) status.textContent = ({queued: "排队中", running: "运行中", cancelling: "正在取消", completed: "已完成", failed: "失败", cancelled: "已中断", archived: "历史记录"})[task.status] || task.status;
  if (progress) {
    const unknownTrainingTarget = task.kind === "train" && !epochStatus?.knownTarget;
    if (progress.parentElement) progress.parentElement.hidden = unknownTrainingTarget || task.status === "archived";
    const generation=task.kind==='composer_v2_generate'?composerV2ProgressState(task):null;
    const amount = ["train","composer_v2_train"].includes(task.kind) && epochStatus?.knownTarget
      ? epochStatus.progress : generation?.rows.length?generation.progress:task.progress||0;
    const percentage=generation&&['queued','running'].includes(task.status)?Math.min(99,Math.floor(amount*100)):Math.round(amount*100);
    progress.style.width = `${percentage}%`;
    const indeterminate=['queued','running'].includes(task.status) && (task.kind==='composer_v2_train'?epochStatus?.indeterminate:generation?.indeterminate);
    progress.parentElement?.classList?.toggle('indeterminate',!!indeterminate);
    if(task.kind==='composer_v2_train')progress.parentElement?.setAttribute?.('aria-label','训练总计划进度');
  }
  if (message) message.textContent = task.cancel_error
    ? `取消失败，任务尚未确认停止；请重试取消：${task.cancel_error}` : task.error || (
      task.kind==='composer_v2_generate'
        ? composerV2GenerationMessage(task)
        : task.kind==='composer_v2_train' && task.message==='composer_v2_checkpoint'?'保存检查点':task.message)||"";
  if (epoch) {
    epoch.hidden = !epochStatus;
    if (epochStatus) epoch.textContent = epochStatus.text + (
      task.result?.selection_metric === "training_loss" ? "；best 依据训练 loss（未配置验证集）"
      : task.result?.selection_metric === "validation_loss" ? "；best 依据验证 loss" : "");
  }
  hydrateTrainingMetricHistory(task);
  renderTaskTrainingMetrics(task);
  renderGenerationProgress(task);
  if (window.MaichartLogs) window.MaichartLogs.renderTask(task);
  else if (logs) logs.textContent = (task.logs || []).join("\n") || "暂无日志";
  window.MaichartWorkflow?.recordTask(task);

  const cancel = element("#cancelButton");
  if (cancel) {
    const retry = task.cancel_retryable === true;
    cancel.disabled = pendingTaskCancellations.has(task.task_id) || task.cancel_pending
      || (!["queued", "running"].includes(task.status) && !retry);
    cancel.textContent = retry ? "重试取消" : "取消任务";
  }
  const logPath = element("#taskLogPath");
  const path = task.result?.log_path;
  if (logPath) {
    logPath.hidden = !path;
    if (path) logPath.textContent = `完整日志：${path}`;
  }

  renderTaskResults(task);
  if (task.kind === "generate") renderGenerationResult(task);
  if (task.kind === "acceptance") renderAcceptanceFromTask(task);
  if (task.kind === "gold") renderGoldResult(task);
}

function composerGenerationSummary(task) {
  if(task.kind==="composer_v2_generate") {
    const report=task.result?.composer_v2_report||{},settings=task.result?.effective_settings||{};
    if(Array.isArray(task.result?.composer_v2_batch?.charts))return composerV2CompletionMessage(task);
    if(settings.count>1)return `Composer v2 批量任务：${settings.count} 张，起始 seed ${settings.seed}；逐张独立保存，完成后在顶部预览切换。`;
    const scope=report.recovery_budget_scope==="forward_progress_v2"?"单处":"全曲";
    const sampling=report.sampling_policy;
    const settingsSummary=sampling?.rng_policy==='separate_timing_action_v1'
      ? `采音 ${sampling.timing_temperature} / ${sampling.timing_top_p}，配置 ${sampling.action_temperature} / ${sampling.action_top_p}（温度 / Top-p）；`
      : '';
    const rankPolicy=report.phrase_search?.ranking_policy??report.phrase_search?.rank_policy;
    const rankSummary=rankPolicy==='likelihood_veto_v1'?'似然重排＋手感底线降序；':rankPolicy==='comfort_v2'?'左右手手感评分；':'';
    return `Composer v2；实际 seed ${report.seed??settings.seed??"未记录"}；${settingsSummary}${rankSummary}候选数 ${report.candidates??settings.candidates??"未记录"}；${scope}冲突恢复预算 ${report.recovery_budget??report.rollback_windows??settings.rollback_windows??2} 次；全曲实际撤回 ${report.rollbacks?.length??0} 次，Hold时值修复 ${report.duration_repairs?.length??0} 次；放行冲突事件 ${report.accepted_with_conflicts_count??0} 个；详见 .composer_v2.json 和冲突报告，仍需试听与游玩检查。`;
  }
  if (task.kind !== "composer_generate") return null;
  const result=task.result||{},report=result.composer_report||{},settings=result.effective_settings||{};
  const condition=report.condition||settings.condition;
  const profile=report.song_profile||{},controls=profile.effective_controls||condition||{};
  const profileDiagnostics=profile.prediction_diagnostics?.rows?.[0] ?? (Array.isArray(profile.prediction_diagnostics)?profile.prediction_diagnostics[0]:profile.prediction_diagnostics);
  const seed=report.seed??result.seed??"未记录",temperature=report.temperature??settings.temperature;
  const pointerTemperature=report.pointer_temperature??settings.pointer_temperature??temperature;
  const actionTemperature=report.action_temperature??settings.action_temperature??temperature;
  const greedy=Number(pointerTemperature)===0 && Number(actionTemperature)===0;
  const parts=[`实际 seed：${seed}；${greedy?"贪心解码，seed 不改变结果":`候选温度 ${pointerTemperature??"未记录"}；动作温度 ${actionTemperature??"未记录"}`}`];
  const architecture=report.architecture_profile??settings.architecture_profile;
  if(architecture)parts.unshift(`训练架构：${({event_set_stream_v1:'整拍集合连续生成',audio_grounded_v3:'音频锚定',music_context_v2:'音乐上下文基线'})[architecture]??architecture}（${architecture}）；更新 ${report.optimizer_steps??'未记录'} 步`);
  if(report.checkpoint??result.checkpoint)parts.push(`检查点：${report.checkpoint??result.checkpoint}`);
  if(controls.radar?.note_density!=null||controls.radar?.peak!=null)parts.push(`实际风格条件：整体密度 ${controls.radar?.note_density??'未记录'}；峰值密度 ${controls.radar?.peak??'未记录'}（均为归一化值）`);
  if(profileDiagnostics?.selected_component!=null)parts.push(`自动整曲风格选中写法：${profileDiagnostics.selected_component}`);
  if(report.timing_output_policy)parts.push(`选时结构：${report.timing_output_policy==='absolute_grid_v1'?'绝对拍位与节奏网格联合预测':'相对拍距预测'}`);
  if(report.event_family_policy)parts.push(`配置类型：${report.event_family_policy==='shared_continuation_v1'?'音乐、历史与持续状态共同决定续写':report.event_family_policy==='audio_state_phrase_v4'?'音频先验与持续状态短句':report.event_family_policy==='audio_prior_transition_v3'?'音乐决定类型底座，历史只决定是否延续':report.event_family_policy==='audio_semantic_v2'?'音乐与类型历史选择类型集合，完整历史组织几何':report.event_family_policy==='semantic_mass_v1'?'先学类型集合，类内再选形状与细节':'旧形状与字面分支'}`);
  const retryPolicy=report.action_retry_policy??settings.action_retry_policy;
  if(retryPolicy!==undefined)parts.push(`动作失败重试：${retryPolicy}`);
  const gridPolicy=report.pointer_grid_policy??settings.pointer_grid_policy;
  if(gridPolicy!==undefined)parts.push(`节奏网格：${({grid_family_coherent_v3:'段落节奏与32分进入校准',grid_family_coherent_v2:'按段落保持节奏（旧版）',grid_level_mass_v1:'保留各网格概率（不看段落）',off:'旧版区间内温度'})[gridPolicy]??gridPolicy}`);
  const repetitionGuard=report.repetition_guard??settings.repetition_guard;
  if(repetitionGuard!==undefined)parts.push(`重复抑制：${({official_tail_v2:'仅超出官谱长度',legacy_v1:'旧版固定抑制',off:'关闭'})[repetitionGuard]??repetitionGuard}`);
  const structure=report.structure_metrics;
  const gridObservation=structure?.absolute_grid_observation;
  if(gridObservation)parts.push(`常规16分网格对齐 ${(100*gridObservation.quarter_aligned_rate).toFixed(1)}%；最长恒定非零偏移 ${gridObservation.longest_same_nonquarter_phase_run} 音（描述性统计，细分和三连不一律判错）`);
  if(structure&&structure.events)parts.push(`结构指标：小节键位呼应 ${(100*structure.layout_echo_rate).toFixed(1)}%（官谱约15%）；三连音类 ${(100*structure.ternary_rate).toFixed(1)}%（官谱约4%），其中零散 ${structure.ternary_isolated_rate==null?'无':(100*structure.ternary_isolated_rate).toFixed(1)+'%'}（官谱约6%）；交互 ${(100*structure.pingpong_rate).toFixed(1)}%`);
  const historyStrength=report.pointer_history_strength??settings.pointer_history_strength;
  if (report.pointer_sampling_policy==='joint_t1_bin_mass_within_bin_temperature_v1') parts.push(Number(pointerTemperature)===0?'选时采用全体候选贪心':'选时保留模型预测的时间区间概率，温度只调整区间内部');
  if (historyStrength!==undefined) parts.push(`选时历史权重 ${historyStrength}（1 为完整历史，0 为音频分支）`);
  const actionHistory=report.action_history_strength??settings.action_history_strength;
  if (actionHistory!==undefined) parts.push(`动作历史权重 ${actionHistory}（1 为完整历史，0 为音频分支）`);
  const typeTemperature=report.action_type_temperature??settings.action_type_temperature;
  const typeMinP=report.action_type_min_p??settings.action_type_min_p;
  if(typeTemperature!==undefined || typeMinP!==undefined) parts.push(`动作类别温度 ${typeTemperature??'未记录'}；min-p ${typeMinP??'未记录'}（不直接改Touch环区比例；温度0仅保留最高类别，温度1沿用旧类别概率）`);
  const layout=report.touch_layout_policy??settings.touch_layout_policy;
  if(layout)parts.push(`Touch布局：${layout==='open'?'开放':'旧双头限制'}`);
  if(report.phrase_search||settings.search_policy){const search=report.phrase_search||{};parts.push(`短句生成：${search.policy??settings.search_policy}；候选 ${search.candidate_count??settings.phrase_candidates??1}；采音连续性 ${report.audio_tracking_strength??settings.audio_tracking_strength??0}；手感软评分 ${settings.softcomfort_strength??1}；当前重采 ${search.current_event_retries_used??0}；试写回退 ${search.private_tail_retries_used??0}；历史编辑 ${search.repairs_used??0}；撤回重采 ${search.rollbacks_used??0}`);}
  const motifPolicy=report.motif_memory_policy;
  const qualityPolicy=report.configuration_quality_policy;
  if(qualityPolicy) {
    parts.push(`配置质量架构：${qualityPolicy==='off'?'关闭（新增强度不生效）':qualityPolicy==='song_paired_v3'?'全曲音乐与历史配对（新版）':qualityPolicy==='cooperative_v2'?'稳定配置协同':'配置协同初版'}`);
    if(qualityPolicy!=='off') parts.push(`配置关系强度 ${report.quality_context_strength??settings.quality_context_strength??1}；短句计划影响 ${report.quality_plan_strength??settings.quality_plan_strength??1}；微前瞻强度 ${report.quality_lookahead_strength??settings.quality_lookahead_strength??0}`);
  }
  const motifStrength=report.motif_strength??settings.motif_strength;
  if(motifStrength!==undefined)parts.push(`动机记忆：${motifPolicy==='off'?'模型关闭，强度调整无效':motifPolicy??'待生成报告确认'}；请求强度 ${motifStrength}`);
  const touchRequested=report.touch_history_guard_requested??settings.touch_history_guard;
  if (touchRequested!==undefined) {
    const touchEffective=report.touch_history_guard;
    const bound=report.touch_history_log_odds_bound??report.touch_history_bound_requested??settings.touch_history_bound;
    parts.push(`Touch 历史保护：${touchEffective===true?'已生效':touchEffective===false?(touchRequested?'已请求，当前模型或动作温度不适用':'关闭'):(touchRequested?'已请求，待生成报告确认':'关闭')}；放大上限 ${bound??'未记录'}`);
  }
  if (condition) parts.push(`实际等级 ${condition.level}；Style：Tap ${condition.style?.tap_bias??"—"} / Slide ${condition.style?.slide_bias??"—"} / Hold ${condition.style?.hold_bias??"—"} / Touch ${condition.style?.touch_bias??"—"}`);
  if (settings.condition_source?.includes("file")) parts.push("条件文件中的明确字段已覆盖界面同名条件");
  if (report.weights_sha256) parts.push(`权重 ${report.weights_sha256.slice(0,12)}；更新 ${report.optimizer_steps??"未记录"} 步`);
  if (report.events_sha256) parts.push(`音符序列 ${report.events_sha256.slice(0,12)}`);
  const curve=report.learned_workload_curve,guard=report.guard||{},playability=report.playability||{};
  if (curve && Array.isArray(curve.times_sec) && curve.times_sec.length) {
    parts.push(`已预测学习负荷曲线：${curve.times_sec.length} 个时点；引导强度 ${report.curve_strength??settings.curve_strength??1}；完整曲线见 .composer.json`);
  } else if (report.schema || task.status==='completed') {
    parts.push('未提供学习负荷曲线；请核对检查点是否具备曲线能力');
  }
  if (guard.checks!==undefined || guard.rejections!==undefined || guard.skipped_candidates!==undefined) {
    parts.push(`配置守卫检查 ${guard.checks??0} 次；拒绝 ${guard.rejections??0} 次；跳过候选 ${guard.skipped_candidates??0} 个`);
  }
  if (report.min_same_head_ms!==undefined) parts.push(`同键最短间隔 ${report.min_same_head_ms} ms；过密拒绝 ${guard.rejection_reasons?.repeated_head_too_soon??0} 次`);
  if (report.intercept_tap_on_slide!==undefined || settings.intercept_tap_on_slide!==undefined) parts.push(`TapOnSlide 拦截：${(report.intercept_tap_on_slide??settings.intercept_tap_on_slide)?"开启":"关闭（仅报告）"}`);
  const density=report.density_observation?.exported;
  if (density?.min_same_head_interval_ms!=null) parts.push(`谱面最短同键间隔 ${Number(density.min_same_head_interval_ms).toFixed(1)} ms`);
  if (Number.isFinite(Number(playability.remaining_hard)) && playability.remaining_hard!==null) {
    parts.push(`MiaCode 未解决硬冲突 ${Number(playability.remaining_hard)} 项（可处理 ${playability.remaining_actionable_hard??'未记录'} 项）；仅报告的硬冲突仍计入总数`);
  }
  if (report.stop_reason) parts.push(`结束原因：${report.stop_reason}`);
  parts.push("每次任务独立保存；条件为软输入，接线生效不保证模型已学会服从");
  return parts.join("。 ");
}

function renderTaskResults(task) {
  const container = element("#taskResults");
  if (!container) return;
  container.replaceChildren();
  const composerSummary=composerGenerationSummary(task);
  if (composerSummary) {
      const notice=document.createElement("p");notice.className="inline-status"+(task.result?.composer_v2_report?.accepted_with_conflicts_count>0?" warn":"");
    notice.textContent=composerSummary;container.appendChild(notice);
  }
    if(task.kind==='composer_v2_generate' && task.result?.composer_v2_report?.conflict_report) {
      const link=document.createElement('a');
      link.href='/api/download?path='+encodeURIComponent(task.result.composer_v2_report.conflict_report);
      link.textContent='下载未解决冲突报告';container.appendChild(link);
    }
  if (task.kind === "form_generate" && task.status === "completed" && task.artifacts_missing) {
    const notice = document.createElement("p");
    notice.className = "inline-status warn";
    notice.textContent = "生成任务已完成，但服务端未返回谱面文件或下载信息。请检查任务日志和输出目录。";
    container.appendChild(notice);
  }
  const repair = task.kind === "form_generate" ? task.result?.repair_summary : null;
  if (repair && repair.remaining_actionable_hard !== undefined && repair.remaining_actionable_hard !== null) {
    const remaining = Number(repair.remaining_actionable_hard);
    if (Number.isFinite(remaining)) {
      const notice = document.createElement("p");
      notice.className = "inline-status warn";
      const repairNotes = [repair.budget_exhausted === true ? "修订预算已耗尽。" : "",
        repair.coverage_status === "partial" ? "部分语法或路径未覆盖。" : ""].filter(Boolean).join("");
      notice.textContent = `MiaCode 检查剩余可处理硬冲突：${remaining} 项。${repairNotes}有限修订结果不构成可玩性认证，请查看生成报告并复核。`;
      container.appendChild(notice);
    }
  }
  renderTaskOnsetDiagnostics(task);
  if (task.kind === "chart_stream_generate" && task.result?.stream_report) {
    const report = task.result.stream_report;
    const timing = report.timing_diagnostics || {};
    const notice = document.createElement("p");
    notice.className = "inline-status";
    notice.textContent = `${report.generation_status === "incomplete" ? "生成未完成" : "生成已结束"}：${report.status?.reason || "未知"}；${report.event_count ?? 0} 个起音；最长起音空白 ${Number(timing.max_gap_sec || 0).toFixed(2)} 秒；最大导出分音 {${timing.max_export_division ?? "未知"}}。空白可能是合理休止，具体位置与候选概率见 stream.json。`;
    if (report.reference_playability) notice.textContent += ` MiaCode 参考运行时硬冲突 ${report.reference_playability.runtime_hard_count ?? "未知"} 项；不构成可玩认证。`;
    if (report.checkpoint_training) notice.textContent += ` 本次权重：保存 epoch ${report.checkpoint_training.epoch ?? "未知"}，宽度 ${report.model_config?.d_model ?? "未知"}。`;
    container.appendChild(notice);
  }
  if (task.kind === "form_generate") {
    let load = task.result?.local_load_summary;
    if (!load) for (const line of [...(task.logs || [])].reverse()) {
      const start = line.indexOf("form_local_load ");
      if (start < 0) continue;
      try { load = JSON.parse(line.slice(start + "form_local_load ".length)); break; } catch (_) {}
    }
    if (load) {
      const notice = document.createElement("p");
      notice.className = "inline-status";
      const heads = Number(load.peak_all_head_nps), events = Number(load.peak_occupied_events_nps);
      const bursts = Array.isArray(load.burst_intervals) ? load.burst_intervals : [];
      notice.textContent = `局部负荷：一秒峰值 ${Number.isFinite(heads) ? heads : "未知"} 个音符头 / ${Number.isFinite(events) ? events : "未知"} 个起音（两项分别统计）。按键多押集中段 ${load.burst_interval_count ?? bursts.length} 处；详细时间见生成报告，统计不代表这些段落一定不合理。`;
      container.appendChild(notice);
    }
  }
  const files = task.result?.files || [];
  const urls = task.result?.download_urls || [];
  const paths = task.result?.paths || [];
  files.forEach((name, index) => {
    const link = document.createElement("a");
    link.textContent = name;
    link.href = urls[index] || "#";
    link.title = paths[index] || "";
    if (!urls[index]) link.classList.add("disabled");
    const row = document.createElement("div");
    row.appendChild(link);
    container.appendChild(row);
  });
  if (globalThis.ChartPreview?.supportsTask(task)) {
    const previewButton = document.createElement("button");
    previewButton.type = "button";
    previewButton.className = "secondary-button";
    previewButton.textContent = "预览这次生成";
    previewButton.addEventListener("click", () => globalThis.ChartPreview.openTask(task));
    container.appendChild(previewButton);
    void globalThis.ChartPreview.offerTask(task);
  }
}

function renderGenerationResult(task) {
  const status = element("#qualityStatus");
  const diagnostics = element("#qualityDiagnostics");
  const reports = task.result?.quality_reports || [];
  const report = reports.find((item) => item?.diagnostics) || reports[0];
  const payload = report?.diagnostics;
  if (status) {
    status.hidden = false;
    if (!report) {
      status.textContent = "本次生成没有质量报告。";
    } else if (payload) {
      const warnings = payload.note_warning_count ?? 0;
      const verdict = payload.generation_status || (payload.syntax_valid ? "success" : "failed");
      const playability = payload.playability;
      let playabilityText = "";
      if (playability && typeof playability === "object") {
        const statusLabel = playability.status === "disabled" ? "未启用"
          : playability.status === "partial" ? "部分静态检测"
            : playability.status === "engine_reference" ? (playability.coverage?.implementation === "python" ? "MiaCode Python 检测" : "MiaCode 检测")
            : playability.status === "static_only" ? "静态参考检测"
              : String(playability.status || "未知");
        const hardCount = Array.isArray(playability.remaining_hard) ? playability.remaining_hard.length
          : Number(playability.remaining_hard || 0);
        const actionableHard = Number(playability.remaining_actionable_hard ?? hardCount);
        const repairs = Array.isArray(playability.repairs) ? playability.repairs : [];
        const repairCount = repairs.length || Number(playability.repairs || 0);
        const deleteRepairs = repairs.filter((item) => !item?.operation || item.operation === "delete").length;
        const editRepairs = repairs.filter((item) => item?.operation && item.operation !== "delete").length;
        const details = Array.isArray(playability.remaining_details) ? playability.remaining_details
          : Array.isArray(playability.details) ? playability.details : [];
        const softCount = details.filter((item) => item?.severity === "soft").length;
        const attempts = (Array.isArray(playability.windows) ? playability.windows : [])
          .flatMap((window) => Array.isArray(window?.attempts) ? window.attempts : []);
        const retryCount = attempts.length ? Math.max(0, attempts.length - attempts.filter((item) => Number(item?.attempt || 0) === 0).length) : 0;
        const coverage = playability.coverage || {};
        const unsupported = Array.isArray(coverage.unsupported) ? coverage.unsupported.length
          : Number(coverage.unsupported || 0);
        const parseErrors = Array.isArray(coverage.parse_errors) ? coverage.parse_errors.length
          : Number(coverage.parse_errors || 0);
        const crossWindow = Array.isArray(playability.cross_window) ? playability.cross_window : [];
        const acceptedCross = crossWindow.filter((item) => item?.status === "accepted");
        const acceptedRewinds = acceptedCross.filter((item) => item?.operation === "regenerate_source_window").length;
        const acceptedSustain = acceptedCross.filter((item) => ["shorten_hold", "slide_to_head_tap"].includes(item?.operation)).length;
        const backtrackChanges = acceptedRewinds || Number(playability.backtrack_modified ?? playability.backtrack_modifications ?? 0);
        const sustainChanges = acceptedSustain || Number(playability.sustain_modified ?? playability.sustain_repairs ?? 0);
        const deletionCount = deleteRepairs || (repairCount && !editRepairs ? repairCount : 0);
        playabilityText = `冲突参考：${statusLabel}；物理 Muri ${hardCount}（可处理 ${actionableHard}）；软参考 ${softCount}；删除修复 ${deletionCount}；微调修复 ${editRepairs}；窗口重试 ${retryCount}；已接受前置回溯 ${backtrackChanges}；已接受前置时值修复 ${sustainChanges}；未覆盖项 ${unsupported}；解析问题 ${parseErrors}。`;
        if (softCount) playabilityText += "Warning 不直接删除，需要结合谱面与游玩复核。";
        if (["static_only", "partial"].includes(playability.status)) {
          playabilityText += "检测范围不完整，不能据此认定谱面可玩。";
        }
      }
      const curveWarnings = task.result?.curve_condition_warnings || payload.intensity_curve?.condition_warnings || [];
      status.textContent = `生成状态：${verdict}；音符级警告 ${warnings} 条。${playabilityText} ${curveWarnings.join("；")} ${task.result?.quality_notice || ""}`;
    } else {
      status.textContent = `质量报告不可读：${report.error || "未知原因"}`;
    }
  }
  if (diagnostics) {
    diagnostics.hidden = !payload;
    if (payload) diagnostics.textContent = JSON.stringify(payload, null, 2);
  }
  // Attach the best candidate's path so the acceptance panel can reuse it.
  const acceptanceMaidata = element("#acceptanceMaidata");
  if (acceptanceMaidata && !acceptanceMaidata.value.trim()) {
    const best = (task.result?.paths || []).find((path) => String(path).endsWith("maidata.txt"));
    if (best) acceptanceMaidata.value = best;
  }
  const acceptanceAudio = element("#acceptanceAudio");
  if (acceptanceAudio && !acceptanceAudio.value.trim()) {
    acceptanceAudio.value = valueOrDefault("#audioPath", "").trim();
  }
  const acceptance = acceptanceFromQuality(task);
  if (acceptance) {
    renderAcceptance(acceptance);
  } else if (payload) {
    setAcceptanceStatus("生成完成。填好 gold_stats.json 后点「验收谱面」可得横向比较结果。", "warn");
  }
}

function acceptanceFromQuality(task) {
  const reports = (task.result?.quality_reports || []).map((item) => item?.diagnostics).filter(Boolean);
  for (const report of reports) {
    if (report?.acceptance) return report.acceptance;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------
async function upload(kind, file, targetSelector, statusSelector) {
  if (!file) return null;
  const status = element(statusSelector);
  if (status) status.textContent = "正在上传...";
  const form = new FormData();
  form.append("kind", kind);
  form.append("file", file, file.name);
  const response = await fetch("/api/upload", { method: "POST", body: form });
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { error: text }; }
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  const target = element(targetSelector);
  if (target) target.value = data.path;
  if (status) status.textContent = `已上传：${data.name}`;
  saveSettings();
  return data;
}

async function waitForExtraction(uploadId, data) {
  const started = Date.now();
  const timeoutMs = 30 * 60 * 1000;
  const status = element("#datasetUploadStatus");
  while (Date.now() - started < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    let state;
    try { state = await api("/api/status"); } catch (_) { continue; }
    const info = state.extractions?.[uploadId];
    if (info?.status === "failed") throw new Error(`解压失败：${info.error || "未知错误"}`);
    if (info?.status === "done" || (state.datasets || []).some((item) => item.upload_id === uploadId)) return;
    const waited = Math.round((Date.now() - started) / 1000);
    if (status) status.textContent = `已接收 ${data.name}，正在后台解压...（已等待 ${waited} 秒）`;
  }
  throw new Error("解压等待超时（30 分钟）。可直接粘贴本机数据集文件夹路径。");
}

function populateDatasets(datasets) {
  uploadedDatasets = Array.isArray(datasets) ? datasets : [];
  datasetWorkspaceState.items = uploadedDatasets.map((item) => ({ ...item,
    path: item.path || item.raw_root || item.root || "",
    manifest: item.manifest || item.manifest_path || "",
    manifest_hint: item.manifest || item.manifest_path || "",
    name: item.name || item.dataset_name || item.path || "未命名数据集",
  })).filter((item) => item.path || item.manifest);
  if (!datasetWorkspaceState.activePath || !datasetWorkspaceState.items.some((item) => item.path === datasetWorkspaceState.activePath || item.manifest === datasetWorkspaceState.activePath)) {
    const current = valueOrDefault("#manifestPath", "").trim() || valueOrDefault("#rawRoot", "").trim();
    datasetWorkspaceState.activePath = datasetWorkspaceState.items.find((item) => item.manifest === current || item.path === current)?.path || datasetWorkspaceState.items[0]?.path || "";
  }
  const select = element("#datasetSelect");
  if (!select) return;
  const current = datasetPaths().raw_root;
  select.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = uploadedDatasets.length ? "选择已上传/已放置的训练集" : "尚未发现训练集";
  select.appendChild(placeholder);
  uploadedDatasets.forEach((data) => {
    const option = document.createElement("option");
    option.value = data.path;
    option.textContent = `${data.name}${data.prepared ? "（已预处理）" : "（待预处理）"}`;
    select.appendChild(option);
  });
  select.value = uploadedDatasets.some((data) => data.path === current) ? current : "";
  renderDatasetWorkspace();
  if (typeof refreshAutoQualityAuditStatus === "function") void refreshAutoQualityAuditStatus().catch(() => {});
}

function datasetKey(item) { return String(item?.manifest || item?.path || ""); }

function datasetCardHtml(item) {
  const key = datasetKey(item);
  const checked = datasetWorkspaceState.selected.has(key);
  const active = item.path === datasetWorkspaceState.activePath || item.manifest === datasetWorkspaceState.activePath;
  const charts = item.chart_count ?? item.charts ?? item.difficulty_count ?? "?";
  const preprocessAction = item.prepared ? "" : `<button class="small-button dataset-card-preprocess" type="button" data-dataset-path="${escapeHtml(item.path || "") }">去训练页预处理</button>`;
  const errorMessage = item.error ? `<p class="inline-status warn">这个来源读取失败：${escapeHtml(String(item.error))}。请检查路径或文件后刷新。</p>` : "";
  return `<article class="dataset-card${active ? " active" : ""}" data-dataset-path="${escapeHtml(item.path || "")}" data-dataset-manifest="${escapeHtml(item.manifest || "")}">
    <label class="dataset-card-check"><input class="dataset-select-checkbox" type="checkbox" data-dataset-key="${escapeHtml(key)}" ${checked ? "checked" : ""}> <strong>用于下次训练</strong></label>
    <button class="dataset-card-open" type="button" data-dataset-path="${escapeHtml(item.path || "")}"><strong>${escapeHtml(item.name || "未命名数据集")}</strong></button>
    <div class="dataset-card-meta">${item.prepared ? "已预处理" : "待预处理"} · 谱面 ${escapeHtml(String(charts))}${item.reviewed_regions!=null?` · 已确认 ${escapeHtml(String(item.reviewed_regions))} 段`:''}</div>
    <div class="dataset-card-path">${escapeHtml(item.path || item.manifest || "")}</div>${errorMessage}${preprocessAction}${globalThis.CandidatePool?.card(item)||""}
  </article>`;
}

function renderDatasetWorkspace() {
  const cards = element("#datasetCards");
  if (!cards) return;
  cards.innerHTML = datasetWorkspaceState.items.map(datasetCardHtml).join("") || `<div class="muted dataset-empty">${datasetWorkspaceState.catalogLoading ? "正在读取训练素材，请稍候…" : datasetWorkspaceState.catalogError ? "素材列表读取失败，请点击“刷新数据集”重试。" : "尚未发现可用数据集。请先上传 ZIP，或在训练页填写原始数据集目录并预处理。"}</div>`;
  const summary = element("#datasetSelectionSummary");
  if (summary) summary.textContent = datasetWorkspaceState.selected.size ? `已选择 ${datasetWorkspaceState.selected.size} 个数据集` : "尚未选择数据集";
  const trainingSources = element("#trainingDatasetSelection");
  if (trainingSources) trainingSources.textContent = selectedDatasetItems().length
    ? `本次训练来源：${selectedDatasetItems().map((item) => item.name).join(" + ")}。检查和开始训练都会使用这些勾选来源，浏览谱面不会改变选择。`
    : datasetWorkspaceState.selectionTouched ? "尚未勾选训练数据集，请到“训练素材”页选择。" : "单数据集模式：使用下方当前训练清单。";
  const preparedManifest = element("#effectiveTrainingManifest");
  if (preparedManifest) preparedManifest.textContent = datasetWorkspaceState.prepared?.manifest
    ? `已准备的训练清单（只读）：${datasetWorkspaceState.prepared.manifest}`
    : "训练清单将在检查所选来源或开始训练时自动准备；当前预处理输出保持独立。";
  const status = element("#datasetWorkspaceStatus");
  if (status) status.textContent = datasetWorkspaceState.catalogLoading ? "正在刷新素材列表；已有卡片仍可浏览。" : datasetWorkspaceState.catalogError ? `素材列表读取失败：${datasetWorkspaceState.catalogError}。已有卡片已保留，请点击“刷新数据集”重试。` : `共发现 ${datasetWorkspaceState.items.length} 个数据集；点击卡片查看目录，勾选后用于训练。`;
  if (status && !datasetWorkspaceState.catalogLoading && !datasetWorkspaceState.catalogError && datasetWorkspaceState.catalogWarnings?.length) status.textContent += ` 部分来源读取失败：${datasetWorkspaceState.catalogWarnings.map(warning => typeof warning === "string" ? warning : `${warning.manifest || warning.path || "来源"}：${warning.error || warning.message || "读取失败"}`).join("；")}。其他来源仍可使用。`;
  const cardsPanel = element("#datasetCardsPanel");
  const catalogPanel = element("#datasetCatalogPanel");
  if (cardsPanel) cardsPanel.hidden = datasetWorkspaceState.view !== "cards";
  if (catalogPanel) catalogPanel.hidden = datasetWorkspaceState.view !== "catalog";
  if (typeof updateZeroUnmarkedTrainingNotice === "function") updateZeroUnmarkedTrainingNotice();
  globalThis.CandidatePool?.render();
  updateDatasetButtons();
}

function isZeroUnmarkedDataset(item) {
  return Boolean(item && (item.candidate_pool || item.reviewed_only || item.reviewed_regions != null));
}

function updateZeroUnmarkedTrainingNotice() {
  const warning = element("#v2TrainZeroUnmarkedWarning");
  if (!warning) return;
  const zeroItems = selectedDatasetItems().filter(isZeroUnmarkedDataset);
  const disabled = !checkedOrDefault("#v2TrainQualityEnabled", true);
  warning.hidden = !(currentModelFamily() === "chart_composer_v2" && zeroItems.length > 0 && disabled);
  if (!warning.hidden) {
    warning.textContent = `所选的 ${zeroItems.map((item) => item.name || item.path).join("、")} 只学习已确认内容。请开启「使用已保存的质量标注和训练示范」后再开始训练。`;
  }
}

function activeDatasetItem() {
  if (datasetWorkspaceState.view === 'catalog' && datasetWorkspaceState.auditCatalogItem) return datasetWorkspaceState.auditCatalogItem;
  return datasetWorkspaceState.items.find((item) => item.path === datasetWorkspaceState.activePath || item.manifest === datasetWorkspaceState.activePath) || null;
}

let datasetCatalogFlight = null;
function loadDatasetCatalog() {
  if (datasetCatalogFlight) return datasetCatalogFlight;
  datasetCatalogFlight = fetchDatasetCatalog().finally(() => { datasetCatalogFlight = null; });
  return datasetCatalogFlight;
}
async function fetchDatasetCatalog() {
  const request = ++datasetWorkspaceState.catalogRequest;
  datasetWorkspaceState.catalogLoading = true;
  datasetWorkspaceState.catalogError = "";
  renderDatasetWorkspace();
  try {
    const data = await api("/api/datasets/catalog", {
      current_manifest: valueOrDefault("#manifestPath", "").trim() || null,
      manifests: [...datasetWorkspaceState.selected].filter((path) => String(path).toLowerCase().endsWith(".json")),
    });
    if (request !== datasetWorkspaceState.catalogRequest) return;
    const items = Array.isArray(data.datasets) ? data.datasets : (Array.isArray(data.items) ? data.items : null);
    if (!items) throw new Error("后台未返回素材列表");
    datasetWorkspaceState.catalogWarnings = Array.isArray(data.warnings) ? data.warnings : [];
    if (items) {
      datasetWorkspaceState.items = items.map((item) => ({ ...item, path: item.path || item.raw_root || item.root || "", manifest: item.manifest || item.manifest_path || "", manifest_hint: item.manifest || item.manifest_path || "", name: item.name || item.dataset_name || item.path || "未命名数据集" })).filter((item) => item.path || item.manifest);
      const known = new Set(datasetWorkspaceState.items.map(datasetKey));
      datasetWorkspaceState.selected = new Set([...datasetWorkspaceState.selected].filter((key) => known.has(key)));
      if (!activeDatasetItem()) { datasetWorkspaceState.activePath = ""; datasetWorkspaceState.view = "cards"; }
      renderDatasetWorkspace();
    }
  } catch (error) {
    if (request === datasetWorkspaceState.catalogRequest) datasetWorkspaceState.catalogError = error.message || String(error);
  } finally {
    if (request === datasetWorkspaceState.catalogRequest) {
      datasetWorkspaceState.catalogLoading = false;
      renderDatasetWorkspace();
    }
  }
}

async function registerCurrentDataset() {
  const path = valueOrDefault("#rawRoot", "").trim();
  if (!path) throw new Error("请先填写原始训练集目录。");
  const manifest = valueOrDefault("#manifestPath", "").trim() || null;
  try {
    const data = await api("/api/datasets/register", { path, raw_root: path, manifest });
    const item = data.dataset || data.item || data;
    if (item && (item.path || item.raw_root || item.manifest)) {
      const normalized = { ...item, path: item.path || item.raw_root || path, manifest: item.manifest || item.manifest_path || manifest || "", manifest_hint: item.manifest || item.manifest_path || manifest || "", name: item.name || item.dataset_name || path };
      const key = datasetKey(normalized);
      datasetWorkspaceState.items = [...datasetWorkspaceState.items.filter((candidate) => datasetKey(candidate) !== key), normalized];
      datasetWorkspaceState.selected.add(key);
      datasetWorkspaceState.selectionTouched = true;
      datasetWorkspaceState.activePath = normalized.path;
      selectDataset(normalized);
      renderDatasetWorkspace(); saveDatasetWorkspace();
      setStatus("已登记当前训练集；如尚未生成 manifest，请继续点击预处理。", true);
      return;
    }
  } catch (error) {
    throw new Error(`数据集登记失败：${error.message || error}。请重启 WebUI 服务后重试。`);
  }
}

function selectedDatasetItems() {
  return datasetWorkspaceState.items.filter((item) => datasetWorkspaceState.selected.has(datasetKey(item)));
}

async function addReviewedTrainingSource(result) {
  // Preserve implicit single-dataset selection before explicitly adding one.
  const existing=selectedDatasetItems().length?null:datasetPaths().manifest;
  const register=async(manifest,name,reviewGroup)=>{
    const response=await api('/api/datasets/register',{path:manifest,name,review_group:reviewGroup});
    const item=response.dataset;
    if(!item?.manifest)throw new Error('素材登记没有返回训练清单，请刷新后重试。');
    const key=datasetKey(item);
    if(reviewGroup)for(const row of datasetWorkspaceState.items)if(row.review_group===reviewGroup)datasetWorkspaceState.selected.delete(datasetKey(row));
    datasetWorkspaceState.items=[...datasetWorkspaceState.items.filter(row=>datasetKey(row)!==key && !(reviewGroup && row.review_group===reviewGroup)),item];
    datasetWorkspaceState.selected.add(key);
    return item;
  };
  if(existing && existing!==result.manifest)await register(existing);
  await register(result.manifest,result.name,result.review_group);
  if(!result.regions)datasetWorkspaceState.selected.delete(datasetKey(datasetWorkspaceState.items.find(item=>item.manifest===result.manifest)));
  datasetWorkspaceState.selectionTouched=true;
  if(element('#v2TrainQualityEnabled'))element('#v2TrainQualityEnabled').checked=true;
  datasetWorkspaceState.prepared=null;renderDatasetWorkspace();saveDatasetWorkspace();saveSettings();
  return {sources:selectedDatasetItems().length};
}

async function composeSelectedDatasets() {
  if (datasetWorkspaceState.applying) return;
  datasetWorkspaceState.selectionTouched = true;
  datasetWorkspaceState.applying = true;
  const buttons = [element("#datasetComposeButton"), element("#trainingDatasetApplyButton")].filter(Boolean);
  buttons.forEach((button) => { button.disabled = true; button.textContent = "正在应用…"; });
  setManifestStatus("正在检查勾选来源并准备训练清单…");
  const compositionSummary = element("#datasetCompositionSummary");
  if (compositionSummary) {
    compositionSummary.hidden = false;
    compositionSummary.className = "inline-status";
    compositionSummary.textContent = "正在检查已勾选来源并生成训练清单…";
  }
  try {
    assertTrainingSelection();
    const items = selectedDatasetItems();
    if (!items.length) throw new Error("请至少勾选一个训练数据集。");
    const manifests = items.map((item) => item.manifest).filter(Boolean);
    const paths = items.map((item) => item.path).filter(Boolean);
    const family = currentModelFamily();
    let data;
    try {
      data = await api("/api/datasets/compose", { manifests, dataset_paths: paths, datasets: items, model_family: family });
    } catch (error) {
      if (!String(error.message || "").match(/404|Not Found|不存在|未找到/i)) throw error;
      throw new Error("当前 WebUI 后端尚未提供多数据集合并接口，请重启/更新服务后再应用多选数据集。");
    }
    if (family !== currentModelFamily() || JSON.stringify(manifests) !== JSON.stringify(selectedTrainingManifests())) {
      throw new Error("应用过程中训练来源或模型族已变化，请按当前选择重新应用。");
    }
    const manifest = data.manifest || data.effective_manifest || data.composed_manifest || data.path;
    if (!manifest) throw new Error("后端没有返回合并后的 manifest 路径。");
    datasetWorkspaceState.prepared = { ...data, manifest };
    invalidateDatasetStatus(`已应用 ${items.length} 个数据集，请检查所选训练来源。`);
    renderDatasetWorkspace();
    saveDatasetWorkspace();
    const summary = data.summary || {};
    const message = `已完成：应用 ${items.length} 个来源；${summary.song_count ?? "?"} 首歌曲，${summary.difficulty_count ?? "?"} 张谱面；训练 manifest：${manifest}`
      + `；按原文件去重 ${summary.duplicate_source_charts ?? 0} 张；汇总质量标注 ${summary.quality_records ?? 0} 条。`
      + (summary.stale_quality_annotations ? ` ${summary.stale_quality_annotations} 条标注因原谱或音频变化已失效，请回原数据集重新检查并保存。` : "");
    if (compositionSummary) {
      compositionSummary.hidden = false;
      compositionSummary.className = "inline-status good";
      compositionSummary.textContent = message;
    }
    setManifestStatus(`已应用 ${items.length} 个来源；清单已准备。可检查可用谱面数量或开始训练。`, "good");
    setStatus(`已应用 ${items.length} 个训练数据集。`, true);
  } catch (error) {
    setManifestStatus(`应用失败：${error?.message || error}`, "warn");
    if (compositionSummary) {
      compositionSummary.hidden = false;
      compositionSummary.className = "inline-status warn";
      compositionSummary.textContent = `应用失败：${error?.message || error}`;
    }
    throw error;
  } finally {
    datasetWorkspaceState.applying = false;
    buttons.forEach((button) => { button.textContent = "应用勾选的数据集"; });
    updateDatasetButtons();
  }
}

async function openDataset(item, { autoScope = null, autoFilter = '' } = {}) {
  if (!item) return;
  if (item.prepared === false || !item.manifest) throw new Error("这个数据集尚未预处理，请点击卡片上的“去训练页预处理”。");
  datasetWorkspaceState.auditCatalogItem = item.auditReadOnly ? item : null;
  if (!item.auditReadOnly) datasetWorkspaceState.activePath = item.path || item.manifest || "";
  qualityState.autoScope = autoScope;
  qualityState.autoFilter = autoFilter;
  if (element('#qualityAutoFilter')) element('#qualityAutoFilter').value = autoFilter;
  globalThis.CandidatePool?.activate(item);
  globalThis.ChartPreview?.setTrainingDataset(item.manifest);
  datasetWorkspaceState.view = "catalog";
  renderDatasetWorkspace();
  saveDatasetWorkspace();
  qualityState.catalogRequest += 1;
  qualityState.detailRequest += 1;
  qualityState.previewLoading=false;
  qualityState.previewError='';
  qualityState.items = []; qualityState.page = 1; qualityState.pages = 1; qualityState.total = 0;
  qualityState.folderTotal = 0; qualityState.selectionRequest += 1; qualityState.selectionLoading = false; qualityState.saving = false; qualityState.editedWeights.clear();
  qualityState.selected.clear(); qualityState.detail = null; qualityState.regions = []; qualityState.correctedPath = "";
  qualityState.annotationPath = ""; qualityState.annotationManifest = ""; qualityState.summary = null;
  renderQualityCatalog();
  if (element("#qualityDetail")) element("#qualityDetail").hidden = true;
  if (element("#qualityDetailAudio")) { element("#qualityDetailAudio").removeAttribute("src"); element("#qualityDetailAudio").load(); }
  if (element("#qualityCorrectedPath")) element("#qualityCorrectedPath").textContent = "";
  if (item.prepared === false || !item.manifest) throw new Error("这个数据集还没有预处理 manifest，请先点击卡片上的“去训练页预处理”。");
  if (element("#datasetCatalogTitle")) element("#datasetCatalogTitle").textContent = `${item.name || "数据集"} · 谱面目录${item.auditReadOnly ? '（查看结果，请在原库保存标注）' : ''}`;
  renderDatasetWorkspace();
  await loadQualityCatalog(true);
}

function showDatasetCards() {
  globalThis.CandidatePool?.activate(null);
  datasetWorkspaceState.view = "cards";
  datasetWorkspaceState.auditCatalogItem = null;
  qualityState.catalogRequest += 1;
  qualityState.detailRequest += 1;
  qualityState.previewLoading=false;
  renderDatasetWorkspace();
  void refreshAutoQualityAuditStatus("quality");
}

async function openDatasetWorkspace() {
  setWorkspaceMode("quality");
  showDatasetCards();
  await loadDatasetCatalog();
}

function selectDataset(data) {
  if (element("#datasetZip")) element("#datasetZip").value = "";
  if (element("#rawRoot")) element("#rawRoot").value = data.path;
  if (element("#cacheDir")) element("#cacheDir").value = data.cache_dir;
  if (element("#manifestPath")) element("#manifestPath").value = data.manifest;
  const select = element("#datasetSelect");
  if (select) select.value = data.path;
  qualityState.items = []; qualityState.page = 1; qualityState.total = 0;
  qualityState.folderTotal = 0; qualityState.selectionRequest += 1; qualityState.selectionLoading = false; qualityState.saving = false; qualityState.editedWeights.clear();
  qualityState.annotationPath = ""; qualityState.annotationManifest = "";
  qualityState.summary = null; qualityState.selected.clear(); qualityState.detail = null;
  qualityState.regions = []; qualityState.correctedPath = "";
  renderQualityCatalog();
  if (element("#qualityCorrectedPath")) element("#qualityCorrectedPath").textContent = "";
  invalidateDatasetStatus(selectedTrainingManifests().length
    ? "当前预处理目录已切换；本次训练仍使用‘训练素材’的勾选来源。"
    : "当前单集目录已切换，请先预处理或检查训练来源。");
  renderDatasetWorkspace();
}

function invalidateDatasetStatus(message) {
  datasetVersion += 1;
  manifestRequest += 1;
  invalidateCheckpointInspection("数据路径或 manifest 已变化，请重新检查 checkpoint。");
  setManifestStatus(message, "warn");
  saveSettings();
}

function invalidateCheckpointInspection(message) {
  checkpointInspection = { key: "", data: null };
  if (message) setCheckpointStatus(message, "warn");
}

function currentCheckpointKey() {
  return JSON.stringify([
    currentModelFamily(),
    currentTrainingMode(),
    valueOrDefault("#trainingCheckpointPath", "").trim(),
    valueOrDefault("#manifestPath", "").trim(),
    valueOrDefault("#validationManifestPath", "").trim(),
    selectedTrainingManifests().slice().sort(),
    datasetVersion,
  ]);
}

// ---------------------------------------------------------------------------
// Status bootstrap
// ---------------------------------------------------------------------------
let datasetWorkspaceBootstrapped = false;
async function loadStatus() {
  const state = await api("/api/status");
  if (!datasetWorkspaceBootstrapped) {
    datasetWorkspaceBootstrapped = true;
    if (datasetWorkspaceState.view === "catalog" && !qualityState.annotationManifest) datasetWorkspaceState.view = "cards";
  }
  globalThis.maichartLastStatus = state;
  globalThis.CandidatePool?.capabilities(state);
  intensityCurveState.projectRoot = String(state.project_root || "");
  invalidateLearnedCurveForCheckpoint();
  const engineStatus = element("#playabilityEngineStatus");
  if (engineStatus) {
    const available = state.playability_engine?.available === true;
    engineStatus.className = `inline-status ${available ? "good" : "warn"}`;
    engineStatus.textContent = available
      ? (state.playability_engine?.implementation === "python"
        ? "MiaCode Python 检测已就绪，路径数据随项目提供，无需 DLL 或 Qt。"
        : "当前后端仍使用旧版 MiaCode 运行时；重启 WebUI 服务后启用 Python 检测。")
      : state.playability_engine?.error || "Python 检测模块或路径数据不可用，请检查项目文件并重启 WebUI 服务。";
  }
  setStatus(`本地服务正常 · workspace: ${state.workspace}`, true);
  if (!datasetWorkspaceState.items.length) populateDatasets(state.datasets || []);
  void loadDatasetCatalog();
  const running = (state.tasks || []).find((task) => ["queued", "running", "cancelling"].includes(task.status)
    && !String(task.kind || "").startsWith("intensity"));
  window.MaichartWorkflow?.renderTasks(state.tasks || []);
  window.MaichartWorkflow?.refresh();
  const composerTraining = (state.tasks || []).filter(task => ["composer_train","composer_v2_train"].includes(task.kind)
    && ["queued", "running", "cancelling"].includes(task.status));
  for (const task of composerTraining) trackComposerTrainingTask(task);
  const selectedTaskId = activeTaskId;
  for (const task of composerTraining) watchTask(task.task_id);
  activeTaskId = selectedTaskId;
  if (running && !activeTaskId) {
    renderTask(running);
    watchTask(running.task_id);
  }
}

// ---------------------------------------------------------------------------
// Preprocess / manifest inspection
// ---------------------------------------------------------------------------
async function startPreprocess() {
  const payload = preprocessPayload();
  if (!payload.raw_root) throw new Error("请先填写或选择原始训练集目录");
  preprocessStarting = true;
  updateDatasetButtons();
  setManifestStatus("正在预处理训练集...");
  try {
    const data = await api("/api/preprocess", payload);
    preprocessTasks.set(payload.manifest, data.task_id);
    watchTask(data.task_id, async (task) => {
      preprocessStarting = false;
      preprocessTasks.delete(payload.manifest);
      updateDatasetButtons();
      if (task.status === "completed") {
        setManifestStatus("预处理完成，正在检查训练集状态...", "good");
        await loadStatus();
        await inspectManifest();
      } else {
        setManifestStatus(`预处理${task.status === "cancelled" ? "已中断" : "失败"}：${task.error || ""}`, "warn");
      }
    });
  } catch (error) {
    preprocessStarting = false;
    updateDatasetButtons();
    throw error;
  }
}

function formatProcessingError(item, index) {
  if (typeof item === "string") return `${index + 1}. ${item}`;
  if (!item || typeof item !== "object") return `${index + 1}. ${String(item)}`;
  const location = [item.song_id, item.difficulty_index !== undefined ? `D${item.difficulty_index}` : "",
    item.path || item.file || item.source].filter(Boolean).join("/");
  const reason = item.error || item.message || item.exception || item.reason || JSON.stringify(item);
  return `${index + 1}. ${location ? `${location}: ` : ""}${reason}`;
}

function renderProcessingErrors(errors) {
  const box = element("#processingErrors");
  if (!box) return;
  if (!Array.isArray(errors) || !errors.length) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  box.hidden = false;
  box.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = `发现 ${errors.length} 条处理/筛选问题（最多显示前 50 条）`;
  box.appendChild(title);
  const list = document.createElement("ol");
  errors.slice(0, 50).forEach((item, index) => {
    const entry = document.createElement("li");
    entry.textContent = formatProcessingError(item, index);
    list.appendChild(entry);
  });
  box.appendChild(list);
}

async function inspectManifest() {
  assertTrainingSelection();
  await assertDatasetBackend();
  const sourcePaths = trainingSourcePaths();
  const manifest = sourcePaths.manifest;
  if (!manifest && !selectedTrainingManifests().length) throw new Error("请先选择训练数据集");
  const request = ++manifestRequest;
  setManifestStatus("正在检查训练集...");
  const data = await api("/api/inspect-manifest", {
    ...sourcePaths, manifests: selectedTrainingManifests(), dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean), model_family: currentModelFamily(),
  });
  if (request !== manifestRequest) return data;
  if (data.ready === false) {
    setManifestStatus(data.message || "训练集尚未准备好", "warn");
    renderProcessingErrors([]);
    return data;
  }
  datasetWorkspaceState.prepared = data.dataset_selection || { manifest: data.manifest };
  renderDatasetWorkspace();
  const reasons = data.top_filter_reasons || [];
  const reasonText = reasons.length
    ? reasons.map(([reason, count]) => `${reason}=${count}`).join(", ")
    : "无";
  const lines = [
    `歌曲 ${data.song_count} 首 / 难度 ${data.difficulty_count} 个`,
    `${data.event_manifest ? "event QC " : ""}清单可用难度 ${data.usable_difficulty_count} 个，过滤 ${data.filtered_difficulty_count} 个`,
    `过滤原因：${reasonText}`,
    `音频状态：${Object.entries(data.audio_statuses || {}).map(([key, count]) => `${key}=${count}`).join(", ") || "无"}`,
  ];
  if(['chart_composer_v2','chart_composer_v1','song_form_v1','song_form_v2','chart_stream_v1'].includes(currentModelFamily()))lines.push('这是清单 QC 统计；实际训练还会重新解析、应用训练过滤和质量标注，并按验证设置划分');
  setManifestStatus(lines.join("｜"), data.usable_difficulty_count > 0 ? "good" : "warn");
  renderProcessingErrors(data.processing_errors || []);
  const validation = element("#validationManifestPath");
  if (validation && !validation.value && data.preprocessing?.target_schema) {
    // Nothing to auto-fill, but keep the field in sync with reality for clarity.
  }
  return data;
}

// ---------------------------------------------------------------------------
// Checkpoint inspection
// ---------------------------------------------------------------------------
function applyCheckpointModelConfig(data) {
  const model = data?.model_config || {};
  const contract = data?.data_contract || {};
  const familyParams = contract.family_params || {};
  const mapping = {
    "#dModel": model.d_model,
    "#nhead": model.nhead,
    "#numLayers": model.num_layers,
    "#dimFeedforward": model.dim_feedforward,
    "#dropout": model.dropout,
    "#maxLen": model.max_len,
    "#trainMaxTokens": model.max_tokens ?? contract.max_tokens,
    "#segmentBeats": contract.segment_beats,
    "#maxAudioTokens": contract.max_audio_tokens,
  };
  Object.entries(mapping).forEach(([selector, value]) => {
    const target = element(selector);
    if (target && value !== undefined && value !== null) target.value = String(value);
  });
  const conditionParams = contract.condition_params || {};
  if (conditionParams.default !== undefined && element("#conditionDefault")) {
    element("#conditionDefault").value = String(conditionParams.default);
  }
  if (conditionParams.mask_prob !== undefined && element("#conditionMaskProb")) {
    element("#conditionMaskProb").value = String(conditionParams.mask_prob);
  }
  const difficulty = contract.difficulty_filter || {};
  const difficultyMapping = {
    "#minDifficultyIndex": difficulty.min_index,
    "#maxDifficultyIndex": difficulty.max_index,
    "#minLevel": difficulty.min_level,
    "#maxLevel": difficulty.max_level,
  };
  Object.entries(difficultyMapping).forEach(([selector, value]) => {
    const target = element(selector);
    if (target && value !== undefined && value !== null) target.value = String(value);
  });
  const familyMapping = {
    "#contextBeats": familyParams.context_beats,
    "#audioMarginBeats": familyParams.audio_margin_beats,
    "#contextTokens": currentTrainingMode() === "finetune"
      && element("#contextTokens")?.dataset.touched === "1"
      ? undefined : familyParams.context_tokens,
    "#overlapBeats": familyParams.overlap_beats,
    "#contextDropout": familyParams.context_dropout,
  };
  Object.entries(familyMapping).forEach(([selector, value]) => {
    const target = element(selector);
    if (target && value !== undefined && value !== null) target.value = String(value);
  });
  if (conditionParams.dropout !== undefined && element("#conditionDropout")) {
    element("#conditionDropout").value = String(conditionParams.dropout);
  }
  if (currentTrainingMode() === "resume") {
    const onsetParams = contract.onset_params || {};
    if (element("#localPoolingPolicy")) {
      element("#localPoolingPolicy").value = model.local_pooling_policy || "legacy_uniform";
    }
    if (element("#actionStateProfile")) element("#actionStateProfile").value = model.action_state_profile || "none";
    const onsetTraining = data.onset_training || {};
    const onsetProfile = onsetParams.profile || model.onset_profile || "none";
    const onsetWeight = onsetParams.loss_weight ?? onsetTraining.loss_weight ?? 0.2;
    if (element("#onsetProfile")) element("#onsetProfile").value = onsetProfile;
    if (element("#onsetLossWeight")) element("#onsetLossWeight").value = String(onsetWeight);
  }
  // A normal resume must reproduce the checkpoint's sampling contract. Older
  // checkpoints only recorded the combined slide/touch multiplier; interpret
  // that value as the legacy Touch multiplier. Finetune intentionally keeps
  // the user's new values so sampling can be changed while reusing weights.
  if (currentTrainingMode() === "resume") {
    const slide = familyParams.slide_oversample;
    const touch = familyParams.touch_oversample ?? slide;
    if (slide !== undefined && element("#slideOversample")) {
      element("#slideOversample").value = String(slide);
    }
    if (touch !== undefined && element("#touchOversample")) {
      element("#touchOversample").value = String(touch);
    }
    const fine = familyParams.fine_rhythm_oversample ?? 1;
    if (element("#fineRhythmOversample")) {
      element("#fineRhythmOversample").value = String(fine);
    }
    const threshold = familyParams.fine_rhythm_ratio_threshold ?? 0.25;
    if (element("#fineRhythmRatioThreshold")) {
      element("#fineRhythmRatioThreshold").value = String(threshold);
    }
  }
  const profile = contract.feature_profile;
  if ((profile === "basic_v1" || profile === "rhythm_v2" || profile === "event_audio_v2_rhythm") && element("#featureProfile")) {
    element("#featureProfile").value = profile;
  }
  saveSettings();
}

function renderCheckpointInspection(data) {
  const valid = data.valid === true;
  const mode = currentTrainingMode();
  const capability = mode === "resume"
    ? `可断点继续：${data.can_resume ? "是" : "否"}`
    : `可加载权重：${data.can_init ? "是" : "否"}`;
  const state = valid ? `checkpoint 可用；${capability}` : "checkpoint 不可用于当前模式";
  const caps = data.generation_capabilities || {};
  const capabilityLine = Object.keys(caps).length
    ? `生成能力：内嵌无理检测 ${caps.conflict_detection ? "可用" : "不可用"}，细碎音符调制 ${caps.fine_note_strength ? "可用" : "不可用"}，局部采音对齐 ${caps.local_onset_guidance === true ? "可用" : "不可用"}`
    : "";
  const details = [
    capabilityLine,
    `状态：${state}`,
    `模型族：${data.model_family_label || data.model_family || "未知"}`,
    `schema：${data.schema || "未记录"}`,
    data.epoch !== undefined ? `保存 epoch：${data.epoch}` : "保存 epoch：未记录",
    data.model_config?.dropout !== undefined ? `模型内部 Dropout（dropout）：${data.model_config.dropout}` : "",
    data.data_contract?.condition_params?.dropout !== undefined ? `条件随机丢弃（condition_dropout）：${data.data_contract.condition_params.dropout}；当前控件：${valueOrDefault("#conditionDropout", "未显示")}` : "",
    data.data_contract?.family_params?.context_dropout !== undefined ? `上下文随机丢弃（context_dropout）：${data.data_contract.family_params.context_dropout}` : "",
    `音频池化：${data.model_config?.local_pooling_policy || "legacy_uniform"}；生成按 checkpoint 继承，迁移请用加强训练`,
    `上下文预算策略：${data.data_contract?.input_policies?.context_budget_policy || "legacy_dynamic_v1"}`,
    `谱面过滤策略：${data.data_contract?.input_policies?.chart_filter_policy || "legacy_strict_v1"}`,
    `持续动作摘要：${data.model_config?.action_state_profile || "none"}；有效训练：${data.action_state_training?.trained === true ? "是" : "否"}`,
    data.data_contract?.onset_params?.profile !== undefined
      ? `采音辅助合同：${data.data_contract.onset_params.profile}，loss weight=${data.data_contract.onset_params.loss_weight ?? "未记录"}` : "",
    data.onset_training?.trained !== undefined
      ? `采音辅助训练：${data.onset_training.trained ? "已训练" : "未训练"}` : "",
    `模型结构：${JSON.stringify(data.model_config || {})}`,
    `数据合同：${JSON.stringify(data.data_contract || {})}`,
    data.model_config?.input_dim !== undefined ? `事件输入维度：${data.model_config.input_dim}` : "",
    data.model_config?.plan_dim !== undefined ? `Planner 输出维度：${data.model_config.plan_dim}（按所选 profile 解释）` : "",
    data.data_contract?.condition_params?.null_policy ? `Null 条件策略：${data.data_contract.condition_params.null_policy}` : "",
    data.warnings?.length ? `警告：\n${formatList(data.warnings)}` : "",
    data.errors?.length ? `错误：\n${formatList(data.errors)}` : "",
  ].filter(Boolean).join("\n");
  const box = element("#checkpointDetails");
  if (box) {
    box.hidden = false;
    box.textContent = details;
  }
  const conditionHint = valid && data.data_contract?.condition_params?.dropout !== undefined
    ? `；条件随机丢弃（condition_dropout）=${data.data_contract.condition_params.dropout}` : "";
  setCheckpointStatus(state + conditionHint + (data.warnings?.length ? `；警告 ${data.warnings.length} 条` : ""), valid ? "good" : "warn");
}

function applyChartComposerCheckpointConfig(data, mode, preserveEditable = false) {
  const config = data?.model_config || {};
  if(config.architecture_profile==='event_set_stream_v1'||data?.schema==='maichart-composer-event-set-checkpoint-v1') {
    const args=data.args||{};
    const fields={composerTrainingWindowPolicy:'training_window_policy',composerMaxPhraseEvents:'max_phrase_events',composerPhraseSeamEvents:'phrase_seam_events',composerPhraseExecutionPolicy:'phrase_execution_policy',composerPhraseModes:'phrase_modes',composerPhraseRelationWeight:'phrase_relation_weight',composerDetailRepresentationPolicy:'detail_representation_policy',composerDurationFeaturePolicy:'duration_feature_policy',composerSeed:'seed',composerMaxSteps:'max_steps',composerLimitCharts:'limit_charts',composerRecoveryInterval:'recovery_interval',composerRecoverySamples:'recovery_samples',composerRecoverySteps:'recovery_steps',composerRecoveryWeight:'recovery_weight',learningRate:'lr',composerGenerationEvalSeeds:'generation_eval_seeds',composerValidationManifest:'validation_manifest',composerArchitectureProfile:'architecture_profile',composerDModel:'d_model',composerDimFeedforward:'dim_feedforward',composerLayers:'layers',composerNhead:'nhead',composerHistoryEvents:'history_events',composerRolloutSteps:'rollout_steps',composerWindowSeconds:'window_seconds',composerAudioGridSeconds:'audio_grid_seconds',composerPlanWeight:'plan_weight',composerPlanBlankWindowSeconds:'plan_blank_window_seconds',composerLrSchedule:'lr_schedule',composerWarmupSteps:'warmup_steps',composerMinLrRatio:'min_lr_ratio',composerValidationMaxBatches:'validation_max_batches',composerValidationBatchSize:'validation_batch_size',composerGenerationEvalIntervalEpochs:'generation_eval_interval_epochs',composerGenerationEvalSongs:'generation_eval_songs',composerPrefetchBatches:'prefetch_batches',composerAudioCacheGb:'audio_cache_gb',composerDropout:'dropout',composerSymbolicPretrainEpochs:'symbolic_pretrain_epochs',batchSize:'batch_size',gradAccumulation:'grad_accumulation'};
    for(const [id,key] of Object.entries(fields)){if(id==='composerValidationManifest'&&mode!=='resume')continue;if(mode==='finetune'&&['batchSize','gradAccumulation','learningRate','composerAudioCacheGb'].includes(id))continue;const node=element('#'+id),value=config[key]??args[key];if(id==='composerValidationManifest'&&node&&mode==='resume'&&value!==undefined&&node.dataset.validationInherited!=='1'){node.dataset.beforeResumeValidation=node.value;node.dataset.validationInherited='1';}const retainReset=mode==='resume' && preserveEditable && ((element('#composerResetLearningRate')?.checked && ['learningRate','composerLrSchedule','composerWarmupSteps','composerMinLrRatio'].includes(id)) || (element('#composerRebatchResume')?.checked && ['batchSize','gradAccumulation'].includes(id)));if(node&&value!==undefined&&!retainReset){node.value=String(value);node.disabled=mode==='resume';}}
    for(const [id,key,fallback] of [['composerSongProfilePolicy','song_profile_policy','off'],['composerSongResponsePolicy','song_response_policy','off'],['composerSongProfileComponents','song_profile_components',4],['composerSongProfileWeight','song_profile_weight',.1],['composerSongResponseModes','song_response_modes',4],['composerSongResponseGroupSize','song_response_group_size',4]]){const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(config[key]??fallback):(key.endsWith('_policy')?'auto':fallback));}}
    {const node=element('#composerBatchOrderPolicy');if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(args.batch_order_policy??'song_local_v1'):'auto');}}
    for(const [id,key,fallback] of [['composerTrainingWindowPolicy','training_window_policy','block_v1'],['composerPhraseExecutionPolicy','phrase_execution_policy','off'],['composerDetailRepresentationPolicy','detail_representation_policy','private_v1'],['composerDurationFeaturePolicy','duration_feature_policy','legacy_v1']]){const node=element('#'+id);if(node && (mode==='resume'||!preserveEditable))node.value=String(mode==='resume'?(config[key]??args[key]??fallback):'auto');}
    for(const [id,key,fallback] of [['composerDraftLossWeight','draft_loss_weight',.1],['composerDraftResponsibilityTemperature','draft_responsibility_temperature',1]]){const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(config[key]??args[key]??fallback):fallback);}}
    for(const [id,key,fallback] of [['composerDraftObjectivePolicy','draft_objective_policy','balanced_mean_v1'],['composerConfigurationRecoverySteps','configuration_recovery_steps',8],['composerRecoveryBridgeSteps','recovery_bridge_steps',8]]){const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(config[key]??args[key]??fallback):(key==='draft_objective_policy'?'auto':fallback));}}
    for(const [id,key,fallback] of [['composerDurationObjectivePolicy','duration_objective_policy','categorical_v1'],['composerDurationCalibrationWeight','duration_calibration_weight',.1]]){const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(config[key]??args[key]??fallback):(key==='duration_objective_policy'?'auto':fallback));}}
    for(const [id,key,fallback] of [['composerCurvePreviewPolicy','curve_preview_policy','off'],['composerCurvePreviewWeight','curve_preview_weight',.05]]){const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(config[key]??args[key]??fallback):(key.endsWith('_policy')?'auto':fallback));}}
    const deploymentFraction=element('#composerDeploymentConditionFraction');if(deploymentFraction){deploymentFraction.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)deploymentFraction.value=String(mode==='resume'?(config.deployment_condition_fraction??args.deployment_condition_fraction??1):.25);}
    const policy=element('#composerLearningPolicy'),weight=element('#composerTransitionExitWeight');
    if(policy){policy.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)policy.value=mode==='resume'?'checkpoint':'auto';policy.dataset.savedPolicies=JSON.stringify({transition_policy:config.transition_policy??'off',timing_group_policy:config.timing_group_policy??'phase_only_v1',literal_output_policy:config.literal_output_policy??'flat_family_v1',decoder_policy:config.decoder_policy??'postnorm_v1',window_balance_policy:config.window_balance_policy??'equal_window_v1',history_window_policy:config.history_window_policy??'rolling_legacy_v1',continuation_policy:config.continuation_policy??'off',sensor_load_policy:config.sensor_load_policy??'button_only_v1',gap_continuation_policy:config.gap_continuation_policy??'additive_v1',motion_continuation_policy:config.motion_continuation_policy??'off',lane_load_policy:config.lane_load_policy??'off',recovery_policy:config.recovery_policy??'persistent_v1',timing_output_policy:config.timing_output_policy??'relative_delta_v1',event_family_policy:config.event_family_policy??'legacy_v1',recovery_target_policy:config.recovery_target_policy??'authored_next_v1',prefix_augmentation_input_policy:config.prefix_augmentation_input_policy??'causal_window_config_time_first_decision_v3',timing_grid_audio_policy:config.timing_grid_audio_policy??'off',phrase_context_policy:config.phrase_context_policy??'off',duration_audio_policy:config.duration_audio_policy??'off',physical_policy:config.physical_policy??'legacy_report_v1'});}
    if(weight){weight.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)weight.value=String(mode==='resume'?(config.transition_exit_weight??0):0);}
    if(policy?.dataset.savedPolicies){const saved=JSON.parse(policy.dataset.savedPolicies);saved.song_choice_objective_policy=config.song_choice_objective_policy??'profile_only_v1';saved.draft_objective_policy=config.draft_objective_policy??'balanced_mean_v1';saved.literal_set_policy=config.literal_set_policy??'canonical_v1';saved.draft_policy=config.draft_policy??'off';saved.plan_kind_policy=config.plan_kind_policy??'off';saved.template_policy=config.template_policy??'cross_family_v1';saved.candidate_state_policy=config.candidate_state_policy??'cursor_v1';policy.dataset.savedPolicies=JSON.stringify(saved);}
    for(const [id,key,fallback] of [['composerPhraseContextPolicy','phrase_context_policy','off'],['composerDurationAudioPolicy','duration_audio_policy','off'],['composerPhraseLossWeight','phrase_loss_weight',.05]]){
      const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(mode==='resume'?(config[key]??args[key]??fallback):(key==='phrase_loss_weight' ? .05 : 'auto'));}
    }
    for(const [id,value] of [['composerPrefixAugmentationRate',mode==='resume'?(config.prefix_augmentation_rate??0):.15],['composerGenerationEvalTemperature',mode==='resume'?(args.generation_eval_temperature??.65):.65],['composerEventFamilyAudioWeight',mode==='resume'?(config.event_family_audio_weight??0):.25]]){const node=element('#'+id);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=String(value);}}
    for(const key of Object.keys(EVENT_SET_LEARNING_POLICIES)){const node=element('#eventSetPolicy_'+key);if(node){node.disabled=mode==='resume';if(mode==='resume'||!preserveEditable)node.value=mode==='resume'?(config[key]??(key==='recovery_target_policy'?'authored_next_v1':key==='literal_set_policy'?'canonical_v1':key==='song_choice_objective_policy'?'profile_only_v1':'auto')):'auto';}}
    for(const node of document.querySelectorAll('[data-generation-eval-key]')){const key='generation_eval_'+node.dataset.generationEvalKey;node.disabled=mode==='resume';if(args[key]!==undefined)node.value=String(args[key]);}
    for(const id of ['composerArchitectureProfile','composerDModel','composerLayers','composerNhead','composerDimFeedforward','composerHistoryEvents','composerRolloutSteps','composerAudioGridSeconds','composerPlanBlankWindowSeconds','composerDropout']){const node=element('#'+id);if(node){node.disabled=true;node.title='继承检查点；改变此项需要从头训练';}}
    updateEventSetControls();return;
  }
  const mapping = {
    "#composerDModel": config.d_model,
    "#composerDimFeedforward": config.dim_feedforward,
    "#composerLayers": config.layers,
    "#composerNhead": config.nhead,
    "#composerHistoryEvents": config.history_events,
    "#composerCandidateCount": config.candidate_count,
    "#composerActionSlots": config.action_slots,
    "#composerDropout": config.dropout,
    "#composerHistoryAnchorPolicy": mode==='resume'?(config.history_anchor_policy??'legacy_digit_v1'):undefined,
    "#composerArchitectureProfile": mode==='resume'?config.architecture_profile:undefined,
    "#composerLensWidth": mode==='resume'?config.lens_width:undefined,
    "#composerHistoryResidualBound": mode==='resume'?config.history_residual_bound:undefined,
  };
  Object.entries(mapping).forEach(([selector, value]) => {
    const target = element(selector);
    if (target && value !== undefined && value !== null) target.value = String(value);
    if (target) target.disabled = ['#composerArchitectureProfile','#composerLensWidth','#composerHistoryResidualBound','#composerHistoryAnchorPolicy','#composerHistoryMemoryRmsBound'].includes(selector)
      ? mode==='resume' : mode !== "fresh";
  });
  const historyMemoryBound=element('#composerHistoryMemoryRmsBound');
  for(const [id,key,fallback] of [['composerLrSchedule','lr_schedule','constant'],
    ['composerWarmupSteps','warmup_steps',0],['composerMinLrRatio','min_lr_ratio',1]]) {
    const target=element('#'+id);if(!target)continue;
    target.disabled=mode==='resume';
    if(mode==='resume')target.value=String(data?.args?.[key]??fallback);
    else if(!preserveEditable)target.value=id==='composerLrSchedule'?'auto':'';
  }
  if(historyMemoryBound) {
    if(mode!=='fresh' && (!preserveEditable || mode==='resume'))
      historyMemoryBound.value=String(config.history_memory_rms_bound??8);
    historyMemoryBound.disabled=mode==='resume';
  }
  for (const [id,key,fallback] of [
    ['composerActionObjectivePolicy','action_objective_policy','event_balanced_v1'],
    ['composerActionStatePolicy','action_state_policy','committed_legacy_v1'],
    ['composerMotifMemoryPolicy','motif_memory_policy','off'],
    ['composerSlideFactorPolicy','slide_factor_policy','marginal_v1'],
    ['composerMotifMaxEvents','motif_max_events',16],
    ['composerStructurePolicy','structure_policy','off'],
    ['composerStructureBackend','structure_backend','native'],
    ['composerStructureDropout','structure_dropout',.15],
    ['composerContactPolicy','contact_policy','off'],
    ['composerRecoveryTouchLayoutPolicy','touch_layout_policy','legacy_two_heads'],
    ['composerHistoryMemoryPolicy','history_memory_policy','unbounded_v1'],
    ['composerHistoryAttentionPolicy','history_attention_policy','auto_v1'],
    ['composerHeadTimingPolicy','head_timing_policy','merged_pads_v1'],
    ['composerCrossEventRelationPolicy','cross_event_relation_policy','off'],
    ['composerFutureHorizonPolicy','future_horizon_policy','off'],
    ['composerEchoPriorPolicy','echo_prior_policy','off'],
    ['composerTimingDistributionPolicy','timing_distribution_policy','hierarchical_v1'],
    ['composerButtonLoadPolicy','button_load_policy','off'],
    ['composerFineTimingPolicy','fine_timing_policy','off'],
    ['composerDensityFeedbackPolicy','density_feedback_policy','off'],
    ['composerActionHeadPolicy','action_head_policy','flat_v1'],
    ['composerConfigurationQualityPolicy','configuration_quality_policy','off'],
    ['composerSampleSequencePolicy','sample_sequence_policy','independent_events_v1'],
    ['composerConditionMaskProb','condition_mask_prob',0],
    ['composerRecoverySamplingPolicy','recovery_sampling_policy',data?.args?.recovery_sampling_policy??data?.action_sampling_policy??'calibrated_kind_touch_rings_v2']]) {
    const target=element('#'+id);
    const newStagePolicy=['head_timing_policy','cross_event_relation_policy','future_horizon_policy','echo_prior_policy','timing_distribution_policy','button_load_policy','action_head_policy','configuration_quality_policy','sample_sequence_policy','recovery_sampling_policy'].includes(key);
    if(target && mode!=='fresh' && (mode==='resume' || (!preserveEditable && !newStagePolicy)))
      target.value=String(config[key]??data?.args?.[key]??fallback);
    if(target)target.disabled=mode==='resume';
  }
  if (mode!=='fresh' && (!preserveEditable || mode==='resume')) {
    const args=data?.args||{};
    const pointer=element('#composerRecoveryPointerTemperature');
    const action=element('#composerRecoveryActionTemperature');
    if (pointer) pointer.value=String(args.recovery_pointer_temperature??1);
    if (action) action.value=String(args.recovery_action_temperature??.85);
  }
  const precision = element("#composerPrecision");
  const cpuThreads = element('#composerCpuThreads');
  if (cpuThreads) {
    cpuThreads.disabled = false;
    if (!preserveEditable) cpuThreads.value = String(mode === 'resume' ? (data?.args?.cpu_threads ?? 0) : 2);
  }
  const sampleWorkers = element('#composerSampleWorkers');
  if (sampleWorkers && mode === 'resume' && !preserveEditable)
    sampleWorkers.value = String(data?.args?.sample_workers ?? 4);
  if (precision && !preserveEditable && mode !== "fresh" && data?.args?.precision) precision.value = String(data.args.precision);
  for (const id of ['composerLensChunkSize','composerHistoryDropout','composerLedgerDropout',
    'composerPlacementWeight','composerPlannerWeight','composerRecoveryInterval',
    'composerRecoverySamples','composerRecoverySteps','composerRecoveryWeight',
    'composerSampleLocality','composerSymbolicDropout','composerAudioPointerWeight',
    'composerLrSchedule','composerWarmupSteps','composerMinLrRatio']) {
    const target=element('#'+id);if(target) target.disabled=mode==='resume';
  }
  if (mode==='resume') {
    const args=data?.args||{};
    for (const [id,key] of [['composerLensChunkSize','lens_chunk_size'],['composerHistoryDropout','history_dropout'],
      ['composerLedgerDropout','ledger_dropout'],['composerPlacementWeight','placement_weight'],['composerPlannerWeight','planner_weight'],
      ['composerRecoveryInterval','recovery_interval'],['composerRecoverySamples','recovery_samples'],
      ['composerRecoverySteps','recovery_steps'],['composerRecoveryWeight','recovery_weight'],
      ['composerRecoveryPointerTemperature','recovery_pointer_temperature'],['composerRecoveryActionTemperature','recovery_action_temperature']]) {
      const target=element('#'+id),value=config[key]??args[key];
      if (target && value!==undefined) target.value=String(value);
      if (target) target.disabled=true;
    }
    for (const [id,key] of [['composerSampleLocality','sample_locality'],
      ['composerSymbolicDropout','symbolic_dropout'],['composerAudioPointerWeight','audio_pointer_weight']]) {
      const target=element('#'+id),value=config[key]??args[key];
      if (target && value!==undefined) target.value=String(value);
      if (target) target.disabled=true;
    }
    for (const [id,key,fallback] of [['batchSize','batch_size'],['gradAccumulation','grad_accumulation',1],
      ['learningRate','lr'],['composerSeed','seed',0],['composerMaxSteps','max_steps',0],
      ['composerGenerationEvalSongs','generation_eval_songs',2],
      ['composerGenerationEvalIntervalEpochs','generation_eval_interval_epochs',1],
      ['composerValidationManifest','validation_manifest',''],['composerHoldoutFraction','holdout_fraction',.05],
      ['composerValidationMaxBatches','validation_max_batches',32],['composerValidationBatchSize','validation_batch_size',8],
      ['composerGenerationEvalSeeds','generation_eval_seeds','0,1'],['composerArchiveCheckpoints','archive_checkpoints','auto']]) {
      const scheduled=args.lr_schedule==='cosine'||Number(args.warmup_steps||0)>0;
      const target=element('#'+id),value=(key==='lr'?(scheduled?args.lr:data.optimizer_lr):undefined)??args[key]??fallback;
      if (['batch_size','grad_accumulation'].includes(key)) {
        if (element('#composerRebatchResume')?.checked) continue;
      } else if (preserveEditable) continue;
      if (target && value!==undefined) target.value=String(value);
    }
  }
}

function renderChartComposerCheckpointInspection(data) {
  const valid = data?.valid === true;
  const config = data?.model_config || {};
  if(config.architecture_profile==='event_set_stream_v1'||data?.schema==='maichart-composer-event-set-checkpoint-v1'){
    const details=[
      `状态：${valid?'整拍集合 checkpoint 可用':'整拍集合 checkpoint 不可用'}；可续训：${data?.can_resume?'是':'否'}；可微调：${data?.can_init?'是':'否'}`,
      `schema：${data?.schema||'未记录'}；epoch：${data?.epoch??'未记录'}；优化器更新：${data?.optimizer_steps??'未记录'}`,
      `类型计划：${config.event_family_policy??'legacy_v1'}；模板保持：${config.template_policy??'cross_family_v1'}；候选持续状态：${config.candidate_state_policy??'cursor_v1'}`,
      `短句窗口：${config.training_window_policy??data?.args?.training_window_policy??'block_v1'}；最多事件：${config.max_phrase_events??data?.args?.max_phrase_events??64}；接缝：${config.phrase_seam_events??data?.args?.phrase_seam_events??2}；写法：${config.phrase_execution_policy??'off'} / ${config.phrase_modes??4}；关系权重：${config.phrase_relation_weight??data?.args?.phrase_relation_weight??0}；细节：${config.detail_representation_policy??'private_v1'}；时长：${config.duration_feature_policy??'legacy_v1'}`,
      `短句组织：${config.phrase_context_policy??'off'}；持续区间音频：${config.duration_audio_policy??'off'}`,
      `短段草稿：${config.draft_policy??'off'}；草稿学习权重：${config.draft_loss_weight??0}；责任温度：${config.draft_responsibility_temperature??1}；已知结构：96维／最多16锚点`,
      `数据内容身份：${data?.dataset_content_identity?.schema??'未记录'}；指纹：${data?.dataset_content_identity?.fingerprint??'未记录'}`,
      data?.training_migration||'',data?.error||''
    ].filter(Boolean).join('\n');
    const box=element('#checkpointDetails');if(box){box.hidden=false;box.textContent=details;}
    setCheckpointStatus(valid?(data?.training_migration||`整拍集合 checkpoint 可用；${currentTrainingMode()==='resume'?'续训沿用检查点合同':'可在新目录微调'}`):(data?.error||'整拍集合 checkpoint 不可用'),valid&&!data?.training_migration?'good':'warn');
    return;
  }
  const details = [
    `状态：${valid ? "chart_composer_v1 checkpoint 可用" : "chart_composer_v1 checkpoint 不可用"}`,
    "模型族：chart_composer_v1（事件候选与动作槽位）",
    `schema：${data?.schema || "未记录"}`,
    `架构：${config.architecture_profile || "未记录"}`,
    `配置学习目标：${data?.args?.action_objective_policy??'event_balanced_v1'}`,
    `动机输入策略：${config.motif_memory_policy==='off'?'关闭':data?.motif_input_policy??'旧版（续训须检查迁移说明）'}`,
    `历史几何：${config.history_anchor_policy??'legacy_digit_v1'}；持续状态：${config.action_state_policy??'committed_legacy_v1'}`,
    `历史记忆幅度策略：${config.history_memory_policy??'unbounded_v1'}；历史记忆RMS上限：${config.history_memory_rms_bound??8}`,
    `历史跨注意力后端：${config.history_attention_policy??'auto_v1'}`,
    `按键计时通道：${config.head_timing_policy??'merged_pads_v1'}；跨拍相对键位：${config.cross_event_relation_policy??'off'}`,
    `相位呼应先验：${config.echo_prior_policy??'off'}`,
    `短句窗口：${config.training_window_policy??data?.args?.training_window_policy??'block_v1'}；最多事件：${config.max_phrase_events??data?.args?.max_phrase_events??64}；接缝：${config.phrase_seam_events??data?.args?.phrase_seam_events??2}；写法：${config.phrase_execution_policy??'off'} / ${config.phrase_modes??4}；关系权重：${config.phrase_relation_weight??data?.args?.phrase_relation_weight??0}；细节：${config.detail_representation_policy??'private_v1'}；时长：${config.duration_feature_policy??'legacy_v1'}`,
    `短句组织训练：${config.phrase_context_policy??'off'}；持续区间音频训练：${config.duration_audio_policy??'off'}；短句辅助损失：${config.phrase_loss_weight??data?.args?.phrase_loss_weight??.05}`,
    `短段草稿：${config.draft_policy??'off'}；草稿学习权重：${config.draft_loss_weight??0}；责任温度：${config.draft_responsibility_temperature??1}；已知结构：96维／最多16锚点`,
    `选时分布：${config.timing_distribution_policy??'hierarchical_v1'}`,
    `逐按钮历史负担：${config.button_load_policy??'off'}`,
    `恢复采样策略：${config.recovery_sampling_policy??data?.args?.recovery_sampling_policy??data?.action_sampling_policy??'calibrated_kind_touch_rings_v2'}；选位 / 动作温度：${data?.args?.recovery_pointer_temperature??1} / ${data?.args?.recovery_action_temperature??.85}`,
    `动机记忆：${config.motif_memory_policy??'off'}；事件上限：${config.motif_max_events??16}；Slide因子：${config.slide_factor_policy??'marginal_v1'}`,
    `结构条件：${config.structure_policy??'off'}；后端：${config.structure_backend??'native'}；遮挡：${config.structure_dropout??.15}；接触预算：${config.contact_policy??'off'}`,
    `恢复Touch布局：${data?.args?.touch_layout_policy??'legacy_two_heads'}`,
    data?.epoch !== undefined ? `保存 epoch：${data.epoch}` : "保存 epoch：未记录",
    data?.optimizer_steps !== undefined ? `优化器更新：${data.optimizer_steps}` : "",
    `模型结构：d_model=${config.d_model ?? "未记录"}；layers=${config.layers ?? "未记录"}；nhead=${config.nhead ?? "未记录"}；历史=${config.history_events ?? "未记录"}；候选=${config.candidate_count ?? "未记录"}；动作槽位=${config.action_slots ?? "未记录"}`,
    data?.training_semantics ? `训练语义：${data.training_semantics}` : "",
    data?.error ? `错误：${data.error}` : "",
  ].filter(Boolean).join("\n");
  const box = element("#checkpointDetails");
  if (box) { box.hidden = false; box.textContent = details; }
  setCheckpointStatus(valid ? `chart_composer_v1 checkpoint 可用；epoch ${data.epoch ?? 0}；${currentTrainingMode() === "resume" ? "resume 将继承模型结构" : "可加强训练"}` : (data.error || "chart_composer_v1 checkpoint 不可用"), valid ? "good" : "warn");
}

function applyChartStreamCheckpointConfig(data, mode, preserveEditable = false) {
  const config = data?.model_config || {};
  const args = data?.args || {};
  STREAM_GROUNDING_FIELDS.forEach(([id,key,fallback])=>{
    const target=element('#'+id); if(!target) return;
    if (mode==='fresh') { if (!preserveEditable) target.value=String(fallback); } else if (!(preserveEditable && mode==='finetune')) target.value=String(config[key] ?? args[key] ?? (key==='grounding_profile'?'none':key==='continuation_profile'?'reset_v1':fallback));
    target.disabled=mode==='resume';
  });
  const mapping = {
    "#streamDModel": config.d_model,
    "#streamLayers": config.layers,
    "#streamNhead": config.nhead,
    "#streamDimFeedforward": config.dim_feedforward,
    "#streamMaxTokens": config.max_tokens,
    "#streamMaxHistoryTokens": config.max_history_tokens,
    "#streamMemoryTokens": config.max_stream_memory,
    "#streamDropout": config.dropout,
    "#streamSeed": args.seed,
    "#streamPlanLossWeight": args.plan_loss_weight,
    "#streamEventDecisionLossWeight": args.event_decision_loss_weight,
    "#streamPrecision": args.precision,
    "#streamAttentionBackend": args.attention_backend || "auto",
    "#streamHandBackend": args.hand_backend || "auto",
    "#streamExecutionMode": args.execution_mode || "auto",
    "#streamCudaGraphBudgetGb": args.cuda_graph_budget_gb ?? 2,
    "#streamWeightDecay": args.weight_decay,
    "#streamAudioCacheGb": args.audio_cache_gb,
    "#streamPrepareWorkers": args.prepare_workers,
    "#streamPrefetchBatches": args.prefetch_batches,
    "#streamDataWorkers": args.data_workers,
    "#streamCpuThreads": args.cpu_threads,
    "#streamInputCacheMode": args.input_cache_mode,
    "#streamStructureCacheMb": args.structure_cache_mb,
    "#streamOptimizerImplementation": args.optimizer_implementation,
    "#streamInputResidency": args.input_residency,
    "#streamResidentCacheGb": args.resident_cache_gb,
    "#streamResidentMemoryReserveGb": args.resident_memory_reserve_gb,
    "#streamCacheValidationRecords": args.cache_validation_records,
    "#streamMaxRegionSeconds": args.max_region_seconds,
    "#streamLimitCharts": args.limit_charts,
    "#streamLogInterval": args.log_interval,
    "#streamCheckpointIntervalSteps": args.checkpoint_interval_steps,
    "#streamCacheDir": args.cache_dir,
  };
  if (!(preserveEditable && mode==='finetune') && element("#qualityEnabled") && typeof args.quality_enabled === "boolean") {
    element("#qualityEnabled").checked=args.quality_enabled;
  }
  Object.entries(mapping).forEach(([selector, value]) => {
    const target = element(selector);
    if (!target || value === undefined || value === null) return;
    if (preserveEditable && mode==='finetune' && !['#streamDModel','#streamLayers','#streamNhead',
        '#streamDimFeedforward','#streamMaxTokens','#streamMaxHistoryTokens','#streamMemoryTokens','#streamDropout'].includes(selector)) return;
    target.value = String(value);
  });
  const locked = mode !== "fresh";
  ["#streamDModel", "#streamLayers", "#streamNhead", "#streamDimFeedforward",
    "#streamMaxTokens", "#streamMaxHistoryTokens", "#streamMemoryTokens", "#streamDropout"].forEach((selector) => {
    const target = element(selector);
    if (target) target.disabled = locked;
  });
}

function renderChartStreamCheckpointInspection(data) {
  const valid = data?.valid === true;
  const config = data?.model_config || {};
  const args = data?.args || {};
  const caps = data?.generation_capabilities || {};
  const state = valid ? "chart_stream_v1 checkpoint 可用" : "chart_stream_v1 checkpoint 不可用";
  const details = [
    `状态：${state}`,
    "模型族：chart_stream_v1（连续整曲事件流）",
    `schema：${data?.schema || "未记录"}`,
    data?.epoch !== undefined ? `保存 epoch：${data.epoch}` : "保存 epoch：未记录",
    `模型结构：d_model=${config.d_model ?? "未记录"}；layers=${config.layers ?? "未记录"}；nhead=${config.nhead ?? "未记录"}；FFN=${config.dim_feedforward ?? "未记录"}`,
    `连续上下文：${config.max_tokens ?? "未记录"} token；近期历史 ${config.max_history_tokens ?? "未记录"} token；记忆槽 ${config.max_stream_memory ?? "未记录"}`,
    `关联输入：${caps.grounding_profile ?? 'none'}；数值时钟=${caps.causal_numeric_clock?'启用':'未启用'}；局部音频与远期记忆=${caps.native_local_audio?'启用':'未启用'}；选音关联监督=${caps.position_alignment_trained?'已有训练记录':'尚无训练记录'}`,
    `连续上下文：${caps.continuation_profile === 'overlap_v1' ? '重叠事件续写' : '原上下文重置'}；重放预算=${caps.context_replay_tokens ?? 256} token（重叠方案生效）`,
    `训练参数：precision=${args.precision ?? "未记录"}；lr=${args.lr ?? "未记录"}；weight_decay=${args.weight_decay ?? "未记录"}；plan_loss_weight=${args.plan_loss_weight ?? "未记录"}；对称增广=${args.symmetry_augmentation ?? "none"}`,
    `生成能力：连续整曲=${caps.continuous_stream ? "可用" : "不可用"}；精确有理事件=${caps.rational_event_positions ? "可用" : "不可用"}；持续动作记忆=${caps.sustained_action_memory ? "可用" : "不可用"}`,
    "连续整曲、有理事件和持续动作能力按 checkpoint 合同检查；语法可用不等于音乐质量或可玩性合格。",
    "计算块只用于显存管理，不代表音乐边界；旧 event / song-form checkpoint 不能互载。",
    data?.warnings?.length ? `警告：\n${formatList(data.warnings)}` : "",
    data?.errors?.length ? `错误：\n${formatList(data.errors)}` : "",
  ].filter(Boolean).join("\n");
  const box = element("#checkpointDetails");
  if (box) { box.hidden = false; box.textContent = details; }
  setCheckpointStatus(valid
    ? (data.can_resume === false && currentTrainingMode() === "resume"
      ? "此 checkpoint 可生成、可 finetune；持续动作时钟监督已升级，请改选加强训练并使用新输出目录。"
      : `chart_stream_v1 checkpoint 可用；epoch ${data.epoch ?? 0}；模型宽度 ${config.d_model ?? "未知"}；优化器更新 ${data.optimizer_steps ?? "未记录"} 步`)
    : ((data.errors || []).join("；") || "chart_stream_v1 checkpoint 不可用"), valid ? "good" : "warn");
}

function applySongFormCheckpointConfig(data, mode) {
  const config = data?.model_config || {}, locked = mode !== "fresh";
  const structure = [["#formDModel", "d_model"], ["#formLayers", "layers"], ["#formNhead", "nhead"], ["#formMaxTokens", "max_tokens"], ["#formMaxMemoryTokens", "max_memory_tokens"], ["#formDropout", "dropout"], ["#formDimFeedforward", "dim_feedforward"]];
  structure.forEach(([selector, key]) => { const node = element(selector); if (!node) return; if (locked && config[key] != null) node.value = String(config[key]); node.disabled = locked; });
  const plannerWidth=element("#formPlannerWidth");
  if (plannerWidth) { if (locked && data?.blueprint_config?.width != null) plannerWidth.value=String(data.blueprint_config.width); plannerWidth.disabled=locked; }
  const trainingMeter = element("#formTrainingMeterMap");
  if (trainingMeter) {
    const storedMeters = data?.resume_settings?.meter_map ?? data?.dataset_contract?.meter_map;
    if (locked && data?.valid) trainingMeter.value = Array.isArray(storedMeters) && storedMeters.length
      ? JSON.stringify(storedMeters, null, 2) : "";
    trainingMeter.disabled = mode === "resume" && data?.valid === true;
  }
  const qualityProfile = element("#formQualityProfile");
  const savedQuality = config.quality_profile || "legacy";
  if (qualityProfile && locked && data?.valid) {
    if (mode === "resume" || qualityProfile.dataset?.evidenceManual !== "1") {
      qualityProfile.value = mode === "resume" ? savedQuality : "auto";
    }
    if (qualityProfile.dataset && mode === "resume") {
      qualityProfile.dataset.evidenceManual = "";
      qualityProfile.dataset.evidenceInheritedProfile = savedQuality;
      qualityProfile.dataset.evidenceInheritedValue = qualityProfile.value;
    }
  }
  for (const [id, value] of [
    ["formPositionAlignmentLossWeight", data?.resume_settings?.position_alignment_loss_weight ?? data?.args?.position_alignment_loss_weight ?? 0],
    ["formPositionAlignmentToleranceMs", data?.resume_settings?.position_alignment_tolerance_ms ?? data?.args?.position_alignment_tolerance_ms ?? 15],
    ["formPlannerContextSeconds", data?.blueprint_config?.context_seconds ?? config.planner_context_seconds ?? 0],
    ["formCurveBinSeconds", config.curve_bin_seconds ?? 2],
  ]) {
    const node = element(`#${id}`);
    if (node && mode === "resume" && data?.valid) {
      node.value = String(value);
      if (node.dataset) {
        node.dataset.evidenceManual = "";
        node.dataset.evidenceInheritedProfile = savedQuality;
        node.dataset.evidenceInheritedValue = node.value;
      }
    } else if (node && mode === "finetune" && data?.valid && node.dataset?.evidenceManual !== "1") {
      node.value = "";
      if (node.dataset) {
        node.dataset.evidenceInheritedProfile = savedQuality;
        node.dataset.evidenceInheritedValue = "";
      }
    }
  }
  const geometryProfile = element("#formRhythmGeometryProfileSelect");
  if (geometryProfile && locked) {
    const stored = data?.model_config?.rhythm_geometry_profile;
    // Resume inherits the saved contract exactly. Finetune keeps the selector
    // editable so a v1 checkpoint can migrate forward to v2.
    if (mode === "resume") geometryProfile.value = stored === "exact_fraction_v1" ? "exact_fraction_v1"
      : stored === "exact_fraction_v2" ? "exact_fraction_v2" : "none";
    else if (stored) geometryProfile.value = stored;
  }
  const inherited = [["#learningRate", "lr"], ["#batchSize", "batch_size"], ["#gradAccumulation", "grad_accumulation"],
    ["#trainingSeed", "seed"], ["#weightDecay", "weight_decay"], ["#formConditionDropout", "condition_drop_prob"],
    ["#formBlueprintLossWeight", "blueprint_loss_weight"], ["#formCurveLossWeight", "load_curve_loss_weight"],
    ["#formRelationLossWeight", "relation_loss_weight"], ["#qualityEnabled", "quality_enabled"],
    ["#maxSteps", "max_steps"], ["#formLimitCharts", "limit_charts"], ["#formPrecision", "precision"],
    ["#formMaxRegionSeconds", "max_region_seconds"], ["#formBlueprintBatchSize", "blueprint_batch_size"],
    ["#formEventDecisionLossWeight", "event_decision_loss_weight"], ["#formRhythmLossWeight", "rhythm_loss_weight"],
    ["#formPositionEventLossWeight", "position_event_loss_weight"],
    ["#formPerturbApplyProb", "perturb_apply_prob"], ["#formPerturbCurveNoiseStd", "perturb_curve_noise_std"],
    ["#formPerturbCurveDropoutProb", "perturb_curve_dropout_prob"], ["#formPerturbHistoryCorruptionProb", "perturb_history_corruption_prob"],
    ["#formPerturbOnsetDropProb", "perturb_onset_drop_prob"], ["#formPerturbOnsetAddProb", "perturb_onset_add_prob"],
    ["#formPerturbOnsetShiftMs", "perturb_onset_shift_ms"], ["#formDiversityLossWeight", "diversity_loss_weight"],
    ["#formRepetitionPenaltyWeight", "repetition_penalty_weight"]];
  inherited.forEach(([selector]) => { const node = element(selector); if (node) node.disabled = false; });
  const settings = {...(data?.args || {}), ...(data?.resume_settings || {})};
  for (const [, key] of inherited) {
    if ((key.startsWith("perturb_") || ["repetition_penalty_weight", "diversity_loss_weight"].includes(key))
      && settings[key] == null) settings[key] = key === "perturb_apply_prob" ? 1 : 0;
  }
  if ((data?.model_family || currentModelFamily()) === "song_form_v2" && settings.blueprint_batch_size == null) {
    settings.blueprint_batch_size = 1;
  }
  if ((data?.model_family || currentModelFamily()) === "song_form_v2" && settings.event_decision_loss_weight == null) {
    settings.event_decision_loss_weight = 0;
  }
  if ((data?.model_family || currentModelFamily()) === "song_form_v2" && settings.rhythm_loss_weight == null) settings.rhythm_loss_weight = 0;
  if ((data?.model_family || currentModelFamily()) === "song_form_v2" && settings.position_event_loss_weight == null) {
    settings.position_event_loss_weight = 0;
  }
  if (settings.max_region_seconds == null && data?.dataset_contract?.max_region_seconds != null) {
    settings.max_region_seconds = data.dataset_contract.max_region_seconds;
  }
  const effective = [];
  if (mode === "resume") inherited.forEach(([selector, key]) => {
    const node = element(selector);
    if (!node || settings[key] == null) return;
    if (node.type === "checkbox") node.checked = settings[key] === true;
    else node.value = String(settings[key]);
    node.disabled = true;
    effective.push(`${key}=${settings[key]}`);
  });
  if (locked) structure.forEach(([selector, key]) => {
    if (element(selector) && config[key] != null) effective.push(`${key}=${config[key]}`);
  });
  const geometry = data?.model_config?.rhythm_geometry_profile || "none";
  const status = element("#checkpointInspectStatus"); if (status && data.valid) status.textContent = `${data.model_family || currentModelFamily()}：epoch ${data.epoch ?? 0}；全曲规划 ${data.blueprint_steps || 0} 次；配置训练 ${data.optimizer_steps || 0} 步；节奏几何 ${geometry}；生成${data.can_generate ? "可用" : "未就绪"}，修订${data.can_revise ? "可用" : "未就绪"}${locked ? `；${mode === "resume" ? "resume 继承值" : "checkpoint 固定结构"}：${effective.join(" · ") || "未记录"}` : ""}。`;
  updateSongFormRhythmGeometryControls();
  if (typeof updateSongFormQualityProfileControls === "function") updateSongFormQualityProfileControls();
  if (typeof updateSongFormQualityControls === "function") updateSongFormQualityControls(mode);
}

async function inspectCheckpoint({ autoPreprocess = false, preserveEditableConfig = false } = {}) {
  // Starting finetune always rechecks the checkpoint. Keep explicit upgrades
  // and overrides made after inspection instead of replacing them just before
  // the payload is serialized. Resume still inherits all saved values.
  const editable = preserveEditableConfig && currentTrainingMode() === "finetune"
    ? ["#formTrainingMeterMap", "#formQualityProfile", "#formPositionAlignmentLossWeight", "#formPositionAlignmentToleranceMs",
      "#formPlannerContextSeconds", "#formCurveBinSeconds", "#formRhythmGeometryProfileSelect", "#formRhythmLossWeight"]
      .filter((selector) => {
        const node = element(selector);
        return !(node?.dataset?.evidenceInheritedProfile && node.dataset.evidenceManual !== "1");
      }).map((selector) => [selector, element(selector)?.value]) : [];
  if (["song_form_v1", "song_form_v2"].includes(currentModelFamily())) applySongFormCheckpointConfig({valid: false}, "fresh");
  assertTrainingSelection({ autoPreprocess });
  await assertDatasetBackend();
  const mode = currentTrainingMode();
  if (mode === "fresh") {
    checkpointInspection = { key: "", data: null };
    setCheckpointStatus("从头训练不需要 checkpoint。", "good");
    updateTrainingControls();
    return null;
  }
  const checkpointPath = valueOrDefault("#trainingCheckpointPath", "").trim();
  const manifest = trainingSourcePaths({ autoPreprocess }).manifest;
  if (["song_form_v1", "song_form_v2"].includes(currentModelFamily())) {
    if (!checkpointPath) throw new Error("请先填写已有 song-form checkpoint 路径");
    const request = ++checkpointInspectRequest; setCheckpointStatus("正在检查 song-form checkpoint…");
    const data = await api("/api/form/inspect", { checkpoint: checkpointPath, operation: "train", mode });
    if (request !== checkpointInspectRequest || mode !== currentTrainingMode()
        || !["song_form_v1", "song_form_v2"].includes(currentModelFamily())
        || checkpointPath !== valueOrDefault("#trainingCheckpointPath", "").trim()) return null;
    if (data.valid && data.model_family !== currentModelFamily()) {
      data.valid=false;data.can_train=false;
      data.errors=[`该模型属于 ${data.model_family}，请切换对应模型族续训；新有理数架构须从头训练。`];
    }
    checkpointInspection = { key: currentCheckpointKey(), data };
    if (data.valid && data.can_train) {
      applySongFormCheckpointConfig(data, mode);
      editable.forEach(([selector, value]) => { const node = element(selector); if (node && value !== undefined) node.value = value; });
      setCheckpointStatus(`${data.model_family} checkpoint 可用；${mode === "resume" ? "权重与优化器可加载，实际数据合同将在启动时核对" : "可加强训练"}`, "good");
    }
    else setCheckpointStatus((data.errors || []).join("；") || "song-form checkpoint 不可用于当前模式", "warn");
    updateTrainingControls(); return data;
  }
  if (currentModelFamily() === "chart_composer_v2") {
    if(!checkpointPath)throw new Error("请填写 Composer v2 checkpoint 路径");
    const request=++checkpointInspectRequest,key=currentCheckpointKey();
    const data=await api("/api/composer-v2/inspect",{checkpoint:checkpointPath});
    if(request!==checkpointInspectRequest || key!==currentCheckpointKey())return null;
    const allowed=!!data.valid && (mode==="resume"?data.can_resume:data.can_init);
    data.valid=allowed;checkpointInspection={key,data};
    if(allowed)applyComposerV2Inspection(data);
    setCheckpointStatus(allowed?"Composer v2 checkpoint 可用于当前训练方式":data.error||(data.errors||[]).join("；")||"检查点不支持当前训练方式",allowed?"good":"warn");
    const box=element("#checkpointDetails");if(box){box.hidden=false;box.textContent=JSON.stringify(data,null,2);}
    updateTrainingControls();return data;
  }
  if (currentModelFamily() === "chart_composer_v1") {
    if (!checkpointPath) throw new Error("请先填写已有 chart_composer_v1 checkpoint 路径");
    const request = ++checkpointInspectRequest;
    const key = currentCheckpointKey();
    setCheckpointStatus("正在检查 chart_composer_v1 checkpoint…");
    const box = element("#checkpointDetails");
    if (box) box.hidden = true;
    const data = await api("/api/composer/inspect", { checkpoint: checkpointPath });
    data.model_family = data.model_family || data.family || "chart_composer_v1";
    if (request !== checkpointInspectRequest || key !== currentCheckpointKey()
        || currentModelFamily() !== "chart_composer_v1") return null;
    if (data.valid && data.model_family !== currentModelFamily()) {
      data.valid = false; data.can_train = false;
      data.errors = [`该模型属于 ${data.model_family}，请切换对应模型族续训。`];
    }
    // Finetune controls express the requested new-stage policy even when a
    // manifest edit invalidated the previous compatibility check. Recheck the
    // parent, but do not silently replace those choices before submission.
    const preserveComposerEdits=preserveEditableConfig && (mode==='finetune'
      || (checkpointInspection.key===key && checkpointInspection.data?.valid===true));
    checkpointInspection = { key, data };
    if (mode==='resume' && data.can_resume!==true) {
      data.valid=false;data.error=data.error||data.training_migration||'该文件不含优化器状态；请选择完整训练 checkpoint 或改用 finetune。';
      data.errors=[data.error];
    }
    if (data.valid === true) applyChartComposerCheckpointConfig(data, mode, preserveComposerEdits);
    renderChartComposerCheckpointInspection(data);
    updateTrainingControls();
    return data;
  }
  if (currentModelFamily() === "chart_stream_v1") {
    if (!checkpointPath) throw new Error("请先填写已有 chart_stream_v1 checkpoint 路径");
    const request = ++checkpointInspectRequest;
    const key = currentCheckpointKey();
    setCheckpointStatus("正在检查 chart_stream_v1 checkpoint…");
    const box = element("#checkpointDetails");
    if (box) box.hidden = true;
    const data = await api("/api/checkpoint/inspect", {
      checkpoint_path: checkpointPath, model_family: "chart_stream_v1", mode,
      manifest, auto_preprocess: autoPreprocess,
      manifests: selectedTrainingManifests(), dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
    });
    if (request !== checkpointInspectRequest || key !== currentCheckpointKey()
        || currentModelFamily() !== "chart_stream_v1") return null;
    checkpointInspection = { key, data };
    if (data.valid === true) applyChartStreamCheckpointConfig(data, mode, preserveEditableConfig);
    renderChartStreamCheckpointInspection(data);
    updateTrainingControls();
    return data;
  }
  if (!checkpointPath) throw new Error("请先上传或填写已有模型 checkpoint 路径");
  if (!manifest && !selectedTrainingManifests().length) throw new Error("请先选择训练数据集");
  const request = ++checkpointInspectRequest;
  const key = currentCheckpointKey();
  setCheckpointStatus("正在检查 checkpoint，并与当前 manifest 比对结构合同...");
  const box = element("#checkpointDetails");
  if (box) box.hidden = true;
  const data = await api("/api/checkpoint/inspect", {
    checkpoint_path: checkpointPath, model_family: currentModelFamily(), mode, manifest,
    auto_preprocess: autoPreprocess,
    manifests: selectedTrainingManifests(), dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
  });
  if (request !== checkpointInspectRequest || key !== currentCheckpointKey()) return null;
  checkpointInspection = { key, data };
  if (data.valid === true) applyCheckpointModelConfig(data);
  renderCheckpointInspection(data);
  if (data.valid === true) {
    const contextTokens = element("#contextTokens");
    const contextNote = mode === "finetune" && contextTokens?.dataset.touched === "1"
      ? `；保留手动设置的上下文 token 上限 ${contextTokens.value}` : "";
    setCheckpointStatus(
      `${mode === "resume" ? "已从 checkpoint 载入模型结构与训练合同" : "已从 checkpoint 载入模型结构；加强训练保留采样等可调设置"}${contextNote}${data.warnings?.length ? `；另有 ${data.warnings.length} 项提示，可展开模型合同查看` : ""}`,
      data.warnings?.length ? "warn" : "good",
    );
  }
  updateTrainingControls();
  return data;
}

async function inspectGenerationCheckpoint() {
  const checkpointPath = valueOrDefault("#checkpointPath", "").trim();
  if (!checkpointPath) throw new Error("请先上传或填写生成使用的 checkpoint 路径");
  const request = ++generationInspectRequest;
  setGenerationStatus("正在检查生成 checkpoint...");
  // Generation follows the checkpoint's own schema, independently of the
  // training-family selection. The common inspector delegates form models.
  const selectedFamily = currentModelFamily();
  const data = await api("/api/checkpoint/inspect", { checkpoint_path: checkpointPath, mode: "generate" });
  data.model_family = data.model_family || data.family;
  if (request !== generationInspectRequest || checkpointPath !== valueOrDefault("#checkpointPath", "").trim()) return null;
  generationInspection = { path: checkpointPath, data };
  updateFamilyFields();
  updateEventSetControls();
  if(data.model_family==="chart_composer_v2") {
    setGenerationStatus(data.valid?`Composer v2 checkpoint 可用于生成；${data.generation_capabilities?.continuous_time?'连续事件（无固定拍数输出边界）':'旧窗口方式'}；条件策略 ${data.condition_contract?.policy||'level_only_v1'}${data.condition_contract?.policy==='radar_style_v1'?'':'；未训练 Radar / Style 能力'}`:data.error||(data.errors||[]).join("；")||"Composer v2 checkpoint 不可用",data.valid?"good":"warn");
    const box=element("#generationPlannerStatus");if(box)box.textContent=`模型族：chart_composer_v2；${data.generation_capabilities?.continuous_time?"连续事件；仅歌曲结束有EOS":"旧点：固定窗口结束符"}`;
    window.MaichartWorkflow?.refresh();
    return data;
  }
  if (data.model_family === "chart_composer_v1") {
    invalidateLearnedCurveForCheckpoint(data);
    const composerReady = data.valid === true && data.can_generate !== false;
    const architecture = data.model_config?.architecture_profile || (isEventSetGeneration() ? 'event_set_stream_v1' : '未记录');
    const architectureName = ({event_set_stream_v1:'整拍集合连续生成',audio_grounded_v3:'音频锚定',music_context_v2:'音乐上下文基线'})[architecture] || architecture;
    const architectureSummary = `训练架构：${architectureName}（${architecture}）`;
    const composerStatus = element("#generationPlannerStatus");
    if (composerStatus) composerStatus.textContent = composerReady
      ? `${architectureSummary}。${isEventSetGeneration()?"整体学习同拍配置与相位，使用独立生成器。":"按候选拍位生成事件动作；与整拍集合架构使用不同的检查点和控件。"}`
      : "当前 chart_composer_v1 checkpoint 未通过检查，请先处理 checkpoint 检查结果。";
    const composerNotice = element("#generationCapabilityNotice");
    if (composerNotice) { composerNotice.hidden = false; composerNotice.textContent = composerReady
      ? (isEventSetGeneration()?"整拍集合 checkpoint 可用；可选择1／2／4个短句候选，采音连续性默认关闭。短句组织和持续区间音频是否学过，以检查点能力说明为准。":"Composer checkpoint 可用；温度、失败重试策略、等级、最大事件数和可选 condition 文件会传给独立生成器。")
      : "Composer checkpoint 不可用；请使用 chart_composer_v1 自己的 checkpoint。"; composerNotice.classList.toggle("warn", !composerReady); }
    setGenerationStatus(composerReady
      ? `模型族：chart_composer_v1；${architectureSummary}；保存 epoch ${data.epoch ?? 0}；更新 ${data.optimizer_steps ?? '未记录'} 步；生成可用`
      : (data.error || "chart_composer_v1 checkpoint 不可用于生成"), composerReady ? "good" : "warn");
    return data;
  }
  if (["song_form_v1", "song_form_v2"].includes(data.model_family)) {
    const conditionNote = data.condition_controls_trained ? "Radar / Style 条件监督已训练" : "Radar / Style 仍是未校准条件，请先做 A/B 验证";
    const curveNote = data.load_curve_trained ? "song_form 强度曲线已训练" : "song_form 强度曲线尚未训练";
    const curveBox = element("#formCurveCapability"); if (curveBox) { curveBox.hidden = false; curveBox.textContent = `${curveNote}；${conditionNote}。`; curveBox.classList.toggle("warn", !data.load_curve_trained); }
    const unreadyNote = data.planner_trained === false
      ? "Planner 尚无有效负荷监督，暂不能生成；请用非零规划与曲线损失完成训练。"
      : "song-form checkpoint 尚未完成事件训练";
    setGenerationStatus(data.valid && data.can_generate ? `模型族：${data.model_family}；epoch ${data.epoch ?? 0}；全曲规划 ${data.blueprint_steps || 0} 次；生成可用；${conditionNote}；${curveNote}` : ((data.errors || []).join("；") || unreadyNote), data.valid && data.can_generate ? "good" : "warn");
    return data;
  }
  const family = data.model_family_label || data.model_family || "未知";
  const genCaps = data.generation_capabilities || {};
  if (data.model_family === "chart_stream_v1") {
    const streamReady = data.valid === true && genCaps.continuous_stream === true
      && genCaps.rational_event_positions === true;
    const streamStatus = element("#generationPlannerStatus");
    if (streamStatus) streamStatus.textContent = streamReady
      ? "当前为 chart_stream_v1：连续生成有理拍事件，由模型决定整曲结束；计算块不产生乐句边界。"
      : "当前 chart_stream_v1 checkpoint 未通过连续整曲能力检查，请先处理 checkpoint 检查结果。";
    const streamNotice = element("#generationCapabilityNotice");
    if (streamNotice) {
      streamNotice.hidden = false;
      streamNotice.textContent = streamReady
        ? "连续整曲、有理事件位置和持续动作记忆可用；Radar / Style 作为软条件参与生成。"
        : "连续整曲生成能力未就绪；请使用 chart_stream_v1 自己的 checkpoint。";
      streamNotice.classList.toggle("warn", !streamReady);
    }
    setGenerationStatus(streamReady
      ? `模型族：chart_stream_v1；保存 epoch ${data.epoch ?? 0}；宽度 ${data.model_config?.d_model ?? "未知"}；更新 ${data.optimizer_steps ?? "未记录"} 步；连续整曲生成可用；精确有理事件${genCaps.rational_event_positions ? "可用" : "不可用"}`
      : ((data.errors || []).join("；") || "chart_stream_v1 checkpoint 不可用于连续整曲生成"), streamReady ? "good" : "warn");
    updateGenerationCapabilityControls();
    return data;
  }
  const actionStatus = element("#generationActionStateStatus");
  if (actionStatus) actionStatus.textContent = genCaps.sustained_action_memory
    ? "此模型已训练持续动作摘要：生成时自动读取前段仍未结束的动作，修复或回溯后会按新谱面重建。"
    : (data.model_config?.action_state_profile === "sustain_state_v1"
      ? "此模型的持续动作模块尚无有效训练记录，需继续训练后再生成。"
      : "此模型没有持续动作模块，继续使用原生成路径；需要新增时选择加强训练并启用持续动作摘要。");
  const capSuffix = Object.keys(genCaps).length
    ? `；内嵌无理检测${genCaps.conflict_detection ? "可用" : "不可用"}，细碎音符调制${genCaps.fine_note_strength ? "可用" : "不可用"}，局部采音对齐${genCaps.local_onset_guidance === true ? "可用" : "不可用"}`
    : "";
  setGenerationStatus(`模型族：${family}；schema：${data.schema || "未记录"}；生成入口：${data.generation_cli || "自动"}${capSuffix}`,
    data.valid ? "good" : "warn");
  const plannerStatus = element("#generationPlannerStatus");
  if (plannerStatus) {
    const profile = data.model_config?.planner_profile || data.data_contract?.planner_params?.profile || "legacy_v1";
    plannerStatus.textContent = !data.valid
      ? "模型检查未通过，请先处理检查结果，再选择强度模式。"
      : currentGenerationFamily() === "legacy_frame_v2"
        ? "当前为旧固定帧模型；本页事件模型的强度曲线与 planner 控制不适用。"
        : !data.valid
          ? "checkpoint 合同检查未通过，请先查看检查结果。"
        : profile === "music_structure_v2" || profile === "music_structure_v3"
          ? profile === "music_structure_v3" && genCaps.fine_note_strength
            ? "当前为 music_structure_v3：auto / model 使用学得的局部规划；可启用细碎音符柔性调制。曲线控制局部强弱，不指定节奏类别。"
            : "当前为 music_structure_v2：auto / model 使用学得的局部规划；曲线控制局部强弱，不指定节奏类别。"
          : "当前为 legacy_v1：auto 使用多维音频参考，model 不支持；密度和风格是软条件，模型响应可能较弱。";
  }
  const capabilityBox = element("#generationCapabilityNotice");
  if (capabilityBox) {
    capabilityBox.hidden = false;
    capabilityBox.textContent = !data.valid
      ? ""
      : genCaps.conflict_detection
        ? (genCaps.fine_note_strength
            ? `该 checkpoint 支持静态冲突参考检测与细碎音符柔性调制；${genCaps.local_onset_guidance === true ? "也支持局部采音对齐指导。" : "局部采音对齐指导需要训练过 chart_onsets_v1 的 checkpoint。"}`
            : `该 checkpoint 支持静态冲突参考检测；细碎音符调制需要 music_structure_v3。${genCaps.local_onset_guidance === true ? "局部采音对齐指导可用。" : "局部采音对齐指导需要训练过 chart_onsets_v1 的 checkpoint。"}`)
        : "非事件模型（或未知），不提供冲突检测或局部采音对齐指导。";
  }
  updateGenerationCapabilityControls();
  const notice = element("#generationCheckpointNotice");
  if (notice && data.warnings?.length) notice.textContent = formatList(data.warnings);
  return data;
}

// ---------------------------------------------------------------------------
// Gold set
// ---------------------------------------------------------------------------
function updateGoldFields() {
  const mode = valueOrDefault("#goldMode", "folder");
  const rootField = element("#goldRootField");
  const designersField = element("#goldDesignersField");
  const songsField = element("#goldSongIdsField");
  const listButton = element("#goldDesignerListButton");
  if (rootField) rootField.hidden = mode !== "folder";
  if (designersField) designersField.hidden = mode !== "designers";
  if (songsField) songsField.hidden = mode !== "designers";
  if (listButton) listButton.hidden = mode !== "designers";
}

function renderGoldScan(report) {
  const box = element("#goldSummary");
  if (!box) return;
  box.hidden = false;
  if (!report.exists) {
    box.textContent = report.message || "未找到 gold 文件夹";
    return;
  }
  const titles = report.titles || [];
  const lines = [
    `gold 文件夹：${report.gold_root}`,
    `可用金标曲：${report.songs || 0} 首，可用难度：${report.usable_difficulties ?? "?"} 个`,
    titles.length ? `示例：${titles.slice(0, 5).join("、")}` : "",
    report.skipped ? `跳过：${report.skipped} 首` : "",
    (report.skipped_examples || []).slice(0, 5).map((item) =>
      `跳过 ${item.path}：${item.message || item.reason}`).join("\n"),
  ].filter(Boolean);
  box.textContent = lines.join("\n");
}

async function scanGoldFolder() {
  const data = await api("/api/gold/scan", {
    manifest: datasetPaths().manifest, raw_root: datasetPaths().raw_root,
    gold_root: valueOrDefault("#goldRoot", "").trim() || null, model_family: currentModelFamily(),
  });
  renderGoldScan(data);
  setGoldStatus(data.exists ? `找到 ${data.songs || 0} 首可用金标曲。` : (data.message || "未找到 gold 文件夹"),
    data.exists && data.songs ? "good" : "warn");
  return data;
}

async function listGoldDesigners() {
  const data = await api("/api/gold/designers", { manifest: datasetPaths().manifest });
  const names = (data.designers || []).map((item) => `${item.name}（${item.count}）`);
  setGoldStatus(`可用难度 ${data.usable_difficulties} 个；谱师：${names.slice(0, 20).join("、") || "无"}`, "good");
  return data;
}

async function startGoldBuild() {
  const payload = {
    manifest: datasetPaths().manifest,
    raw_root: datasetPaths().raw_root,
    cache_dir: datasetPaths().cache_dir,
    model_family: currentModelFamily(),
    mode: valueOrDefault("#goldMode", "folder"),
    gold_root: valueOrDefault("#goldRoot", "").trim() || null,
    designers: valueOrDefault("#goldDesigners", "").split("\n").map((item) => item.trim()).filter(Boolean),
    song_ids: valueOrDefault("#goldSongIds", "").split("\n").map((item) => item.trim()).filter(Boolean),
    val_ratio: Number(valueOrDefault("#goldValRatio", 0.3)),
    min_note_count: Number(valueOrDefault("#goldMinNotes", 300)),
    out_dir: valueOrDefault("#goldOutDir", "").trim() || null,
  };
  setGoldStatus("正在生成金标集切分...");
  const data = await api("/api/gold/build", payload);
  watchTask(data.task_id, (task) => {
    if (task.status === "completed") setGoldStatus("金标集切分完成。", "good");
    else setGoldStatus(`金标集切分${task.status === "cancelled" ? "已中断" : "失败"}：${task.error || ""}`, "warn");
  });
}

function renderGoldResult(task) {
  const summary = task.result?.gold_summary;
  const box = element("#goldSummary");
  if (!box) return;
  box.hidden = false;
  if (!summary) {
    box.textContent = task.result?.gold_error || "未找到 gold_summary.json";
    return;
  }
  const lines = [
    `预训练集：${summary.pretrain?.songs ?? "?"} 首`,
    `金标微调集：${summary.gold_finetune?.songs ?? "?"} 首`,
    `金标验证集：${summary.gold_val?.songs ?? "?"} 首`,
    summary.gold_stats ? `金标统计：${summary.gold_stats.charts ?? "?"} 张谱面` : "",
    `输出目录：${task.result?.gold_dir || ""}`,
  ].filter(Boolean);
  box.textContent = lines.join("\n");
  // Point the training panel at the produced manifests.
  if (task.result?.pretrain_manifest_exists && element("#manifestPath")) {
    element("#manifestPath").value = task.result.pretrain_manifest;
  }
  if (task.result?.gold_val_manifest_exists && element("#validationManifestPath")) {
    element("#validationManifestPath").value = task.result.gold_val_manifest;
  }
  if (task.result?.gold_stats_exists && element("#goldStatsPath")) {
    element("#goldStatsPath").value = task.result.gold_stats;
  }
  saveSettings();
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------
async function startTrain(options = {}) {
  if (liveComposerTrainingTasks.size) throw new Error("已有 Composer 训练正在运行或取消，请等待任务退出后再提交训练。");
  if (trainingStarting || trainingPreflight) return;
  trainingPreflight = true;
  updateDatasetButtons();
  try { return await submitTrain(options); }
  finally { trainingPreflight = false; updateDatasetButtons(); }
}

async function submitTrain(options = {}) {
  setWorkspaceMode("training");
  if (datasetUploading) throw new Error("请等待上传完成后再开始训练。");
  assertTrainingSelection({ autoPreprocess: !!options.autoPreprocess });
  await assertDatasetBackend();
  if (currentTrainingMode() !== "fresh") {
    if (options.autoPreprocess && currentModelFamily() === "legacy_frame_v2") {
      throw new Error("旧固定帧模型的续训/加强训练请先点击“预处理当前目录”，完成后使用“开始训练”，以便检查完整的数据合同。");
    }
    const inspection = await inspectCheckpoint({ autoPreprocess: !!options.autoPreprocess, preserveEditableConfig: true });
    if (!inspection || inspection.valid !== true) {
      throw new Error(`已有模型检查未通过：${formatList(inspection?.errors) || "请确认路径、模型族与训练来源"}`);
    }
  }
  const payload = { ...trainingPayload({ autoPreprocess: !!options.autoPreprocess }), auto_preprocess: !!options.autoPreprocess };
  const hasZeroUnmarkedSource = typeof selectedDatasetItems === "function"
    && selectedDatasetItems().some((item) => item && (item.candidate_pool || item.reviewed_only || item.reviewed_regions != null));
  if (hasZeroUnmarkedSource && payload.quality_enabled === false) {
    throw new Error("当前已勾选候选库或审核示范库（未审部分默认零权重），必须开启质量标注与训练示范开关，防止未审内容泄露或示范失效。");
  }
  if(payload.model_family==="chart_composer_v2") {
    validateV2NumericInputs('#composerV2TrainingControls');
    if(options.autoPreprocess)throw new Error("Composer v2 请先准备训练 manifest");
    const backend=await api("/api/status");
    if(payload.onset_readout && payload.onset_readout!=='auto' && !backend.training_capabilities?.composer_v2_onset_readout)
      throw new Error('独立起音参考训练需要更新后的 WebUI 后台，请重启当前后台后提交。');
    if(payload.intensity_curve && !backend.training_capabilities?.composer_v2_intensity_curve)throw new Error('新版强度曲线训练需要更新后的 WebUI 后台，请重启后台后提交。');
    if(payload.rhythm_structure && !backend.training_capabilities?.composer_v2_rhythm_evidence_v2)
      throw new Error('逐拍参考证据与混合分音学习需要更新后的 WebUI 后台，请重启当前后台后提交。');
    if(backend.training_capabilities?.composer_v2_api_contract!=="composer_v2_continuous_events_v3")throw new Error("当前后台不支持 Composer v2 条件与自动节拍合同，请重启为更新后的 WebUI 后台再提交；请先妥善处理正在运行的任务。");
    if (backend.training_capabilities?.composer_v2_memory_budgets !== true &&
        ((payload.cuda_memory_budget_gb ?? 0) !== 0 || (payload.audio_cache_gb ?? 0.25) !== 0.25)) {
      throw new Error('自定义显存与音频缓存预算需要更新后的 WebUI 后台，请重启后台后提交。');
    }
    if ((payload.timing_recovery ?? 'auto') !== 'auto' && backend.training_capabilities?.composer_v2_timing_recovery !== true) {
      throw new Error('指定节奏前文纠错方式需要更新后的 WebUI 后台，请重启后台后提交。');
    }
    if ((payload.activation_checkpointing ?? 'auto') !== 'auto' && backend.training_capabilities?.composer_v2_activation_checkpointing !== true) {
      throw new Error('指定训练显存优化方式需要更新后的 WebUI 后台，请重启后台后提交。');
    }
    if ((payload.reviewed_reference_policy ?? 'auto') !== 'auto' && backend.training_capabilities?.composer_v2_reviewed_reference_policy !== true) {
      throw new Error('指定示范库前后呼应参考策略需要更新后的 WebUI 后台，请重启后台后提交。');
    }
    if ((payload.auto_quality_policy ?? 'auto')!=='auto' && !backend.training_capabilities?.composer_v2_auto_quality) {
      throw new Error('指定自动质检策略需要新版后台，请重启 WebUI 后提交。');
    }
    if ((payload.difficulty_source==='fitted' || payload.min_level!=null || payload.max_level!=null
        || checkpointInspection?.data?.training_contract?.difficulty_selection)
        && !backend.training_capabilities?.composer_v2_difficulty_selection) {
      throw new Error('训练谱面难度筛选需要更新后的 WebUI 后台。请重启后台后提交。');
    }
    if ((payload.difficulty_source==='chartgate' ||
        (payload.mode==='resume' && checkpointInspection?.data?.training_contract?.difficulty_selection?.source==='chartgate'))
        && !backend.training_capabilities?.composer_v2_chartgate) {
      throw new Error('请重启 WebUI 后再使用 Chart;Gate。');
    }
    trainingStarting=true;updateDatasetButtons();
    try{const data=await api("/api/composer-v2/train",payload);watchTask(data.task_id,()=>{trainingStarting=false;updateDatasetButtons();});}
    catch(error){trainingStarting=false;updateDatasetButtons();throw error;}
    return;
  }
  if (payload.model_family === "chart_composer_v1") {
    if (options.autoPreprocess) throw new Error("chart_composer_v1 请先准备 Composer manifest，再开始训练。");
    if (!payload.manifest) throw new Error("请先填写已准备的 Composer manifest");
    const backend=await api('/api/status');
    if(payload.architecture_profile==='event_set_stream_v1'&&backend.capabilities?.event_set_stream!==true)
      throw new Error('当前后台尚未加载整拍集合架构。请连接更新后的 WebUI 后台；保留正在训练的旧服务，不要在旧端口启动新架构。');
    if(payload.reset_learning_rate===true)assertEventSetResumeLearningRateCapability(payload,backend);
    if(payload.architecture_profile==='event_set_stream_v1' && backend.training_capabilities?.composer_detail_compute!=='packed_v1') throw new Error('当前后台尚未加载细节计算方式设置，请重启 WebUI 后台。');
    if(payload.architecture_profile==='event_set_stream_v1' && payload.rebatch_resume===true && backend.training_capabilities?.composer_event_set_rebatch!==true) throw new Error('当前后台尚未加载续训批次调整，请重启 WebUI 后台。');
    if(payload.architecture_profile==='event_set_stream_v1' && payload.data_workers>0 && backend.training_capabilities?.composer_process_prefetch!==true) throw new Error('当前后台尚未加载独立数据准备进程，请重启 WebUI 后台。');
    if(payload.architecture_profile==='event_set_stream_v1' && backend.training_capabilities?.composer_song_style_contract!=='masked_joint_response_v1') throw new Error('当前后台尚未加载整曲风格训练，请重启 WebUI 后台。');
    if(payload.architecture_profile==='event_set_stream_v1' && backend.training_capabilities?.composer_phrase_contract!=='phrase_modes_v1') throw new Error('当前后台尚未加载短句与细节学习策略，请更新 WebUI 后台。');
    if (backend.training_capabilities?.composer_training_runtime!=='music_context_shared_slots_v2'
        || backend.training_capabilities?.composer_training_controls!=='causal_motif_runtime_controls_v11_fresh_training_defaults')
      throw new Error('当前服务尚未加载 Composer 本轮重训默认设置。请重启为更新后的WebUI后端，再应用重训默认值；有活动训练时先保留原服务监控训练。');
    if ((payload.lr_schedule==='cosine'||Number(payload.warmup_steps||0)>0)
        && backend.training_capabilities?.composer_learning_rate_schedule!=='optimizer_updates_v1')
      throw new Error('当前服务尚未加载学习率预热与退火。请保留活动训练，在更新后的WebUI服务中启动新阶段。');
    const effectiveStructure=payload.structure_policy==='auto'
      ?(payload.mode==='fresh'?'metrical_context_v2':checkpointInspection?.data?.model_config?.structure_policy??'off')
      :payload.structure_policy;
    const structureCapability=backend.training_capabilities?.composer_structure_context;
    if (effectiveStructure!=='off' && !(structureCapability==='metrical_context_v2'
        || (effectiveStructure!=='metrical_context_v2' && structureCapability==='metrical_context_v1')))
      throw new Error('当前服务尚未加载所选版本的音乐结构信号，请保留活动训练，在更新后的WebUI服务中启动新阶段。');
    // Even explicit off/flat selections need the new adapter: an old service
    // can discard unknown fields while its new CLI child enables auto defaults.
    if (backend.training_capabilities?.composer_scheme_abc!=='horizon_grouped_condition_v1')
      throw new Error('当前服务尚未加载未来音频视野、动作分类头及条件遮挡控件。请保留活动训练，在更新后的WebUI服务中启动新阶段。');
    trainingStarting = true; updateDatasetButtons();
    try { const data = await api("/api/composer/train", payload); watchTask(data.task_id, () => { trainingStarting = false; updateDatasetButtons(); }); }
    catch (error) { trainingStarting = false; updateDatasetButtons(); throw error; }
    return;
  }
  if (payload.model_family === "chart_stream_v1") {
    if (options.autoPreprocess) throw new Error("chart_stream_v1 请先完成训练素材预处理，再从已准备 manifest 开始训练。");
    if (!payload.manifest) throw new Error("请先填写已准备的 manifest");
    const backend=await api('/api/status');
    if (backend.training_capabilities?.stream_grounded_context !== 'causal_audio_v1')
      throw new Error('当前后端尚未加载连续流关联训练参数，请重启 WebUI 后再训练。');
    if (backend.training_capabilities?.stream_training_runtime !== 'resident_pipeline_v3')
      throw new Error('当前后端尚未加载连续流输入驻留设置，请重启 WebUI 后再训练。');
    if (backend.training_capabilities?.stream_cycle_exit_normalization !== 'main_valid_tokens_v1')
      throw new Error('当前后端尚未加载有界循环退出监督，请在无活动任务时重启 WebUI 后再训练。');
    if (backend.training_capabilities?.stream_motor_contract !== 'causal_hands_v1')
      throw new Error('当前后端尚未加载 v14 因果遮罩与双手特征，请在无活动任务时重启 WebUI。旧权重训练请选择 finetune 和新目录。');
    if (backend.training_capabilities?.stream_runtime_optimization !== 'native_hand_pipeline_v3')
      throw new Error('当前后端尚未加载连续流速度优化和注意力后端设置，请在无活动任务时重启 WebUI 后再训练。');
    trainingStarting = true; updateDatasetButtons();
    try { const data = await api("/api/chart-stream/train", payload); watchTask(data.task_id, () => { trainingStarting = false; updateDatasetButtons(); }); }
    catch (error) { trainingStarting = false; updateDatasetButtons(); throw error; }
    return;
  }
  if (["song_form_v1", "song_form_v2"].includes(payload.model_family)) {
    if (options.autoPreprocess) throw new Error("song-form 请先在训练素材页完成预处理，再从已准备 manifest 开始训练。");
    if (!payload.manifest) throw new Error("请先在训练素材中选择或填写已准备的 manifest");
    trainingStarting = true; updateDatasetButtons();
    try { const data = await api("/api/form/train", payload); watchTask(data.task_id, () => { trainingStarting = false; updateDatasetButtons(); }); }
    catch (error) { trainingStarting = false; updateDatasetButtons(); throw error; }
    return;
  }
  if (payload.model_family !== "legacy_frame_v2") {
    const inputStatus = await api("/api/status");
    if (inputStatus.training_capabilities?.versioned_input_policies !== true) {
      throw new Error("当前后端未加载上下文预算、独立音频池化和星星过滤修正，请重启 WebUI 服务后再训练。");
    }
    if (payload.action_state_profile === "sustain_state_v1" && inputStatus.training_capabilities?.sustained_action_training !== true) {
      throw new Error("当前后端未加载持续动作摘要训练，请重启 WebUI 服务后再训练。");
    }
    const status = await api("/api/status");
    if (status.training_capabilities?.event_learning_rate_control !== true) {
      throw new Error("当前后端尚未加载事件模型学习率修复，请重启 WebUI 服务后再训练。");
    }
    if (payload.onset_profile === "chart_onsets_v1"
        && status.training_capabilities?.local_onset_training !== true) {
      throw new Error("当前后端未加载 chart_onsets_v1 训练支持；auto / none 仍可使用，启用新 profile 请先加载新服务。");
    }
  }
  if (payload.model_family === "event_ctx_v1") {
    const status = await api("/api/status");
    if (status.training_capabilities?.independent_touch_oversample !== true) {
      throw new Error("已更新前端但后端未加载独立Touch采样，请在暂停训练后重启WebUI服务。");
    }
  }
  if (payload.model_family !== "legacy_frame_v2") {
    const needsMusicStructure = payload.planner_profile !== "legacy_v1" || payload.preference_enabled;
    if (needsMusicStructure) {
      const status = await api("/api/status");
      if (status.music_structure_capability !== true) {
        throw new Error("当前后端未加载质量加权/音乐结构 planner 新能力，请重启 WebUI 服务后再开始训练");
      }
    }
  }
  if (payload.quality_enabled || payload.preference_enabled) {
    const status = await api("/api/status");
    if (status.training_capabilities?.quality_training_capability !== true) {
      throw new Error("当前服务未提供质量加权训练能力，请重启 WebUI 服务后再试。");
    }
  }
  trainingStarting = true;
  updateDatasetButtons();
  try {
    const data = await api("/api/train", payload);
    watchTask(data.task_id, () => {
      trainingStarting = false;
      updateDatasetButtons();
    });
  } catch (error) {
    trainingStarting = false;
    updateDatasetButtons();
    throw error;
  }
}

// ---------------------------------------------------------------------------
// Quality-weighted training workspace. All edits go through the API; the
// browser never asks users to author an annotation manifest by hand.
// ---------------------------------------------------------------------------
const QUALITY_LABELS = { excellent: "优秀", normal: "普通", reduced: "减权", excluded: "排除", negative: "反例" };
const QUALITY_DEFAULT_WEIGHTS = { excellent: 2, normal: 1, reduced: 0.25, excluded: 0, negative: 0 };

function qualityManifest() { return activeDatasetItem()?.manifest || datasetPaths().manifest; }
function qualityLabelText(label) { return QUALITY_LABELS[label] || label || "未标注"; }
function qualitySummaryText(summary) {
  if (!summary) return "尚未加载当前 manifest。";
  if (typeof summary === "string") return summary;
  const counts = summary.counts || summary.labels || {};
  const parts = Object.entries(QUALITY_LABELS).map(([key, text]) => `${text} ${counts[key] ?? 0}`);
  return `${parts.join(" · ")}\n已保存标注 ${summary.annotated ?? summary.annotation_count ?? "?"}；反例配对 ${summary.preference_pairs ?? summary.corrected_pairs ?? 0}`;
}

function renderQualitySummary() {
  const text = qualitySummaryText(qualityState.summary);
  ["#qualitySummary", "#qualityTrainingSummary"].forEach((selector) => { const node = element(selector); if (node) node.textContent = text; });
}

const autoAuditViews = { quality: { request: 0 }, training: { request: 0 } };
let completedCombinedAudit = null;
try { completedCombinedAudit = JSON.parse(sessionStorage.getItem('maichart-last-combined-audit')); } catch (_) { /* storage unavailable */ }

function autoAuditSource(panel) {
  const item = panel === "quality" && datasetWorkspaceState.view === "catalog" ? activeDatasetItem() : null;
  if (item) {
    return { catalog: true, readOnly: !!item.auditReadOnly, key: JSON.stringify([item.manifest]),
      label: item.name, payload: { manifest: item.manifest } };
  }
  const selected = selectedDatasetItems();
  const manifests = selected.map(item => item.manifest).filter(Boolean);
  const manifest = manifests[0] || (!datasetWorkspaceState.selectionTouched ? datasetPaths().manifest : "");
  const paths = manifests.length ? [...manifests].sort() : manifest ? [manifest] : [];
  return { key: paths.length ? JSON.stringify(paths) : "",
    label: selected.length ? selected.map(item => item.name).join(" + ") : manifest,
    payload: manifests.length ? { manifest, manifests } : { manifest } };
}

function setAutoAuditStatus(panel, message) {
  const node = element(panel === "quality" ? "#qualityAutoAuditStatus" : "#v2AutoQualityAuditStatus");
  if (node) node.textContent = message;
}

function updateAutoAuditTarget(panel, source = autoAuditSource(panel)) {
  const node = element(panel === "quality" ? "#qualityAutoAuditTarget" : "#v2AutoQualityAuditTarget");
  if (node) node.textContent = source.label ? `体检对象：${source.label}` : "请先打开或勾选已预处理的数据集。";
  if (panel === "quality") {
    const button = element('#qualityAutoAuditCard [data-run-auto-audit="full"]');
    if (button) button.textContent = source.catalog ? "体检当前数据集" : "体检所选训练集";
    for (const action of document.querySelectorAll('#qualityAutoAuditCard [data-run-auto-audit]')) action.hidden = !!source.readOnly;
  }
}

const AUTO_AUDIT_ISSUE_LABELS = {
  auto_audio_misalignment: "整首音画严重错位",
  auto_excessive_muri: "硬伤坏段过多",
  auto_hard_muri: "硬冲突片段",
  auto_conversion_jitter: "转谱微抖动",
  auto_jack_spam: "高速同键连打",
  auto_local_tempo_drift: "局部对音漂移",
  auto_abandoned_gap: "高能空白",
  comfort: "手顺降权",
  audio_unavailable: "音频无法核对",
  audio_identity_unavailable: "音频身份无法核对",
  no_chart_events: "空谱",
  parse_error: "谱面解析失败",
  partial_coverage: "检测覆盖不足",
  Overlap: "同键重叠",
  MultiTouch: "多押超出双手上限",
  TouchMultiAction: "Touch 同手兼顾冲突",
  SlideTooFast: "滑星过快",
  TouchUnreachable: "Touch 不可达",
  TapOnSlide: "Slide 撞尾",
  InnerMuri: "内屏交叉无理",
};

function renderAutoQualityAuditDashboard(data, panel, source, preferredScope) {
  const box = element(panel === "quality" ? "#qualityAutoAuditDashboard" : "#v2AutoQualityDashboard");
  if (!box) return;
  const preview = preferredScope === "preview" ? !!data?.preview_report : !data?.report && !!data?.preview_report;
  const report = preview ? data.preview_report : data?.report;
  box.hidden = !report;
  box.replaceChildren();
  if (!report) return;
  const count = value => Math.max(0, Number(value) || 0);
  const pillGroups = [
    ["整谱排除原因", report.excluded_by_reason, "张"],
    ["已屏蔽冲突", report.masked_conflicts_by_type, "处"],
    ["专项问题", report.domain_pathologies_by_type, "处"],
  ];
  const unverifiedReasons = { ...report.unverified_by_reason };
  if (!report.unverified_by_reason) {
    for (const row of report.unverified_charts || []) {
      const reason = row.reason || "其他待核对";
      unverifiedReasons[reason] = (unverifiedReasons[reason] || 0) + 1;
    }
  }
  pillGroups.push(["待核对原因", unverifiedReasons, "张"]);
  const issueHtml = pillGroups.map(([title, values, unit]) => {
    const pills = Object.entries(values || {}).filter(([, n]) => count(n) > 0).map(([key, n]) => {
      const label = AUTO_AUDIT_ISSUE_LABELS[key.replace(/^miacode_/, "")] || key;
      return `<span class="audit-issue-pill">${escapeHtml(label)} <strong>${count(n)} ${unit}</strong></span>`;
    }).join("");
    return pills ? `<div class="audit-issue-pills"><span class="muted">${title}</span>${pills}</div>` : "";
  }).join("");
  const stat = (name, value, style = "") =>
    `<div class="audit-stat-chip ${style}"><span>${name}</span><strong>${value}</strong></div>`;
  const use = preview ? "抽查预览" : data.has_annotations ? "正式标注已保存" : "只有报告，尚无匹配的正式标注";
  let adoption = "新训练与微调按训练页的采用策略读取";
  if (preview) adoption = "供检查和试听";
  else if (panel === 'training' && currentTrainingMode() === 'resume') adoption = "续训沿用检查点设置";
  else if (panel === 'training' && data.has_annotations && (source.payload.manifests?.length || 0) <= 1) {
    adoption = data.training_effective ? "连续事件训练可按当前设置采用" : "当前设置未采用正式标注";
  }
  const scope = preview ? "preview" : "formal";
  const manifest = report.manifest || data.manifest || source.payload.manifest;
  box.dataset.auditManifest = manifest || "";
  box.dataset.auditScope = scope;
  box.dataset.auditName = source.label || manifest || "";
  const canInspect = manifest && (preview ? data.has_preview_annotations : data.has_annotations);
  box.innerHTML = `<div class="auto-audit-dashboard-head">
    <span class="audit-mode-pill ${preview || !data.has_annotations ? "warn" : "good"}">${use}</span>
    <span class="muted">${adoption}</span>
    </div><p class="muted">${escapeHtml(manifest || source.label || "")}</p>
    <div class="audit-stat-grid">
    ${stat("已检查谱面", `${count(report.audited_charts)} 张`)}
    ${stat("干净通过", `${count(report.clean_charts)} 张`, "clean")}
    ${stat("整谱自动排除", `${count(report.excluded_charts)} 张`, "danger")}
    ${stat("局部问题", `${count(report.partially_masked_charts)} 张${report.partially_masked_regions == null ? "" : ` · ${count(report.partially_masked_regions)} 处`}`, "warn")}
    ${stat("手顺降权", `${count(report.soft_weighted_charts)} 张 · ${count(report.soft_weighted_regions ?? report.problem_regions_by_reason?.comfort)} 处`, "amber")}
    ${stat("优选好段", `${count(report.highlighted_charts)} 张 · ${count(report.highlighted_regions)} 处`, "good")}
    ${stat("待核对", `${count(report.unverified_chart_count)} 张`, "warn")}
    </div>${issueHtml}
    <div class="button-row compact-button-row audit-jump-row">
    ${data.report && data.preview_report ? `<button type="button" class="small-button" data-auto-audit-switch="${preview ? "formal" : "preview"}">${preview ? "查看正式结果" : "查看抽查结果"}</button>` : ""}
    ${canInspect ? `<button type="button" class="small-button" data-auto-audit-jump="flagged">查看问题谱面</button>
      <button type="button" class="small-button" data-auto-audit-jump="good">试听优选好段</button>
      <button type="button" class="small-button" data-auto-audit-jump="excluded">查看整谱排除</button>
      <button type="button" class="small-button" data-auto-audit-jump="unverified">查看待核对</button>
      <button type="button" class="small-button" data-auto-audit-jump="">显示全部谱面</button>` : ""}
    </div>`;
}

async function refreshAutoQualityAuditStatus(panel = "quality", scope) {
  const view = autoAuditViews[panel];
  const source = autoAuditSource(panel);
  if (source.key !== view.key) {
    view.key = source.key;
    view.scope = undefined;
    renderAutoQualityAuditDashboard(null, panel, source);
  }
  if (scope) view.scope = scope;
  updateAutoAuditTarget(panel, source);
  const request = ++view.request;
  if (!source.key) {
    setAutoAuditStatus(panel, "");
    return;
  }
  const combined = completedCombinedAudit?.key === source.key ? completedCombinedAudit.manifest : null;
  const selection = combined ? { manifest: combined }
    : source.payload.manifests?.length > 1 ? { manifests: source.payload.manifests } : source.payload;
  try {
    const data = await api("/api/composer-v2/auto_quality_status", {
      ...selection, quality_enabled: checkedOrDefault("#v2TrainQualityEnabled", true),
      mode: currentTrainingMode(),
      auto_quality_policy: valueOrDefault("#v2TrainAutoQualityPolicy", "auto"),
    });
    if (request !== view.request || source.key !== autoAuditSource(panel).key) return;
    renderAutoQualityAuditDashboard(data, panel, source, view.scope);
    if (!element("#v2AutoQualityAuditButton")?.disabled) {
      const message = data.report || data.preview_report ? ""
        : data.has_annotations ? "已找到正式标注，当前没有对应的统计报告。"
        : source.payload.manifests?.length > 1
          ? "可体检这组训练来源，也可打开单个库查看它的报告。"
          : "这里还没有体检报告。";
      setAutoAuditStatus(panel, message);
    }
  } catch (error) {
    if (request === view.request && source.key === autoAuditSource(panel).key) {
      renderAutoQualityAuditDashboard(null, panel, source);
      setAutoAuditStatus(panel, `读取体检状态失败：${error.message}`);
    }
  }
}

function finishAutoAudit(panel, source, report, scope) {
  if (source.payload.manifests?.length > 1 && report.manifest) {
    completedCombinedAudit = { key: source.key, manifest: report.manifest };
    try { sessionStorage.setItem('maichart-last-combined-audit', JSON.stringify(completedCombinedAudit)); } catch (_) { /* storage unavailable */ }
  }
  if (source.key !== autoAuditSource(panel).key) return;
  if (panel === "quality" && source.catalog) {
    void changeCatalogAuditScope(scope).catch(error => setAutoAuditStatus(panel, error.message));
    return;
  }
  void refreshAutoQualityAuditStatus(panel, scope);
}

async function changeCatalogAuditScope(scope) {
  const manifest = qualityManifest();
  const chartId = qualityState.detail?.chart_id;
  const request = ++qualityState.detailRequest;
  qualityState.autoScope = scope;
  qualityState.previewLoading = false;
  await loadQualityCatalog(true);
  if (chartId && request === qualityState.detailRequest && manifest === qualityManifest()
      && document.body.dataset.workspace === 'quality') {
    await openQualityReview(manifest, chartId, request, { scroll: false });
  }
}

async function jumpToAutoAuditCatalog(box, filter) {
  const manifest = box?.dataset.auditManifest;
  if (!manifest) return;
  const item = datasetWorkspaceState.items.find(row => sameAuditManifest(row.manifest, manifest));
  const target = item || { manifest, path: manifest, name: "体检清单", prepared: true, auditReadOnly: true };
  setWorkspaceMode("quality");
  await openDataset(target, { autoScope: box.dataset.auditScope, autoFilter: filter });
  element("#datasetCatalogPanel")?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function sameAuditManifest(left, right) {
  const normalize = path => {
    const value = String(path || '').replaceAll('\\', '/');
    return /^[a-z]:\//i.test(value) || value.startsWith('//') ? value.toLowerCase() : value;
  };
  return normalize(left) === normalize(right);
}

function markActivePresetChip(attrName, value) {
  document.querySelectorAll(`[${attrName}]`).forEach((chip) => {
    chip.classList.toggle("is-active", chip.getAttribute(attrName) === value);
  });
}

function applyControlMap(changes) {
  for (const [id, val] of Object.entries(changes)) {
    const node = element("#" + id);
    if (!node || node.disabled) continue;
    if (node.type === "checkbox") node.checked = Boolean(val);
    else node.value = String(val);
  }
}

function applyAutoQualityPreset(presetName) {
  const presets = {
    standard: {
      values: {
        v2AutoQualityLimit: 0, v2AutoQualityTailMode: "mask_hard", v2AutoQualityTailMs: 200,
        v2AutoQualityInnerMuriMode: "allow", v2AutoQualityTouchPolicy: "palm_span_v1", v2AutoQualityTouchRadius: 180,
        v2AutoQualityDomainMode: "maimai_dx_v1", v2AutoQualityAcousticMode: "auto",
        v2AutoQualityHighlightMode: "auto", v2AutoQualityGoodQuantile: 0.82,
        v2AutoQualityMaxProblemFraction: 0.45, v2AutoQualityHandFlowQuantile: 0.05,
      },
      note: "已套用「标准清洗＋优选好段（推荐）」：屏蔽硬无理与音画错位，轻降权 5% 别扭手顺，提权前 18% 精品乐句。点击「一键自动体检所选训练集」即可开始。",
    },
    strict: {
      values: {
        v2AutoQualityLimit: 0, v2AutoQualityTailMode: "mask_all", v2AutoQualityTailMs: 220,
        v2AutoQualityInnerMuriMode: "mask_hard", v2AutoQualityTouchPolicy: "palm_span_v1", v2AutoQualityTouchRadius: 165,
        v2AutoQualityDomainMode: "maimai_dx_v1", v2AutoQualityAcousticMode: "auto",
        v2AutoQualityHighlightMode: "acoustic_and_flow_v1", v2AutoQualityGoodQuantile: 0.86,
        v2AutoQualityMaxProblemFraction: 0.35, v2AutoQualityHandFlowQuantile: 0.10,
      },
      note: "已套用「严格排错（适合杂乱自制谱合集）」：连同撞尾警告与内屏硬无理一并屏蔽，问题超 35% 整张排除，好段必须声学与手顺双达标。",
    },
    conservative: {
      values: {
        v2AutoQualityLimit: 0, v2AutoQualityTailMode: "mask_hard", v2AutoQualityTailMs: 200,
        v2AutoQualityInnerMuriMode: "allow", v2AutoQualityTouchPolicy: "palm_span_v1", v2AutoQualityTouchRadius: 180,
        v2AutoQualityDomainMode: "maimai_dx_v1", v2AutoQualityAcousticMode: "auto",
        v2AutoQualityHighlightMode: "off", v2AutoQualityGoodQuantile: 0.82,
        v2AutoQualityMaxProblemFraction: 0.45, v2AutoQualityHandFlowQuantile: 0,
      },
      note: "已套用「仅排错不提权（保守清洗）」：只屏蔽硬无理、转谱顽疾与音画错位片段，不额外放大任何段落权重。",
    },
  };
  const chosen = presets[presetName];
  if (!chosen) return;
  applyControlMap(chosen.values);
  markActivePresetChip("data-auto-audit-preset", presetName);
  for (const selector of ["#qualityAutoAuditStatus", "#v2AutoQualityAuditStatus"]) {
    const node = element(selector);
    if (node) node.textContent = chosen.note;
  }
  saveSettings();
}

const V2_TRAIN_SMOKE_PLAN = Object.freeze({
  v2TrainEpochs: '1',
  v2TrainLimitCharts: '20',
  v2TrainMaxSteps: '32',
  v2TrainValidationMaxBatches: '4',
});
let v2TrainPlanBeforeSmoke = null;

function restoreV2TrainPlan() {
  if (!v2TrainPlanBeforeSmoke) return false;
  applyControlMap(v2TrainPlanBeforeSmoke);
  v2TrainPlanBeforeSmoke = null;
  return true;
}

function applyV2TrainPreset(presetName) {
  const presets = {
    vram8g: {
      title: '8G–12G 显卡省显存',
      values: {
        v2TrainBatchSize: 4, v2TrainGradAccumulation: 16, v2TrainActivationCheckpointing: 'on',
        v2TrainAudioCacheGb: 0.25, v2TrainPrecision: 'auto',
      },
      memory: '开启显存重算，音频缓存 0.25 GiB',
    },
    standard16g: {
      title: '16G+ 显卡标准速度',
      values: {
        v2TrainBatchSize: 8, v2TrainGradAccumulation: 8, v2TrainActivationCheckpointing: 'auto',
        v2TrainAudioCacheGb: 0.5, v2TrainPrecision: 'auto',
      },
      memory: '显存重算自动选择，音频缓存 0.5 GiB',
    },
  };
  const status = element('#v2TrainPresetStatus');
  const resume = currentTrainingMode() === 'resume';
  if (presetName === 'smoke') {
    if (resume) return;
    v2TrainPlanBeforeSmoke ??= Object.fromEntries(Object.keys(V2_TRAIN_SMOKE_PLAN)
      .map(id => [id, element('#' + id).value]));
    applyControlMap(V2_TRAIN_SMOKE_PLAN);
    status.textContent = '已套用短跑计划。训练 1 轮，最多 20 张谱面、32 次更新尝试和 4 个验证批次。切换到显卡预设即可恢复短跑前的计划。';
  } else {
    const chosen = presets[presetName];
    if (!chosen) return;
    const restoredPlan = restoreV2TrainPlan();
    applyControlMap(chosen.values);
    const batch = `批大小 ${chosen.values.v2TrainBatchSize} × 累积 ${chosen.values.v2TrainGradAccumulation}，精度自动选择`;
    status.textContent = resume
      ? `已应用「${chosen.title}」中的可调设置，${chosen.memory}。训练计划、批大小、累积与精度沿用检查点。`
      : `已应用「${chosen.title}」，${batch}，${chosen.memory}。${restoredPlan ? '已恢复短跑前的训练计划。' : ''}`;
  }
  markActivePresetChip('data-v2-train-preset', presetName);
  updateTrainingControls();
  saveSettings();
}

function applyV2GenPreset(presetName) {
  const presets = {
    balanced: {
      values: {
        v2GeneratePhraseSearch: "layout", v2GeneratePhraseRankPolicy: "comfort_v1", v2GenerateCandidates: "2",
        v2GenerateSelfRefine: "off", v2GeneratePhraseBeats: 4, v2GeneratePhraseCommitBeats: 2,
        v2GenerateInterceptTapOnSlide: true, v2GenerateInterceptSlideHeadTap: false,
        v2GenerateForwardFeasibility: "balanced_v1", v2GenerateCount: 1,
        v2GenerateTemperature: 0.9, v2GenerateActionTemperature: 0.9,
      },
      note: '已套用「推荐稳妥写谱」。先定节奏，再试写两种配置，并启用防撞尾与下笔前防锁。',
    },
    refined: {
      values: {
        v2GeneratePhraseSearch: "layout", v2GeneratePhraseRankPolicy: "comfort_v2", v2GenerateCandidates: "4",
        v2GenerateSelfRefine: "worst_spans_v1", v2GeneratePhraseBeats: 4, v2GeneratePhraseCommitBeats: 2,
        v2GenerateInterceptTapOnSlide: true, v2GenerateInterceptSlideHeadTap: true,
        v2GenerateForwardFeasibility: "balanced_v1", v2GenerateCount: 1,
        v2GenerateTemperature: 0.85, v2GenerateActionTemperature: 0.9,
      },
      note: '已套用「高质精雕细琢」。每次最多试写 4 种配置，按左右手手感选优，开启星头抢拍检查并尝试精修薄弱段。',
    },
    fast: {
      values: {
        v2GeneratePhraseSearch: "event", v2GenerateCandidates: "1", v2GenerateSelfRefine: "off",
        v2GenerateInterceptTapOnSlide: true, v2GenerateCount: 1,
      },
      note: '已套用「快速试写预览」。逐步生成 1 张谱面，开启防撞尾。',
    },
    batch4: {
      values: {
        v2GeneratePhraseSearch: "layout", v2GeneratePhraseRankPolicy: "comfort_v2", v2GenerateCandidates: "2",
        v2GenerateSelfRefine: "off", v2GenerateInterceptTapOnSlide: true, v2GenerateCount: 4, v2GenerateParallel: 1,
      },
      note: '已套用「一次抽 4 张挑最好」。依次生成 4 张完整谱面，完成后可在顶部「谱面预览」切换。',
    },
  };
  const chosen = presets[presetName];
  if (!chosen) return;
  const {v2GeneratePhraseRankPolicy: rank, ...controls} = chosen.values;
  applyControlMap(controls);
  updateV2BatchControls();
  if (rank) applyControlMap({v2GeneratePhraseRankPolicy: rank});
  const legacy = element('#v2GeneratePhraseSearch')?.disabled;
  markActivePresetChip('data-v2-gen-preset', legacy ? '' : presetName);
  const status = element("#v2GenPresetStatus");
  if (status) status.textContent = legacy
    ? '已应用当前模型支持的设置。该模型使用旧窗口生成，比较整段写法与自动精修停用。'
    : chosen.note;
  saveSettings();
}

function qualityRowHtml(item, readOnly = false) {
  const pool=!!item.candidate_status;
  const id = escapeHtml(item.chart_id || item.id || "");
  const label = item.label || item.record?.label || "";
  const rowId = String(item.chart_id || item.id || "");
  const weight = Number(qualityState.editedWeights.get(rowId) ?? item.weight ?? item.record?.weight ?? QUALITY_DEFAULT_WEIGHTS[label] ?? 1);
  const title = item.maidata_title || item.title || item.song_title || item.song_id || "未知歌曲";
  const folder = item.folder_name || item.folder || item.song_folder || item.song_id || "";
  const source = item.source_dataset_name || item.dataset_name || item.source_dataset || "";
  const difficulty = item.level !== undefined ? `${item.level} · ${item.difficulty_index ?? ""}` : (item.difficulty_index ?? "");
  const selectedId = String(item.chart_id || item.id || "");
  const disabled = qualityState.selectionLoading || qualityState.saving ? "disabled" : "";
  const manualScope = item.auto_override_scope === 'chart' ? '有人工整谱决定' : item.auto_override_scope === 'region' ? '有人工片段标注' : '';
  const autoBadge = !pool && item.auto_badge
    ? `<div class="auto-audit-row-badges"><span class="auto-audit-badge ${escapeHtml(item.auto_status || "clean")}">${escapeHtml(item.auto_badge)}</span>${manualScope ? `<span class="auto-audit-override">${manualScope}</span>` : ""}</div>`
    : "";
  return `<tr data-chart-id="${id}"><td ${pool||readOnly?'hidden':''}>${pool||readOnly?'':`<input class="quality-row-select" type="checkbox" data-chart-id="${id}" ${qualityState.selected.has(selectedId) ? "checked" : ""} ${disabled}>`}</td>`
    + `<td><strong>${escapeHtml(title)}</strong><br><span class="muted">难度 ${escapeHtml(String(difficulty))}</span><br><span class="dataset-row-folder">文件夹：${escapeHtml(folder)}</span>${source ? `<br><span class="muted">来源：${escapeHtml(source)}</span>` : ""}${globalThis.CandidatePool?.badge(item)||""}${autoBadge}</td>`
    + `<td>${escapeHtml(item.designer || item.chart_designer || "未标注")}</td><td><span class="quality-label ${escapeHtml(label)}">${escapeHtml(pool?(globalThis.CandidatePool?.reviewText(item)||'尚未确认学习'):qualityLabelText(label))}</span></td>`
    + `<td>${readOnly ? escapeHtml(String(weight)) : pool?(item.confirmed_regions?`${item.confirmed_regions} 段`:'—'):`<input class="quality-weight" data-chart-id="${id}" type="number" min="0" max="4" step="0.05" value="${Number.isFinite(weight) ? weight : 1}" ${disabled}>`}</td>`
    + `<td><button class="small-button quality-detail-button" type="button" data-chart-id="${id}">${readOnly?'预览':'预览与标注'}</button></td></tr>`;
}

function renderQualityCatalog() {
  const body = element("#qualityCatalogBody"); if (!body) return;
  const pool=!!activeDatasetItem()?.candidate_pool;
  const readOnly=!!activeDatasetItem()?.auditReadOnly;
  if(element('#qualityBulkControls'))element('#qualityBulkControls').hidden=pool||readOnly;
  if(element('#qualityLegacyDetails'))element('#qualityLegacyDetails').hidden=readOnly;
  if(element('#qualityLabelHeading'))element('#qualityLabelHeading').textContent=pool?'你的学习决定':'标签';
  if(element('#qualityWeightHeading'))element('#qualityWeightHeading').textContent=pool?'已确认片段':'权重';
  if(element('#qualityLabelFilter'))element('#qualityLabelFilter').hidden=pool;
  if(element('#qualityAutoFilter'))element('#qualityAutoFilter').hidden=pool;
  if(element('#qualitySummary'))element('#qualitySummary').hidden=pool;
  if(element('#qualitySelectAll'))element('#qualitySelectAll').closest('th').hidden=pool||readOnly;
  body.innerHTML = qualityState.items.map(item=>qualityRowHtml(item,readOnly)).join("") || `<tr><td colspan="6" class="muted">没有匹配的谱面。</td></tr>`;
  const pageCount = Math.max(1, qualityState.pages || Math.ceil(qualityState.total / qualityState.pageSize));
  if (element("#qualityPageStatus")) element("#qualityPageStatus").textContent = `第 ${qualityState.page} / ${pageCount} 页（共 ${qualityState.total} 张）`;
  if (element("#qualityPageInput")) element("#qualityPageInput").value = String(qualityState.page);
  if (element("#qualityPrevButton")) element("#qualityPrevButton").disabled = qualityState.page <= 1;
  if (element("#qualityNextButton")) element("#qualityNextButton").disabled = qualityState.page >= pageCount;
  updateQualitySelectionControls();
  updateTrainingPreviewNavigation();
}

function updateTrainingPreviewNavigation() {
  const navigation=element('#trainingPreviewNavigation');if(!navigation)return;
  const preview=globalThis.ChartPreview?.getEntry(),context=preview?.training_context;
  const training=preview?.source_kind==='training_source';
  navigation.hidden=!training;
  const matches=training && context?.manifest===qualityManifest();
  const index=matches?qualityState.items.findIndex(item=>String(item.chart_id)===context.chart_id):-1;
  const busy=qualityState.previewLoading || qualityState.previewNavigating;
  element('#trainingPreviewPrevious').disabled=!!busy || index<0 || (index===0 && qualityState.page<=1);
  element('#trainingPreviewNext').disabled=!!busy || index<0 || (index===qualityState.items.length-1 && qualityState.page>=qualityState.pages);
  const label=element('#trainingPreviewPosition');
  label.textContent=qualityState.previewError?'载入失败，可重试':index<0?'请选择目录中的谱面':`${(qualityState.page-1)*qualityState.pageSize+index+1} / ${qualityState.total} 张 · ${qualityState.items[index].title||qualityState.items[index].song_id||''}`;
  label.title=qualityState.previewError || (index<0?'':`${qualityState.items[index].title||''} · 难度 ${qualityState.items[index].level||''}`);
  for(const row of document.querySelectorAll('#qualityCatalogBody tr[data-chart-id]')) {
    const current=matches && row.dataset.chartId===context.chart_id;
    if(current)row.setAttribute('aria-current','true');else row.removeAttribute('aria-current');
    row.querySelector('.quality-detail-button')?.setAttribute('aria-pressed',String(current));
  }
}

async function navigateTrainingPreview(direction) {
  if(qualityState.previewLoading || qualityState.previewNavigating)return;
  const context=globalThis.ChartPreview?.getEntry()?.training_context,manifest=qualityManifest();
  if(context?.manifest!==manifest)return;
  const index=qualityState.items.findIndex(item=>String(item.chart_id)===context.chart_id);
  if(index<0)return;
  const request=qualityState.detailRequest;
  let navigationRequest=request;
  const originalPage=qualityState.page;
  qualityState.previewNavigating=true;qualityState.previewError='';updateTrainingPreviewNavigation();
  try {
    let row=qualityState.items[index+direction];
    if(!row) {
      const page=qualityState.page+direction;
      if(page<1 || page>qualityState.pages)return;
      qualityState.page=page;await loadQualityCatalog(false);
      if(manifest!==qualityManifest() || request!==qualityState.detailRequest)return;
      row=direction>0?qualityState.items[0]:qualityState.items.at(-1);
    }
    if(row) {
      const opening=openQualityDetail(row.chart_id,{scroll:false});
      navigationRequest=qualityState.detailRequest;await opening;
    }
  } catch(error) {
    if(manifest===qualityManifest() && navigationRequest===qualityState.detailRequest) {
      qualityState.page=originalPage;qualityState.previewError=error.message;renderQualityCatalog();
    }
    throw error;
  } finally {qualityState.previewNavigating=false;updateTrainingPreviewNavigation();}
}

async function browseTrainingPreview() {
  if(!element('#generationPreviewCollapse')?.hidden)element('#generationPreviewCollapse').click();
  setWorkspaceMode('quality');
  if(datasetWorkspaceState.view==='catalog') {
    const id=globalThis.ChartPreview?.getEntry()?.training_context?.chart_id;
    const row=[...document.querySelectorAll('#qualityCatalogBody tr[data-chart-id]')].find(row=>row.dataset.chartId===id);
    (row||element('#datasetCatalogPanel'))?.scrollIntoView({behavior:'smooth',block:row?'center':'start'});
  } else {
    await loadDatasetCatalog();element('#datasetCardsPanel')?.scrollIntoView({behavior:'smooth',block:'start'});
  }
}
element('#trainingPreviewBrowse')?.addEventListener('click',guard(browseTrainingPreview));
element('#trainingPreviewCatalog')?.addEventListener('click',guard(browseTrainingPreview));
element('#trainingPreviewPrevious')?.addEventListener('click',guard(()=>navigateTrainingPreview(-1)));
element('#trainingPreviewNext')?.addEventListener('click',guard(()=>navigateTrainingPreview(1)));
window.addEventListener('chart-preview-updated',updateTrainingPreviewNavigation);
window.addEventListener('chart-preview-annotations-updated',updateTrainingPreviewNavigation);

function updateQualitySelectionControls() {
  const busy = qualityState.selectionLoading || qualityState.saving;
  if (element("#qualitySelectedCount")) element("#qualitySelectedCount").textContent = qualityState.selected.size;
  const selectAll = element("#qualitySelectAll");
  const selectedOnPage = qualityState.items.filter((item) => qualityState.selected.has(String(item.chart_id || item.id))).length;
  if (selectAll) { selectAll.checked = qualityState.items.length > 0 && selectedOnPage === qualityState.items.length; selectAll.indeterminate = selectedOnPage > 0 && selectedOnPage < qualityState.items.length; selectAll.disabled = busy; }
  if (element("#qualitySelectFolderButton")) element("#qualitySelectFolderButton").disabled = busy || !qualityState.folderTotal;
  if (element("#qualityClearSelectionButton")) element("#qualityClearSelectionButton").disabled = qualityState.saving || !qualityState.selected.size && !qualityState.selectionLoading;
  if (element("#qualitySaveSelectedButton")) element("#qualitySaveSelectedButton").disabled = busy || !qualityState.selected.size;
  const status = element("#qualitySelectionStatus");
  if (status) status.textContent = qualityState.selectionLoading ? "正在读取当前文件夹的全部谱面…"
    : qualityState.saving ? `正在保存 ${qualityState.selected.size} 张谱面的标注…`
    : `已选 ${qualityState.selected.size} 张 / 当前文件夹共 ${qualityState.folderTotal} 张。文件夹全选跨越所有分页，包含搜索和标签筛选隐藏的条目；选择后再保存标注。`;
}

function clearQualitySelection() {
  qualityState.selectionRequest += 1;
  qualityState.selectionLoading = false;
  qualityState.selected.clear();
  renderQualityCatalog();
}

async function selectQualityFolder() {
  const manifest = qualityManifest();
  if (!manifest) throw new Error("请先打开一个已预处理的数据集目录。");
  const request = ++qualityState.selectionRequest;
  qualityState.selectionLoading = true;
  renderQualityCatalog();
  try {
    let data;
    try { data = await api("/api/quality/selection", { manifest }); }
    catch (error) {
      if (/404|Not found/i.test(String(error.message))) throw new Error("当前后端尚未加载跨页全选，请重启 WebUI 服务后再试。");
      throw error;
    }
    if (request !== qualityState.selectionRequest || manifest !== qualityManifest()) return;
    if (!Array.isArray(data.chart_ids)) throw new Error("后端没有返回完整谱面列表，请重启 WebUI 服务。");
    qualityState.selected = new Set(data.chart_ids.map(String));
    qualityState.folderTotal = Number(data.total || qualityState.selected.size);
    for (const id of qualityState.editedWeights.keys()) if (!qualityState.selected.has(id)) qualityState.editedWeights.delete(id);
  } finally {
    if (request === qualityState.selectionRequest) { qualityState.selectionLoading = false; renderQualityCatalog(); }
  }
}

async function loadQualityCatalog(resetPage = false) {
  const manifest = qualityManifest();
  if (!manifest) throw new Error("请先填写训练 manifest 路径。");
  if (resetPage) qualityState.page = 1;
  const request = ++qualityState.catalogRequest;
  const sortByUi = valueOrDefault("#qualitySortBy", "difficulty");
  const sortBy = sortByUi === "difficulty" ? "level" : sortByUi;
  const sortDirection = valueOrDefault("#qualitySortDirection", "asc");
  const isPool = Boolean(activeDatasetItem()?.candidate_pool);
  const autoFilterValue = isPool ? "" : (qualityState.autoFilter || valueOrDefault("#qualityAutoFilter", ""));
  const data = await api("/api/quality/catalog", { manifest, query: qualityState.query, page: qualityState.page, page_size: qualityState.pageSize, label_filter: isPool ? null : qualityState.label || null, auto_filter: autoFilterValue || null, sort_by: sortBy, sort_order: sortDirection,
    auto_scope: qualityState.autoScope || 'formal',
    candidate_status:valueOrDefault('#candidatePoolStatusFilter',''),
    candidate_review:isPool?valueOrDefault('#candidatePoolDecisionFilter',''):'' });
  if (request !== qualityState.catalogRequest || manifest !== qualityManifest()) return;
  if (!qualityState.autoScope && !data.auto_quality?.has_annotations && data.auto_quality?.has_preview_annotations) {
    qualityState.autoScope = 'preview';
    return loadQualityCatalog(resetPage);
  }
  qualityState.items = Array.isArray(data.items) ? data.items : [];
  qualityState.page = Math.max(1, Number(data.page ?? qualityState.page) || 1);
  qualityState.pageSize = Math.max(1, Number(data.page_size ?? qualityState.pageSize) || qualityState.pageSize);
  qualityState.pages = Math.max(1, Number(data.pages ?? Math.ceil(Number(data.total || 0) / qualityState.pageSize)) || 1);
  qualityState.total = Number(data.total || 0);
  qualityState.folderTotal = Number(data.folder_total ?? data.total ?? 0);
  qualityState.annotationPath = data.annotation_path || qualityState.annotationPath || "";
  qualityState.annotationManifest = manifest;
  qualityState.summary = data.summary || null;
  const source = autoAuditSource('quality');
  const view = autoAuditViews.quality;
  view.key = source.key;
  view.scope = qualityState.autoScope;
  ++view.request;
  updateAutoAuditTarget('quality', source);
  renderAutoQualityAuditDashboard(data.auto_quality, 'quality', source, view.scope);
  renderQualitySummary(); renderQualityCatalog();
}

function isZeroUnmarkedQualityDetail(detail = qualityState.detail) {
  return Boolean(detail && (detail.zero_unmarked || detail.record?.unmarked_weight === 0 || isZeroUnmarkedDataset(activeDatasetItem())));
}

function updateQualityDetail() {
  const detail = qualityState.detail; if (!detail) return;
  const previewParts = [];
  if (detail.preview) previewParts.push(`事件预览\n${typeof detail.preview === "string" ? detail.preview : JSON.stringify(detail.preview, null, 2)}`);
  if (detail.maidata) previewParts.push(`maidata\n${detail.maidata}`);
  if (detail.inote) previewParts.push(`inote\n${detail.inote}`);
  const preview = previewParts.join("\n\n") || "";
  if (element("#qualityDetailTitle")) element("#qualityDetailTitle").textContent = `${detail.title || detail.chart_id || "谱面详情"} · ${detail.level || ""}`;
  if (element("#qualityDetailPreview")) element("#qualityDetailPreview").textContent = preview;
  const audio = element("#qualityDetailAudio"); if (audio && detail.audio_url && audio.getAttribute("src")!==detail.audio_url) audio.src = detail.audio_url;
  const zeroUnmarked = typeof isZeroUnmarkedQualityDetail === "function"
    ? isZeroUnmarkedQualityDetail(detail)
    : Boolean(detail && (detail.zero_unmarked || detail.record?.unmarked_weight === 0));
  const disabledAttr = zeroUnmarked ? "disabled" : "";
  for (const selector of ["#qualityAddRegionButton", "#qualitySaveRegionsButton"]) {
    const btn = element(selector); if (btn) btn.disabled = zeroUnmarked;
  }
  document.querySelectorAll("[data-quality-span-tag]").forEach((btn) => { btn.disabled = zeroUnmarked; });
  const spanStatus = element("#qualitySpanStatus");
  if (spanStatus && zeroUnmarked) {
    spanStatus.textContent = "这个来源只学习已确认片段。请点「回到谱面预览」，在时间轴圈选后保存学习内容。";
  } else if (spanStatus && spanStatus.textContent.startsWith("这个来源只学习已确认片段")) {
    spanStatus.textContent = "";
  }
  const list = element("#qualityRegionsList"); if (!list) return;
  list.innerHTML = qualityState.regions.map((region, index) => `<div class="quality-region-row" data-region-index="${index}" data-entry-start-sec="${region.entry_start_sec ?? ''}"><input type="number" min="0" step="0.01" value="${Number(region.start_sec ?? region.start ?? 0)}" data-region-field="start_sec" ${disabledAttr}><input type="number" min="0" step="0.01" value="${Number(region.end_sec ?? region.end ?? 0)}" data-region-field="end_sec" ${disabledAttr}><select aria-label="片段评价" data-region-field="label" ${disabledAttr}>${[['good','优秀'],['problem','有问题'],['uncertain','不确定']].map(([value,label])=>`<option value="${value}" ${(region.label||'uncertain')===value?'selected':''}>${label}</option>`).join('')}</select><select aria-label="问题类型" data-region-field="reason" ${disabledAttr}>${[['','未指定'],['timing','采音'],['layout','配置'],['comfort','手感']].map(([value,label])=>`<option value="${value}" ${region.reason===value?'selected':''}>${label}</option>`).join('')}</select><details class="quality-span-weights"><summary>高级权重</summary><input type="number" min="0" max="4" step="0.05" value="${Number(region.weight ?? 0)}" data-region-field="weight" ${disabledAttr}><input aria-label="采音权重" title="采音权重" type="number" min="0" max="4" step="0.05" value="${Number(region.timing_weight ?? (region.label==='good'?1.5:region.label==='problem' && region.reason==='timing'?0:1))}" data-region-field="timing_weight" ${disabledAttr}><input aria-label="配置权重" title="配置权重" type="number" min="0" max="4" step="0.05" value="${Number(region.component_weight ?? (region.label==='good'?1.5:region.label==='problem' && ['layout','comfort'].includes(region.reason)?0:1))}" data-region-field="component_weight" ${disabledAttr}></details><button class="small-button quality-remove-region" type="button" ${disabledAttr}>删除</button></div>`).join("") || `<span class="muted">${zeroUnmarked ? "尚未确认学习片段；请在页面顶部「谱面预览」中圈选并确认。" : "暂无局部标注。片段评价会调整对应的采音或配置监督，其余前文仍保留。"}</span>`;
}

async function openQualityDetail(chartId,{scroll=true}={}) {
  const manifest = qualityManifest();
  const workspace=document.body.dataset.workspace;
  const request = ++qualityState.detailRequest;
  qualityState.previewLoading=true;qualityState.previewError='';updateTrainingPreviewNavigation();
  const payload = {manifest, chart_id:chartId, auto_scope:qualityState.autoScope || 'formal'};
  try {
    const [data,preview]=await Promise.all([
      api('/api/quality/detail',payload),
      api('/api/quality/chart-preview',payload),
    ]);
    if(request!==qualityState.detailRequest || manifest!==qualityManifest() || workspace!==document.body.dataset.workspace)return;
    qualityState.detail={...data,chart_id:chartId};
    const record=data.record||{};
    qualityState.regions=Array.isArray(record.regions)?structuredClone(record.regions):[];
    qualityState.correctedPath=record.corrected_path||'';
    if(element('#qualityCorrectedPath'))element('#qualityCorrectedPath').textContent=qualityState.correctedPath;
    if(element('#qualityDetail'))element('#qualityDetail').hidden=false;
    updateQualityDetail();openScopedQualityPreview(preview,scroll);
  } catch(error) {
    if(request===qualityState.detailRequest)qualityState.previewError=error.message;
    throw error;
  } finally {if(request===qualityState.detailRequest)qualityState.previewLoading=false;updateTrainingPreviewNavigation();}
}

async function openQualityReview(manifest=qualityManifest(),chartId=qualityState.detail?.chart_id,request=qualityState.detailRequest,{scroll=true}={}) {
  if(!chartId)return;
  const workspace=document.body.dataset.workspace;
  const preview=await api('/api/quality/chart-preview',{manifest,chart_id:chartId,auto_scope:qualityState.autoScope || 'formal'});
  if(request===qualityState.detailRequest && manifest===qualityManifest() && workspace===document.body.dataset.workspace)openScopedQualityPreview(preview,scroll);
}
function openScopedQualityPreview(preview, scroll) {
  if (preview.training_context) preview.training_context.audit_read_only = !!activeDatasetItem()?.auditReadOnly;
  globalThis.ChartPreview?.openSource(preview,{scroll});
}
element('#qualityOpenReview')?.addEventListener('click',()=>void openQualityReview().catch(error=>setStatus(error.message,false)));

window.addEventListener('training-source-annotated',event=>{
  const {manifest,chart_id,result}=event.detail;
  if(manifest!==qualityManifest())return;
  if(qualityState.detail?.chart_id===chart_id) {
    const record=result.records?.find(row=>row.chart_id===chart_id);
    if(record) {qualityState.detail.record=record;qualityState.regions=structuredClone(record.regions||[]);updateQualityDetail();}
  }
  qualityState.summary=result.summary;renderQualitySummary();
  const record=result.records?.find(row=>row.chart_id===chart_id);
  const item=qualityState.items.find(row=>row.chart_id===chart_id);
  if(item){item.annotation=record||null;item.label=record?.label||'normal';item.weight=record?.weight??1;
    item.candidate_chart_review=record?.candidate_chart_review;
    item.confirmed_regions=['excluded','negative'].includes(item.label)||!item.weight?0:(record?.regions||[]).filter(row=>row.label==='good').length;
    renderQualityCatalog();}
  if (item?.auto_badge && !item.candidate_status) {
    void loadQualityCatalog(false).catch(error=>setStatus(error.message,false));
    const preview = globalThis.ChartPreview?.getEntry()?.training_context;
    if (preview?.manifest === manifest && preview.chart_id === chart_id) {
      void openQualityReview(manifest,chart_id,qualityState.detailRequest,{scroll:false}).catch(error=>setStatus(error.message,false));
    }
  }
});

async function saveQualitySelection() {
  if (qualityState.selectionLoading || qualityState.saving) throw new Error("请等待当前选择或保存操作完成。");
  const manifest = qualityManifest();
  const request = qualityState.selectionRequest;
  const chartIds = [...qualityState.selected]; if (!chartIds.length) throw new Error("请先选择谱面。");
  const label = valueOrDefault("#qualityBulkLabel", "normal");
  const bulkWeight = Math.max(0, Math.min(4, numberOrDefault("#qualityBulkWeight", QUALITY_DEFAULT_WEIGHTS[label] ?? 1)));
  const groups = new Map();
  chartIds.forEach((id) => {
    const item = qualityState.items.find((candidate) => String(candidate.chart_id || candidate.id) === String(id));
    const weight = Math.max(0, Math.min(4, qualityState.editedWeights.get(id) ?? (item?._weightEdited ? Number(item.weight) : bulkWeight)));
    const key = String(weight); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(id);
  });
  const reason = valueOrDefault("#qualityBulkReason", "").trim();
  qualityState.saving = true; renderQualityCatalog();
  try {
    let data = {};
    for (const [weight, ids] of groups) {
      data = await api("/api/quality/save", { manifest, chart_ids: ids, label, weight: Number(weight), reason });
    }
    if (request !== qualityState.selectionRequest || manifest !== qualityManifest()) return;
    qualityState.annotationPath = data.annotation_path || qualityState.annotationPath;
    qualityState.annotationManifest = manifest;
    qualityState.summary = data.summary || qualityState.summary;
    qualityState.selected.clear(); qualityState.editedWeights.clear();
    renderQualitySummary(); await loadQualityCatalog(false);
  } finally {
    if (request === qualityState.selectionRequest) { qualityState.saving = false; renderQualityCatalog(); }
  }
}

function collectQualityRegions() {
  return [...document.querySelectorAll(".quality-region-row")].map((row) => {
    const value = name => row.querySelector(`[data-region-field="${name}"]`)?.value;
    const entryRaw = row.dataset?.entryStartSec;
    const entryStart = entryRaw !== undefined && entryRaw !== "" && Number.isFinite(Number(entryRaw)) ? Number(entryRaw) : null;
    return {
      start_sec: Number(value("start_sec")),
      end_sec: Number(value("end_sec")),
      ...(entryStart !== null ? { entry_start_sec: entryStart } : {}),
      weight: Math.max(0, Math.min(4, Number(value("weight")))),
      label: value("label") || "uncertain",
      timing_weight: Number(value("timing_weight")),
      component_weight: Number(value("component_weight")),
      reason: value("reason") || "",
    };
  }).filter((item) => Number.isFinite(item.start_sec) && Number.isFinite(item.end_sec) && item.end_sec > item.start_sec);
}

function replaceQualityRegionSpan(regions, replacement) {
  const start=replacement.start_sec,end=replacement.end_sec,result=[];
  for(const region of regions) {
    const a=Number(region.start_sec),b=Number(region.end_sec);
    if(b<=start || a>=end)result.push({...region});
    else {
      if(a<start)result.push({...region,end_sec:start});
      if(b>end)result.push({...region,start_sec:end});
    }
  }
  result.push({...replacement});
  return result.sort((a,b)=>a.start_sec-b.start_sec);
}

async function saveQualityRegions() {
  if (!qualityState.detail?.chart_id) throw new Error("请先打开一张谱面。");
  const zeroDetail = typeof isZeroUnmarkedQualityDetail === "function"
    ? isZeroUnmarkedQualityDetail(qualityState.detail)
    : Boolean(qualityState.detail && (qualityState.detail.zero_unmarked || qualityState.detail.record?.unmarked_weight === 0));
  if (zeroDetail) throw new Error("候选库与示范库请在页面顶部「谱面预览」中确认学习片段，避免旧版表格覆盖零权重默认设置。");
  const chartId=qualityState.detail.chart_id,manifest=qualityManifest(),request=qualityState.detailRequest;
  const record = qualityState.detail.record || {};
  const data = await api("/api/quality/save", { manifest, chart_ids: [chartId], label: record.label || "normal", weight: Number(record.weight ?? 1), reason: record.reason || "", regions: collectQualityRegions(), annotation_scope: 'region' });
  if(chartId!==qualityState.detail?.chart_id || manifest!==qualityManifest() || request!==qualityState.detailRequest)return;
  qualityState.annotationPath = data.annotation_path || qualityState.annotationPath; qualityState.summary = data.summary || qualityState.summary;
  qualityState.annotationManifest = qualityManifest();
  renderQualitySummary(); await loadQualityCatalog(false);
  if(chartId===qualityState.detail?.chart_id && manifest===qualityManifest() && request===qualityState.detailRequest)await openQualityDetail(chartId);
}

async function saveQualityCorrection() {
  if (!qualityState.detail?.chart_id || !qualityState.correctedPath) throw new Error("请先上传修正版 maidata，并选择难度槽。");
  const difficulty = valueOrDefault("#qualityCorrectedDifficulty", ""); if (!difficulty) throw new Error("请选择修正版难度槽。");
  const data = await api("/api/quality/save", { manifest: qualityManifest(), chart_ids: [qualityState.detail.chart_id], label: "negative", weight: 0, reason: "反面案例与修正版配对", corrected_path: qualityState.correctedPath, corrected_difficulty: Number(difficulty) });
  qualityState.annotationPath = data.annotation_path || qualityState.annotationPath; qualityState.summary = data.summary || qualityState.summary;
  qualityState.annotationManifest = qualityManifest();
  renderQualitySummary(); await loadQualityCatalog(false); await openQualityDetail(qualityState.detail.chart_id);
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------
function intensityAudioPath() {
  return valueOrDefault("#audioPath", "").trim();
}

function intensityProfileValid(profile) {
  const points = profile?.points || [];
  const duration = Number(profile?.duration_sec);
  const hash = String(profile?.audio_sha256 || "");
  return profile?.schema === "maichart-intensity-curve-v1" && Array.isArray(points) && points.length >= 2
    && Number.isFinite(duration) && duration > 0 && /^[a-f0-9]{64}$/i.test(hash)
    && points.every((point, index) => Number.isFinite(Number(point.time_sec))
      && Number(point.time_sec) >= 0 && Number(point.time_sec) <= duration
      && Number(point.intensity) >= 0 && Number(point.intensity) <= 1
      && (index === 0 || Number(point.time_sec) > Number(points[index - 1].time_sec)))
    && (profile.components === undefined || (profile.components !== null
      && typeof profile.components === "object" && !Array.isArray(profile.components)
      && Object.values(profile.components).every((values) => Array.isArray(values) && values.length === points.length
        && values.every((value) => Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 1))));
}

function updateIntensityButtons() {
  const hasProfile = intensityProfileValid(intensityCurveState.profile);
  ["#intensityContrastButton", "#applyIntensitySectionButton"]
    .forEach((selector) => { const node = element(selector); if (node) node.disabled = !hasProfile; });
  const save = element("#intensitySaveButton");
  if (save) save.disabled = !hasProfile || intensityCurveState.contextStale === true;
  const undo = element("#intensityUndoButton");
  if (undo) undo.disabled = !hasProfile || intensityCurveState.history.length === 0;
  const reset = element("#intensityResetButton");
  if (reset) reset.disabled = !hasProfile || !intensityProfileValid(intensityCurveState.automatic);
  updateIntensityViewControls();
  updateCurveConditionNotice();
}

function updateCurveConditionNotice() {
  const node = element("#intensityCurveConditionNotice");
  if (!node) return;
  const show = intensityCurveState.conditionChanged && intensityCurveState.profile
    && checkedOrDefault("#intensityCurveEnabled", false);
  node.hidden = !show;
  node.textContent = show ? "等级、Radar 或 Style 已变化。未重新预测时仍沿用原曲线，生成使用当前等级与风格；你也可以按需重新生成曲线。" : "";
}

function setIntensityStatus(message, kind = "") {
  const node = element("#intensityCurveStatus");
  if (node) { node.textContent = message; node.className = `inline-status ${kind}`.trim(); }
}

function setIntensitySong(path) {
  const name = path ? path.split(/[\\/]/).pop() : "尚未选择歌曲";
  if (element("#intensitySongName")) element("#intensitySongName").textContent = name;
  if (element("#intensitySongPath")) element("#intensitySongPath").textContent = path || "";
  const audio = element("#intensityAudio");
  const preview = intensityCurveState.profile?.audio_preview_url;
  if (audio && path && preview && audio.getAttribute("src") !== preview) audio.src = preview;
}

function cloneIntensityProfile(profile) {
  return profile ? JSON.parse(JSON.stringify(profile)) : null;
}

function pushIntensityHistory() {
  if (!intensityCurveState.profile) return;
  intensityCurveState.history.push(cloneIntensityProfile(intensityCurveState.profile));
  if (intensityCurveState.history.length > 50) intensityCurveState.history.shift();
}

function enhanceIntensityContrast() {
  const profile = intensityCurveState.profile;
  if (!intensityProfileValid(profile)) throw new Error("请先生成有效的强度曲线");
  const points = profile.points.map((point) => ({
    ...point,
    intensity: Number(Math.max(0, Math.min(1, 0.5 + 1.25 * (Number(point.intensity) - 0.5))).toFixed(4)),
  }));
  if (points.every((point, index) => point.intensity === Number(profile.points[index].intensity))) {
    setIntensityStatus("曲线已处于基准或上下限，本次没有可扩大的强弱落差。", "warn");
    return;
  }
  pushIntensityHistory();
  profile.points = points;
  profile.edited = true;
  intensityCurveState.saved = false;
  saveIntensityCurveState();
  drawIntensityCurve();
  updateIntensityButtons();
  setIntensityStatus("已增强全曲强弱对比：围绕 0.5 放大差值 1.25 倍。可撤销；确认后请保存并返回生成页。", "good");
}

function resampleIntensityComponents(sourcePoints, components, targetPoints) {
  if (components === undefined) return undefined;
  if (!components || typeof components !== "object" || Array.isArray(components)) throw new Error("曲线基础通道格式无效。");
  const rows = sourcePoints.map((point, index) => ({time: Number(point.time_sec), index}))
    .filter((row) => Number.isFinite(row.time)).sort((a, b) => a.time - b.time);
  // Keep the last point and its channels at duplicate times, matching profile
  // normalization. Components remain baseline predictions: scalar edits do not
  // replace the learned type proportions with the manually chosen intensity.
  const unique = [];
  rows.forEach((row) => {
    if (unique.length && unique[unique.length - 1].time === row.time) unique[unique.length - 1] = row;
    else unique.push(row);
  });
  return Object.fromEntries(Object.entries(components).map(([key, values]) => {
    if (!Array.isArray(values) || values.length !== sourcePoints.length
        || values.some((value) => !Number.isFinite(Number(value)) || Number(value) < 0 || Number(value) > 1)
        || !unique.length) throw new Error(`曲线基础通道 ${key} 与控制点不一致，请重新生成曲线。`);
    let right = 0;
    const sampled = targetPoints.map((point) => {
      const time = Number(point.time_sec);
      while (right < unique.length - 1 && unique[right].time < time) right++;
      if (right === 0) return Number(values[unique[0].index]);
      if (time >= unique[unique.length - 1].time) return Number(values[unique[unique.length - 1].index]);
      const left = right - 1;
      const mix = (time - unique[left].time) / (unique[right].time - unique[left].time);
      return Number(values[unique[left].index]) * (1 - mix) + Number(values[unique[right].index]) * mix;
    });
    return [key, sampled];
  }));
}

function normalizeIntensityProfile(profile) {
  if (!profile || !Array.isArray(profile.points)) return null;
  const sorted = profile.points.map((point) => ({
    time_sec: Number(point.time_sec),
    intensity: Math.max(0, Math.min(1, Number(point.intensity ?? point.value ?? 0.5))),
  })).filter((point) => Number.isFinite(point.time_sec)).sort((a, b) => a.time_sec - b.time_sec);
  const points = [];
  sorted.forEach((point) => {
    if (points.length && points[points.length - 1].time_sec === point.time_sec) points[points.length - 1] = point;
    else points.push(point);
  });
  const components = resampleIntensityComponents(profile.points, profile.components, points);
  const duration_sec = Number(profile.duration_sec || 0);
  return { ...profile, schema: "maichart-intensity-curve-v1", duration_sec, points,
    ...(components !== undefined ? {components} : {}),
    rest_regions: normalizeRestRegions(profile.rest_regions, duration_sec) };
}

function normalizeRestRegions(regions, duration) {
  const safeDuration = Math.max(0, Number(duration) || 0);
  const sorted = (Array.isArray(regions) ? regions : []).map((region) => ({
    start_sec: Math.max(0, Math.min(safeDuration, Number(region?.start_sec))),
    end_sec: Math.max(0, Math.min(safeDuration, Number(region?.end_sec))),
  })).filter((region) => Number.isFinite(region.start_sec) && Number.isFinite(region.end_sec)
    && region.end_sec > region.start_sec).sort((a, b) => a.start_sec - b.start_sec);
  const merged = [];
  sorted.forEach((region) => {
    const last = merged[merged.length - 1];
    if (last && region.start_sec <= last.end_sec) last.end_sec = Math.max(last.end_sec, region.end_sec);
    else merged.push({ ...region });
  });
  return merged;
}

function updateRestRegion(start, end, add) {
  const profile = intensityCurveState.profile;
  if (!profile) return;
  const duration = Number(profile.duration_sec || 0);
  start = Math.max(0, Math.min(duration, Number(start)));
  end = Math.max(0, Math.min(duration, Number(end)));
  if (!Number.isFinite(start) || !Number.isFinite(end) || !(end > start)) throw new Error("请框选至少两个控制点，或填写有效的开始与结束时间；结束时间必须大于开始时间");
  pushIntensityHistory();
  const current = normalizeRestRegions(profile.rest_regions, duration);
  if (add) current.push({ start_sec: start, end_sec: end });
  else {
    const remaining = [];
    current.forEach((region) => {
      if (region.end_sec <= start || region.start_sec >= end) remaining.push(region);
      else {
        if (region.start_sec < start) remaining.push({ start_sec: region.start_sec, end_sec: start });
        if (region.end_sec > end) remaining.push({ start_sec: end, end_sec: region.end_sec });
      }
    });
    profile.rest_regions = remaining;
  }
  if (add) profile.rest_regions = normalizeRestRegions(current, duration);
  profile.edited = true; intensityCurveState.saved = false; saveIntensityCurveState(); drawIntensityCurve(); updateIntensityButtons();
  setIntensityStatus(`${add ? "已标记" : "已取消"}休止段 ${formatIntensityTime(start, true)}–${formatIntensityTime(end, true)}（半开区间）`, "good");
}

function applyIntensityRest(add) {
  try {
    updateRestRegion(valueOrDefault("#intensitySectionStart", 0), valueOrDefault("#intensitySectionEnd", 0), add);
  } catch (error) {
    setIntensityStatus(error.message || String(error), "warn");
  }
}

function curveCanvasGeometry(canvas, profile) {
  const width = canvas.clientWidth || 1200;
  const height = Math.max(320, Math.round(width * 0.38));
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pad = { left: 48, right: 18, top: 22, bottom: 36 };
  const duration = Math.max(0.001, Number(profile?.duration_sec || 1));
  const end = Math.max(0.001, Math.min(duration, intensityView.end ?? duration));
  const start = Math.max(0, Math.min(end - 0.001, intensityView.start));
  return { ctx, width, height, pad, plotWidth: width - pad.left - pad.right, plotHeight: height - pad.top - pad.bottom,
    duration, start, end, span: end - start };
}

function resetIntensityView() {
  intensityView.start = 0; intensityView.end = null; intensityView.selected.clear(); intensityView.gesture = null;
  intensityView.spectrogram = null; intensityView.specSource = null;
  intensityView.sectionSelectionKey = "";
}

function updateIntensityViewControls() {
  const profile = intensityCurveState.profile;
  if (!profile) {
    ["#markIntensityRestButton", "#clearIntensityRestButton"].forEach(selector => { if (element(selector)) element(selector).disabled = true; });
    if (element("#intensityRestCount")) element("#intensityRestCount").textContent = "0";
    updateIntensityRestList([]);
    return;
  }
  const duration = Number(profile.duration_sec), end = intensityView.end ?? duration;
  const span = Math.max(0.001, end - intensityView.start), zoom = duration / span;
  if (element("#intensityZoom")) element("#intensityZoom").value = zoom;
  if (element("#intensityZoomLabel")) element("#intensityZoomLabel").textContent = `${zoom.toFixed(zoom < 10 ? 1 : 0)}×`;
  const pan = element("#intensityPan");
  if (pan) { pan.disabled = zoom <= 1.001; pan.value = duration > span ? intensityView.start / (duration - span) * 1000 : 0; }
  if (element("#intensityVisibleRange")) element("#intensityVisibleRange").textContent = `${formatIntensityTime(intensityView.start, true)}–${formatIntensityTime(end, true)}`;
  intensityView.selected = new Set([...intensityView.selected].filter(index => index >= 0 && index < profile.points.length));
  const count = intensityView.selected.size;
  const selectedTimes = [...intensityView.selected].map(index => Number(profile.points[index].time_sec)).sort((a, b) => a - b);
  const selectionKey = selectedTimes.length >= 2 ? `${selectedTimes[0]}:${selectedTimes[selectedTimes.length - 1]}` : "";
  if (selectionKey && selectionKey !== intensityView.sectionSelectionKey) {
    element("#intensitySectionStart").value = selectedTimes[0];
    element("#intensitySectionEnd").value = selectedTimes[selectedTimes.length - 1];
  }
  intensityView.sectionSelectionKey = selectionKey;
  if (element("#intensitySelectionCount")) element("#intensitySelectionCount").textContent = count ? `已选择 ${count} 个控制点` : "未选择控制点";
  ["#intensityFitSelection", "#intensitySelectionDown", "#intensitySelectionUp", "#intensitySelectionSet"]
    .forEach(selector => { if (element(selector)) element(selector).disabled = !count; });
  ["#markIntensityRestButton", "#clearIntensityRestButton"].forEach(selector => { if (element(selector)) element(selector).disabled = !profile; });
  if (element("#intensityRestCount")) element("#intensityRestCount").textContent = String(normalizeRestRegions(profile.rest_regions, duration).length);
  updateIntensityRestList(normalizeRestRegions(profile.rest_regions, duration));
}

function updateIntensityRestList(regions) {
  const container = element("#intensityRestRegions");
  if (!container) return;
  const key = JSON.stringify(regions);
  if (container.dataset.regions === key) return;
  container.dataset.regions = key;
  container.replaceChildren();
  if (!regions.length) {
    const empty = document.createElement("span"); empty.className = "intensity-rest-empty";
    empty.textContent = "尚未标记休止区间。"; container.append(empty); return;
  }
  regions.forEach((region, index) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "intensity-rest-region";
    button.textContent = `休止 ${index + 1} · ${formatIntensityTime(region.start_sec, true)}–${formatIntensityTime(region.end_sec, true)}`;
    button.title = "点击放大查看该休止区，并填入下方开始与结束时间";
    button.addEventListener("click", () => {
      element("#intensitySectionStart").value = region.start_sec;
      element("#intensitySectionEnd").value = region.end_sec;
      const margin = Math.max(.5, (region.end_sec - region.start_sec) * .15);
      setIntensityViewport(region.start_sec - margin, region.end_sec - region.start_sec + margin * 2);
      element("#intensityCurveCanvas").scrollIntoView({ block: "nearest" });
    });
    container.append(button);
  });
}

function setIntensityViewport(start, span) {
  const duration = Number(intensityCurveState.profile?.duration_sec); if (!(duration > 0)) return;
  span = Math.max(duration / 64, Math.min(duration, span));
  intensityView.start = Math.max(0, Math.min(duration - span, start));
  intensityView.end = intensityView.start + span; drawIntensityCurve();
}

function zoomIntensityView(zoom, anchorSec = null, anchorRatio = 0.5) {
  const profile = intensityCurveState.profile; if (!profile) return;
  const end = intensityView.end ?? profile.duration_sec, span = end - intensityView.start;
  const anchor = anchorSec ?? (intensityView.start + span * anchorRatio);
  const newSpan = profile.duration_sec / Math.max(1, Math.min(64, Number(zoom)));
  setIntensityViewport(anchor - newSpan * anchorRatio, newSpan);
}

function fitIntensitySelection() {
  if (!intensityView.selected.size) return;
  const points = [...intensityView.selected].map(index => intensityCurveState.profile.points[index]);
  const low = Math.min(...points.map(point => point.time_sec)), high = Math.max(...points.map(point => point.time_sec));
  const padding = Math.max(0.5, (high - low) * 0.15);
  setIntensityViewport(low - padding, high - low + 2 * padding);
}

function intensitySpectrogramCanvas(spec) {
  if (!spec?.values?.length || !spec.times?.length) return null;
  if (intensityView.specSource === spec) return intensityView.spectrogram;
  const rows = spec.values.length, cols = spec.times.length, raster = document.createElement("canvas");
  raster.width = cols; raster.height = rows; const ctx = raster.getContext("2d");
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const norm = Math.max(0, Math.min(1, Number(spec.values[y]?.[x] ?? 0) / 255));
    ctx.fillStyle = `hsla(${210 - norm * 190},85%,${18 + norm * 42}%,${0.18 + norm * 0.42})`;
    ctx.fillRect(x, rows - 1 - y, 1, 1);
  }
  intensityView.specSource = spec; intensityView.spectrogram = raster; return raster;
}

function drawIntensityCurve() {
  const canvas = element("#intensityCurveCanvas");
  const profile = intensityCurveState.profile;
  if (!canvas?.getContext) return;
  const geo = curveCanvasGeometry(canvas, profile); const { ctx, width, height, pad, plotWidth, plotHeight, duration, start, end, span } = geo;
  ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#171c20"; ctx.fillRect(0, 0, width, height);
  if (!profile) return;
  const spec = profile.spectrogram;
  const raster = intensitySpectrogramCanvas(spec);
  if (raster) ctx.drawImage(raster, start / duration * raster.width, 0, span / duration * raster.width, raster.height, pad.left, pad.top, plotWidth, plotHeight);
  ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1; ctx.fillStyle = "rgba(255,255,255,.75)"; ctx.font = "11px system-ui";
  for (let i = 0; i <= 4; i++) { const y = pad.top + (i / 4) * plotHeight; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotWidth, y); ctx.stroke(); ctx.fillText((1 - i / 4).toFixed(2), 8, y + 4); }
  if (spec?.frequencies_hz?.length) {
    [0, 0.5, 1].forEach((ratio) => { const y = pad.top + (1 - ratio) * plotHeight; const idx = Math.round(ratio * (spec.frequencies_hz.length - 1)); ctx.fillText(`${Math.round(Number(spec.frequencies_hz[idx]))}Hz`, width - 52, y + 4); });
  }
  const points = profile.points || [];
  const xAt = time => pad.left + (time - start) / span * plotWidth;
  ctx.save(); ctx.beginPath(); ctx.rect(pad.left, pad.top, plotWidth, plotHeight); ctx.clip();
  const visibleRests = normalizeRestRegions(profile.rest_regions, duration).filter(region => region.end_sec > start && region.start_sec < end);
  visibleRests.forEach((region) => {
    const left = Math.max(start, region.start_sec), right = Math.min(end, region.end_sec);
    const x = xAt(left), regionWidth = xAt(right) - x;
    ctx.save(); ctx.beginPath(); ctx.rect(x, pad.top, regionWidth, plotHeight); ctx.clip();
    ctx.fillStyle = "rgba(214, 44, 65, .38)"; ctx.fillRect(x, pad.top, regionWidth, plotHeight);
    ctx.strokeStyle = "rgba(255, 173, 183, .48)"; ctx.lineWidth = 1;
    for (let offset = -plotHeight; offset < regionWidth; offset += 14) {
      ctx.beginPath(); ctx.moveTo(x + offset, pad.top + plotHeight); ctx.lineTo(x + offset + plotHeight, pad.top); ctx.stroke();
    }
    ctx.fillStyle = "#f16b79"; ctx.fillRect(x, pad.top + plotHeight - 7, Math.max(regionWidth, 2), 7);
    ctx.restore();
  });
  if (points.length) {
    ctx.strokeStyle = "#ffd35a"; ctx.lineWidth = 3; ctx.beginPath();
    points.forEach((point, index) => { const x = xAt(point.time_sec); const y = pad.top + (1 - point.intensity) * plotHeight; if (!index) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    points.forEach((point, index) => {
      if (point.time_sec < start || point.time_sec > end) return;
      const x = xAt(point.time_sec), y = pad.top + (1 - point.intensity) * plotHeight, selected = intensityView.selected.has(index);
      ctx.fillStyle = selected ? "#5be1f5" : "#ff6d55"; ctx.beginPath(); ctx.arc(x, y, selected ? 6 : 4, 0, Math.PI * 2); ctx.fill();
      if (selected) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); }
    });
  }
  // Draw rest borders and labels above the curve/spectrogram so neither can
  // hide the annotation. Narrow regions retain a bright boundary and band;
  // their exact time range is always available in the list below the chart.
  visibleRests.forEach((region) => {
    const left = Math.max(start, region.start_sec), right = Math.min(end, region.end_sec);
    const x = xAt(left), regionWidth = xAt(right) - x;
    ctx.strokeStyle = "#ff9aa6"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
    [region.start_sec, region.end_sec].filter(time => time >= start && time <= end).forEach(time => {
      ctx.beginPath(); ctx.moveTo(xAt(time), pad.top); ctx.lineTo(xAt(time), pad.top + plotHeight); ctx.stroke();
    }); ctx.setLineDash([]);
    ctx.font = "bold 12px system-ui";
    const fullLabel = `休止 ${formatIntensityTime(region.start_sec, true)}–${formatIntensityTime(region.end_sec, true)}`;
    const label = ctx.measureText(fullLabel).width + 12 <= regionWidth ? fullLabel : "休止";
    const labelWidth = ctx.measureText(label).width + 12;
    if (labelWidth <= regionWidth) {
      const labelX = x + (regionWidth - labelWidth) / 2;
      ctx.fillStyle = "#862033"; ctx.fillRect(labelX, pad.top + 5, labelWidth, 22);
      ctx.fillStyle = "#fff"; ctx.fillText(label, labelX + 6, pad.top + 20);
    }
  });
  const gesture = intensityView.gesture;
  if (gesture?.kind === "select") {
    const x = Math.min(gesture.x, gesture.toX), y = Math.min(gesture.y, gesture.toY);
    ctx.fillStyle = "rgba(91,225,245,.16)"; ctx.fillRect(x, y, Math.abs(gesture.toX - gesture.x), Math.abs(gesture.toY - gesture.y));
    ctx.strokeStyle = "#5be1f5"; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.strokeRect(x, y, Math.abs(gesture.toX - gesture.x), Math.abs(gesture.toY - gesture.y)); ctx.setLineDash([]);
  }
  const playback = Number(element("#intensityAudio")?.currentTime || 0);
  if (playback >= start && playback <= end) { ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(xAt(playback), pad.top); ctx.lineTo(xAt(playback), pad.top + plotHeight); ctx.stroke(); }
  ctx.restore();
  ctx.fillStyle = "rgba(255,255,255,.82)";
  const rough = span / 6, magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 0.001)));
  const step = [1, 2, 5, 10].find(value => value * magnitude >= rough) * magnitude;
  for (let sec = Math.ceil(start / step) * step; sec <= end + 1e-8; sec += step) ctx.fillText(formatIntensityTime(sec, step < 1), xAt(sec) - 12, height - 12);
  updateIntensityViewControls();
}

function formatIntensityTime(sec, precise = false) { const total = Math.round(Math.max(0, Number(sec)) * (precise ? 10 : 1)) / (precise ? 10 : 1); return `${Math.floor(total / 60)}:${(total % 60).toFixed(precise ? 1 : 0).padStart(precise ? 4 : 2, "0")}`; }

function intensityEventCoordinates(event) {
  const canvas = element("#intensityCurveCanvas"), rect = canvas.getBoundingClientRect();
  const geo = curveCanvasGeometry(canvas, intensityCurveState.profile);
  return { ...geo, x: (event.clientX - rect.left) * geo.width / rect.width, y: (event.clientY - rect.top) * geo.height / rect.height };
}

function intensityPointAt(event) {
  const canvas = element("#intensityCurveCanvas"), profile = intensityCurveState.profile; if (!canvas || !profile?.points?.length) return -1;
  const { x, y, pad, plotWidth, plotHeight, start, end, span } = intensityEventCoordinates(event);
  if (x < pad.left || x > pad.left + plotWidth || y < pad.top || y > pad.top + plotHeight) return -1;
  let best = -1, distance = Infinity;
  profile.points.forEach((point, index) => { if (point.time_sec < start || point.time_sec > end) return; const px = pad.left + (point.time_sec - start) / span * plotWidth, py = pad.top + (1 - point.intensity) * plotHeight; const d = Math.hypot(px - x, py - y); if (d < distance) { distance = d; best = index; } });
  return distance <= 12 ? best : -1;
}

function updateIntensityPoint(event) {
  const profile = intensityCurveState.profile, gesture = intensityView.gesture; if (!profile || !gesture) return;
  const geo = intensityEventCoordinates(event);
  if (gesture.kind === "select") {
    gesture.toX = Math.max(geo.pad.left, Math.min(geo.pad.left + geo.plotWidth, geo.x));
    gesture.toY = Math.max(geo.pad.top, Math.min(geo.pad.top + geo.plotHeight, geo.y));
    const lowX = Math.min(gesture.x, gesture.toX), highX = Math.max(gesture.x, gesture.toX), lowY = Math.min(gesture.y, gesture.toY), highY = Math.max(gesture.y, gesture.toY);
    intensityView.selected = new Set(gesture.previous);
    profile.points.forEach((point, index) => {
      const x = geo.pad.left + (point.time_sec - geo.start) / geo.span * geo.plotWidth, y = geo.pad.top + (1 - point.intensity) * geo.plotHeight;
      if (point.time_sec >= geo.start && point.time_sec <= geo.end && x >= lowX && x <= highX && y >= lowY && y <= highY) intensityView.selected.add(index);
    });
  } else if (gesture.kind === "drag") {
    const values = [...gesture.values.values()], low = -Math.min(...values), high = 1 - Math.max(...values);
    const delta = Math.max(low, Math.min(high, (gesture.y - geo.y) / geo.plotHeight));
    if (!gesture.changed && Math.abs(delta) > 1e-6) { pushIntensityHistory(); gesture.changed = true; intensityCurveState.saved = false; }
    if (gesture.changed) { gesture.values.forEach((value, index) => { profile.points[index].intensity = Number((value + delta).toFixed(4)); }); profile.edited = true; }
    if (element("#intensityPointReadout")) element("#intensityPointReadout").textContent = `${gesture.values.size} 点 · 整体${delta >= 0 ? "提高" : "降低"} ${Math.abs(delta).toFixed(2)}`;
  }
  drawIntensityCurve();
}

function adjustIntensitySelection(delta = null) {
  const profile = intensityCurveState.profile; if (!profile || !intensityView.selected.size) return;
  const values = [...intensityView.selected].map(index => Number(profile.points[index].intensity));
  const target = Number(valueOrDefault("#intensitySelectionValue", 0.3));
  if (delta === null && (!Number.isFinite(target) || target < 0 || target > 1)) throw new Error("选中点目标强度必须在 0–1 之间");
  const shift = delta === null ? null : Math.max(-Math.min(...values), Math.min(1 - Math.max(...values), delta));
  if (shift === 0) return;
  pushIntensityHistory();
  intensityView.selected.forEach(index => { profile.points[index].intensity = Number((shift === null ? target : profile.points[index].intensity + shift).toFixed(4)); });
  profile.edited = true; intensityCurveState.saved = false; saveIntensityCurveState(); drawIntensityCurve();
}

function showIntensityProfile(profile, audioPreviewUrl = null) {
  const normalized = normalizeIntensityProfile(profile); if (!intensityProfileValid(normalized)) throw new Error("后端返回的强度曲线格式不完整");
  intensityCurveState.profile = normalized; intensityCurveState.automatic = cloneIntensityProfile(normalized); intensityCurveState.history = []; intensityCurveState.saved = false; intensityCurveState.contextStale = false; intensityCurveState.conditionChanged = false;
  intensityCurveState.contextStaleReason = null;
  resetIntensityView();
  if (audioPreviewUrl) intensityCurveState.profile.audio_preview_url = audioPreviewUrl;
  if (audioPreviewUrl) intensityCurveState.automatic.audio_preview_url = audioPreviewUrl;
  intensityCurveState.path = intensityAudioPath(); intensityCurveState.hash = normalized.audio_sha256 || "";
  setIntensitySong(intensityCurveState.path); drawIntensityCurve(); updateIntensityButtons(); saveIntensityCurveState();
}

async function pollIntensityTask(taskId, sourcePath) {
  try {
    const task = await api(`/api/task?id=${encodeURIComponent(taskId)}`);
    if (intensityCurveState.taskId !== taskId) return;
    if (["queued", "running", "cancelling"].includes(task.status)) { intensityCurveState.timer = setTimeout(() => pollIntensityTask(taskId, sourcePath), 800); return; }
    if (task.status !== "completed") throw new Error(task.error || "强度曲线分析失败");
    if (intensityAudioPath() !== sourcePath) { setIntensityStatus("分析已完成，但歌曲已更换；已丢弃旧曲线，请重新生成。", "warn"); return; }
    const result = task.result || {}; const profile = result.intensity_profile || result.profile;
    if (result.intensity_profile_path) profile.analysis_profile_path = result.intensity_profile_path;
    showIntensityProfile(profile, result.audio_preview_url || profile?.audio_preview_url || null);
    // A completed new analysis is fresher than an earlier checkpoint
    // inspection. Do not compare it with that inspection's old epoch/step.
    intensityCurveState.checkpointRevision = null;
    setIntensityStatus("自动强度曲线已生成。拖动控制点或按时间段调整后保存。", "good");
    invalidateLearnedCurveForCheckpoint(profile.checkpoint_identity || null);
  } catch (error) { setIntensityStatus(`分析失败：${error.message || error}`, "warn"); }
}

async function generateIntensityCurve() {
  const audio = intensityAudioPath(); if (!audio) throw new Error("请先在生成页导入音频或填写音频路径");
  setWorkspaceMode("intensity"); setIntensitySong(audio); setIntensityStatus("正在分析节奏、频谱、和声变化与响度…");
  const status = await api("/api/status");
  if (status.intensity_curve_capability !== true) throw new Error("当前后端尚未加载多维强度曲线分析能力，请重启 WebUI 服务后再试");
  intensityCurveState.projectRoot = String(status.project_root || intensityCurveState.projectRoot || "");
  const generation = generationPayload();
  const family = generation.model_family || currentGenerationFamily();
  const curveSource = valueOrDefault("#intensityCurveSource", "auto");
  const data = await api("/api/intensity/analyze", {
    audio, resolution_sec: 2, local_context_sec: 12,
    curve_source: curveSource,
    model_family: family,
    checkpoint: curveSource === "audio" ? null : generation.checkpoint || null,
    condition: generation.condition, condition_file: generation.condition_file,
    level: generation.level, bpm: generation.bpm,
    first: generation.first, tempo_mode: generation.tempo_mode,
    tempo_detection:generation.tempo_detection,cache_dir:generation.cache_dir,
    tempo_map: generation.tempo_map,
    meter_map: generation.meter_map,
  });
  if (intensityCurveState.timer) clearTimeout(intensityCurveState.timer);
  intensityCurveState.taskId = data.task_id; await pollIntensityTask(data.task_id, audio);
}

function applyIntensitySection() {
  const profile = intensityCurveState.profile; if (!profile) return;
  let start = Number(valueOrDefault("#intensitySectionStart", 0));
  let end = Number(valueOrDefault("#intensitySectionEnd", profile.duration_sec));
  const target = Math.max(0, Math.min(1, Number(valueOrDefault("#intensitySectionValue", 0.3))));
  const blend = Math.max(0, Number(valueOrDefault("#intensitySectionBlend", 1)));
  if (![start, end, target, blend].every(Number.isFinite)) throw new Error("时间和强度必须填写有效数字");
  start = Math.max(0, Math.min(Number(profile.duration_sec), start)); end = Math.max(0, Math.min(Number(profile.duration_sec), end));
  if (!(end > start)) throw new Error("结束时间必须大于开始时间");
  if (profile.points.length > 2044) throw new Error("当前曲线点过多，无法安全插入段落边界");
  const source = profile.points.map((point) => ({ ...point }));
  const at = (time) => {
    if (time <= source[0].time_sec) return Number(source[0].intensity);
    if (time >= source[source.length - 1].time_sec) return Number(source[source.length - 1].intensity);
    let right = source.findIndex((point) => point.time_sec >= time); const left = right - 1;
    const mix = (time - source[left].time_sec) / (source[right].time_sec - source[left].time_sec);
    return Number(source[left].intensity) * (1 - mix) + Number(source[right].intensity) * mix;
  };
  const boundaryTimes = [start, end];
  if (blend > 0) boundaryTimes.push(Math.max(0, start - blend), Math.min(Number(profile.duration_sec), end + blend));
  const times = [...new Set([...source.map((point) => point.time_sec), ...boundaryTimes])].sort((a, b) => a - b);
  if (times.length > 2048) throw new Error("应用段落后曲线点将超过 2048 个");
  const points = times.map((time_sec) => {
    const original = at(time_sec);
    let weight = time_sec >= start && time_sec <= end ? 1 : 0;
    if (blend > 0 && time_sec < start) weight = Math.max(0, Math.min(1, (time_sec - (start - blend)) / blend));
    if (blend > 0 && time_sec > end) weight = Math.max(0, Math.min(1, ((end + blend) - time_sec) / blend));
    return { time_sec, intensity: Number((original * (1 - weight) + target * weight).toFixed(4)) };
  });
  const components = resampleIntensityComponents(source, profile.components, points);
  pushIntensityHistory(); intensityCurveState.saved = false;
  profile.points = points;
  if (components !== undefined) profile.components = components;
  profile.edited = true; intensityView.selected.clear(); saveIntensityCurveState(); drawIntensityCurve(); updateIntensityButtons();
}

function saveIntensityCurveAndReturn() {
  if (!intensityProfileValid(intensityCurveState.profile)) throw new Error("请先生成强度曲线");
  invalidateLearnedCurveForCheckpoint();
  if (intensityCurveState.contextStale) throw new Error(intensityCurveState.contextStaleReason === "checkpoint"
    ? "检查点已变化，请重新生成自动曲线后再保存" : "节拍设置已变化，请重新生成模型来源曲线后再保存");
  // Saving acknowledges the current curve; only actual editing operations
  // turn an automatic preview into an explicit generation instruction.
  intensityCurveState.saved = true; saveIntensityCurveState();
  const checkbox = element("#intensityCurveEnabled"); if (checkbox) checkbox.checked = true;
  setWorkspaceMode("generation"); setIntensityStatus(intensityCurveState.profile.edited === true
    ? "曲线已保存；生成页将使用这份编辑曲线。" : "自动曲线已保存；保留自动预览身份，实际调整后才作为编辑曲线使用。", "good"); updateCurveConditionNotice(); saveSettings();
}

function undoIntensityCurve() {
  const previous = intensityCurveState.history.pop(); if (!previous) return;
  intensityCurveState.profile = previous; intensityCurveState.saved = false; intensityView.selected.clear(); saveIntensityCurveState(); drawIntensityCurve(); updateIntensityButtons();
}

function resetIntensityCurve() {
  if (!intensityCurveState.automatic) return;
  pushIntensityHistory(); intensityCurveState.profile = cloneIntensityProfile(intensityCurveState.automatic); intensityCurveState.profile.edited = false; intensityCurveState.saved = false; intensityView.selected.clear(); saveIntensityCurveState(); setIntensitySong(intensityCurveState.path); drawIntensityCurve(); updateIntensityButtons();
}

function invalidateIntensityForAudio() {
  const path = intensityAudioPath();
  if (intensityCurveState.profile && intensityCurveState.path !== path) {
    intensityCurveState.profile = null; intensityCurveState.automatic = null; intensityCurveState.history = []; intensityCurveState.saved = false; intensityCurveState.contextStale = false; intensityCurveState.conditionChanged = false;
    resetIntensityView();
    setIntensitySong(path); drawIntensityCurve(); updateIntensityButtons();
    const audio = element("#intensityAudio"); if (audio) { audio.pause(); audio.removeAttribute("src"); audio.load(); }
    saveIntensityCurveState(); setIntensityStatus("音频已变化，旧曲线已失效，请重新生成。", "warn");
  } else setIntensitySong(path);
}

function invalidateLearnedCurveForTiming() {
  if (typeof v2CurveInvalidateTiming === "function") v2CurveInvalidateTiming();
  const profile = intensityCurveState.profile;
  if (!profile || !/^(learned_music_structure_|learned_song_form_|learned_chart_stream_|learned_chart_composer_|learned_audio_grounded_|learned_audio_workload_|learned_music_context_)/.test(String(profile.method || ""))) return;
  intensityCurveState.saved = false;
  intensityCurveState.contextStale = true;
  intensityCurveState.contextStaleReason = "timing";
  saveIntensityCurveState();
  updateIntensityButtons();
  setIntensityStatus("节拍设置已变化，模型来源曲线需要重新生成。", "warn");
}

function normalizeAutomaticCurveEditFlag(profile, chordMultiplier = 1) {
  if (profile?.edited !== true || !/^(learned_|audio_reference_)/.test(String(profile.method || ""))) return profile;
  const points = profile.points, reference = profile.reference_points;
  if (!Array.isArray(points) || !points.length || !Array.isArray(reference) || points.length !== reference.length) return profile;
  if (profile.rest_regions != null && (!Array.isArray(profile.rest_regions) || profile.rest_regions.length)) return profile;
  if (Number(profile.chord_multiplier ?? 1) !== 1 || Number(chordMultiplier) !== 1) return profile;
  if (points.every((point, index) => Number(point.time_sec) === Number(reference[index].time_sec)
      && Number(point.intensity) === Number(reference[index].intensity))) profile.edited = false;
  return profile;
}

function invalidateLearnedCurveForCheckpoint(identity = null) {
  const profile = normalizeAutomaticCurveEditFlag(intensityCurveState.profile,
    Number(valueOrDefault("#composerChordMultiplier", 1)));
  if (!profile?.checkpoint || !(String(profile.method || "").startsWith("learned_") || profile.model_prediction_available === false)) return;
  // The backend resolves relative paths against PROJECT_ROOT. Comparing
  // that absolute result directly with the input made every fresh relative
  // checkpoint curve immediately stale. Resolve the same lexical path here;
  // never substitute a basename (two directories may hold different models).
  const normalize = value => {
    let path = String(value || "").trim().replaceAll("\\", "/");
    let absolute = /^(?:[a-z]:\/|\/)/i.test(path);
    const root = String(intensityCurveState.projectRoot || "").trim().replaceAll("\\", "/");
    if (path && !absolute && root && !path.startsWith("~")) {
      path = `${root.replace(/\/$/, "")}/${path}`;
      absolute = /^(?:[a-z]:\/|\/)/i.test(path);
    }
    const prefix = path.startsWith("//") ? "//" : path.match(/^[a-z]:\//i)?.[0] || (path.startsWith("/") ? "/" : "");
    const parts = [];
    for (const part of path.slice(prefix.length).split("/")) {
      if (!part || part === ".") continue;
      if (part === ".." && parts.length && parts.at(-1) !== "..") parts.pop();
      else if (part !== ".." || !absolute) parts.push(part);
    }
    return { path: (prefix + parts.join("/")).toLowerCase(), absolute };
  };
  const selected = valueOrDefault("#checkpointPath", "");
  const recordedPath = normalize(profile.checkpoint), currentPath = normalize(selected);
  // Before the initial /api/status returns, an absolute/relative pair is
  // unresolved rather than evidence of a changed model.
  const pathKnown = recordedPath.absolute === currentPath.absolute || !currentPath.path;
  const differentPath = pathKnown && recordedPath.path !== currentPath.path;
  const recorded = profile.checkpoint_identity || {};
  const revision = identity?.checkpoint_identity || identity;
  const number = value => value !== null && value !== undefined && typeof value !== "boolean"
    && !(typeof value === "string" && value.trim() === "")
    && Number.isFinite(Number(value)) ? Number(value) : null;
  if (pathKnown && !differentPath && revision && ["epoch", "optimizer_steps"].some(key => number(revision[key]) !== null)) {
    intensityCurveState.checkpointRevision = { path: selected,
      ...Object.fromEntries(["epoch", "optimizer_steps"].filter(key => number(revision[key]) !== null)
        .map(key => [key, number(revision[key])])) };
  }
  const checked = intensityCurveState.checkpointRevision;
  const current = checked && normalize(checked.path).path === currentPath.path ? checked : {};
  const differentRevision = pathKnown && !differentPath && ["epoch", "optimizer_steps"].some(key => number(recorded[key]) !== null
    && number(current[key]) !== null && number(recorded[key]) !== number(current[key]));
  const clearCheckpointStale = () => {
    if (intensityCurveState.contextStaleReason !== "checkpoint") return false;
    intensityCurveState.contextStale = false; intensityCurveState.contextStaleReason = null;
    return true;
  };
  if (!differentPath && !differentRevision) {
    if (pathKnown && clearCheckpointStale()) {
      saveIntensityCurveState(); updateIntensityButtons();
      setIntensityStatus("当前检查点与曲线一致，可以保存使用。", "good");
    }
    return;
  }
  if (profile.edited === true) {
    // Deliberate edits remain an arrangement request, including when their
    // original automatic prediction came from another checkpoint.
    intensityCurveState.conditionChanged = true;
    clearCheckpointStale();
    if (!intensityCurveState.contextStale) setIntensityStatus("检查点已变化；保留已编辑曲线，新模型将按当前条件解释这份编排要求。", "warn");
  } else {
    intensityCurveState.saved = false;
    // A checkpoint warning must not erase an independently stale musical
    // clock. Returning to the old checkpoint cannot validate a BPM edit.
    if (!intensityCurveState.contextStale || !intensityCurveState.contextStaleReason) {
      intensityCurveState.contextStaleReason = "checkpoint";
    }
    intensityCurveState.contextStale = true;
    setIntensityStatus("检查点已变化，旧自动曲线已停用；请重新生成当前模型的自动曲线。", "warn");
  }
  saveIntensityCurveState(); updateIntensityButtons();
}

function invalidateLearnedCurveForCondition() {
  const profile = intensityCurveState.profile;
  if (!profile || !/^(learned_music_structure_|learned_song_form_|learned_chart_stream_|learned_chart_composer_|learned_audio_grounded_|learned_audio_workload_|learned_music_context_)/.test(String(profile.method || ""))) return;
  intensityCurveState.conditionChanged = true;
  saveIntensityCurveState();
  updateIntensityButtons();
  if (!intensityCurveState.contextStale) {
    setIntensityStatus("等级、Radar 或 Style 已变化；可继续使用原曲线，或按需重新生成。", "warn");
  }
}

async function restoreIntensityBackdrop() {
  const profile = intensityCurveState.profile;
  if (!profile || profile.spectrogram || !profile.analysis_profile_path) return;
  try {
    const original = await api(`/api/download?path=${encodeURIComponent(profile.analysis_profile_path)}`);
    if (intensityCurveState.profile !== profile || original.audio_sha256 !== profile.audio_sha256) return;
    profile.spectrogram = original.spectrogram;
    if (intensityCurveState.automatic) intensityCurveState.automatic.spectrogram = original.spectrogram;
    drawIntensityCurve();
  } catch (_) { setIntensityStatus("已恢复编辑曲线；频谱文件不可用，可重新分析当前歌曲。", "warn"); }
}

async function startGenerate() {
  if (generationSubmitting) return;
  generationSubmitting = true;
  const button = element("#generateButton");
  if (button) button.disabled = true;
  try { return await submitGenerate(); }
  finally { generationSubmitting = false; if (button) button.disabled = false; }
}

async function submitGenerate() {
  setWorkspaceMode("generation");
  const checkpoint=valueOrDefault('#checkpointPath','').trim();
  if((typeof currentGenerationFamily==='function'?currentGenerationFamily():null)==='chart_composer_v2' && checkpoint &&
      (generationInspection.path!==checkpoint || !generationInspection.data?.valid)) {
    const checked=await inspectGenerationCheckpoint();
    if(!checked?.valid)throw new Error(checked?.error||'Composer v2 检查点不可用');
  }
  const payload = generationPayload();
  if (!payload.checkpoint) throw new Error("请先上传或填写 checkpoint 路径");
  if (!payload.audio) throw new Error("请先导入音频或填写音频路径");
  if(payload.model_family==="chart_composer_v2") {
    validateV2NumericInputs('#composerV2GenerationControls');validateV2NumericInputs('#generationTargetGroup');
    const status=await api("/api/status");
    if(status.training_capabilities?.composer_v2_api_contract!=="composer_v2_continuous_events_v3")throw new Error("当前后台不支持 Composer v2 条件与自动节拍合同，请重启为更新后的 WebUI 后台再提交；请先妥善处理正在运行的任务。");
    if(payload.intensity_profile && !status.training_capabilities?.composer_v2_intensity_curve)throw new Error('编辑曲线需要更新后的 WebUI 后台。');
    requireV2ExhaustionSupport(status,payload);
    requireV2SplitSamplingSupport(status,payload);
    const data=await api("/api/composer-v2/generate",payload);
    watchTask(data.task_id,task=>{if(task.status==="completed")setAcceptanceStatus(composerV2CompletionMessage(task),(task.result?.composer_v2_report?.accepted_with_conflicts_count>0 || task.result?.composer_v2_batch?.failed>0 || task.result?.composer_v2_batch?.charts?.some(x=>x.accepted_with_conflicts_count>0))?"warn":"good");});
    return;
  }
  if (payload.model_family === "chart_composer_v1") {
    const status = await api("/api/status");
    if(isEventSetGeneration()&&status.capabilities?.event_set_stream!==true)
      throw new Error('当前后台尚未加载整拍集合生成。请连接更新后的 WebUI 后台，再检查新 checkpoint；保留正在训练的旧服务。');
    if (payload.contact_policy==='contact_groups_v1'
        && status.training_capabilities?.composer_contact_budget!=='conservative_groups_v1')
      throw new Error('当前服务尚未加载接触预算策略，请重启更新后的 WebUI 服务。');
    if(isEventSetGeneration() && generationInspection?.data?.generation_capabilities?.song_profile===true && status.training_capabilities?.composer_song_style_contract!=='masked_joint_response_v1') throw new Error('当前后台尚未加载整曲风格生成，请重启 WebUI 后台。');
    if(isEventSetGeneration() && status.training_capabilities?.composer_phrase_contract!=='phrase_modes_v1') throw new Error('当前后台尚未加载短句写法温度，请更新 WebUI 后台。');
    if (status.composer_generation_contract !== "composer_generation_controls_v13_cooperative_quality") {
      throw new Error("当前服务尚未加载 Composer 独立曲线控件。请使用更新后的 WebUI 服务；有活动训练时保留原服务监控训练。");
    }
    const data = await api("/api/composer/generate", payload);
    watchTask(data.task_id, (task) => { if (task.status === "completed") {
      const report=task.result?.composer_report||{},playability=report.playability||{};
      const remainingHard=Number(playability.remaining_hard);
      const hasHard=Number.isFinite(remainingHard)&&remainingHard>0;
      const partial=playability.coverage?.status==='partial'||playability.status==='partial';
      setAcceptanceStatus(hasHard?`chart_composer_v1 生成完成，仍有 ${remainingHard} 项硬冲突；请查看 .composer.json 并复核。`
        :partial?'chart_composer_v1 生成完成，MiaCode 检测覆盖不完整；请查看 .composer.json 并复核。'
        :'chart_composer_v1 生成完成；学习负荷曲线与检查结果见 .composer.json，仍需人工试听与游玩检查。',hasHard||partial?'warn':'good');
    }});
    return;
  }
  if (payload.model_family === "chart_stream_v1") {
    const status = await api("/api/status");
    if (status.stream_generation_contract !== "chart_stream_quality_controls_v8") {
      throw new Error("当前服务尚未加载连续流质量控制。请使用已更新的 WebUI 服务；若此端口正在训练，请等训练结束后重启，避免新参数被旧后端忽略。");
    }
    const data = await api("/api/chart-stream/generate", payload);
    watchTask(data.task_id, (task) => { if (task.status === "completed") {
      const incomplete = task.result?.generation_status === "incomplete";
      const review = task.result?.generation_status === "needs_review";
      setAcceptanceStatus(incomplete ? "chart_stream_v1 生成未完成；请查看搜索／事件／约束预算原因及报告，已保存完整事件。"
        : review ? "chart_stream_v1 已结束，但参考检查仍发现硬冲突；请查看报告并复核。"
        : "chart_stream_v1 连续整曲生成已结束；请检查连续性、休止和谱面可玩性。", incomplete || review ? "warn" : "good");
    }});
    return;
  }
  if (["song_form_v1", "song_form_v2"].includes(payload.model_family)) {
    payload.requested_output = payload.output;
    payload.output = uniqueSongFormOutputPath(payload.output);
    const data = await api("/api/form/generate", payload);
    watchTask(data.task_id, (task) => { if (task.status === "completed") setAcceptanceStatus("song-form 生成完成；请查看报告并进行人工试听与游玩检查。", "good"); });
    return;
  }
  if (payload.conflict_detection_enabled) {
    const status = await api("/api/status");
    if (status.playability_policy_version !== 3) {
      throw new Error("当前后端尚未加载零阈值与不限回溯次数的无理策略，请重启 WebUI 服务后再生成");
    }
    if (status.playability_engine?.available !== true) {
      throw new Error(status.playability_engine?.error || "当前后端尚未加载 MiaCode Python 检测，请重启 WebUI 服务后再生成");
    }
  }
  if (payload.local_onset_bias_strength > 0) {
    const caps = generationInspection.path === payload.checkpoint && generationInspection.data?.valid === true
      ? generationInspection.data.generation_capabilities || {} : {};
    if (caps.local_onset_guidance !== true) {
      throw new Error("请先检查 checkpoint；当前模型没有已训练的局部采音对齐辅助头，不能开启非零指导。");
    }
  }
  if (payload.intensity_curve_enabled) {
    if (!payload.intensity_curve) throw new Error("已启用谱面张力曲线，但当前歌曲还没有生成并保存曲线");
    if (intensityCurveState.path !== payload.audio) throw new Error("音频路径已变化，请为当前歌曲重新生成强度曲线");
  }
  const usesIntensityFeature = payload.intensity_curve_enabled || (payload.intensity_follow && payload.intensity_mode === "multifeature");
  if (payload.model_family !== "legacy_frame_v2" && usesIntensityFeature) {
    const status = await api("/api/status");
    if (status.intensity_curve_capability !== true) throw new Error("后端尚未加载多维强度跟随/曲线能力，请重启 WebUI 服务后再生成");
    const hasExplicitRest = Array.isArray(payload.intensity_curve?.rest_regions) && payload.intensity_curve.rest_regions.length > 0;
    if (((payload.intensity_curve_enabled && ["follow", "hard"].includes(payload.intensity_curve_mode)) || hasExplicitRest)
        && status.density_feedback_capability !== true) {
      throw new Error(hasExplicitRest
        ? "当前后端尚未加载显式休止区密度反馈能力，请重启 WebUI 服务后再生成"
        : "当前后端尚未加载 follow / hard 密度反馈能力，请重启 WebUI 服务后再生成");
    }
  }
  if (payload.model_family !== "legacy_frame_v2"
      && (payload.intensity_curve_enabled || payload.intensity_mode === "auto" || payload.intensity_mode === "model")) {
    const status = await api("/api/status");
    if (status.music_structure_capability !== true) {
      throw new Error("当前后端未加载音乐结构 planner 与曲线控制，请重启 WebUI 服务后再生成");
    }
  }
  const data = await api("/api/generate", payload);
  if (data.curve_condition_warnings?.length) {
    intensityCurveState.conditionChanged = true;
    saveIntensityCurveState(); updateCurveConditionNotice();
    setIntensityStatus(data.curve_condition_warnings.join("；"), "warn");
  }
  watchTask(data.task_id, (task) => {
    if (task.status === "completed") setAcceptanceStatus("生成完成，可点击「验收谱面」做横向比较。", "good");
  });
}

function clearGenerate() {
  resetGenerationDefaults();
  ["#checkpointPath", "#audioPath", "#featurePath", "#tempoMap", "#formGenerationMeterMap", "#composerConditionFile", "#goldStatsPath",
    "#acceptanceMaidata", "#acceptanceAudio", "#qualityStatus", "#qualityDiagnostics"]
    .forEach((selector) => {
      const target = element(selector);
      if (!target) return;
      if (target.tagName === "TEXTAREA" || target.tagName === "INPUT") target.value = "";
      else target.textContent = "";
    });
  ["#checkpointFile", "#audioFile", "#featureFile"].forEach((selector) => {
    const target = element(selector); if (target) target.value = "";
  });
  if (element("#outputPath")) element("#outputPath").value = "webui_workspace/generated/maidata.txt";
  generationInspection = {path: "", data: null};
  updateFamilyFields();
  ["#checkpointUploadStatus", "#audioUploadStatus", "#featureUploadStatus"].forEach((selector) => {
    if (element(selector)) element(selector).textContent = "未导入文件";
  });
  const panel = element("#acceptancePanel");
  if (panel) panel.hidden = true;
  setGenerationStatus("已清空导入路径，请重新选择歌曲与 checkpoint。");
  setAcceptanceStatus("已清空导入路径与生成参数，输出路径恢复默认；训练设置未改动。");
  saveSettings();
}

const VERDICT_BADGE = { pass: "PASS", warn: "WARN", fail: "FAIL", unknown: "N/A" };

function renderAcceptance(report) {
  const panel = element("#acceptancePanel");
  const verdictBox = element("#acceptanceVerdict");
  if (panel) panel.hidden = false;
  const verdict = report?.verdict || "unknown";
  if (verdictBox) {
    verdictBox.className = `acceptance-verdict ${verdict}`;
    const summary = verdict === "pass"
      ? `全部通过；无失败项。`
      : `失败 ${report?.failures ?? 0} 项、警告 ${report?.warnings ?? 0} 项。`;
    verdictBox.textContent = `${VERDICT_BADGE[verdict] || "?"} ${summary}`
      + (report?.reference_charts ? `（参照：${report.reference || "金标"} ${report.reference_charts} 张）` : "");
  }
  const criteria = element("#acceptanceCriteria");
  if (criteria) {
    criteria.replaceChildren();
    (report?.criteria || []).forEach((item) => {
      const row = document.createElement("div");
      row.className = `acceptance-row ${item.verdict}`;
      const range = item.low === undefined || item.low === null
        ? ""
        : ` / 金标 ${Number(item.low).toFixed(3)}–${Number(item.high).toFixed(3)}`;
      row.innerHTML = `<strong>${escapeHtml(item.label || item.key)}</strong>`
        + `<span>${Number(item.value).toFixed(3)}${range}</span>`
        + `<em>${VERDICT_BADGE[item.verdict] || "?"}</em>`
        + `<p>${escapeHtml(item.message || "")}</p>`;
      criteria.appendChild(row);
    });
  }
  if (report?.timeline) drawTimeline(report.timeline);
  setAcceptanceStatus("验收完成。语法通过不等于可玩，请人工试玩确认。", verdict === "pass" ? "good" : "warn");
}

function renderAcceptanceFromTask(task) {
  if (task.result?.acceptance_report) {
    renderAcceptance(task.result.acceptance_report);
  } else if (task.result?.acceptance_error) {
    setAcceptanceStatus(task.result.acceptance_error, "warn");
  }
}

async function startAcceptance() {
  const maidata = valueOrDefault("#acceptanceMaidata", "").trim() || valueOrDefault("#outputPath", "").trim();
  if (!maidata) throw new Error("请填写待验收的 maidata.txt 路径");
  setAcceptanceStatus("正在验收谱面...");
  const data = await api("/api/acceptance/evaluate", {
    maidata,
    audio: valueOrDefault("#acceptanceAudio", "").trim() || valueOrDefault("#audioPath", "").trim() || null,
    gold_stats: valueOrDefault("#goldStatsPath", "").trim() || null,
    difficulty: Number(valueOrDefault("#acceptanceDifficulty", 5)),
    bpm: valueOrDefault("#bpm", "").trim() || null,
    cache_dir: datasetPaths().cache_dir,
  });
  watchTask(data.task_id);
}

function drawTimeline(timeline) {
  const canvas = element("#acceptanceChart");
  const hint = element("#acceptanceChartHint");
  if (!canvas || !canvas.getContext) return;
  const points = timeline?.points || [];
  const spans = timeline?.blank_spans || [];
  const duration = Number(timeline?.duration_sec || 0);
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { left: 42, right: 12, top: 12, bottom: 22 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fffdf9";
  context.fillRect(0, 0, width, height);
  if (!points.length || duration <= 0) {
    context.fillStyle = "#77736b";
    context.fillText("没有可绘制的时序数据（需要音频与谱面）", pad.left, height / 2);
    if (hint) hint.textContent = "填好音频后可显示强度/密度/空白对照图。";
    return;
  }
  const densityMax = Math.max(1, ...points.map((point) => point.density || 0));
  // Blank measures are shaded first so the curves read on top of them.
  context.fillStyle = "rgba(217, 95, 73, .12)";
  spans.forEach(([start, end]) => {
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;
    const x0 = pad.left + (start / duration) * plotWidth;
    const x1 = pad.left + (end / duration) * plotWidth;
    context.fillRect(x0, pad.top, Math.max(1, x1 - x0), plotHeight);
  });
  // Intensity (audio) as a filled envelope.
  context.beginPath();
  points.forEach((point, index) => {
    const x = pad.left + (point.t / duration) * plotWidth;
    const y = pad.top + plotHeight - Math.min(1, point.intensity || 0) * plotHeight * 0.6;
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.strokeStyle = "#167d7c";
  context.lineWidth = 1.5;
  context.stroke();
  // Density as bars.
  const barWidth = Math.max(1, plotWidth / points.length - 1);
  context.fillStyle = "rgba(229, 189, 69, .85)";
  points.forEach((point) => {
    const x = pad.left + (point.t / duration) * plotWidth;
    const value = (point.density || 0) / densityMax;
    const barHeight = value * plotHeight * 0.5;
    context.fillRect(x, pad.top + plotHeight - barHeight, barWidth, barHeight);
  });
  context.strokeStyle = "#d9d3c8";
  context.lineWidth = 1;
  context.strokeRect(pad.left, pad.top, plotWidth, plotHeight);
  context.fillStyle = "#77736b";
  context.font = "11px system-ui, sans-serif";
  context.fillText(`密度峰值 ${densityMax}/窗口`, pad.left, pad.top + 10);
  context.fillText(`${duration.toFixed(1)}s`, pad.left + plotWidth - 34, height - 6);
  if (hint) {
    hint.textContent = `蓝线=歌曲相对强度，黄柱=谱面密度，红底=空白小节（${spans.length} 段）。`;
  }
}

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------
async function refreshLogs() {
  const data = await api("/api/logs?limit=100");
  const container = element("#logList");
  const status = element("#logsStatus");
  if (status) status.textContent = `日志目录：${data.root}（共 ${(data.logs || []).length} 条）`;
  if (!container) return;
  container.hidden = false;
  container.replaceChildren();
  (data.logs || []).forEach((entry) => {
    const row = document.createElement("div");
    row.className = "log-row";
    const label = `${entry.started || ""} · ${TASK_KIND_LABEL[entry.kind] || entry.kind} · ${entry.status}`;
    row.textContent = `${label} · ${formatBytes(entry.bytes)} · ${entry.task_id || ""}`
      + (entry.error ? ` · ${String(entry.error).split("\n")[0]}` : "");
    if (entry.download) {
      const link = document.createElement("a");
      link.href = entry.download;
      link.textContent = "下载";
      link.className = "log-link";
      row.appendChild(link);
      if (entry.kind === "form_train") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "small-button";
        button.textContent = "查看训练走势";
        button.addEventListener("click", guard(() => viewArchivedTrainingLog(entry, button)));
        row.appendChild(button);
      }
    }
    container.appendChild(row);
  });
}

async function copyLogsFolder() {
  const data = await api("/api/logs?limit=1");
  const text = data.root || "";
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`日志目录已复制：${text}`, true);
  } catch (_) {
    setStatus(`日志目录：${text}`, true);
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
setWorkspaceMode(initialWorkspace());
controlRows("radarControls", radarKeys);
controlRows("styleControls", styleKeys);
loadIntensityCurveState(false);
generationDefaults = captureGenerationDefaults();
initializeEventSetParameterGroups();
const restoredSettings = restoreSettings();
invalidateLearnedCurveForCheckpoint();
if (restoredSettingNotice) {
  const status = element("#conditionPresetStatus");
  if (status) status.textContent = restoredSettingNotice;
}
if (restoredSettings) {
  setTrainingModeDefaults();
}
updateTrainingControls();
updateGoldFields();
if (document.body.dataset.workspace === "quality") guard(() => loadQualityCatalog(true))();
if (intensityCurveState.profile) {
  invalidateIntensityForAudio();
  drawIntensityCurve();
  updateIntensityButtons();
  restoreIntensityBackdrop();
}

// Persist any edit, in the capture phase so a handler that stops propagation
// still does not lose the change.
["input", "change"].forEach((type) => {
  document.addEventListener(type, (event) => {
    const target = event.target;
    if (!target || !target.id || target.type === "file") return;
    if (target.id === "learningRate") target.dataset.touched = "1";
    if (target.id === "contextTokens") target.dataset.touched = "1";
    if (["formQualityProfile", "formPositionAlignmentLossWeight", "formPositionAlignmentToleranceMs",
      "formPlannerContextSeconds", "formCurveBinSeconds"].includes(target.id)) markSongFormEvidenceInput(target);
    if (target.id === "fineNoteStrength") {
      const output = element("#fineNoteStrengthValue");
      if (output) output.textContent = Number(target.value).toFixed(2);
    }
    if (target.id === "localOnsetBiasStrength") {
      const output = element("#localOnsetBiasStrengthValue");
      if (output) output.textContent = Number(target.value).toFixed(2);
    }
    updateGenerationCapabilityControls();
    saveSettings();
  }, true);
});
window.addEventListener("beforeunload", flushSettings);

// Mode tabs
on("#trainingModeTab", "click", () => setWorkspaceMode("training"));
on("#qualityModeTab", "click", async()=>{
  setWorkspaceMode('quality');await loadDatasetCatalog();
  if(datasetWorkspaceState.view==='catalog' && qualityManifest() && !qualityState.items.length)await loadQualityCatalog(false);
});
on("#generationModeTab", "click", () => setWorkspaceMode("generation"));
on("#intensityModeTab", "click", () => { setIntensitySong(intensityAudioPath()); setWorkspaceMode("intensity"); drawIntensityCurve(); });
on("#conflictPreset", "change", () => {
  const preset = valueOrDefault("#conflictPreset", "custom");
  if (Object.prototype.hasOwnProperty.call(CONFLICT_PRESETS, preset)) applyConflictPreset(preset);
  else saveSettings();
});
CONFLICT_KIND_FIELDS.forEach(([, id]) => on(`#${id}`, "change", () => {
  const preset = element("#conflictPreset");
  if (preset) preset.value = "custom";
  saveSettings();
}));
on("#tapOnSlideThresholdMs", "change", () => {
  const preset = element("#conflictPreset");
  if (preset) preset.value = "custom";
  saveSettings();
});

// Training panel
on("#modelFamily", "change", () => {
  generationInspectRequest += 1;
  if(currentModelFamily()==="chart_composer_v2" && currentTrainingMode()==="fresh") {
    const directory=element("#checkpointDir");if(directory)directory.value="checkpoints/chart_composer_v2";
  } else if(element("#checkpointDir")?.value==="checkpoints/chart_composer_v2") {
    element("#checkpointDir").value="checkpoints/"+currentModelFamily();
  }
  generationInspection = { path: "", data: null }; checkpointInspection = { key: "", data: null };
  applySongFormCheckpointConfig({valid: false}, "fresh");
  if (currentTrainingMode() === "fresh" && currentModelFamily() === "song_form_v2") {
    for (const [id, v] of Object.entries({formDModel:256, formLayers:4, formNhead:8, formDimFeedforward:0, formPlannerWidth:128})) {
      const node = element("#" + id); if (node) node.value = String(v);
    }
    if (["checkpoints/chart_stream_v1", "checkpoints/chart_composer_v1"].includes(element("#checkpointDir")?.value)) element("#checkpointDir").value = "checkpoints/song_form_v2";
  }
  if (currentTrainingMode() === "fresh" && currentModelFamily() === "chart_stream_v1") {
    for (const [id, v] of Object.entries({streamDModel:256, streamLayers:4, streamNhead:8, streamDimFeedforward:768,
      streamMaxTokens:1024, streamMaxHistoryTokens:256, streamMemoryTokens:32, streamPlanLossWeight:.15,
      streamDropout:.1, streamSeed:0,
      streamHistoryPositionDropout:.1, streamHistoryPositionSpanEvents:4, streamHistoryPositionKeepEvents:2, streamConditionDropout:.1,
      streamPrecision:"auto", streamAttentionBackend:"auto", streamHandBackend:"auto", streamExecutionMode:"auto", streamCudaGraphBudgetGb:2,
      streamWeightDecay:.01, streamAudioCacheGb:8, streamPrepareWorkers:4,
      streamMaxRegionSeconds:15, streamLimitCharts:0, streamLogInterval:25, streamCacheDir:"cache/chart_stream_v1"})) {
      const node = element("#" + id); if (node) node.value = String(v);
    }
    if (!element("#checkpointDir")?.value || ["checkpoints/song_form_v2", "checkpoints/chart_composer_v1"].includes(element("#checkpointDir")?.value)) element("#checkpointDir").value = "checkpoints/chart_stream_v1";
  }
  if (currentTrainingMode() === "fresh" && currentModelFamily() === "chart_composer_v1") {
    applyComposerFreshTrainingDefaults();
  }
  updateTrainingControls(); updateFamilyFields(); saveSettings();
});
on("#v2TrainValidationSource", "change", () => {updateTrainingControls();saveSettings();});
on("#v2TrainDifficultySource", "change", () => {updateV2DifficultyControls();saveSettings();});
on("#v2TrainChartgateSetup", "click", initializeV2Chartgate);
on("#v2TrainQualityEnabled", "change", () => {updateTrainingControls();saveSettings();});
on('#v2GenerateCount','input',()=>{updateV2BatchControls();saveSettings();});
on('#v2GenerateParallel','input',()=>{updateV2BatchControls();saveSettings();});
on('#v2GeneratePhraseSearch','change',()=>{updateV2BatchControls();saveSettings();});
on('#v2GenerateCandidates','input',()=>{updateV2BatchControls();saveSettings();});
on("#v2AutoQualityAuditButton", "click", () => { void startV2AutoQualityAudit(); });
on("#trainingMode", "change", () => { const validationField=element('#composerValidationManifest');if(currentTrainingMode()!=='resume' && validationField?.dataset.validationInherited==='1'){validationField.value=validationField.dataset.beforeResumeValidation??'';delete validationField.dataset.validationInherited;delete validationField.dataset.beforeResumeValidation;} applySongFormCheckpointConfig({valid: false}, "fresh"); for(const id of ['composerActionObjectivePolicy','composerHistoryAnchorPolicy','composerHistoryMemoryPolicy','composerHistoryAttentionPolicy','composerHeadTimingPolicy','composerCrossEventRelationPolicy','composerFutureHorizonPolicy','composerEchoPriorPolicy','composerTimingDistributionPolicy','composerButtonLoadPolicy','composerFineTimingPolicy','composerDensityFeedbackPolicy','composerActionHeadPolicy','composerConfigurationQualityPolicy','composerSampleSequencePolicy','composerActionStatePolicy','composerMotifMemoryPolicy','composerSlideFactorPolicy','composerStructurePolicy','composerStructureBackend','composerContactPolicy','composerRecoveryTouchLayoutPolicy','composerRecoverySamplingPolicy']){const target=element('#'+id);if(target && currentTrainingMode()!=='resume')target.value='auto';}setTrainingModeDefaults({resetComposer:true}); updateTrainingControls(); });
on("#composerApplyFreshDefaults",'click',()=>{if(applyComposerFreshTrainingDefaults()){updateTrainingControls();saveSettings();setStatus(valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1'?'已应用整拍集合默认值：50轮、固定学习率0.0005、batch16短段×累积4、历史128、展开32事件、有序草稿。':'已应用 Composer 重训默认值：20轮、batch64×累积4，启用动机与配置恢复。',true);}});
on('#composerLearningPolicy','change',()=>{const legacy=valueOrDefault('#composerLearningPolicy','auto')==='legacy';for(const [id,value] of [['composerTransitionExitWeight',0],['composerPrefixAugmentationRate',legacy?0:.15]]){const node=element('#'+id);if(node&&!node.disabled)node.value=String(value);}updateEventSetControls();saveSettings();});
on('#composerResetLearningRate','change',()=>{const reset=element('#composerResetLearningRate');if(!reset?.checked && checkpointInspection.key===currentCheckpointKey() && checkpointInspection.data?.valid){const args=checkpointInspection.data.args||{};for(const [id,key] of [['learningRate','lr'],['composerLrSchedule','lr_schedule'],['composerWarmupSteps','warmup_steps'],['composerMinLrRatio','min_lr_ratio']]){const node=element('#'+id);if(node&&args[key]!==undefined)node.value=String(args[key]);}}updateEventSetControls();});
on('#composerSongResponsePolicy','change',()=>{updateEventSetControls();saveSettings();});
on('#composerValidationSource','change',()=>{updateEventSetControls();saveSettings();});
on('#composerArchitectureProfile','change',()=>{if(valueOrDefault('#composerArchitectureProfile','')==='event_set_stream_v1')applyEventSetFreshDefaults();updateEventSetControls();saveSettings();});
updateEventSetControls();
on('#composerRecoverySamplingPolicy','change',()=>{updateComposerRecoverySamplingControls();saveSettings();});
on('#composerRebatchResume','change',()=>{updateTrainingControls();saveSettings();});
on("#formRhythmGeometryProfileSelect", "change", () => { updateSongFormRhythmGeometryControls(); saveSettings(); });
on("#formQualityApplyRecipe", "click", applySongFormQualityRecipe);
for (const id of ["formPerturbApplyProb", "formPerturbCurveNoiseStd", "formPerturbCurveDropoutProb", "formPerturbHistoryCorruptionProb", "formPerturbOnsetDropProb", "formPerturbOnsetAddProb", "formPerturbOnsetShiftMs", "formDiversityLossWeight"]) {
  on(`#${id}`, "input", () => { const recipe = element("#formQualityRecipe"); if (recipe) recipe.value = "custom"; saveSettings(); });
}
on("#formResumeDataPolicy", "change", () => { updateTrainingControls(); saveSettings(); });
on("#trainingCheckpointPath", "input", () => { if (["song_form_v1", "song_form_v2"].includes(currentModelFamily())) applySongFormCheckpointConfig({valid: false}, "fresh"); invalidateCheckpointInspection("checkpoint 路径已变化，请重新检查。"); });
on("#goldMode", "change", () => { updateGoldFields(); saveSettings(); });
on("#preprocessButton", "click", startPreprocess);
on("#inspectManifestButton", "click", inspectManifest);
on("#checkpointInspectButton", "click", inspectCheckpoint);
on("#trainButton", "click", () => startTrain());
on("#autoPipelineButton", "click", () => startTrain({ autoPreprocess: true }));
on("#openQualityWorkspaceButton", "click", openDatasetWorkspace);
on("#chooseTrainingDatasetsButton", "click", openDatasetWorkspace);
on("#datasetWorkspaceRefreshButton", "click", async () => { await loadStatus(); });
on("#datasetComposeButton", "click", composeSelectedDatasets);
on("#trainingDatasetApplyButton", "click", composeSelectedDatasets);
on("#datasetImportButton", "click", () => { setWorkspaceMode("training"); window.MaichartWorkflow?.focusSetting("rawRoot"); setStatus("上传训练集 ZIP，或填写本机目录并点击“登记当前目录到训练素材”。", true); });
on("#registerDatasetButton", "click", registerCurrentDataset);
on("#datasetBackButton", "click", showDatasetCards);
on("#qualityRefreshButton", "click", () => loadQualityCatalog(true));
on("#qualityLabelFilter", "change", async (event) => { qualityState.label = event.target.value; await loadQualityCatalog(true); });
on("#qualityAutoFilter", "change", async (event) => { qualityState.autoFilter = event.target.value; await loadQualityCatalog(true); });
on("#qualitySearch", "change", async (event) => { qualityState.query = event.target.value.trim(); await loadQualityCatalog(true); });
on("#qualitySearch", "keydown", async (event) => { if (event.key === "Enter") { qualityState.query = event.target.value.trim(); await loadQualityCatalog(true); } });
on("#qualitySortBy", "change", () => loadQualityCatalog(true));
on("#qualitySortDirection", "change", () => loadQualityCatalog(true));
on("#qualityPageSize", "change", (event) => { qualityState.pageSize = Math.max(1, Number(event.target.value) || 25); loadQualityCatalog(true); });
on("#qualitySaveSelectedButton", "click", saveQualitySelection);
on("#qualitySelectFolderButton", "click", selectQualityFolder);
on("#qualityClearSelectionButton", "click", clearQualitySelection);
on("#qualityPrevButton", "click", async () => { if (qualityState.page > 1) { qualityState.page -= 1; await loadQualityCatalog(false); } });
on("#qualityNextButton", "click", async () => { if (qualityState.page < (qualityState.pages || Math.ceil(qualityState.total / qualityState.pageSize))) { qualityState.page += 1; await loadQualityCatalog(false); } });
on("#qualityPageJumpButton", "click", async () => {
  const pageCount = Math.max(1, qualityState.pages || Math.ceil(qualityState.total / qualityState.pageSize));
  const requested = Math.min(pageCount, Math.max(1, Math.floor(Number(valueOrDefault("#qualityPageInput", "1")) || 1)));
  qualityState.page = requested;
  await loadQualityCatalog(false);
});
on("#qualityPageInput", "keydown", async (event) => { if (event.key === "Enter") element("#qualityPageJumpButton")?.click(); });
on("#qualitySelectAll", "change", (event) => { qualityState.items.forEach((item) => { const id = String(item.chart_id || item.id); if (event.target.checked) qualityState.selected.add(id); else qualityState.selected.delete(id); }); renderQualityCatalog(); });
on("#qualityCloseDetailButton", "click", () => { const panel = element("#qualityDetail"); if (panel) panel.hidden = true; });
on("#qualityAddRegionButton", "click", () => { qualityState.regions.push({ start_sec: Number(element("#qualityDetailAudio")?.currentTime||0), end_sec: Number(element("#qualityDetailAudio")?.currentTime||0)+1, weight: 1, label:"uncertain", reason: "", timing_weight:1, component_weight:1 }); updateQualityDetail(); });
on("#qualitySaveRegionsButton", "click", saveQualityRegions);
on("#qualitySaveCorrectionButton", "click", saveQualityCorrection);
on("#qualityBulkLabel", "change", (event) => { const weight = element("#qualityBulkWeight"); if (weight) weight.value = QUALITY_DEFAULT_WEIGHTS[event.target.value] ?? 1; });
document.addEventListener("change", (event) => {
  const regionRow=event.target.closest?.('.quality-region-row');
  if(regionRow && ['label','reason'].includes(event.target.dataset.regionField)) {
    const field=name=>regionRow.querySelector(`[data-region-field="${name}"]`);
    const label=field('label').value,reason=field('reason').value;
    field('weight').value=1;
    field('timing_weight').value=label==='good'?1.5:label==='problem' && (!reason || reason==='timing')?0:1;
    field('component_weight').value=label==='good'?1.5:label==='problem' && (!reason || reason==='layout' || reason==='comfort')?0:1;
  }
  const row = event.target.closest?.(".quality-row-select");
  if (row) { const id = row.dataset.chartId; if (row.checked) qualityState.selected.add(id); else qualityState.selected.delete(id); updateQualitySelectionControls(); }
  const weight = event.target.closest?.(".quality-weight");
  if (weight) { const item = qualityState.items.find((candidate) => String(candidate.chart_id || candidate.id) === String(weight.dataset.chartId)); if (item) { item.weight = Math.max(0, Math.min(4, Number(weight.value))); item._weightEdited = true; qualityState.editedWeights.set(String(weight.dataset.chartId), item.weight); } }
}, true);
document.addEventListener("click", (event) => {
  const auditPreset = event.target.closest?.("[data-auto-audit-preset]");
  if (auditPreset) { applyAutoQualityPreset(auditPreset.dataset.autoAuditPreset); return; }
  const trainPreset = event.target.closest?.("[data-v2-train-preset]");
  if (trainPreset) { applyV2TrainPreset(trainPreset.dataset.v2TrainPreset); return; }
  const genPreset = event.target.closest?.("[data-v2-gen-preset]");
  if (genPreset) { applyV2GenPreset(genPreset.dataset.v2GenPreset); return; }
  const runAudit = event.target.closest?.("[data-run-auto-audit]");
  if (runAudit) {
    const mode = runAudit.dataset.runAutoAudit;
    const panel = runAudit.closest('#qualityAutoAuditCard') ? 'quality' : 'training';
    void startV2AutoQualityAudit({ panel, limitCharts: mode === "preview20" ? 20 : 0 });
    return;
  }
  const switchAudit = event.target.closest?.("[data-auto-audit-switch]");
  if (switchAudit) {
    const panel = switchAudit.closest('#qualityAutoAuditDashboard') ? 'quality' : 'training';
    const scope = switchAudit.dataset.autoAuditSwitch;
    if (panel === 'quality' && datasetWorkspaceState.view === 'catalog') {
      guard(() => changeCatalogAuditScope(scope))();
    } else {
      void refreshAutoQualityAuditStatus(panel, scope);
    }
    return;
  }
  const jumpAudit = event.target.closest?.("[data-auto-audit-jump]");
  if (jumpAudit) {
    guard(() => jumpToAutoAuditCatalog(jumpAudit.closest('.auto-audit-dashboard'), jumpAudit.dataset.autoAuditJump))();
    return;
  }
  const preprocessButton = event.target.closest?.(".dataset-card-preprocess");
  if (preprocessButton) {
    const item = datasetWorkspaceState.items.find((candidate) => candidate.path === preprocessButton.dataset.datasetPath);
    if (item) {
      if (item.path) selectDataset(item);
      setWorkspaceMode("training");
      setStatus("已切换到该数据集，请点击“预处理训练集”。", true);
    }
    return;
  }
  const openDatasetButton = event.target.closest?.(".dataset-card-open");
  if (openDatasetButton) {
    const item = datasetWorkspaceState.items.find((candidate) => candidate.path === openDatasetButton.dataset.datasetPath);
    if (item) guard(() => openDataset(item))();
  }
  const button = event.target.closest?.(".quality-detail-button");
  if (button) guard(()=>openQualityDetail(button.dataset.chartId))();
  const remove = event.target.closest?.(".quality-remove-region");
  if (remove) { const row = remove.closest(".quality-region-row"); const index = Number(row?.dataset.regionIndex); if (Number.isInteger(index)) { qualityState.regions.splice(index, 1); updateQualityDetail(); } }
});
document.addEventListener("change", (event) => {
  const checkbox = event.target.closest?.(".dataset-select-checkbox");
  if (!checkbox) return;
  datasetWorkspaceState.selectionTouched = true;
  const key = checkbox.dataset.datasetKey;
  if (checkbox.checked) datasetWorkspaceState.selected.add(key); else datasetWorkspaceState.selected.delete(key);
  if(selectedDatasetItems().some(isZeroUnmarkedDataset)) {
    const quality=element("#v2TrainQualityEnabled");if(quality)quality.checked=true;
    const legacyQuality=element("#qualityEnabled");if(legacyQuality)legacyQuality.checked=true;
  }
  updateZeroUnmarkedTrainingNotice();
  datasetWorkspaceState.prepared = null;
  const compositionSummary = element("#datasetCompositionSummary");
  if (compositionSummary) {
    compositionSummary.hidden = false;
    compositionSummary.className = "inline-status warn";
    compositionSummary.textContent = "勾选来源已变化；检查或开始训练时将按当前选择自动准备清单。";
  }
  invalidateDatasetStatus("本次训练来源已变化，请重新检查训练集与 checkpoint。");
  saveDatasetWorkspace();
  renderDatasetWorkspace();
  void refreshAutoQualityAuditStatus('quality');
}, true);
window.addEventListener('workspace-mode-changed', event => {
  const panel = event.detail.mode;
  if (panel === 'quality' || panel === 'training') void refreshAutoQualityAuditStatus(panel);
});
for (const id of ['v2TrainQualityEnabled', 'v2TrainAutoQualityPolicy', 'trainingMode', 'manifestPath']) {
  on('#' + id, 'change', () => {
    if (document.body.dataset.workspace === 'training') void refreshAutoQualityAuditStatus('training');
  });
}
on("#qualityCorrectedFile", "change", async (event) => {
  const data = await upload("maidata", event.target.files[0], "#qualityCorrectedPath", "#qualityCorrectedPath");
  if (data) { qualityState.correctedPath = data.path; if (element("#qualityCorrectedPath")) element("#qualityCorrectedPath").textContent = data.path; }
});
on("#goldScanButton", "click", scanGoldFolder);
on("#goldBuildButton", "click", startGoldBuild);
on("#goldDesignerListButton", "click", listGoldDesigners);
on("#refreshLogsButton", "click", refreshLogs);
on("#openLogsFolderButton", "click", copyLogsFolder);

on("#datasetSelect", "change", async (event) => {
  const data = uploadedDatasets.find((item) => item.path === event.target.value);
  if (!data) return;
  selectDataset(data);
  setStatus("已切换当前预处理目录。训练来源请在‘训练素材’勾选，再点击‘检查所选训练来源’。", true);
});

["#rawRoot", "#cacheDir", "#manifestPath"].forEach((selector) => {
  const target = element(selector);
  if (!target) return;
  target.addEventListener("input", () => {
    const select = element("#datasetSelect");
    if (select) select.value = "";
    invalidateDatasetStatus("路径已修改，请确认它们属于同一份数据，然后重新检查或预处理。");
  });
});

on("#trainingCheckpointFile", "change", async (event) => {
  const data = await upload("checkpoint", event.target.files[0], "#trainingCheckpointPath", "#trainingCheckpointUploadStatus");
  if (data) invalidateCheckpointInspection("checkpoint 已更新，请点击『检查 checkpoint』确认它与当前 manifest 兼容。");
});

on("#datasetZip", "change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  datasetUploading = true;
  datasetVersion += 1;
  invalidateCheckpointInspection("ZIP 正在上传；完成后需要重新检查 checkpoint。");
  setManifestStatus("ZIP 正在上传和解压，请等待完成。");
  updateDatasetButtons();
  try {
    const data = await upload("dataset_zip", file, "#rawRoot", "#datasetUploadStatus");
    if (data.extracting) await waitForExtraction(data.upload_id, data);
    populateDatasets([data, ...uploadedDatasets.filter((item) => item.path !== data.path)]);
    selectDataset(data);
    setManifestStatus("已上传 ZIP，但还没有生成训练清单。下一步：点击『预处理训练集』。", "warn");
  } finally {
    datasetUploading = false;
    updateDatasetButtons();
  }
});

// Generation panel
on("#audioFile", "change", async (event) => { const data = await upload("audio", event.target.files[0], "#audioPath", "#audioUploadStatus"); invalidateIntensityForAudio(); return data; });
on("#audioPath", "input", invalidateIntensityForAudio);
on("#intensityCurveEnabled", "change", updateCurveConditionNotice);
on("#acceptanceDifficulty", "change", invalidateLearnedCurveForCondition);
on("#tempoMap", "input", invalidateLearnedCurveForTiming);
on("#tempoMap", "change", invalidateLearnedCurveForTiming);
on("#formGenerationMeterMap", "input", invalidateLearnedCurveForTiming);
on("#formGenerationMeterMap", "change", invalidateLearnedCurveForTiming);
on("#composerConditionFile", "input", invalidateLearnedCurveForCondition);
on("#composerConditionFile", "change", invalidateLearnedCurveForCondition);
on("#formQualityProfile", "change", () => { updateSongFormQualityProfileControls(); saveSettings(); });
["#bpm", "#first"].forEach((selector) => {
  on(selector, "input", invalidateLearnedCurveForTiming);
  on(selector, "change", invalidateLearnedCurveForTiming);
});
["#tempoMode", "#v2GenerateTempoDetection", "#v2GenerateAutoFirst", "#v2GenerateTempoOctavePolicy", "#v2GenerateDownbeatAlignment"].forEach((selector) => {
  on(selector, "change", invalidateLearnedCurveForTiming);
});
on("#featureFile", "change", (event) => upload("audio_features", event.target.files[0], "#featurePath", "#featureUploadStatus"));
on("#checkpointFile", "change", async (event) => {
  const data = await upload("checkpoint", event.target.files[0], "#checkpointPath", "#checkpointUploadStatus");
  if (data) {
    invalidateLearnedCurveForCheckpoint();
    generationInspectRequest += 1;
    generationInspection = { path: "", data: null };
    updateFamilyFields();
    updateEventSetControls();
    if (element("#generationPlannerStatus")) element("#generationPlannerStatus").textContent = "checkpoint 已更换，点击上方检查按钮或直接提交生成可确认模型能力。";
    setGenerationStatus("已上传 checkpoint；点击『检查生成 checkpoint 类型』确认模型族。", "warn");
  }
});
on("#checkpointPath", "input", () => {
  invalidateLearnedCurveForCheckpoint();
  generationInspectRequest += 1;
  generationInspection = { path: "", data: null };
  updateFamilyFields();
  updateEventSetControls();
  if (element("#generationPlannerStatus")) element("#generationPlannerStatus").textContent = "checkpoint 路径已修改，点击上方检查按钮或直接提交生成可确认模型能力。";
});
document.querySelectorAll("[data-fine-note-value]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = element("#fineNoteStrength");
    if (!target || target.disabled) return;
    target.value = button.dataset.fineNoteValue;
    const output = element("#fineNoteStrengthValue");
    if (output) output.textContent = Number(target.value).toFixed(2);
    saveSettings();
  });
});
on("#generationInspectButton", "click", inspectGenerationCheckpoint);on("#v2GenerateAutoFirst", "change", updateFamilyFields);
on("#openIntensityCurveButton", "click", () => { setIntensitySong(intensityAudioPath()); setWorkspaceMode("intensity"); drawIntensityCurve(); restoreIntensityBackdrop(); });
on("#intensityBackToGeneration", "click", () => setWorkspaceMode("generation"));
on("#generateIntensityCurveButton", "click", generateIntensityCurve);
on("#intensitySaveButton", "click", saveIntensityCurveAndReturn);
on("#intensityUndoButton", "click", undoIntensityCurve);
on("#intensityResetButton", "click", resetIntensityCurve);
on("#intensityContrastButton", "click", enhanceIntensityContrast);
on("#applyIntensitySectionButton", "click", applyIntensitySection);
on("#markIntensityRestButton", "click", () => applyIntensityRest(true));
on("#clearIntensityRestButton", "click", () => applyIntensityRest(false));
on("#intensityZoom", "input", event => zoomIntensityView(event.target.value));
on("#intensityZoomIn", "click", () => zoomIntensityView(Number(element("#intensityZoom").value) * 2));
on("#intensityZoomOut", "click", () => zoomIntensityView(Number(element("#intensityZoom").value) / 2));
on("#intensityFitAll", "click", () => { intensityView.start = 0; intensityView.end = null; drawIntensityCurve(); });
on("#intensityFitSelection", "click", fitIntensitySelection);
on("#intensityPan", "input", event => {
  const duration = intensityCurveState.profile?.duration_sec; if (!duration) return;
  const span = (intensityView.end ?? duration) - intensityView.start;
  setIntensityViewport(Number(event.target.value) / 1000 * (duration - span), span);
});
on("#intensitySelectVisible", "click", () => {
  const profile = intensityCurveState.profile; if (!profile) return;
  intensityView.selected = new Set(profile.points.map((point, index) => point.time_sec >= intensityView.start && point.time_sec <= (intensityView.end ?? profile.duration_sec) ? index : -1).filter(index => index >= 0)); drawIntensityCurve();
});
on("#intensityClearSelection", "click", () => { intensityView.selected.clear(); drawIntensityCurve(); });
on("#intensitySelectionDown", "click", () => adjustIntensitySelection(-0.05));
on("#intensitySelectionUp", "click", () => adjustIntensitySelection(0.05));
on("#intensitySelectionSet", "click", () => adjustIntensitySelection());
on("#intensityAudio", "timeupdate", drawIntensityCurve);
on("#intensityAudio", "error", () => { if (intensityCurveState.profile) setIntensityStatus("试听音频不可用。若刚重启了服务，请重新分析当前歌曲以恢复试听；已保存的编辑曲线仍可用于生成。", "warn"); });
const intensityCanvas = element("#intensityCurveCanvas");
if (intensityCanvas) {
  intensityCanvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !intensityCurveState.profile) return;
    const geo = intensityEventCoordinates(event);
    if (geo.x < geo.pad.left || geo.x > geo.pad.left + geo.plotWidth || geo.y < geo.pad.top || geo.y > geo.pad.top + geo.plotHeight) return;
    const index = intensityPointAt(event);
    const selecting = event.shiftKey || (valueOrDefault("#intensityEditMode", "edit") === "select"
      && !(index >= 0 && intensityView.selected.has(index) && !event.ctrlKey && !event.metaKey));
    if (selecting) {
      const previous = event.ctrlKey || event.metaKey ? new Set(intensityView.selected) : new Set();
      intensityView.selected = new Set(previous);
      if (index >= 0) intensityView.selected.add(index);
      intensityView.gesture = { kind: "select", x: geo.x, y: geo.y, toX: geo.x, toY: geo.y, previous };
      intensityCanvas.setPointerCapture(event.pointerId);
    } else if (index >= 0) {
      if (event.ctrlKey || event.metaKey) { if (intensityView.selected.has(index)) intensityView.selected.delete(index); else intensityView.selected.add(index); }
      else if (!intensityView.selected.has(index)) intensityView.selected = new Set([index]);
      if (intensityView.selected.has(index)) {
        intensityCurveState.dragging = index;
        intensityView.gesture = { kind: "drag", y: geo.y, changed: false, values: new Map([...intensityView.selected].map(i => [i, intensityCurveState.profile.points[i].intensity])) };
        intensityCanvas.setPointerCapture(event.pointerId);
      }
    } else {
      intensityView.selected.clear(); const audio = element("#intensityAudio");
      if (audio) audio.currentTime = geo.start + (geo.x - geo.pad.left) / geo.plotWidth * geo.span;
    }
    event.preventDefault(); drawIntensityCurve();
  });
  intensityCanvas.addEventListener("pointermove", (event) => updateIntensityPoint(event));
  const finishDrag = () => {
    const changed = intensityView.gesture?.changed; intensityView.gesture = null; intensityCurveState.dragging = -1;
    if (changed) saveIntensityCurveState(); drawIntensityCurve(); updateIntensityButtons();
  };
  intensityCanvas.addEventListener("pointerup", finishDrag); intensityCanvas.addEventListener("pointercancel", finishDrag);
  intensityCanvas.addEventListener("pointerleave", (event) => { if (event.buttons === 0) finishDrag(); });
  intensityCanvas.addEventListener("wheel", event => {
    if (!event.ctrlKey || !intensityCurveState.profile || intensityView.gesture) return;
    event.preventDefault(); const geo = intensityEventCoordinates(event);
    const ratio = Math.max(0, Math.min(1, (geo.x - geo.pad.left) / geo.plotWidth));
    zoomIntensityView(geo.duration / geo.span * (event.deltaY < 0 ? 1.25 : 0.8), geo.start + geo.span * ratio, ratio);
  }, { passive: false });
}
window.addEventListener("resize", () => drawIntensityCurve());
on("#applyConditionPreset", "click", async () => {
  if (!levelPresetProfile) await loadLevelPresetProfile();
  if (levelPresetProfile) { invalidateLearnedCurveForCondition(); applyConditionPreset(); }
});
on("#level", "change", () => { invalidateLearnedCurveForCondition(); if (levelPresetProfile) applyConditionPreset(); });
on("#levelRange", "input", event => { invalidateLearnedCurveForCondition(); setConditionValue("level", Number(event.target.value)); if (levelPresetProfile) applyConditionPreset(); });
on("#conditionPreferenceStrength", "input", () => { invalidateLearnedCurveForCondition(); if (levelPresetProfile) applyConditionPreset(); else updateConditionRecipeDisplay(); });
for (const definition of Object.values(preferenceDefinitions)) {
  on(`#${definition.field}`, "change", () => { invalidateLearnedCurveForCondition(); if (levelPresetProfile) applyConditionPreset(); else updateConditionRecipeDisplay(); });
}
on("#clearConditionPreferences", "click", () => {
  invalidateLearnedCurveForCondition();
  for (const definition of Object.values(preferenceDefinitions)) { const input = element(`#${definition.field}`); if (input) input.checked = false; }
  if (levelPresetProfile) applyConditionPreset(); else { updateConditionRecipeDisplay(); saveSettings(); }
});
for (const [key] of [...radarKeys, ...styleKeys]) {
  const markManual = () => { invalidateLearnedCurveForCondition(); lastConditionRecipe = null; conditionEditedBeforeProfile = true; updateConditionRecipeDisplay(); };
  on(`#${key}`, "input", markManual);
  document.querySelector(`[data-number-for="${key}"]`)?.addEventListener("input", markManual);
}
on("#v2EvaluateButton", "click", () => startV2QualityEvaluation());
on("#v2AuditButton", "click", () => startV2QualityEvaluation("candidate_audit"));
on("#generateButton", "click", startGenerate);
on("#clearGenerate", "click", clearGenerate);
on("#acceptanceButton", "click", startAcceptance);
$("#resetGenerationButton").addEventListener("click", () => {
  resetGenerationDefaults();
});
$("#resetSettingsButton").addEventListener("click", () => {
  discardSettings();
  location.reload();
});
on("#cancelButton", "click", async () => {
  const taskId = activeTaskId;
  if (!taskId || pendingTaskCancellations.has(taskId)) return;
  pendingTaskCancellations.add(taskId);
  const button = element("#cancelButton");
  if (button) button.disabled = true;
  let response;
  try {
    response = await api("/api/cancel", { task_id: taskId });
  } finally {
    pendingTaskCancellations.delete(taskId);
    const task = response || taskWatchers.get(taskId)?.task;
    if (activeTaskId === taskId && task) renderTask(task);
    else if (activeTaskId === taskId && button) button.disabled = false;
  }
});

// Initial server probe.  Failure must not disable the rest of the page.
guard(loadStatus)();
guard(loadLevelPresetProfile)();
