"use strict";

// Presentation and navigation only. Existing form IDs, payload builders and
// server endpoints remain the source of truth for every job.
(function () {
  const $ = (selector) => document.querySelector(selector);
  const tasks = new Map();
  const value = (id, fallback = "") => document.getElementById(id)?.value ?? fallback;
  const checked = (id) => document.getElementById(id)?.checked === true;
  const nameOf = (id) => {
    const control = document.getElementById(id);
    return control?.selectedOptions?.[0]?.textContent.trim() || value(id);
  };

  function v2DifficultySummary() {
    const source = value("v2TrainDifficultySource");
    const fitted = source === "chartgate" || source === "fitted";
    const minDeclared = Number(value("v2TrainMinDeclaredLevel", "1"));
    const slots = [[4, "EXPERT"], [5, "MASTER"], [6, "Re:MASTER"]]
      .filter(([slot]) => (source !== "chartgate" || slot !== 4) && checked("v2TrainDifficultySlot" + slot))
      .map(([, name]) => name);
    const range = `${value("v2TrainMinLevel") || 12}–${value("v2TrainMaxLevel") || (fitted ? 16 : 15)}`;
    const declared = fitted ? (minDeclared > 0 ? ` · 标称定数 ≥ ${minDeclared}` : " · 标称校验关闭") : "";
    return `训练定数 ${nameOf("v2TrainDifficultySource")} ${range}${declared} · 槽位 ${slots.join("、") || "未选择"}`;
  }

  function v2GenerationSearchSummary() {
    if (document.getElementById('v2GeneratePhraseSearch')?.disabled) return '写谱方式 · 旧模型窗口生成';
    const mode = value('v2GeneratePhraseSearch', 'layout');
    const candidates = Number(value('v2GenerateCandidates', '2'));
    const timingTemperature = value('v2GenerateTemperature', '0.9');
    const actionTemperature = document.getElementById('v2GenerateActionTemperature')?.disabled
      ? timingTemperature : value('v2GenerateActionTemperature').trim() || timingTemperature;
    const timingBranches = mode === 'joint' || (mode === 'hybrid' && candidates > 2);
    // The generator collapses duplicate candidates when every varied decision is greedy.
    const deterministic = actionTemperature.trim() !== '' && Number(actionTemperature) === 0
      && (!timingBranches || (timingTemperature.trim() !== '' && Number(timingTemperature) === 0));
    const compare = mode !== 'event' && candidates > 1 && !deterministic;
    const search = compare
      ? `${nameOf('v2GeneratePhraseSearch')} · 每次最多比较 ${candidates} 种 · 偏好 ${nameOf('v2GeneratePhraseRankPolicy')}`
      : '逐步生成';
    const refine = !document.getElementById('v2GenerateSelfRefine')?.disabled && value('v2GenerateSelfRefine', 'off') !== 'off';
    return `写谱方式 · ${search}${refine ? ` · ${nameOf('v2GenerateSelfRefine')}` : ''}`;
  }

  const selectedMode = () => value("trainingMode", "fresh");
  const taskLabels = {review_bundle_export:'导出已确认片段',review_bundle_preview:'预览片段包',review_bundle_import:'导入片段包',candidate_budget: "训练来源学习份额估算", candidate_scan: "待审候选库扫描", preprocess: "预处理", train: "训练", generate: "生成", acceptance: "诊断", gold: "金标工具",
    chart_stream_train: "连续谱面训练", chart_stream_generate: "连续谱面生成",
    composer_v2_train: "Composer v2 训练", composer_v2_generate: "Composer v2 生成",
    composer_v2_auto_quality_audit: "两级自动素材体检", composer_v2_evaluate: "采音与选优评估",
    composer_v2_candidate_audit: "候选选优分析", composer_v2_curve: "强度曲线预测", composer_v2_media: "音频波形准备",
    composer_v2_chartgate_setup: "初始化 Chart;Gate",
    composer_train: "chart_composer_v1 训练", composer_generate: "chart_composer_v1 生成"};
  const stateLabels = {queued: "排队中", running: "运行中", cancelling: "正在取消", completed: "已完成", failed: "失败", cancelled: "已中断"};

  function reveal(target) {
    if (!target) return;
    let parent = target.parentElement;
    while (parent) {
      if (parent.tagName === "DETAILS") parent.open = true;
      parent = parent.parentElement;
    }
    target.scrollIntoView({block: "center", behavior: "smooth"});
    if (!target.disabled && typeof target.focus === "function") target.focus({preventScroll: true});
    const field = target.closest(".field, .checkbox-field, .workflow-group") || target;
    field.classList.add("parameter-highlight");
    setTimeout(() => field.classList.remove("parameter-highlight"), 1800);
  }

  function focusSetting(id) {
    const target = document.getElementById(id);
    if (!target) return;
    if (target.closest(".training-panel")) setWorkspaceMode("training");
    if (target.closest(".generation-panel")) setWorkspaceMode("generation");
    setLogsExpanded(false);
    reveal(target);
  }

  function setLogsExpanded(expanded) {
    document.body.classList.toggle("logs-expanded", expanded);
    const button = $("#expandLogsButton");
    if (button) {
      button.textContent = expanded ? "返回参数与日志并排视图" : "全宽查看日志";
      button.setAttribute("aria-expanded", String(expanded));
    }
    if (expanded) $(".task-panel")?.scrollIntoView({block: "start", behavior: "smooth"});
  }

  function fieldName(control) {
    const field = control.closest(".field, .checkbox-field, .control-row");
    if (!field) return control.getAttribute("aria-label") || control.id;
    const clone = field.cloneNode(true);
    clone.querySelectorAll("input, select, textarea, small, button, output, details").forEach((node) => node.remove());
    return clone.textContent.replace(/\s+/g, " ").trim().slice(0, 90) || control.getAttribute("aria-label") || control.id;
  }

  function isFamilyHidden(control) {
    return Boolean(control.closest("[hidden], [data-retired-curve], #generationCurveGroup, .intensity-panel"));
  }

  function searchParameters(input) {
    const scope = input.dataset.parameterSearch;
    const panel = $(`.${scope}-panel`);
    const results = document.getElementById(`${scope}ParameterResults`);
    const query = input.value.trim().toLowerCase();
    results.replaceChildren();
    results.hidden = !query;
    if (!query) return;
    const seen = new Set();
    let matches = 0;
    for (const control of panel.querySelectorAll("input[id], select[id], textarea[id]")) {
      if (control === input || control.type === "file" || isFamilyHidden(control) || control.closest(".quick-settings")) continue;
      const label = fieldName(control);
      const snake = control.id.replace(/[A-Z]/g, (letter) => "_" + letter.toLowerCase());
      const field = control.closest(".field, .checkbox-field, .control-row");
      if (!(label + " " + control.id + " " + snake + " " + (field?.textContent || "")).toLowerCase().includes(query)) continue;
      if (seen.has(field || control)) continue;
      seen.add(field || control);
      matches++;
      if (matches > 12) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "parameter-result";
      const section = control.closest(".v2-step-card, .settings-section, .workflow-group");
      const title = section?.querySelector(":scope > .v2-step-head h4, :scope > summary > span, :scope > summary, :scope > .workflow-group-heading h3")?.textContent?.trim() || "常用设置";
      button.textContent = `${label} · ${title}${control.disabled ? "（当前不可用）" : ""}`;
      button.addEventListener("click", () => { results.hidden = true; reveal(control); });
      results.append(button);
    }
    const hint = document.createElement("p");
    hint.textContent = matches ? `找到 ${matches} 项${matches > 12 ? "，先显示前 12 项，请缩小关键词" : "，点击展开并定位"}。` : "没有找到当前模型可见的参数，试试中文名称或参数英文名。";
    results.prepend(hint);
  }

  function list(id, lines) {
    const target = document.getElementById(id);
    if (!target) return;
    const text = JSON.stringify(lines);
    if (target.dataset.content === text) return;
    target.dataset.content = text;
    target.replaceChildren(...lines.filter(Boolean).map((line) => {
      const li = document.createElement("li"); li.textContent = line; return li;
    }));
  }

  function advancedCount(scope) {
    let count = 0;
    for (const control of $(`.${scope}-panel`).querySelectorAll("[data-advanced] input, [data-advanced] select, [data-advanced] textarea")) {
      if (control.disabled || control.type === "file" || isFamilyHidden(control)) continue;
      let changed;
      if (control.type === "checkbox") changed = control.checked !== control.defaultChecked;
      else if (control.tagName === "SELECT") changed = control.value !== ([...control.options].find((option) => option.defaultSelected) || control.options[0])?.value;
      else changed = control.value !== control.defaultValue;
      if (changed) count++;
    }
    const target = document.getElementById(`${scope}AdvancedCount`);
    if (target) target.textContent = count ? `${count} 项与页面初始值不同（含模型继承值）` : "当前使用页面初始值";
  }

  function refresh() {
    const mode = selectedMode();
    const checkpointGroup = $("#trainingCheckpointGroup");
    if (checkpointGroup) checkpointGroup.hidden = mode === "fresh";
    const sources = typeof selectedDatasetItems === "function" ? selectedDatasetItems() : [];
    const modeLabel = {fresh: "从头训练", resume: "从断点继续训练", finetune: "已有模型加强训练"}[mode];
    const selectedFamily = value("modelFamily");
    const limits = Number(value(selectedFamily==='chart_composer_v2'?'v2TrainMaxSteps':'maxSteps', 0));
    const formFamily = ["song_form_v1", "song_form_v2"].includes(selectedFamily);
    const streamFamily = selectedFamily === "chart_stream_v1";
    const composerFamily = selectedFamily === 'chart_composer_v1';
    const eventSetFamily = composerFamily && value('composerArchitectureProfile') === 'event_set_stream_v1';
    const chartLimit = Number(value(selectedFamily==='chart_composer_v2'?'v2TrainLimitCharts':composerFamily?'composerLimitCharts':'formLimitCharts', 0));
    list("trainingRunSummary", [
      `起点：${modeLabel}${mode !== "fresh" ? ` · ${value("trainingCheckpointPath").trim() || "尚未选择 checkpoint"}` : ""}`,
      sources.length ? `素材：${sources.map((item) => item.name || item.path).join("、")}` : `素材：当前单份清单 · ${value("manifestPath").trim() || "尚未准备"}`,
      `${eventSetFamily ? "总目标" : mode === "resume" ? "追加" : "本阶段"} ${value("epochs")} 轮 · ${composerFamily
        ? `${chartLimit>0?`最多 ${chartLimit} 张谱面`:'全部合格谱面'} · 每轮 ${Number(value('composerMaxSteps',0))||'全部'} batch`
        : streamFamily
        ? `${chartLimit > 0 ? `最多 ${chartLimit} 张谱面` : "全部合格谱面"}${limits > 0 ? ` · continuation batch 上限 ${limits}` : " · 不限 continuation batch"}`
        : formFamily
        ? `${chartLimit > 0 ? `最多 ${chartLimit} 张谱面` : "全部合格谱面"}${limits > 0 ? ` · 配置 batch 上限 ${limits}` : " · 不限配置 batch"}`
        : `每轮${limits > 0 ? `最多 ${limits} 个训练 batch（短跑）` : "完整训练集"}`}`,
      `学习率 ${value("learningRate")} · batch ${value("batchSize")} × 累积 ${value("gradAccumulation")} ≈ 每次更新 ${Number(value("batchSize")) * Number(value("gradAccumulation"))} 个片段`,
      ...(eventSetFamily ? [`结构 ${value('composerDModel')} 维 / ${value('composerLayers')} 层 / FFN ${value('composerDimFeedforward')} · 精度 ${nameOf('composerPrecision')}`,
        `历史 ${value('composerHistoryEvents')} 个事件 · 短段展开 ${value('composerRolloutSteps')} 个事件`,
        `短句：${value('composerPhraseContextPolicy')==='auto'?(value('composerLearningPolicy')==='legacy'?'关闭（旧策略）':'有序配置短句（v2）'):nameOf('composerPhraseContextPolicy')} · 类型学习：${value('eventSetPolicy_event_family_policy')==='auto'?(value('composerLearningPolicy')==='legacy'?'旧版基线（v1）':'共享续写状态（推荐）'):nameOf('eventSetPolicy_event_family_policy')}`,
        `验证：${value('composerValidationManifest').trim()?'独立验证集':Number(value('composerHoldoutFraction',0))>0?`自动留出 ${Number(value('composerHoldoutFraction'))*100}%（开发＋最终留出合计）`:'关闭自动留出，无验证集'}`,
        `后台预取 ${value('composerPrefetchBatches')} batch · 音频预算 ${value('composerAudioCacheGb')} GiB`,
        '完整续训权重 composer_last.pt；推理权重 composer_inference.pt'] : composerFamily ? [`结构 ${value('composerDModel')} 维 / ${value('composerLayers')} 层 / FFN ${value('composerDimFeedforward')} · 动作槽位 ${value('composerActionSlots')}`,
        `历史 ${value('composerHistoryEvents')} 个事件 · 候选 ${value('composerCandidateCount')} · 精度 ${nameOf('composerPrecision')}`,
        `后台预取 ${value('composerPrefetchBatches')} batch · 音频预算 ${value('composerAudioCacheGb')} GiB · 日志每 ${value('composerLogInterval')} batch`,
        `完整续训权重 composer_last.pt；额外推理权重${checked('composerExportInference')?'开启':'关闭'}`] : streamFamily ? [`种子 ${value("trainingSeed").trim() || "0"} · 权重衰减 ${value("streamWeightDecay")} · 精度 ${nameOf("streamPrecision")}`,
        `连续上下文 ${value("streamMaxTokens")} token · 近期历史 ${value("streamMaxHistoryTokens")} token · 记忆槽 ${value("streamMemoryTokens")}`,
        `计划辅助 ${value("streamPlanLossWeight")} · 音频缓存 ${value("streamAudioCacheGb")} GB · 准备线程 ${value("streamPrepareWorkers")}`] : formFamily ? [`种子 ${value("trainingSeed").trim() || "0"} · 权重衰减 ${value("weightDecay")} · 条件省略 ${value("formConditionDropout")}`,
        `全曲规划损失 ${value("formBlueprintLossWeight")} · 负荷损失 ${value("formCurveLossWeight")} · 段落关联损失 ${value("formRelationLossWeight")} · 质量加权${checked("qualityEnabled") ? "开启" : "关闭"}`,
        `精度 ${nameOf("formPrecision")} · 单次音频上限 ${value("formMaxRegionSeconds")} 秒 · 隐藏层 ${value("formDModel")} / ${value("formLayers")} 层 / ${value("formNhead")} 头`] : []),
      ...(!composerFamily ? [`验证：${value("validationManifestPath").trim() ? "使用独立验证集" : "不使用；best 按训练 loss 保存"}`,
        `质量加权${checked("qualityEnabled") ? "开启" : "关闭"} · 修正版偏好训练${checked("preferenceEnabled") ? "开启" : "关闭"}`,
        `Slide / Touch / 细碎节奏倍率：${value("slideOversample")} / ${value("touchOversample")} / ${value("fineRhythmOversample")}`] : []),
      `输出目录：${value("checkpointDir")}`,
    ]);
    if(selectedFamily==='chart_composer_v2')list("trainingRunSummary",[
      `起点：${modeLabel}${mode!=="fresh"?` · ${value("trainingCheckpointPath").trim()||"尚未选择 checkpoint"}`:""}`,
      sources.length?`素材：${sources.map(item=>item.name||item.path).join("、")}`:`素材：${value("manifestPath").trim()||"尚未准备"}`,
      `总目标 ${value("v2TrainEpochs")} 轮 · 批大小 ${value("v2TrainBatchSize")} × 累积 ${value("v2TrainGradAccumulation")}（单步等效 ${Number(value("v2TrainBatchSize"))*Number(value("v2TrainGradAccumulation"))}） · 精度 ${nameOf("v2TrainPrecision")}`,
      `显存优化：${nameOf("v2TrainActivationCheckpointing")} · CUDA 预算 ${Number(value("v2TrainCudaMemoryBudgetGb")) > 0 ? `${value("v2TrainCudaMemoryBudgetGb")} GiB` : "自动"} · 音频内存缓存 ${value("v2TrainAudioCacheGb", "0.25")} GiB`,
      `学习率 ${value("v2TrainLr")} · ${nameOf("v2TrainLrSchedule")} · 音频特征 ${nameOf("v2TrainAudioBackend")}`,
      v2DifficultySummary(),
      `素材清洗与标注：${checked("v2TrainQualityEnabled") ? `开启（${nameOf("v2TrainAutoQualityPolicy")}；手动标注优先）` : "关闭（整谱原始均匀学习）"}`,
      `验证：${mode==="resume"?"继承 checkpoint 已保存的训练与验证清单":value('v2TrainValidationSource')==='auto'?Number(value('v2TrainHoldoutFraction'))===0?'关闭（自动留出比例为0）':`自动留出 ${Math.round(Number(value('v2TrainHoldoutFraction'))*100)}%（按整首歌严格隔离）`:value('v2TrainValidationSource')==='manual'?`手动清单 ${value('v2TrainValidationManifest').trim()||'尚未选择'}`:'关闭'}`,
      mode==='resume'?'严格续训继承原训练总计划；已完成计划后，新增训练请使用新目录 finetune。':`谱面上限 ${chartLimit||'不限'} · 更新尝试上限 ${limits||'不限'}`,
      ...(mode!=='resume' && value('v2TrainValidationSource')!=='off' && (value('v2TrainValidationSource')!=='auto' || Number(value('v2TrainHoldoutFraction'))>0)
        ? [`每轮验证批次 ${Number(value('v2TrainValidationMaxBatches')) || '全部'}`] : []),
      `输出目录：${value("checkpointDir")}`]);
    const hint = $("#trainingRunHint");
    const problems = [];
    if (mode !== "fresh" && !value("trainingCheckpointPath").trim()) problems.push("先选择已有模型并检查兼容性。");
    if (Number(value("prepLimit")) > 0) problems.push(`当前设置了 ${value("prepLimit")} 首素材上限，正式训练前请确认。`);
    if (limits > 0 || ((formFamily || selectedFamily==='chart_composer_v2') && chartLimit > 0)) problems.push("当前设置了短跑上限，完成不代表已经跑完整个训练集。");
    if (hint) { hint.textContent = problems.join(" ") || "确认以上来源、轮数与输出目录后开始训练。"; hint.classList.toggle("warn", problems.length > 0); }

    const curve = checked("intensityCurveEnabled");
    const saved = typeof intensityCurveState !== "undefined" && intensityCurveState.saved && intensityCurveState.profile;
    const curveSummary = $("#generationCurveSummary");
    if (curveSummary) {
      curveSummary.textContent = curve
        ? (saved ? `使用已保存曲线 · ${nameOf("intensityCurveMode")} · 指导 ${value("intensityCurveStrength")}；重新编辑后记得保存。`
          : "已启用曲线，但还没有已保存的曲线。请打开曲线编辑，分析并保存后再生成。")
        : "未启用手动曲线。模型仍按自身规划生成；自动跟随设置在下方高级选项中。";
      curveSummary.classList.toggle("warn", curve && !saved);
    }
    const detection = checked("conflictDetectionEnabled");
    const repair = value("conflictRepairMode");
    const repairSummary = $("#generationRepairSummary");
    if (repairSummary) repairSummary.textContent = !detection ? "无理检测当前关闭；下面的检测、修复与回溯设置不会生效。"
      : repair === "local" ? `局部修复 · ${nameOf("noteRepairMode")} · 每窗重试 ${value("conflictRetries")} 次 · 前置长音：${nameOf("sustainRepairMode")}`
      : repair === "reject" ? "检测开启：存在参与处理的残留 Muri 或解析失败时拒绝候选；仅报告类别不触发拒绝。"
      : "检测开启，仅报告问题；不进行局部修复或重采样。";
    list("generationRunSummary", [
      `歌曲：${value("audioPath").trim() || "尚未选择音频"}`,
      `模型：${value("checkpointPath").trim() || "尚未选择 checkpoint"}`,
      `目标等级 ${value("level")} · ${value("candidates")} 个候选 · 种子 ${value("seed").trim() || "自动"}`,
      `时钟：${value("tempoMap").trim() ? "手动 tempo map 优先" : `BPM ${value("bpm").trim() || "自动估计"}`} · 首拍偏移 ${value("first")} 秒`,
      `无理处理：${detection ? nameOf("conflictRepairMode") : "关闭"}`,
      `输出：${value("outputPath")}`,
    ]);
    if(typeof currentGenerationFamily==="function" && currentGenerationFamily()==="chart_composer_v2")list("generationRunSummary",[
      `歌曲：${value("audioPath").trim()||"尚未选择音频"}`,
      `模型：${value("checkpointPath").trim()||"尚未选择 checkpoint"}`,
      `目标等级 ${value("level")} · 生成 ${value("v2GenerateCount","1")} 张完整谱面${Number(value("v2GenerateCount","1"))>1?`（并行 ${Math.min(Number(value("v2GenerateCount","1")),Number(value("v2GenerateParallel","1")))}）`:""} · 种子 ${value("seed").trim()||"自动"}`,
      v2GenerationSearchSummary(),
      `防无理保护：${checked("v2GenerateInterceptTapOnSlide")?`防撞尾（${value("v2GenerateTapOnSlideThresholdMs")}ms）`:"不拦撞尾"}${checked("v2GenerateInterceptSlideHeadTap")?" ＋ 防星头抢拍":""} · Touch：${nameOf("v2GenerateTouchReachabilityPolicy")}`,
      `采音温度 ${value("v2GenerateTemperature")}${document.getElementById('v2GenerateOnsetStrength')?.disabled?'':`（重音贴合 ${value("v2GenerateOnsetStrength","1")}）`} · 配置温度 ${document.getElementById('v2GenerateActionTemperature')?.disabled?"跟随采音":value("v2GenerateActionTemperature")||"跟随采音"} · 冲突恢复 ${value("v2GenerateRollbackWindows")} 次`,
      `Radar / Style：${typeof v2ConditionCapability==='function' && v2ConditionCapability().enabled ? `手动指定 ${Object.values(v2ConditionState).filter(item=>item.override).length} 项；其余自动由音乐决定` : '由音乐与等级自动决定'}`,
      `时钟：${value("tempoMap").trim()?"手动 tempo map 优先":value("bpm").trim()?`手动 BPM ${value("bpm").trim()}`:`BPM 自动（${nameOf("v2GenerateTempoDetection")}）`} · ${document.getElementById('v2GenerateAutoFirst')?.checked?'第一拍自动对准':`手动首拍 ${value("first")} 秒`}`,
      `输出：${value("outputPath")||"outputs/composer_v2/maidata.txt"}；附带报告 .composer_v2.json`]);
    advancedCount("training");
  }

  function applyValues(changes, message) {
    for (const [id, next] of Object.entries(changes)) {
      const target = document.getElementById(id);
      if (!target || target.disabled) continue;
      target.value = String(next);
      target.dispatchEvent(new Event("input", {bubbles: true}));
      target.dispatchEvent(new Event("change", {bubbles: true}));
    }
    $("#trainingRecipeStatus").textContent = message;
    refresh();
  }

  function renderTasks(incoming = []) {
    incoming.forEach((task) => { if (task?.task_id) tasks.set(task.task_id, {...tasks.get(task.task_id), ...task}); });
    const select = $("#taskSelector");
    if (!select) return;
    const selected = typeof activeTaskId !== "undefined" ? activeTaskId : select.value;
    const options = [...tasks.values()].reverse();
    const signature = JSON.stringify(options.map((task) => [task.task_id, task.status, task.message]));
    if (select.dataset.signature !== signature) {
      select.dataset.signature = signature;
      select.replaceChildren(...(options.length ? options.map((task) => {
        const option = document.createElement("option"); option.value = task.task_id;
        option.textContent = `${taskLabels[task.kind] || task.kind} · ${stateLabels[task.status] || task.status} · ${task.task_id}`;
        return option;
      }) : [new Option("暂无任务", "")]));
    }
    if (selected && tasks.has(selected)) select.value = selected;
  }

  function reportError(message) {
    const scope = ["training", "generation", "quality", "intensity"].includes(document.body.dataset.workspace)
      ? document.body.dataset.workspace : "training";
    const panel = $(`.${scope}-panel`);
    if (!panel) return;
    let alert = panel.querySelector(".workflow-error");
    if (!alert) { alert = document.createElement("p"); alert.className = "workflow-error"; alert.setAttribute("role", "alert"); panel.prepend(alert); }
    alert.textContent = message;
    alert.hidden = false;
    alert.scrollIntoView({block: "nearest", behavior: "smooth"});
  }

  window.MaichartWorkflow = {refresh, renderTasks, recordTask: (task) => renderTasks([task]), reveal, focusSetting, setLogsExpanded, reportError};
  document.querySelectorAll("[data-parameter-search]").forEach((input) => {
    input.addEventListener("input", () => searchParameters(input));
    input.addEventListener("keydown", (event) => { if (event.key === "Escape") { input.value = ""; searchParameters(input); } });
  });
  document.querySelectorAll("[data-expand-settings], [data-collapse-settings]").forEach((button) => button.addEventListener("click", () => {
    const scope = button.dataset.expandSettings || button.dataset.collapseSettings;
    document.querySelectorAll(`.${scope}-panel details[data-advanced]`).forEach((section) => { section.open = Boolean(button.dataset.expandSettings); });
  }));
  document.querySelectorAll("[data-focus-setting]").forEach((button) => button.addEventListener("click", () => focusSetting(button.dataset.focusSetting)));
  document.querySelectorAll("[data-workspace-link]").forEach((button) => button.addEventListener("click", () => setWorkspaceMode(button.dataset.workspaceLink)));
  document.querySelectorAll("[data-open-task-logs]").forEach((button) => button.addEventListener("click", () => setLogsExpanded(true)));
  $("#expandLogsButton")?.addEventListener("click", () => setLogsExpanded(!document.body.classList.contains("logs-expanded")));
  $("#taskSelector")?.addEventListener("change", (event) => { if (event.target.value) watchTask(event.target.value); });
  $("#refreshTaskSelector")?.addEventListener("click", async () => { try { renderTasks((await api("/api/status")).tasks || []); } catch (error) { reportError(error.message); } });
  $("#applyTrainingRunRecipe")?.addEventListener("click", () => {
    const composer=value('modelFamily')==='chart_composer_v1';
    if (value("trainingRunRecipe") === "smoke") applyValues({epochs: 1, [composer?'composerMaxSteps':'maxSteps']:32, validationMaxSteps: 2}, "已应用短跑：1 轮、最多 32 个训练 batch。没有启动训练。");
    else applyValues({[composer?'composerMaxSteps':'maxSteps']:0, validationMaxSteps: 0}, `已取消 batch 上限，训练轮数仍为 ${value("epochs")}；请按本次目标设置轮数。没有启动训练。`);
  });
  $("#applyTrainingBatchRecipe")?.addEventListener("click", () => {
    const [batchSize, gradAccumulation] = value("trainingBatchRecipe").split(",").map(Number);
    applyValues({batchSize, gradAccumulation}, `已应用 batch ${batchSize}、梯度累积 ${gradAccumulation}。模型结构与学习率未改动。`);
  });
  // Verify external workbench buttons wired in candidate_feedback.js / v2_intensity.js / review_exchange.js / chart_preview.js.
  const externalModuleButtons = [
    document.getElementById("trainingPreviewBrowse"),
    document.getElementById("trainingPreviewPrevious"),
    document.getElementById("trainingPreviewNext"),
    document.getElementById("trainingPreviewCatalog"),
    document.getElementById("generationPreviewExpand"),
    document.getElementById("generationPreviewCollapse"),
    document.getElementById("generationReviewToolsToggle"),
    document.getElementById("generationPreviewSwitchSource"),
    document.getElementById("feedbackZoomIn"),
    document.getElementById("feedbackZoomStepOut"),
    document.getElementById("feedbackZoomOut"),
    document.getElementById("feedbackDetailView"),
    document.getElementById("feedbackPlayRange"),
    document.getElementById("feedbackPause"),
    document.getElementById("feedbackClear"),
    document.getElementById("feedbackMarkStart"),
    document.getElementById("feedbackMarkEnd"),
    document.getElementById("candidatePoolWhole"),
    document.getElementById("candidatePoolReject"),
    document.getElementById("candidatePoolNextChart"),
    document.getElementById("candidatePoolUndo"),
    document.getElementById("candidatePoolUsePreview"),
    document.getElementById("feedbackPreviewA"),
    document.getElementById("feedbackPreviewB"),
    document.getElementById("feedbackCollectTraining"),
    document.getElementById("feedbackExportBundle"),
    document.getElementById("feedbackAddTraining"),
    document.getElementById("candidatePoolBudget"),
    document.getElementById("candidatePoolBudgetCancel"),
    document.getElementById("reviewBundleExport"),
    document.getElementById("reviewBundleChoose"),
    document.getElementById("reviewBundleApply"),
    document.getElementById("reviewBundleOpen"),
    document.getElementById("reviewBundleAddTraining"),
    document.getElementById("candidatePoolNext"),
    document.getElementById("candidatePoolSync"),
    document.getElementById("candidatePoolExcluded"),
    document.getElementById("candidatePoolUse"),
    document.getElementById("qualityOpenReview"),
    document.getElementById("qualitySpanStart"),
    document.getElementById("qualitySpanEnd"),
    document.getElementById("qualitySpanWaveLoad"),
    document.getElementById("qualitySpanZoomIn"),
    document.getElementById("qualitySpanZoomOut"),
    document.getElementById("v2CurvePredict"),
    document.getElementById("v2CurveUndo"),
    document.getElementById("v2CurveReset"),
    document.getElementById("v2CurveExport"),
    document.getElementById("v2CurveLoadAudio"),
    document.getElementById("v2CurveZoomIn"),
    document.getElementById("v2CurveZoomOut"),
    document.getElementById("v2CurveFit"),
    document.getElementById("v2CurvePlayView"),
    document.getElementById("v2CurveSelectView"),
    document.getElementById("v2CurveClearSelection"),
    document.getElementById("v2CurveAdd"),
    document.getElementById("v2CurveDelete"),
    document.getElementById("v2CurveBatchApply"),
    document.getElementById("v2CurvePlayPoint"),
  ];
  void externalModuleButtons;
  let pending;
  ["input", "change"].forEach((type) => document.addEventListener(type, () => { clearTimeout(pending); pending = setTimeout(refresh, 0); }));
  refresh();
})();
