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
  const selectedMode = () => value("trainingMode", "fresh");
  const taskLabels = {preprocess: "预处理", train: "训练", generate: "生成", acceptance: "诊断", gold: "金标工具",
    chart_stream_train: "连续谱面训练", chart_stream_generate: "连续谱面生成",
    composer_train: "chart_composer_v1 训练", composer_generate: "chart_composer_v1 生成"};
  const stateLabels = {queued: "排队中", running: "运行中", completed: "已完成", failed: "失败", cancelled: "已中断"};

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
    return Boolean(control.closest("[hidden]"));
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
      const section = control.closest(".settings-section, .workflow-group");
      const title = section?.querySelector(":scope > summary > span, :scope > .workflow-group-heading h3")?.textContent || "常用设置";
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
    const limits = Number(value("maxSteps", 0));
    const selectedFamily = value("modelFamily");
    const formFamily = ["song_form_v1", "song_form_v2"].includes(selectedFamily);
    const streamFamily = selectedFamily === "chart_stream_v1";
    const composerFamily = selectedFamily === 'chart_composer_v1';
    const eventSetFamily = composerFamily && value('composerArchitectureProfile') === 'event_set_stream_v1';
    const chartLimit = Number(value(composerFamily?'composerLimitCharts':'formLimitCharts', 0));
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
        `短句：${value('composerPhraseContextPolicy')==='auto'?(value('composerLearningPolicy')==='legacy'?'关闭（旧策略）':'有序配置短句（v2）'):nameOf('composerPhraseContextPolicy')} · 类型学习：${value('eventSetPolicy_event_family_policy')==='auto'?(value('composerLearningPolicy')==='legacy'?'旧版基线（v1）':'音频先验＋持续状态与短句（v4）'):nameOf('eventSetPolicy_event_family_policy')}`,
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
    const hint = $("#trainingRunHint");
    const problems = [];
    if (mode !== "fresh" && !value("trainingCheckpointPath").trim()) problems.push("先选择已有模型并检查兼容性。");
    if (Number(value("prepLimit")) > 0) problems.push(`当前设置了 ${value("prepLimit")} 首素材上限，正式训练前请确认。`);
    if (limits > 0 || (formFamily && chartLimit > 0)) problems.push("当前设置了短跑上限，完成不代表已经跑完整个训练集。");
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
      `曲线${curve ? (saved ? "已启用并保存" : "已启用，待保存") : "未启用"} · 无理处理：${detection ? nameOf("conflictRepairMode") : "关闭"}`,
      `输出：${value("outputPath")}`,
    ]);
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
  let pending;
  ["input", "change"].forEach((type) => document.addEventListener(type, () => { clearTimeout(pending); pending = setTimeout(refresh, 0); }));
  refresh();
})();
