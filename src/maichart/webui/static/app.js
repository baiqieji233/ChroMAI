"use strict";

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
  if (!phase && ["queued", "running"].includes(task?.status)) {
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

function renderTaskTrainingMetrics(task) {
  const panel = element("#taskTrainingMetrics");
  const source = element("#taskTrainingMetricsSource");
  const groups = element("#taskTrainingMetricsGroups");
  if (!panel || !groups) return;
  groups.replaceChildren();
  if (task?.kind !== "form_train") {
    panel.hidden = true;
    return;
  }

  const events = trainingMetricEvents(task);
  const livePhase = events.phase;
  const live = ["queued", "running"].includes(task.status) && livePhase !== null;
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
  if (!task) return null;
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
  if (["queued", "running"].includes(task.status) || result.last_epoch !== undefined) {
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
  "tempoMode", "division",                                  // 节拍与时间轴
  "level", "conditionPreferenceStrength",                   // 等级基准与叠加偏好
  "preferKeyboard", "preferStars", "preferTechnical", "preferChord", "preferHold", "preferTouch", "preferMovement",
  "generationDevice",                                       // 生成运行设备
  "formPositionComplexity", "formPositionBeamSize", "formEarlyEosPenalty",
  "seed", "candidates", "temperature", "sampleEvents",       // 采样与随机性
  "topP", "repetitionPenalty", "repetitionWindow", "maxTokens",
  "intensityFollow", "intensityFollowSpan",                  // 逐窗口强度跟随
  "intensityMode", "intensityCurveEnabled",
  "intensityCurveStrength", "intensityCurveMode", "intensityCurveSource",
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
const qualityState = { items: [], page: 1, pageSize: 25, pages: 1, total: 0, folderTotal: 0, query: "", label: "", annotationPath: "", annotationManifest: "", summary: null, selected: new Set(), editedWeights: new Map(), selectionRequest: 0, selectionLoading: false, saving: false, detail: null, regions: [], correctedPath: "", catalogRequest: 0, detailRequest: 0 };
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
    && ["queued", "running", "completed", "failed", "cancelled"].includes(data.status);
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
  const value = ["generation", "intensity", "quality"].includes(mode) ? mode : "training";
  document.body.dataset.workspace = value;
  element("#trainingModeTab")?.classList.toggle("active", value === "training");
  element("#qualityModeTab")?.classList.toggle("active", value === "quality");
  element("#generationModeTab")?.classList.toggle("active", value === "generation");
  element("#intensityModeTab")?.classList.toggle("active", value === "intensity");
  if (value === "intensity") requestAnimationFrame(drawIntensityCurve);
  window.MaichartWorkflow?.setLogsExpanded(false);
  window.MaichartWorkflow?.refresh();
  try { localStorage.setItem(workspaceStorageKey, value); } catch (_) { /* storage disabled */ }
}

function initialWorkspace() {
  try { return localStorage.getItem(workspaceStorageKey) || "training"; } catch (_) { return "training"; }
}

function loadIntensityCurveState() {
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
      contextVersion: 2,
      conditionChanged: intensityCurveState.conditionChanged === true,
    }));
  } catch (_) { /* curve is still usable for this page */ }
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------
function settingsFields() {
  return Array.from(document.querySelectorAll("input[id], select[id], textarea[id]"))
    .filter((element) => element.type !== "file" && element.id !== "datasetSelect"
      && !["taskSelector", "trainingRunRecipe", "trainingBatchRecipe"].includes(element.id)
      && !element.id.endsWith("ParameterSearch")
      && !element.id.startsWith("intensitySection"));
}

function collectSettings() {
  const values = {};
  settingsFields().forEach((target) => {
    values[target.id] = target.type === "checkbox" ? target.checked : target.value;
  });
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
  migrateSongFormQualityDefaults(values);
  let restored = 0;
  settingsFields().forEach((target) => {
    if (!Object.prototype.hasOwnProperty.call(values, target.id)) return;
    const value = values[target.id];
    if (target.type === "checkbox") target.checked = value === true;
    else if (value !== undefined && value !== null) target.value = String(value);
    restored += 1;
  });
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
  if (levelPresetProfile) applyConditionPreset();
  updateGenerationCapabilityControls();
  saveSettings();
  const status = element("#resetGenerationStatus");
  if (status) status.textContent = "已恢复生成面板默认值；训练参数未改动。";
}

function setConditionValue(key, value) {
  const target = document.getElementById(key);
  if (!target) return;
  target.value = String(value);
  const mirror = document.querySelector(`[data-number-for="${key}"]`);
  if (mirror) mirror.value = String(value);
}

function applyConditionPreset() {
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
    if (!restoredSettings && !conditionEditedBeforeProfile) applyConditionPreset();
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
  return valueOrDefault("#modelFamily", "song_form_v2");
}

function currentGenerationFamily() {
  const selected = currentModelFamily();
  const inspected = generationInspection.data?.model_family;
  return inspected && generationInspection.path === valueOrDefault("#checkpointPath", "").trim() ? inspected : selected;
}

function updateGenerationCapabilityControls() {
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
  const curveMode = element("#intensityCurveMode");
  const legacyCurveModes = curveMode ? [...curveMode.options].filter((option) => ["soft", "strict"].includes(option.value)) : [];
  legacyCurveModes.forEach((option) => { option.disabled = isSongForm; });
  const hardMode = curveMode?.querySelector('option[value="hard"]');
  if (hardMode) hardMode.textContent = isSongForm ? "hard（放宽幅度裁剪）" : "hard（严苛跟随）";
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
}

function updateFamilyFields() {
  document.querySelectorAll("[data-training-family]").forEach((target) => {
    target.hidden = !familyFieldVisible(target.dataset.trainingFamily, currentModelFamily());
  });
  document.querySelectorAll("[data-generation-family]").forEach((target) => {
    target.hidden = !familyFieldVisible(target.dataset.generationFamily, currentGenerationFamily());
  });
  const songTraining = ["song_form_v1", "song_form_v2"].includes(currentModelFamily());
  ["#dModel", "#nhead", "#numLayers", "#dimFeedforward", "#trainMaxTokens", "#segmentBeats", "#maxAudioTokens",
   "#conditionMaskProb", "#conditionDefault", "#conditionDropout", "#contextDropout", "#dropout", "#weightDecay",
   "#plannerProfile", "#planLossWeight", "#planTeacherMix", "#onsetProfile", "#onsetLossWeight", "#actionStateProfile",
   "#localPoolingPolicy", "#noAmp", "#audioCacheGb", "#cudaMemoryBudgetGb", "#prewarmWorkers", "#maxSteps", "#validationMaxSteps",
   "#minDifficultyIndex", "#maxDifficultyIndex", "#minLevel", "#maxLevel", "#slideOversample", "#touchOversample", "#fineRhythmOversample",
   "#fineRhythmRatioThreshold", "#contextBeats", "#audioMarginBeats", "#contextTokens", "#overlapBeats",
   "#preferenceEnabled", "#preferenceWeight", "#preferenceMaxPairs"].forEach((selector) => {
    const node = element(selector); const field = node?.closest(".field, .checkbox-field");
    if (field && songTraining && !field.dataset.trainingFamily) field.hidden = true;
    else if (field && !songTraining && !field.dataset.trainingFamily) field.hidden = false;
  });
  const songHint = element("#songFormTrainingControls"); if (songHint) songHint.hidden = !songTraining;
  const songGenerationHint = element("#songFormGenerationControls"); if (songGenerationHint) songGenerationHint.hidden = !["song_form_v1", "song_form_v2"].includes(currentGenerationFamily());
  if (songTraining) {
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
    familyHelp.textContent = family === "song_form_v2"
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
    dirHelp.textContent = ["song_form_v1", "song_form_v2"].includes(family)
      ? "song-form 产物为 form_last.pt；resume 严格继承合同，finetune 请使用新的输出目录。"
      : family === "legacy_frame_v2"
        ? "旧基线输出 generator_last.pt / generator_epoch_NNN.pt，不能作为 event checkpoint 使用。"
        : "event_last.pt / event_best.pt / event_epoch_NNN.pt。建议使用独立输出目录。";
  }
  const bpmHelp = element("#bpmHelp");
  if (bpmHelp) {
    bpmHelp.textContent = currentGenerationFamily() === "legacy_frame_v2"
      ? "旧基线：留空自动分析；填写则覆盖为单一 BPM，tempo map 可进一步覆盖。"
      : "未填 tempo map 时，留空使用自动估计 BPM，填写则采用该值；auto 会额外使用高置信度变速点。填写 tempo map 后，其 beat 0 BPM 与变速点优先。";
  }
  const preprocessHelp = element("#preprocessFamilyStatus");
  if (preprocessHelp) preprocessHelp.textContent = ["song_form_v1", "song_form_v2"].includes(family)
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
function generationPayload() {
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
function setTrainingModeDefaults() {
  const mode = currentTrainingMode();
  const learningRate = element("#learningRate");
  if (learningRate && learningRate.dataset.touched !== "1") {
    learningRate.value = mode === "finetune" ? "0.00002" : "0.0002";
  }
  const help = element("#learningRateHelp");
  if (help) help.textContent = mode === "finetune" ? "finetune 默认 2e-5。" : "fresh 默认 2e-4。";
  updateTrainingModeHelp(mode);
}

function updateTrainingModeHelp(mode = currentTrainingMode()) {
  const songForm = ["song_form_v1", "song_form_v2"].includes(element("#modelFamily")?.value);
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
        ? songForm
          ? "本次继续训练的轮数，从 checkpoint 已完成轮次的下一轮开始；迁移时重跑未完成轮次，保留其中已学到的权重。"
          : "resume 的 epochs 是从断点额外训练的轮数；会从 checkpoint 的 epoch 接着计数，早停会按验证 loss 停止。"
        : "finetune 从新阶段 epoch 1 开始计数；epochs 是本次加强训练的轮数，早停会按验证 loss 停止。";
  }
}

function updateDatasetButtons() {
  const busy = datasetUploading || preprocessStarting || trainingStarting || trainingPreflight;
  const mapping = {
    "#datasetComposeButton": busy || datasetWorkspaceState.applying,
    "#trainingDatasetApplyButton": busy || datasetWorkspaceState.applying,
    "#preprocessButton": busy,
    "#inspectManifestButton": busy,
    "#trainButton": busy,
    "#autoPipelineButton": busy || selectedTrainingManifests().length > 1,
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
  const needsCheckpoint = currentTrainingMode() !== "fresh";
  ["#trainingCheckpointFile", "#trainingCheckpointPath", "#checkpointInspectButton"].forEach((selector) => {
    const target = element(selector);
    if (target) target.disabled = !needsCheckpoint || busy;
  });
  const resetOptimizer = element("#resetOptimizer");
  if (resetOptimizer) resetOptimizer.disabled = currentTrainingMode() !== "finetune" || busy;
}

function updateTrainingControls() {
  updateFamilyFields();
  updateDatasetButtons();
  updateTrainingModeHelp();
  const resume = currentTrainingMode() === "resume";
  const resumePolicy = element("#formResumeDataPolicyGroup");
  if (resumePolicy) resumePolicy.hidden = !resume || !["song_form_v1", "song_form_v2"].includes(currentModelFamily());
  ["#onsetProfile", "#onsetLossWeight", "#localPoolingPolicy", "#actionStateProfile"].forEach((selector) => {
    const target = element(selector);
    if (target) target.disabled = resume;
  });
  updateSongFormRhythmGeometryControls();
  updateSongFormQualityControls();
  if (typeof updateSongFormQualityProfileControls === "function") updateSongFormQualityProfileControls();
  window.MaichartWorkflow?.refresh();
}

// ---------------------------------------------------------------------------
// Task handling
// ---------------------------------------------------------------------------
const TASK_KIND_LABEL = {
  preprocess: "预处理", train: "训练", generate: "生成", form_train: "song-form 训练", form_generate: "song-form 生成", form_analyze: "song-form 分析", form_revise: "song-form 修订", intensity: "强度曲线", gold: "金标集", acceptance: "验收",
};
const taskWatchers = new Map();
const trainingMetricSupportCache = new Map();

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

function watchTask(taskId, onFinished = null) {
  activeTaskId = taskId;
  const existing = taskWatchers.get(taskId);
  if (existing) {
    if (onFinished) existing.callbacks.add(onFinished);
    if (existing.task) renderTask(existing.task);
    return;
  }
  taskWatchers.set(taskId, { callbacks: new Set(onFinished ? [onFinished] : []), task: null });
  pollTask(taskId);
}

async function pollTask(taskId) {
  const watcher = taskWatchers.get(taskId);
  if (!watcher) return;
  try {
    const task = await api(`/api/task?id=${encodeURIComponent(taskId)}`);
    watcher.task = task;
    window.MaichartWorkflow?.recordTask(task);
    if (activeTaskId === taskId) renderTask(task);
    if (["queued", "running"].includes(task.status)) {
      setTimeout(() => pollTask(taskId), 800);
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
  if (status) status.textContent = ({queued: "排队中", running: "运行中", completed: "已完成", failed: "失败", cancelled: "已中断", archived: "历史记录"})[task.status] || task.status;
  if (progress) {
    const unknownTrainingTarget = task.kind === "train" && !epochStatus?.knownTarget;
    if (progress.parentElement) progress.parentElement.hidden = unknownTrainingTarget || task.status === "archived";
    const amount = task.kind === "train" && epochStatus?.knownTarget
      ? epochStatus.progress : task.progress || 0;
    progress.style.width = `${Math.round(amount * 100)}%`;
  }
  if (message) message.textContent = task.error || task.message || "";
  if (epoch) {
    epoch.hidden = !epochStatus;
    if (epochStatus) epoch.textContent = epochStatus.text + (
      task.result?.selection_metric === "training_loss" ? "；best 依据训练 loss（未配置验证集）"
      : task.result?.selection_metric === "validation_loss" ? "；best 依据验证 loss" : "");
  }
  hydrateTrainingMetricHistory(task);
  renderTaskTrainingMetrics(task);
  if (window.MaichartLogs) window.MaichartLogs.renderTask(task);
  else if (logs) logs.textContent = (task.logs || []).join("\n") || "暂无日志";
  window.MaichartWorkflow?.recordTask(task);

  const cancel = element("#cancelButton");
  if (cancel) cancel.disabled = !["queued", "running"].includes(task.status);
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

function renderTaskResults(task) {
  const container = element("#taskResults");
  if (!container) return;
  container.replaceChildren();
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
}

function datasetKey(item) { return String(item?.manifest || item?.path || ""); }

function datasetCardHtml(item) {
  const key = datasetKey(item);
  const checked = datasetWorkspaceState.selected.has(key);
  const active = item.path === datasetWorkspaceState.activePath || item.manifest === datasetWorkspaceState.activePath;
  const charts = item.chart_count ?? item.charts ?? item.difficulty_count ?? "?";
  const preprocessAction = item.prepared ? "" : `<button class="small-button dataset-card-preprocess" type="button" data-dataset-path="${escapeHtml(item.path || "") }">去训练页预处理</button>`;
  return `<article class="dataset-card${active ? " active" : ""}" data-dataset-path="${escapeHtml(item.path || "")}" data-dataset-manifest="${escapeHtml(item.manifest || "")}">
    <label class="dataset-card-check"><input class="dataset-select-checkbox" type="checkbox" data-dataset-key="${escapeHtml(key)}" ${checked ? "checked" : ""}> <strong>加入训练</strong></label>
    <button class="dataset-card-open" type="button" data-dataset-path="${escapeHtml(item.path || "")}"><strong>${escapeHtml(item.name || "未命名数据集")}</strong></button>
    <div class="dataset-card-meta">${item.prepared ? "已预处理" : "待预处理"} · 谱面 ${escapeHtml(String(charts))}</div>
    <div class="dataset-card-path">${escapeHtml(item.path || item.manifest || "")}</div>${preprocessAction}
  </article>`;
}

function renderDatasetWorkspace() {
  const cards = element("#datasetCards");
  if (!cards) return;
  cards.innerHTML = datasetWorkspaceState.items.map(datasetCardHtml).join("") || `<div class="muted dataset-empty">尚未发现可用数据集。请先上传 ZIP，或在训练页填写原始数据集目录并预处理。</div>`;
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
  if (status && datasetWorkspaceState.items.length) status.textContent = `共发现 ${datasetWorkspaceState.items.length} 个数据集；点击卡片查看目录，勾选后用于训练。`;
  const cardsPanel = element("#datasetCardsPanel");
  const catalogPanel = element("#datasetCatalogPanel");
  if (cardsPanel) cardsPanel.hidden = datasetWorkspaceState.view !== "cards";
  if (catalogPanel) catalogPanel.hidden = datasetWorkspaceState.view !== "catalog";
  updateDatasetButtons();
}

function activeDatasetItem() {
  return datasetWorkspaceState.items.find((item) => item.path === datasetWorkspaceState.activePath || item.manifest === datasetWorkspaceState.activePath) || null;
}

async function loadDatasetCatalog() {
  const request = ++datasetWorkspaceState.catalogRequest;
  try {
    const data = await api("/api/datasets/catalog", {
      current_manifest: valueOrDefault("#manifestPath", "").trim() || null,
      manifests: [...datasetWorkspaceState.selected].filter((path) => String(path).toLowerCase().endsWith(".json")),
    });
    if (request !== datasetWorkspaceState.catalogRequest) return;
    const items = Array.isArray(data.datasets) ? data.datasets : (Array.isArray(data.items) ? data.items : null);
    if (items) {
      datasetWorkspaceState.items = items.map((item) => ({ ...item, path: item.path || item.raw_root || item.root || "", manifest: item.manifest || item.manifest_path || "", manifest_hint: item.manifest || item.manifest_path || "", name: item.name || item.dataset_name || item.path || "未命名数据集" })).filter((item) => item.path || item.manifest);
      const known = new Set(datasetWorkspaceState.items.map(datasetKey));
      datasetWorkspaceState.selected = new Set([...datasetWorkspaceState.selected].filter((key) => known.has(key)));
      if (!activeDatasetItem()) { datasetWorkspaceState.activePath = ""; datasetWorkspaceState.view = "cards"; }
      renderDatasetWorkspace();
    }
  } catch (_) {
    // Older servers expose the same uploaded list through /api/status.
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

async function openDataset(item) {
  if (!item) return;
  if (item.prepared === false || !item.manifest) throw new Error("这个数据集尚未预处理，请点击卡片上的“去训练页预处理”。");
  datasetWorkspaceState.activePath = item.path || item.manifest || "";
  datasetWorkspaceState.view = "catalog";
  renderDatasetWorkspace();
  saveDatasetWorkspace();
  qualityState.catalogRequest += 1;
  qualityState.detailRequest += 1;
  qualityState.items = []; qualityState.page = 1; qualityState.pages = 1; qualityState.total = 0;
  qualityState.folderTotal = 0; qualityState.selectionRequest += 1; qualityState.selectionLoading = false; qualityState.saving = false; qualityState.editedWeights.clear();
  qualityState.selected.clear(); qualityState.detail = null; qualityState.regions = []; qualityState.correctedPath = "";
  qualityState.annotationPath = ""; qualityState.annotationManifest = ""; qualityState.summary = null;
  renderQualityCatalog();
  if (element("#qualityDetail")) element("#qualityDetail").hidden = true;
  if (element("#qualityDetailAudio")) { element("#qualityDetailAudio").removeAttribute("src"); element("#qualityDetailAudio").load(); }
  if (element("#qualityCorrectedPath")) element("#qualityCorrectedPath").textContent = "";
  if (item.prepared === false || !item.manifest) throw new Error("这个数据集还没有预处理 manifest，请先点击卡片上的“去训练页预处理”。");
  if (element("#datasetCatalogTitle")) element("#datasetCatalogTitle").textContent = `${item.name || "数据集"} · 谱面目录`;
  renderDatasetWorkspace();
  await loadQualityCatalog(true);
}

function showDatasetCards() {
  datasetWorkspaceState.view = "cards";
  qualityState.catalogRequest += 1;
  qualityState.detailRequest += 1;
  renderDatasetWorkspace();
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
async function loadStatus() {
  const state = await api("/api/status");
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
  const intensityNotice = state.intensity_curve_capability === true ? " · 多维强度曲线可用" : " · 多维强度曲线需重启后端";
  setStatus(`本地服务正常 · workspace: ${state.workspace}${intensityNotice}`, state.intensity_curve_capability === true);
  populateDatasets(state.datasets || []);
  await loadDatasetCatalog();
  const running = (state.tasks || []).find((task) => ["queued", "running"].includes(task.status)
    && !String(task.kind || "").startsWith("intensity"));
  window.MaichartWorkflow?.renderTasks(state.tasks || []);
  window.MaichartWorkflow?.refresh();
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
    `${data.event_manifest ? "event QC " : ""}可用难度 ${data.usable_difficulty_count} 个，过滤 ${data.filtered_difficulty_count} 个`,
    `过滤原因：${reasonText}`,
    `音频状态：${Object.entries(data.audio_statuses || {}).map(([key, count]) => `${key}=${count}`).join(", ") || "无"}`,
  ];
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
  if (!checkpointPath) throw new Error("请先上传或填写已有模型 checkpoint 路径");
  const manifest = trainingSourcePaths({ autoPreprocess }).manifest;
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
  if (["song_form_v1", "song_form_v2"].includes(currentModelFamily())) {
    const data = await api("/api/form/inspect", { checkpoint: checkpointPath, operation: "generate", mode: "fresh" });
    if (request !== generationInspectRequest) return null;
    generationInspection = { path: checkpointPath, data }; updateFamilyFields();
    const conditionNote = data.condition_controls_trained ? "Radar / Style 条件监督已训练" : "Radar / Style 仍是未校准条件，请先做 A/B 验证";
    const curveNote = data.load_curve_trained ? "song_form 强度曲线已训练" : "song_form 强度曲线尚未训练";
    const curveBox = element("#formCurveCapability"); if (curveBox) { curveBox.hidden = false; curveBox.textContent = `${curveNote}；${conditionNote}。`; curveBox.classList.toggle("warn", !data.load_curve_trained); }
    const unreadyNote = data.planner_trained === false
      ? "Planner 尚无有效负荷监督，暂不能生成；请用非零规划与曲线损失完成训练。"
      : "song-form checkpoint 尚未完成事件训练";
    setGenerationStatus(data.valid && data.can_generate ? `模型族：${data.model_family}；epoch ${data.epoch ?? 0}；全曲规划 ${data.blueprint_steps || 0} 次；生成可用；${conditionNote}；${curveNote}` : ((data.errors || []).join("；") || unreadyNote), data.valid && data.can_generate ? "good" : "warn");
    return data;
  }
  const data = await api("/api/checkpoint/inspect", { checkpoint_path: checkpointPath, mode: "generate" });
  if (request !== generationInspectRequest) return null;
  generationInspection = { path: checkpointPath, data };
  updateFamilyFields();
  const family = data.model_family_label || data.model_family || "未知";
  const genCaps = data.generation_capabilities || {};
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

function qualityRowHtml(item) {
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
  return `<tr data-chart-id="${id}"><td><input class="quality-row-select" type="checkbox" data-chart-id="${id}" ${qualityState.selected.has(selectedId) ? "checked" : ""} ${disabled}></td>`
    + `<td><strong>${escapeHtml(title)}</strong><br><span class="muted">难度 ${escapeHtml(String(difficulty))}</span><br><span class="dataset-row-folder">文件夹：${escapeHtml(folder)}</span>${source ? `<br><span class="muted">来源：${escapeHtml(source)}</span>` : ""}</td>`
    + `<td>${escapeHtml(item.designer || item.chart_designer || "未标注")}</td><td><span class="quality-label ${escapeHtml(label)}">${escapeHtml(qualityLabelText(label))}</span></td>`
    + `<td><input class="quality-weight" data-chart-id="${id}" type="number" min="0" max="4" step="0.05" value="${Number.isFinite(weight) ? weight : 1}" ${disabled}></td>`
    + `<td><button class="small-button quality-detail-button" type="button" data-chart-id="${id}">查看</button></td></tr>`;
}

function renderQualityCatalog() {
  const body = element("#qualityCatalogBody"); if (!body) return;
  body.innerHTML = qualityState.items.map(qualityRowHtml).join("") || `<tr><td colspan="6" class="muted">没有匹配的谱面。</td></tr>`;
  const pageCount = Math.max(1, qualityState.pages || Math.ceil(qualityState.total / qualityState.pageSize));
  if (element("#qualityPageStatus")) element("#qualityPageStatus").textContent = `第 ${qualityState.page} / ${pageCount} 页（共 ${qualityState.total} 张）`;
  if (element("#qualityPageInput")) element("#qualityPageInput").value = String(qualityState.page);
  if (element("#qualityPrevButton")) element("#qualityPrevButton").disabled = qualityState.page <= 1;
  if (element("#qualityNextButton")) element("#qualityNextButton").disabled = qualityState.page >= pageCount;
  updateQualitySelectionControls();
}

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
  const data = await api("/api/quality/catalog", { manifest, query: qualityState.query, page: qualityState.page, page_size: qualityState.pageSize, label_filter: qualityState.label || null, sort_by: sortBy, sort_order: sortDirection });
  if (request !== qualityState.catalogRequest || manifest !== qualityManifest()) return;
  qualityState.items = Array.isArray(data.items) ? data.items : [];
  qualityState.page = Math.max(1, Number(data.page ?? qualityState.page) || 1);
  qualityState.pageSize = Math.max(1, Number(data.page_size ?? qualityState.pageSize) || qualityState.pageSize);
  qualityState.pages = Math.max(1, Number(data.pages ?? Math.ceil(Number(data.total || 0) / qualityState.pageSize)) || 1);
  qualityState.total = Number(data.total || 0);
  qualityState.folderTotal = Number(data.folder_total ?? data.total ?? 0);
  qualityState.annotationPath = data.annotation_path || qualityState.annotationPath || "";
  qualityState.annotationManifest = manifest;
  qualityState.summary = data.summary || null;
  renderQualitySummary(); renderQualityCatalog();
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
  const audio = element("#qualityDetailAudio"); if (audio && detail.audio_url) audio.src = detail.audio_url;
  const list = element("#qualityRegionsList"); if (!list) return;
  list.innerHTML = qualityState.regions.map((region, index) => `<div class="quality-region-row" data-region-index="${index}"><input type="number" min="0" step="0.01" value="${Number(region.start_sec ?? region.start ?? 0)}" data-region-field="start_sec"><input type="number" min="0" step="0.01" value="${Number(region.end_sec ?? region.end ?? 0)}" data-region-field="end_sec"><input type="number" min="0" max="4" step="0.05" value="${Number(region.weight ?? 0)}" data-region-field="weight"><input type="text" value="${escapeHtml(region.reason || "")}" data-region-field="reason" placeholder="理由"><button class="small-button quality-remove-region" type="button">删除</button></div>`).join("") || `<span class="muted">暂无局部标注。负例/排除片段会跳过交叠训练窗口。</span>`;
}

async function openQualityDetail(chartId) {
  const manifest = qualityManifest();
  const request = ++qualityState.detailRequest;
  const data = await api("/api/quality/detail", { manifest, chart_id: chartId });
  if (request !== qualityState.detailRequest || manifest !== qualityManifest()) return;
  qualityState.detail = { ...data, chart_id: chartId };
  const record = data.record || {};
  qualityState.regions = Array.isArray(record.regions) ? JSON.parse(JSON.stringify(record.regions)) : [];
  qualityState.correctedPath = record.corrected_path || "";
  if (element("#qualityCorrectedPath")) element("#qualityCorrectedPath").textContent = qualityState.correctedPath;
  const panel = element("#qualityDetail"); if (panel) panel.hidden = false;
  updateQualityDetail();
  panel?.scrollIntoView({ behavior: "smooth", block: "start" });
}

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
    return { start_sec: Number(value("start_sec")), end_sec: Number(value("end_sec")), weight: Math.max(0, Math.min(4, Number(value("weight")))), reason: value("reason") || "" };
  }).filter((item) => Number.isFinite(item.start_sec) && Number.isFinite(item.end_sec) && item.end_sec > item.start_sec);
}

async function saveQualityRegions() {
  if (!qualityState.detail?.chart_id) throw new Error("请先打开一张谱面。");
  const record = qualityState.detail.record || {};
  const data = await api("/api/quality/save", { manifest: qualityManifest(), chart_ids: [qualityState.detail.chart_id], label: record.label || "normal", weight: Number(record.weight ?? 1), reason: record.reason || "", regions: collectQualityRegions() });
  qualityState.annotationPath = data.annotation_path || qualityState.annotationPath; qualityState.summary = data.summary || qualityState.summary;
  qualityState.annotationManifest = qualityManifest();
  renderQualitySummary(); await loadQualityCatalog(false); await openQualityDetail(qualityState.detail.chart_id);
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
    if (["queued", "running"].includes(task.status)) { intensityCurveState.timer = setTimeout(() => pollIntensityTask(taskId, sourcePath), 800); return; }
    if (task.status !== "completed") throw new Error(task.error || "强度曲线分析失败");
    if (intensityAudioPath() !== sourcePath) { setIntensityStatus("分析已完成，但歌曲已更换；已丢弃旧曲线，请重新生成。", "warn"); return; }
    const result = task.result || {}; const profile = result.intensity_profile || result.profile;
    if (result.intensity_profile_path) profile.analysis_profile_path = result.intensity_profile_path;
    showIntensityProfile(profile, result.audio_preview_url || profile?.audio_preview_url || null);
    setIntensityStatus("自动强度曲线已生成。拖动控制点或按时间段调整后保存。", "good");
  } catch (error) { setIntensityStatus(`分析失败：${error.message || error}`, "warn"); }
}

async function generateIntensityCurve() {
  const audio = intensityAudioPath(); if (!audio) throw new Error("请先在生成页导入音频或填写音频路径");
  setWorkspaceMode("intensity"); setIntensitySong(audio); setIntensityStatus("正在分析节奏、频谱、和声变化与响度…");
  const status = await api("/api/status");
  if (status.intensity_curve_capability !== true) throw new Error("当前后端尚未加载多维强度曲线分析能力，请重启 WebUI 服务后再试");
  const generation = generationPayload();
  const data = await api("/api/intensity/analyze", {
    audio, resolution_sec: 2, local_context_sec: 12,
    curve_source: valueOrDefault("#intensityCurveSource", "auto"),
    checkpoint: generation.checkpoint || null,
    condition: generation.condition, bpm: generation.bpm,
    first: generation.first, tempo_mode: generation.tempo_mode,
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
  if (intensityCurveState.contextStale) throw new Error("节拍设置已变化，请重新生成模型来源曲线后再保存");
  intensityCurveState.profile.edited = true; intensityCurveState.saved = true; saveIntensityCurveState();
  const checkbox = element("#intensityCurveEnabled"); if (checkbox) checkbox.checked = true;
  setWorkspaceMode("generation"); setIntensityStatus("曲线已保存；生成页将使用这份人工曲线。", "good"); updateCurveConditionNotice(); saveSettings();
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
  const profile = intensityCurveState.profile;
  if (!profile || !/^(learned_music_structure_|learned_song_form_)/.test(String(profile.method || ""))) return;
  intensityCurveState.saved = false;
  intensityCurveState.contextStale = true;
  saveIntensityCurveState();
  updateIntensityButtons();
  setIntensityStatus("节拍设置已变化，模型来源曲线需要重新生成。", "warn");
}

function invalidateLearnedCurveForCondition() {
  const profile = intensityCurveState.profile;
  if (!profile || !/^(learned_music_structure_|learned_song_form_)/.test(String(profile.method || ""))) return;
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
  const payload = generationPayload();
  if (!payload.checkpoint) throw new Error("请先上传或填写 checkpoint 路径");
  if (!payload.audio) throw new Error("请先导入音频或填写音频路径");
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
  ["#checkpointPath", "#audioPath", "#featurePath", "#tempoMap", "#formGenerationMeterMap", "#goldStatsPath",
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
loadIntensityCurveState();
generationDefaults = captureGenerationDefaults();
const restoredSettings = restoreSettings();
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
on("#qualityModeTab", "click", openDatasetWorkspace);
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
on("#modelFamily", "change", () => { generationInspection = { path: "", data: null }; checkpointInspection = { key: "", data: null }; applySongFormCheckpointConfig({valid: false}, "fresh"); if (currentModelFamily()==="song_form_v2" && currentTrainingMode()==="fresh") { for (const [id,v] of Object.entries({formDModel:256,formLayers:4,formNhead:8,formDimFeedforward:0,formPlannerWidth:128})) { const node=element("#"+id);if(node)node.value=String(v); } } updateTrainingControls(); updateFamilyFields(); saveSettings(); });
on("#trainingMode", "change", () => { applySongFormCheckpointConfig({valid: false}, "fresh"); setTrainingModeDefaults(); updateTrainingControls(); });
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
on("#qualityAddRegionButton", "click", () => { qualityState.regions.push({ start_sec: 0, end_sec: 1, weight: 0, reason: "" }); updateQualityDetail(); });
on("#qualitySaveRegionsButton", "click", saveQualityRegions);
on("#qualitySaveCorrectionButton", "click", saveQualityCorrection);
on("#qualityBulkLabel", "change", (event) => { const weight = element("#qualityBulkWeight"); if (weight) weight.value = QUALITY_DEFAULT_WEIGHTS[event.target.value] ?? 1; });
document.addEventListener("change", (event) => {
  const row = event.target.closest?.(".quality-row-select");
  if (row) { const id = row.dataset.chartId; if (row.checked) qualityState.selected.add(id); else qualityState.selected.delete(id); updateQualitySelectionControls(); }
  const weight = event.target.closest?.(".quality-weight");
  if (weight) { const item = qualityState.items.find((candidate) => String(candidate.chart_id || candidate.id) === String(weight.dataset.chartId)); if (item) { item.weight = Math.max(0, Math.min(4, Number(weight.value))); item._weightEdited = true; qualityState.editedWeights.set(String(weight.dataset.chartId), item.weight); } }
}, true);
document.addEventListener("click", (event) => {
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
  if (button) openQualityDetail(button.dataset.chartId);
  const remove = event.target.closest?.(".quality-remove-region");
  if (remove) { const row = remove.closest(".quality-region-row"); const index = Number(row?.dataset.regionIndex); if (Number.isInteger(index)) { qualityState.regions.splice(index, 1); updateQualityDetail(); } }
});
document.addEventListener("change", (event) => {
  const checkbox = event.target.closest?.(".dataset-select-checkbox");
  if (!checkbox) return;
  datasetWorkspaceState.selectionTouched = true;
  const key = checkbox.dataset.datasetKey;
  if (checkbox.checked) datasetWorkspaceState.selected.add(key); else datasetWorkspaceState.selected.delete(key);
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
}, true);
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
on("#formQualityProfile", "change", () => { updateSongFormQualityProfileControls(); saveSettings(); });
["#bpm", "#first", "#tempoMode"].forEach((selector) => {
  on(selector, "change", invalidateLearnedCurveForTiming);
});
on("#featureFile", "change", (event) => upload("audio_features", event.target.files[0], "#featurePath", "#featureUploadStatus"));
on("#checkpointFile", "change", async (event) => {
  const data = await upload("checkpoint", event.target.files[0], "#checkpointPath", "#checkpointUploadStatus");
  if (data) {
    generationInspection = { path: "", data: null };
    updateFamilyFields();
    if (element("#generationPlannerStatus")) element("#generationPlannerStatus").textContent = "checkpoint 已更换，请重新检查 planner 类型。";
    setGenerationStatus("已上传 checkpoint；点击『检查生成 checkpoint 类型』确认模型族。", "warn");
  }
});
on("#checkpointPath", "input", () => {
  generationInspection = { path: "", data: null };
  updateFamilyFields();
  if (element("#generationPlannerStatus")) element("#generationPlannerStatus").textContent = "checkpoint 路径已修改，请重新检查 planner 类型。";
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
on("#generationInspectButton", "click", inspectGenerationCheckpoint);
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
  if (!activeTaskId) return;
  await api("/api/cancel", { task_id: activeTaskId });
});

// Initial server probe.  Failure must not disable the rest of the page.
guard(loadStatus)();
guard(loadLevelPresetProfile)();
