"use strict";

(function exposeFormLab(root) {
  const RATIONAL = /^\+?\d+(?:\/\d+)?$/;

  function requiredText(value, name) {
    const result = String(value ?? "").trim();
    if (!result) throw new Error(`${name}不能为空。`);
    return result;
  }

  function checkpointFile(value) {
    const path = requiredText(value, "song_form_v1 checkpoint 文件");
    if (!/\.pt$/i.test(path)) throw new Error("请选择 .pt checkpoint 文件路径。");
    return path;
  }

  function finiteNumber(value, name, {min = -Infinity, max = Infinity, integer = false} = {}) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < min || number > max || (integer && !Number.isInteger(number))) {
      throw new Error(`${name}取值无效。`);
    }
    return number;
  }

  function exactBeat(value, name) {
    const text = requiredText(value, name);
    if (!RATIONAL.test(text)) throw new Error(`${name}必须是非负整数或有理数，如 16/3。`);
    const [numeratorText, denominatorText] = text.split("/");
    const numerator = BigInt(numeratorText);
    const denominator = BigInt(denominatorText || "1");
    if (denominator <= 0n) throw new Error(`${name}分母必须大于 0。`);
    return {text: `${numerator}/${denominator}`, numerator, denominator};
  }

  function compareBeat(left, right) {
    const delta = left.numerator * right.denominator - right.numerator * left.denominator;
    return delta < 0n ? -1 : delta > 0n ? 1 : 0;
  }

  function groupingValues(value, numerator, label) {
    const text = String(value ?? "").trim();
    if (!text) return [];
    const groups = text.split(",").map((part) => finiteNumber(part.trim(), `${label}分组`, {min: 1, integer: true}));
    if (groups.reduce((sum, item) => sum + item, 0) !== numerator) {
      throw new Error(`${label}分组之和必须等于拍号分子。`);
    }
    return groups;
  }

  function meterChange(beat, numeratorValue, denominatorValue, grouping, label) {
    const position = exactBeat(beat, `${label}起始拍`);
    const numerator = finiteNumber(numeratorValue, `${label}分子`, {min: 1, integer: true});
    const denominator = finiteNumber(denominatorValue, `${label}分母`, {min: 1, integer: true});
    return {
      beat: position.text,
      numerator,
      denominator,
      grouping: groupingValues(grouping, numerator, label),
      confidence: 1,
      source: "manual",
      _position: position,
    };
  }

  function buildMeterMap(values, changes = []) {
    const known = values.meter_known === true || values.meter_known === "true" || values.meter_known === "on";
    const rows = (changes || []).filter((row) =>
      [row.beat, row.numerator, row.grouping].some((value) => String(value ?? "").trim())
    );
    if (!known) {
      if (rows.length) throw new Error("填写拍号变更前，请先提供起始拍号。");
      return null;
    }
    const result = [meterChange("0", values.meter_numerator, values.meter_denominator, values.meter_grouping, "起始拍号")];
    let previous = result[0]._position;
    for (const [index, row] of rows.entries()) {
      const change = meterChange(row.beat, row.numerator, row.denominator, row.grouping, `拍号变更 ${index + 1}`);
      if (compareBeat(change._position, previous) <= 0) throw new Error("拍号变更必须按起始拍严格递增，且晚于第 0 拍。");
      previous = change._position;
      result.push(change);
    }
    return result.map(({_position, ...item}) => item);
  }

  function buildTempoMap(value) {
    const text = String(value ?? "").trim();
    if (!text) return null;
    let entries;
    try { entries = JSON.parse(text); } catch (_) { throw new Error("tempo_map 必须是有效 JSON。"); }
    if (!Array.isArray(entries) || entries.length === 0) throw new Error("tempo_map 必须是非空数组。");
    const result = entries.map((entry, index) => {
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error("tempo_map 每项都需要 beat 和 bpm。");
      const position = exactBeat(entry.beat, `tempo_map 第 ${index + 1} 项拍位`);
      return {
        beat: position.text,
        bpm: finiteNumber(entry.bpm, `tempo_map 第 ${index + 1} 项 BPM`, {min: Number.MIN_VALUE}),
        _position: position,
      };
    });
    if (result[0]._position.numerator !== 0n) throw new Error("tempo_map 必须从第 0 拍开始。");
    for (let index = 1; index < result.length; index += 1) {
      if (compareBeat(result[index]._position, result[index - 1]._position) <= 0) {
        throw new Error("tempo_map 拍位必须严格递增。");
      }
    }
    return result.map(({_position, ...entry}) => entry);
  }

  function buildClock(values, changes = []) {
    return {
      bpm: finiteNumber(values.bpm ?? 120, "BPM", {min: Number.MIN_VALUE}),
      first: finiteNumber(values.first ?? 0, "first"),
      meter_map: buildMeterMap(values, changes),
      tempo_map: buildTempoMap(values.tempo_map_json),
    };
  }

  function buildAnalysisPayload(values, changes = []) {
    return {
      audio: requiredText(values.audio, "音频路径"),
      ...buildClock(values, changes),
      max_region_seconds: finiteNumber(values.max_region_seconds ?? 14, "单次音频计算上限", {min: 2, max: 14}),
    };
  }

  function buildTrainingPayload(values, sharedMeterMap = null) {
    const mode = values.mode || "fresh";
    if (!["fresh", "resume", "finetune"].includes(mode)) throw new Error("训练模式无效。");
    const checkpoint = String(values.init_checkpoint ?? "").trim();
    if (mode !== "fresh" && !checkpoint) throw new Error("继续训练或加强训练时需要填写初始化 checkpoint。");
    const dModel = finiteNumber(values.d_model ?? 256, "d_model", {min: 8, integer: true});
    const nhead = finiteNumber(values.nhead ?? 8, "Attention heads", {min: 1, integer: true});
    if (dModel % nhead !== 0) throw new Error("d_model 必须能被 Attention heads 整除。");
    const payload = {
      manifest: requiredText(values.manifest, "训练素材合并清单"),
      checkpoint_dir: requiredText(values.checkpoint_dir, "Checkpoint 输出目录"),
      mode,
      init_checkpoint: mode === "fresh" ? null : checkpoint,
      epochs: finiteNumber(values.epochs ?? 20, "Epochs", {min: 1, integer: true}),
      batch_size: finiteNumber(values.batch_size ?? 4, "Batch size", {min: 1, integer: true}),
      d_model: dModel,
      layers: finiteNumber(values.layers ?? 4, "Layers", {min: 1, integer: true}),
      nhead,
      max_tokens: finiteNumber(values.max_tokens ?? 2048, "Max chart tokens", {min: 1, integer: true}),
      max_memory_tokens: finiteNumber(values.memory_tokens ?? values.max_memory_tokens ?? 512, "Max memory tokens", {min: 1, integer: true}),
      lr: finiteNumber(values.lr ?? 0.0002, "学习率", {min: Number.MIN_VALUE}),
      device: values.device || "cpu",
    };
    if (!["cpu", "cuda"].includes(payload.device)) throw new Error("设备只能选择 CPU 或 CUDA。");
    for (const [key, label] of [["max_steps", "最大训练步数"], ["limit_charts", "谱面数量上限"]]) {
      if (String(values[key] ?? "").trim()) payload[key] = finiteNumber(values[key], label, {min: 1, integer: true});
    }
    if (values.use_analysis_meter === true || values.use_analysis_meter === "true" || values.use_analysis_meter === "on") {
      if (!sharedMeterMap) throw new Error("启用统一拍号前，请在上方提供明确拍号。");
      payload.meter_map = sharedMeterMap;
    }
    return payload;
  }

  function buildGenerationPayload(values, sharedClock = {}, changes = []) {
    const clock = buildClock(sharedClock, changes);
    const threshold = finiteNumber(values.onset_threshold ?? 0.5, "谱面起音贴合阈值", {min: 0.01, max: 0.99});
    return {
      checkpoint: checkpointFile(values.checkpoint),
      audio: requiredText(values.audio, "音频路径"),
      output: requiredText(values.output, "输出路径"),
      level: finiteNumber(values.level ?? 13.3, "难度等级", {min: 1, max: 15.9}),
      ...clock,
      memory_strength: finiteNumber(values.memory_strength ?? 0.6, "Motif 记忆强度", {min: 0, max: 1}),
      onset_threshold: threshold,
      repair: values.repair === true || values.repair === "true" || values.repair === "on",
      max_checks: finiteNumber(values.max_checks ?? 24, "最多检查次数", {min: 1, integer: true}),
      max_edits: finiteNumber(values.max_edits ?? 4, "最多编辑数", {min: 0, integer: true}),
      deadline_seconds: finiteNumber(values.deadline_seconds ?? 10, "修订时限", {min: 1}),
    };
  }

  function buildRevisionPayload(values) {
    const start = exactBeat(values.editable_start, "可编辑区开始");
    const end = exactBeat(values.editable_end, "可编辑区结束");
    if (compareBeat(end, start) <= 0) throw new Error("可编辑区结束拍必须晚于开始拍。");
    return {
      checkpoint: checkpointFile(values.checkpoint),
      audio: requiredText(values.audio, "音频路径"),
      maidata_path: requiredText(values.maidata_path, "来源 maidata 路径"),
      output: requiredText(values.output, "修订输出路径"),
      difficulty_index: finiteNumber(values.difficulty_index, "Difficulty index", {min: 0, integer: true}),
      editable_start: start.text,
      editable_end: end.text,
      memory_strength: finiteNumber(values.memory_strength ?? 0.6, "Motif 记忆强度", {min: 0, max: 1}),
    };
  }

  function musicalRegions(report) {
    const spans = Array.isArray(report?.spans) ? report.spans : [];
    const segmentation = report?.segmentation || {};
    const spanRegions = segmentation.span_region_indices || [];
    const regions = new Map((segmentation.region_spans || []).map((region, index) => [
      Number(region.index ?? index), region,
    ]));
    if (!regions.size) {
      spans.forEach((span, index) => {
        const regionId = Number(spanRegions[index] ?? index);
        const existing = regions.get(regionId);
        if (existing) existing.end = span.end;
        else regions.set(regionId, {...span, index: regionId});
      });
    }
    return regions;
  }

  function mapRelationsToRegions(report) {
    const regions = musicalRegions(report);
    return (report?.recurrence?.links || []).flatMap((relation) => {
      const sourceRegion = regions.get(Number(relation.source));
      const targetRegion = regions.get(Number(relation.target));
      return sourceRegion && targetRegion ? [{...relation, sourceRegion, targetRegion}] : [];
    });
  }

  function resolvePreparedDataset(items, saved) {
    const datasets = Array.isArray(saved?.datasets) ? saved.datasets : [];
    if (datasets.length > 1 || !saved?.manifest) return null;
    const item = (items || []).find((candidate) =>
      candidate.prepared === true && candidate.manifest === saved.manifest
    );
    if (!item) return null;
    if (datasets.length === 1) {
      const selected = datasets[0];
      const key = typeof selected === "string" ? selected : String(selected?.manifest || selected?.path || "");
      if (key !== item.manifest && key !== item.path) return null;
    }
    return item;
  }

  const payloadBuilders = {
    buildAnalysisPayload,
    buildMeterMap,
    buildTrainingPayload,
    buildGenerationPayload,
    buildRevisionPayload,
    mapRelationsToRegions,
    resolvePreparedDataset,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = payloadBuilders;
  if (!root || !root.document) return;
  root.FormLabPayload = payloadBuilders;

  const $ = (selector, scope = document) => scope.querySelector(selector);
  const meterRows = $("#meterChanges");
  const taskLog = $("#taskLogs");
  let activeTimer = null;
  let activeTaskId = "";
  let activeTaskAction = "";
  let isBusy = false;
  let loadedReportTask = "";

  function formValues(form) {
    const values = {};
    for (const field of form.elements) {
      if (!field.name) continue;
      values[field.name] = field.type === "checkbox" ? field.checked : field.value;
    }
    return values;
  }

  function savedDatasetSettings() {
    try {
      return JSON.parse(localStorage.getItem("maichart.training.dataset.v1") || "null");
    } catch (_) {
      return null;
    }
  }

  async function loadTrainingDatasets() {
    const picker = $("#trainingDatasetPicker");
    const manifestInput = $("#trainingForm [name='manifest']");
    const status = $("#trainingDatasetStatus");
    const previousChoice = picker.value;
    const previousManualPath = !previousChoice ? manifestInput.value.trim() : "";
    picker.disabled = true;
    picker.replaceChildren(new Option("正在读取数据集目录…", ""));
    try {
      const data = await request("/api/datasets/catalog", {});
      const catalog = Array.isArray(data.datasets) ? data.datasets : Array.isArray(data.items) ? data.items : [];
      const prepared = catalog.filter((item) => item?.prepared === true && item.manifest);
      const placeholder = new Option(prepared.length ? "选择一份已准备素材…" : "没有找到已准备素材", "");
      picker.replaceChildren(placeholder);
      for (const item of prepared) {
        const option = new Option(`${item.name || "数据集"} · ${item.charts ?? "?"} 张谱面`, item.manifest);
        picker.append(option);
      }
      picker.disabled = prepared.length === 0;

      const currentChoice = prepared.find((item) => item.manifest === previousChoice);
      const saved = savedDatasetSettings();
      const restored = resolvePreparedDataset(prepared, saved);
      if (currentChoice) {
        picker.value = currentChoice.manifest;
        manifestInput.value = currentChoice.manifest;
        status.textContent = `训练清单已设置：${currentChoice.manifest}`;
      } else if (previousChoice) {
        manifestInput.value = "";
        status.textContent = "上次选择的清单已不在目录中，请重新选择。";
      } else if (previousManualPath) {
        manifestInput.value = previousManualPath;
        status.textContent = "保留了高级路径输入。";
      } else if (restored) {
        picker.value = restored.manifest;
        manifestInput.value = restored.manifest;
        status.textContent = `已恢复唯一匹配的已准备清单：${restored.manifest}`;
      } else if (Array.isArray(saved?.datasets) && saved.datasets.length > 1) {
        status.textContent = "检测到上次选择了多个来源。请返回主工作台的“训练素材”页完成合并，再刷新并选择合并后的清单。";
      } else if (prepared.length) {
        status.textContent = `找到 ${prepared.length} 份已准备素材；选择后会自动填入训练清单。`;
      } else {
        status.textContent = "没有找到已准备素材。请返回主工作台的“训练素材”页导入并预处理。";
      }
    } catch (error) {
      picker.replaceChildren(new Option("数据集目录读取失败", ""));
      status.textContent = `读取数据集目录失败：${error.message}。可在高级选项中粘贴清单路径。`;
    }
  }

  function meterChanges() {
    return Array.from(meterRows.querySelectorAll(".form-lab-meter-row"), (row) => ({
      beat: $("[name='change_beat']", row).value,
      numerator: $("[name='change_numerator']", row).value,
      denominator: $("[name='change_denominator']", row).value,
      grouping: $("[name='change_grouping']", row).value,
    }));
  }

  function addMeterRow() {
    const row = document.createElement("div");
    row.className = "form-lab-meter-row";
    const field = (label, name, type, options = {}) => {
      const wrapper = document.createElement("label");
      wrapper.className = "field";
      wrapper.append(document.createTextNode(label));
      let input;
      if (type === "select") {
        input = document.createElement("select");
        for (const value of [2, 4, 8, 16]) {
          const option = document.createElement("option");
          option.value = String(value);
          option.textContent = String(value);
          if (value === 4) option.selected = true;
          input.append(option);
        }
      } else {
        input = document.createElement("input");
        input.type = type;
        Object.assign(input, options);
      }
      input.name = name;
      wrapper.append(input);
      row.append(wrapper);
    };
    field("起始拍", "change_beat", "text", {placeholder: "如 32 或 32/3"});
    field("分子", "change_numerator", "number", {min: "1", step: "1"});
    field("分母", "change_denominator", "select");
    field("拍组", "change_grouping", "text", {placeholder: "3,3"});
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary-button meter-remove";
    remove.textContent = "移除";
    remove.setAttribute("aria-label", "移除拍号变更");
    remove.addEventListener("click", () => row.remove());
    row.append(remove);
    meterRows.append(row);
  }

  function setTaskMessage(message, error = false) {
    const target = $("#taskMessage");
    target.textContent = message;
    target.classList.toggle("workflow-error", error);
  }

  async function request(path, body) {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      cache: "no-store",
      ...(body === undefined ? {} : {headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}),
    });
    const text = await response.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch (_) { data = {error: text || `HTTP ${response.status}`}; }
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function artifactEntries(task) {
    const result = task.result || {};
    const entries = [];
    const reportPath = result.report_path || result.report_file || (typeof result.report === "string" ? result.report : "");
    if (reportPath) entries.push({label: "下载分析报告", path: reportPath});
    const artifacts = result.artifacts || result.artifact_paths || [];
    const list = Array.isArray(artifacts) ? artifacts : Object.entries(artifacts).map(([name, path]) => ({name, path}));
    for (const item of list) {
      const path = typeof item === "string" ? item : item?.path;
      if (!path) continue;
      if (!entries.some((entry) => entry.path === path)) entries.push({label: item?.name || item?.label || "下载产物", path});
    }
    return entries;
  }

  function renderTask(task) {
    $("#taskKind").textContent = `${task.kind || "form"} · ${task.task_id || ""}`;
    $("#taskStatus").textContent = ({queued: "排隊中", running: "運行中", completed: "已完成", failed: "失敗", cancelled: "已中斷"})[task.status] || task.status || "未知";
    const progress = Math.max(0, Math.min(1, Number(task.progress || 0)));
    $("#taskProgress").style.width = `${Math.round(progress * 100)}%`;
    if (task.error || task.message) setTaskMessage(task.error || task.message, Boolean(task.error));
    if (root.MaichartLogs) root.MaichartLogs.renderTask(task);
    else {
      const logValue = Array.isArray(task.logs) ? task.logs.join("\n") : typeof task.logs === "string" ? task.logs : "";
      if (logValue) taskLog.textContent = logValue;
    }
    const artifacts = $("#taskArtifacts");
    const entries = artifactEntries(task);
    if (entries.length) {
      artifacts.replaceChildren();
      for (const entry of entries) {
        const link = document.createElement("a");
        link.href = `/api/download?path=${encodeURIComponent(entry.path)}`;
        link.textContent = `${entry.label}: ${entry.path}`;
        link.download = "";
        artifacts.append(link);
      }
    }
    const running = ["queued", "running"].includes(task.status);
    if (!running) setBusy(false);
    if (!running && task.status === "completed" && activeTaskAction === "analysis" && loadedReportTask !== task.task_id) {
      loadedReportTask = task.task_id;
      loadAnalysisReport(task).catch((error) => {
        $("#analysisResult").hidden = false;
        $("#analysisSummary").textContent = `分析报告无法读取：${error.message}`;
      });
    }
  }

  function beatValue(value) {
    const [numerator, denominator = "1"] = String(value ?? "0").split("/");
    const top = Number(numerator);
    const bottom = Number(denominator);
    return Number.isFinite(top) && Number.isFinite(bottom) && bottom !== 0 ? top / bottom : NaN;
  }

  function boundaryLabel(span) {
    const source = String(span.source || "");
    if (source === "resource_cap") return "计算上限切分（非乐句）";
    if (source.startsWith("audio_novelty")) return "音频变化候选（待验证）";
    if (source === "meter_change") return "拍号变更边界";
    if (source === "song_start" || source === "song_end") return "歌曲边界";
    return source || "未标注边界";
  }

  function svgNode(name, attributes = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    return node;
  }

  function renderAnalysisReport(report) {
    if (!report || !Array.isArray(report.spans)) throw new Error("报告中没有 spans 数组。");
    const panel = $("#analysisResult");
    const timeline = $("#analysisTimeline");
    const linksBox = $("#analysisLinks");
    panel.hidden = false;
    timeline.replaceChildren();
    linksBox.replaceChildren();
    const spans = report.spans;
    const regionIndices = report.segmentation?.span_region_indices || [];
    const regions = musicalRegions(report);
    const links = mapRelationsToRegions(report);
    const total = Math.max(...spans.map((span) => beatValue(span.end)), 1);
    const width = 860;
    const plotLeft = 44;
    const plotWidth = width - plotLeft - 24;
    const rowHeight = 28;
    const top = 20;
    const height = top + Math.max(1, spans.length) * rowHeight + 28;
    const svg = svgNode("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "按四分音符拍排列的计算段与相似关系候选"});
    const axis = svgNode("line", {x1: plotLeft, y1: 10, x2: width - 20, y2: 10, stroke: "#aaa399", "stroke-width": 1});
    svg.append(axis);
    const zero = svgNode("text", {x: plotLeft, y: 9, fill: "#77736b", "font-size": 9, "text-anchor": "start"});
    zero.textContent = "0 拍";
    svg.append(zero);
    const endLabel = svgNode("text", {x: width - 20, y: 9, fill: "#77736b", "font-size": 9, "text-anchor": "end"});
    endLabel.textContent = `${report.spans.at(-1)?.end || "0"} 拍`;
    svg.append(endLabel);

    const rowGroups = new Map();
    spans.forEach((span, index) => {
      const regionId = Number(regionIndices[index] ?? span.index ?? index);
      const group = rowGroups.get(regionId) || {first: index, last: index};
      group.last = index;
      rowGroups.set(regionId, group);
    });
    for (const [regionId, group] of rowGroups) {
      const region = regions.get(regionId);
      if (!region) continue;
      const x = plotLeft + (beatValue(region.start) / total) * plotWidth;
      const regionWidth = Math.max(2, ((beatValue(region.end) - beatValue(region.start)) / total) * plotWidth);
      const y = top + group.first * rowHeight - 3;
      const regionRect = svgNode("rect", {
        x, y, width: regionWidth, height: (group.last - group.first + 1) * rowHeight - 4,
        fill: "#e5bd45", "fill-opacity": 0.12, stroke: "#9a8960", "stroke-opacity": 0.35, "stroke-width": 0.7,
      });
      const title = svgNode("title");
      title.textContent = `音乐区域 ${regionId + 1}：${region.start}–${region.end} 拍`;
      regionRect.append(title);
      svg.append(regionRect);
    }
    for (const relation of links) {
      const source = rowGroups.get(Number(relation.source));
      const target = rowGroups.get(Number(relation.target));
      if (!source || !target) continue;
      const sourceRegion = relation.sourceRegion;
      const targetRegion = relation.targetRegion;
      const x1 = plotLeft + ((beatValue(sourceRegion.start) + beatValue(sourceRegion.end)) / 2 / total) * plotWidth;
      const x2 = plotLeft + ((beatValue(targetRegion.start) + beatValue(targetRegion.end)) / 2 / total) * plotWidth;
      const y1 = top + ((source.first + source.last) / 2) * rowHeight + 8;
      const y2 = top + ((target.first + target.last) / 2) * rowHeight + 8;
      const curve = svgNode("path", {
        d: `M ${x1} ${y1} Q ${(x1 + x2) / 2} ${Math.min(y1, y2) - 8} ${x2} ${y2}`,
        fill: "none", stroke: "#167d7c", "stroke-width": 1.5, "stroke-opacity": 0.55,
      });
      const title = svgNode("title");
      title.textContent = `相似候选区域 ${Number(relation.source) + 1} → ${Number(relation.target) + 1}，相似度 ${Number(relation.similarity).toFixed(2)}；不是人工段落标签`;
      curve.append(title);
      svg.append(curve);
    }

    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const title of ["段", "起止拍", "拍号 / 分组", "边界与区域", "关系"]) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = title;
      headerRow.append(cell);
    }
    thead.append(headerRow);
    table.append(thead);
    const tbody = document.createElement("tbody");
    spans.forEach((span, index) => {
      const y = top + index * rowHeight;
      const x = plotLeft + (beatValue(span.start) / total) * plotWidth;
      const rectWidth = Math.max(3, ((beatValue(span.end) - beatValue(span.start)) / total) * plotWidth);
      const known = span.numerator != null && span.denominator != null;
      const rect = svgNode("rect", {
        x, y, width: rectWidth, height: 16, rx: 2,
        fill: known ? ["#167d7c", "#d95f49", "#638c67", "#d19b31"][Number(regionIndices[index] || 0) % 4] : "#8a8883",
      });
      const rectTitle = svgNode("title");
      rectTitle.textContent = `段 ${index + 1}：${known ? `${span.numerator}/${span.denominator}` : "拍号未知"}`;
      rect.append(rectTitle);
      svg.append(rect);

      const regionId = Number(regionIndices[index] ?? span.index ?? index);
      const related = links.filter((item) => Number(item.source) === regionId || Number(item.target) === regionId);
      const relationText = related.map((item) => `${Number(item.source) + 1}→${Number(item.target) + 1} (${Number(item.similarity).toFixed(2)})`).join("；") || "—";
      const grouping = Array.isArray(span.grouping) && span.grouping.length ? span.grouping.join("+") : "分组未知";
      const cells = [
        `段 ${index + 1} · 区域 ${regionId + 1}`,
        `${span.start}–${span.end}`,
        known ? `${span.numerator}/${span.denominator} · ${grouping}` : "拍号未知 · 不推断小节",
        boundaryLabel(span),
        relationText,
      ];
      const row = document.createElement("tr");
      for (const value of cells) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      tbody.append(row);
    });
    table.append(tbody);
    timeline.append(svg, table);

    const knownCount = spans.filter((span) => span.numerator != null).length;
    $("#analysisSummary").textContent = `${spans.length} 个计算段，${links.length} 条相似关系候选；${knownCount}/${spans.length} 段有明确拍号。资源上限切分不是乐句标签。`;
    if (links.length) {
      const summary = document.createElement("p");
      summary.className = "inline-status";
      summary.textContent = "绿色连线仅表示音频相似度候选，不代表已确认的副歌、重复段或编曲意图。";
      linksBox.append(summary);
    }
  }

  async function loadAnalysisReport(task) {
    const result = task.result || {};
    let report = result.analysis_report || result.report_data;
    if (!report && result.report && typeof result.report === "object") report = result.report;
    if (!report) {
      const path = result.report_path || result.report_file || (typeof result.report === "string" ? result.report : "");
      if (!path) throw new Error("任务结果没有报告路径。");
      report = await request(`/api/download?path=${encodeURIComponent(path)}`);
    }
    if (activeTaskId !== task.task_id) return;
    renderAnalysisReport(report);
  }

  function setBusy(busy) {
    isBusy = Boolean(busy);
    document.querySelectorAll(".form-lab-main form button[type='submit']").forEach((button) => {
      button.disabled = isBusy;
    });
    $("#cancelTask").disabled = !isBusy || !activeTaskId;
  }

  async function pollTask(taskId) {
    try {
      const task = await request(`/api/task?id=${encodeURIComponent(taskId)}`);
      renderTask(task);
      if (["queued", "running"].includes(task.status)) {
        activeTimer = setTimeout(() => pollTask(taskId), 1000);
      } else {
        activeTimer = null;
      }
    } catch (error) {
      setTaskMessage(`状态读取失败，将重试：${error.message}`, true);
      activeTimer = setTimeout(() => pollTask(taskId), 3000);
    }
  }

  async function launch(endpoint, payload, label) {
    if (activeTimer) clearTimeout(activeTimer);
    $("#taskKind").textContent = label;
    $("#taskStatus").textContent = "提交中";
    $("#taskProgress").style.width = "0%";
    $("#taskArtifacts").replaceChildren();
    $("#analysisResult").hidden = true;
    $("#analysisTimeline").replaceChildren();
    $("#analysisLinks").replaceChildren();
    taskLog.textContent = "";
    activeTaskId = "";
    activeTaskAction = endpoint.endsWith("/analyze") ? "analysis" : "other";
    setTaskMessage("正在提交任务…");
    const data = await request(endpoint, payload);
    if (!data.task_id) throw new Error("服务端未返回 task_id。");
    activeTaskId = data.task_id;
    $("#taskKind").textContent = `${label} · ${data.task_id}`;
    $("#cancelTask").disabled = false;
    pollTask(data.task_id);
  }

  function bindForm(id, endpoint, builder, label, extra = null) {
    const form = $(`#${id}`);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (isBusy) return;
      setBusy(true);
      try {
        const values = formValues(form);
        const payload = extra ? builder(values, ...extra()) : builder(values);
        await launch(endpoint, payload, label);
      } catch (error) {
        setTaskMessage(error.message || String(error), true);
        $("#taskStatus").textContent = "未提交";
        activeTaskId = "";
        setBusy(false);
      }
    });
  }

  const meterToggle = $("[name='meter_known']");
  meterToggle.addEventListener("change", () => {
    $("#initialMeterFields").hidden = !meterToggle.checked;
  });
  $("#trainingDatasetPicker").addEventListener("change", (event) => {
    const manifest = event.target.value;
    $("#trainingForm [name='manifest']").value = manifest;
    if (manifest) $("#trainingDatasetStatus").textContent = `训练清单已设置：${manifest}`;
  });
  $("#trainingForm [name='manifest']").addEventListener("input", () => {
    if (!$("#trainingForm [name='manifest']").value.trim()) return;
    $("#trainingDatasetPicker").value = "";
    $("#trainingDatasetStatus").textContent = "使用了高级路径输入。";
  });
  $("#refreshTrainingDatasets").addEventListener("click", loadTrainingDatasets);
  loadTrainingDatasets();
  $("#addMeterChange").addEventListener("click", addMeterRow);
  bindForm("analysisForm", "/api/form/analyze", buildAnalysisPayload, "音频分析", meterChanges);
  bindForm("trainingForm", "/api/form/train", buildTrainingPayload, "Song Form 训练", () => {
    const training = formValues($("#trainingForm"));
    return [training.use_analysis_meter ? buildMeterMap(formValues($("#analysisForm")), meterChanges()) : null];
  });
  bindForm("generationForm", "/api/form/generate", buildGenerationPayload, "Song Form 生成", () => [
    formValues($("#analysisForm")), meterChanges(),
  ]);
  bindForm("revisionForm", "/api/form/revise", buildRevisionPayload, "Song Form 修订");
  $("#cancelTask").addEventListener("click", async () => {
    if (!activeTaskId || !isBusy) return;
    $("#cancelTask").disabled = true;
    try {
      await request("/api/cancel", {task_id: activeTaskId});
      setTaskMessage("已发送中断请求，等待任务状态更新。");
    } catch (error) {
      $("#cancelTask").disabled = false;
      setTaskMessage(`中断失败：${error.message}`, true);
    }
  });
})(typeof globalThis !== "undefined" ? globalThis : this);
