"use strict";

// ---------------------------------------------------------------------------
// Condition sliders.  Each entry is [key, label, help, default]; the help text
// cites where the default came from, and the defaults are the corpus medians
// (maichart/generation/conditions.py GENERATION_DEFAULTS) so a fresh session
// starts where real charts are instead of at the neutral 0.5 mask value.
// ---------------------------------------------------------------------------
const radarKeys = [
  ["note_density", "整体密度", "notes/sec 的整体目标：一段时间内平均写多少可玩音符。不是响度，也不是峰值。语料中位等级带 0.24–0.37，中位 0.305。", "0.305"],
  ["peak", "峰值密度", "高潮或爆发段的局部密度目标；与整体密度一起决定普通段和密集段的落差。语料中位等级带 0.25–0.56，中位 0.375。", "0.375"],
  ["sweep", "滑条量（sweep）", "Slide 的总体使用量/运动量倾向；只提高它不等于每段都会出现 Slide。语料中位等级带 0.06–0.18，中位 0.109。", "0.109"],
  ["slide_tricky", "滑条复杂度", "Slide 路径和转折的复杂程度倾向；数值越高越偏向更复杂的 Slide 写法。语料中位等级带 0.05–0.19，中位 0.114。", "0.114"],
  ["slide_sequence", "连续滑条", "连续安排多个 Slide 的倾向；过高可能造成长段 Slide 堆叠，应配合 sweep 和 slide_bias 调整。语料中位等级带 0.02–0.13，中位 0.066。", "0.066"],
  ["jack", "同键连打（jack）", "相邻时刻重复同一个按键的倾向；高值会增加 Jack 感，但不代表整体难度自动提高。语料中位等级带 0.11–0.28，中位 0.188。", "0.188"],
  ["chord", "和弦比例", "同一时刻同时按多个键的目标比例；应与 Style 中的 chord_bias 同方向设置。语料中位等级带 0.16–0.41，中位 0.280。", "0.280"],
];
const styleKeys = [
  ["tap_bias", "Tap 倾向", "Tap 在整体谱面中的使用倾向；数值越高越偏向普通按键和较少特殊 note。语料中位等级带 0.59–0.83，中位 0.720。", "0.720"],
  ["slide_bias", "Slide 倾向", "Slide 类型在整体谱面中的使用倾向；建议与 Radar 的 sweep、slide_tricky、slide_sequence 同方向设置。语料中位等级带 0.07–0.23，中位 0.140。", "0.140"],
  ["hold_bias", "Hold 倾向", "Hold 长条在整体谱面中的使用倾向；新 music_structure_v2 planner 有 Hold 比例预测，旧 planner 没有独立 Hold 控制。此滑条始终是软条件，不是强制配额。语料中位等级带 0.03–0.14，中位 0.077。", "0.077"],
  ["touch_bias", "Touch 倾向", "Touch / Touch Hold 在整体谱面中的使用倾向；高值不保证每个窗口都出现 Touch。语料中位等级带 0.00–0.14，中位 0.039。", "0.039"],
  ["chord_bias", "和弦倾向", "和弦在整体谱面中的表现倾向；它和 Radar 的 chord 是一对，建议同时提高或降低。语料中位等级带 0.16–0.41，中位 0.280。", "0.280"],
  ["melody_following", "节奏贴合（非旋律，melody_following）", "后端条件名为 melody_following。当前项目主要把它解释为跟随节奏/起音结构，不等于直接识别人声旋律，也不等于响度。训练侧实测中位约 0.895。", "0.900"],
  ["pattern_reuse", "模式复用", "重复使用已经出现过的节奏/配置模式的倾向；低值鼓励更多变化，高值允许更多动机再现。当前模型已有长段重复风险，建议保持较低。语料中位数 0.028。", "0.028"],
  ["directionality", "方向性", "后端统计相邻单键位置变化在顺时针/逆时针方向上的一致程度；高值偏向连续同向移动，不是 Slide 路径长度。语料中位等级带 0.01–0.14，中位 0.043。", "0.043"],
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

// Generation knobs only.  Song data (bpm/first/tempoMap), file paths, metadata
// and every training control are deliberately absent: resetting the generation
// panel must never endanger a training configuration the user settled on.
const GENERATION_PARAM_FIELDS = [
  "tempoMode", "division",                                  // 节拍与时间轴
  "level", "conditionPreferenceStrength",                   // 等级基准与叠加偏好
  "preferKeyboard", "preferStars", "preferTechnical", "preferChord", "preferHold", "preferTouch", "preferMovement",
  "generationDevice",                                       // 生成运行设备
  "seed", "candidates", "temperature", "sampleEvents",       // 采样与随机性
  "topP", "repetitionPenalty", "repetitionWindow", "maxTokens",
  "intensityFollow", "intensityFollowSpan",                  // 逐窗口强度跟随
  "intensityMode", "intensityCurveEnabled",
  "intensityCurveStrength", "intensityCurveMode", "intensityCurveSource",
  "tailDamping", "tailDampingWindows", "tailDampingStrength", // 收尾衰减
  "audioBoundsTrim", "audioBoundsLead", "silenceThreshold", "silenceWindowShare", // 音频边界
  "guidanceScale", "presenceThreshold", "buttonThreshold",
];

let activeTaskId = null;
let datasetUploading = false;
let preprocessStarting = false;
let trainingStarting = false;
let datasetVersion = 0;
let manifestRequest = 0;
let checkpointInspectRequest = 0;
let generationInspectRequest = 0;
let uploadedDatasets = [];
let checkpointInspection = { key: "", data: null };
let generationInspection = { path: "", data: null };
let generationDefaults = {};
let intensityCurveState = { path: "", hash: "", profile: null, automatic: null, history: [], taskId: null, timer: null, dragging: -1, saved: false };
const qualityState = { items: [], page: 1, pageSize: 25, pages: 1, total: 0, folderTotal: 0, query: "", label: "", annotationPath: "", annotationManifest: "", summary: null, selected: new Set(), editedWeights: new Map(), selectionRequest: 0, selectionLoading: false, saving: false, detail: null, regions: [], correctedPath: "", catalogRequest: 0, detailRequest: 0 };
// Dataset selection is intentionally separate from qualityState: selected
// sources feed training, while activeDataset is the source currently being
// browsed/annotated. This prevents a merged training manifest from hiding the
// origin of a quality label.
const datasetWorkspaceState = { items: [], selected: new Set(), activePath: "", catalog: null, catalogRequest: 0, selectionTouched: false, view: "cards", prepared: null };
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
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
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
      history: [], saved: saved.saved === true };
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
    }));
  } catch (_) { /* curve is still usable for this page */ }
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------
function settingsFields() {
  return Array.from(document.querySelectorAll("input[id], select[id], textarea[id]"))
    .filter((element) => element.type !== "file" && element.id !== "datasetSelect"
      && !element.id.startsWith("intensitySection"));
}

function collectSettings() {
  const values = {};
  settingsFields().forEach((target) => {
    values[target.id] = target.type === "checkbox" ? target.checked : target.value;
  });
  return values;
}

function saveSettings() {
  if (settingsSaveTimer) clearTimeout(settingsSaveTimer);
  settingsSaveTimer = setTimeout(flushSettings, 200);
}

function flushSettings() {
  if (settingsSaveTimer) { clearTimeout(settingsSaveTimer); settingsSaveTimer = null; }
  try {
    localStorage.setItem(settingsStorageKey, JSON.stringify({ version: 1, values: collectSettings() }));
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
  let restored = 0;
  settingsFields().forEach((target) => {
    if (!Object.prototype.hasOwnProperty.call(values, target.id)) return;
    const value = values[target.id];
    if (target.type === "checkbox") target.checked = value === true;
    else if (value !== undefined && value !== null) target.value = String(value);
    restored += 1;
  });
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
  Object.entries(generationDefaults).forEach(([id, value]) => {
    const target = document.getElementById(id);
    if (!target) return;
    if (target.type === "checkbox") target.checked = value === true;
    else target.value = String(value);
    const mirror = document.querySelector(`[data-number-for="${id}"]`);
    if (mirror) mirror.value = target.value;
  });
  if (levelPresetProfile) applyConditionPreset();
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
  return valueOrDefault("#modelFamily", "event_ctx_v1");
}

function currentGenerationFamily() {
  return generationInspection.data?.model_family || "event_v1";
}

// A field marked for one family applies to its descendants too: the
// cross-window model is the event pipeline plus context, so every event_v1
// setting still applies to it.  A field may declare several families at once.
function familyFieldVisible(declared, current) {
  return String(declared).split(/\s+/).filter(Boolean).some((name) => (
    name === current || (name === "event_v1" && current === "event_ctx_v1")
  ));
}

function updateFamilyFields() {
  document.querySelectorAll("[data-training-family]").forEach((target) => {
    target.hidden = !familyFieldVisible(target.dataset.trainingFamily, currentModelFamily());
  });
  document.querySelectorAll("[data-generation-family]").forEach((target) => {
    target.hidden = !familyFieldVisible(target.dataset.generationFamily, currentGenerationFamily());
  });
  const family = currentModelFamily();
  const familyHelp = element("#modelFamilyHelp");
  if (familyHelp) {
    familyHelp.textContent = family === "event_v1"
      ? "事件模型使用 112 维 event_audio_v2_rhythm 和显式 null embedding；Planner profile 选择 music_structure_v2 时使用 12D 音乐结构规划器，选择 legacy_v1 才是旧 planner。"
      : family === "event_ctx_v1"
        ? "跨窗口事件模型：使用 112 维 event_audio_v2_rhythm 和显式 null embedding；每个窗口额外看到「前面已写的音符」和「前后的音乐」。Planner profile 选择 music_structure_v2 时学习局部音乐结构；checkpoint 与 event_v1 互不通用，必须单独训练。"
        : "旧固定帧基线，仅用于对照；不具备 event 模型的表示能力。";
  }
  const dirHelp = element("#checkpointDirHelp");
  if (dirHelp) {
    dirHelp.textContent = family === "legacy_frame_v2"
      ? "旧基线输出 generator_last.pt / generator_epoch_NNN.pt，不能作为 event checkpoint 使用。"
      : "event_last.pt / event_best.pt / event_epoch_NNN.pt。建议使用独立输出目录。";
  }
  const bpmHelp = element("#bpmHelp");
  if (bpmHelp) {
    bpmHelp.textContent = currentGenerationFamily() === "legacy_frame_v2"
      ? "旧基线：留空自动分析；填写则覆盖为单一 BPM，tempo map 可进一步覆盖。"
      : "留空即使用自动估计的 BPM；填写则完全采用你给的值。auto 会额外采用高置信度的变速点，建议仅在确认歌曲变速时使用。";
  }
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
    context_tokens: numberOrDefault("#contextTokens", 192),
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

// Shared by "开始训练" and the one-click button, so both send the same settings.
function trainingPayload(options = {}) {
  const mode = currentTrainingMode();
  // planner_profile values: "auto", "music_structure_v2", "legacy_v1".
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

function generationPayload() {
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
    // only take effect for a checkpoint whose contract advertises the extended
    // family; older checkpoints silently ignore them.
    conflict_detection_enabled: checkedOrDefault("#conflictDetectionEnabled", false),
    conflict_repair_mode: valueOrDefault("#conflictRepairMode", "none"),
    detect_soft_conflicts: checkedOrDefault("#detectSoftConflicts", false),
  };
}

function intensityCurvePayload() {
  if (!checkedOrDefault("#intensityCurveEnabled", false)) return null;
  const profile = intensityCurveState.profile;
  if (!intensityCurveState.saved || !profile || !Array.isArray(profile.points) || !profile.points.length) {
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

// ---------------------------------------------------------------------------
// Training-mode dependent controls
// ---------------------------------------------------------------------------
function setTrainingModeDefaults(force = false) {
  const mode = currentTrainingMode();
  const learningRate = element("#learningRate");
  if (learningRate && (force || !learningRate.dataset.touched)) {
    learningRate.value = mode === "finetune" ? "0.00002" : "0.0002";
  }
  const help = element("#learningRateHelp");
  if (help) help.textContent = mode === "finetune" ? "finetune 默认 2e-5。" : "fresh 默认 2e-4。";
  const modeHelp = element("#trainingModeHelp");
  if (modeHelp) {
    modeHelp.textContent = mode === "fresh"
      ? "从随机初始化开始；不需要已有 checkpoint。"
      : mode === "resume"
        ? "从断点继续：需要与当前 manifest、模型族完全一致的 checkpoint。"
        : "已有模型加强训练：加载权重并新建优化器；可以只用比 full 更小的精选集。允许调整质量标注和 Slide/Touch 采样，模型族、词表、窗口与规划器须保持兼容。";
  }
  const epochsHelp = element("#epochsHelp");
  if (epochsHelp) {
    epochsHelp.textContent = mode === "fresh"
      ? "fresh 模式的总训练轮数。配了验证 manifest 时建议填大（如 80），由早停决定实际轮数。"
      : mode === "resume"
        ? "resume 的 epochs 是从断点额外训练的轮数；会从 checkpoint 的 epoch 接着计数，早停会按验证 loss 停止。"
        : "finetune 从新阶段 epoch 1 开始计数；epochs 是本次加强训练的轮数，早停会按验证 loss 停止。";
  }
}

function updateDatasetButtons() {
  const busy = datasetUploading || preprocessStarting || trainingStarting;
  const mapping = {
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
}

// ---------------------------------------------------------------------------
// Task handling
// ---------------------------------------------------------------------------
const TASK_KIND_LABEL = {
  preprocess: "预处理", train: "训练", generate: "生成", gold: "金标集", acceptance: "验收",
};

function watchTask(taskId, onFinished = null) {
  activeTaskId = taskId;
  pollTask(taskId, onFinished);
}

async function pollTask(taskId, onFinished = null) {
  try {
    const task = await api(`/api/task?id=${encodeURIComponent(taskId)}`);
    if (activeTaskId === taskId) renderTask(task);
    if (["queued", "running"].includes(task.status)) {
      setTimeout(() => pollTask(taskId, onFinished), 800);
    } else if (onFinished) {
      await onFinished(task);
    }
  } catch (error) {
    const message = element("#taskMessage");
    if (message) message.textContent = error.message;
  }
}

function renderTask(task) {
  const kind = element("#taskKind");
  const status = element("#taskStatus");
  const progress = element("#progressBar");
  const message = element("#taskMessage");
  const logs = element("#taskLogs");
  if (kind) kind.textContent = `${TASK_KIND_LABEL[task.kind] || task.kind} · ${task.task_id}`;
  if (status) status.textContent = task.status;
  if (progress) progress.style.width = `${Math.round((task.progress || 0) * 100)}%`;
  if (message) message.textContent = task.error || task.message || "";
  if (logs) logs.textContent = (task.logs || []).join("\n") || "暂无日志";

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
      status.textContent = `生成状态：${verdict}；音符级警告 ${warnings} 条。${task.result?.quality_notice || ""}`;
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
    ? `上次准备的训练清单（只读）：${datasetWorkspaceState.prepared.manifest}`
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
  datasetWorkspaceState.selectionTouched = true;
  const items = selectedDatasetItems();
  if (!items.length) throw new Error("请至少勾选一个训练数据集。");
  const unprepared = items.filter((item) => !item.manifest || item.prepared === false);
  if (unprepared.length) throw new Error(`所选数据集尚未全部预处理：${unprepared.map((item) => item.name || item.path).join("、")}。请先进入对应卡片完成预处理。`);
  const manifests = items.map((item) => item.manifest).filter(Boolean);
  const paths = items.map((item) => item.path).filter(Boolean);
  if (manifests.length <= 1 && paths.length <= 1) {
    const item = items[0];
    if (item.path) selectDataset(item);
    if (item.manifest && element("#manifestPath")) element("#manifestPath").value = item.manifest;
    datasetWorkspaceState.prepared = { manifest: item.manifest };
    renderDatasetWorkspace();
    saveDatasetWorkspace();
    setStatus("已应用 1 个训练数据集。", true);
    return;
  }
  let data;
  try {
    data = await api("/api/datasets/compose", { manifests, dataset_paths: paths, datasets: items, model_family: currentModelFamily() });
  } catch (error) {
    // Older servers can still accept the first manifest. Never silently claim
    // that multiple sources were merged when the compose endpoint is absent.
    if (!String(error.message || "").match(/404|Not Found|不存在|未找到/i)) throw error;
    throw new Error("当前 WebUI 后端尚未提供多数据集合并接口，请重启/更新服务后再应用多选数据集。");
  }
  const manifest = data.manifest || data.effective_manifest || data.composed_manifest || data.path;
  if (!manifest) throw new Error("后端没有返回合并后的 manifest 路径。");
  datasetWorkspaceState.prepared = { ...data, manifest };
  renderDatasetWorkspace();
  invalidateDatasetStatus(`已合并 ${items.length} 个数据集，请重新检查训练集状态。`);
  saveDatasetWorkspace();
  const summary = data.summary || {};
  const summaryBox = element("#datasetCompositionSummary");
  if (summaryBox) {
    summaryBox.hidden = false;
    summaryBox.textContent = `已应用 ${items.length} 个来源：${summary.song_count ?? "?"} 首歌曲，${summary.difficulty_count ?? "?"} 张谱面；按原文件去重 ${summary.duplicate_source_charts ?? 0} 张；汇总质量标注 ${summary.quality_records ?? 0} 条。`
      + (summary.stale_quality_annotations ? ` ${summary.stale_quality_annotations} 条标注因原谱或音频变化已失效，请回原数据集重新检查并保存。` : "");
  }
  setStatus(`已应用 ${items.length} 个训练数据集。`, true);
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
  const intensityNotice = state.intensity_curve_capability === true ? " · 多维强度曲线可用" : " · 多维强度曲线需重启后端";
  setStatus(`本地服务正常 · workspace: ${state.workspace}${intensityNotice}`, state.intensity_curve_capability === true);
  populateDatasets(state.datasets || []);
  await loadDatasetCatalog();
  const running = (state.tasks || []).find((task) => ["queued", "running"].includes(task.status)
    && !String(task.kind || "").startsWith("intensity"));
  if (running) {
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
    "#contextTokens": familyParams.context_tokens,
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
    ? `生成能力：冲突检测 ${caps.conflict_detection ? "✓" : "✗"}，采音倾向 ${caps.rhythm_bias ? "✓" : "✗"}`
      + (caps.rhythm_bias ? "" : "（该 checkpoint 未在细碎节奏目标上训练，倾向参数会被忽略）")
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
    `模型结构：${JSON.stringify(data.model_config || {})}`,
    `数据合同：${JSON.stringify(data.data_contract || {})}`,
    data.model_config?.input_dim !== undefined ? `事件输入维度：${data.model_config.input_dim}` : "",
    data.model_config?.plan_dim !== undefined ? `Planner 维度：${data.model_config.plan_dim}（5 个连续规划量 + 4 个配置类别 logits）` : "",
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

async function inspectCheckpoint() {
  assertTrainingSelection();
  await assertDatasetBackend();
  const mode = currentTrainingMode();
  if (mode === "fresh") {
    checkpointInspection = { key: "", data: null };
    setCheckpointStatus("从头训练不需要 checkpoint。", "good");
    updateTrainingControls();
    return null;
  }
  const checkpointPath = valueOrDefault("#trainingCheckpointPath", "").trim();
  if (!checkpointPath) throw new Error("请先上传或填写已有模型 checkpoint 路径");
  const manifest = trainingSourcePaths().manifest;
  if (!manifest && !selectedTrainingManifests().length) throw new Error("请先选择训练数据集");
  const request = ++checkpointInspectRequest;
  const key = currentCheckpointKey();
  setCheckpointStatus("正在检查 checkpoint，并与当前 manifest 比对结构合同...");
  const box = element("#checkpointDetails");
  if (box) box.hidden = true;
  const data = await api("/api/checkpoint/inspect", {
    checkpoint_path: checkpointPath, model_family: currentModelFamily(), mode, manifest, manifests: selectedTrainingManifests(), dataset_paths: selectedDatasetItems().map((item) => item.path).filter(Boolean),
  });
  if (request !== checkpointInspectRequest || key !== currentCheckpointKey()) return null;
  checkpointInspection = { key, data };
  if (data.valid === true) applyCheckpointModelConfig(data);
  renderCheckpointInspection(data);
  updateTrainingControls();
  return data;
}

async function inspectGenerationCheckpoint() {
  const checkpointPath = valueOrDefault("#checkpointPath", "").trim();
  if (!checkpointPath) throw new Error("请先上传或填写生成使用的 checkpoint 路径");
  const request = ++generationInspectRequest;
  setGenerationStatus("正在检查生成 checkpoint...");
  const data = await api("/api/checkpoint/inspect", { checkpoint_path: checkpointPath, mode: "generate" });
  if (request !== generationInspectRequest) return null;
  generationInspection = { path: checkpointPath, data };
  updateFamilyFields();
  const family = data.model_family_label || data.model_family || "未知";
  const genCaps = data.generation_capabilities || {};
  const capSuffix = Object.keys(genCaps).length
    ? `；可用后处理：冲突检测${genCaps.conflict_detection ? "✓" : "✗"}，采音倾向${genCaps.rhythm_bias ? "✓" : "✗"}`
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
        : profile === "music_structure_v2"
          ? "当前为 music_structure_v2：auto / model 使用学得的局部规划，强度跟随幅度不限制其自然起伏。要明确调整局部强弱，请生成、编辑并保存曲线。"
          : "当前为 legacy_v1：auto 使用多维音频参考，model 不支持；密度和风格是软条件，模型响应可能较弱。";
  }
  const capabilityBox = element("#generationCapabilityNotice");
  if (capabilityBox) {
    capabilityBox.hidden = false;
    capabilityBox.textContent = !data.valid
      ? ""
      : genCaps.conflict_detection
        ? (genCaps.rhythm_bias
            ? "该 checkpoint 支持冲突检测与采音倾向。"
            : "该 checkpoint 支持冲突检测（后处理，无需重训）；但未在细碎节奏目标上训练，「节奏倾向 / 细碎音符强度」将被忽略。")
        : "非事件模型（或未知），不提供冲突检测。";
  }
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
  setWorkspaceMode("training");
  if (datasetUploading) throw new Error("请等待上传完成后再开始训练。");
  assertTrainingSelection({ autoPreprocess: !!options.autoPreprocess });
  await assertDatasetBackend();
  const payload = { ...trainingPayload({ autoPreprocess: !!options.autoPreprocess }), auto_preprocess: !!options.autoPreprocess };
  if (payload.model_family !== "legacy_frame_v2") {
    const status = await api("/api/status");
    if (status.training_capabilities?.event_learning_rate_control !== true) {
      throw new Error("当前后端尚未加载事件模型学习率修复，请重启 WebUI 服务后再训练。");
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
      && (index === 0 || Number(point.time_sec) > Number(points[index - 1].time_sec)));
}

function updateIntensityButtons() {
  const hasProfile = intensityProfileValid(intensityCurveState.profile);
  ["#intensityContrastButton", "#intensitySaveButton", "#applyIntensitySectionButton"]
    .forEach((selector) => { const node = element(selector); if (node) node.disabled = !hasProfile; });
  const undo = element("#intensityUndoButton");
  if (undo) undo.disabled = !hasProfile || intensityCurveState.history.length === 0;
  const reset = element("#intensityResetButton");
  if (reset) reset.disabled = !hasProfile || !intensityProfileValid(intensityCurveState.automatic);
  updateIntensityViewControls();
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

function normalizeIntensityProfile(profile) {
  if (!profile || !Array.isArray(profile.points)) return null;
  const points = profile.points.map((point) => ({
    time_sec: Number(point.time_sec),
    intensity: Math.max(0, Math.min(1, Number(point.intensity ?? point.value ?? 0.5))),
  })).filter((point) => Number.isFinite(point.time_sec)).sort((a, b) => a.time_sec - b.time_sec);
  const duration_sec = Number(profile.duration_sec || 0);
  return { ...profile, schema: "maichart-intensity-curve-v1", duration_sec, points,
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
  intensityCurveState.profile = normalized; intensityCurveState.automatic = cloneIntensityProfile(normalized); intensityCurveState.history = []; intensityCurveState.saved = false;
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
  pushIntensityHistory(); intensityCurveState.saved = false;
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
  profile.points = times.map((time_sec) => {
    const original = at(time_sec);
    let weight = time_sec >= start && time_sec <= end ? 1 : 0;
    if (blend > 0 && time_sec < start) weight = Math.max(0, Math.min(1, (time_sec - (start - blend)) / blend));
    if (blend > 0 && time_sec > end) weight = Math.max(0, Math.min(1, ((end + blend) - time_sec) / blend));
    return { time_sec, intensity: Number((original * (1 - weight) + target * weight).toFixed(4)) };
  });
  profile.edited = true; intensityView.selected.clear(); saveIntensityCurveState(); drawIntensityCurve(); updateIntensityButtons();
}

function saveIntensityCurveAndReturn() {
  if (!intensityProfileValid(intensityCurveState.profile)) throw new Error("请先生成强度曲线");
  intensityCurveState.profile.edited = true; intensityCurveState.saved = true; saveIntensityCurveState();
  const checkbox = element("#intensityCurveEnabled"); if (checkbox) checkbox.checked = true;
  setWorkspaceMode("generation"); setIntensityStatus("曲线已保存；生成页将使用这份人工曲线。", "good"); saveSettings();
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
    intensityCurveState.profile = null; intensityCurveState.automatic = null; intensityCurveState.history = []; intensityCurveState.saved = false;
    resetIntensityView();
    setIntensitySong(path); drawIntensityCurve(); updateIntensityButtons();
    const audio = element("#intensityAudio"); if (audio) { audio.pause(); audio.removeAttribute("src"); audio.load(); }
    saveIntensityCurveState(); setIntensityStatus("音频已变化，旧曲线已失效，请重新生成。", "warn");
  } else setIntensitySong(path);
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
  setWorkspaceMode("generation");
  const payload = generationPayload();
  if (!payload.checkpoint) throw new Error("请先上传或填写 checkpoint 路径");
  if (!payload.audio) throw new Error("请先导入音频或填写音频路径");
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
  watchTask(data.task_id, (task) => {
    if (task.status === "completed") setAcceptanceStatus("生成完成，可点击「验收谱面」做横向比较。", "good");
  });
}

function clearGenerate() {
  resetGenerationDefaults();
  ["#checkpointPath", "#audioPath", "#outputPath", "#tempoMap", "#goldStatsPath",
    "#acceptanceMaidata", "#acceptanceAudio", "#qualityStatus", "#qualityDiagnostics"]
    .forEach((selector) => {
      const target = element(selector);
      if (!target) return;
      if (target.tagName === "PRE" || target.tagName === "TEXTAREA") target.textContent = "";
    });
  ["#checkpointUploadStatus", "#audioUploadStatus", "#featureUploadStatus"].forEach((selector) => {
    if (element(selector)) element(selector).textContent = "未导入文件";
  });
  const panel = element("#acceptancePanel");
  if (panel) panel.hidden = true;
  setAcceptanceStatus("已清空生成参数；训练设置未改动。");
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
if (restoredSettings) {
  // Recompute mode-dependent defaults, then put the user's own lr back on top.
  const learningRate = valueOrDefault("#learningRate", "");
  setTrainingModeDefaults(true);
  if (learningRate && element("#learningRate")) element("#learningRate").value = learningRate;
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
    saveSettings();
  }, true);
});
window.addEventListener("beforeunload", flushSettings);

// Mode tabs
on("#trainingModeTab", "click", () => setWorkspaceMode("training"));
on("#qualityModeTab", "click", openDatasetWorkspace);
on("#generationModeTab", "click", () => setWorkspaceMode("generation"));
on("#intensityModeTab", "click", () => { setIntensitySong(intensityAudioPath()); setWorkspaceMode("intensity"); drawIntensityCurve(); });

// Training panel
on("#modelFamily", "change", () => { updateTrainingControls(); saveSettings(); });
on("#trainingMode", "change", () => { setTrainingModeDefaults(true); updateTrainingControls(); });
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
on("#datasetImportButton", "click", () => { setWorkspaceMode("training"); element("#rawRoot")?.scrollIntoView({ block: "center" }); element("#rawRoot")?.focus(); setStatus("上传训练集 ZIP，或填写本机目录并点击“登记当前目录到训练素材”。", true); });
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
on("#featureFile", "change", (event) => upload("audio_features", event.target.files[0], "#featurePath", "#featureUploadStatus"));
on("#checkpointFile", "change", async (event) => {
  const data = await upload("checkpoint", event.target.files[0], "#checkpointPath", "#checkpointUploadStatus");
  if (data) {
    generationInspection = { path: "", data: null };
    if (element("#generationPlannerStatus")) element("#generationPlannerStatus").textContent = "checkpoint 已更换，请重新检查 planner 类型。";
    setGenerationStatus("已上传 checkpoint；点击『检查生成 checkpoint 类型』确认模型族。", "warn");
  }
});
on("#checkpointPath", "input", () => {
  generationInspection = { path: "", data: null };
  if (element("#generationPlannerStatus")) element("#generationPlannerStatus").textContent = "checkpoint 路径已修改，请重新检查 planner 类型。";
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
  if (levelPresetProfile) applyConditionPreset();
});
on("#level", "change", () => { if (levelPresetProfile) applyConditionPreset(); });
on("#levelRange", "input", event => { setConditionValue("level", Number(event.target.value)); if (levelPresetProfile) applyConditionPreset(); });
on("#conditionPreferenceStrength", "input", () => { if (levelPresetProfile) applyConditionPreset(); else updateConditionRecipeDisplay(); });
for (const definition of Object.values(preferenceDefinitions)) {
  on(`#${definition.field}`, "change", () => { if (levelPresetProfile) applyConditionPreset(); else updateConditionRecipeDisplay(); });
}
on("#clearConditionPreferences", "click", () => {
  for (const definition of Object.values(preferenceDefinitions)) { const input = element(`#${definition.field}`); if (input) input.checked = false; }
  if (levelPresetProfile) applyConditionPreset(); else { updateConditionRecipeDisplay(); saveSettings(); }
});
for (const [key] of [...radarKeys, ...styleKeys]) {
  const markManual = () => { lastConditionRecipe = null; conditionEditedBeforeProfile = true; updateConditionRecipeDisplay(); };
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
