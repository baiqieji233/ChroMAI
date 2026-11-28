"use strict";

(() => {
  const $ = id => document.getElementById(id);
  if (!$("batchGenerationPanel")) return;
  const storageKey = "chromai.generationQueue.v1";
  const fields = new Map();
  const inherited = new Set(["action_temperature", "action_top_p", "onset_strength", "seed", "bpm", "first"]);
  const integerFields = new Set(["seed", "count", "parallel", "difficulty", "cpu_threads", "window_retries", "rollback_windows", "max_events"]);
  const controls = {
    "目标与采样": {
      level:"level", difficulty:"acceptanceDifficulty", temperature:"v2GenerateTemperature", action_temperature:"v2GenerateActionTemperature",
      top_p:"v2GenerateTopP", action_top_p:"v2GenerateActionTopP", seed:"seed", onset_strength:"v2GenerateOnsetStrength",
    },
    "写法与候选": {
      count:"v2GenerateCount", parallel:"v2GenerateParallel", candidates:"v2GenerateCandidates", phrase_search:"v2GeneratePhraseSearch",
      phrase_rank_policy:"v2GeneratePhraseRankPolicy", phrase_beats:"v2GeneratePhraseBeats", phrase_commit_beats:"v2GeneratePhraseCommitBeats",
      self_refine:"v2GenerateSelfRefine", configuration_recovery:"v2GenerateConfigurationRecovery", intent_mode:"v2GenerateIntentMode",
      reference_selection:"v2GenerateReferenceSelection", reference_strength:"v2GenerateReferenceStrength",
      recent_configuration_strength:"v2GenerateRecentConfigurationStrength", eos_suppression:"v2GenerateEosSuppression",
    },
    "节拍与检查": {
      tempo_detection:"v2GenerateTempoDetection", tempo_octave_policy:"v2GenerateTempoOctavePolicy", downbeat_alignment:"v2GenerateDownbeatAlignment",
      exhaustion_policy:"v2GenerateExhaustionPolicy", intercept_tap_on_slide:"v2GenerateInterceptTapOnSlide",
      tap_on_slide_threshold_ms:"v2GenerateTapOnSlideThresholdMs", intercept_slide_head_tap:"v2GenerateInterceptSlideHeadTap",
      forward_feasibility:"v2GenerateForwardFeasibility", forward_tap_on_slide_ms:"v2GenerateForwardTapOnSlideMs",
      touch_reachability_policy:"v2GenerateTouchReachabilityPolicy", touch_cohold_radius:"v2GenerateTouchCoholdRadius",
      window_retries:"v2GenerateWindowRetries", rollback_windows:"v2GenerateRollbackWindows",
    },
    "设备与输出": {
      device:"v2GenerateDevice", audio_device:"v2GenerateAudioDevice", inference_backend:"v2GenerateInferenceBackend",
      cpu_threads:"v2GenerateCpuThreads", cache_dir:"v2GenerateCacheDir", max_events:"v2GenerateMaxEvents",
      record_candidate_probabilities:"v2GenerateRecordCandidateProbabilities", designer:"designer",
    },
  };
  const labels = {
    level:"目标定数", temperature:"采音温度", action_temperature:"配置温度", top_p:"采音 Top-p", action_top_p:"配置 Top-p",
    candidates:"内部候选数", count:"每首输出张数", parallel:"每首同时生成数", seed:"种子", difficulty:"难度槽位",
  };
  function register(group, key, id, label, rowOnly = false) {
    const source = $(id);
    if (!source) return;
    const holder = source.closest("label");
    fields.set(key, {id, group, rowOnly, source, label:label || labels[key] ||
      holder?.querySelector("span")?.textContent.trim() || holder?.firstChild?.textContent.trim() || key});
  }
  for (const [group, entries] of Object.entries(controls)) {
    for (const [key,id] of Object.entries(entries)) register(group,key,id);
  }
  register("曲目信息", "checkpoint", "checkpointPath", "此曲模型", true);
  register("曲目信息", "title", "title", "输出曲名", true);
  register("曲目信息", "artist", "artist", "输出曲师", true);
  register("曲目信息", "bpm", "bpm", "BPM", true);
  register("曲目信息", "first", "first", "首拍偏移（秒）", true);
  register("Radar / Style", "radar_style_guidance", "v2RadarStyleGuidance", "启用 Radar / Style 引导");
  for (const [key,label] of [...radarKeys, ...styleKeys]) register("Radar / Style", "condition."+key, key, label);

  const defaults = {level:13.3, count:1, parallel:1, candidates:2, temperature:.9, action_temperature:.9, seed:null};
  const summaryKeys = ["level","count","candidates","temperature","action_temperature","seed"];
  const songInputs = new Set(["title","artist","bpm","first","tempo_map","meter_map","intensity_profile","intensity_strength"]);
  const hiddenFields = new Map([
    ["tempo_map","分段 BPM 表"],["meter_map","拍号表"],["intensity_profile","歌曲强度曲线"],["intensity_strength","强度曲线权重"],
  ]);
  let state;
  try { state = JSON.parse(localStorage.getItem(storageKey)); } catch (_) { /* start with an empty draft */ }
  if (!Array.isArray(state?.songs) || !Array.isArray(state?.presets)) {
    state = {base:{}, layers:[], songs:[], presets:[], selected:null, runs:[], currentRun:null};
  }
  state.singleLayers ??= [];
  let validation = null;
  let searching = false;
  let uploading = false;
  let runData = null;
  let timer = null;
  const dialog = $("batchDialog");
  const running = new Set(["queued", "running", "cancelling"]);

  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (className) el.className = className;
    return el;
  }
  function button(text, onclick, disabled = false, className = "secondary-button") {
    const el = node("button",text,className);
    el.type = "button";
    el.disabled = disabled;
    el.addEventListener("click",onclick);
    return el;
  }
  function option(value, text) {
    const el = node("option",text);
    el.value = value;
    return el;
  }
  function labeled(text, input) {
    const label = node("label",text,"field");
    label.append(input);
    return label;
  }
  function report(text, error = false) {
    const el = $("batchMessage");
    el.textContent = text;
    el.hidden = !text;
    el.classList.toggle("batch-error", error);
  }
  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(state)); }
    catch (_) { report("浏览器存储无法写入，本页仍可提交当前队列。", true); }
  }
  function dirty() {
    validation = null;
    persist();
    render();
  }
  function uid() {
    return Array.from(crypto.getRandomValues(new Uint32Array(3)), x=>x.toString(16)).join("-");
  }
  function fileName(path) {
    return String(path ?? "").split(/[\\/]/).pop() || "";
  }
  function titleFor(song) {
    return song.name || fileName(song.audio);
  }
  function preset(id) {
    return state.presets.find(p=>p.id===id);
  }
  function compose(song = null) {
    const values = {}, source = {};
    function add(settings, owner) {
      for (const [key,value] of Object.entries(settings || {})) {
        values[key] = value;
        source[key] = owner;
      }
    }
    add(state.base, "基准");
    for (const id of state.layers) { const p = preset(id); if (p) add(p.values,"全批 · "+p.name); }
    for (const id of song?.layers || []) { const p = preset(id); if (p) add(p.values,"单曲 · "+p.name); }
    add(song?.overrides, "单曲手动");
    return {values, source};
  }
  function request() {
    return {songs:state.songs.map(song=>{
      const settings = {model_family:"chart_composer_v2", audio:song.audio, condition:{}};
      const {values,source}=compose(song);
      for (const [key,value] of Object.entries(values)) {
        if (key.startsWith("condition.")) {
          if (value !== null) settings.condition[key.slice(10)] = value;
        } else {
          settings[key] = value;
        }
      }
      return {id:song.id, name:titleFor(song), settings, sources:source};
    })};
  }
  function fieldName(key) {
    return fields.get(key)?.label || hiddenFields.get(key) || key;
  }
  function display(key,value) {
    if (value === null || value === undefined) return key.startsWith("action_") ? "跟随采音" : "自动";
    if (key === "intensity_curve" || key === "intensity_profile") return "已关联";
    if (Array.isArray(value)) return value.length + " 段";
    if (typeof value === "boolean") return value ? "开启" : "关闭";
    if (key === "candidates") return String(value) + " 种";
    const opt = Array.from(fields.get(key)?.source.options || []).find(x=>x.value===String(value));
    return opt?.textContent || String(value);
  }
  function layerList(ids, scope, modified = dirty) {
    const block = node("div");
    const choices = node("select");
    choices.setAttribute("aria-label", "添加"+scope+"预设");
    choices.append(option("", "选择预设"));
    for (const p of state.presets.filter(p=>!ids.includes(p.id))) choices.append(option(p.id,p.name));
    const add = button("加入",()=>{
      if (!choices.value) return;
      ids.push(choices.value);
      modified();
    },true);
    choices.onchange = ()=>{add.disabled = !choices.value;};
    const controls = node("div",undefined,"batch-actions");
    controls.append(choices,add,node("small","越靠下优先级越高","muted"));
    block.append(controls);
    if (!ids.length) return block;
    const list = node("ol",undefined,"batch-stack");
    list.setAttribute("aria-label",scope+"预设优先级");
    ids.forEach((id,index)=>{
      const p = preset(id);
      if (!p) return;
      const item = node("li");
      const text = node("div");
      const keys = Object.keys(p.values);
      text.append(node("strong",p.name),node("small",keys.slice(0,3).map(fieldName).join(" · ")+(keys.length>3 ? " 等 "+keys.length+" 项" : "")));
      const shadowed = keys.filter(key=>ids.slice(index+1).some(later=>Object.hasOwn(preset(later)?.values || {},key)));
      if (shadowed.length) text.append(node("small","被下方预设覆盖 · "+shadowed.map(fieldName).join("、")));
      const actions = node("div",undefined,"batch-actions");
      for (const [delta,label] of [[-1,"上移"],[1,"下移"]]) {
        const move = button(label,()=>{
          [ids[index],ids[index+delta]] = [ids[index+delta],ids[index]];
          modified();
          const target = scope==="生成页" ? dialog : $(scope==="全批" ? "batchLayers" : "batchSongSettings");
          Array.from(target.querySelectorAll("button[data-layer]")).find(b=>b.dataset.layer===id && !b.disabled)?.focus();
        },index+delta<0 || index+delta>=ids.length,"batch-row-action");
        move.dataset.layer = id;
        move.setAttribute("aria-label",label+" "+p.name);
        actions.append(move);
      }
      actions.append(button("移除",()=>{ids.splice(index,1);modified();},false,"batch-row-action"));
      item.append(text,actions);
      list.append(item);
    });
    block.append(list);
    return block;
  }
  function settingsTable(song, all = false) {
    const {values,source} = compose(song);
    const validated = validation?.jobs?.find(job=>job.id===song.id);
    const effective = validated?.settings || {};
    const merged = {...defaults,...values,...effective, audio:song.audio};
    delete merged.condition;
    for (const [k,v] of Object.entries(effective.condition || {})) merged["condition."+k]=v;
    const first = ["checkpoint","audio",...summaryKeys];
    const keys = all ? [...new Set([...first,...fields.keys(),...hiddenFields.keys()])]
      .filter(key=>Object.hasOwn(merged,key)) : summaryKeys;
    const table = node("table",undefined,"batch-source-table");
    const head = node("tr");
    for (const title of ["参数","最终值","来源"]) head.append(node("th", title));
    table.createTHead().append(head);
    const body = table.createTBody();
    for (const key of keys) {
      const row = body.insertRow();
      const value = key==="seed" && validated?.automatic_seed ? null : merged[key];
      const text = key==="parallel" && effective.effective_parallel && effective.effective_parallel!==value
        ? effective.effective_parallel+"（设定 "+value+"）" : display(key,value);
      row.append(node("td",key==="audio" ? "音频" : fieldName(key)),
        node("td",text,"batch-value"),
        node("td",source[key] || (key==="audio" ? "单曲" : "生成默认值")));
    }
    return table;
  }
  function render() {
    $("batchCheckpoint").value = state.base.checkpoint || "";
    $("batchLayers").replaceChildren(layerList(state.layers,"全批"));
    $("batchSongCount").textContent = state.songs.length ? String(state.songs.length) : "";
    $("batchEmpty").hidden = state.songs.length > 0;
    const list = $("batchSongs");
    list.replaceChildren();
    let selected = state.songs.find(s=>s.id===state.selected);
    if (!selected) selected = state.songs[0];
    if (selected) state.selected = selected.id;
    state.songs.forEach((song,index)=>{
      const li = node("li");
      if (song === selected) li.setAttribute("aria-current","true");
      const select = button("",()=>{state.selected=song.id;persist();render();},false,"batch-song-choice");
      const {values} = compose(song);
      const status = validation?.jobs?.find(job=>job.id===song.id);
      select.title = song.audio;
      select.append(node("strong",titleFor(song)),
        node("small",status?.error || `${values.level ?? defaults.level} 定数 · ${values.count ?? 1} 张`,status?.error ? "batch-error" : ""));
      li.append(select);
      const up = button("↑",()=>{[state.songs[index-1],state.songs[index]]=[state.songs[index],state.songs[index-1]];dirty();},index===0,"batch-row-action");
      up.setAttribute("aria-label","提前执行 "+titleFor(song));
      const remove = button("移除",()=>{state.songs.splice(index,1);dirty();},false,"batch-row-action");
      remove.setAttribute("aria-label","移除 "+titleFor(song));
      li.append(up,remove);
      list.append(li);
    });
    const detail = $("batchSongSettings");
    detail.hidden = !selected;
    detail.replaceChildren();
    if (selected) {
      const heading = node("div",undefined,"batch-heading");
      const actions = node("div",undefined,"batch-actions");
      actions.append(button("复制此曲",()=>{
        const copy = structuredClone(selected);
        copy.id = uid();
        copy.name = titleFor(selected) + "（对照）";
        state.songs.splice(state.songs.indexOf(selected)+1,0,copy);
        state.selected = copy.id;
        dirty();
      }),button("编辑单曲参数",()=>editSettings("song",selected)));
      heading.append(node("h3",titleFor(selected)),actions);
      detail.append(heading,node("p","单曲预设覆盖全批设置，手动参数最后生效。","muted"),
        layerList(selected.layers,"单曲"),settingsTable(selected),button("查看全部生效参数",()=>inspectSettings(selected)));
      const invalid = validation?.jobs?.find(job=>job.id===selected.id)?.error;
      if (invalid) detail.append(node("p",invalid,"batch-error"));
    }
    const count = state.songs.reduce((sum,s)=>sum+(Number(compose(s).values.count)||1),0);
    $("batchPlanSummary").textContent = state.songs.length ? `${state.songs.length} 首歌曲 · ${count} 张谱面` : "队列为空";
    $("batchPreview").disabled = searching || uploading || !state.songs.length;
    $("batchStart").disabled = searching || uploading || !state.songs.length;
  }

  function showDialog(title, content, footer = null) {
    if (dialog.open) dialog.close();
    const head = node("header",undefined,"batch-heading");
    const label = node("h3",title);
    label.id = "batchDialogTitle";
    dialog.setAttribute("aria-labelledby",label.id);
    head.append(label,button("关闭",()=>dialog.close()));
    dialog.replaceChildren(head,content);
    if (footer) dialog.append(footer);
    dialog.showModal();
  }
  function inspectSettings(song) {
    const text = node("div");
    text.append(settingsTable(song,true));
    showDialog(titleFor(song)+" · 生效参数",text);
  }
  function inputFor(key,value) {
    const meta = fields.get(key);
    const source = meta.source;
    let el;
    if (source.tagName==="SELECT") {
      el = node("select");
      for (const opt of source.options) el.append(option(opt.value,opt.textContent));
      if (!Array.from(el.options).some(o=>o.value===String(value))) el.append(option(String(value),String(value)));
      el.value = String(value);
    } else {
      el = node("input");
      const conditional = key.startsWith("condition.");
      el.type = source.type==="checkbox" ? "checkbox" :
        conditional || source.type==="number" ? "number" : "text";
      if (el.type==="checkbox") el.checked=Boolean(value);
      else el.value = value ?? "";
      if (el.type==="number") {
        el.min = conditional ? 0 : key==="level" ? 1 : source.min || "";
        el.max = conditional ? 1 : source.max || "";
        el.step = integerFields.has(key) ? "1" : "any";
        if (conditional || inherited.has(key)) el.placeholder = display(key,null);
      }
    }
    return el;
  }
  function readInput(key, el) {
    if (el.type==="checkbox") return el.checked;
    const raw = el.value.trim();
    if (el.type !== "number") return integerFields.has(key) ? Number(raw) : raw;
    if (!raw && (inherited.has(key) || key.startsWith("condition."))) return null;
    const n = Number(raw);
    if (!raw || !el.checkValidity() || !Number.isFinite(n) ||
        (integerFields.has(key) && !Number.isSafeInteger(n)) || (key.endsWith("top_p") && n<=0)) {
      throw new Error("请检查「"+fieldName(key)+"」的数值");
    }
    return n;
  }
  function liveValue(key) {
    const meta = fields.get(key);
    if (key.startsWith("condition.")) {
      const c = v2ConditionState[key.slice(10)];
      return c?.override ? Number(c.value) : null;
    }
    if (key==="first" && $("v2GenerateAutoFirst")?.checked) return null;
    const source = meta.source;
    if (source.type==="checkbox") return source.checked;
    if (source.type==="number" || integerFields.has(key)) return source.value.trim()==="" ? null : Number(source.value);
    return source.value.trim();
  }
  function editSettings(mode, item = null, single = false) {
    const isPreset = mode==="preset", isSong = mode==="song";
    const working = {...(isPreset ? item?.values : isSong ? item.overrides : state.base)};
    const registered = [...fields.entries()].filter(([key,f])=>isSong || !f.rowOnly);
    const content = node("div");
    const error = node("p",undefined,"batch-error");
    error.setAttribute("role","alert");
    const name = node("input");
    name.type = "text";name.maxLength=80;name.value=item?.name || "";
    if (isPreset) content.append(labeled("预设名称",name));
    let path, songName, clockInputs;
    if (isSong) {
      songName = node("input");
      songName.value = titleFor(item);
      songName.maxLength=160;
      path = node("input");
      path.value = item.audio;
      path.spellcheck=false;
      content.append(labeled("队列显示名称",songName),labeled("音频路径",path));
      const clocks = node("details",undefined,"settings-section");
      clocks.append(node("summary","分段 BPM、拍号与强度曲线"));
      const options = node("div",undefined,"settings-content");
      clockInputs = new Map();
      for (const key of ["tempo_map","meter_map"]) {
        const input = node("textarea");
        input.value = working[key] ? JSON.stringify(working[key],null,2) : "";
        input.rows = 3;
        input.spellcheck = false;
        if (key==="tempo_map") input.placeholder='[{"beat":"0","bpm":180},{"beat":"32","bpm":200}]';
        options.append(labeled(fieldName(key),input));
        clockInputs.set(key,input);
      }
      options.append(node("small","留空使用当前歌曲的自动时钟，或使用上面填写的恒定 BPM。","muted"));
      if (working.intensity_profile) {
        const clear = button("解除当前强度曲线",()=>{
          delete working.intensity_profile;
          delete working.intensity_strength;
          clear.disabled=true;
          clear.textContent="已解除强度曲线";
        });
        options.append(clear);
      }
      clocks.append(options);
      content.append(clocks);
    }
    content.append(node("p",isPreset ? "添加要保存的参数，其余由上层或生成默认值决定。" :
      isSong ? "添加的参数只覆盖这首歌。移除一项即可恢复继承。" : "基准参数会被全批预设覆盖。","muted"));
    const addRow = node("div",undefined,"batch-actions");
    const select = node("select");
    select.setAttribute("aria-label","选择要添加的参数");
    const inputs = new Map();
    const editor = node("div",undefined,"batch-param-editor");
    const add = button("添加参数",()=>{
      const key = select.value;
      if (!key) return;
      const {values} = compose(isSong ? item : null);
      const actual = isPreset && single ? liveValue(key) : Object.hasOwn(values,key) ? values[key] : liveValue(key);
      working[key] = actual ?? null;
      renderFields(key);
    },true);
    select.onchange = ()=>{add.disabled = !select.value;};
    addRow.append(select,add);
    if (isPreset && single) {
      addRow.append(button("加入生成页已设置的参数",()=>{
        for (const [key,meta] of registered) {
          const val = liveValue(key);
          if (!meta.source.disabled && val !== null && val !== "" && !Object.hasOwn(working,key)) working[key]=val;
        }
        renderFields();
      }));
    }
    content.append(addRow,editor,error);
    function renderFields(focusKey) {
      const old = new Map(inputs);
      inputs.clear();
      editor.replaceChildren();
      select.replaceChildren(option("","选择参数"));
      for (const group of new Set(registered.map(([,f])=>f.group))) {
        const options = node("optgroup");
        options.label = group;
        for (const [key,f] of registered.filter(([,f])=>f.group===group)) {
          if (!Object.hasOwn(working,key)) options.append(option(key,f.label));
          else {
            const el = old.get(key) || inputFor(key,working[key]);
            el.id="batch-param-"+key.replace(".","-");
            inputs.set(key,el);
            const label=node("label",f.label);
            label.htmlFor=el.id;
            const row=node("div",undefined,"batch-param");
            row.append(label,el,button("恢复继承",()=>{delete working[key];inputs.delete(key);renderFields();select.focus();},false,"batch-row-action"));
            editor.append(row);
          }
        }
        if (options.children.length) select.append(options);
      }
      add.disabled=true;
      inputs.get(focusKey)?.focus();
    }
    renderFields();
    const footer = node("div",undefined,"batch-actions");
    footer.append(button(isPreset ? "保存预设" : "保存设置",()=>{
      try {
        for (const [key,el] of inputs) working[key] = readInput(key,el);
        if (isPreset) {
          const title = name.value.trim();
          if (!title || !inputs.size) throw new Error("请填写名称并添加至少一个参数");
          if (state.presets.some(p=>p.id!==item?.id && p.name===title)) throw new Error("已有同名预设");
          if (item) Object.assign(item,{name:title,values:working});
          else state.presets.push({id:uid(),name:title,values:working});
          report("已保存「"+title+"」。");
        } else if (isSong) {
          const audio=path.value.trim();
          if (!audio) throw new Error("请输入音频路径");
          for (const [key,input] of clockInputs) {
            const raw=input.value.trim();
            let value=null;
            if (raw) {
              try { value=JSON.parse(raw); }
              catch (_) { throw new Error(fieldName(key)+"需要有效的 JSON 数组"); }
              if (!Array.isArray(value) || !value.length) throw new Error(fieldName(key)+"需要非空数组");
            }
            if (audio!==item.audio && JSON.stringify(value)===JSON.stringify(item.overrides[key] ?? null)) value=null;
            if (value===null) delete working[key]; else working[key]=value;
          }
          if (audio!==item.audio) {
            delete working.intensity_profile;
            delete working.intensity_strength;
            for (const key of ["title","artist","bpm","first"]) {
              if (Object.is(item.overrides[key],working[key])) delete working[key];
            }
          }
          const alias=songName.value.trim();
          item.name=audio!==item.audio && alias===titleFor(item) ? fileName(audio) : alias || fileName(audio);
          item.audio=audio;
          item.overrides=working;
        } else {
          state.base=working;
        }
        dialog.close();dirty();
        if (isPreset && single) openLibrary(true);
      } catch (ex) { error.textContent=ex.message; }
    },false,"primary-button"));
    showDialog(isPreset ? (item ? "编辑预设" : "保存预设") : isSong ? "编辑单曲参数" : "编辑基准参数",content,footer);
  }

  function applyToSingle(values) {
    if (currentGenerationFamily()!=="chart_composer_v2") throw new Error("请在生成页选择 Composer v2 模型");
    const skipped=[];
    function apply(key, value) {
      if (key.startsWith("condition.")) {
        if (!v2ConditionCapability().enabled && value!==null) {skipped.push(fieldName(key));return;}
        const entry=v2ConditionState[key.slice(10)];
        entry.override=value!==null;
        entry.edited=value!==null;
        if (value!==null) entry.value=value;
        return;
      }
      const input = fields.get(key)?.source;
      if (!input || input.disabled) {skipped.push(fieldName(key));return;}
      if (input.type==="checkbox") input.checked=value;
      else input.value=value ?? "";
    }
    const deferred = new Set(["parallel","phrase_rank_policy"]);
    for (const [key,value] of Object.entries(values)) if (!deferred.has(key)) apply(key,value);
    updateV2BatchControls();
    for (const key of deferred) if (Object.hasOwn(values,key)) apply(key,values[key]);
    updateV2ConditionControls();
    markActivePresetChip("data-v2-gen-preset","");
    saveSettings();
    window.MaichartWorkflow?.refresh();
    const status=$("v2GenPresetStatus");
    if (status) status.textContent=skipped.length ? "已套用可用参数。当前模型未启用 "+skipped.join("、")+"。" : "已套用保存的参数。";
  }
  function openLibrary(single=false) {
    const content=node("div");
    const error=node("p",undefined,"batch-error");
    const list=node("ul",undefined,"batch-library");
    if (single && state.presets.length) {
      const area=node("section",undefined,"batch-shared");
      const rows=node("div");
      const apply=button("套用组合",()=>{
        const values={};
        for (const id of state.singleLayers) Object.assign(values,preset(id)?.values);
        try { applyToSingle(values);dialog.close(); } catch (ex) { error.textContent=ex.message; }
      },!state.singleLayers.length,"primary-button");
      const refresh=()=>{
        persist();
        rows.replaceChildren(layerList(state.singleLayers,"生成页",refresh));
        apply.disabled=!state.singleLayers.length;
      };
      refresh();
      area.append(node("strong","套用到当前生成表单"),rows,apply);
      content.append(area);
    }
    for (const p of state.presets) {
      const li=node("li");
      const info=node("div");
      info.append(node("strong",p.name),node("small",Object.entries(p.values).map(([k,v])=>fieldName(k)+" "+display(k,v)).join(" · ")));
      const actions=node("div",undefined,"batch-actions");
      if (single) actions.append(button("套用",()=>{
        try { applyToSingle(p.values);dialog.close(); }
        catch (ex) { error.textContent=ex.message; }
      }));
      actions.append(button("编辑",()=>editSettings("preset",p,single)),button("删除",()=>{
        if (!window.confirm('删除「'+p.name+'」并将它从当前队列的预设列表中移除？')) return;
        state.presets=state.presets.filter(x=>x.id!==p.id);
        state.layers=state.layers.filter(id=>id!==p.id);
        state.singleLayers=state.singleLayers.filter(id=>id!==p.id);
        for (const s of state.songs) s.layers=s.layers.filter(id=>id!==p.id);
        dirty();openLibrary(single);
      }));
      li.append(info,actions);list.append(li);
    }
    content.append(button(single ? "从当前设置保存预设" : "保存新预设",()=>editSettings("preset",null,single)),
      node("p", "预设保存在当前浏览器，可重复用于不同歌曲。","muted"));
    if (!state.presets.length) content.append(node("p","还没有保存的预设。","muted"));
    content.append(list,error);
    showDialog("生成预设库",content);
  }

  function addSong(audio,name,overrides={}) {
    const song={id:uid(),name,audio,layers:[],overrides};
    state.songs.push(song);
    state.selected=song.id;
  }
  function pastePaths() {
    const box=node("textarea");
    box.rows=5;box.spellcheck=false;
    const filePaths=labeled("每行一条本机音频路径",box);
    const e=node("p",undefined,"batch-error");
    const wrap=node("div");
    wrap.append(filePaths,e);
    const add=button("加入队列",()=>{
      const paths=box.value.split(/\r?\n/).map(p=>p.trim().replace(/^"|"$/g,"")).filter(Boolean);
      if (!paths.length) {
        e.textContent="请至少填写一条音频路径。";return;
      }
      for (const path of paths) addSong(path,fileName(path));
      dialog.close();dirty();
    },false,"primary-button");
    showDialog("添加音频路径",wrap,add);
  }
  async function selectFiles(event) {
    const files=[...event.target.files];
    event.target.value="";
    if (!files.length) return;
    uploading=true;render();
    const failed=[];
    for (const file of files) {
      report("正在导入 "+file.name+" …");
      try {
        const form=new FormData();
        form.append("kind","audio");form.append("file",file,file.name);
        const response=await fetch("/api/upload",{method:"POST",body:form});
        const data=await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "上传失败");
        addSong(data.path,file.name);
      } catch (ex) { failed.push(file.name+" · "+ex.message); }
    }
    uploading=false;
    dirty();
    report(failed.length ? failed.join("；") : "已导入 "+files.length+" 首歌曲。",Boolean(failed.length));
  }
  async function send(start) {
    searching=true;
    render();
    report(start ? "正在提交队列…" : "正在核对每首歌的实际生成参数…");
    const body=request();
    const signature=JSON.stringify(body);
    try {
      const data=await api("/api/composer-v2/generation_queue"+(start ? "" : "_preview"),body);
      if (data.task_id) {
        state.runs=[data.task_id,...state.runs.filter(id=>id!==data.task_id)].slice(0,20);
        state.currentRun=data.task_id;runData=null;
        persist();report("已启动批量生成。");
        void pollRun();
      } else if (signature===JSON.stringify(request())) {
        validation=data;
        report(data.valid ? "所有歌曲的参数已通过核对。" : "有歌曲的参数需要调整，详情见队列。",!data.valid);
      }
    } catch (ex) { report(ex.message,true); }
    finally { searching=false;render(); }
  }
  const statusNames = {queued:"等待",running:"生成中",completed:"完成",partial:"部分完成",failed:"失败",cancelled:"已停止",cancelling:"正在停止",interrupted:"已中断"};
  function renderRun() {
    const panel=$("batchRuns");
    if (!state.runs.length) {panel.hidden=true;return;}
    panel.hidden=false;
    const head=node("div",undefined,"batch-heading");
    head.append(node("h3","批量生成任务"));
    const history=node("select");
    history.setAttribute("aria-label","选择批量任务");
    state.runs.forEach(id=>history.append(option(id,id===state.runs[0]?"最近一批 · "+id:id)));
    history.value=state.currentRun;
    history.onchange=()=>{state.currentRun=history.value;runData=null;persist();void pollRun();};
    head.append(history);
    panel.replaceChildren(head);
    if (!runData) return;
    const {result,status}=runData;
    const progress=node("progress");
    progress.max=1;
    progress.value=Math.min(1,(runData.progress || 0)+(result.active_job?.progress || 0)/Math.max(1,result.jobs.length));
    progress.setAttribute("aria-label","批量进度");
    const canStop=running.has(status) && (status!=="cancelling" || runData.cancel_retryable);
    if (canStop) head.append(button(runData.cancel_retryable ? "重试停止" : "停止这批",async()=>{
      try { await api("/api/cancel",{task_id:runData.task_id});void pollRun(); }
      catch (e) { report(e.message,true); }
    }));
    if (result.manifest) {
      const link=node("a","下载批次参数与结果");
      link.href="/api/download?path="+encodeURIComponent(result.manifest);
      head.append(link);
    }
    panel.append(progress,node("p",(statusNames[status] || status)+(runData.error ? " · "+runData.error : ""),"muted"));
    for (const job of result.jobs) {
      const row=node("div",undefined,"batch-run-row");
      const info=node("div");
      const current=job.task_id===result.active_task_id ? result.active_job : null;
      info.append(node("strong",job.name),node("small",(statusNames[job.status] || job.status)+
        (job.produced ? " · "+job.produced+" 张" : "")+(job.settings ? " · 种子 "+job.settings.seed : "")+
        (current?.message ? " · "+current.message : "")));
      if (job.error) info.append(node("small",job.error.split("\n")[0],"batch-error"));
      row.append(info);
      if (job.task_id) {
        const actions=node("div",undefined,"batch-actions");
        if (job.status==="completed" || job.status==="partial") actions.append(button("预览谱面",async()=>{
          setWorkspaceMode("generation");
          await ChartPreview.openTask({task_id:job.task_id,kind:"composer_v2_generate",status:"completed"});
        }));
        actions.append(button("日志",()=>{watchTask(job.task_id);window.MaichartWorkflow?.setLogsExpanded(true);}));
        row.append(actions);
      }
      panel.append(row);
    }
  }
  async function pollRun() {
    clearTimeout(timer);
    const id=state.currentRun;
    if (!id) {renderRun();return;}
    try {
      const data=await api("/api/composer-v2/generation_queue?id="+encodeURIComponent(id));
      if (id!==state.currentRun) return;
      runData=data;
      renderRun();
      if (running.has(data.status)) timer=setTimeout(pollRun,1500);
    } catch (ex) {
      if (id!==state.currentRun) return;
      renderRun();
      $("batchRuns").append(node("p",ex.message,"batch-error"),button("重新读取任务",pollRun));
    }
  }

  $("batchGenerationModeTab").onclick=()=>setWorkspaceMode("batch");
  $("batchCheckpoint").onchange=e=>{state.base.checkpoint=e.target.value.trim();dirty();};
  $("batchPresetNew").onclick=()=>editSettings("preset");
  $("batchPresetLibrary").onclick=()=>openLibrary();
  $("batchSinglePresets").onclick=()=>openLibrary(true);
  $("batchEditBase").onclick=()=>editSettings("base");
  $("batchCopyBase").onclick=()=>{
    if (currentGenerationFamily()!=="chart_composer_v2") {report("请先在生成页选择 Composer v2 模型。",true);return;}
    const checkpoint=$("checkpointPath").value.trim();
    const values={};
    for (const [key,f] of fields) {
      if (f.rowOnly || f.source.disabled) continue;
      const value=liveValue(key);
      if (value===null && !["seed","action_temperature","action_top_p"].includes(key)) continue;
      if (!Number.isNaN(value) && value!=="") values[key]=value;
    }
    state.base={...values,checkpoint:checkpoint || state.base.checkpoint || ""};
    report("已复制生成页设置到基准。");
    dirty();
  };
  $("batchAddPaths").onclick=pastePaths;
  $("batchChooseAudio").onclick=()=>$("batchAudioFiles").click();
  $("batchAudioFiles").onchange=selectFiles;
  $("batchAddCurrent").onclick=()=>{
    try {
      const p=composerV2GenerationPayload();
      if (!p.audio) throw new Error("请先在生成页导入一首歌曲");
      const song={};
      for (const key of songInputs) if (p[key]!==undefined && p[key]!==null) song[key]=p[key];
      if ($("seed").value.trim()) song.seed=p.seed;
      if (!state.base.checkpoint && p.checkpoint) state.base.checkpoint=p.checkpoint;
      addSong(p.audio,p.title || $("audioPath").dataset.originalName || fileName(p.audio),song);
      dirty();
    } catch (ex) {report(ex.message,true);}
  };
  $("batchPreview").onclick=()=>send(false);
  $("batchStart").onclick=()=>send(true);
  window.addEventListener("workspace-mode-changed", event=>{
    if (event.detail.mode!=="batch") return;
    if (!state.base.checkpoint && $("checkpointPath").value.trim()) {
      state.base.checkpoint=$("checkpointPath").value.trim();
      persist();
    }
    render();
    void pollRun();
  });
  if (document.body.dataset.workspace === "batch") {
    render();
    void pollRun();
  }
})();
