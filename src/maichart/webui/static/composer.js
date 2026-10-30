(() => {
  const $ = (s) => document.querySelector(s);
  const formData = (form) => Object.fromEntries([...form.elements].filter((x) => x.name).map((x) => [x.name, x.type === "checkbox" ? x.checked : x.value]));
  const settingsKey = 'maichart.composer.controls.v3';
  const inheritedKeys = ['architecture_profile','lens_width','lens_chunk_size','history_residual_bound',
    'history_dropout','ledger_dropout','placement_weight','planner_weight','recovery_interval',
    'recovery_samples','recovery_steps','recovery_weight','sample_locality','symbolic_dropout',
    'audio_pointer_weight','d_model','dim_feedforward','layers','nhead','history_events',
    'candidate_count','action_slots','lr_schedule','warmup_steps','min_lr_ratio',
    'structure_policy','structure_backend','structure_dropout','contact_policy','configuration_quality_policy','sample_sequence_policy'];
  const forms = ['trainForm','generateForm'].map((id) => document.getElementById(id));
  function saveControls() {
    try { localStorage.setItem(settingsKey, JSON.stringify(Object.fromEntries(forms.map((form) => [form.id,formData(form)])))); } catch (_) {}
  }
  try {
    const saved = JSON.parse(localStorage.getItem(settingsKey) || '{}');
    for (const form of forms) for (const field of form.elements) {
      const value=saved[form.id]?.[field.name];
      if (field.name && value!==undefined) { if (field.type==='checkbox') field.checked=Boolean(value); else field.value=String(value); }
    }
  } catch (_) {}
  for (const form of forms) { form.addEventListener('input',saveControls); form.addEventListener('change',saveControls); }
  const request = async (url, body) => { const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); const data = await r.json(); if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`); return data; };
  let active = "", timer = 0;
  function showError(error) { $("#taskMessage").textContent = error.message || String(error); $("#taskMessage").className = "bad"; }
  function render(task) {
    $("#taskKind").textContent = `${task.kind || "composer"} · ${task.task_id || ""}`;
    $("#taskState").textContent = ({queued: "排队中", running: "运行中", completed: "完成", failed: "失败", cancelled: "已中断"})[task.status] || task.status || "未知";
    $("#taskMessage").textContent = task.error || task.message || ""; $("#taskMessage").className = task.error ? "bad" : "";
    const p = Number(task.progress); $("#progress").style.width = `${Math.max(0, Math.min(1, Number.isFinite(p) ? p : 0)) * 100}%`;
    const result = task.result || {}, effective=result.effective_settings||{};
    const settings=task.kind==='composer_train'
      ? [`架构 ${effective.architecture_profile||'未记录'}`,result.settings_policy==='resume_inherited'?'继承 checkpoint 参数':'当前请求参数',
          ...inheritedKeys.filter((key)=>key!=='architecture_profile'&&effective[key]!==undefined).map((key)=>`${key}=${effective[key]}`)]
      : effective.pointer_history_strength===undefined?[]:[`选时历史权重 ${effective.pointer_history_strength}（1 完整历史，0 音频分支）`];
    $("#progressText").textContent = [result.epoch, result.batch, result.loss, result.events_per_sec,...settings].filter((x) => x !== undefined && x !== null).join(" · ");
    $("#logs").textContent = Array.isArray(task.logs) ? task.logs.join("\n") : (task.logs || ""); $("#logs").scrollTop = $("#logs").scrollHeight;
    const paths = []; for (const key of ["output", "checkpoint_dir", "log_path"]) if (result[key]) paths.push([key, result[key]]); if (task.kind === "composer_generate" && result.output) paths.push(["composer.json", result.output.replace(/\.[^.]+$/, ".composer.json")]);
    $("#artifacts").replaceChildren(...(paths.length ? paths.map(([label, path]) => { const line = document.createElement("div"); const a = document.createElement("a"); a.href = `/api/download?path=${encodeURIComponent(path)}`; a.target = "_blank"; a.textContent = `${label}: ${path}`; line.append(a); return line; }) : [document.createTextNode("完成后显示")]));
    $("#cancel").disabled = !["queued", "running"].includes(task.status);
  }
  async function poll(id) { active = id; clearTimeout(timer); const tick = async () => { try { const task = await (await fetch(`/api/task?id=${encodeURIComponent(id)}`)).json(); render(task); if (!["completed", "failed", "cancelled"].includes(task.status)) timer = setTimeout(tick, 1000); } catch (error) { showError(error); timer = setTimeout(tick, 2000); } }; tick(); }
  async function launch(form, endpoint) { try { $("#taskMessage").textContent = "正在提交…"; const data = formData(form); if (form.id === "trainForm") {
    data.resume = data.mode === "resume" ? data.parent : ""; data.init_checkpoint = data.mode === "finetune" ? data.parent : "";
    if (data.mode==='resume') {
      const inspection=await request('/api/composer/inspect',{checkpoint:data.parent});
      if (!inspection.can_resume) throw new Error(inspection.error||'checkpoint 不能 resume，请使用 finetune');
      for (const key of inheritedKeys) {
        const value=inspection.model_config?.[key]??inspection.args?.[key]??({lr_schedule:'constant',warmup_steps:0,min_lr_ratio:1,
          sample_sequence_policy:'independent_events_v1',configuration_quality_policy:'off',structure_policy:'off',structure_backend:'native',structure_dropout:.15,contact_policy:'off'}[key]);
        if (value!==undefined) { data[key]=value; const field=form.elements.namedItem(key); if(field) field.value=String(value); }
      }
      saveControls();
    }
    delete data.parent;
    if(data.structure_policy!=='off') {
      const status=await(await fetch('/api/status')).json();
      const capability=status.training_capabilities?.composer_structure_context;
      const enhanced=data.structure_policy==='metrical_context_v2'||(data.mode==='fresh'&&data.structure_policy==='auto');
      if(!(capability==='metrical_context_v2'||(!enhanced&&capability==='metrical_context_v1')))
        throw new Error('当前服务未加载音乐结构训练，请重启更新后的 WebUI 服务。');
    }
    if(data.lr_schedule==='cosine'||Number(data.warmup_steps||0)>0) {
      const status=await(await fetch('/api/status')).json();
      if(status.training_capabilities?.composer_learning_rate_schedule!=='optimizer_updates_v1')
        throw new Error('当前服务未加载学习率预热与退火，请在更新后的服务启动新阶段。');
    }
  } else if (form.id === "generateForm") { for (const k of ["bpm", "condition_file", "intensity_curve_file"]) if (!data[k]) delete data[k]; } const result = await request(endpoint, data); poll(result.task_id); } catch (error) { showError(error); } }
  $("#trainForm").addEventListener("submit", (e) => { e.preventDefault(); launch(e.currentTarget, "/api/composer/train"); }); $("#generateForm").addEventListener("submit", (e) => { e.preventDefault(); launch(e.currentTarget, "/api/composer/generate"); });
  $("#inspectForm").addEventListener("submit", async (e) => { e.preventDefault(); try { $("#inspection").textContent = JSON.stringify(await request("/api/composer/inspect", formData(e.currentTarget)), null, 2); } catch (error) { $("#inspection").textContent = error.message; } });
  $("#cancel").addEventListener("click", async () => { if (!active) return; try { await request("/api/cancel", {task_id: active}); } catch (error) { showError(error); } });
  fetch("/api/status").then((r) => r.json()).then((data) => { const task = (data.tasks || []).filter((x) => String(x.kind || "").startsWith("composer_")).at(-1); if (task) { render(task); if (["queued", "running"].includes(task.status)) poll(task.task_id); } }).catch(() => {});
})();
