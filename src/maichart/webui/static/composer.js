(() => {
  const $ = (s) => document.querySelector(s);
  const formData = (form) => Object.fromEntries([...form.elements].filter((x) => x.name).map((x) => [x.name, x.type === "checkbox" ? x.checked : x.value]));
  const request = async (url, body) => { const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); const data = await r.json(); if (!r.ok || data.error) throw new Error(data.error || `HTTP ${r.status}`); return data; };
  let active = "", timer = 0;
  function showError(error) { $("#taskMessage").textContent = error.message || String(error); $("#taskMessage").className = "bad"; }
  function render(task) {
    $("#taskKind").textContent = `${task.kind || "composer"} · ${task.task_id || ""}`;
    $("#taskState").textContent = ({queued: "排队中", running: "运行中", completed: "完成", failed: "失败", cancelled: "已中断"})[task.status] || task.status || "未知";
    $("#taskMessage").textContent = task.error || task.message || ""; $("#taskMessage").className = task.error ? "bad" : "";
    const p = Number(task.progress); $("#progress").style.width = `${Math.max(0, Math.min(1, Number.isFinite(p) ? p : 0)) * 100}%`;
    const result = task.result || {}; $("#progressText").textContent = [result.epoch, result.batch, result.loss, result.events_per_sec].filter((x) => x !== undefined && x !== null).join(" · ");
    $("#logs").textContent = Array.isArray(task.logs) ? task.logs.join("\n") : (task.logs || ""); $("#logs").scrollTop = $("#logs").scrollHeight;
    const paths = []; for (const key of ["output", "checkpoint_dir", "log_path"]) if (result[key]) paths.push([key, result[key]]); if (task.kind === "composer_generate" && result.output) paths.push(["composer.json", result.output.replace(/\.[^.]+$/, ".composer.json")]);
    $("#artifacts").replaceChildren(...(paths.length ? paths.map(([label, path]) => { const line = document.createElement("div"); const a = document.createElement("a"); a.href = `/api/download?path=${encodeURIComponent(path)}`; a.target = "_blank"; a.textContent = `${label}: ${path}`; line.append(a); return line; }) : [document.createTextNode("完成后显示")]));
    $("#cancel").disabled = !["queued", "running"].includes(task.status);
  }
  async function poll(id) { active = id; clearTimeout(timer); const tick = async () => { try { const task = await (await fetch(`/api/task?id=${encodeURIComponent(id)}`)).json(); render(task); if (!["completed", "failed", "cancelled"].includes(task.status)) timer = setTimeout(tick, 1000); } catch (error) { showError(error); timer = setTimeout(tick, 2000); } }; tick(); }
  async function launch(form, endpoint) { try { $("#taskMessage").textContent = "正在提交…"; const data = formData(form); if (form.id === "trainForm") { data.resume = data.mode === "resume" ? data.parent : ""; data.init_checkpoint = data.mode === "finetune" ? data.parent : ""; delete data.parent; } else if (form.id === "generateForm") { for (const k of ["bpm", "condition_file"]) if (!data[k]) delete data[k]; } const result = await request(endpoint, data); poll(result.task_id); } catch (error) { showError(error); } }
  $("#trainForm").addEventListener("submit", (e) => { e.preventDefault(); launch(e.currentTarget, "/api/composer/train"); }); $("#generateForm").addEventListener("submit", (e) => { e.preventDefault(); launch(e.currentTarget, "/api/composer/generate"); });
  $("#inspectForm").addEventListener("submit", async (e) => { e.preventDefault(); try { $("#inspection").textContent = JSON.stringify(await request("/api/composer/inspect", formData(e.currentTarget)), null, 2); } catch (error) { $("#inspection").textContent = error.message; } });
  $("#cancel").addEventListener("click", async () => { if (!active) return; try { await request("/api/cancel", {task_id: active}); } catch (error) { showError(error); } });
  fetch("/api/status").then((r) => r.json()).then((data) => { const task = (data.tasks || []).filter((x) => String(x.kind || "").startsWith("composer_")).at(-1); if (task) { render(task); if (["queued", "running"].includes(task.status)) poll(task.task_id); } }).catch(() => {});
})();
