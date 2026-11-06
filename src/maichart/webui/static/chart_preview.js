"use strict";

// The player only reads completed generation artifacts. It never uses the
// currently editable audio/tempo fields to reinterpret an older result.
(function () {
  const panel = document.getElementById("generationPreviewPanel");
  if (!panel) return;
  const mount = document.getElementById("generationPreviewPlayer");
  const status = document.getElementById("generationPreviewStatus");
  const source = document.getElementById("generationPreviewSource");
  const badge = document.getElementById("generationPreviewBadge");
  const selector = document.getElementById("generationPreviewChart");
  const installButton = document.getElementById("generationPreviewInstall");
  const reloadButton = document.getElementById("generationPreviewReload");
  const licenseLink = document.getElementById("generationPreviewLicense");
  const kinds = new Set(["generate", "form_generate", "composer_generate", "composer_v2_generate", "chart_stream_generate"]);
  let entry = null, player = null, playerKey = "", failedKey = "", selected = 0;
  let loadSerial = 0, requestSerial = 0, requestedTask = null, offeredTask = null;
  let installTimer = null;

  function message(text, error = false) {
    status.textContent = text;
    status.classList.toggle("warn", error);
    if (error) badge.textContent = "预览未就绪 · 展开查看";
  }
  async function json(url, options) {
    const response = await fetch(url, {cache: "no-store", ...options});
    if (!response.ok) {
      let detail;
      try { detail = await response.json(); } catch (_) {}
      throw new Error(detail?.error || (response.status === 404
        ? "当前后台尚未提供谱面预览，请更新并重启 WebUI。" : `预览请求失败（${response.status}）`));
    }
    return response.json();
  }
  function release() {
    ++loadSerial;
    if (player) player.destroy();
    player = null; playerKey = "";
    mount.replaceChildren();
    mount.hidden = true;
  }
  function pause() {
    if (player) player.pause().catch(() => {});
  }
  function visible() {
    return panel.open && !document.hidden && panel.getClientRects().length > 0;
  }
  async function activate(force = false) {
    if (!entry || !visible()) return;
    const chart = entry.charts[selected];
    if (!chart) return;
    const key = `${entry.task_id}:${entry.revision}:${selected}`;
    if (!force && (playerKey === key || failedKey === key)) return;
    if (force) failedKey = "";
    release();
    const serial = loadSerial;
    playerKey = key;
    badge.textContent = "正在载入预览…";
    message("正在准备谱面预览…");
    try {
      const runtime = await json("/api/preview/status");
      if (serial !== loadSerial) return;
      if (!runtime.available) {
        playerKey = "";
        installButton.hidden = false;
        installButton.disabled = !!runtime.installing;
        message(runtime.error || runtime.message || "首次使用需下载预览播放器（约 35 MB），之后可离线使用。", !!runtime.error);
        if (runtime.installing) pollInstall();
        return;
      }
      installButton.hidden = true;
      licenseLink.href = new URL("LICENSE_SCOPE.md", new URL(runtime.renderer_url, location.href)).href;
      licenseLink.hidden = false;
      const sdk = await import(runtime.sdk_url);
      if (serial !== loadSerial || !visible()) { if (serial === loadSerial) playerKey = ""; return; }
      mount.hidden = false;
      const current = sdk.createPreview(mount, {
        rendererUrl: runtime.renderer_url,
        resolution: {width: 1280, height: 720},
        maxFps: 30, theme: "light", diagnostics: false,
      });
      player = current;
      current.on("error", ({message: text}) => {
        if (serial === loadSerial) { message(`预览未能完成加载：${text}`, true); failedKey = key; }
      });
      current.on("progress", ({stage, progress}) => {
        if (serial === loadSerial) message(`正在载入谱面与音频… ${Math.round(Math.min(1, Math.max(0, progress || 0)) * 100)}%`);
      });
      await current.ready;
      if (serial !== loadSerial) return;
      const input = {maidata: {url: chart.url}};
      if (chart.audio_url) input.audio = chart.audio_url;
      const info = await current.load(input);
      if (serial !== loadSerial) return;
      message(`已载入 ${info.title || chart.title || chart.label} · ${info.noteCount ?? "—"} 个音符。点击播放器中的播放按钮开始。${chart.audio_url ? "" : "当前为静音预览。"}`);
      badge.textContent = "已载入 · 点击播放";
      reloadButton.disabled = false;
      if (!visible()) pause();
    } catch (error) {
      if (serial !== loadSerial) return;
      failedKey = key;
      release();
      message(`预览加载失败：${error.message}。谱面导出结果不受影响，可点击“重新载入”。`, true);
      reloadButton.disabled = false;
    }
  }
  function show(value) {
    if (!value?.charts?.length) return;
    if (entry?.task_id === value.task_id && entry?.revision === value.revision) return;
    release(); failedKey = ""; entry = value; selected = 0;
    selector.replaceChildren();
    value.charts.forEach((chart, index) => {
      const option = document.createElement("option");
      option.value = String(index); option.textContent = chart.label;
      selector.appendChild(option);
    });
    selector.disabled = value.charts.length < 2;
    source.textContent = `任务 ${value.task_id} · ${value.charts[0].title} · ${value.charts[0].audio_name ? `音频：${value.charts[0].audio_name}` : "无对应音频"}`;
    source.title = value.charts[0].path;
    reloadButton.disabled = false;
    badge.textContent = `${value.charts[0].title} · 展开预览`;
    badge.title = value.charts[0].title;
    message("谱面已就绪，展开即可载入；播放器载入后保持暂停。");
    void activate();
  }
  function supportsTask(task) { return task?.status === "completed" && kinds.has(task.kind); }
  async function offerTask(task, {force = false} = {}) {
    if (!supportsTask(task)) return;
    if (!force && (offeredTask === task.task_id || requestedTask === task.task_id)) return;
    const serial = ++requestSerial;
    requestedTask = task.task_id;
    try {
      const value = await json(`/api/preview/task?id=${encodeURIComponent(task.task_id)}`);
      if (serial !== requestSerial) return;
      offeredTask = task.task_id;
      show(value);
    } catch (error) {
      if (serial === requestSerial) { offeredTask = task.task_id; message(error.message, true); }
    } finally {
      if (serial === requestSerial) requestedTask = null;
    }
  }
  async function openTask(task) {
    document.querySelector('[data-mode="generation"]')?.click();
    await offerTask(task, {force: true});
    panel.open = true;
    panel.scrollIntoView({behavior: "smooth", block: "start"});
    void activate();
  }
  function pollInstall() {
    clearTimeout(installTimer);
    installTimer = setTimeout(async () => {
      try {
        const runtime = await json("/api/preview/status");
        message(runtime.error || runtime.message || "正在准备播放器…", !!runtime.error);
        if (runtime.available) { installButton.hidden = true; void activate(true); }
        else if (runtime.installing) pollInstall();
        else { installButton.disabled = false; installButton.hidden = false; }
      } catch (error) { installButton.disabled = false; message(error.message, true); }
    }, 1000);
  }
  selector.addEventListener("change", () => {
    selected = Number(selector.value);
    const chart = entry?.charts[selected];
    if (chart) { source.textContent = `任务 ${entry.task_id} · ${chart.title} · ${chart.audio_name || "静音预览"}`; source.title = chart.path; }
    void activate(true);
  });
  reloadButton.addEventListener("click", () => {
    if (entry) void offerTask({task_id: entry.task_id, kind: "generate", status: "completed"}, {force: true}).then(() => activate(true));
  });
  installButton.addEventListener("click", async () => {
    installButton.disabled = true;
    try { await json("/api/preview/install", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"}); pollInstall(); }
    catch (error) { installButton.disabled = false; message(error.message, true); }
  });
  panel.addEventListener("toggle", () => {
    if (panel.open) void activate();
    else { release(); if (entry) badge.textContent = `${entry.charts[selected].title} · 展开预览`; message(entry ? "预览已收起并释放资源；展开后重新载入。" : "生成完成后会自动更新结果。"); }
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  const observer = new IntersectionObserver(items => {
    if (items.some(item => item.isIntersecting)) void activate();
    else pause();
  });
  observer.observe(panel);
  window.addEventListener("pagehide", () => { clearTimeout(installTimer); observer.disconnect(); release(); }, {once: true});
  globalThis.ChartPreview = {offerTask, openTask, supportsTask,
    getState: () => player ? player.getState() : Promise.resolve(null)};
  // The latest completed artifact survives a WebUI restart via task log metadata.
  const initialSerial = requestSerial;
  json("/api/preview/latest").then(value => {
    if (requestSerial === initialSerial && !entry && value.preview) show(value.preview);
  }).catch(() => {});
})();
