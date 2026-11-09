"use strict";

// Review existing training charts and complete outputs with their own audio.
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
  const downloadLink = document.getElementById("generationPreviewDownload");
  const expandButton = document.getElementById('generationPreviewExpand');
  const collapseButton = document.getElementById('generationPreviewCollapse');
  const resizeHandle = document.getElementById('generationReviewResize');
  const toolsToggle = document.getElementById('generationReviewToolsToggle');
  const sourceSwitch = document.getElementById('generationPreviewSwitchSource');
  const previewTitle = document.getElementById('generationPreviewTitle');
  const previewBody = panel.querySelector('.generation-preview-body');
  const trainingEmpty = document.getElementById('trainingPreviewEmpty');
  let generationEntry=null,trainingEntry=null;
  const kinds = new Set(["generate", "form_generate", "composer_generate", "composer_v2_generate", "chart_stream_generate"]);
  let entry = null, player = null, playerKey = "", failedKey = "", selected = 0;
  let pendingSeek = null;
  let playerLoaded = false, lastTransport = null;
  let pendingPlay = false, activeRange = null, watchTimer = null;
  let activation = null, activationKey = '';
  let selectionRequest = 0;
  let transportRequest = 0, loopReturning = false;
  let savedOverflow = null, focusReturn = null;
  let loadSerial = 0, requestSerial = 0, requestedTask = null, offeredTask = null, taskRequestSerial=0;
  let installTimer = null;

  function readableLabel(chart) {
    return chart.label || chart.title || '完整谱面';
  }
  function setExpanded(expanded) {
    if(expanded===panel.classList.contains('review-expanded'))return;
    panel.classList.toggle('review-expanded',expanded);
    if(expandButton)expandButton.hidden=expanded;
    if(collapseButton)collapseButton.hidden=!expanded;
    if(resizeHandle)resizeHandle.hidden=!expanded;
    if(toolsToggle)toolsToggle.hidden=!expanded;
    if(expanded) {
      savedOverflow=document.body.style.overflow;focusReturn=document.activeElement;
      document.body.style.overflow='hidden';panel.setAttribute('role','dialog');
      panel.setAttribute('aria-modal','true');panel.setAttribute('aria-label','谱面审阅');
      panel.open=true;collapseButton?.focus();void activate();
    } else {
      document.body.style.overflow=savedOverflow??'';savedOverflow=null;
      panel.removeAttribute('role');panel.removeAttribute('aria-modal');panel.removeAttribute('aria-label');
      if(focusReturn?.isConnected)focusReturn.focus();focusReturn=null;
    }
  }
  expandButton?.addEventListener('click',()=>setExpanded(true));
  collapseButton?.addEventListener('click',()=>setExpanded(false));
  toolsToggle?.addEventListener('click',()=>{
    const hidden=panel.classList.toggle('review-tools-hidden');
    toolsToggle.textContent=hidden?'显示工具栏':'收起工具栏';
    toolsToggle.setAttribute('aria-expanded',String(!hidden));
  });
  function previewHeight(height) {
    const maximum=Math.max(220,innerHeight*.68),value=Math.max(220,Math.min(maximum,height));
    panel.style.setProperty('--review-preview-height',value+'px');
    resizeHandle?.setAttribute('aria-valuenow',String(Math.round(value)));
  }
  let resizeDrag=null;
  resizeHandle?.addEventListener('pointerdown',event=>{
    event.preventDefault();resizeHandle.setPointerCapture(event.pointerId);
    resizeDrag={y:event.clientY,height:mount.getBoundingClientRect().height};
  });
  resizeHandle?.addEventListener('pointermove',event=>{
    if(resizeDrag)previewHeight(resizeDrag.height+event.clientY-resizeDrag.y);
  });
  resizeHandle?.addEventListener('pointerup',event=>{
    if(!resizeDrag)return;previewHeight(resizeDrag.height+event.clientY-resizeDrag.y);
    resizeDrag=null;resizeHandle.releasePointerCapture(event.pointerId);
  });
  resizeHandle?.addEventListener('pointercancel',()=>{resizeDrag=null;});
  resizeHandle?.addEventListener('dblclick',()=>panel.style.removeProperty('--review-preview-height'));
  resizeHandle?.addEventListener('keydown',event=>{
    if(['ArrowUp','ArrowDown','Home'].includes(event.key)) {
      event.preventDefault();
      if(event.key==='Home')panel.style.removeProperty('--review-preview-height');
      else previewHeight(mount.getBoundingClientRect().height+(event.key==='ArrowDown'?32:-32));
    }
  });
  document.addEventListener('keydown',event=>{
    if(!panel.classList.contains('review-expanded'))return;
    if(event.key==='Escape') {event.preventDefault();setExpanded(false);return;}
    if(event.key==='Tab') {
      const items=[...panel.querySelectorAll('button,input,select,a[href],iframe,[tabindex="0"]')]
        .filter(item=>!item.disabled && item.getClientRects().length);
      const first=items[0],last=items[items.length-1];
      if(event.shiftKey && document.activeElement===first) {event.preventDefault();last?.focus();}
      else if(!event.shiftKey && document.activeElement===last) {event.preventDefault();first?.focus();}
    }
  });

  const sameAudio = (first, second) => {
    const a=first?.audio_sha256 || first?.audio_path || first?.audio_url;
    const b=second?.audio_sha256 || second?.audio_path || second?.audio_url;
    return !!a && a===b;
  };
  function clearRange() { activeRange=null; }
  function stopWatch() { clearInterval(watchTimer);watchTimer=null; }
  async function repeatRange(current, serial) {
    const bounds=activeRange;
    if(loopReturning || !bounds || current!==player || serial!==loadSerial || !visible())return;
    loopReturning=true;
    try {
      await current.seek(bounds.startSec);
      if(current===player && serial===loadSerial && bounds===activeRange && visible())await current.play();
    } finally {loopReturning=false;}
  }
  function watchPlayback() {
    stopWatch();
    const current=player,serial=loadSerial;
    if(!current)return;
    let polling=false;
    watchTimer=setInterval(async()=>{
      if(polling || !visible() || current!==player)return;
      polling=true;
      try {
        const transport=transportRequest;
        const state=await current.getState();
        if(serial!==loadSerial || current!==player || transport!==transportRequest)return;
        lastTransport=state;
        window.dispatchEvent(new CustomEvent('chart-preview-playback',{detail:state}));
        if(activeRange && state.playing && state.currentTime>=activeRange.endSec) {
          await repeatRange(current,serial);
        }
        if(!state.playing)stopWatch();
      } catch (_) { if(current===player)stopWatch(); }
      finally { polling=false; }
    },250);
  }
  function updateSource(sourceKind='main') {
    const chart=entry?.charts[selected];if(!chart)return;
    source.textContent=`${chart.title} · ${readableLabel(chart)} · ${chart.audio_name || '无对应音频'}`;
    source.title=chart.path;
    if(downloadLink) {downloadLink.href=chart.url;downloadLink.download='maidata.txt';downloadLink.hidden=false;}
    window.dispatchEvent(new CustomEvent('chart-preview-versionchanged', {detail:{index:selected,chart,source:sourceKind}}));
  }

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
    stopWatch();
    if (player) player.destroy();
    player = null; playerLoaded=false;playerKey = "";
    mount.replaceChildren();
    mount.hidden = true;
  }
  async function pause() {
    ++transportRequest;clearRange();stopWatch();pendingPlay=false;
    if(lastTransport)lastTransport={...lastTransport,playing:false};
    if (player) await player.pause().catch(() => {});
  }
  function visible() {
    return panel.open && !document.hidden && panel.getClientRects().length > 0;
  }
  function activate(force = false) {
    if(!entry || !visible())return Promise.resolve();
    const key=`${entry.task_id}:${entry.revision}:${selected}`;
    if(!force && activation && activationKey===key)return activation;
    const pending=loadPreview(force);activation=pending;activationKey=key;
    return pending.finally(()=>{if(activation===pending)activation=null;});
  }
  async function loadPreview(force = false) {
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
        // No extra cap: requestAnimationFrame follows the active display.
        maxFps: "unlimited", theme: "light", diagnostics: false,
      });
      player = current;
      current.on('playing',()=>{if(serial===loadSerial)watchPlayback();});
      current.on('paused',()=>{if(serial===loadSerial)stopWatch();});
      current.on('ended',()=>{void repeatRange(current,serial).catch(error=>message(error.message,true));});
      current.on('seeked',()=>{
        const transport=transportRequest;
        if(serial===loadSerial)void current.getState().then(state=>{
          if(serial===loadSerial && playerLoaded && transport===transportRequest) {lastTransport=state;window.dispatchEvent(new CustomEvent('chart-preview-playback',{detail:state}));}
        }).catch(()=>{});
      });
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
      playerLoaded=true;
      if(Number.isFinite(pendingSeek)) {const second=pendingSeek;pendingSeek=null;await current.seek(second);if(serial!==loadSerial)return;}
      if(pendingPlay && visible()) {pendingPlay=false;await current.play();if(serial!==loadSerial)return;watchPlayback();}
      const state=await current.getState();
      if(serial!==loadSerial)return;
      lastTransport=state;window.dispatchEvent(new CustomEvent('chart-preview-playback',{detail:state}));
      message(`已载入 ${info.title || chart.title || readableLabel(chart)} · ${info.noteCount ?? "—"} 个音符。点击播放器中的播放按钮开始。${chart.audio_url ? "" : "当前为静音预览。"}`);
      badge.textContent = "已载入 · 点击播放";
      reloadButton.disabled = false;
      if (!visible()) pause();
    } catch (error) {
      if (serial !== loadSerial) return;
      failedKey = key;
      release();
      message(`预览加载失败：${error.message}。点击“重新载入”重试。`, true);
      reloadButton.disabled = false;
    }
  }
  function show(value) {
    if (!value?.charts?.length) return;
    // Older backends also exposed internal trial files. Never treat them as versions.
    value={...value,charts:value.charts.filter(chart=>!chart.partial_preview && chart.complete!==false
      && (!chart.content_kind || chart.content_kind==='complete_chart')
      && !/[\\/]phrase_candidates[\\/]/.test(chart.path||''))};
    if(!value.charts.length) {message('这个任务没有可预览的完整谱面。');return;}
    if(value.source_kind==='training_source')trainingEntry=value;else generationEntry=value;
    previewBody.hidden=false;trainingEmpty.hidden=true;
    previewTitle.textContent=value.source_kind==='training_source'?'训练集谱面预览':'生成谱面预览';
    document.getElementById('generationPreviewChoice').hidden=value.source_kind==='training_source';
    document.getElementById('trainingPreviewNavigation').hidden=value.source_kind!=='training_source';
    if(sourceSwitch) {
      sourceSwitch.hidden=value.source_kind!=='training_source' && !trainingEntry;
      sourceSwitch.textContent=value.source_kind==='training_source'?'查看生成结果':'查看训练素材';
    }
    if (entry?.task_id === value.task_id && entry?.revision === value.revision) return;
    const previousPath=entry?.task_id===value.task_id?entry.charts[selected]?.path:null;
    ++selectionRequest;release(); failedKey = "";clearRange();pendingSeek=null;pendingPlay=false;lastTransport=null;entry = value;
    selected=Math.max(0,value.charts.findIndex(chart=>chart.path===previousPath));
    selector.replaceChildren();
    value.charts.forEach((chart, index) => {
      const option = document.createElement("option");
      option.value = String(index); option.textContent = readableLabel(chart);
      selector.appendChild(option);
    });
    selector.value=String(selected);
    selector.disabled = value.charts.length < 2;
    const chart=value.charts[selected];
    source.textContent = `${chart.title} · ${readableLabel(chart)} · ${chart.audio_name || '无对应音频'}`;
    source.title = chart.path;
    if(downloadLink) {downloadLink.href=chart.url;downloadLink.download='maidata.txt';downloadLink.hidden=false;}
    reloadButton.disabled = false;
    badge.textContent = `${chart.title} · ${value.charts.length} 张完整谱面`;
    badge.title = chart.title;
    message("谱面已就绪，展开即可载入；播放器载入后保持暂停。");
    window.dispatchEvent(new CustomEvent("chart-preview-updated", {detail:value}));
    void activate();
  }
  function supportsTask(task) { return task?.status === "completed" && kinds.has(task.kind); }
  async function offerTask(task, {force = false} = {}) {
    if (!supportsTask(task)) return;
    if(entry?.source_kind==='training_source' && !force)return;
    if (!force && (offeredTask === task.task_id || requestedTask === task.task_id)) return;
    const serial = ++requestSerial;
    requestedTask = task.task_id;taskRequestSerial=serial;
    try {
      const value = await json(`/api/preview/task?id=${encodeURIComponent(task.task_id)}`);
      if (serial !== requestSerial) return;
      offeredTask = task.task_id;
      generationEntry=value;
      if(force || document.body.dataset.workspace!=='quality')show(value);
    } catch (error) {
      if (serial === requestSerial) { offeredTask = task.task_id; message(error.message, true); }
    } finally {
      if (serial === taskRequestSerial) requestedTask = null;
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
  async function chooseChart(index,{startSec=null,restart=false,source:sourceKind='main'}={}) {
    if(!entry?.charts[index])return;
    const request=++selectionRequest,transport=transportRequest,previous=entry.charts[selected],currentPlayer=player;
    const state=currentPlayer && playerLoaded?await currentPlayer.getState().catch(()=>lastTransport):lastTransport;
    if(request!==selectionRequest)return;
    const compatible=sameAudio(previous,entry.charts[index]);
    if(state)lastTransport=state;
    if(!compatible)clearRange();
    const time=compatible && !restart && Number.isFinite(state?.currentTime)?state.currentTime:startSec;
    pendingSeek=Number.isFinite(time)?Math.max(0,time):null;
    pendingPlay=compatible && !!state?.playing && transport===transportRequest;
    const changed=selected!==index;selected=index;selector.value=String(index);panel.open=true;
    updateSource(sourceKind);
    await activate(changed);
    if(!changed && player) {
      if(Number.isFinite(pendingSeek))await player.seek(pendingSeek);
      pendingSeek=null;pendingPlay=false;
    }
  }
  async function playRange({startSec,endSec}) {
    const intent=++transportRequest;
    if(!Number.isFinite(startSec) || !Number.isFinite(endSec) || startSec<0 || endSec<=startSec)
      throw new Error('请先选择有效的试听范围。');
    panel.open=true;await activate();
    if(intent!==transportRequest)return;
    if(!player)throw new Error('请先载入谱面预览。');
    const state=await player.getState(),duration=Number(state.duration);
    if(intent!==transportRequest)return;
    const stop=Number.isFinite(duration) && duration>0?Math.min(endSec,duration):endSec;
    if(stop<=startSec)throw new Error('所选范围超出了这张谱面的播放时长。');
    activeRange={startSec,endSec:stop};await player.seek(startSec);await player.play();watchPlayback();
  }
  async function seek(second) {
    if(!Number.isFinite(second) || second<0)throw new Error('请选择有效的谱面位置。');
    const selection=selectionRequest,intent=++transportRequest;
    clearRange();pendingSeek=second;panel.open=true;await activate();
    if(selection!==selectionRequest || intent!==transportRequest)return;
    if(!player)throw new Error('请先载入谱面预览。');
    pendingSeek=null;await player.seek(second);
  }
  selector.addEventListener("change", () => {void chooseChart(Number(selector.value)).catch(error=>message(error.message,true));});
  reloadButton.addEventListener("click", () => {
    if(entry?.source_kind==='training_source') {
      const request=++requestSerial;
      void json('/api/quality/chart-preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(entry.training_context)}).then(value=>{if(request===requestSerial){show(value);void activate(true);}}).catch(error=>{if(request===requestSerial)message(error.message,true);});return;
    }
    if (entry) void offerTask({task_id: entry.task_id, kind: "generate", status: "completed"}, {force: true}).then(() => activate(true));
  });
  sourceSwitch?.addEventListener('click',async()=>{
    try {
      const value=entry?.source_kind==='training_source'?(generationEntry||(await json('/api/preview/latest')).preview):trainingEntry;
      if(value)show(value);else message('选择训练素材或完成一次生成后，可在这里切换。');
    } catch(error){message(error.message,true);}
  });
  installButton.addEventListener("click", async () => {
    installButton.disabled = true;
    try { await json("/api/preview/install", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"}); pollInstall(); }
    catch (error) { installButton.disabled = false; message(error.message, true); }
  });
  panel.addEventListener("toggle", () => {
    if (panel.open) void activate();
    else { setExpanded(false);clearRange();pendingPlay=false;release(); if (entry) badge.textContent = `${entry.charts[selected].title} · 展开预览`; message(entry ? "预览已收起并释放资源；展开后重新载入。" : "生成完成后会自动更新结果。"); }
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  const observer = new IntersectionObserver(items => {
    if (items.some(item => item.isIntersecting)) void activate();
    else pause();
  });
  observer.observe(panel);
  function workspacePreview(mode) {
    const cached=mode==='quality'?trainingEntry:mode==='generation'?generationEntry:null;
    if(mode!=='quality' && mode!=='generation')return;
    ++requestSerial;
    if(cached) {show(cached);if(mode==='quality')panel.open=true;return;}
    ++selectionRequest;release();entry=null;selected=0;
    pendingSeek=null;pendingPlay=false;lastTransport=null;clearRange();
    setExpanded(false);previewBody.hidden=true;
    trainingEmpty.hidden=mode!=='quality';
    previewTitle.textContent=mode==='quality'?'训练集谱面预览':'生成谱面预览';
    badge.textContent=mode==='quality'?'打开训练集，选择一张谱面':'生成后自动更新';
    panel.open=mode==='quality';
  }
  window.addEventListener('workspace-mode-changed',event=>workspacePreview(event.detail.mode));
  window.addEventListener("pagehide", () => { clearTimeout(installTimer); observer.disconnect(); release(); }, {once: true});
  globalThis.ChartPreview = {offerTask, openTask, supportsTask,chooseChart,playRange,pause,clearRange,seek,
    openSource(value,{scroll=true}={}) {++requestSerial;show(value);panel.open=true;if(scroll)panel.scrollIntoView({behavior:'smooth',block:'start'});void activate();},
    setTrainingDataset(manifest) {
      if(trainingEntry?.training_context?.manifest===manifest)return;
      trainingEntry=null;
      if(document.body.dataset.workspace==='quality')workspacePreview('quality');
    },
    getEntry: () => entry,
    selectChart: chooseChart,
    getState: () => player && playerLoaded ? player.getState() : Promise.resolve(lastTransport)};
  // The latest completed artifact survives a WebUI restart via task log metadata.
  workspacePreview(document.body.dataset.workspace);
  json("/api/preview/latest").then(value => {
    if (!value.preview)return;
    if(!generationEntry)generationEntry=value.preview;
    if (!entry && !requestedTask && document.body.dataset.workspace!=='quality')show(generationEntry);
  }).catch(() => {});
})();
