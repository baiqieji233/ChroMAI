"use strict";

// A preview can be changed freely. Only an explicitly applied clock reaches generation.
const ComposerReferenceTempo = (() => {
  const byId = id => document.getElementById(id);
  const audio = () => byId("audioPath").value.trim();
  const slotNames = ["", "EASY", "BASIC", "ADVANCED", "EXPERT", "MASTER", "Re:MASTER"];
  const storageKey = "maichart.composer-v2.reference-clock.v1";
  const pageSize = 64;
  let draft = null, active = null, sequence = 0, page = 0, previousAudio = audio(), draftAudio = "";
  const file = byId("v2ReferenceFile"), selector = byId("v2ReferenceSlot");
  const selected = () => draft?.slots[Number(selector.value)] ?? null;
  const current = () => active && active.audio === audio() ? active : null;
  const slotName = slot => slot.index == null ? "BPM 表" : slotNames[slot.index] + (slot.level ? " " + slot.level : "");
  const label = slot => [draft.title || draft.filename, slotName(slot)].join(" · ");

  function timestamp(seconds) {
    const ms = Math.round(Math.abs(seconds) * 1000);
    return (seconds < 0 ? "−" : "") + Math.floor(ms / 60000) + ":" +
      String(Math.floor(ms / 1000) % 60).padStart(2, "0") + "." + String(ms % 1000).padStart(3, "0");
  }

  function showSlot() {
    const slot = selected();
    byId("v2ReferencePreview").hidden = !slot;
    const rows = slot?.rows || [];
    const firstSource = {default:"谱面省略，按 0 秒", json:"BPM 表", auto:"生成时自动对准"};
    if (slot) {
      const source = firstSource[slot.first_source] || "&" + slot.first_source;
      const offset = slot.first == null ? source : slot.first + " 秒（" + source + "）";
      byId("v2ReferenceInfo").textContent = slot.error || label(slot) + " · 首拍 " + offset + " · " + rows.length + " 段速度";
      byId("v2ReferenceTimeHeading").textContent = slot.first == null ? "距第 0 拍" : "音频位置";
    }
    const matches = rows.length && slot.index != null ? draft.slots.filter(other =>
      other !== slot && other.first === slot.first && other.rows?.length === rows.length &&
      other.rows.every((row, i) => row.beat === rows[i].beat && row.bpm === rows[i].bpm)) : [];
    const matchNote = byId("v2ReferenceMatching");
    matchNote.hidden = !matches.length;
    matchNote.textContent = matches.length ? "相同 BPM 表和首拍的难度还有 " + matches.map(s => slotNames[s.index]).join("、") + "。" : "";
    byId("v2ReferenceTable").hidden = !rows.length;
    const body = byId("v2ReferenceRows");
    const start = page * pageSize, end = Math.min(start + pageSize, rows.length);
    const fragment = document.createDocumentFragment();
    for (const item of rows.slice(start, end)) {
      const tr = document.createElement("tr");
      for (const value of [item.beat, timestamp(item.seconds), item.bpm]) {
        tr.insertCell().textContent = value;
      }
      fragment.append(tr);
    }
    body.replaceChildren(fragment);
    byId("v2ReferenceTable").scrollTop = 0;
    byId("v2ReferencePages").hidden = rows.length <= pageSize;
    byId("v2ReferencePrev").disabled = !page;
    byId("v2ReferenceNext").disabled = end >= rows.length;
    byId("v2ReferenceRange").textContent = (start + 1) + "–" + end + " / " + rows.length;
    byId("v2ReferenceExport").disabled = !rows.length;
    updateControls();
  }

  function updateControls(v2 = currentGenerationFamily() === "chart_composer_v2") {
    const used = current();
    const applied = Boolean(used && v2);
    byId("v2AutomaticClockFields").hidden = applied;
    byId("generationTimingGroup").classList.toggle("reference-active", applied);
    byId("v2ReferenceJump").textContent = used ? "查看已应用的参考 BPM 表" : "从 maidata 导入 BPM 表";
    byId("v2ReferenceBadge").textContent = used ? "已应用 · " + used.label : "可选";
    byId("v2ReferenceActive").hidden = !used;
    if (used) byId("v2ReferenceActive").textContent = "当前生成使用 " + used.label +
      " · " + used.tempo_map.length + " 段速度 · " + (used.first == null ? "自动首拍" : "首拍 " + used.first + " 秒");
    byId("v2ReferenceClear").hidden = !used;
    const isApplied = used && used.slot === selected();
    const button = byId("v2ReferenceApply");
    button.disabled = !audio() || !selected()?.rows?.length || Boolean(isApplied);
    button.textContent = isApplied ? "已用于当前音频" : "将此表用于当前音频";
  }

  function setDraft(data, slotIndex) {
    draft = data;
    selector.replaceChildren();
    if (draft) {
      draft.slots.forEach((slot, index) => {
        selector.add(new Option(slotName(slot) + (slot.error ? "（无法使用）" : ""), String(index)));
      });
      selector.value = String(slotIndex);
      byId("v2ReferenceSlotField").hidden = draft.slots.length === 1 && draft.slots[0].index == null;
    }
    page = 0;
    showSlot();
  }

  function remember() {
    try {
      if (active) {
        sessionStorage.setItem(storageKey, JSON.stringify({audio: active.audio, draft, selection: Number(selector.value)}));
      } else sessionStorage.removeItem(storageKey);
    } catch (_) {
      // A quota failure must not restore a previously applied clock on reload.
      try { sessionStorage.removeItem(storageKey); } catch (_) { /* Storage is unavailable. */ }
      if (active) byId("v2ReferenceState").textContent = "已用于本次生成。浏览器未能保存这张表，刷新后请重新导入。";
    }
  }

  function changed() {
    updateControls();
    if (typeof v2CurveInvalidateTiming === "function") v2CurveInvalidateTiming();
    window.MaichartWorkflow?.refresh();
  }

  function activate() {
    const slot = selected();
    if (!audio() || !slot?.rows?.length) return;
    active = {audio: audio(), slot, label: label(slot), first: slot.first,
      tempo_map: slot.rows.map(({beat, bpm}) => ({beat, bpm}))};
    draftAudio = active.audio;
    byId("v2ReferenceState").textContent = "";
    remember();
    changed();
  }

  function audioChanged() {
    const next = audio();
    if (next === previousAudio) return;
    previousAudio = next;
    if ((active && active.audio !== next) || (draftAudio && draftAudio !== next)) {
      ++sequence;
      const hadReference = Boolean(active || draft || file.files.length);
      active = null;
      draftAudio = "";
      remember();
      file.value = "";
      setDraft(null);
      byId("v2ReferenceState").textContent = hadReference ? "音频已更换，请重新选择与这首歌对应的参考文件。" : "";
      changed();
    } else {
      if (draft && next) byId("v2ReferenceState").textContent = "核对后点击「将此表用于当前音频」。";
      updateControls();
    }
  }

  function readFile(input) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = () => reject(new Error("本地文件读取失败"));
      reader.readAsDataURL(input);
    });
  }

  file.addEventListener("change", async () => {
    const id = ++sequence, input = file.files[0];
    draftAudio = input ? audio() : "";
    setDraft(null);
    const status = byId("v2ReferenceState");
    if (!input) { status.textContent = ""; return; }
    try {
      if (input.size > 4 * 1024 * 1024) throw new Error("请选择不超过 4 MiB 的文件");
      status.textContent = "正在读取变速点…";
      const data = await api("/api/composer-v2/reference_tempo", {content: await readFile(input)});
      if (id !== sequence) return;
      data.filename = input.name;
      const preferred = data.slots.findIndex(s => s.index === 5 && !s.error);
      setDraft(data, preferred >= 0 ? preferred : Math.max(0, data.slots.findIndex(s => !s.error)));
      status.textContent = audio() ? "正在预览。核对后点击「将此表用于当前音频」。" : "正在预览。选好生成音频后即可应用。";
    } catch (error) {
      if (id === sequence) status.textContent = "读取失败：" + error.message;
    }
  });
  selector.addEventListener("change", () => { page = 0; showSlot(); });
  byId("v2ReferencePrev").addEventListener("click", () => { --page; showSlot(); });
  byId("v2ReferenceNext").addEventListener("click", () => { ++page; showSlot(); });
  byId("v2ReferenceApply").addEventListener("click", activate);
  byId("v2ReferenceClear").addEventListener("click", () => {
    active = null; remember(); changed();
    byId("v2ReferenceState").textContent = "已恢复原节拍设置。";
  });
  byId("v2ReferenceExport").addEventListener("click", () => {
    const slot = selected();
    if (!slot?.rows?.length) return;
    const content = {first: slot.first, tempo_map: slot.rows.map(({beat, bpm}) => ({beat, bpm}))};
    const url = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], {type:"application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = (draft.title || draft.filename).replace(/[<>:"/\\|?*]/g, "_") + (slot.index ? "-slot" + slot.index : "") + ".bpm.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  byId("audioPath").addEventListener("change", audioChanged);
  file.disabled = false;
  updateControls();
  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey));
    if (saved?.audio && saved.audio === audio() && saved.draft?.slots?.[saved.selection]?.rows?.length) {
      setDraft(saved.draft, saved.selection);
      activate();
    } else sessionStorage.removeItem(storageKey);
  } catch (_) { /* Unavailable or outdated session data. */ }
  return {current, updateControls, audioChanged};
})();
