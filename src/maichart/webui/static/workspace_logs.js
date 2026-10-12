(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.MaichartLogs = api;
})(typeof window === "undefined" ? null : window, function () {
  "use strict";

  const KEY_EVENT = /\b(epoch|loss|best|checkpoint|saved|saving|validation|generation|sample|complete|finished|failed|resume)\b|playability_|action_state_|audio_bounds|prewarm|event_audio_loads|已完成|保存|验证|生成|采样|失败|轮次|预热/i;

  function severityOf(line) {
    if (/\b(error|exception|traceback|failed|fatal)\b|\b\w*(?:Error|Exception):|错误|失败|异常/i.test(line)) return "error";
    if (/\b(warn(?:ing)?)\b|警告/i.test(line)) return "warning";
    if (/\b(debug|trace)\b/i.test(line)) return "debug";
    return "info";
  }

  function isKeyEvent(line) {
    return ["error", "warning"].includes(severityOf(line)) || KEY_EVENT.test(line);
  }

  function filterLines(lines, options = {}) {
    const query = String(options.query || "").trim().toLocaleLowerCase();
    const severity = options.severity || "all";
    return lines.filter((line) => {
      if (severity === "key" && !isKeyEvent(line)) return false;
      if (severity !== "all" && severity !== "key" && severityOf(line) !== severity) return false;
      return !query || line.toLocaleLowerCase().includes(query);
    });
  }

  function downloadUrl(path) {
    return path ? `/api/download?path=${encodeURIComponent(String(path))}` : "";
  }

  function installControls(logs) {
    const panel = logs.closest(".task-panel");
    if (!panel) return null;
    let toolbar = panel.querySelector(".workspace-log-toolbar");
    if (toolbar) return toolbar;

    toolbar = document.createElement("div");
    toolbar.className = "workspace-log-toolbar";

    const filterLabel = document.createElement("label");
    filterLabel.className = "workspace-log-filter";
    filterLabel.append("显示 ");
    const severity = document.createElement("select");
    severity.setAttribute("aria-label", "按日志类型筛选");
    [["all", "全部日志"], ["error", "错误"], ["warning", "警告"], ["key", "关键事件"], ["info", "普通信息"], ["debug", "调试"]]
      .forEach(([value, label]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        severity.appendChild(option);
      });
    filterLabel.appendChild(severity);

    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "搜索日志";
    search.setAttribute("aria-label", "搜索日志");

    const followLabel = document.createElement("label");
    followLabel.className = "workspace-log-toggle";
    const follow = document.createElement("input");
    follow.type = "checkbox";
    follow.checked = true;
    followLabel.append(follow, document.createTextNode("跟随最新"));

    const wrapLabel = document.createElement("label");
    wrapLabel.className = "workspace-log-toggle";
    const wrap = document.createElement("input");
    wrap.type = "checkbox";
    wrap.checked = true;
    wrapLabel.append(wrap, document.createTextNode("自动换行"));

    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "small-button";
    copy.textContent = "复制显示内容";

    const download = document.createElement("a");
    download.className = "workspace-log-download";
    download.hidden = true;
    download.textContent = "下载完整日志";

    const summary = document.createElement("span");
    summary.className = "workspace-log-summary";
    summary.setAttribute("aria-live", "polite");
    summary.setAttribute("aria-atomic", "true");

    toolbar.append(filterLabel, search, followLabel, wrapLabel, copy, download, summary);
    logs.parentNode.insertBefore(toolbar, logs);

    const state = { logs, toolbar, severity, search, follow, wrap, copy, download, summary,
      taskId: "", sourceLines: [], sourceSignature: null, visibleLines: [], initialized: false };
    toolbar._maichartLogState = state;
    logs.addEventListener("scroll", () => {
      if (logs.scrollTop + logs.clientHeight < logs.scrollHeight - 8 && follow.checked) follow.checked = false;
    });
    const rerender = () => renderVisible(state, false);
    severity.addEventListener("change", rerender);
    search.addEventListener("input", rerender);
    follow.addEventListener("change", () => {
      if (follow.checked) logs.scrollTop = logs.scrollHeight;
    });
    wrap.addEventListener("change", () => logs.classList.toggle("workspace-log-nowrap", !wrap.checked));
    copy.addEventListener("click", async () => {
      const text = state.visibleLines.join("\n");
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
        } else {
          const field = document.createElement("textarea");
          field.value = text;
          field.setAttribute("readonly", "");
          field.style.position = "fixed";
          field.style.opacity = "0";
          document.body.appendChild(field);
          field.select();
          const copied = document.execCommand("copy");
          field.remove();
          if (!copied) throw new Error("copy command unavailable");
        }
        summary.textContent = `已复制 ${state.visibleLines.length} 行`;
      } catch (_) {
        summary.textContent = "复制失败，请选择日志后手动复制";
      }
    });
    return toolbar;
  }

  function renderVisible(state, followLatest) {
    const { logs, severity, search, summary } = state;
    const visible = filterLines(state.sourceLines, { severity: severity.value, query: search.value });
    const text = visible.join("\n") || (state.sourceLines.length ? "没有符合筛选条件的日志" : "");
    if (logs.textContent !== text) {
      const previousTop = logs.scrollTop;
      logs.textContent = text;
      if (followLatest) logs.scrollTop = logs.scrollHeight;
      else logs.scrollTop = previousTop;
    }
    state.visibleLines = visible;
    const latest = [...state.sourceLines].reverse().find(isKeyEvent);
    const lineCount = `${visible.length}/${state.sourceLines.length} 行`;
    summary.textContent = latest ? `${lineCount} · 最新事件：${latest.slice(0, 120)}` : lineCount;
  }

  function renderTask(task) {
    if (typeof document === "undefined") return;
    const logs = document.querySelector("#taskLogs");
    if (!logs) return;
    const toolbar = installControls(logs);
    if (!toolbar) return;
    const state = toolbar._maichartLogState;
    const taskId = String(task.task_id || "");
    const rawLines = Array.isArray(task.logs) ? task.logs.map((line) => String(line)) : [];
    const initializing = ["queued", "running"].includes(task.status) && rawLines.length === 0;
    const source = rawLines.length ? rawLines : initializing ? ["等待任务开始输出日志…"] : [];
    const signature = JSON.stringify(source);
    const taskChanged = state.taskId !== taskId;
    const changed = taskChanged || state.sourceSignature !== signature;
    if (taskChanged) {
      state.taskId = taskId;
      state.initialized = false;
      state.follow.checked = true;
      state.search.value = "";
      state.severity.value = "all";
    }
    state.sourceLines = source;
    if (changed) state.sourceSignature = signature;
    state.download.hidden = !task.result?.log_path;
    if (task.result?.log_path) state.download.href = downloadUrl(task.result.log_path);
    if (changed) {
      renderVisible(state, !state.initialized || state.follow.checked);
      state.initialized = true;
      if (rawLines.length === 0) {
        state.summary.textContent = ["queued", "running"].includes(task.status)
          ? "正在初始化，尚无日志"
          : "此任务没有记录日志";
      }
    } else if (rawLines.length === 0) {
      const noLogs = rawLines.length === 0;
      state.summary.textContent = noLogs
        ? (["queued", "running"].includes(task.status) ? "正在初始化，尚无日志" : "此任务没有记录日志")
        : `${state.visibleLines.length}/${rawLines.length} 行`;
    }
  }

  return { severityOf, isKeyEvent, filterLines, downloadUrl, renderTask };
});
