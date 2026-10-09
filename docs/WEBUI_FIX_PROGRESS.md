# WebUI 修复进度

## 事故与恢复（重要）

**事故**：修复过程中我用 PowerShell `Get-Content -Raw` / `Set-Content` 改写 `src/maichart/webui/static/app.js`。
在中文 Windows 上该命令按 ANSI(GBK) 读取 UTF-8 源文件，导致整份脚本文本损坏；随后我又用一份约 109 行的精简脚本覆盖了原约 1650 行的完整前端。

**后果**：页面能打开，但绝大多数按钮没有绑定事件、上传/任务轮询/金标/验收/日志渲染全部失效 —— 也就是“几乎所有按键都无法使用”。

**恢复**：按 `index.html`（控件与按钮清单）与 `server.py`（API 契约）重建了完整前端，包含：

- 训练 / 生成 双工作区页签切换（`body[data-workspace]` 控制显隐）
- 全部 19 个按钮的事件绑定
- 上传（数据集 ZIP / 音频 / 特征 / checkpoint），含后台解压轮询
- 任务轮询、进度、日志、结果文件下载链接
- 训练集检查（manifest 摘要、过滤原因、处理错误列表）
- checkpoint 检查与模型合同回填（含难度筛选、`max_audio_tokens`、`condition_default`）
- 金标集扫描 / 列表 / 切分，并把产出的 pretrain / val / gold_stats 回填到训练与验收
- 生成、质量报告渲染、谱面验收（PASS/WARN/FAIL + 时序 canvas 绘制）
- 持久化日志列表与目录复制
- 设置持久化（含每个 epoch 随机种子、训练难度区间）与两个重置按钮

**教训**：不再用 PowerShell 读写项目源码文件；所有源码改动只走 `read`/`edit`/`write` 工具。

## 训练集筛选（本次新增）

训练页新增并已贯通到训练命令：

| 控件 | payload 键 | CLI 参数 |
|---|---|---|
| 最低/最高难度槽 (1–6) | `min_difficulty_index` / `max_difficulty_index` | `--min-difficulty-index` / `--max-difficulty-index` |
| 最低/最高定数 (1–15) | `min_level` / `max_level` | `--min-level` / `--max-level` |
| 音频最大 token | `max_audio_tokens` | `--max-audio-tokens` |
| 条件遮罩默认值 | `condition_default` | `--condition-default` |

这些值会写入 dataset signature 与 checkpoint `data_contract`，续训时校验，避免同名 checkpoint 对应不同训练集。

## 验证证据

- `node --check src/maichart/webui/static/app.js` 通过。
- 前端启动桩检查（`.scratch/webui_boot_check.mjs`）：`WEBUI_BOOT_OK buttons=19`，即在最小 DOM 环境下脚本完整加载、19 个按钮全部绑定、点击不抛未预期异常。
- 临时实例实测：`/`=200（62370 B）、`/app.js`=200（62048 B）、`/api/status`=200。
- 你的运行实例（端口 7860）已在提供修复后的 `app.js`，**刷新页面即可**，不需要重启服务。
- 全量测试：`588 passed, 11 skipped`。
- 新增 API 回归测试验证 WebUI 将难度槽、定数范围、max_audio_tokens 和 condition_default 正确传至训练 CLI：`14 passed`。
- 前端训练/生成参数表单和 API 测试：`68 passed`。

## 当前状态

未启动训练、GPU benchmark 或真实音频生成。下一步可由用户在“训练模型”页设定难度槽与定数区间后，先跑 1–2 个 epoch 的短跑训练。
