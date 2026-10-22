# chart_composer_v1 WebUI handoff

日期：2026-10-22

本次新增独立入口 `/composer`，对应模型族 `chart_composer_v1` 和 checkpoint schema `maichart-chart-composer-checkpoint-v2`。入口页不依赖主工作台的 `app.js`，只使用现有 WebUI 的任务运行时和下载路由。

## 已实现

- `src/maichart/webui/composer_api.py`
  - `POST /api/composer/train`
  - `POST /api/composer/generate`
  - `POST /api/composer/inspect`
  - 训练参数映射到 `python -m maichart.composer.train`，支持 `fresh`、`resume`、`finetune`、CPU/CUDA/auto、precision、manifest、缓存目录和模型结构参数。
  - 生成参数映射到 `python -m maichart.composer.generate`，支持 BPM、first、level、temperature、seed、max-events、condition-file。
  - inspection 使用 CPU `torch.load(..., weights_only=False)`，再调用 `maichart.composer.checkpoint.inspect_payload`。
- `src/maichart/webui/static/composer.html/js/css`
  - 独立训练、生成、checkpoint inspection 页面。
  - 任务状态轮询 `/api/task?id=...`，完整显示内存日志；任务仍由 server 的持久化日志机制保存。
- `server.py`
  - 新增 `/composer` 页面和静态资源路由。
  - 新增 composer 三个 POST dispatcher，复用 `_create_task`、`_run_subprocess_task`、`/api/cancel`、`/api/download`。
- 主入口 `index.html` 增加 Composer 链接。

## 核心模块对接假设

核心实现需要提供：

- `python -m maichart.composer.train`，并接受用户约定的训练 CLI 参数；完成后在 `checkpoint-dir/composer_last.pt` 生成产物。
- `python -m maichart.composer.generate`，并按输出路径生成 `.txt` 与同 stem 的 `.composer.json`。
- `maichart.composer.checkpoint.inspect_payload(payload)`。

当前工作区在本轮开始时仅存在 composer 的模型/检查点基础模块，训练与生成 CLI 尚未出现；API 已按上述契约接好，待核心模块落地后可直接运行。

## 任务状态

训练任务类型为 `composer_train`，生成任务类型为 `composer_generate`。训练进度解析依赖核心 stdout 中的 `epoch=...`；核心 JSON 日志会原样进入任务日志。结果状态通过现有 `TaskState` 返回，页面会显示 `result` 中的输出路径、checkpoint 目录和日志路径。

## 边界

本轮没有修改 composer 核心、stream API 或共享训练参数；没有运行测试或启动服务。`manifest` 列表目前取第一个非空 manifest，因为核心 CLI 合约是单个 `--manifest`；需要多 manifest 合并时应在准备阶段生成合并清单，或由核心 CLI 扩展参数后再同步此适配层。
