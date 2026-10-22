# Chart Composer v1 主工作台迁移交接

日期：2026-10-22
分支：`codex/chart-composer-v1-20261022`

## 已完成

`chart_composer_v1` 已接入主 WebUI 工作台。模型族下拉框、训练参数区、生成参数区和 checkpoint 检查会根据当前族切换；主训练按钮调用 `/api/composer/train`，主生成按钮调用 `/api/composer/generate`，生成检查调用 `/api/composer/inspect`。独立 `/composer` 页面仍作为隐藏兼容入口保留，主工作台不再显示它的导航链接。

Composer 的结构参数包括 `d_model`、`dim_feedforward`、层数、注意力头数、历史事件数、候选拍位数和动作槽位数；其中 `dim_feedforward` 已从主界面贯通到 checkpoint 和训练 CLI，避免只提高模型宽度而让前馈层停留在过窄的默认值。

服务端现在能够从 Composer checkpoint schema 识别模型族，并在主 `/api/train`、`/api/generate` 和 `/api/checkpoint/inspect` 路由中转发到 Composer 适配器。多数据集选择先经过现有数据集合并器，Composer 使用合并后的 event source manifest，不再只取列表中的第一份清单。Composer 训练日志中的 JSON 进度会转换为主工作台任务进度和 epoch 状态；`composer_last.pt`、生成的 `.txt` 和 `.composer.json` 会进入任务产物列表。

数据集合同仍复用 event-preprocessed manifest。`chart_composer_v1` 只接受 `maichart-chart-composer-checkpoint-v1`，与 event、song-form、chart-stream 和 legacy checkpoint 不兼容。resume 会从 checkpoint 读取结构；finetune 也不会用 WebUI 新结构参数重建模型，结构控件会在检查 checkpoint 后锁定。

## 验证边界

- 已完成 Composer 核心 CPU 训练、checkpoint 保存、CPU 生成和既有 WebUI Node 回归。
- 本轮补充了主工作台接线、服务端 schema 识别、路由、联合 manifest 和任务进度解析；未启动 GPU 训练，也未把短跑结果当作音乐质量证据。
- 需要重启正在运行的 WebUI 进程后，7860/7861 才会加载本轮服务端和静态资源改动；旧进程仍可能只显示旧模型族列表。

## 后续边界

Composer 当前仍是事件级原型：候选召回率、复杂 Slide 路径结构化、EOS/长休止决策和整段质量修订尚未达到 stream 的完整能力。正式训练前应先用小 manifest 做 CPU 或短 CUDA 烟雾，确认候选覆盖率和产物链路，再扩大数据规模。
