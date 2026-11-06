# 项目文档

第一次使用从[项目 README](../README.md)开始。当前开发路线是主工作台中的 `chart_composer_v2` 连续事件模型。

| 目录 | 内容 |
| --- | --- |
| [使用与技术说明](explanations/) | 操作、参数、模型设计、问题复盘和实测记录。带日期的报告描述当时的实现与任务。 |
| [环境与兼容性](compatibility/README.md) | 启动环境、旧模型入口、旧参数和检查点迁移。 |
| [维护交接](handoffs/README.md) | 各阶段的接手资料、未决问题和下一步工作。 |
| [验证资料](evidence/) | 报告使用的 JSON、图表和原始记录。早期流式模型的证据另见 [audits](audits/)。 |
| [历史归档](archive/20261015/README.md) | 已归档的旧质量改进材料。 |

## Composer v2

- [WebUI 操作与参数](explanations/COMPOSER_V2_WEBUI_ZH.md)覆盖生成、预览、数据准备、训练和检查点。
- [连续事件改造](explanations/COMPOSER_V2_CONTINUOUS_EVENTS_20261104_ZH.md)说明拍位编码、上下文整理、持续状态和训练目标。
- [节奏前文纠错与成谱复查](explanations/COMPOSER_V2_QUALITY_REVIEW_20261105_ZH.md)记录修复依据、对照产物和验收边界。
- [BPM 与自动偏移](explanations/COMPOSER_V2_TIMING_AUDIT_20261105_ZH.md)记录恒速、变速实测及需要手动确认的情况。
- [生成后预览](explanations/MIAVIEW_GENERATION_PREVIEW_20261105_ZH.md)说明播放器、任务绑定和本地资源。
- [冲突恢复与报告](explanations/COMPOSER_V2_EXHAUSTION_REPORT_20261105_ZH.md)说明恢复预算耗尽后的处理和诊断。

## 代码入口

| 环节 | 位置 |
| --- | --- |
| 主工作台、任务与参数传递 | [WebUI](../src/maichart/webui/) |
| 数据、模型与训练 | [Composer v2](../src/maichart/composer_v2/) |
| 连续时间编码和生成 | [序列编码](../src/maichart/composer_v2/stream_codec.py)、[生成循环](../src/maichart/composer_v2/stream_generate.py) |
| 谱面时钟和原生冲突检查 | [节拍时钟](../src/maichart/tempo_clock.py)、[MiaCode 检测](../src/maichart/generation/miacode_native.py) |

架构演进和已经踩过的问题见[技术复盘](explanations/CHART_GENERATION_TECHNICAL_RETROSPECTIVE_ZH.md)。旧报告里的默认值、运行端口、实验路径和任务状态属于当时的记录，现用参数以本分支的 WebUI、CLI 和检查点能力为准。
