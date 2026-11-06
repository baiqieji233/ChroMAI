# 维护交接

这里保存各阶段的变更范围、已经验证的内容、遗留问题与接手入口。运行方式与参数统一放在[使用与技术说明](../README.md)，兼容性放在[环境与兼容性](../compatibility/README.md)。

## 接手 Composer v2

1. 从[连续事件改造](../explanations/COMPOSER_V2_CONTINUOUS_EVENTS_20261104_ZH.md)确认实际时间编码和上下文边界。
2. 从[成谱质量复查](../explanations/COMPOSER_V2_QUALITY_REVIEW_20261105_ZH.md)找到节奏前文纠错的原始对照、训练设置与尚待验收的情况。
3. 遇到生成失败时先查任务的诊断 JSON，按[单处恢复预算](../explanations/COMPOSER_V2_RECOVERY_FRONTIER_20261105_ZH.md)、[持续动作修复](../explanations/COMPOSER_V2_SUSTAIN_REARTICULATION_20261105_ZH.md)和[冲突放行报告](../explanations/COMPOSER_V2_EXHAUSTION_REPORT_20261105_ZH.md)定位。
4. 接手训练前从日志和检查点确认模型族、序列策略、数据清单、训练进度及实际运行参数。文档中的目录、端口与进度是对应日期的记录。

## 早期阶段

- [Composer v1 全量架构排查](COMPOSER_FULL_ARCHITECTURE_AUDIT_HANDOFF_20261027_ZH.md)
- [Composer v1 训练与生成的节奏交接](COMPOSER_GENERATION_RHYTHM_AND_REPETITION_HANDOFF_20261029_ZH.md)
- [全曲音乐与历史配对](COMPOSER_SONG_PAIRED_ARCHITECTURE_HANDOFF_20261029_ZH.md)
- [整拍集合生成实现](COMPOSER_EVENT_SET_STREAM_IMPLEMENTATION_20261030_ZH.md)
- [Stream v18 质量与音频路线](STREAM_V18_AUDIO_FIDELITY_REPETITION_HANDOFF_20261021.md)
- [Windows 本机运行环境恢复记录](WINDOWS_RUNTIME_RECOVERY_20261026_ZH.md)

其余交接按文件名中的主题与日期查阅。新增交接时记录当时的提交或分支、模型和数据范围、验收证据与后续事项。可长期使用的操作说明另存到 `docs/explanations/`，原始报告附件放在 `docs/evidence/`。
