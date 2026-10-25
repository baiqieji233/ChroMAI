# Composer v3 音频、相位与可玩性交接文档（中文版）

原文：[COMPOSER_V3_AUDIO_PHASE_PLAYABILITY_HANDOFF_20261022.md](file:///e:/simai_transformer/docs/COMPOSER_V3_AUDIO_PHASE_PLAYABILITY_HANDOFF_20261022.md)  
日期：2026-10-22

## 范围

`chart_composer_v1` 保持相同的模型张量形状和动作词表（action catalog），但由于候选点/音频输入行为发生了变化，v3 使用全新的训练输出目录。`Prism_composer_v2` 保留为上一阶段的基线，不做任何修改。

## 变更内容

- 候选点生成现在在训练与推理两端共享同一套相位转移规则。精细的 `{48}`/`{64}` 相对偏移依然可用，同时阻止二进制/三连音混合相位持续滞留在会编译为 `{192}` 的 `/48` 分母死锁状态中。
- 候选时刻插值引入了显式的 `attack_confidence_v2` 策略。Mel、RMS 和 chroma 保持平滑插值；当查询时刻远离稀疏证据节点时，窄窗攻击（attack）派生列会被衰减。特征总宽度仍为 120，但绝不能将旧检查点当作输入分布未变而直接续训（resume）。
- 生成阶段会在每个候选时刻检查活动的 Hold/Touch Hold 以及在途 Slide/Wifi 状态，限制可用手数并屏蔽已知被占用的起始键位（heads）；完整的 MiaCode Python 审查在生成结束后运行一次，并仅在删除操作能够减少硬冲突数量时保守地删除已映射的硬冲突。软冲突或未映射的发现项保留在 `.composer.json` 报告中。
- 训练运行时会在启动记录和数据集指纹（dataset fingerprint）中报告 `phase_filter_misses` 以及音频特征策略。

## 训练合同（Training contract）

v3 为全新训练（fresh run），运行参数与已停止的 v2 任务保持一致：宽度 384、6 层、FFN 1536、8 头、batch 32、BF16、学习率 `2e-4`、候选数 96、历史长度 128、动作槽位 8、预取深度 2、4 个数据准备进程、fused/auto 优化器、共 80 轮（epochs），仅写入 `checkpoints/Prism_composer_v3`。

现有的 v2 检查点保留用于 A/B 对比。请勿将其 resume 到 v3 中；特征策略和相位过滤的变化要求使用全新的数据集指纹。音乐质量的最终确认仍需结合固定音频生成、冲突报告、密度/空白跨度检查以及人工试玩评审。

全新启动记录：WebUI 任务 `a3824e280d30`，输出目录 `checkpoints/Prism_composer_v3`，于 2026-10-22 按上述参数启动。该任务已进入第 1 轮数据批次，无父检查点。

## 验证边界

在启动 v3 之前，请先运行针对相位转移、攻击置信度衰减、候选张量形状、后处理导入以及检查点/模型构建的 CPU/静态检查。切勿将这些基础检查当作音乐质量的证明。
