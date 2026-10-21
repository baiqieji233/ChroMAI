# Stream v18 审阅问题修复交接

日期：2026-10-21

## 本轮确认并修复

1. `density_interval_cost` 的门控方向已修正。短于 Planner 目标间隔的候选，在谱师选择支持弱时处罚更强；有明确选择支持时只减免部分处罚。长休止仍不受该项处罚。
2. 主训练序列不再完全跳过位置遮罩。`decoder_input` 中的旧事件会采样位置遮罩事件，但只清除对应时钟几何回声，保留原始 `P:` token，避免破坏当前 Fraction 的 teacher forcing。最近事件保留为锚点。
3. `condition_dropout` 已同步到 local grounding。`condition_null` 现在同时作用于全局音频编码、local condition projection 和 alignment selection 输入，不会只屏蔽一条路径。
4. `PositionEvidence` 将 alignment concentration 与 selection contrast 分开记录。selection 代价不再被 alignment concentration 单独压低；同时保留 selection 平坦时的保守停用行为。
5. 均匀十六分不做全局惩罚。只有持续时间足够长、局部负荷偏低、谱师选择支持弱且间隔足够短时，才施加渐进疲劳项，以保留正常扫键、交互和切分。

## 对审阅结论的修正

- 审阅者关于 `density_interval_cost` 门控方向的判断正确。
- 审阅者关于主序列位置遮罩未生效的判断正确，但 `CausalClock` 的列含义并不是“7:15 的 grid_run”；当前列布局是 geometry、秒轴、间隔、提交数、pending 和 BPM。
- 审阅者关于 selection 与 alignment 置信度耦合的问题正确，但当前修复保留了 selection 平坦时的保守关闭策略。
- `rhythm_cycle_cost` 豁免均匀间隔是有意保护正常音乐游戏节奏的设计，不应直接改成普遍加罚。

## 额外发现

高温结构逃逸中的 novelty 终端分数改变了最终候选目标，因此 `_mixture_correction` 仍然只是在校正提案混合分布，不能严格称为对原始模型分布无偏。该行为是有意的生成质量启发式，后续诊断应把它记录为目标重排。

## 验证

- Stream grounding、generation evidence、causal motor、pipeline、training contract、structural repeat、timing search、configuration search：134 passed。
- 仍需在实际 fresh/finetune 训练后用固定歌曲、固定 seed 和固定生成参数做音乐质量 A/B；本交接文档只证明代码合同和定向逻辑。
