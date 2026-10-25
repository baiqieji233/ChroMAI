# Composer 事件视图与高密同键修复交接文档（中文版）

原文：[COMPOSER_EVENT_VIEWS_DENSITY_HANDOFF_20261023.md](file:///e:/simai_transformer/docs/COMPOSER_EVENT_VIEWS_DENSITY_HANDOFF_20261023.md)  
日期：2026-10-23 · 分支：`codex/composer-music-context-v2-20261023`

## 授权与快照

用户授权修改训练合同，并要求从根因上修复近期生成中出现的超高密度同键连打问题（而非仅靠输出后过滤）。  
任务 `f3e488d5e119`（`Prism_composer_v6`）已通过 WebUI 取消，保留的最后检查点位于优化步 `8500`（最后观察到的实时日志为第 4 轮 / 批次 5375 / 优化步 8762），未删除或覆盖任何检查点。源码与工作区备份位于 `backups/composer_density_stack_20261023_142449/` 及 `backups/composer_event_density_completion_20261023/`。

## 合同决策（Contract decisions）

- 由单一、显式且不可变的事件视图策略（`EventViews`）统一定义：有效音频范围内的事件、完整的因果上下文（包含被质量排除的原谱事件）、可信标签事件（trusted label events）以及候选可达的受监督子集。
- 被排除/未知（excluded/unknown）的时间区间绝不能被当作休止（rest）或提前结束（EOS）标签。
- 负荷（workload）标签使用可复用的精确三角核前缀查询（`ExactWorkloadQuery`）；对称负荷掩码与历史窗口（trailing）掩码相互独立，并严格匹配各自的真实时间支持域。
- `future` 与 `EOS` 标签携带显式的有效性掩码；原始未知尾部绝不伪造为 `EOS` 正样本。
- 训练与新生成共享基于解析器的触发头/类型语义（`event_metadata`）；若 `action_slots` 设置过小导致截断原谱事件，必须直接报错拒绝而非静默截断。
- 数据集指纹（fingerprint）完整纳入实际事件内容、变速/首拍、音频身份与标签策略；旧权重仍可用于生成和显式微调迁移，但禁止按同合同直接 resume。

## 阶段进展

1. **根因排查**：确认问题根源来自候选相位恢复中的微小桥接步长、缺少逐键时间间隔输入（per-pad temporal input）、事件视图口径不一致、稀疏网格插值的负荷标签，以及原生门控（guard）对首事件和纯 Tap 的绕过。原生 `MiaCode` 并非人类单键连打速度（jack-speed）校验器，因此需要独立的、可配置的生成期同键安全策略。
2. **代码实现**：统一的 `EventViews` 已接入数据集；精确负荷查询与独立标签掩码已接通；逐键因果时间特征（per-pad causal timing）已接入全新训练、生成与显式迁移；候选策略 `v6` 移除了仅用于相位恢复的 `< 1/16` 拍超细桥接提案；`EventGuard` 事务校验已覆盖首个事件与纯 Tap 事件，并集成了 WebUI 同键间隔安全控件。
3. **集中测试**：按约定延后至实现与集成审查完成后执行（编写阶段尚未运行新增测试）。

## 续接工作与兼容性说明

当前训练语义标识升级为：`chart_composer_training_v6_unified_events_head_timing`。  
新版 `music_context_v2` 模型启用 `head_timing_dim=99`：对全部 33 个按键位置分别编码“是否已出现标志（seen flag）”、“距上次触发的对数秒数 `log1p(age_sec)`”和“距上次触发的对数拍数 `log1p(age_beat)`”。快照严格在目标事件发生之前提取；在模态 Dropout 下该特征与符号历史一并隐藏，且不进入纯音频辅助分支。新增的线性投影层仅含 `99 * d_model`（加偏置共 `100 * d_model`）个参数，并在迁移时以全零初始化。

旧权重仍可正常生成；旧优化器状态不能在新合同下直接 resume（请使用新目录进行显式 finetune 或启动全新训练）。当旧版 v5 检查点在生成时升级使用 v6 安全候选策略时，生成报告会分别记录训练期策略（`trained_candidate_policy`）与实际生效策略（`candidate_policy`）。

默认的 `50ms` 同键最短间隔（`min_same_head_ms`）是一项显式的产品安全策略，而非 `MiaCode` 内置规则。CLI `--min-same-head-ms` 与主 WebUI 控件支持在 `0..200ms` 范围内调节；设为 `0` 仅关闭该同键间隔策略，原生 `MiaCode` 冲突检测始终保持启用。训练端不对原谱事件施加该间隔过滤。生成报告会完整记录门控拒绝原因以及后处理前后的密度诊断数据。
