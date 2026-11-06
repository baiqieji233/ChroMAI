# Composer music-context v2 实现交接文档（中文版）

原文：[COMPOSER_MUSIC_CONTEXT_V2_HANDOFF_20261023.md](COMPOSER_MUSIC_CONTEXT_V2_HANDOFF_20261023.md)\
日期：2026-10-23

## 范围与备份

2026-10-23，用户授权修改模型/数据/生成代码并停止当前训练任务。  
源码备份位于 `backups/composer_music_alignment_20261023_101702/source_before.zip`（同目录含工作区补丁与活动任务快照）。任务 `657e02a68227`（`Prism_composer_v5`）已通过 WebUI 取消，其已有检查点保持完好。

## 实现前的设计决策

- 将新训练架构版本化为 `music_context_v2`；旧模型保留各自的架构与生成策略，严禁跨合同静默续训（resume）。
- 新增从已提交前驱到候选点的跳过区间声学特征（interval audio）、近期历史交叉注意力（cross-attention）、纯音频选位辅助目标（audio-only pointer objective）以及一致性符号上下文遮挡（coherent symbolic-context dropout）。绝不通过将任意生成前缀拼接到原谱后缀来凭空捏造恢复标签。
- 使预测的过去 `1s/4s` 事件率/头率（trailing workload）与因果台账（causal ledger）窗口严格对齐；保留原谱对称负荷监督与仅使用预测值作为决策输入的原则。被质量排除的谱面区域视为“未知（unknown）”——仅屏蔽监督，绝不将其标注为休止（rest）。
- 在各条件动作槽位之间共享合法原子输出矩阵（shared legal-atom output matrix），保留 `STOP` 监督与完整合法词表，不打包剔除零目标。
- 缓存有序的原谱前缀统计量，减少声学特征库的重复拷贝与逐张量 CUDA 同步；不做全量“事件 × 候选”特征物化，不使用有损缓存，也不靠篡改采样分布来宣称提速。
- 提供轻量级的负荷/台账/历史干预诊断、相位恢复与覆盖率诊断工具；这些诊断仅输出证据，绝不硬性拦截谱面导出。

测试在实现阶段全部完成后统一运行；短测试仅验证合同与数值路径，不代表音乐质量或彻底消除循环。

## 实现与验证

### 第一阶段：核心实现完成

- `model.py`：新增可选的 `music_context_v2` profile、候选到历史的交叉注意力、纯音频指针交叉熵（audio-only pointer CE）、一致性谱面模态 Dropout（默认 `0.15`）、可训练的 `1s/4s` 历史窗口速率头（trailing-rate heads）及阻断梯度的同窗口偏差投影（discrepancy projection）、共享合法原子输出层加槽位嵌入（slot embeddings）。对称负荷曲线头对决策损失保持梯度阻断（`detach`），其投影层正常参与训练。
- `data.py`, `workload.py`：在 v2 上采用仅基于均匀全曲探针（uniform-only）的对称负荷标签、带通道级排除掩码的同窗口 trailing 标签、可复用的有序头数前缀、有界的原谱类型/节奏退出加权（exit weighting），以及事件内原子字典序规范化排序（canonical ordering）。目标标签绝不进入模型输入。
- `candidates.py`：`absolute_recovery_v5` 包含细/粗粒度相位逃逸锚点以及稀疏的 `24/32/48/64/96/128` 拍长休止候选（受 64 秒与音频末尾边界约束）。旧检查点继续使用原有的 v4 策略。候选未命中类别会被分类记录。
- `generate.py`：接入匹配的跳过区间与 trailing 输入，将已提交事件锚点（committed anchor）与搜索游标（search cursor）严格区分，仅对 v2 启用规范化槽位排序，维护增量式头数前缀，并在报告中记录历史窗口覆盖的秒数/拍数分位数。不设任何音乐密度硬门限。
- `checkpoint.py`：完整保留旧模型生成支持；`v1 -> v2` 初始化作为显式微调（finetune）处理，对原 8 个槽位分类器取均值并零初始化新增证据路径（属于近似初始化，不等同于无损等效生成，且禁止跨架构恢复优化器状态）。全新训练或微调禁止覆盖已存在的 `composer_last.pt`。
- `acoustics.py`：已验证的原生特征库（`validated_bank=True`）在构建紧凑上下文时跳过重复的全量扫描、排序与整曲索引拷贝；未验证/旧版入口保留完整清洗路径。
- `train.py`：合并非有限值检查（单次设备到主机同步），用全局梯度范数取代逐参数标量同步；修复当 `locality` 大于 `batch_size` 时的批次越界问题、梯度累积窗口跳过逻辑以及 FP16 跳过更新时的 scaler 状态；新增可选的 CUDA 事件分段计时（H2D/前向/反向/更新）与预取统计。
- WebUI 主工作台与独立页面新增支持 `symbolic_dropout`、`audio_pointer_weight` 与 `profile_steps` 控件，训练看板展示新增的两项损失，服务端能力标识升级为 `music_context_shared_slots_v2`。
- `diagnose.py`：提供只读的固定状态负荷与台账干预、声学证据重排、一致性符号遮挡、删除最近一次已提交事件（同步重算状态/台账/候选）以及一/两步精确相位可达性诊断。

### 第二阶段：短测验证与基线观察

- 聚焦 Python 测试套件：**58 passed**，覆盖 Composer 训练、生成、双 Hold 延续性、精确轮内断点续训、负荷、门控、映射特征库、内存预算及新版 `music_context_v2` 路径。
- JavaScript 测试：**11 passed**，涵盖主工作台控件与训练看板。
- 在 RTX 5070 Ti Laptop 上对 18,454 词表（batch 32，384 维 / 6 层）进行的合成 CUDA 基准对比显示：
  - 旧模型：`85,832,255` 参数（动作头 `56,838,320` 参数），峰值显存 `2.204 GiB`，每步更新 `73.1 ms`；
  - 新模型：`36,715,626` 参数（动作头 `7,104,790` 参数），峰值显存 `1.353 GiB`，每步更新 `68.3 ms`。

### 运行与迁移说明

全新训练自动启用 `music_context_v2`。若基于已有 workload 检查点迁移，请选择 **finetune（微调）** 并指定全新的输出目录（如 `checkpoints/Prism_composer_v6`）。推荐起始参数：`symbolic_dropout = 0.15`，`audio_pointer_weight = 0.15`，`curve_strength = 1.0`。

只读诊断命令示例：
```powershell
python -m maichart.composer.diagnose --checkpoint checkpoints/Prism_composer_v5/composer_last.pt --manifest E:\全量官谱\training_manifest.json --limit-charts 2 --samples 8 --device cuda --output .scratch/composer_interventions.json
```
