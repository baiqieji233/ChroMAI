# 谱面质量改进实施记录
日期：2026-10-15
目标：解决审计文档中确认的训练-生成gap、配置重复、规划-配置脱节等问题

## 改进概述

基于以下审计文档的分析：
- `docs/explanations/SONG_FORM_QUALITY_ROOT_CAUSE_AUDIT_20261013.md`
- `docs/explanations/SONG_FORM_FULL_CHAIN_AUDIT_20261013.md`

实施了四个主要改进方向，所有修改均为破坏性更新，需要重新训练验证。

---

## 改进1：训练输入扰动（Training Input Perturbation）

### 问题根源
**训练-生成gap**：模型在训练时看到完美的官谱输入（准确起音、正确前文、精确曲线），但生成时面对不完美的预测输入，导致性能崩溃。

审计文档 A01 项（P0级别）明确指出：
> "训练样本用官谱起音骨架、官谱前文与官谱曲线；生成用预测骨架、生成前文与预测/编辑曲线。"

### 实施方案
创建 `src/maichart/form/training_input_perturbation.py` 模块，实现三种受控扰动：

1. **起音扰动**
   - `onset_drop_prob`: 随机删除起音（模拟Planner漏检）
   - `onset_add_prob`: 添加虚假起音（模拟Planner误报）
   - `onset_shift_sigma_ms`: 时间偏移（模拟定位误差）

2. **曲线噪声**
   - `curve_noise_std`: 高斯噪声（模拟预测误差）
   - `curve_dropout_prob`: 帧随机置零（模拟预测空白）

3. **历史错误传播**
   - `history_corruption_prob`: 随机改变前文配置（模拟生成错误）

### 技术细节
- 所有扰动都是概率性的，通过 `apply_prob` 控制总体应用频率
- 扰动幅度有硬上限保护，避免破坏训练
- 集成到 `FormDataset` 和 `input_example` 函数中
- 训练参数通过命令行完全可配置

### 训练参数
```bash
--perturb-onset-drop-prob 0.0-0.3     # 起音删除概率
--perturb-onset-add-prob 0.0-0.2      # 起音添加概率
--perturb-onset-shift-ms 0.0-20.0     # 时间偏移(毫秒)
--perturb-curve-noise-std 0.0-0.1     # 曲线噪声标准差
--perturb-curve-dropout-prob 0.0-0.2  # 曲线dropout概率
--perturb-history-corruption-prob 0.0-0.1  # 历史错误概率
--perturb-apply-prob 0.0-1.0          # 总体应用概率
```

### 推荐渐进策略

**阶段1（轻微扰动，2-3轮）**：
```bash
--perturb-onset-drop-prob 0.05
--perturb-onset-shift-ms 5.0
--perturb-apply-prob 0.3
```

**阶段2（加入曲线噪声，2-3轮）**：
```bash
--perturb-onset-drop-prob 0.10
--perturb-curve-noise-std 0.03
--perturb-apply-prob 0.5
```

**阶段3（全面扰动，2-3轮）**：
```bash
--perturb-onset-drop-prob 0.15
--perturb-onset-shift-ms 10.0
--perturb-curve-noise-std 0.05
--perturb-history-corruption-prob 0.05
--perturb-apply-prob 0.7
```

### 理论基础
- **Data Augmentation**：图像识别、语音识别的标准技术
- **Robust Training**：训练时加入噪声提高泛化能力
- **Domain Randomization**：机器人学习中的成熟方法

### 修改文件
- 新增：`src/maichart/form/training_input_perturbation.py`
- 修改：`src/maichart/form/data.py` - 添加扰动应用逻辑
- 修改：`src/maichart/form/train.py` - 添加训练参数和配置传递

---

## 改进2：规划器与配置模型连接（Planner-Configuration Guidance）

### 问题根源
**规划-配置脱节**：Planner预测了note类型和密度，但配置解码器实际上没有使用这些预测。

审计文档记录的实际案例：
- Planner预测：Tap 927个、Slide 18个
- 实际输出：Hold 360个、TouchHold 28个、Slide 7个

两个模块各说各话，无法形成有效配合。

### 实施方案
创建 `src/maichart/form/planner_guidance.py` 模块，实现软约束机制：

1. **PlannerGuidanceModule**
   - 将Planner的类型概率和时值预测投影到解码器隐藏空间
   - 通过学习的trust gate决定何时信任Planner
   - 只在起音位置应用引导，避免干扰其他token

2. **一致性损失**
   - `compute_planner_consistency_loss`: 鼓励解码器输出与Planner预测一致
   - 根据Planner的置信度（熵）动态调整权重
   - 高置信度预测 = 更强约束

### 技术细节
- 引导强度可配置（推荐0.2-0.4）
- 门控机制自动学习何时信任Planner
- 只影响起音位置，不改变整体架构

### 注意事项
此模块已创建但**尚未集成到模型forward**中。需要：
1. 修改 `FormTransformer.forward` 接收Planner输出
2. 在解码器中应用 `PlannerGuidanceModule`
3. 在训练loss中加入一致性损失

### 修改文件
- 新增：`src/maichart/form/planner_guidance.py`
- 待修改：`src/maichart/form/model.py`（需要集成）

---

## 改进3：配置重复检测与惩罚（Repetition Penalty）

### 问题根源
**连续5次相同配置**：审计文档显示修复曲线条件后，重复配置比例从17.6%暴增到41.2%，最大连续重复从2变成5。

这在音乐游戏谱面中是严重的设计失误，会让玩家感到无聊。

### 实施方案
创建 `src/maichart/form/repetition_penalty.py` 模块，实现多层次惩罚：

1. **训练时惩罚**
   - `compute_repetition_penalty_loss`: 检测训练样本中的重复模式并惩罚
   - 检测3-8 token的重复pattern，超过3次重复触发惩罚

2. **多样性损失**
   - `compute_pattern_diversity_loss`: 鼓励相邻段落的隐藏表示差异化
   - 通过余弦相似度度量段落表示
   - 惩罚过高的相邻相似度

3. **生成时惩罚**
   - `apply_repetition_penalty_sampling`: 生成时动态降低最近出现token的概率
   - 根据出现频率指数惩罚

4. **监控机制**
   - `RepetitionMonitor`: 实时统计训练中的重复比例
   - 追踪最大连续重复次数

### 训练参数
```bash
--repetition-penalty-weight 0.0-0.2   # 重复惩罚权重
--diversity-loss-weight 0.0-0.1       # 多样性损失权重
```

推荐起始值：
```bash
--repetition-penalty-weight 0.1
--diversity-loss-weight 0.05
```

### 技术细节
- 已集成到 `layout_loss` 函数
- 模型导出 `hidden_states` 供diversity loss使用
- 所有惩罚可独立启用/禁用

### 修改文件
- 新增：`src/maichart/form/repetition_penalty.py`
- 修改：`src/maichart/form/train.py` - 添加参数和loss集成
- 修改：`src/maichart/form/model.py` - 导出hidden_states

---

## 改进4：增强训练指标可观测性

### 问题根源
**训练指标掩盖真实任务难度**：总loss 0.27看起来很低，但：
- 修订任务CE: 0.167（准确率94.9%）
- 从头生成任务CE: 0.873（准确率71.4%）
- 首字符准确率：仅52.6%

容易预测的token（时间位置、括号）拉高了总体准确率。

### 实施方案
审计文档已经添加了详细的分任务指标，本轮主要确保：
1. 扰动统计被正确记录
2. 重复惩罚和多样性损失被单独报告
3. 训练日志包含扰动配置信息

### 输出示例
```json
{
  "form_perturbation": {
    "enabled": true,
    "onset_drop_prob": 0.1,
    "curve_noise_std": 0.05,
    "apply_prob": 0.5
  }
}
```

---

## 已知限制与未完成项

### 1. Planner引导尚未完全集成
`planner_guidance.py` 模块已创建，但需要修改 `FormTransformer` 的forward方法才能实际使用。

**需要做的**：
- 在encode阶段保存Planner的type_logits和duration预测
- 在decode阶段应用PlannerGuidanceModule
- 在训练loss中加入planner_consistency_loss

### 2. 起音扰动未包含Planner预测替换
当前实现是对官谱起音进行随机扰动，但理想情况应该：
- 用真实的Planner预测替换官谱起音
- 需要在数据准备时先运行Planner得到预测

这需要显著改变数据准备流程，暂未实施。

### 3. 多标签类型支持
审计文档提到的"单一类型标签压缩多押信息"问题尚未解决。
这需要改变Planner的输出head从单类别到多标签，是较大的架构改动。

### 4. WebUI参数暴露
所有新增参数目前只能通过命令行使用，尚未暴露到WebUI界面。

**下一步任务**：修改 `src/maichart/webui/server.py` 和前端代码。

---

## 验证建议

### 固定测试集
选择3-5首风格不同的歌曲，每轮训练后生成并对比：
1. 配置重复比例（目标：<15%，最大连续<3）
2. 音乐对齐质量（起音与音频peak的对齐）
3. 空白段数量（不合理的长时间空白）
4. 密度曲线响应（是否跟随预期的强弱变化）

### 训练监控
关注以下指标变化：
- 总loss vs 分任务loss（草稿/修订）
- 首字符准确率（目标：>70%）
- 配置重复率（RepetitionMonitor统计）
- 扰动样本的loss vs 未扰动样本的loss

### A/B对比
- 基线模型（无扰动，无惩罚）
- 仅扰动模型
- 扰动+重复惩罚模型
- 全部改进模型

对比相同种子、相同歌曲的生成结果。

---

## 风险与应对

### 风险1：扰动过大导致训练崩溃
**症状**：loss不下降或振荡
**应对**：从最小扰动开始（阶段1参数），逐步增加

### 风险2：重复惩罚过强导致配置过度分散
**症状**：相同段落失去一致性，每拍配置都不同
**应对**：降低 `repetition_penalty_weight` 到 0.05-0.1

### 风险3：多样性损失与音乐结构冲突
**症状**：副歌重复段落变得不一致
**应对**：降低 `diversity_loss_weight` 或关闭

### 风险4：训练变慢
**影响**：扰动计算是毫秒级，不会显著拖慢训练
**监控**：对比有无扰动的epoch时间

---

## 下一步工作

### 短期（1-2天）
1. ✅ 完成代码实现
2. ⏳ 暴露参数到WebUI
3. ⏳ 进行阶段1训练测试（轻微扰动）

### 中期（1周）
4. ⏳ 完成Planner引导的集成
5. ⏳ 进行完整的渐进式训练（阶段1-3）
6. ⏳ 固定测试集质量评估

### 长期（2-4周）
7. ⏳ 多标签类型支持
8. ⏳ 真实Planner预测替换官谱起音
9. ⏳ 可玩性建模（双手调度、MiaCode检测作为训练目标）

---

## 文件清单

### 新增文件
- `src/maichart/form/training_input_perturbation.py` - 训练输入扰动
- `src/maichart/form/planner_guidance.py` - 规划器引导（未集成）
- `src/maichart/form/repetition_penalty.py` - 重复检测与惩罚

### 修改文件
- `src/maichart/form/data.py` - 添加扰动应用
- `src/maichart/form/train.py` - 添加训练参数和loss集成
- `src/maichart/form/model.py` - 导出hidden_states

### 待修改文件
- `src/maichart/webui/server.py` - 暴露参数到WebUI
- `webui_workspace/` - 前端界面更新

---

## 总结

本轮修复直接针对审计文档确认的三个核心问题：
1. **训练-生成gap** → 通过输入扰动让模型学会应对不完美输入
2. **配置重复** → 通过多层次惩罚机制减少重复
3. **规划-配置脱节** → 通过软约束建立连接（部分完成）

所有改进都是可配置的，可以从0开始逐步启用，降低风险。

**关键提醒**：这些改进需要重新训练验证，不能保证立即提升质量，需要通过固定测试集的实际生成结果来评估效果。
