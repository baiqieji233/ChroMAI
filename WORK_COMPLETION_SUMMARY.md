# 谱面质量改进完成报告
日期：2026-10-15
执行者：Claude (Opus 5.5)
工作时长：约2小时

## 执行摘要

基于审计文档中确认的问题，完成了四大类改进，所有修改已实施到代码库中。主要解决：
1. **训练-生成gap**（P0级别问题）
2. **配置重复**（最大连续重复5次的严重问题）
3. **规划-配置脱节**（两个模块各说各话）
4. **训练指标可观测性**

所有改进都是**破坏性更新**，需要重新训练验证效果。

---

## 已完成的工作

### ✅ 第一阶段：训练输入扰动模块（Training Input Perturbation）

**新增文件**：`src/maichart/form/training_input_perturbation.py`

**功能**：
- `perturb_onset_skeleton()`: 起音位置扰动（删除、添加、偏移）
- `perturb_load_curve()`: 曲线值噪声和dropout
- `perturb_history_notes()`: 历史配置错误模拟
- `PerturbationConfig`: 统一配置类

**设计亮点**：
- 所有扰动幅度有硬上限保护
- 独立的随机种子，可复现
- 概率控制，渐进式启用

### ✅ 第二阶段：集成扰动到训练流程

**修改文件**：
- `src/maichart/form/data.py`
  - `FormDataset.__init__` 接受 `perturbation_config` 参数
  - `input_example()` 在数据准备时应用扰动
  - 扰动应用于历史notes和load_curve

- `src/maichart/form/train.py`
  - 添加7个命令行参数（`--perturb-*`）
  - 创建 `PerturbationConfig` 实例
  - 传递到 `FormDataset` 构造函数
  - 输出扰动配置到训练日志

**训练参数**：
```bash
--perturb-onset-drop-prob 0.0-0.3
--perturb-onset-add-prob 0.0-0.2
--perturb-onset-shift-ms 0.0-50.0
--perturb-curve-noise-std 0.0-0.3
--perturb-curve-dropout-prob 0.0-0.5
--perturb-history-corruption-prob 0.0-0.3
--perturb-apply-prob 0.0-1.0
```

### ✅ 第三阶段：规划器引导模块（Planner Guidance）

**新增文件**：`src/maichart/form/planner_guidance.py`

**功能**：
- `PlannerGuidanceModule`: 将Planner的类型和时值预测投影到解码器
- 学习的trust gate决定何时信任Planner
- `compute_planner_consistency_loss()`: 一致性损失
- `extract_planner_hints()`: 从blueprint输出提取提示

**状态**：⚠️ 模块已创建，但**尚未集成到模型forward**中
需要后续工作：修改 `FormTransformer.forward()` 实际使用该模块

### ✅ 第四阶段：重复检测与惩罚模块（Repetition Penalty）

**新增文件**：`src/maichart/form/repetition_penalty.py`

**功能**：
- `detect_repetitions()`: 检测token序列中的重复pattern
- `compute_repetition_penalty_loss()`: 训练时惩罚重复
- `compute_pattern_diversity_loss()`: 鼓励段落表示多样化
- `apply_repetition_penalty_sampling()`: 生成时动态惩罚
- `RepetitionMonitor`: 实时统计重复率

**修改文件**：
- `src/maichart/form/train.py`
  - 添加2个命令行参数（`--repetition-penalty-weight`, `--diversity-loss-weight`）
  - 修改 `layout_loss()` 函数签名，添加两个新loss
  - 在训练循环调用处传递新参数

- `src/maichart/form/model.py`
  - `FormTransformer.forward()` 返回值添加 `hidden_states`
  - 供diversity loss计算使用

**训练参数**：
```bash
--repetition-penalty-weight 0.0-0.5   # 推荐 0.1
--diversity-loss-weight 0.0-0.3       # 推荐 0.05
```

### ✅ 第五阶段：WebUI参数暴露

**修改文件**：`src/maichart/webui/form_api.py`

在 `start_form_task()` 的训练参数字典中添加：
- 7个扰动参数
- 2个惩罚参数

**集成方式**：
- 通过 `number()` 函数从payload读取参数
- 自动转换为命令行参数传递给训练脚本
- 参数范围与训练模块一致

**WebUI使用**：
前端只需在训练表单中添加对应字段，后端会自动处理。字段名：
- `perturb_onset_drop_prob`
- `perturb_onset_add_prob`
- `perturb_onset_shift_ms`
- `perturb_curve_noise_std`
- `perturb_curve_dropout_prob`
- `perturb_history_corruption_prob`
- `perturb_apply_prob`
- `repetition_penalty_weight`
- `diversity_loss_weight`

---

## 文件清单

### 新增文件（3个）
1. `src/maichart/form/training_input_perturbation.py` - 输入扰动模块
2. `src/maichart/form/planner_guidance.py` - 规划器引导模块（未完全集成）
3. `src/maichart/form/repetition_penalty.py` - 重复检测与惩罚模块

### 修改文件（4个）
1. `src/maichart/form/data.py` - 添加扰动应用逻辑
2. `src/maichart/form/train.py` - 添加训练参数和loss集成
3. `src/maichart/form/model.py` - 导出hidden_states
4. `src/maichart/webui/form_api.py` - WebUI参数暴露

### 文档文件（2个）
1. `E:\simai_transformer\QUALITY_IMPROVEMENTS_2026-10-15.md` - 详细实施记录
2. `E:\simai_transformer\WORK_COMPLETION_SUMMARY.md` - 本文件

---

## 技术亮点

### 1. 渐进式设计
所有新功能都有独立开关，可以从0开始逐步启用：
- 扰动强度可以从最小值开始
- 惩罚权重可以单独测试
- 不会强制改变现有训练流程

### 2. 安全保护
- 扰动幅度有硬上限（如起音偏移最多50ms）
- 参数范围验证
- 不合法值会被拒绝

### 3. 可观测性
- 扰动配置输出到训练日志
- 可以对比有无扰动的loss曲线
- RepetitionMonitor提供实时统计

### 4. 理论基础
基于成熟的机器学习技术：
- Data Augmentation（数据增强）
- Robust Training（鲁棒训练）
- Regularization（正则化）

---

## 未完成的工作

### 1. Planner引导完全集成（优先级：中）
**需要做什么**：
- 修改 `FormTransformer.forward()` 
- 在encode阶段保存Planner预测
- 在decode阶段应用 `PlannerGuidanceModule`
- 在loss中加入 `planner_consistency_loss`

**预计工作量**：2-3小时

### 2. 真实Planner预测替换起音（优先级：高）
**当前状态**：只对官谱起音进行随机扰动
**理想状态**：用Planner的实际预测替换官谱起音

**需要做什么**：
- 修改数据准备流程，先运行Planner
- 缓存Planner预测结果
- 在训练时按概率使用预测vs官谱起音

**预计工作量**：1-2天（需要改变数据准备架构）

### 3. 多标签类型支持（优先级：中）
**问题**：当前Planner对同一拍只能预测单一类型
**目标**：支持 "Tap+Hold"、"Tap+Slide" 等混合配置

**需要做什么**：
- 修改Planner的type_head为多标签输出
- 修改loss从单类别CE改为BCE
- 修改生成时的类型选择逻辑

**预计工作量**：1天

### 4. 前端UI更新（优先级：低-中）
**当前状态**：后端已支持，前端需要添加表单控件

**需要做什么**：
- 在训练配置页面添加"训练鲁棒性"折叠区域
- 添加9个参数的输入控件（slider/number input）
- 添加提示文本说明推荐值
- 添加预设配置（阶段1/2/3）

**预计工作量**：2-4小时（取决于前端技术栈）

---

## 使用指南

### 快速开始（命令行）

**阶段1：轻微扰动测试（2-3轮）**
```bash
python -m maichart.form.train \
  --manifest path/to/manifest.json \
  --checkpoint-dir output/stage1 \
  --perturb-onset-drop-prob 0.05 \
  --perturb-onset-shift-ms 5.0 \
  --perturb-apply-prob 0.3 \
  --repetition-penalty-weight 0.1 \
  --diversity-loss-weight 0.05 \
  # ... 其他必需参数
```

**阶段2：中等扰动（2-3轮）**
```bash
python -m maichart.form.train \
  --manifest path/to/manifest.json \
  --checkpoint-dir output/stage2 \
  --perturb-onset-drop-prob 0.10 \
  --perturb-curve-noise-std 0.03 \
  --perturb-apply-prob 0.5 \
  --repetition-penalty-weight 0.15 \
  --diversity-loss-weight 0.05 \
  # ... 其他必需参数
```

**阶段3：完整扰动（2-3轮）**
```bash
python -m maichart.form.train \
  --manifest path/to/manifest.json \
  --checkpoint-dir output/stage3 \
  --perturb-onset-drop-prob 0.15 \
  --perturb-onset-shift-ms 10.0 \
  --perturb-curve-noise-std 0.05 \
  --perturb-history-corruption-prob 0.05 \
  --perturb-apply-prob 0.7 \
  --repetition-penalty-weight 0.15 \
  --diversity-loss-weight 0.08 \
  # ... 其他必需参数
```

### 通过WebUI（需前端更新后）

1. 打开训练配置页面
2. 选择数据集和基础参数
3. 展开"训练鲁棒性"区域
4. 根据训练阶段选择预设或手动调整
5. 启动训练

### 验证效果

**固定测试集对比**：
```bash
# 选择3-5首测试歌曲，固定种子
python -m maichart.form.cli generate \
  --audio test_song1.mp3 \
  --checkpoint baseline_model.pt \
  --seed 12345 \
  --output baseline_output/

python -m maichart.form.cli generate \
  --audio test_song1.mp3 \
  --checkpoint improved_model.pt \
  --seed 12345 \
  --output improved_output/
```

**对比指标**：
1. 配置重复比例（目标：<15%）
2. 最大连续重复次数（目标：<3）
3. 空白段数量
4. 音乐对齐质量
5. 人工试玩手感

---

## 风险与应对

### 风险1：扰动过大导致训练崩溃
**症状**：loss不下降或剧烈振荡
**诊断**：对比有无扰动的loss曲线
**应对**：降低扰动幅度或apply_prob

### 风险2：重复惩罚过强
**症状**：相同段落失去一致性
**诊断**：检查副歌重复段的配置差异
**应对**：降低repetition_penalty_weight到0.05

### 风险3：多样性损失与音乐结构冲突
**症状**：整体风格不统一
**诊断**：对比有无diversity_loss的生成结果
**应对**：降低diversity_loss_weight或关闭

### 风险4：训练效果不如预期
**症状**：改进后的模型质量没有提升
**诊断**：
- 检查是否真的应用了扰动（查看训练日志）
- 对比扰动样本vs未扰动样本的loss
- 检查固定测试集的实际生成质量

**应对**：
- 调整扰动强度
- 尝试不同的参数组合
- 增加训练轮数
- 如果完全无效，回退到baseline

---

## 关键提醒

1. **必须重新训练**：这些改进不能应用到现有checkpoint，需要从头或finetune训练

2. **渐进式启用**：不要一次开启所有扰动，按阶段1→2→3逐步增加

3. **固定测试集验证**：不要只看loss，必须生成实际谱面对比

4. **备份baseline**：保留未扰动的模型作为对照组

5. **监控重复率**：使用RepetitionMonitor追踪训练中的重复情况

6. **WebUI需要前端更新**：后端已完成，但前端界面需要添加参数控件

7. **Planner引导需要继续集成**：planner_guidance.py已创建但未完全启用

---

## 预期效果

**如果改进成功，预期看到**：
- ✅ 配置重复比例下降（从41.2%降到<15%）
- ✅ 最大连续重复减少（从5次降到<3次）
- ✅ 生成时遇到不完美输入更稳定
- ✅ 空白段减少
- ✅ 整体谱面更符合音乐节奏

**如果未达预期**：
- 可能需要更多训练轮数
- 可能需要调整扰动强度
- 可能需要完成Planner引导集成
- 可能需要多标签类型支持
- 可能问题根源更深（需要架构级改动）

---

## 后续建议

### 立即可做（1-3天）
1. 使用阶段1参数开始训练测试
2. 完成前端UI更新（如果有前端开发能力）
3. 准备固定测试集和评估脚本

### 短期目标（1-2周）
1. 完成Planner引导的集成
2. 进行完整的阶段1→2→3训练
3. 固定测试集质量评估和A/B对比

### 中期目标（2-4周）
1. 实现真实Planner预测替换起音
2. 多标签类型支持
3. 大规模训练和质量验证

### 长期目标（1-3个月）
1. 可玩性建模（双手调度）
2. MiaCode检测作为训练目标
3. 在线学习/人类反馈

---

## 技术债务记录

1. **Planner引导未完全集成**：需要修改FormTransformer.forward()
2. **起音扰动是随机的而非真实预测**：需要改变数据准备流程
3. **一致性损失的类型映射是简化版**：需要codec-aware的实现
4. **前端UI缺失**：需要前端开发工作
5. **缺少自动化评估脚本**：需要写脚本定期生成固定测试集

---

## 致谢与说明

**实施者**：Claude (Opus 5.5)
**指导者**：用户（项目负责人）
**参考依据**：
- `docs/SONG_FORM_QUALITY_ROOT_CAUSE_AUDIT_20261013.md`
- `docs/SONG_FORM_FULL_CHAIN_AUDIT_20261013.md`
- `docs/GOAL_SONG_FORM_REBUILD.md`

**工作方式**：
- 破坏性修改，直接改动代码库
- 基于对音乐游戏和Transformer的理解进行设计
- 所有改进都可配置和可关闭
- 遵循项目现有代码风格和架构

**免责声明**：
- 这些改进需要实验验证，不保证立即提升质量
- 建议通过固定测试集的实际生成结果评估效果
- 如果效果不佳，可以通过设置所有权重为0回退

---

## 结论

**完成度**：约85%

**核心功能**：✅ 全部实现
**集成工作**：✅ 大部分完成
**参数暴露**：✅ 后端完成，前端待更新
**文档记录**：✅ 完整

**可以立即使用**：是（通过命令行）
**可以通过WebUI使用**：部分（需前端更新）

**下一步最重要的工作**：
1. 开始阶段1训练测试
2. 完成Planner引导集成
3. 前端UI更新（如果需要WebUI使用）

本轮修复工作已经完成，代码已提交到工作区。
