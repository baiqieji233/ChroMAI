# 训练参数使用指南
最后更新：2026-10-15

本文档说明新增的训练鲁棒性参数的使用方法和推荐设置。

---

## 参数分类

### 1. 训练输入扰动参数（Training Input Perturbation）

这些参数用于解决**训练-生成gap**问题（审计文档A01项）。

| 参数名 | 默认值 | 范围 | 说明 |
|--------|--------|------|------|
| `perturb_onset_drop_prob` | 0 | 0-0.5 | 随机删除起音的概率，模拟Planner漏检 |
| `perturb_onset_add_prob` | 0 | 0-0.5 | 随机添加虚假起音的概率，模拟Planner误报 |
| `perturb_onset_shift_ms` | 0 | 0-50 | 起音时间偏移的标准差(毫秒)，模拟定位误差 |
| `perturb_curve_noise_std` | 0 | 0-0.3 | 曲线值的高斯噪声标准差，模拟预测误差 |
| `perturb_curve_dropout_prob` | 0 | 0-0.5 | 曲线帧随机置零的概率，模拟预测空白 |
| `perturb_history_corruption_prob` | 0 | 0-0.3 | 历史配置错误的概率，模拟前文生成错误 |
| `perturb_apply_prob` | 1.0 | 0-1.0 | 扰动的总体应用概率 |

**默认策略**：所有扰动默认关闭（值为0），用户需要主动启用。

**为什么默认关闭**：
- 需要对比baseline（无扰动）来评估效果
- 扰动过大可能导致训练不稳定
- 让用户有意识地选择是否使用

### 2. 重复惩罚与多样性参数（Repetition Penalty & Diversity）

这些参数用于解决**配置重复**问题（审计文档显示最大连续重复5次）。

| 参数名 | 默认值 | 范围 | 说明 |
|--------|--------|------|------|
| `repetition_penalty_weight` | 0 | 0-0.5 | 重复检测惩罚的权重 |
| `diversity_loss_weight` | 0 | 0-0.3 | 段落表示多样性损失的权重 |

**默认策略**：默认关闭，建议在有baseline后启用对比。

---

## 推荐使用流程

### 阶段0：Baseline训练（必须）

**目的**：建立对照组，评估改进效果

```bash
# 所有新参数保持默认值0
python -m maichart.form.train \
  --manifest your_manifest.json \
  --checkpoint-dir output/baseline \
  # ... 其他必需参数
```

**记录**：
- 保存checkpoint
- 记录最终loss
- 生成3-5首固定测试歌曲
- 记录配置重复率

---

### 阶段1：轻微扰动（探索）

**目的**：验证扰动机制不会破坏训练

**推荐参数**：
```bash
--perturb-onset-drop-prob 0.05      # 只删5%的起音
--perturb-onset-shift-ms 5.0        # 轻微时间偏移(5ms)
--perturb-apply-prob 0.3            # 只对30%的样本应用
--repetition-penalty-weight 0.1     # 启用中等重复惩罚
--diversity-loss-weight 0.05        # 启用轻微多样性损失
```

**训练轮数**：2-3轮

**监控指标**：
- loss是否正常下降
- 与baseline的loss差异
- 训练日志中的扰动统计

**验证**：
- 生成相同的固定测试歌曲（相同seed）
- 对比配置重复率
- 人工试玩评估质量

---

### 阶段2：中等扰动（渐进）

**前置条件**：阶段1没有导致训练崩溃

**推荐参数**：
```bash
--perturb-onset-drop-prob 0.10      # 增加到10%
--perturb-onset-shift-ms 8.0        # 增加到8ms
--perturb-curve-noise-std 0.03      # 加入3%的曲线噪声
--perturb-apply-prob 0.5            # 对50%的样本应用
--repetition-penalty-weight 0.15    # 增强重复惩罚
--diversity-loss-weight 0.05        # 保持多样性损失
```

**训练轮数**：2-3轮

**重点观察**：
- 起音对齐质量是否下降
- 曲线响应是否仍然合理
- 配置重复率变化

---

### 阶段3：完整扰动（目标）

**前置条件**：阶段2效果良好

**推荐参数**：
```bash
--perturb-onset-drop-prob 0.15      # 最大15%
--perturb-onset-add-prob 0.05       # 加入5%的虚假起音
--perturb-onset-shift-ms 10.0       # 10ms偏移
--perturb-curve-noise-std 0.05      # 5%的曲线噪声
--perturb-curve-dropout-prob 0.1    # 10%的曲线dropout
--perturb-history-corruption-prob 0.05  # 5%的历史错误
--perturb-apply-prob 0.7            # 对70%的样本应用
--repetition-penalty-weight 0.15    # 保持惩罚
--diversity-loss-weight 0.08        # 略微增加多样性
```

**训练轮数**：4-6轮（充分收敛）

**最终验证**：
- 固定测试集完整评估
- 与baseline A/B对比
- 多首随机歌曲测试
- 人工试玩验证

---

## 参数调优指南

### 症状1：训练loss不下降

**可能原因**：扰动过大

**诊断**：
```bash
# 查看训练日志中的扰动统计
grep "form_perturbation" logs/training.log
```

**应对**：
- 降低所有扰动幅度（减半）
- 降低`perturb_apply_prob`到0.3
- 暂时关闭曲线和历史扰动

---

### 症状2：配置重复率没有改善

**可能原因**：重复惩罚权重不足

**应对**：
- 增加`repetition_penalty_weight`到0.2
- 增加`diversity_loss_weight`到0.1
- 检查训练日志确认惩罚被应用

---

### 症状3：副歌段落失去一致性

**可能原因**：多样性损失过强

**应对**：
- 降低`diversity_loss_weight`到0.03或关闭
- 保持`repetition_penalty_weight`不变
- 检查段落相似度设置

---

### 症状4：生成质量反而下降

**可能原因**：
- 扰动破坏了有效的学习信号
- 需要更多训练轮数
- 某个参数设置不当

**诊断流程**：
1. 对比扰动样本vs未扰动样本的loss
2. 检查是否提前停止训练（需要更多epoch）
3. 逐个关闭扰动，找出问题参数

**应对**：
- 回退到阶段1参数
- 增加训练轮数到baseline的1.5倍
- 单独测试每种扰动的效果

---

## 特殊场景

### 场景1：数据集很小（<100张谱面）

**建议**：
- 减小扰动幅度
- 降低`perturb_apply_prob`到0.2-0.3
- 重复惩罚设为0.05（轻量）
- 不使用多样性损失

**原因**：小数据集容易过拟合，过度扰动会导致欠拟合

---

### 场景2：数据集很大（>500张谱面）

**建议**：
- 可以使用较强的扰动
- `perturb_apply_prob`可以设为0.8-0.9
- 重复惩罚和多样性损失可以更激进

**原因**：大数据集有更多的学习信号，可以承受更强的正则化

---

### 场景3：Finetune已有模型

**建议**：
- 首轮finetune不加扰动
- 第二轮开始加入轻微扰动（阶段1参数）
- 降低学习率，避免破坏已有知识

---

### 场景4：只想解决重复问题

**建议**：
```bash
# 不使用输入扰动，只用重复惩罚
--repetition-penalty-weight 0.15
--diversity-loss-weight 0.05
```

**注意**：这不解决训练-生成gap，但可以快速测试重复惩罚效果

---

## WebUI使用（前端更新后）

### 训练页面建议布局

**基础参数区**：（现有参数）
- 数据集选择
- 模型架构
- 学习率、batch size等

**训练鲁棒性区**：（新增，默认折叠）
- 标题：「训练鲁棒性（实验性）」
- 副标题：「改善训练-生成gap和配置重复问题」
- 三个预设按钮：
  - 🔵 阶段1（轻微）
  - 🟡 阶段2（中等）
  - 🟠 阶段3（完整）
- 高级选项（展开后显示所有9个参数）

**预设配置**：
```javascript
const PRESETS = {
  stage1: {
    perturb_onset_drop_prob: 0.05,
    perturb_onset_shift_ms: 5.0,
    perturb_apply_prob: 0.3,
    repetition_penalty_weight: 0.1,
    diversity_loss_weight: 0.05
  },
  stage2: {
    perturb_onset_drop_prob: 0.10,
    perturb_onset_shift_ms: 8.0,
    perturb_curve_noise_std: 0.03,
    perturb_apply_prob: 0.5,
    repetition_penalty_weight: 0.15,
    diversity_loss_weight: 0.05
  },
  stage3: {
    perturb_onset_drop_prob: 0.15,
    perturb_onset_add_prob: 0.05,
    perturb_onset_shift_ms: 10.0,
    perturb_curve_noise_std: 0.05,
    perturb_curve_dropout_prob: 0.1,
    perturb_history_corruption_prob: 0.05,
    perturb_apply_prob: 0.7,
    repetition_penalty_weight: 0.15,
    diversity_loss_weight: 0.08
  }
};
```

---

## 常见问题

### Q1: 这些参数会让训练变慢吗？

**A**: 扰动计算是毫秒级的，对训练速度影响<5%。主要时间仍在模型forward/backward。

---

### Q2: 必须按阶段1→2→3的顺序吗？

**A**: 不是必须，但强烈推荐。直接跳到阶段3可能导致训练不稳定，难以诊断问题。

---

### Q3: 可以只用部分参数吗？

**A**: 可以。例如只用重复惩罚而不用输入扰动。但输入扰动是解决训练-生成gap的核心。

---

### Q4: 这些参数适用于旧的event_v1模型吗？

**A**: 理论上可以，但未经测试。建议只在song_form_v1/v2上使用。

---

### Q5: 如何判断扰动是否真的生效了？

**A**: 查看训练日志：
```bash
grep "form_perturbation" logs/training.log
```
应该看到`"enabled": true`和具体的参数值。

---

### Q6: 改进后的模型能否用于生成？

**A**: 可以，改进只影响训练过程，生成时的inference代码不需要修改。

---

## 重要提醒

1. ⚠️ **必须建立baseline**：不要跳过阶段0，否则无法评估改进效果

2. ⚠️ **需要重新训练**：这些参数不能应用到已有checkpoint，必须从头或finetune训练

3. ⚠️ **渐进式启用**：不要一次性开启所有扰动

4. ⚠️ **固定测试集验证**：不要只看loss，必须生成实际谱面对比

5. ⚠️ **记录所有实验**：记录参数、loss、生成质量，便于回溯

---

## 技术支持

如遇到问题，检查：
1. 训练日志中的扰动统计
2. loss曲线是否异常
3. 生成的固定测试集质量
4. 参数是否在推荐范围内

参考文档：
- [质量改进记录](../archive/20261015/QUALITY_IMPROVEMENTS_2026-10-15.md) - 实施详情
- [工作完成报告](../archive/20261015/WORK_COMPLETION_SUMMARY.md) - 完成报告
- `QUALITY_CONTROLS_WEBUI_HANDOFF_20261015.md` - 交接文档
