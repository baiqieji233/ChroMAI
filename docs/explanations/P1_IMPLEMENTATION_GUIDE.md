# P1阶段实施指南：集成与训练

**基于Goal文档**: GOAL_COMPLEX_RHYTHM_PLAYABILITY_FINAL.md  
**完成状态**: P0完成，P1阶段说明  
**创建日期**: 2026-10-10

---

## 一、已完成的模块清单

### P0模块（已完成 7/7）
1. ✓ maimai_physics.py - 按键布局和手部分配
2. ✓ conflict_detector.py - 冲突检测（5类）
3. ✓ conflict_repair.py - 冲突修复（3种模式）
4. ✓ rhythm_analysis.py - 细粒度节奏特征提取
5. ✓ judgment_system.py - 判定系统完整建模
6. ✓ generation_controls_extended.py - WebUI配置
7. ✓ annotate_rhythm_patterns.py - 训练集标注脚本

### P1新增模块
8. ✓ post_process.py - 生成后处理集成
9. ✓ generation_integration.py - WebUI API集成

---

## 二、P1优先级工作流

### 阶段1：数据准备（1-2天）

#### 1.1 标注训练集（可选但推荐）
```bash
# 为full训练集添加节奏标注
python scripts/annotate_rhythm_patterns.py \
  --dataset webui_workspace/datasets/full训练集/manifest.json \
  --output webui_workspace/datasets/full训练集/rhythm_annotated_manifest.json \
  --summary
```

**输出**：
- 每个segment添加rhythm_analysis字段
- 统计摘要：各类节奏模式分布
- 高复杂度窗口标记

#### 1.2 筛选复杂节奏专项集（可选）
```python
# scripts/build_rhythm_subset.py
# 从full训练集筛选 fine_note_ratio > 0.3
# 生成专项manifest用于finetune
```

**目的**：
- Stage 2强化训练数据来源
- 针对性改进复杂节奏采样

---

### 阶段2：生成流程集成（2-3天）

#### 2.1 修改event_ctx_cli.py
```python
# 添加新参数
parser.add_argument("--rhythm-bias", default="auto", 
                   help="auto/uniform/triplet/dotted/mixed")
parser.add_argument("--fine-note-strength", type=float, default=0.5,
                   help="0-1, fine rhythm sampling strength")
parser.add_argument("--enable-conflict-detection", action="store_true",
                   help="Enable playability conflict detection")

# 集成后处理
from maichart.generation.generation_integration import integrate_generation_config
from maichart.generation.post_process import run_post_process

config = integrate_generation_config(base_config, {
    "rhythm_bias": args.rhythm_bias,
    "fine_note_strength": args.fine_note_strength,
    "enable_conflict_detection": args.enable_conflict_detection,
})

# 生成后运行后处理
result = run_post_process(notes_dict, config["post_process_config"])
```

#### 2.2 修改WebUI API (webui/server.py)
```python
def _start_generate(self, payload):
    # 提取新参数
    generation_params = {
        "rhythm_bias": payload.get("rhythm_bias", "auto"),
        "fine_note_strength": payload.get("fine_note_strength", 0.5),
        "enable_conflict_detection": payload.get("enable_conflict_detection", True),
        # ...
    }
    
    config = integrate_generation_config(base_config, generation_params)
    # ... 生成 ...
    result = run_post_process(notes_dict, config["post_process_config"])
```

#### 2.3 修改WebUI前端 (app.js)
```javascript
// 添加节奏倾向控制区（参考generation_controls_extended.py中的HTML）
// 添加冲突检测开关
// 修改生成API调用，传入新参数
```

---

### 阶段3：训练执行（3-7天）

#### 3.1 Stage 1: 基础全量训练（15 epochs）

**命令**：
```bash
python -m maichart.generation.event_ctx_cli train \
  --dataset webui_workspace/datasets/full训练集/manifest.json \
  --output checkpoints/complex_rhythm_v1 \
  --fresh \
  --planner-profile music_structure_v2 \
  --difficulty-range 4 6 \
  --level-range 12.0 15.0 \
  --epochs 15 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --save-every 2
```

**预期结果**：
- checkpoint存储在checkpoints/complex_rhythm_v1/
- 每2 epochs保存一次checkpoint
- loss稳定下降
- token accuracy持续提升

**监控指标**：
- Loss: 应从高值逐步下降（通常1.0->0.3）
- Token Accuracy: 应从低值逐步提升
- 显存占用：应稳定在一定范围
- 训练时间：估计2-3天（RTX 4090）

#### 3.2 中间评估（1天）
生成Stage 1 checkpoint的测试结果：
```bash
python -m maichart.generation.event_ctx_cli generate \
  --checkpoint checkpoints/complex_rhythm_v1/epoch_015.pt \
  --audio test_song.mp3 \
  --output test_output.txt \
  --rhythm-bias auto \
  --fine-note-strength 0.5
```

**评估项**：
- 细碎音符比例：与baseline对比
- 冲突检出率：应 < 1%
- 人工试玩：难度曲线、可玩性

**决策**：
- ✓ 质量提升 → 进入Stage 2
- ✗ 质量下降 → 分析原因，调整Stage 2参数或延长Stage 1

#### 3.3 Stage 2: 复杂节奏强化训练（5-10 epochs）

**前置**：建立专项训练集（见1.2）

**命令**：
```bash
python -m maichart.generation.event_ctx_cli train \
  --dataset webui_workspace/datasets/full训练集/manifest.json \
  --init-checkpoint checkpoints/complex_rhythm_v1/epoch_015.pt \
  --output checkpoints/complex_rhythm_v1_fine \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --oversample-fine-rhythm 2.5 \
  --save-every 1
```

**参数说明**：
- --init-checkpoint: 加载Stage 1的最佳epoch
- --oversample-fine-rhythm: 复杂节奏样本过采样倍率
- learning-rate: 降低50%，避免过度调整

**预期结果**：
- 细碎节奏采样提升30-50%
- 总体loss可能略微上升（正常，因为数据分布变化）
- token accuracy维持或略有下降（正常）

---

## 三、关键决策点

### Q1: 是否需要标注训练集？
**推荐**：可选，但推荐
- **支持标注**：提供更精确的训练信号
- **不标注**：依赖模型自学习，可能较慢但风险低

### Q2: 是否启用所有冲突检测？
**推荐**：逐步启用
1. 仅启用硬冲突（detect_hard_conflicts=true, detect_soft_conflicts=false）
2. 验证硬冲突检出率 < 1%
3. 可选：加入软冲突检测

### Q3: 冲突修复模式如何选择？
**推荐**：
- 开发阶段：none（仅检测）
- 测试阶段：local（局部修复）
- 生产阶段：none或local（reject可能过于严格）

### Q4: Stage 1和Stage 2需要多长？
**推荐**：
- Stage 1: 15 epochs（基础学习）
- Stage 2: 5-10 epochs（微调）
- 总耗时：2-3周（含等待和评估）

---

## 四、故障排查

### 问题1：训练loss下降缓慢或停滞
**检查**：
- 学习率是否太低/太高
- 数据集是否正确加载
- 显存是否充足

**解决**：
- 调整learning_rate（±50%）
- 检查manifest.json和数据文件
- 查看nvidia-smi确认显存

### 问题2：冲突检测触发过于频繁
**原因**：
- 约束条件过严（软冲突阈值）
- 训练数据本身包含约束违反

**解决**：
- 关闭soft冲突检测
- 提高jump_distance和min_jump_time
- 使用"none"修复模式分析问题

### 问题3：生成结果质量下降
**原因**：
- Stage 2过采样比例过高
- 学习率设置不当
- 专项训练集过于单一

**解决**：
- 降低oversample倍率（2.5 → 1.5）
- 检查专项集构成（是否包含非音乐片段）
- 延长Stage 1或使用不同的学习率曲线

---

## 五、成功标准

### MVP验收
- [ ] 生成流程成功集成节奏参数
- [ ] WebUI显示新控件并传参正确
- [ ] Stage 1训练收敛，loss > 0

### 完整验收
- [ ] 细碎节奏采样率提升 ≥ 30%（相比baseline）
- [ ] 硬冲突率 < 1%
- [ ] 人工可玩性评分提升 ≥ 0.5分（5点量表）
- [ ] 复杂节奏段落质量明显改善

---

## 六、临时粗糙边界（可接受）

这些问题可以在后续优化：
1. post_process.py中的note字典->events转换简化
2. WebUI前端HTML需要实际集成（当前只有模板）
3. 训练命令需要验证against实际CLI接口
4. 冲突修复后的notes_dict重构尚未实现

---

## 七、下一步行动清单

### 立即可执行
- [ ] 运行annotate_rhythm_patterns.py标注示例数据
- [ ] 修改event_ctx_cli.py添加新参数
- [ ] 在webui/server.py集成generation_integration.py
- [ ] 测试端到端流程

### 短期（本周）
- [ ] 准备Stage 1训练数据和checkpoint
- [ ] 启动Stage 1训练（15 epochs）
- [ ] 监控训练指标

### 中期（1-2周）
- [ ] 评估Stage 1结果
- [ ] 构建复杂节奏专项集（如需）
- [ ] 启动Stage 2训练

### 长期（2-3周）
- [ ] 完成全部训练和评估
- [ ] 生成最终质量报告
- [ ] 考虑P2优化项

---

## 八、参考资源

- Goal文档: `docs/explanations/GOAL_COMPLEX_RHYTHM_PLAYABILITY_FINAL.md`
- 实施总结: `docs/explanations/IMPLEMENTATION_PROGRESS.md`
- 验证报告: `docs/explanations/GOAL_VERIFICATION.md`
- 模块代码: `src/maichart/generation/*.py`

