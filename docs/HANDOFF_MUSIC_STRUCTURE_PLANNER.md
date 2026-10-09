# 交接文档：音乐结构、Planner 与 Null Condition 改造

> 项目：`E:/simai_transformer`
> 当前日期：2026-10-05
> 文档用途：用户准备切换运行环境，供下一位智能体继续当前任务。
> 当前任务未完成，不要据此启动正式训练。

## 1. 当前任务目标

在现有 `event_ctx` / event Transformer 框架上继续改造：

1. 处理变 BPM 和拍位问题，使音频短时节奏特征以完整 tempo map 对齐，但不要求自动 BPM/变速检测完全准确；目标是拍位基本正确、谱面时序稳定。
2. 增加窗口级谱面规划器（planner），先预测窗口级谱面意图，再用事件 decoder 生成具体 Simai event token。
3. 增加 planner 辅助监督，减少模型只靠 token CE 学习导致的稀疏、窗口平坦和尾部失控。
4. 缺省条件使用显式 null embedding，而不是用统一 `0.5` 伪装“无条件”。
5. 同步 checkpoint contract、训练/生成链路和 WebUI。
6. 只做代码和 smoke 验证；用户明确要求在真实训练/真实生成前暂停。

## 2. 已完成并通过验证的既有修复

- 事件模型条件由逐帧常量加法改为 encoder memory condition token。
- `max_audio_tokens` 默认从 512 提升到 1024。
- 训练数据支持难度槽与定数范围筛选：
  - `min_difficulty_index` / `max_difficulty_index`
  - `min_level` / `max_level`
- 筛选范围、条件参数和模型参数进入 dataset signature / checkpoint data contract。
- 训练时 condition dropout 保留 level，避免整条条件向量被清零。
- 训练和生成均有窗口级 intensity follow；但它不是最终解决方案，真正的 planner/辅助监督仍需实验验证。
- 生成 merge 后再次进行 audible span 硬过滤。
- 修复 CFG null 分支、temperature/top-p 顺序、overlap stride、候选评分和候选 rank。
- WebUI 已增加训练/生成模式页签。
- WebUI 已增加训练筛选控件：难度槽范围、定数范围、`maxAudioTokens`、`conditionDefault`。
- WebUI 参数已经通过 `/api/train` 转发到训练 CLI。
- 现有全量回归在前一阶段达到：`594 passed, 11 skipped`。
- WebUI 专项和 intensity/bounds 测试此前达到：`68 passed`。

相关文档：

- [TRAINING_GENERATION_FIX_PROGRESS.md](TRAINING_GENERATION_FIX_PROGRESS.md)
- [WEBUI_FIX_PROGRESS.md](WEBUI_FIX_PROGRESS.md)
- [STRUCTURE_AUDIT.md](STRUCTURE_AUDIT.md)
- [HANDOFF_TRAINING_GENERATION_REVIEW.md](HANDOFF_TRAINING_GENERATION_REVIEW.md)

## 3. 当前轮已经修改的代码

### 3.1 变 BPM / 节奏分析

文件：`src/maichart/generation/event_audio.py`

已经存在并继续使用的正确基础：

- `audio_segment_tensor()` 使用 `tempo_changes` 积分 beat phase。
- `_timing_context()` 计算 local BPM、beat phase、bar phase、phrase phase。
- 变 BPM 情况不能把固定秒窗口直接等同固定拍长。

本轮新增了 `_beat_normalized_rhythm()`，用于按当前局部 beat duration 计算：

- `beat_onset_local`
- `beat_onset_half`
- `beat_flux_local`
- `beat_rms_local`

重要设计决定：

- 舞萌密度仍然以绝对时间计算，即 `notes / second`。
- beat-normalized rhythm 只用于节奏形态分析，不能取代绝对 NPS。
- 为避免一次性破坏当前 108 维输入合同，新 rhythm 通道目前没有拼进 event model 输入；相关分析函数保留为后续结构特征实验基础。
- 不要把自动变 BPM 当成绝对真值。当前目标是 tempo map/拍位大致正确；若自动变速检测不可靠，应优先保持稳定全局 BPM 或显式 tempo map。

### 3.2 Window planner

文件：

- `src/maichart/generation/event_model.py`
- `src/maichart/training/event_dataset.py`
- `src/maichart/training/event_ctx_dataset.py`
- `src/maichart/training/train_events.py`

已加入：

- `EventTransformerConfig.plan_dim = 9`
- `EventConditionedTransformer.plan_head`
- `forward(..., return_plan=True)` 返回 `(logits, plan_pred)`；默认仍只返回 logits。
- dataset item 增加 `plan_target`，当前 6 维目标含义；模型 planner 输出为 9 维：前 5 维连续回归，后 4 维配置类别 logits：
  1. 窗口事件数量的归一化代理（count / 32）
  2. **绝对秒密度**的归一化代理（notes / second / 16）
  3. 和弦比例
  4. Slide 比例
  5. 最后事件在窗口中的相对位置
  6. 粗粒度配置类别（plain/chord/slide/touch）
- `train_events.run_epoch()` 在 token CE 之外增加 planner MSE，当前暂定权重 `0.1`。

重要：planner 仍不是独立两阶段生成器，但当前 `plan_condition` 已把预测的连续规划和配置 logits 注入 decoder memory，使窗口意图参与具体事件 token 生成。

### 3.3 Null embedding

文件：`src/maichart/generation/event_model.py`、数据集和训练循环。

已加入：

- `condition_null_embedding` 参数。
- 模型 `forward` 接受 `condition_null`。
- encoder condition token 在 `condition_null=True` 时叠加 null embedding。
- dataset 计划输出 `condition_null`；collate 和训练 forward 正在接通。
- 旧 checkpoint 缺少 `condition_null_embedding`、planner head 参数时，compat loader 允许缺失并使用当前模型初始化值。

当前设计意图：

- “缺省 style 条件”不是把每个指标填为统一 0.5。
- 数值默认向量与 null embedding 分开：默认向量可作为语料中位数，null embedding 表示“用户未提供该组条件”。
- 生成端默认没有显式 null flag 时为 false；若要真正使用 CFG/null 语义，需继续把 null flag 沿 `_decode`/`_decode_windows`/WebUI contract 传递。

## 4. 当前已知未完成/高风险问题

### P0：先修训练 planner loss 参数 bug

最近加入了 CLI：

```text
--plan-loss-weight
```

`run_epoch()` 已增加 `plan_loss_weight` 参数，但代码中仍有一处需要检查：训练循环里曾使用 `args.plan_loss_weight`，而 `run_epoch()` 内没有 `args` 变量。应改成：

```python
loss = loss + float(plan_loss_weight) * plan_loss
```

并在训练和验证调用处显式传入：

```python
plan_loss_weight=args.plan_loss_weight
```

否则正式训练会在进入 planner loss 路径时 `NameError`。

### P0：event_audio 输入维度刚刚修复过，必须先全量测试

本轮中 rhythm channel 曾临时加入 `FEATURE_KEYS`，导致输入从 108 变成 112，破坏已有 phase/shape 测试。后来已恢复主输入为 108，并保留 rhythm 分析但不拼入模型输入。

恢复后的目标：

```text
INPUT_DIM = 108
```

必须先运行：

```text
python -m pytest -q tests/test_event_audio.py tests/test_event_pipeline.py tests/test_event_model.py tests/test_event_ctx.py tests/test_training_v25_smoke.py
```

如果再次出现 112 维，优先检查：

- `FEATURE_KEYS` 是否包含 `RHYTHM_CONTEXT_KEYS`
- `audio_segment_tensor()` 是否 concatenate rhythm
- `compress_event_audio()` 空音频分支是否 concatenate rhythm
- `EventTransformerConfig.input_dim` 是否被误改成 112

### P0：condition null 仍需完整贯通

当前需要继续确认：

- `event_dataset.py` 的 `condition_null` 是否出现在 plain/context item。
- `collate_events()` 是否输出 batch `condition_null`。
- `train_events.run_epoch()` 是否传 `condition_null=batch["condition_null"]`。
- `EventConditionedTransformer.forward()` 是否将 `condition_null` 传给 `encode_audio()`。
- 生成端 `_decode()` 的 conditional/null branch 是否明确传 `condition_null=False/True`。
- `condition_params` contract 是否记录 null policy/version。
- `event_checkpoints.compare_event_contract()` 是否需要新增 null policy key。

### P1：planner target/训练接口

必须检查：

- EventCtxDataset 使用 `_window_plan(ref, start, end)`，不能传多余 sample_index。
- planner target 的密度必须是绝对秒密度，不是 beat density。
- planner loss 对空窗口/极稀疏窗口不能产生异常值。
- planner loss 只作为辅助，默认权重建议 `0.05–0.1`，不要一开始设太大。
- planner head 新参数需要写入 model config，旧 checkpoint compat 允许缺失。
- 加一个 smoke 测试确认 `return_plan=True` 返回 `[B,5]`。

## 5. 关于变 BPM 的最终原则

舞萌谱面的实际密度定义是：

```text
notes per second
```

而不是 notes per beat。速度变化时，同样的每拍音符数会对应不同实际密度，因此 planner target 必须以 tempo map 积分出的窗口秒长计算。

同时，音频结构分析需要两套坐标：

1. **绝对时间坐标**：用于真实谱面密度、note/second、首尾音频边界、可听区检查。
2. **完整 tempo map 的 beat 坐标**：用于 beat phase、bar/phrase phase、局部节奏形态和拍位稳定性。

不能用任意一种坐标替代另一种。

自动变 BPM 检测不需要完全准确。优先级应该是：

1. 用户显式 tempo map；
2. 训练 chart 自带 tempo changes；
3. 稳定 global BPM + phase；
4. 自动局部 tempo 仅在置信度足够时采用。

## 6. 当前训练建议（尚未执行）

用户目前只考虑 full 训练集，不打算纳入低难度谱面。应保持：

```text
训练难度槽范围：4–6
训练定数范围：12–15
```

在 planner/null 改动稳定前，不要长训。建议下一环境先做：

1. full 数据集短跑 1 epoch，独立 checkpoint 目录。
2. 检查 dataset audit、planner loss、token loss、显存和 checkpoint 保存。
3. 固定同一音频，对比不同 level/style/null 条件。
4. 再决定是否长训。

不要用旧 checkpoint resume 新 planner/null contract；使用 fresh。旧 checkpoint 只用于兼容加载测试，不是新模型的训练起点。

## 7. 当前环境问题

当前环境在本轮多次出现源码文本/编码污染风险，尤其是 WebUI `app.js` 曾被 PowerShell ANSI/UTF-8 读写破坏。已经重建为可解析的精简实现并通过 WebUI 测试，但切换环境后仍应先执行：

```text
node --check src/maichart/webui/static/app.js
python -m py_compile src/maichart/generation/event_audio.py src/maichart/generation/event_model.py src/maichart/training/event_dataset.py src/maichart/training/event_ctx_dataset.py src/maichart/training/train_events.py
python -m pytest -q
```

不要使用 PowerShell `Get-Content -Raw` / `Set-Content` 写 Python/JS 源码；优先使用 harness 的 `read` / `edit` / `write`。特别注意当前某些中文字符串可能已经被替换为乱码，功能优先但不要再做整文件编码转换。

## 8. 下一步最小执行清单

1. 修复 `plan_loss_weight` 的 `args` 引用。
2. 完整贯通 `condition_null` 并加入 contract。
3. 运行 event_audio/model/dataset/training smoke。
4. 运行全量 pytest。
5. 仅在用户确认后启动 full 训练短跑。

