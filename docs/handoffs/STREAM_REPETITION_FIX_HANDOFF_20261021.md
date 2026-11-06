# `chart_stream_v1` 训练期快速复读退化修复交接文档

> 后续独立审计发现本文部分因果归因证据不足，且其中字符级时值/局部换键等策略会误伤正常手法。实际修复和 v14 合同以 `STREAM_CAUSAL_MOTOR_FIX_HANDOFF_20261021.md` 为准。本文保留为当时的调查记录，不代表所有推断已获验证。

日期：2026-10-21
关联模型：`Prism_stream_v15`（`chart_stream_v1`，`causal_audio_v1` + `overlap_v1`）
对比测试输出：
- 倒数第三轮（无严重复读，Epoch 2 / 4,764 optimizer steps）：[`webui_workspace/generated/81a4b6cb85b8/maidata.txt`](../../webui_workspace/generated/81a4b6cb85b8/maidata.txt)
- 近期倒数第二轮（Epoch 5 / 11,910 optimizer steps）：[`webui_workspace/generated/17195d1139df/maidata.txt`](../../webui_workspace/generated/17195d1139df/maidata.txt)
- 近期最新一轮（Epoch 6 / 14,292 optimizer steps）：[`webui_workspace/generated/4e7dc1cc89b5/maidata.txt`](../../webui_workspace/generated/4e7dc1cc89b5/maidata.txt)

---

## 1. 背景与五维复读现象汇总

在相同生成参数（`temperature=0.9`, `position_temperature=0.3`, `style_balance_strength=1.0`, `local_chord_strength=1.0`, `rhythm_diversity_strength=0.6`, `position_interval_strength=1.0`）下，`Prism_stream_v15` 仅从 Epoch 2 训练至 Epoch 5–6 即在五个维度上出现显著复读退化：

| 复读维度 | Epoch 2 (`81a4b6cb85b8`) | Epoch 5 (`17195d1139df`) | Epoch 6 (`4e7dc1cc89b5`) |
| --- | --- | --- | --- |
| **1. 节奏间隔锁死（`1/2` 拍 `{8}` 连续流）** | 最长连续 `1/2` 拍仅 **15** 音 | 最长连续 `1/2` 拍达 **294** 音（60s–196s） | 最长连续 `1/2` 拍达 **260** 音（14.9s–138.6s） |
| **2. 双押—单键周期 2 交替震荡** | 正常短交替 | 连续 **32** 事件严格 `双押, 单键, 双押, 单键` | 连续 **42** 事件严格 `1/7, 3, 2/4, 3, 2/4...` |
| **3. 连续纯 Touch 洪流** | 最长连续 Touch 仅 **1** 音 | 正常零散 Touch | 连续 **33** 个 `E` 区 Touch 音符（`E8, E1, E2...`） |
| **4. 时值括号 `[a:b]` 垄断** | `[8:1]` 占 50.0%，`[16:3]` 占 22.9% | `[8:1]` 占 **86.3%**，最长连续 **52** 次 | `[16:3]` 占 **88.3%**，最长连续 **61** 次 |
| **5. 键位局部子集锁死（`{3,4,5,6}` 底部四键）** | 各键位分布均衡 | 局部双键反复横跳 | `{3,4,5,6}` 占全曲按键 **64.7%**，多处 14–16 音死锁于 `{3,4,5,6}` |

---

## 2. 根因定位与代码级修复详情

### 修复 1：切断 `plan_head -> intent_projection` 的 Token CE 反向梯度劫持，并保护负值间隔预测
- **涉及文件**：
  - [`src/maichart/stream/model.py`](../../src/maichart/stream/model.py#L211-L218)
  - [`src/maichart/stream/timing.py`](../../src/maichart/stream/timing.py#L20-L33)
- **根因分析**：
  1. 在 [`ChartStreamTransformer.decode`](../../src/maichart/stream/model.py#L211-L217) 中，`plan = self.plan_head(hidden)` 原本只应在事件边界 token（`BOS` / `<end>`）上受 `plan_labels`（权重 `0.15`）监督；但随后 `intent = plan.gather(...)` **未做 `.detach()`** 就直接送入 `hidden = hidden + self.grounding.intent_projection(intent)`。
  2. 相比同文件第 288 行对密度曲线显式执行了 `curve.detach()`，`plan` 未截断梯度导致**每个解码步的 Token CE 损失（权重 `1.0`）都会通过 `intent_projection` 反向更新 `plan_head`**。随训练推进（Epoch 2 -> Epoch 6），`intent_projection` 范数增长 `2.46x`，Token CE 梯度将 `plan_head` 劫持为普通隐状态投影层，把 `plan[..., 0]`（预测下一事件 `log1p` 间隔）在 **62.1%** 的事件上拉成负值（均值 `-0.089`，最小值 `-0.871`）。
  3. 在推理端 [`interval_consistency_cost`](../../src/maichart/stream/timing.py#L27-L30) 中，当 `prediction < 0` 时，旧代码执行 `target = max(0., min(1., float(prediction))) * 8. = 0.0`（即误以为模型预测了 `0` 秒同时音），导致 `1` 拍间隔被罚 `0.453` 分，而 `1/2` 拍仅被罚 `0.211` 分，系统性将拍位搜索锁死在 `1/2` 拍（`{8}`）连续流中。
- **修复内容**：
  1. 在 [`src/maichart/stream/model.py`](../../src/maichart/stream/model.py#L216) 中将 `intent = plan.gather(...)` 改为 `intent = plan.detach().gather(...)`。前向数值 100% 不变，`intent_projection` 仍正常从 Token CE 学习如何利用 `plan` 特征，而 `plan_head` 只受 `plan_labels` 监督，彻底消除梯度劫持。
  2. 在 [`src/maichart/stream/timing.py`](../../src/maichart/stream/timing.py#L27) 中增加 `float(prediction) < 0.` 保护：当已有 checkpoint 的 `plan[..., 0]` 出现负值漂移时直接返回 `0.`，不再将其截断为 `0.0` 拍目标去惩罚正常拍距。

---

### 修复 2：消除 `kinematic_rows` 在遮罩事件上的间隔回声泄露与伪造 Touch 标志，并引入连续 Touch 衰减
- **涉及文件**：
  - [`src/maichart/stream/symmetry.py`](../../src/maichart/stream/symmetry.py#L193-L280)
  - [`src/maichart/stream/data.py`](../../src/maichart/stream/data.py#L605-L614)
- **根因分析**：
  1. 在 [`src/maichart/stream/data.py`](../../src/maichart/stream/data.py#L606-L611) 中，被 `history_note_dropout` / `prefix_note_dropout` 遮罩的事件被置为 `lanes[event] = ()`。
  2. 在 [`kinematic_rows`](../../src/maichart/stream/symmetry.py#L237-L263) 中：
     - `values[16]`（`log1p(now - event_seconds[count - 1])`）与 `values[17]`（前一事件间隔）**完全未随遮罩清零**，导致模型绕过 `masked_clock_inputs`（`clock[..., 15:17]`），直接从 `kinematics[16:18]` 抄写上一拍间隔（`Prism_stream_v15` 到 Epoch 6 时 `kinematic_projection` 第 16 列范数高达 `3.414`，居全模型之首）。
     - 同时，`lanes[event] = ()` 触发了 `values[23] = float(not last) = 1.0`（**纯 Touch 标志位**），使大量被遮罩的普通按键事件在训练时激活了第 23 列（范数升至 `2.267`）。而在推理时，只有真正的纯 Touch 音符才会令 `last == ()` 激活第 23 列，并且连续 Touch 期间 `values[21]`（`run`）恒为 `0.0`、`values[23]` 恒为 `1.0`，形成自激死循环（连续 33 个 `E` 区 Touch 音符）。
- **修复内容**：
  1. [`kinematic_rows`](../../src/maichart/stream/symmetry.py#L193) 新增可选参数 `masked_events=None`，并在 [`src/maichart/stream/data.py`](../../src/maichart/stream/data.py#L606-L614) 中传入 `history_hidden` 与 `prefix_hidden` 对应的事件索引集合。
  2. 当 `count - 1` 属于 `masked_events` 时，将 `values[16] = 0.`、`values[17] = 0.`、`values[23] = 0.`（遮罩事件为未知事件，既不泄露时钟间隔，也不伪装成纯 Touch 音符）；当 `count - 2` 属于 `masked_events` 时同样将 `values[17] = 0.`。
  3. 对真实未遮罩的纯 Touch 事件（`not last and not last_masked`），统计连续纯 Touch 长度 `touch_run`，令 `values[23] = 1. / max(1, touch_run)`：单次 Touch 仍保持 `1.0`（100% 兼容所有单点测试与既有语义），连续 Touch 则按 `1, 1/2, 1/3, ...` 单调衰减，打破推理期固定点吸引子。

---

### 修复 3：补齐 `cycle_exit_targets` 三大结构盲区，并采用有界密度补偿的 `cycle_exit` 归一化
- **涉及文件**：
  - [`src/maichart/stream/continuation.py`](../../src/maichart/stream/continuation.py#L39-L104)
  - [`src/maichart/stream/train.py`](../../src/maichart/stream/train.py#L145-L152)
  - [`tests/test_stream_training_contract.py`](../../tests/test_stream_training_contract.py#L134-L143)
- **根因分析**：
  1. 在 [`cycle_exit_targets`](../../src/maichart/stream/continuation.py#L50-L98) 中：
     - 构造 `shapes` 时排除了 `'duration'` 类 token（`[16:3]`, `[8:1]`, `[4:1]`），导致不同键位上连续重复相同时值括号（如 `1-5[16:3]`, `2-6[16:3]`）后切换时值（如 `3-7[8:1]`）时，时值切换 token **从未被标记为 `cycle_exit`**。
     - 纯 Touch 序列（如 `C`, `E2`, `E4f`）因字符长度不同导致 `shapes` 不相等，且 `lanes` 全为空元组 `()`（被 `all(lanes[i-period:i])` 过滤），导致从连续 Touch 簇退出回普通按键的首个键位 token 漏标。
     - 当连续 6 个以上按键事件被困在 `<= 4` 个键位的局部子集（如 `{3,4,5,6}`）内无固定周期振荡、随后跳出该子集时，跳出子集的键位 token 漏标。
  2. 在 [`src/maichart/stream/train.py`](../../src/maichart/stream/train.py#L149) 中，上一轮将分母从 `exit_mask.sum()` 直接改为 `mask.sum()`，虽然消除了单 token 稀疏批次 `O(T)` 爆炸，但使每个退出 token 的附加梯度固定只有 `+0.25x`（相对均值缩小 `13.6x`），导致 `Prism_stream_v15` 的 `cycle_exit_accuracy` 跌至 `43.76%`。
- **修复内容**：
  1. 在 [`cycle_exit_targets`](../../src/maichart/stream/continuation.py#L40-L104) 中新增三项真实原谱退出标记（不伪造任何负样本）：
     - **时值括号循环退出**：当 `durations[i-2*period:i-period] == durations[i-period:i]` 非空且 `durations[i] != durations[i-period]` 时，标记当前事件首个分叉的 `'duration'` token。
     - **纯 Touch 簇退出**：当 `i >= 2` 且前两事件均为纯 Touch（`not lanes[i-1] and not lanes[i-2]`）、当前事件回到按键（`lanes[i]` 非空）时，标记首个 `'lane'` token。
     - **受限键位邻域退出**：当连续 6 个按键事件的并集 `confined` 满足 `1 <= len(confined) <= 4` 且当前事件首次跳出 `confined` 时，标记首个落在该子集外的 `'lane'` token。
  2. 在 [`src/maichart/stream/train.py`](../../src/maichart/stream/train.py#L149-L150) 中采用有界密度补偿分母：
     ```python
     exit_denom = torch.maximum(exit_mask.sum().float(), mask.sum().float() * .25).clamp_min(1.)
     cycle_exit = (ce * exit_mask * local * quality).sum() / exit_denom
     ```
     单 token 附加梯度倍率严格有界于 `[1.0, 4.0] * cycle_exit_weight`（绝不随序列长度 `T` 无界放大），同时在真实稀疏退出密度（`~7–10%`）下提供 **4 倍于原分母** 的退出监督力度。

---

### 修复 4：`rhythm_cycle_cost` 对超长均匀分音锁死（>48 拍差）施加平滑软抑制
- **涉及文件**：
  - [`src/maichart/stream/timing.py`](../../src/maichart/stream/timing.py#L84-L95)
- **根因分析**：
  - [`rhythm_cycle_cost`](../../src/maichart/stream/timing.py#L70) 对 `len(set(cycle)) <= 1` 的均匀分音流无条件永久豁免（`0.0` 成本），因此在模型间隔预测发生偏移时，连续 260–294 个 `1/2` 拍事件不会受到任何节奏多样性阻尼。
- **修复内容**：
  - 统计尾部连续相同拍差的长度 `steady_run`：**48 个连续相同间隔以内（例如整整 6 小节 8 分音符或 3 小节 16 分音符常规流）仍严格保持 `0.0` 成本**；仅当 `steady_run > 48` 时，对继续维持同一拍差的候选施加平滑有界软成本 `min(2.5, 0.08 * (steady_run - 48)) * strength`，在不伤及正常跑图段落的前提下阻断数百音的单一分音锁死。

---

### 修复 5：推理端 `TypeGuidance` 与 `LocalChordGuard` 周期 2 双押震荡、Touch 洪流、时值垄断与键位子集锁死阻尼，以及 WebUI `fresh` 模式重置修复
- **涉及文件**：
  - [`src/maichart/form/local_load.py`](../../src/maichart/form/local_load.py#L117-L127)
  - [`src/maichart/form/type_guidance.py`](../../src/maichart/form/type_guidance.py#L150-L185)（及 `event`、`commit`、`replay`、`token_bias`）
  - [`src/maichart/webui/static/app.js`](../../src/maichart/webui/static/app.js#L2176)（及第 3139 行）
- **根因分析**：
  1. **周期 2 `双押—单键` 极限环震荡**：`TypeGuidance.recent_chord_run` 一遇单键即归零，因此 `双押, 单键, 双押, 单键`（50% 双押率）永远达不到 `recent_chord_run >= 4`；同时 `LocalChordGuard` 在窗口双押率刚越过 `local_ceiling` 时压出单键、单键后又回落到阈值下方放行双押，形成周期 2 乒乓极限环（连续 32–42 事件）。
  2. **纯 Touch 洪流、时值括号 `[16:3]` 垄断、键位子集 `{3,4,5,6}` 锁死**：`TypeGuidance` 此前未追踪 `recent_touch_run`、`recent_duration_run` 与近期按键集合 `recent_button_lanes`。
  3. **WebUI `fresh` 模式配置残留**：[`applyChartStreamCheckpointConfig`](../../src/maichart/webui/static/app.js#L3139) 在 `mode === 'fresh'` 时未回填 `STREAM_GROUNDING_FIELDS` 的默认 `fallback`（`'auto'`），且 `updateTrainingControls` 在隔离测试上下文中未做 `typeof STREAM_GROUNDING_FIELDS !== "undefined"` 防御。
- **修复内容**：
  1. [`LocalChordGuard.event`](../../src/maichart/form/local_load.py#L120-L125)：检测滑动窗口尾部连续 `(双押, 单键)` 交替对数 `alt_pairs`，当 `alt_pairs >= 3`（即已连续 6 事件严格交替且下一拍正处于周期 2 的双押相）时，追加有界压力 `min(2.0, 0.45 * (alt_pairs - 2))`，打破乒乓极限环。
  2. [`TypeGuidance`](../../src/maichart/form/type_guidance.py#L259-L545)：
     - 追踪 `recent_chord_history`、`recent_touch_run`、`last_duration_bracket` / `recent_duration_run` 与 `recent_button_lanes`（并在 `replay` 中完整重置）。
     - 当 `recent_touch_run >= 3` 时，在 `event()` 中衰减 `style_log[3:5]`，并在 `token_bias` 中（仅当存在可选按键分支时）抑制继续输出 `'A'..'E'`。
     - 当 `alt_pairs >= 3` 时，在 `token_bias` 中对 `'/'` 追加周期 2 交替抑制。
     - 当同一时值括号连续重复 `recent_duration_run >= 4` 且词表存在至少 2 种时值、当前前缀处于 `'['` 括号内并有多于 1 个合法续写字符时，在 `token_bias` 中抑制继续拼出同一时值的字符。
     - 当词表包含至少 5 个按键且近期 6 个按键事件被困在 `<= 4` 个键位（或近期 4 个事件被困在 `<= 2` 个键位）时，在音符起始位置对受困键位字符施加有界负偏置（仅当候选含子集外键位时生效）。所有 `token_bias` 仍严格夹紧在 `[-3.0, 0.0]` 内。
  3. [`src/maichart/webui/static/app.js`](../../src/maichart/webui/static/app.js#L2176)：为 `STREAM_GROUNDING_FIELDS` 增加存在性守卫，并在 `mode === 'fresh'` 时恢复 `fallback` 默认值。

---

## 3. 修改文件清单

| 文件路径 | 改动性质 | 核心内容说明 |
| --- | --- | --- |
| [`src/maichart/stream/model.py`](../../src/maichart/stream/model.py#L216) | 模型训练图修复 | `intent = plan.detach().gather(...)` 阻断 Token CE 经 `intent_projection` 对 `plan_head` 的反向梯度劫持 |
| [`src/maichart/stream/symmetry.py`](../../src/maichart/stream/symmetry.py#L193-L280) | 运动学特征去回声与 Touch 衰减 | `kinematic_rows` 支持 `masked_events` 清零 `16:18` 与 `23` 通道；真实连续 Touch 通道 `23` 按 `1 / touch_run` 衰减 |
| [`src/maichart/stream/data.py`](../../src/maichart/stream/data.py#L605-L614) | 训练数据构造 | 将 `history_hidden` 与 `prefix_hidden` 组装为 `masked_events` 传入 `kinematic_rows` |
| [`src/maichart/stream/continuation.py`](../../src/maichart/stream/continuation.py#L39-L104) | 循环退出标签扩展 | `cycle_exit_targets` 新增时值括号 `[a:b]` 退出、纯 Touch 簇退出、受限键位子集（`<=4` 键）退出检测 |
| [`src/maichart/stream/train.py`](../../src/maichart/stream/train.py#L149-L150) | 损失函数归一化 | `cycle_exit` 采用有界密度补偿分母 `max(exit_count, 0.25 * valid_count)`，单 token 附加梯度严格有界于 `[1x, 4x]` |
| [`src/maichart/stream/timing.py`](../../src/maichart/stream/timing.py#L27-L95) | 推理拍位评分 | `interval_consistency_cost` 忽略负值 `prediction < 0`；`rhythm_cycle_cost` 对 `>48` 连续相同拍差施加有界软成本 |
| [`src/maichart/form/local_load.py`](../../src/maichart/form/local_load.py#L120-L125) | 局部双押守卫 | `LocalChordGuard.event` 检测 `>=3` 对 `(双押, 单键)` 周期 2 交替并施加阻尼压力 |
| [`src/maichart/form/type_guidance.py`](../../src/maichart/form/type_guidance.py#L150-L545) | 推理形态引导 | `TypeGuidance` 增加周期 2 双押交替、连续 Touch、连续相同时值括号 `[a:b]` 与受限按键子集的因果软抑制 |
| [`src/maichart/webui/static/app.js`](../../src/maichart/webui/static/app.js#L2176) | 前端状态同步 | `STREAM_GROUNDING_FIELDS` 隔离上下文守卫及 `fresh` 模式默认值恢复 |
| [`tests/test_stream_training_contract.py`](../../tests/test_stream_training_contract.py#L134-L143) | 契约回归测试 | 验证有界密度补偿 `cycle_exit` 梯度公式与 `<= 4.0` 倍率上限 |
| [`tests/test_stream_geometry_and_sections.py`](../../tests/test_stream_geometry_and_sections.py#L359-L491) | 定向回归测试 | 新增 5 组单元测试覆盖全部 5 项修复 |

---

## 4. 验证与回归测试结果

1. **Python 单元测试与契约回归**：
   - 执行流式与形态核心测试集（15 个测试文件）：
     `pytest -q tests/test_stream_training_contract.py tests/test_stream_geometry_and_sections.py tests/test_stream_grounding.py tests/test_stream_pipeline.py tests/test_stream_quality_constraints.py tests/test_stream_quality_followup.py tests/test_stream_generation_integrity.py tests/test_stream_boundary_clock_contract.py tests/test_chart_stream_v1.py tests/test_stream_training_graph.py tests/test_form_type_guidance.py tests/test_stream_configuration_search.py tests/test_stream_sustain_intent.py tests/test_webui.py tests/test_form_api.py`
   - 结果：**374 passed**，0 failed。
2. **新增 5 项专项回归测试**（位于 [`tests/test_stream_geometry_and_sections.py`](../../tests/test_stream_geometry_and_sections.py#L359-L491)）：
   - `test_plan_head_is_detached_from_token_ce_through_intent_projection`：验证非零 `intent_projection` 下 `logits` 对 `plan_head.weight` 梯度为 `None`，而 `intent_projection.weight` 梯度正常，且 `interval_consistency_cost` 对负预测返回 `0.`。
   - `test_kinematic_rows_masks_interval_echo_and_touch_flag_and_decays_consecutive_touch`：验证 `masked_events` 清零通道 `16:18` 与 `23`，且真实连续 Touch 在通道 `23` 按 `1, 1/2, 1/3` 衰减。
   - `test_cycle_exit_targets_marks_duration_bracket_touch_cluster_and_confined_lane_exits`：验证时值括号退出、Touch 簇退出与 6 事件受限键位子集退出均精确标记到首个分叉 token。
   - `test_rhythm_cycle_cost_spares_normal_steady_streams_and_damps_extreme_subdivision_overruns`：验证 48 拍差均匀流成本为 `0.`，64 拍差超长锁死触发正阻尼成本，切换拍差立即归零。
   - `test_type_guidance_and_local_chords_damp_period2_alternation_touch_flood_duration_and_lane_lockin`：验证周期 2 双押交替、连续 Touch、连续 `[16:3]` 时值、以及 `{3,4,5,6}` 键位锁死均触发预期因果抑制。
3. **WebUI Node 测试**：
   - `node --test tests/test_stream_generation_ui.js tests/test_form_ui_lifecycle.js`：**19 passed, 0 failed**。

---

## 5. 后续使用与审查建议

1. **对现有 `Prism_stream_v15`（Epoch 5/6）直接重新生成验证**：
   - 修复 1（负值 `plan[..., 0]` 保护）、修复 2（推理期连续 Touch 通道 23 衰减）、修复 4（`>48` 拍差均匀流软抑制）和修复 5（`TypeGuidance` / `LocalChordGuard` 四重防复读阻尼）**无需重新训练即可在推理端生效**。
2. **推荐微调（Finetune）或从 Epoch 2（`epoch_0002.pt`）续训/微调**：
   - 由于 Epoch 5–6 的权重中 `plan_head` 已被未截断的 Token CE 梯度拉偏、且 `kinematic_projection` 吸收了遮罩间隔回声，建议以尚未退化的 `epoch_0002.pt`（或从头训练）为起点，在包含修复 1–3 的新训练图下进行训练，使 `plan_head`、`kinematic_projection` 与 `cycle_exit` 退出分布在梯度层面彻底恢复健康。
