# Chart Composer (`audio_grounded_v3`) 质量改造交接与复审文档（供 GPT Astra 审核）

> 后续状态（2026-10-28）：本文件保留上一轮实现记录，部分内容已被后续审查纠正。当前修复、验证与未完成事项见 [最新交接说明](COMPOSER_CURVE_AND_OBJECTIVE_HANDOFF_20261028_ZH.md)。下文的 EOS 连续确认、竞争层类型、曲线通道与零 Hold 拦截描述，不能作为当前代码行为的依据。

本文档汇总了结合此前架构审查与 GPT Astra 评估意见所完成的全部代码改造、设计取舍、数学不变量及回归测试结果，供下一轮复审与训练/生成验收使用。

---

## 一、 总体改造概览与对齐说明

本次改造严格围绕四个核心优先级展开，并采纳了 Astra 提出的全部关键补充与口径修正：

| 优先级 | 改造主题 | 核心修改文件 | 是否兼容现有 `prism_composer_v18` 权重直接推理 |
|---|---|---|---|
| **Priority 1** | 提前停谱修复：区分“最后音符头 / 动作尾 / 整曲结束”、EOS 提议确认机制、中段休止跨越与恢复训练 EOS 监督 | `src/maichart/composer/generate.py`, `src/maichart/composer/recovery.py`, `src/maichart/composer/objectives.py`, `src/maichart/composer/train.py` | **是**（推理立即生效；恢复训练 EOS 监督在续训/微调中生效） |
| **Priority 2** | 跨节奏网格可达性解除、零时值 Hold 守卫与解析后分音诊断指标 | `src/maichart/composer/candidates.py`, `src/maichart/composer/data.py`, `src/maichart/composer/recovery.py`, `src/maichart/composer/guard.py`, `src/maichart/composer/configuration_observation.py` | **是**（推理立即解除三分音锁死，训练与恢复采样同步统一） |
| **Priority 3** | 难度/负荷曲线预测与交互式编辑全链路接通（真实速率域 `expm1` -> 缩放/覆盖 -> `log1p`） | `src/maichart/composer/curve.py`, `src/maichart/composer/generate.py`, `src/maichart/composer/grounded_model.py`, `src/maichart/composer/model.py`, `src/maichart/webui/server.py`, `src/maichart/webui/composer_api.py`, `src/maichart/webui/static/app.js`, `src/maichart/webui/static/index.html` | **是**（直接复用已训练的 `workload_head` 与 `planner` 头，无需重训即可在 WebUI 预测并编辑曲线） |
| **Priority 4** | 细粒度采音跨候选竞争、高斯中心精定位监督、辅助头部分开放梯度与音频动作辅助任务宏观化 | `src/maichart/composer/grounded_model.py`, `src/maichart/composer/audio_lens.py`, `src/maichart/composer/conserved_actions.py`, `src/maichart/composer/objectives.py`, `src/maichart/composer/checkpoint.py`, `src/maichart/composer/train.py` | **是**（旧权重加载时自动保持零步等价；微调/新训时激活新竞争层与新监督） |

---

## 二、 分项详细设计与实现说明

### Priority 1：提前停谱修复与“音符头 / 动作结束 / 整曲结束”职责解耦

#### 1. 问题根因与设计目标
在 `generate.py` 原实现中，只要单步候选的 `sigmoid(eos_logit) >= 0.98`，解码循环便无条件触发 `stop_reason = 'learned_eos'` 并立刻终止整曲生成。由于训练期 `eos_loss` 使用按批次动态计算的 `pos_weight = clamp(negatives / positives, 1.0, 64.0)`，单次高分并不能直接等价于整曲结束，尤其在间奏、长按（Hold/Slide）持续覆盖区或局部静默段极易发生误停（如 `4412dfb7c5ca` 在 145.23s 停谱，而模型自身的 Planner 在 156s 仍预测显著活动）。

#### 2. 具体修改点
1. **显式区分三个时间边界**（`src/maichart/composer/generate.py`）：
   - `last_head_sec`：最后一个已提交事件（音符头）的秒数；
   - `last_action_end_sec`：所有已提交动作（含 Hold / TouchHold / Slide 持续尾端）的最大结束秒数；
   - `effective_end_sec`：音频有效结束边界。
   - 三者均写入 `.composer.json` 报告顶层字段，不再混同。
2. **将 EOS 视为“结束提议（Proposal）”并结合剩余规划确认**（`_has_future_planned_activity`）：
   - 当 `eos_prob >= 0.98` 时，不再直接 `break`，而是记入 `eos_diagnostics['eos_proposals']`。
   - 按以下顺序审查是否允许真正结束整曲：
     1. **人工曲线/休止区审查**：若启用了 `curve_controller` 且后续仍有非休止活动（`curve_controller.has_future_activity`），否决 EOS（记录原因 `curve_future_activity`）；
     2. **持续动作覆盖审查**：若当前候选时间仍在已提交长音尾端之前（`cand_sec + 0.05 < last_action_end_sec`）且距离有效音频末尾超过 `1.5s`，否决 EOS（记录原因 `active_sustain_coverage`）；
     3. **模型自身 Planner / Workload 剩余活动审查**：调用 `_has_future_planned_activity` 检查 `after_sec + 0.5s` 至 `effective_end_sec - 0.5s` 区间内：
        - 若 `planner_prediction` 存在 `expm1(head_rate) >= 0.65` 且 `rest_prob < 0.70` 的未来四拍块，否决 EOS（`planner_activity_remaining`）；
        - 若 `song_workload` 存在 `expm1(head_rate) >= 0.75` 的未来探针，否决 EOS（`workload_activity_remaining`）；
     4. **远端安全护栏**：若距离 `effective_end_sec` 尚有超过 `8.0s` 且未提供任何规划确认证据，需连续 3 次 EOS 提议（`consecutive_eos_proposals >= 3`）才允许在尾段接受停止，否则记录 `premature_horizon_guard` 并继续。
3. **中段休止跨越与历史保留**：
   - 当处于显式标记的 `rest_regions` 或局部候选因中段休止被耗尽而后续仍有活动时，仅推进搜索游标 `search_beat`（保留 `committed_points` 全部已生成谱面历史与 `guard` 状态），累计 `eos_diagnostics['rest_skips']`，借鉴 `event_ctx` “局部窗口结束不等于整曲结束”的控制职责。
4. **恢复训练（Recovery）补充生成历史上的 EOS 监督与 `eos_pos_weight` 记录**：
   - 在 `src/maichart/composer/recovery.py` 的 `RecoveryBuilder` 中新增 `supervise_eos: bool = False`（默认 `False` 以保持旧单元测试向后兼容；在 `train.py:recovery_batch` 中当 `action_objective_policy == RECOVERY_ACTION_OBJECTIVE` 时自动开启为 `True`），从 `views.eos(target_index)` 读取可信的继续/结束标签。
   - 在 `src/maichart/composer/objectives.py:recovery_objective` 中，当 `eos_weight > 0` 时对生成历史上的延续步施加 `0.15 * eos_loss` 监督并记录 `recovery_metrics['eos_loss']`。
   - 在 `src/maichart/composer/train.py:_loss` 中记录实际批次的 `eos_pos_weight` 以便后续校准。

---

### Priority 2：跨节奏网格转换解锁、零时值 Hold 守卫与分音指标

#### 1. 问题根因与设计目标
原 `MUSIC_CANDIDATE_POLICY_V6` 过滤条件要求 `beat.denominator in {1, 2, 3, 4, 6, 8}` **且** `(beat - previous).denominator in {1, 2, 3, 4, 6, 8}`。当上一拍位于三分音相位（如 `previous = 1/3`）时，返回合法二元网格 `3/4`（差值 `5/12`，分母 12）或 `7/8`（差值 `13/24`，分母 24）会被差值分母条件直接禁止，导致模型一旦进入三分音网格就只能等到半拍或整拍才能回到 `1/4`、`1/8` 二元网格，形成连续 `{12}`、`{24}` 相位粘连。

#### 2. 具体修改点
1. **统一跨网格可达性谓词 `is_playable_candidate_transition`**：
   - 定义在 `src/maichart/composer/candidates.py` 中，要求目标绝对位置 `beat.denominator in CANONICAL_BEAT_DENOMINATORS`（`{1, 2, 3, 4, 6, 8}`）。
   - 对于相邻拍差 `delta = beat - previous`：
     - 若 `delta.denominator in {1, 2, 3, 4, 6, 8}`（同网格家族步进或整/半拍对齐），直接允许；
     - 若为跨网格转换（此时因 `lcm(3, 4) = 12`、`lcm(3, 8) = 24`，`delta.denominator in {12, 24}`），只要物理拍距 `delta >= Fraction(1, 16)`（即过滤掉如 `1/3 -> 3/8`、`delta = 1/24` 拍这类极短的跨网格微桥接碎片，而允许 `1/3 -> 3/4`、`delta = 5/12` 及 `1/3 -> 7/8`、`delta = 13/24` 等正常跨网格回归），即判定为合法。
2. **训练、恢复与生成三端统一统计**：
   - `generate.py:_candidate_playability` 直接调用 `is_playable_candidate_transition`；
   - `data.py` 与 `train.py` 新增 `playability_unreachable_targets` 统计，记录官谱目标在解码可达性规则下的真实可达率；
   - `recovery.py` 新增 `playability_unreachable_references` 统计并写入恢复诊断。
3. **拦截零时值 Hold（`nonpositive_hold_duration`）**（`src/maichart/composer/guard.py`）：
   - 在 `EventGuard` 中检查所有 `h[...]` 动作的解析后持续秒数，若 `duration <= 1e-9` 则拒绝并标记原因 `'nonpositive_hold_duration'`，消除 `3h[16:0]` 等退化零长按。
4. **解析后事件级节奏与分音统计**（`src/maichart/composer/configuration_observation.py`）：
   - 按 Astra 建议，不以文本 `{12}`、`{24}` 出现次数代替分音比例，而是在 `configuration_observation(points)` 中直接对解析后的 `EventPoint.beat` 序列统计：
     - `beat_denominator_counts`（绝对拍位分母分布）
     - `interval_denominator_counts`（相邻拍差分母分布）
     - `family_counts`（`aligned` / `binary` / `ternary` 事件数）
     - `ternary_event_ratio`（三分音家族事件占比）
     - `rhythm_family_transitions` 与 `cross_grid_transitions`（节奏家族切换次数）
     - `max_ternary_run_length`（最长连续三分音事件串长度）

---

### Priority 3：难度/负荷曲线预测与交互式编辑接通

#### 1. 设计原理与真实速率域变换
`AudioGroundedComposer`（如 `prism_composer_v18`）中已具备训练好的两个曲线相关预测头：
- `planner(planner_audio, condition, planner_mask) -> [B, T_plan, 6]`：四拍块级规划，通道为 `[log1p(event_rate), log1p(head_rate), multi_ratio, slide_ratio, touch_ratio, rest_prob]`；
- `workload_head([audio, cond]) -> [B, K, 5]`：1 秒窗口局部负荷，通道为 `[log1p(event_rate), log1p(head_rate), hold_load, slide_load, touch_spread]`；
- 以及历史尾迹负荷条件 `trailing_audio -> [B, K, 4]`（`[log1p(event_rate), log1p(head_rate), touch_ratio, slide_ratio]`）。

为避免在 `log1p` 压缩域直接线性乘缩放因子造成高密度区指数级失真，`src/maichart/composer/curve.py` 中的所有缩放与覆盖均严格在**真实速率空间（Real Rate Domain）**执行：
$$\text{rate}_{\text{real}} = \max(\exp(x) - 1, 0) \xrightarrow{\text{scale / blend / cap}} \text{rate}_{\text{mod}} \xrightarrow{} \ln(1 + \text{rate}_{\text{mod}})$$

#### 2. 具体修改点
1. **新建 `src/maichart/composer/curve.py`**：
   - `generate_learned_profile(...)`：加载 `chart_composer_v1` checkpoint，在全曲均匀时间网格上运行 `model.planner` 与 `model.workload_head`，将预测的 `head_rate`、`event_rate` 及各音符类型通道归一化导出为与 WebUI 曲线编辑器完全兼容的 `maichart-intensity-curve-v1` JSON 结构（含 `points`、`components`、`spectrogram`、`baseline_head_rate` 及 `method="learned_chart_composer_v1"`）。
   - `ComposerCurveController` 与 `apply_curve_to_composer(...)`：支持 `follow`、`soft`、`hard`、`strict` 四种模式与 `rest_regions`：
     - `transform_planner(planner_prediction, plan_centers)`：调制 `[1, T_plan, 6]`；
     - `build_window_overrides(candidate_seconds, model_workload, model_trailing)`：生成每个解码步的 `workload_override`（`[1, K, 5]`）、`trailing_override`（`[1, K, 4]`）、候选级 `rest_mask` 与 `strict` 模式下的 `max_heads` 上限。
2. **模型与生成端接入**：
   - `grounded_model.py` 与 `model.py` 新增可选 `trailing_override=None` 参数；
   - `generate.py` 新增 CLI 参数 `--intensity-curve-file`、`--intensity-curve-mode`、`--intensity-curve-span`，在解码循环中将 `planner_override`、`workload_override`、`trailing_override` 传入模型，并在 `rest_regions` 内自动屏蔽新起音候选（允许此前已开始的 Hold/Slide 跨入）。
3. **WebUI 全链路放开**：
   - `server.py`：允许 `chart_composer_v1` 使用 `model` 和 `auto` 曲线来源，调用 `maichart.composer.curve.generate_learned_profile`；
   - `composer_api.py`：当 `intensity_curve_enabled` 开启或传入 `intensity_curve_file` 时，将曲线快照写入任务目录并传递 `--intensity-curve-file`、`--intensity-curve-mode`、`--intensity-curve-span`；
   - `app.js` 与 `index.html`：为 `chart_composer_v1` 展示第 04 步“安排局部强弱（可选）”面板，解除 `model` 曲线来源禁用与保存限制。

---

### Priority 4：细粒度采音跨候选竞争、精定位监督与动作辅助任务重新分工

#### 1. 融合 `AudioLens` 后的轻量跨候选竞争层（`_CandidateCompetition`）
- **原架构缺口**：`AudioGroundedComposer` 的 `self.audio_encoder` 虽然在早期对粗粒度候选特征做了自注意力，但高分辨率 `AudioLens`（31 帧局部 TCN）与 `placement_head` 在 `self.audio_fusion` 中才被融合，融合后便直接送入逐点独立的 `self.pointer` 和 `self.audio_pointer`，导致相邻几个仅相差 `6.6–8.3ms`（如 `1/16` 与 `1/12` 格点）的候选在精细听觉表征层面无法直接横向比峰。
- **修改实现**：
  - 在 `src/maichart/composer/grounded_model.py` 中定义轻量残差跨候选注意力模块 `_CandidateCompetition`，置于 `self.audio_fusion` **之后**、`self.pointer` 与 `self.audio_pointer` **之前**。
  - 其输出投影 `self.out_proj` 采用**精确零初始化**（`nn.init.zeros_`），因此在新初始化或从旧权重迁移的第 0 步，输出与原网络逐元素严格恒等（`candidate_competition(x, mask) == x`），而在反向传播第 1 步即可建立跨候选精细峰竞争梯度。
  - 在 `src/maichart/composer/checkpoint.py` 中做了双向兼容：加载不含 `candidate_competition` 的旧推理权重时自动设 `model.candidate_competition = None`；执行 `--init-checkpoint` 迁移时自动保留零初始化的 `_CandidateCompetition`。

#### 2. 部分开放 `placement_head` 梯度与高斯中心精定位监督
- **部分开放梯度**（`src/maichart/composer/grounded_model.py`）：
  - 定义 `PLACEMENT_GRAD_SCALE = 0.15`，对融合进选时/动作上下文的 `placement[:, :count]` 采用直通缩放梯度：
    ```python
    placement_cand = placement[:, :count]
    placement_fused = placement_cand.detach() + PLACEMENT_GRAD_SCALE * (placement_cand - placement_cand.detach())
    ```
  - 这样主选时损失（`pointer_loss`）与主配置损失（`action_loss`）能够以 `0.15` 的温和比例更新 `placement_head`，同时保持 `planner` 与 `workload_head` 的 `.detach()` 隔离不变（确保可解释曲线头不被主损失扭曲）。
- **新增高斯中心精定位监督**（`PlacementTargets.proximity`）：
  - 保留原有的 `±25ms` 邻域占用标签 `placement_target`（用于感知“附近有音符”），同时在 `src/maichart/composer/audio_lens.py` 中新增 `PlacementTargets.proximity(query_seconds, sigma_sec=0.010)`，对 `±25ms` 内的候选计算以真实音符时刻为中心的高斯核分数 $\exp(-\Delta t^2 / (2\sigma^2))$（$\sigma = 10\text{ms}$）。
  - 在 `data.py` 中打包为 `'placement_proximity'`，并在 `train.py:_loss` 中对正占用位置计算平滑 L1 对齐损失 `placement_alignment_loss`（以 `0.5 * placement_weight` 加入总损失并单独记录指标），专门协助模型在 `6.6–8.3ms` 间距的相邻格点（如 `1/16` vs `1/12`）中区分真正的起音中心。

#### 3. 音频动作辅助任务宏观化与配置损失拆解诊断
- **音频辅助任务宏观化**（`build_audio_macro_groups`）：
  - 原 `audio_action_loss` 要求无跨事件键位历史的纯音频支路去预测具体的 1–8 号键位与旋转对称滑条终点，导致 `audio_action_loss` 长期停滞在 `~5.68` 并向共享动作因子注入对称性噪声。
  - 在 `src/maichart/composer/conserved_actions.py` 中新增 `build_audio_macro_groups(action_factors)`，按**位置无关的声学属性** `(kind_mask, touch_ring, break_ex, hold_duration, slide_duration)`（加 `STOP` 为第 0 组）将动作词表划分为声学宏观组，并注册为 `ConservedGroupedActionHead.audio_macro_group_id`。
  - 在 `src/maichart/composer/train.py:_loss` 中，对 `audio_action_logits` 传入 `macro_group_ids=audio_macro_group_ids, macro_only=True`，通过 `group_logsumexp` 聚合组内对数概率后再计算交叉熵。
  - **关键优势**：既让纯音频分支专注于其真正能从音频判断的“音符类型、Touch 环区、Break/Ex 修饰符、持续时值与同拍多押停止（STOP）”，又**完整保留了 `audio_action_logits` 的 `[B, S, V]` 输出形态**，使推理期 `action_logits_with_touch_guard` 的 Touch 历史防自强化保护无需任何改动即可正常工作。
- **配置损失细化诊断**（`src/maichart/composer/objectives.py:action_objective`）：
  - 当传入 `macro_group_ids` 时，利用精确分解恒等式：
    $$\text{NLL}(a) = \underbrace{-\log \frac{\sum_{a' \in g(a)} e^{z_{a'}}}{\sum_{b} e^{z_b}}}_{\text{macro\_ce}} + \underbrace{-\log \frac{e^{z_a}}{\sum_{a' \in g(a)} e^{z_{a'}}}}_{\text{detail\_ce}}$$
  - 在 `action_objective` 与 `train.py` 训练/验证指标中新增按非 STOP 决策质量池化的 `action_macro_ce`（宏观类型+起手键位选择损失）与 `action_detail_ce`（桶内轨迹形状/终点/时值细化损失），与原有的 `first_slot_ce`、`first_nonstop_ce`、`later_nonstop_ce`、`stop_ce` 共同构成完整的配置学习可观测面板。

---

## 三、 验证与回归测试结果

1. **JavaScript WebUI 曲线测试**（`node --test tests/test_composer_intensity_ui.js`）：
   - 验证 `chart_composer_v1` 选择 `model` 曲线来源时正确向 `/api/intensity/analyze` 传递 `curve_source: "model"` 与 `checkpoint`，且保存按钮与家族切换正常（**2/2 通过**）。
2. **Python Composer 全量单元测试与新增专项测试**（`pytest tests/test_composer*.py`）：
   - 新增专项测试 `tests/test_composer_astra_priority_fixes.py`，覆盖：
     - `1/3 -> 3/4`、`1/3 -> 7/8` 跨网格回归放行与 `1/3 -> 3/8` 极短微步拦截；
     - `1h[16:0]` 零时值 Hold 守卫拦截与 `configuration_observation['rhythm']` 统计；
     - `_has_future_planned_activity` 对提前 EOS 提议的否决；
     - `apply_curve_to_composer` 在真实速率空间的调制与 `rest_regions` 归零；
     - `_CandidateCompetition` 零步恒等性、`build_audio_macro_groups` 旋转对称合并、`macro_ce + detail_ce == nonstop_ce` 恒等分解、`PlacementTargets.proximity` 高斯核中心峰值及恢复训练 `eos_loss` 梯度反传。
   - 全套 `test_composer*.py` 共 **755 项测试全部通过**（`755 passed, 1 skipped`）。

---

## 四、 建议后续实验与验收关注点（供 Astra 参考）

1. **存量权重（`prism_composer_v18`）零训练直接重生成对比**：
   - 使用固定种子重新生成此前在 145.23s / 155.54s 提前停谱的曲目，检查 `.composer.json` 中的 `last_head_sec`、`last_action_end_sec`、`effective_end_sec`、`eos_diagnostics`（观察 `eos_proposals` 与 `eos_veto_reasons`）以及 `configuration_observation.rhythm` 中的 `ternary_event_ratio` 和 `max_ternary_run_length`。
2. **微调（Finetune）实验观察指标**：
   - 基于 `prism_composer_v18` 启动带余弦退火（`--lr-schedule cosine --min-lr-ratio 0.1`）的微调，重点观察验证集新增的：
     - `action_macro_ce` vs `action_detail_ce` 的下降斜率；
     - 宏观化后的 `audio_action_loss` 是否打破 `5.65` 平台持续下降；
     - `placement_alignment_loss` 与 `_CandidateCompetition` 激活后，`pointer_loss` 与生成谱面的精细采音对齐度提升情况。
