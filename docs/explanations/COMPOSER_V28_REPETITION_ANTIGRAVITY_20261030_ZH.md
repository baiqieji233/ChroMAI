# Composer v28 (`event_set_stream_v1`) 复读与配置锁死机制审查、探针实证及修复合同（修订版）

> 实施后续补记：本审查后的逐键直接负担与按钮出口监督已落地，短训从保留的 step 17000 而非 step 8000 开始。已完成同父 400 步双种子对照，最新结果、界限和 v29 运行状态见 [实施交接](COMPOSER_V28_REPETITION_REPAIR_20261030_ZH.md)。以下保留咨询时的论证和假说状态。

> **审查范围与状态说明**：
> - 本文档基于 `src/maichart/composer/` 最新源码（`stream_model.py`、`stream_data.py`、`stream_continuation.py`、`stream_sampling.py`、`stream_recovery.py`、`event_set_codec.py`）与 `.scratch/repetition_v28_20261030/` 下的三份实测文件（[`output_trace.json`](../../.scratch/repetition_v28_20261030/output_trace.json)、[`head_probe.json`](../../.scratch/repetition_v28_20261030/head_probe.json)、[`generated_head_probe.json`](../../.scratch/repetition_v28_20261030/generated_head_probe.json)）完成修订。
> - 严格区分：**（A）已被代码与探针证实的机制事实**、**（B）当前已落地的零初始化兼容修复合同**、以及 **（C）仍需通过短训/双 seed 生成对照实验验证的架构假说**。不将静态源码推断宣称为“未训即 100% 根治”。

---

## 一、 实测现象与探针实证（`output_trace.json` / `head_probe.json` / `generated_head_probe.json`）

### 1.1 生成追踪中的三类复读表征（[`output_trace.json`](../../.scratch/repetition_v28_20261030/output_trace.json)）
在同一首 190 BPM、160.737 秒曲目的不同训练阶段与采样种子下（注：因不同跑次 seed 不同，仅作现象模式诊断，不作严格同 seed 退化归因）：
1. **等间距节奏长程粘滞**：
   - `step 5388`：出现最长 **374 步连续 `1/4` 拍间隔**（按键位置在转动，但时间间隔锁死在 `1/4` 拍）。
   - `step 8000` (`89277c93ec42`)：出现最长 **103 步连续 `0.5` 拍间隔**（如 `beat 171..209` 等段落）。
2. **单键连续连打（Stationary Jack Run）**：
   - `step 6000` (`a9bef4902cc5`)：`beat 368..374.5` 连续 **14 次单键 `3`**，紧接着 `beat 375..387` 连续 **24 次单键 `4`**（前 23 次间隔均为 `0.5` 拍）。
3. **长程纯 Touch（`shape=0`）段落粘滞**：
   - `step 8000` (`89277c93ec42`)：`beat 15..79`（长达 **20.2 秒**）出现 **65 个连续 Touch-only 事件**（`C, E7, B5, E6, E8, D5, E5, E4...`），全部走 `shape == 0` (`fallback`) 分支。

### 1.2 8000 步模型头探针与反事实干预实证
通过 [`head_probe.json`](../../.scratch/repetition_v28_20261030/head_probe.json)（真实首谱 teacher batch 194 个有效 NOTE 目标）与 [`generated_head_probe.json`](../../.scratch/repetition_v28_20261030/generated_head_probe.json)（`89277c93ec42` 自产前缀每 8 步探针），可以定量确认以下两条自强化反馈通路：

1. **`keep_gap_head` 对同间距延续的放大与对切换步（Switch Step）的压制**：
   - 在 194 个真实 NOTE 目标上，`keep_gap_head` 输出均值为 **`+1.831`**（中位数 `+1.893`，上四分位 `+2.556`，最大值 `+4.222`），将 `P(previous_gap)` 的平均概率从无 `keep_gap` 时的 **`0.466`** 推高至 **`0.639`**（中位数从 `0.459` 升至 `0.832`）。
   - **关键双刃剑效应**：
     - 在真值确实延续上一拍 `delta` 的 118 个位置上，`keep_gap` 使真值概率均值从 `0.678` 升至 `0.876`（NLL 平均改善 `+0.385`），因此在全 batch 平均下它是降 loss 的有利分支；
     - **但在真值需要切换 `delta`（`target_matches_previous_gap == false`）的 76 个转折位置上**，`keep_gap` 仍然给出了均值 **`+1.522`** 的正偏置，导致真值目标概率从无 `keep_gap` 时的 **`0.577` 下降到 `0.483`**（中位数从 `0.617` 跌至 `0.441`，NLL 恶化 `-0.320`）！
   - 在自产序列 [`generated_head_probe.json`](../../.scratch/repetition_v28_20261030/generated_head_probe.json) 的纯常规段（`touch_ratio == 0`，36 个探针点）中，`keep` 均值达 **`2.397`**，使 `p_gap` 均值从 `0.730` 推高到 **`0.905`**（中位数 **`0.979`**），再叠加 `pointer_temperature = 0.65` 的缩放，直接解释了跨乐句的等间距长跑难以切出。

2. **历史 Touch 比例对 `shape == 0` (`fallback`) 概率的强自回归反馈**：
   - 在 [`generated_head_probe.json`](../../.scratch/repetition_v28_20261030/generated_head_probe.json) 中：
     - 当最近窗口 `touch_ratio == 0`（36 点）时，`fallback_p` 均值仅为 **`0.051`**（中位数 `0.012`）；
     - 当 `touch_ratio in (0.001, 0.5]`（14 点）时，`fallback_p` 均值升至 **`0.429`**；
     - 当 `touch_ratio >= 0.90`（10 个自产高密度 Touch 前缀位置，如 `step 32..72`, `128..144`, `512`）时，`fallback_p` 均值高达 **`0.888`**！
   - **反事实干预验证**：在这 10 个 `touch_ratio >= 0.90` 的位置上，**保持完全相同的音乐时刻、音频特征与时间步**，仅将历史前缀中的 Touch 替换为普通单键 `1`（`counterfactual_all_past_touch_to_tap1_fallback_p`），10 个位置的 `fallback_p` **100% 全部骤降**，均值从 **`0.888` 跌至 `0.044`**（例如 `step 32` 从 `0.971` 降至 `0.022`，`step 40` 从 `0.939` 降至 `0.019`）。
   - *注：将全部历史 Touch 替换为单键 `1` 属于人为构造的分布外（OOD）干预，不能据此宣称音频完全不起作用，但它直接证实了 Decoder 历史流中的 Touch 状态是维持 `fallback_p ~ 0.9` 的核心自回归通路。*

---

## 二、 六大核心机制问题的精确校准与已落地修复合同

### 问题 1：多目标并行训练（`H=64 + K=16`）与流式推理的历史合同如何严格统一？

#### 1.1 为什么单层 Band Mask 不能统一多层 Transformer 上下文？
- 在多层（如 `layers=4`）Transformer Decoder 中，即便每一层都施加宽度为 $W=65$ 的带状因果掩码（sliding-window causal mask），第 2 层仍能通过第 1 层中间位置的隐状态间接读取到 $2W$ 步之前的信息，$L$ 层堆叠后的有效感受野会扩张到 $\min(\text{seq\_len}, L \times W)$。
- 因此，如果训练输入是长度 `80`（`H=64 + K=16`）的连续序列，而推理输入在进入模型前被硬截断为 `65` 行，单靠每层 attention mask **无法**让训练后部目标槽（`65..80` 行）与推理 `65` 行在多层隐状态上等价。

#### 1.2 已落地的精确等价解：`block_aligned_v2`（保留 16 步并行效率，多层逐位等价）
当前源码（[`stream_data.py:384-389, 627, 775-780`](../../src/maichart/composer/stream_data.py#L384-L389)）已采用**全局固定 `K` 块对齐起点（`block_aligned_v2`）**：
- **训练侧**：将全曲流按固定步长 $K = \text{rollout\_steps} (16)$ 切分为不重叠的目标块 $[mK,\; (m+1)K)$，每个训练样本的输入序列起点固定为：
  $$\text{first} = \max(0,\; mK - H)$$
  块内第 $j$ 个全曲步（$j \in [mK, (m+1)K)$）在因果掩码和因果 `_pad_load` 下可见的精确历史区间是 $[\text{first} .. j]$（长度在 $H+1$ 到 $H+K$ 之间变化），且每个目标步各自读取其所属时间窗权重 `target_window_weights[song][first + i]`（[`stream_data.py:633-635`](../../src/maichart/composer/stream_data.py#L633-L635)）。
- **推理侧**：[`StreamInferenceContext._append_step`](../../src/maichart/composer/stream_data.py#L775-L780) 在已提交步数为 $j$ 时，按**完全相同的公式**计算保留起点：
  $$\text{first} = \max\left(0,\; \lfloor j / K \rfloor \cdot K - H\right)$$
  保留 `keep = j - first + 1` 行。
- **合同性质**：由于 Decoder 自注意力与 `_pad_load` 都是严格因果的（第 $j$ 行绝不依赖 $j$ 之后的行），训练块 $[mK, (m+1)K)$ 中任意目标步 $j$ 所看到的输入行序列 $[\text{first} .. j]$ 与流式推理推进到步 $j$ 时的输入行序列 **在所有 Transformer 层上 100% 严格一致**（已通过 2 层模型逐 target logits 等价性验证），同时完整保留了一次前向计算 16 个目标步的训练吞吐率。

---

### 问题 2：`motion_head`、`keep_gap_head` 与 `event_phase_logits` 的自强化机制与校正

#### 2.1 `motion_phase_features` 在静止同键对（`before == after`）上的三通道重合及其修复
- **代码机制（[`stream_model.py:437-458`](../../src/maichart/composer/stream_model.py#L437-L458)）**：
  - 三个运动先验通道的目标车道为 `destinations = stack(((2*after - before) % 8, before, after), -1)`，分别代表：`KEEP`（延续角速度）、`REVERSE`（折返至前一键）、`STAY`（停留在当前键）。
  - 在旧策略 `motion_continuation_v1`（`distinct=False`）下，当连续两个旋转事件落在同一相位（`before == after == k`）时，`(2k - k) % 8`、`before`、`after` 三个通道同时指向车道 $k$。[`head_probe.json`](../../.scratch/repetition_v28_20261030/head_probe.json#L71-L115) 显示 8000 步时 `motion_head` 三个分量的均值分别为 `[+0.270, -0.145, +0.206]`（上四分位 `[+0.811, +0.436, +0.858]`，`motion_bias_span` 最大达 `2.805`）。当 `before == after` 时，原本用于刻画旋转延续的通道 0 与通道 1 也会叠加到静止车道 $k$ 上。
- **已落地修复（`motion_distinct_v2`，[`stream_model.py:451-455`](../../src/maichart/composer/stream_model.py#L451-L455)）**：
  ```python
  if distinct:
      moving = before.ne(after)
      pushforward = pushforward * torch.stack((moving, moving, torch.ones_like(moving)), -1)[..., None]
  ```
  在对称轨道边缘化（orbit marginalization）之前，要求 `KEEP` 和 `REVERSE` 仅在 `before != after` 的异键转移对上非零，静止同键对仅由第 3 通道 `STAY` 单独表征，消除了三通道在静止点上的代数重合。

#### 2.2 关于 `event_phase_logits` 与负荷传导通路的准确数学表述（订正）
- **原初版文档表述的勘误**：
  - 在 [`event_phase_logits`](../../src/maichart/composer/stream_model.py#L413-L435) 中，当上一拍是单键（`reference_phase_mask` 仅有 1 个 `True`，记为车道 $r$）时，`reference = self.reference_phase_head(context).float().masked_fill(~mask, -torch.inf).log_softmax(-1)` 在 $r$ 处恒为 `0`，其对 `reference_phase_head` 的导数为 `0`。
  - **但这并不等于按键负荷对目标 `phase` 完全断路**：因为 `context`（由 `hidden`、`music`、以及 8 车道 `_prefix_load` 融合而成，见 [`stream_model.py:331-340, 520-543`](../../src/maichart/composer/stream_model.py#L331-L340)）同时包含了历史绝对 `atom_embedding`、绝对 `phase_context` 和 8 车道绝对负担；`relative = self.phase_head(context + self.shape_embedding(shape))` 完全可以通过读取 `context` 来压低 $\Delta\phi = 0$（即不换键）或特定相对位移的概率。
  - 单一合法参考相位下 `reference_phase_head` 导数为 0，仅表示“参考系起点 $r$ 唯一确定，无需在多个对称起点间做权重分配”，而相对相位分布 `relative` 对 `context`（含负荷）的梯度始终是畅通的。

#### 2.3 `keep_gap_head` 从“加性 Logit 偏置”升级为“校准的 Берnoulli 继续/切换分解”（[`stream_model.py:302-329`](../../src/maichart/composer/stream_model.py#L302-L329) & [`stream_sampling.py:38-55`](../../src/maichart/composer/stream_sampling.py#L38-L55)）
- **为什么旧版 `additive_v1` 容易锁死**：
  - 旧版直接做 `delta_logits = delta_logits + keep_gap_head(hidden) * matches`。此时 `previous_delta` 的最终概率不仅取决于 `keep_gap_head`，还与基础 `delta_logits` 在同一分类面上重复叠加；且在采样时直接对全部 `delta_logits` 除以 `pointer_temperature = 0.65`，由于“切换（switch）”概率质量被分散在几十个不同的候选 `delta` 上，而“继续（copy）”独占一个类，平坦低温（`T < 1`）会严重挤压所有分散的切换候选，指数级放大单一 copy 类的中选率。
- **已落地修复（`probability_v2` + `delta_sampling_logits`）**：
  1. **显式二元继续/切换概率分解（[`repeat_delta_log_probs`](../../src/maichart/composer/stream_model.py#L302-L329)）**：
     - 当上一步为合法 NOTE 时，定义继续门控：
       $$g = \text{keep\_gap\_head}(h) + \text{keep\_gap\_audio}\!\left(k_{\text{prev}} - \bar{k}_{\text{allowed}}\right)$$
     - 保持上一间距（`copy`）的对数概率严格等于 $\ln \sigma(g)$；切换至其他合法间距（`other = allowed & ~copy`）的对数概率严格等于 $\ln \sigma(-g) + \text{log\_softmax}_{\text{other}}(\text{logits})$。
  2. **关于候选音频去均值的数学区分（重要订正）**：
     - 在 [`timing_audio_scores`](../../src/maichart/composer/stream_model.py#L298-L300) 的多分类线性点积 $\langle q(h), k_c \rangle$ 中，如果单纯对所有候选 $k_c$ 减去候选均值 $\bar{k}$，每个候选 logit 只会减去同一个与 $c$ 无关的常数 $\langle q(h), \bar{k} \rangle$，`softmax` 对平移严格不变，因此**在线性多分类点积中单纯去均值不会改变任何候选概率**（若加 LayerNorm 则是另一项改变特征尺度的非线性变换，需单独实验）。
     - **但在二元门控 `keep_gap_audio(previous_key - mean)`（[`stream_model.py:318-319`](../../src/maichart/composer/stream_model.py#L318-L319)）中**，因为输出是标量 $g$ 并通过 $\sigma(g)$ 与固定阈值 `0` 比较，用 $k_{\text{prev}} - \bar{k}_{\text{allowed}}$ 作为输入确实提供了有效的“当前延续落点相对于其它合法候选时刻的平均声学对比度”。
  3. **保分组质量的采样温度缩放（[`delta_sampling_logits`](../../src/maichart/composer/stream_sampling.py#L38-L55)）**：
     - 采样时先对二元分组质量 $(\ln P(\text{copy}),\; \ln P(\text{switch}))$ 按温度 $T$ 缩放，再在 `switch` 内部对各候选 `other` 按温度 $T$ 缩放。这彻底阻止了“因为 switch 拆成多个子类而在低温下被单一 copy 类吞噬”的扁平温度偏差。

---

### 问题 3：`fallback`（`shape=0`）的语料结构、Touch 状态补齐与局限性说明

#### 3.1 最新 Codec 覆盖率与 `fallback` 构成的精确数据（订正）
- 在最新 `EventSetCodec` 下：
  - **常规可旋转模板（`shape > 0`）已覆盖 88.01% 的事件**，`fallback`（`shape == 0`）占比为 **11.99%**（而非旧统计的 14.05%）。
  - 在这 `11.99%` 的 fallback 事件中，**Touch 相关事件为 21,407 个，不支持旋转规约的复杂/复合 Slide 事件为 20,590 个**（两者接近 1:1）。
  - **重要推论**：即便未来将全部常规 Touch（`C, A1..E8`）都正规化进旋转对称 `shape > 0` 体系，`fallback` 中仍会保留约 **5.88%** 的复杂 Slide 事件，**不会自动降到 `< 1.5%`**。因此，除了探索 Touch 旋转模板外，**必须直接修复 `shape == 0` 在历史流与负荷中的状态表征**。

#### 3.2 已落地的 Touch 独立传感器负荷与游程状态（`separate_touch_v1` + `typed_runs_v1`）
- **为什么之前 Touch 容易连出 65 个**：
  - 旧版只有 8 个外圈按键的 `atom_lanes`（`1..8`），所有 33 个 Touch 传感器区域（`C, A1..8, B1..8, D1..8, E1..8`）在 `_prefix_load` 中完全没有负荷记录，也没有“当前已连续打出多少个 Touch / 持续了多少秒”的显式状态。
- **已落地修复**：
  1. **33 维独立 Touch Pad 负荷（[`stream_model.py:150-152, 199-210, 337-339, 542-543`](../../src/maichart/composer/stream_model.py#L199-L210)）**：
     - 通过 `head_pad_indices_for_policy(..., 'separate_pads_v2')` 构建 `atom_touch` 缓冲区（`33` 个独立 Touch 传感区，不与 8 个物理按键混同），在 Decoder 输入 `x` 和动作上下文 `action_context_at` 中同时注入 `touch_load_projection(_pad_load(self.atom_touch, ...))`（4 个尺度/间隔特征 $\times 33 = 132$ 维，零初始化以兼容已有权重）。
  2. **独立于 Codec 的 8 维全前缀精确游程特征（[`stream_continuation.py:28-74`](../../src/maichart/composer/stream_continuation.py#L28-L74)）**：
     - `ContinuationState` 在完整已提交前缀上精确统计（不因 64 步窗口滑动而丢失更早起点的累计计数），输出 8 维 `log1p` 特征（通过零初始化的 `continuation_projection` 注入 Decoder 输入 `x`）：
       - `gap_count`, `gap_elapsed`（同 `delta` 连续步数与累计拍数）
       - `type_count`, `type_elapsed`（同音符类型集合连续步数与持续秒数）
       - `touch_count`, `touch_elapsed`（纯 Touch 连续步数与持续秒数）
       - `max_button_run_count`, `max_button_run_elapsed`（单键连续击打次数与持续秒数）

---

### 问题 4：`recovery_loss` 的真实归一化尺度与转折步（Exit Step）联合监督

#### 4.1 关于 `recovery_loss` 梯度尺度的数学订正
- **原初版文档表述的勘误**：
  - 在 [`stream_train.py:298`](../../src/maichart/composer/stream_train.py#L298) 与 [`stream_model.py:721`](../../src/maichart/composer/stream_model.py#L721) 中，recovery 批次显式设置了 `recovered['_loss_normalizer'] = recovered['target_kind'].size(0)`（即按有效 recovery 行数 $B_{\text{rec}} = 2$ 独立求均值），而主训练批次按 $B \times K = 256$ 求均值。
  - 因此，每次触发 recovery 时，其损失是与主训练同量纲的**单步平均 NLL** 乘以 `recovery_weight = 0.1`（平均到每 25 步更新周期的标量权重约为 $0.1 / 25 = 1/250$），**绝不是按原始目标总数 $6400 : 2$ 折算的 $1/32000$**。
- **Recovery 的真实瓶颈在于“采样覆盖面稀疏”而非分母归一化**：
  - 每 25 更仅抽 2 条短轨迹，且旧版 `_anchors` 只包含 `WAIT` 和双押重叠 `repeat`，没有覆盖“等间距长跑切出”和“Touch 段切出”。

#### 4.2 已落地的双重转折强化（主训练 `transition_exit_loss` + Recovery Exit Anchors）
1. **主训练全量覆盖的 `transition_exit_loss`（[`stream_continuation.py:75-85`](../../src/maichart/composer/stream_continuation.py#L75-L85) & [`stream_model.py:820-824`](../../src/maichart/composer/stream_model.py#L820-L824)）**：
   - 当真实前缀满足 `gap_count >= 4` 且当前真值改变了 `delta`，或 `touch_count >= 3` 且当前真值切出纯 Touch 时，标记为真实转折切出步（`exit_mask`），对其完整联合负对数似然 `joint_nll = kind_nll + delta_nll + shape_nll + phase_nll + detail_nll` 额外施加 `transition_exit_weight = 0.5` 的监督权重。
   - 这样既不惩罚原谱中合法的连续等间距/连续 Touch 内部步，又在每一个真实乐句转换边界上把“切出动作 + 切出时间”的联合梯度放大了 $1.5\times$。
2. **Recovery 锚点新增 `touch_exit` 与 `rhythm_exit`（[`stream_recovery.py:77-96`](../../src/maichart/composer/stream_recovery.py#L77-L96)）**：
   - 在真实转折步之前的 2 步（`max(0, index - 2)`）建立恢复采样锚点，使模型在自生成 2 步后直接面对“需要从 Touch 段或等间距段切回常规谱面”的参考监督。

---

### 问题 5：`action_context_at` 的真实结构与音频约束评估（订正）

1. **当前源码中 `action_context_at` 的真实结构（[`stream_model.py:166, 331-340`](../../src/maichart/composer/stream_model.py#L331-L340)）**：
   - 订正初版文档中的过时代码引用：当前 `StreamComposer` 定义为：
     ```python
     self.action_fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
     ```
     并在 `action_context_at` 中执行：
     ```python
     music = self._read_audio(audio_cache, event_seconds, song_inverse)
     result = self.action_fusion(torch.cat((hidden, music.to(hidden)), -1))
     ```
     随后再加上 `load_projection` 与 `touch_load_projection`。
   - 因此，**`hidden` 与候选时刻精确音频 `music` 之间已经具备 `Linear(2d -> d) -> GELU -> LayerNorm(d)` 的非线性融合与归一化**，并非裸线性相加。
2. **为什么具备 `action_fusion` 后在 8000 步仍观察到 `generated_head_probe.json` 中的强历史主导？**
   - 如第一节探针所示，当历史陷入连续 Touch 或连续同 `delta` 时，由于旧版缺少 `touch_load`、缺少 `continuation_features`（游程计数）、且 `keep_gap_head` 是无条件加性偏置，Decoder 输出的 `hidden` 本身已经高度饱和地编码了“继续 Touch / 继续同间距”，压过了 `music` 分支的影响。

---

### 问题 6：当前已就绪修复清单 vs 后续实验与大胆假说路线图

#### 6.1 已在源码中就绪、待新目录 Finetune 与双 Seed 验证的修复合同
1. **`block_aligned_v2` 历史窗口合同**：训练与流式推理按全局 $K=16$ 块对齐输入起点，多层 Transformer 隐状态与负荷前缀严格一致，保留 16 步并行效率。
2. **`motion_distinct_v2` 运动先验正交化**：静止同键对（`before == after`）屏蔽 `KEEP` 与 `REVERSE` 通道，仅保留 `STAY`。
3. **`probability_v2` + `delta_sampling_logits`**：将间距延续从加性 logit 偏置改为包含候选声学对比度 `keep_gap_audio(previous_key - mean)` 的二元 Bernoulli 继续/切换门控，并在采样时对 continue/switch 分组质量与 switch 内部候选分别应用温度。
4. **`separate_touch_v1`（33 Touch Pad 独立负荷）+ `typed_runs_v1`（全前缀 8 维游程状态）**：补齐 Touch 传感器疲劳度与同间距/同类型/同按键连续游程感知，新投影层全部**零初始化**以平滑继承已有 checkpoint 权重。
5. **`transition_exit_weight = 0.5` + Recovery Exit 锚点**：在真实长游程切出边界强化联合分布监督。

#### 6.2 值得在后续迭代中验证的进阶假说（不阻塞当前 Finetune）
1. **假说 A（Touch 8 重旋转对称模板化）**：
   - 将外圈 Touch（`A1..8, B1..8, D1..8, E1..8`）与中心 `C` 纳入 `EventSetCodec` 的旋转等价类，使占 fallback 约一半（21,407 / 42,000）的 Touch 事件转为标准 `shape + phase + detail` 表示，让 `fallback (shape=0)` 仅专职负责剩余 ~5.88% 的复杂多滑条。
2. **假说 B（逐车道旋转等变负荷偏置直接作用于 `event_phase_logits`）**：
   - 虽然 `phase_head(context)` 已能读取 8 键负荷，但若在 `event_phase_logits` 上显式加入一个共享权重的逐车道负荷标量投影（按候选 `(shape, phase)` 实际落点的按键集合求和），可进一步降低样本复杂度，让“高负荷按键自动降权”具备直接的归纳偏置。
3. **假说 C（固定双 Seed 探针回归对比）**：
   - 在新目录从 8000 步 checkpoint 启动短程 finetune 后，使用完全相同的评估脚本重跑 [`head_probe.json`](../../.scratch/repetition_v28_20261030/head_probe.json)（重点观察 `target_matches_previous_gap == false` 时的 `p_target` 是否从 `0.483` 回升）及固定双 seed 生成追踪（对比最大连续单键数、最大连续 Touch 步数、最大连续同 `delta` 步数）。
