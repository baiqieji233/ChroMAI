# Composer v29 复读根因二审与无串行 Rollout 的学习侧根治方案

> **审查范围与约束**：
> - 严格只读审查 `.scratch/repetition_v29_20261030/` 下的全部实测探针（[`output_trace.json`](../../.scratch/repetition_v29_20261030/output_trace.json)、[`prefix_probe.json`](../../.scratch/repetition_v29_20261030/prefix_probe.json)、[`initial400_probe.json`](../../.scratch/repetition_v29_20261030/initial400_probe.json)、[`teacher_probe.json`](../../.scratch/repetition_v29_20261030/teacher_probe.json)、[`motion_rollout.json`](../../.scratch/repetition_v29_20261030/motion_rollout.json)、[`logic_probe.json`](../../.scratch/repetition_v29_20261030/logic_probe.json)）及当前 `v29` 源码（[`stream_model.py`](../../src/maichart/composer/stream_model.py)、[`stream_continuation.py`](../../src/maichart/composer/stream_continuation.py)、[`stream_recovery.py`](../../src/maichart/composer/stream_recovery.py)）。
> - 未修改任何源码、未运行 GPU、未触碰训练任务 `5ea3a7a720b0`。

---

## 一、 核心判断：为什么 `v29` 加了游程特征、负荷偏置和 Exit Loss，仍出现 40 步 `1-8` 交替、16 连同键与 62 步 Touch？

完全赞同你的核心诊断：**根本矛盾在于“Teacher-Forcing 下长游程的统计先验与自生成死锁环路共线”，叠加“新特征埋在 4 层已收敛 Decoder 底部且对交替/同形循环存在结构盲区”，导致显式延续头（`motion_head` / `keep_gap_head`）在闭环中单向放大，而稀疏串行 Recovery 既覆盖不到又容易跨过真实出口。**

具体拆解为以下 **5 项已被探针数据定量证实的根因**：

### 1. 为什么 8 维 `run_features` 训练 4000 步后清零对 `shape` 的 TV 影响仅有 `0.000089`？（位置埋没 + Teacher 共线性饥饿）
1. **注入位置埋在 4 层已收敛 Decoder 底部，未直达输出门控头（[`stream_model.py:544-547`](../../src/maichart/composer/stream_model.py#L544-L547)）**：
   - 在 `v29` 中，`continuation_projection`（零初始化）仅被加在 Decoder 输入 `x` 上（第 547 行），**既没有进入 [`action_context_at`](../../src/maichart/composer/stream_model.py#L331-L340)（决定 `shape_head`、`event_phase_logits`、`motion_head`），也没有进入 [`repeat_delta_log_probs`](../../src/maichart/composer/stream_model.py#L302-L329)（决定 `keep_gap` 门控）**！
   - 当从 8000 步已收敛权重微调时，4 层 Transformer Decoder 的注意力与 FFN 子空间早已固化，底部微弱的零初始化残差很难穿过 4 层自注意力去改变顶层 `shape_head` 和 `keep_gap_head` 的决策（4000 步后 `continuation_projection` 权重范数仅 `0.298`，清零后 `shape` TV 仅 `0.000089`）。
2. **Teacher-Forcing 下“游程越长 $\implies$ 继续概率越高”的统计共线性（[`teacher_probe.json:25-64`](../../.scratch/repetition_v29_20261030/teacher_probe.json#L25-L64)）**：
   - 看 `teacher_probe.json` 揭示的真实谱面先验：
     - `gap_run >= 0`：真值继续率 **60.11%**（模型预测 `58.05%`）
     - `gap_run >= 4`：真值继续率 **77.47%**（模型预测 `73.67%`）
     - `gap_run >= 8`：真值继续率 **83.85%**（模型预测 `78.63%`）
     - `gap_run >= 16`：真值继续率 **97.22%**（72 例中 70 例继续，仅 **2 例**退出！）
   - 在真实谱面中，一旦进入 $\ge 8$ 或 $\ge 16$ 步的等间距流，它往往是一整段合法的 8 分/16 分乐句，**在乐句内部 90%+ 的步数上真值都是“继续”**；而唯一的退出步往往伴随明显的音乐段落边界或长按休止（模型在 `gap_run >= 8` 的 31 个真实退出步上，哪怕不靠 `run_features` 也已经把 `p_copy` 压到了 `0.284`！）。
   - 结果：在纯 Teacher 前缀下，`run_features` 几乎收不到“因为游程长所以必须退出”的净梯度；相反，在自生成闭环中，模型在**本不该长跑的音乐位置**误入循环时，从来没在训练中见过“错误长循环前缀 $\to$ 立即切回常规谱面”的样本！

### 2. 为什么 `v29` 出现 40 步 `1-8` 交替（Trill）与单键 16 连击？（`ContinuationState` 双盲区 + `motion_head` 幅度失衡）
1. **`ContinuationState` 对双键交替（Period-2 Trill `1-8-1-8`）完全失明（[`stream_continuation.py:64-65`](../../src/maichart/composer/stream_continuation.py#L64-L65) & [`logic_probe.json:50`](../../.scratch/repetition_v29_20261030/logic_probe.json#L50)）**：
   - 在 `ContinuationState.append` 中：
     ```python
     self.buttons = {key: (self.buttons[key][0]+1, self.buttons[key][1]) if key in self.buttons else (1, second)
                     for key in buttons}
     ```
   - 当序列为 `1 -> 8 -> 1 -> 8 ...` 时，打 `1` 时清空了 `8` 的计数，打 `8` 时又清空了 `1` 的计数！
   - 因此，**在长达 40 步的 `1-8` 交替中，`run_features` 的按键游程计数恒为 `1`（`log1p(1)=0.693`），按键游程时长恒为 `0.0`，且 `exit_label` 中 `repeated_buttons` 恒为空集**！模型在游程特征和转折损失上把 40 步 `1-8` 死锁当成了“每一步都在换新键”的正常流动！
2. **`type_count` 仅按 5 种粗粒度 `note_type` 统计，无法区分配置重复与正常谱面（[`stream_continuation.py:20`](../../src/maichart/composer/stream_continuation.py#L20) & [`logic_probe.json:50-51`](../../.scratch/repetition_v29_20261030/logic_probe.json#L50-L51)）**：
   - `types = frozenset(n.note_type for n in parsed.notes)` 只有 `'tap', 'hold', 'slide', 'touch', 'touch_hold'` 五种值。
   - 这导致：单键 `1`、双押 `1/2`、散打单键流 `1->2->3->4`、与 `1-8-1-8` 交替或 `5-5-5-5` 连打，全部具有完全相同的 `types = frozenset({'tap'})`（[`logic_probe.json:50`](../../.scratch/repetition_v29_20261030/logic_probe.json#L50) `identical_features_single_vs_alternating_chord: true`）！`type_count` 实际上退化成了“距离上一个 Hold/Slide/Touch 过了多少步”，无法提供任何“同一 Shape/配置模板已重复多少次”的信号。
3. **`motion_head` 的 `REVERSE`（`before`）通道在 `1-8` 交替中满额激活且幅度达 `3.91`，压倒 `0.257` 的 `lane_load`（[`prefix_probe.json`](../../.scratch/repetition_v29_20261030/prefix_probe.json)）**：
   - 在 `1 -> 8` 后，`before = 1, after = 8`，满足 `before != after`，因此 `motion_distinct_v2` 的第 1 通道（`REVERSE` = `before`）将 100% 概率质量放在车道 `1` 上；下一步 `8 -> 1` 后又将 100% 质量放在车道 `8` 上。
   - 在 `prefix_probe.json` 的 41 个 `1-8` 交替位置上，`motion_head` 的相位 logit 跨度高达 **`3.91`**（从旧 400 步的 `2.93` 进一步膨胀），将延续交替相位的概率从无 motion 时的 **`0.450` 拉高到 `0.902`**；而新增的 `lane_load_projection` 范数仅 `0.057`，产生的相位偏置跨度仅 **`0.257`**，相差 **15 倍**！
   - 但同时，[`teacher_probe.json:77-193`](../../.scratch/repetition_v29_20261030/teacher_probe.json#L77-L193) 证明 `motion_head` 在 16 张验证谱的 13 张上改善了相位 NLL（`1.680 -> 1.634`，877 行变好 vs 546 行变差），[`motion_rollout.json`](../../.scratch/repetition_v29_20261030/motion_rollout.json) 也证实仅关掉 `motion_head` 虽能把 `fast_jack` 从 `12/7` 降到 `5/3`（另一 case 从 `28` 降到 `9`），却不能解决 Hold 与节奏锁死。因此**不能一刀切关闭 `motion_head`，而必须对其进行幅度受限与交替衰减改造**。

### 3. 为什么 `exit_label` 与 `recovery_batch` 没能建立有效的“闭环切出”防线？（[`logic_probe.json:1-95`](../../.scratch/repetition_v29_20261030/logic_probe.json#L1-L95)）
1. **`exit_label` 的三处漏判（[`stream_continuation.py:75-89`](../../src/maichart/composer/stream_continuation.py#L75-L89)）**：
   - **排除 `WAIT` / `EOS`**：`if target.kind != 2: return False, False` 导致长连打或长 Touch 段通过休止（`WAIT`）自然结束的真实边界完全没有计入 `transition_exit_loss`（`tap_to_wait: [false, false]`）。
   - **同键转 Hold / 转双押漏判**：`repeated_buttons - buttons` 在 `1, 1, 1 -> 1h[4:1]` 或 `1, 1, 1 -> 1/2` 时因目标仍含键 `1` 而判为 `False`（`same_button_tap_to_hold: [true, false]`），且 `self.buttons[1]` 继续累加。
   - **双键交替（`1-8-1-8`）与同 Shape 重复无 exit 标签**：因为 `repeated_buttons` 只看单步连续同键，双键交替退出时 `button_eligible` 为 `False`。
2. **`recovery_batch` 轨迹过长导致覆盖极窄，且跨越出口后丢失出口监督（[`logic_probe.json:52-75`](../../.scratch/repetition_v29_20261030/logic_probe.json#L52-L75)）**：
   - **覆盖率极低**：`MAX_EPISODE_STEPS = 64`，每 25 update 只走 `steps = 2` 步，一条 episode 要占用 $32 \times 25 = 800$ 步训练！到 `step 6500` 累计仅走过 **19 个独立 episode**（`anchor_cursor: 19`）。
   - **越过出口（Missed Exit）盲区**：锚点设在 `index - 2`，当模型在 `no_grad` 下采样 2 步后，若生成的第 2 步时间已经跨过了原谱在 `index` 处的出口事件（如 `logic_probe.json:52-58` 中错过了 `beat 1/2` 的 `1h[4:1]`），`_next_reference` 只找 `s.end > cursor` 的下一个真谱事件（`beat 3/4` 的 `2`）。结果：**模型在 `beat 1/2` 错过出口的错误决策处于 `no_grad` 中，而后续有梯度的监督步教的却是错过出口之后的普通单键 `2`**！

---

## 二、 保持正常流线、无硬上限、无高昂串行 Rollout 的学习侧根治方案

针对上述定量证据，我们不需要任何生成端硬重复上限，也不需要昂贵的串行长轨迹 Rollout，只需在**特征直连、运动先验有界化、并行前缀扰动（In-Batch Parallel Prefix Perturbation）、以及短平快同拍点恢复**四个层面上做紧凑修改：

### 方案 1：把精确游程与疲劳特征直接接入决策头（1-Hop Direct Head Conditioning）
- **问题根源**：`continuation_features` 和 `touch_load` 只加在 4 层 Decoder 底部的 `x` 上，被已收敛的 Transformer 层稀释（清零 TV 仅 `0.000089`）。
- **改法（极小改动，梯度一步直达）**：
  1. **直接接入 `repeat_delta_log_probs`（[`stream_model.py:313-320`](../../src/maichart/composer/stream_model.py#L313-L320)）**：
     让继续/切换门控 $g$ 直接读取当前的游程特征：
     $$g = \text{keep\_gap\_head}(h) + \text{keep\_gap\_audio}(k_{\text{prev}} - \bar{k}) + \text{keep\_gap\_run}(\text{continuation\_features})$$
     这样在 `transition_exit_loss` 下，`keep_gap_run`（$D_{\text{run}} \to 1$）直接收到一阶梯度，立刻学会当 `gap_count` / `gap_elapsed` 达到乐句长度且声学无强支撑时压低 $g$。
  2. **直接接入 `action_context_at`（[`stream_model.py:331-340`](../../src/maichart/composer/stream_model.py#L331-L340)）**：
     在 `action_context_at` 中直接加上 `self.action_continuation_projection(continuation_features)`（或将 `continuation_features` 拼入 `action_fusion`），使 `shape_head`、`event_phase_logits` 和 `motion_head` 无需穿透 4 层 Decoder 就能直接读取当前的 `touch_count`、`shape_run`、`button_run` 和 `trill_run`。

### 方案 2：扩展 `ContinuationState` 覆盖双键交替（Period-2 Trill）与精确 Shape/配置游程，并修正 `exit_label`
- **改法（[`stream_continuation.py`](../../src/maichart/composer/stream_continuation.py)）**：
  1. **用“精确配置模板游程（`shape_run` / `pattern_run`）”替代粗粒度 5 类 `note_type` 游程**：
     - 将 `types` 改为 `(shape_id, detail_tuple_or_kind)`（或同时跟踪 `same_shape_count` 与 `touch_count`），使连续单键 Tap（`shape=1`）、连续特定 Hold（如 `8h[8:1]`）、连续双押各自独立计数，不再把整首曲子的不同常规配置混为一谈。
  2. **新增双键/两步交替游程（`period2_trill_count` & `period2_trill_elapsed`）**：
     - 在 `ContinuationState` 中保留 `prev_buttons` 和 `prev2_buttons`。若当前步 `buttons` 非空且 `buttons <= (prev_buttons | prev2_buttons)` 且 `len(prev_buttons | prev2_buttons) <= 2`（即始终在至多 2 个按键之间连打或交替，覆盖 `5-5-5-5` 单键连打与 `1-8-1-8` 双键交互），则累加 `trill_count` 与 `trill_elapsed`！
     - 这样 40 步 `1-8` 交替的游程计数将从原来的 `1` 变成真实的 `40`！
  3. **修正 `exit_label` 的三处漏判**：
     - 当 `gap_eligible`、`touch_eligible`、`button_eligible` 或 `trill_eligible`（`trill_count >= 4`）为真时：
       - 若 `target.kind == WAIT`（进入休止），**判定为合法 `exit = True`**！
       - 若单键连打（`count >= 2`）或双键交替（`trill_count >= 4`）后，下一步的 `(shape, buttons)` 改变了按键集合（`buttons != prev_buttons` 且跳出 `{prev_buttons | prev2_buttons}`）或改变了配置形态（如单键 Tap 转为 Hold `1h` 或双押 `1/2`），**均判定为 `exit = True`**！

### 方案 3：改造 `motion_head` 为“有界 + 含新鲜车道第 4 通道”的旋转等变先验（保留 13/16 谱面增益，消除 `3.91` 幅度绑架）
- **为什么不能只关掉 `motion_head`**：`teacher_probe.json` 证实它对常规流线相位很有帮助（13/16 谱改善），但它之所以在 `1-8` 交替中把继续交替概率从 `0.450` 推到 `0.902`，是因为：
  1. `motion_head` 输出无界（跨度达 `3.91`），压倒了 `lane_load`（`0.257`）；
  2. 当前 3 个通道（`KEEP=(2*after-before)%8`、`REVERSE=before`、`STAY=after`）**全部只奖励“延续过去两步的运动轨迹或停在旧键”，没有任何一个通道用于表征“跳出最近两步按键集合的新鲜车道（Fresh Lanes）”**！
- **改法（[`stream_model.py:434-460`](../../src/maichart/composer/stream_model.py#L434-L460)）**：
  1. **增加第 4 个旋转等变通道 `FRESH`（跳出 `{before, after}` 的其余车道均匀分布）**：
     - 在 `motion_phase_features` 中，除原有的 `(KEEP, REVERSE, STAY)` 外，定义第 4 通道：对不属于 `first` 和 `second` 的其余车道赋予归一化正质量。 when 模型处于长交替（`trill_count` 高）或按键疲劳时，模型可以直接给第 4 通道输出正权重（或给 `REVERSE`/`STAY` 输出负权重），从而自然且平滑地跳出 `{1, 8}` 死循环，而不是只能在 `KEEP/REVERSE/STAY` 三者中打转！
  2. **对 `motion_head` 与 `lane_load_projection` 实施软幅度对称约束**：
     - 将 `motion_head(context)` 改为有界输出：`1.5 * torch.tanh(self.motion_head(context))`（最大跨度控制在 `[-1.5, +1.5]`，避免出现 `3.91` 的极端饱和 logit 锁死），同时让 `lane_load` 直接读取包含近期击打频次的负荷特征，使负荷抑制与运动先验处于同一数量级（`~1.0` logit）。

### 方案 4：用“主训练批内因果前缀扰动（In-Batch Prefix Perturbation）”+“锚点同拍一步恢复”替代稀疏 64 步长轨迹 Recovery
- **核心问题**：为什么 Teacher Loss 无法验收闭环，且 ` continuation_projection` / `touch_load_projection` 学不出来？
  因为 **Teacher 前缀里只有“合法的音乐长跑”（此时目标也是继续长跑），几乎没有“在应当切回常规谱面的音乐位置上，历史前缀却处于 `1-8-1-8` 交替、单键连打或纯 Touch 循环”的样本**！而 `recovery_batch` 走 64 步长轨迹不仅太慢（6500 步仅 19 条），还会因为 `cursor` 漂移而跨过真实出口（`logic_probe.json:52-58`）。
- **无串行 Rollout 的高效闭环训练方案（二选一或结合，均无长串行开销）**：
  1. **方案 4A（零额外前向开销）：训练批内合成循环前缀扰动（In-Batch Synthetic Loop Prefix Perturbation）**：
     - 在主训练构造样本（或 Collate / Forward 前）时，以小概率（例如 `15%~20%` 的样本块）选择目标区中的某个真实非循环 NOTE 步 $t$（例如原谱中正常的 Tap/Hold/Slide 步），将 $t$ 之前的 $M \in [4, 12]$ 个历史步（仅修改 `history_*` 输入及对应的 `_prefix_load`、`motion_phase_features`、`continuation_features`，**严格保持各步的 `step_beat`、`step_second`、音乐特征以及第 $t$ 步的真实 `target_*` 不变**）替换为以下四类自生成死锁前缀之一：
       - **双键交替前缀**：将前 $M$ 步历史设为 `k1 <-> k2` 交替（如 `1, 8, 1, 8...`）；
       - **单键连打前缀**：将前 $M$ 步历史设为同一按键 $k$（如 `5, 5, 5, 5...`）；
       - **纯 Touch 前缀**：在非 Touch 乐段将前 $M$ 步历史设为随机 Touch（`shape=0`，如 `E4, E5, B5...`）；
       - **等间距锁死前缀**：在变奏乐段将前 $M$ 步 `history_delta` 设为恒定 `1/4` 或 `1/2`。
     - **为什么这能立竿见影地打破 Teacher 共线性？**
       - 因为第 $t$ 步的真值 `target` 是对齐在真实音乐重音上的正常谱面事件（不是循环的延续），模型在**每一次普通 Batch 前向**中都会直接面对：“当历史已经陷入 8 步 `1-8` 交替 / 8 步 Touch / 8 步单键连打，但当前音乐并不是这类段落时，必须依靠 `continuation_features`、`touch_load`、`lane_load` 和 `music` 立即输出真值配置并切出循环”！
       - 这完全不需要串行 GPU Rollout（开销为 **0**），也不会引入任何硬上限规则（因为在原谱本来就是长连打/长 Touch 的未扰动样本上，模型依然正常学习保持长连打）。
  2. **方案 4B（修复 `recovery_batch` 的覆盖率与越过出口 Bug，[`stream_recovery.py`](../../src/maichart/composer/stream_recovery.py)）**：
     - **将 `MAX_EPISODE_STEPS` 从 `64` 缩短为 `2`（即每次调用都从新的风险锚点出发走 1~2 步后立即监督并轮换新锚点）**：这样每 25 更就能覆盖 2 个全新锚点，4000 步可覆盖 **320 个独立风险出口**（而不是 19 个）！
     - **锚点设在 `index - 1`（出口前 1 步），走 `1` 步自产采样后，若模型错过了 `index` 处的真实出口，直接强制在真实出口步 `index` 的前缀/时刻上监督该出口事件（或用模型刚采样出的那 1 个错误事件替换 `index - 1` 的历史，去监督 `index` 处的真实出口）**！这样永远不会发生 `logic_probe.json:52-58` 中“采样越过 `beat 1/2` 出口导致 `1h[4:1]` 被跳过”的漏监督问题。

---

## 三、 落地优先级与快速验证建议

为快速验证闭环效果，建议按以下最小闭环组合实施并用同一套 `prefix_probe` / `motion_rollout` 复测：

1. **必改项 1（修复特征盲区与直连通路）**：
   - 在 [`stream_continuation.py`](../../src/maichart/composer/stream_continuation.py) 中增加 **双键交替游程（`period2_trill`）** 与 **精确 `shape` 游程**，并修正 `exit_label` 纳入 `WAIT`、同键转 Hold/双押、以及双键交替退出。
   - 在 [`stream_model.py`](../../src/maichart/composer/stream_model.py) 中将 `continuation_features` 直接接入 `repeat_delta_log_probs`（`keep_gap` 门控）和 `action_context_at`（`shape_head` / `phase_head` / `motion_head`），不再只埋在 4 层 Decoder 底部。
2. **必改项 2（约束 `motion_head` 幅度并补齐跳出通道）**：
   - 对 `motion_head` 输出加 `1.5 * tanh(...)` 幅度上限（阻止 `3.91` 跨度绑架相位分布），并在 `motion_phase_features` 中加入第 4 通道 `FRESH = ~(first | second)`（或当 `before` 在近 4 步已重复时衰减 `REVERSE` 通道）。
3. **必改项 3（打破 Teacher 共线性：短锚点 Recovery + 批内循环前缀扰动）**：
   - 将 `stream_recovery.py` 的 episode 改为 1~2 步短周期轮换且禁止跳过目标 `exit` 事件；同时在训练批中加入小比例（如 15%）的合成循环前缀扰动，让零初始化的 `continuation` / `load` 分支在每个训练步都能收到强烈的“跳出死锁”梯度。
4. **验收标准（直接对比 `prefix_probe.json` 与 `motion_rollout.json`）**：
   - 1000~1500 步短训后，重跑 `prefix_probe.json`：验证清零 `run_features` 的 `shape` TV 从 `0.000089` 显著提升至 `> 0.05`，41 步 `1-8` 交替前缀上的 `motion` 相位跨度从 `3.91` 回落至 `<= 1.5`；重跑 `motion_rollout.json` 12 条闭环序列，验证开启 `motion_head` 时 `fast_jack` 与 `period2_rate` 同步下降，同时 `teacher_probe.json` 的 16 谱相位 NLL 保持优于关闭 `motion_head`。
