# Composer 异常 32/48 分音符、大段空白与历史自感知（对比 `event_ctx`）根因排查及修复交接文档

**日期**：2026-10-29  
**针对问题样本**：`webui_workspace/generated/4a69b3df555b/maidata.txt`（由 `checkpoints/prism_composer_v21/composer_last.pt` 生成，BPM 190，Lv 14.4）  
**典型异常片段**：
```simai
{16}2,1,{8}8,{32}1/7,2/6h[32:3]/B3,{16},{32}5h[8:1],4x,{16},{32}3,2,{16},{8}3,7/8,1/6,
```

---

## 一、 问题现象与量化取证

对 `webui_workspace/generated/4a69b3df555b/maidata.txt` 及配套诊断文件 `maidata.composer.json`、`4a69b3df555b_composer_curve.json` 进行逐拍拆解后，定位到三类核心异常：

### 1. 畸形 `{32}` / `{48}` 切分与异常 Hold 时值（如 `2/6h[32:3]/B3`、`Ch[48:1]`）
- **节拍级复盘（以第 48 小节 `beat 192..196` 为例）**：
  - `beat 192` (`{16}2`) $\to$ `192 + 1/4` (`1`) $\to$ `192 + 2/4` (`{8}8`)
  - `beat 193` (`{32}1/7`) $\to$ `193 + 1/8` (`2/6h[32:3]/B3`)：**步长 $\Delta = 1/8\text{ beat}$（即 `{32}` 分音符，BPM 190 下仅 $39.5\text{ ms}$）**，且在该 32 分反拍上塞入了双押 + `[32:3]` 超短 Hold + Touch `B3` 的三复合配置。
  - `beat 193 + 1/8` $\to$ `193 + 4/8` (`5h[8:1]`)：**步长 $\Delta = 3/8\text{ beat}$**，Simai 编译器被迫输出 `{16},{32}` 填补 $3/8$ 拍空隙。
  - 紧接着在 `193.5 -> 193.625 -> 194.0` 与 `194.0 -> 194.125 -> 194.5` 连续三次重复 `1/8 拍 + 3/8 拍` 的 `{32}x,y,{16},` 抽搐切分。
- **全谱统计对比**：
  - 全量训练语料（`781,465` 个事件）中，`beat.denominator == 8`（`{32}` 反拍）仅占 **`1.48%`**，且集中在低 BPM 曲目或连续 32 分交互流中。
  - 而在 `4a69b3df555b`（596 个事件）中，`beat.denominator == 8` 高达 **39 次（`6.54%`）**，`interval_denominator_counts["8"]` 高达 **76 次（`12.8%`，甚至超过了 `{16}` 分音符的 75 次！）**，另有 **5 次 `denominator == 12`（即 `{48}` 分音符，$26.3\text{ ms}$）**，以及 `6h[32:3]`、`5h[32:1]`、`Ch[48:1]` 等罕见畸形 Hold 时值。

### 2. 曲中与曲尾的大段空白（Blank Stretches）
在 `4a69b3df555b` 中出现多处非预期的长空白：
- **曲中活跃段空白**：
  - `beat 76 -> 92`（`24.00s -> 29.05s`）：在曲目活跃段（强度曲线 `0.50`，规划密度 `~4.2 notes/s`）突然跨跳 **16 拍（5.05 秒）** 空白。
  - `beat 116 -> 124`（`36.63s -> 39.16s`）与 `beat 332 -> 340`（`104.84s -> 107.37s`）：在活跃段突然跨跳 **8 拍（2.53 秒）** 空白。
  - 全谱共出现 14 次 $\ge 4$ 拍（$\ge 1.26\text{ s}$）的空白跳跃。
- **曲尾 4.5 秒空白 + 末尾孤立散音（`156.24s -> 160.74s`）**：
  - 用户强度曲线在 `157.0s .. 160.74s` 已设为 `0.0`（尾奏静音），模型在 `156.24s`（`beat 494.75`）已给出 `eos_probability >= 0.98`。
  - 但 EOS 被 `curve_activity_remaining` 否决，随后因 `157s..160.74s` 强度为 0，被迫跳过 **14.25 拍（4.50 秒）** 在音频结束前 `0.01s`（`160.7368s`）硬塞了一个孤立的 `,3,E`。

---

## 二、 根因深度剖析

### 根因 1：为何生成了不合理的 `{32}` / `{48}` 分音符与 `[32:3]` 畸形配置？

1. **[`src/maichart/composer/candidates.py`](../../src/maichart/composer/candidates.py#L28-L56) 中 `is_playable_candidate_transition` 存在恒真短路漏洞（Tautology Bug）与 `{32}` 反拍无约束放行**：
   - 第 47 行已检查 `if beat.denominator not in CANONICAL_BEAT_DENOMINATORS: return False`。
   - 第 55 行却写成：
     ```python
     return delta.denominator in CANONICAL_BEAT_DENOMINATORS or beat.denominator in CANONICAL_BEAT_DENOMINATORS
     ```
     由于右侧 `beat.denominator in CANONICAL_BEAT_DENOMINATORS` 在第 47 行已保证为 `True`，**第 55 行恒为 `True`（对 `delta.denominator` 的检查完全失效）**！
   - 结果：当 `prev` 处于三连音网格（如 `1/3`、`2/3`，即 `{12}`）而候选 `beat` 处于二连音网格（如 `3/4`，即 `{16}`）时，$\Delta = 3/4 - 2/3 = 1/12\text{ beat}$（Simai `{48}` 分音符，BPM 190 下仅 $26.3\text{ ms}$）被无条件放行，直接编译出 `{12}5,{48}4,{12},{16}5`。
   - 同时，`CANONICAL_BEAT_DENOMINATORS = {1, 2, 3, 4, 6, 8}` 将 `denominator == 8`（`{32}` 分音符，即 $1/8$ 拍）当作无条件合法的通用网格点：既允许从整拍跳 $\Delta = 1/8$ 后再跳 $\Delta = 3/8$（形成 `{32}x,y,{16},` 的孤立二连抽搐），也允许从任意拍直接跨 $\Delta = 3/8, 5/8, 7/8$ 落到 `{32}` 反拍上。

2. **[`src/maichart/composer/generate.py`](../../src/maichart/composer/generate.py#L154-L173) 中 `_truncate_probabilities` 与 `_conserved_action_probabilities` 存在 `<= 16` 截断旁路漏洞**：
   - `_truncate_probabilities` 第 156 行写有：
     ```python
     if probabilities.numel() <= 16:
         return probabilities
     ```
   - `_conserved_action_probabilities` 第 345 行写有：
     ```python
     counts = (~padding & torch.isfinite(packed)).sum(-1)
     within = torch.where((counts <= 16)[:, None], within, filtered)
     ```
   - **灾难性后果 A（Pointer 端 Bin 0 内部截断完全失效）**：
     Pointer 将 96 个候选分为 6 个距离桶（`BIN_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0)`）。其中 **Bin 0（$\Delta < 0.5\text{ beat}$）内总共只有 6~8 个候选**（`1/8, 1/6, 1/4, 1/3, 3/8` 等），恒满足 `numel() <= 16`！因此 Bin 0 内部的 `top_p=0.92, min_p=0.05` **从来没有生效过**，`1/8`（`{32}`）和 `3/8`（`{32}`）的尾部分布噪声被原封不动保留并采样！
   - **灾难性后果 B（Action 端 Hold 时值桶截断完全失效）**：
     在 `conserved_grouped_v2` 动作头中，每个宏桶（如 `(hold, pad_6, ring_0)`）内部的合法不同时值原子数通常只有 8~14 个（`<= 16`）。由于 `counts <= 16` 直接跳过 `top_p=0.85, min_p=0.08` 过滤，`6h[32:3]`、`5h[32:1]`、`Ch[48:1]` 等极低概率长尾畸形时值从未被截断！
   - **灾难性后果 C（动作重试失败后把合法的 `{16}` 拍拉黑并退化到 `{32}`）**：
     在 `generate.py` 第 930~1009 行，当指针选中 `193 + 1/4`（正常的 `{16}` 拍）后，若 6 次动作采样因抽到长尾非法动作（如高速 Slide 或冷却冲突）被 `EventGuard` 拒绝，第 1009 行执行 `logits[choice] = float('-inf')` 将 `193 + 1/4` 整拍屏蔽；由于 `_grounded_pointer_probabilities` 保持 Bin 0 的总概率质量不变，屏蔽 `1/4` 后，**Bin 0 的全部概率质量被瞬间转移到了同属 Bin 0 的 `1/8`（`{32}`）和 `3/8`（`{32}`）上**！

---

### 根因 2：为何会出现曲中与曲尾的大段空白？

1. **`_grounded_pointer_probabilities` 中粗粒度距离桶（Bin 0..5）完全未经温度缩放与尾部截断**：
   - 在 [`src/maichart/composer/generate.py`](../../src/maichart/composer/generate.py#L203-L212) 中：
     ```python
     joint = logits.softmax(-1)
     ...
     probabilities[members] = joint[members].sum() * within
     ```
   - `pointer_temperature`（如 `0.5`）和 `top_p`/`min_p` **仅作用于桶内 `within`**，而 6 个大桶（`<0.5, 0.5..1, 1..2, 2..4, 4..8, >=8` 拍）的桶间概率 `joint[members].sum()` 永远处于 **`T = 1.0` 无截断状态**！
   - 当训练早中期（如 Epoch 2）模型在活跃段给 Bin 4（`4..8` 拍）或 Bin 5（`8..128` 拍）留有 `1%~2%` 的 softmax 尾部噪声时，整首歌采样 600 次就会必然命中 6~12 次 Bin 4 / Bin 5，直接在副歌中央制造 8 拍（2.5s）乃至 16 拍（5.0s）的巨型空白！
2. **缺少活跃乐段长跨跳保护（Active-Span Long-Jump Guard）**：
   - 当模型自身的 Planner / Workload 以及外部强度曲线均明确指示当前至未来数秒处于高密度活跃段（非休止区）时，候选集中仍然允许直接跨跳 $\ge 4$ 拍（跳过整小节活跃音乐）的远距休止候选参与随机抽样。
3. **曲尾 `ComposerCurveController.has_future_activity` 线性插值余留尾巴导致假阳性 EOS 否决**：
   - 在 `4a69b3df555b_composer_curve.json` 中，控制点 `(155.0s, 0.3913)` 到 `(157.0s, 0.0)` 之间为线性下降斜坡，`157.0s..160.74s` 全为 `0.0`。
   - 在 `156.24s` 处，`has_future_activity` 从 `after_sec + 0.25 = 156.49s` 开始以 `0.5s` 步长探测；仅凭 `156.49s` 处单个插值点 `0.0997 >= 0.06` 就判定后续仍有活动并否决 EOS，随后又因为 `157.0s..160.74s` 强度为 `0.0`（接近休止），跨过 4.5 秒空白在 `160.7368s` 处塞入最后一颗散音。

---

### 根因 3：评估当前模型能否知道“自己刚写过什么”——与 `event_ctx` 的架构级对比

#### 1. `event_ctx`（[`src/maichart/generation/event_ctx.py`](../../src/maichart/generation/event_ctx.py) / [`event_model.py`](../../src/maichart/generation/event_model.py)）是如何做到的？
- **完整连续的局部前导窗口（Lead-in Window）**：`build_context_events` 将前序 8 拍（`context_beats = 8`）内已生成的全部事件序列化为与解码目标完全同构的细粒度 token 序列（`<ctx> AT:beat CHAR:... <event_end> ... <ctx_end>`），直接作为因果 Transformer Decoder 的前缀。
- **序列级整块 Dropout 而非随机打洞**：`DEFAULT_CONTEXT_DROPOUT = 0.15` 仅以 15% 概率在序列级丢弃整个前导块；在保留前导块的 85% 训练步中，**最近的 `t-1, t-2, t-3, t-4` 事件 100% 连续完整，绝不随机抹除 `t-1`**。
- **直接注意力感知**：Decoder 在预测下一个时间点 `AT:` 和每一个音符原子时，每一层都能直接通过多头自注意力看到最近几拍内：
  1. 刚写过的精确节拍网格与步长（`{8}` / `{12}` / `{16}` 延续性）；
  2. 最近多步（`t-4..t-1`）的按键轨迹（顺/逆时针流、交互 `ABA`/`ABAB`、同键纵连）；
  3. **Slide 的终点落键位置**（例如写完 `1-5[8:1]` 后，明确知道手已经滑向 5 号键）；
  4. 最近刚用过的配置类型（单点/双押/Hold/Slide/Touch 组合）。

#### 2. 当前 `chart_composer_v1`（`AudioGroundedComposer`）为何无法很好地知道自己刚写过什么？
排查 [`src/maichart/composer/grounded_model.py`](../../src/maichart/composer/grounded_model.py#L305-L408) 与 [`src/maichart/composer/events.py`](../../src/maichart/composer/events.py#L101-L112) 发现 **四大结构性失明点**：

1. **训练时对 `t-1` 施加了高达 40.5% 的毁灭性随机抹除**：
   - `symbolic_dropout = 0.15` 整体屏蔽符号分支 15%；
   - `history_dropout = 0.30` 对包括 `t-1`、`t-2` 在内的每个历史位置独立随机打洞；
   - 第 330 行 `tail_valid = (~safe_padding[rows, (lengths-1).clamp_min(0)]) & lengths.gt(0)`：只要 `t-1` 被 `history_dropout` 命中（30% 概率），**`tail`（`memory[t-1]`）、`last_heads` 以及 `previous_heads` 全部被乘以 `0` 清零**（完全不回退读取 `t-2` 或局部窗口）！
   - 叠加后（$1 - 0.85 \times 0.70 = 40.5\%$），**训练中 40.5% 的样本完全看不到 `t-1` 的按键和状态，66% 的样本在最近 3 步 `t-1..t-3` 中存在随机断洞**！
2. **`history_event_features` 对节拍切分（Subdivision）近乎失明**：
   - `history_event_features` 仅传入 `min(1.0, delta_beats / 4.0)` 和 `min(1.0, delta_seconds / 2.0)`。
   - 在 `delta_beats / 4.0` 线性压缩下，`{8}`（`0.125`）、`{12}`（`0.083`）、`{16}`（`0.0625`）、`{24}`（`0.042`）、`{32}`（`0.031`）全被挤压在 `0.03~0.12` 的极窄区间内，没有任何周期相位或对数步长特征，导致模型无法分辨自己刚写的是 `{12}`、`{16}` 还是 `{32}`！
3. **`last_heads` / `previous_heads` 仅保留 `t-1` 单步起手键位，丢失了 `t-2..t-4` 多步轨迹与 Slide 终点落键（Landing Pad）**：
   - `atom_head_lanes[last_ids].any(-2)` 只提取 `t-1` 的 8 个起手按键布尔掩码：
     - 看不到 `t-2, t-3, t-4`，因此无法判断当前是 `1 -> 2 -> 3` 的顺时针流，还是 `2 -> 7 -> 2` 的交互（极易退化为机械 `ABA` 或随机跳键）；
     - 完全丢弃了 **Slide 的终点落键（`end_position`）**——写下 `1-5[8:1]` 后，模型只知道起手在 `1`，完全不知道滑条终点落在 `5`！
4. **局部前导窗口（Local Lead-in）未与候选步长直接比对**：
   - Pointer 在给 96 个候选打分时，`symbolic` 向量是在 `audio_context` 上做了一次全局 `history_attention`（且门控 `history_gate` 初值仅 `0.1`），没有任何模块显式比对“候选步长 $\Delta_{\text{cand}}$ 与刚刚写下的最近几步步长 $\Delta_{t-1}, \Delta_{t-2}$ 是否构成连续等分律动（如连续 `{8}` / `{16}` / `{12}`）”。

---

## 三、 修复方案与架构设计

### 修复 1：彻底根除异常 `{32}` / `{48}` 切分与 `[32:3]` 畸形动作长尾

1. **修复 [`src/maichart/composer/candidates.py`](../../src/maichart/composer/candidates.py) 中的相位与 `{32}` 转移规则**：
   - 移除 `is_playable_candidate_transition` 第 55 行的恒真短路 `or beat.denominator in CANONICAL_BEAT_DENOMINATORS`，严格要求跨拍步长分母合法，禁止三连音反拍（`1/3, 2/3, 1/6, 5/6`）与二连音反拍（`1/4, 3/4, 1/8, 3/8, 5/8, 7/8`）之间发生直接产生 `1/12`（`{48}`）或 `1/24`（`{96}`）微步长的畸形跨网格跳跃；二/三连音切换必须在整拍或半拍锚点（`denominator in {1, 2}`）完成。
   - **严格约束 `{32}` 反拍（`denominator == 8`）的进出规则**：
     - **进入 `{32}` 反拍（`beat.denominator == 8`）**：仅允许从合法二进制网格通过真正的 32 分音符步长 `delta == Fraction(1, 8)` 进入（禁止跨 `3/8, 5/8, 7/8` 拍空白直接落到 `{32}` 反拍上形成 `{32},,,x`）。
     - **离开 `{32}` 反拍（`prev.denominator == 8`）**：一旦处于 `{32}` 反拍上，下一步必须通过 `delta == Fraction(1, 8)` 走完 32 分音符回到 `{16}`/`{8}`/`{4}` 网格（或当近距受阻时恢复到 $\ge 1$ 拍的整拍/半拍锚点 `beat.denominator in {1, 2}`），**严禁 `prev.denominator == 8` 后接 `delta = 3/8, 5/8, 7/8` 形成 `{32}x,y,{16},` 的二连抽搐**！
   - **在生成端（`generate.py`）结合物理拍速与局部上下文约束 `{32}`**：
     - 当单步 $1/8$ 拍的物理时长过短（如 BPM 190 下 $1/8\text{ beat} = 39.5\text{ ms} < 45\text{ ms}$）且前一步并非处于连续 32 分流（`prev.denominator != 8`）时，在有常规 `{16}`/`{12}`/`{8}` 候选可用的情况下不主动开启超高速 `{32}` 孤立碎音；
     - 若处于 $1/8$ 拍（`{32}`）极短步长上，限制动作采样仅允许单键 Tap（禁止在 $39\text{ ms}$ 的 32 分音符上塞入多押 Chord / Hold / Slide）。

2. **修复 [`src/maichart/composer/generate.py`](../../src/maichart/composer/generate.py) 中的 `<= 16` 截断旁路漏洞与非常规时值原子过滤（`_canonical_duration_mask`）**：
   - 将 `_truncate_probabilities` 中的 `if probabilities.numel() <= 16: return probabilities` 修正为 `if probabilities.numel() <= 1: return probabilities`。
   - 将 `_conserved_action_probabilities` 中的 `within = torch.where((counts <= 16)[:, None], within, filtered)` 修正为 `within = torch.where((counts <= 1)[:, None], within, filtered)`。
   - 新增 `_canonical_duration_mask`：当当前按键存在标准 Simai 时值分母（`{1, 2, 3, 4, 6, 8, 12, 16, 24}` 且单音持续 $\le 8$ 拍）的合法动作原子时，自动屏蔽词表中来自变速/演出谱的 `[32:...]`、`[48:...]`、`[64:...]` 或超长 `[4:15]` 等畸形时值原子，彻底杜绝 `2/6h[32:3]/B3`、`A8h[48:1]`、`2<4[32:13]` 等异常配置。

---

### 修复 2：消除曲中与曲尾的大段空白

1. **在 `_grounded_pointer_probabilities` 中支持桶间（Coarse Bin）温度缩放与 `bin_min_p` 尾部剪枝**：
   - 为 `_grounded_pointer_probabilities` 增加可选参数 `bin_temperature=1., bin_min_p=0.`（默认 `1.0 / 0.0` 保持现有单元测试的精确桶质量守恒契约）。
   - 在实际推理生成（`generate()`）调用时，传入与 `pointer_temperature` 联动的 `bin_temperature` 与 `bin_min_p=0.04`，使得当模型将绝大部分概率集中在近距桶（Bin 0/1/2，`< 2` 拍）时，远距桶（Bin 4 `4..8` 拍、Bin 5 `>= 8` 拍）上的 `1%~2%` 尾部噪声被彻底清零，不再被 `multinomial` 误抽中。
2. **新增活跃乐段长跨跳保护（`_suppress_active_span_long_jumps`）**：
   - 在 `generate()` 中，若候选步长 $\Delta \ge 4.0\text{ beats}$（且 $\Delta t \ge 1.2\text{ s}$），检查当前已提交时间 $t_{\text{prev}}$ 到候选时间 $t_{\text{cand}}$ 之间的中间窗口 $[t_{\text{prev}} + 0.25, t_{\text{cand}} - 0.25]$：
   - 若该中间窗口不在 `rest_region` 内，且 Planner / Workload / 强度曲线预测该区间处于明显活跃状态（如平均预期音符密度 $\ge 1.5\text{ notes/s}$ 且休止概率 $< 0.50$），则在存在近距合法候选（$\Delta < 4.0\text{ beats}$）时屏蔽此类跨越多秒活跃乐段的异常长跳。
3. **修复曲尾强度曲线斜坡导致的 EOS 假阳性否决与尾奏零强度落点**：
   - 在 [`ComposerCurveController.has_future_activity`](../../src/maichart/composer/curve.py#L303-L307) 中，不再因 `after_sec + 0.25s` 处向 `0.0` 尾奏过渡的单个线性插值尾巴点而否决 EOS，而是要求后续存在实质性的活动采样点（且若剩余未休止区间最大强度或积分不足以支撑新音符则允许正常结束）。
   - 在 `ComposerCurveController.rest_mask` 中，当曲线尾段（如 `157.0s .. 160.74s`）目标强度已降至 `<= 0.015`（静音尾奏）时，即便处于 `follow` / `hard` 模式，也将尾部零强度静音段视为不可落点的尾奏休止区，彻底杜绝跨过 4.5 秒静音在音频末尾 `160.74s` 硬塞孤立音符。

---

### 修复 3：借鉴 `event_ctx` 升级模型局部历史自感知能力（`_LocalLeadInContext` + Slide 落键几何）

为使 `AudioGroundedComposer` 能够像 `event_ctx` 一样清晰感知“自己刚刚写下的几步配置与节拍律动”，在保持现有模型配置与旧权重兼容性的前提下，进行三项核心升级：

1. **在 [`CompositionalActions`](../../src/maichart/composer/action_factors.py) 中预计算每个动作原子的终点落键几何（`atom_end_lanes`）**：
   - 除原有的起手按键布尔掩码 `atom_head_lanes`（`[vocab, 8]`）外，解析所有 Slide 分支的最终落点 `end_position` 以及 Tap/Hold 的按键位置，注册非持久化缓冲区 `atom_end_lanes`（`[vocab, 8]`）与 `_normalized_end_lanes`（`[vocab, 8]`）。
   - 由此，模型在写下 `1-5[8:1]` 后，不仅知道起手在 `1` 号键，还能精确知道滑条终点落在 `5` 号键。
2. **新增局部前导窗口自感知模块 `_LocalLeadInContext`（[`src/maichart/composer/grounded_model.py`](../../src/maichart/composer/grounded_model.py)）**：
   - 对齐 `event_ctx` 的 8 拍局部前导思想，提取最近 $K = 8$ 个未掩码历史事件（`t-1, t-2, ..., t-8`）：
     - **摆脱单点 `t-1` 丢失致盲**：即使 `t-1` 在训练时被 `history_dropout` 掩码，模块仍能从最近未掩码的局部前导事件序列中提取完整的时序记忆与按键状态，彻底解决 40.5% 训练步完全失明的问题。
     - **多步按键与落键轨迹编码（Multi-step Head & Landing Trajectory）**：显式编码最近 4 步（`t-1..t-4`）的起手键位分布、Slide 终点落键分布、相邻步同键重复（Jack）、隔步往返（`ABA` / `ABAB`）、顺/逆时针流向（CW/CCW Stream）以及上一步 Slide 终点与当前起手的衔接关系。
     - **局部律动延续性比对（Rhythm & Subdivision Continuity）**：提取最近前导事件的步长特征（对数步长、切分网格相位）并与每个候选位置的步长 $\Delta_{\text{cand}}$ 直接计算律动匹配特征（判断候选步长是否延续了刚写下的 `{8}` / `{12}` / `{16}` 律动网格）。
   - 该模块通过零初始化输出投影层（`out_proj`）注入 `symbolic` 上下文，既输入给 Pointer 残差分支（`history_pointer` / `history_bin`），也输入给 Action 解码分支（`action_context` -> `decode_candidate` / `action_slot`）。
3. **升级跨事件相对键位关系（`cross_event_relation_policy = 'relative_heads_v1'`）**：
   - 引入可学习的零初值门控，将 `t-1` 的 Slide 终点落键几何与 `t-2` 的近期起手轨迹融合进 `previous_heads` 表征，使第一槽位（Slot 0）的跨事件相对键位打分同时感知起手键位与滑条落点。

---

## 四、 实测验证对比（复现用例 `4a69b3df555b`）与训练建议

### 1. 同参数实测对比（`4a69b3df555b`，`seed=1517101319`，`BPM=190`）

| 指标 | 修复前（`4a69b3df555b/maidata.txt`） | 修复后（实测验证） |
| :--- | :--- | :--- |
| **总事件数 (`event_count`)** | `596`（被多段长空白吞噬近半） | **`1188`**（完整覆盖全曲活跃段） |
| **`{32}` 事件数 (`beat_denom["8"]`)** | `39` | **`0`** |
| **`{32}` 步长间隔数 (`interval_denom["8"]`)** | `76` | **`0`** |
| **`{48}` 步长间隔数 (`interval_denom["12"]`)** | `5` | **`0`** |
| **畸形时值 (`[32:...]` / `[48:...]` / `[4:15]`)** | 多处（如 `2/6h[32:3]/B3` 等） | **`0`** |
| **曲中/曲尾 $\ge 4$ 拍大段空白数** | `14` 处（最长 16 拍 / `5.05s`） | **`0` 处** |
| **尾奏收尾时间 (`last_head_sec`)** | `156.24s` 空 4.5s 后跳至 `160.74s` 孤立尾音 | **`156.47s` 干净收尾（`learned_eos`）** |

### 2. 训练与验证建议

1. **直接 Fresh 训练（不要断点重训）**：
   - 由于本次修复增强了 `AudioGroundedComposer` 的 `_LocalLeadInContext` 局部前导自感知模块、Slide 终点落键几何以及相位合法转移契约，**若重启训练，请直接在新的 `checkpoint_dir` 下启动 `fresh` 训练，不要从旧的 `prism_composer_v21` 断点恢复（resume）**。
2. **推理端即刻生效的修复**：
   - 即使使用现有已保存的 checkpoint 进行生成，本次在 `candidates.py`、`generate.py` 和 `curve.py` 中落地的 `{32}`/`{48}` 相位转移约束、`<= 16` 截断漏洞修复、非常规时值原子过滤、距离桶尾部剪枝、活跃段长跨跳保护以及尾奏静音区 EOS 修正也**立即生效**，可直接消除 `{32}` 抽搐碎音、`[32:3]` 畸形 Hold 以及曲中/曲尾大段空白。
# 后续复审入口

本次实现已被接续审查，功能保留；近期历史可见性、恢复采样对齐和长跳条件已进一步补齐。当前行为、验证范围和 v21 训练状态见 [分音与前导复审交接](COMPOSER_RHYTHM_REVIEW_HANDOFF_20261029_ZH.md)。

