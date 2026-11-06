# `event_set_stream_v1` 全链路源码复审：关键缺口定位与高可行性落地方案（Antigravity 审查）

日期：2026-10-30  
审查范围：只读审查 `event_set_stream_v1` 全套源码（[event_set_codec.py](../../src/maichart/composer/event_set_codec.py)、[stream_data.py](../../src/maichart/composer/stream_data.py)、[stream_model.py](../../src/maichart/composer/stream_model.py)、[stream_train.py](../../src/maichart/composer/stream_train.py)、[stream_generate.py](../../src/maichart/composer/stream_generate.py)、[stream_sampling.py](../../src/maichart/composer/stream_sampling.py)、[stream_recovery.py](../../src/maichart/composer/stream_recovery.py)、[stream_checkpoint.py](../../src/maichart/composer/stream_checkpoint.py)、[stream_curve.py](../../src/maichart/composer/stream_curve.py)、[event_set_api.py](../../src/maichart/webui/event_set_api.py)）、[实现清单](../handoffs/COMPOSER_EVENT_SET_STREAM_IMPLEMENTATION_20261030_ZH.md) 及语料审计报告 [.scratch/event_set_rework_20261030/corpus_audit.json](../../.scratch/event_set_rework_20261030/corpus_audit.json)。\
遵循约束：未修改任何源码、未启动或停止任何任务、未运行 GPU。

---

## 零、总体结论：骨架方向非常干净，但存在 6 处必须补齐的“底层断路/语义错位”

本轮 `event_set_stream_v1` 的工程重构速度与整洁度极高：已真正落地了 `H=64 + K=16` 连续因果流、旋转等变集合编码（37 个 `shape` 覆盖 85.95% 常规事件且往返解析 0 错配）、对称相位 `logsumexp` 边缘化、显式 `WAIT(2b)` 步、因果 `SustainStateIndex` 跨 `WAIT` 传递、以及独立检查点隔离。**因果方向上未发现未来谱面泄漏**（`target_second` 严格在 `kind_logits` / `delta_logits` 之后才用于 `action_context_at`，`SustainStateIndex` 截止于 `step.start`）。

但通过逐行核对源码，发现当前实现中有 **6 处直接影响采音精度、走位连续性、休止判断与训练效率的关键缺口**（其中 3 处属于“底层特征已算好，但下游预测头根本没读到或算错了周期”的硬伤）：

1. **【选时盲区】`delta_head` 看不见非整拍候选点的局部 10ms 音频**：[`stream_model.py:248-259`](../../src/maichart/composer/stream_model.py#L248-L259) 中，`x` 只读取了 `beat + [0, 1, 2, 3, 4, 5, 6, 7, 8]` **9 个整拍点**的 `cache['local']`（`Conv1d` 感受野仅 `50ms`），而 `delta_head(hidden)` 是直接接在 `hidden` 后的 `nn.Linear(d, 141)`，**完全没有读取各候选 `beat + delta_values[c]`（如 `+1/4, +1/3, +1/2, +3/4` 拍）处的局部音频**！这意味着 [`stream_data.py:164-207`](../../src/maichart/composer/stream_data.py#L164-L207) 精心构建的 `10ms` 网格与 `5ms` 原始瞬态峰，在决定细分落点（`1/4`, `1/3`, `1/2` 拍）时对 `delta_head` 是**完全隐形**的。
2. **【时钟周期 Bug】`_clock` 中 `beats.sin(), beats.cos()` 漏乘 $2\pi$**：[`stream_model.py:123-126`](../../src/maichart/composer/stream_model.py#L123-L126) 直接对拍数算 `beats.sin(), beats.cos()`，其周期是 $2\pi \approx 6.2832$ 拍（无理数拍周期！），而非 1 拍、1/2 拍、1/4 拍或 4 拍小节；且 `self.decoder` 缺少序列内相对步/相对拍位置编码。
3. **【对称双押跨 `8→1` 边界的相位跳变】`phase_mask` 只修了对数似然，未修历史 `history_phase` 与 `reference_phase`**：当对向双押顺时针旋转 `3/7 → 4/8 → 1/5 → 2/6` 时，[`event_set_codec.py:88`](../../src/maichart/composer/event_set_codec.py#L88) 对 `1/5` 总是取最小索引 `phase = 0`（而非与上一拍 `3` 相邻的等价相位 `4`）。结果 [`stream_model.py:229`](../../src/maichart/composer/stream_model.py#L229) 计算出的历史 `movement` 从 `4/8` 到 `1/5` 变成了 `(0 - 3) % 8 = 5`（伪跳变 `-3` 格，而非真实 `+1` 格），且下一拍的 `reference_phase` 也被重置到 `0`。
4. **【规划头断路与 10ms `blank` 语义失真】**：[`stream_data.py:224-240`](../../src/maichart/composer/stream_data.py#L224-L240) 将 `_audio_plan` 建立在 `10ms` 极窄格上，导致即便在每秒 8 音的 16 分爆发段，也有 **92% 的 10ms 格满足 `events == 0`（`blank = 1`）**，`blank` 退化为“非起音帧”而非“音乐休止段”；`stream_loss` 未乘 `audio_plan_exposure`；且 `plan_logits` 在训练和默认生成中**完全没有接入 `kind_logits`（`WAIT` vs `NOTE`）或模型主干**。
5. **【滑条旋转覆盖率与细节头独立性】**：[`event_set_codec.py:54`](../../src/maichart/composer/event_set_codec.py#L54) 仅放行直线 `-`，将同样严格满足 $\mathbb{Z}_8$ 旋转等变的 `v, p, q, s, z, w` 及折线 `V` 全部打入 `literal` 回退（占全部事件的 14.05%），且回退事件 `shape=0` 会导致 `rotatable=False` 丢失相位参考；同时 [`stream_model.py:110, 312`](../../src/maichart/composer/stream_model.py#L110) 的 `detail_heads` 既不知道当前选中的 `phase`，也在多槽间独立采样（易产出 `1b/5` 半边 Break 或双滑条不同时值）。
6. **【闭环重试与 CPU 缓存瓶颈】**：[`stream_generate.py:145-159`](../../src/maichart/composer/stream_generate.py#L145-L159) 在 `sample_guarded_event` 盲抽 8 次失败后，直接强制插入 `WAIT(2b)` 跳过整整两拍；而 [`stream_data.py:381, 410`](../../src/maichart/composer/stream_data.py#L381) 将 `_banks` 和 `_states` 的缓存硬编码为 **4 首歌**，导致每轮训练/验证/恢复反复在 CPU 上重算 `10ms` 音频银行、`5ms` 原始动态峰与 `SustainStateIndex`。

---

## 一、逐题深度审查与源码证据

### 1. 首选路线是否有关键未落实或只做表面？

| 首选路线核心模块 | 当前源码落实状态 | 关键未落实 / 仅表面接通处（含源码行号） |
| :--- | :--- | :--- |
| **1) 整拍集合 + 旋转相位 + 对称折叠** | **80% 已落实**：`EventSetCodec` 实现了无序集合规范化与 `phase_equivalents`；`stream_loss` 用 `logsumexp` 边缘化等价相位 | **半闭环**：1）仅损失端边缘化了 `target_phase_mask`，而历史输入 `history_phase`（[`stream_data.py:430`](../../src/maichart/composer/stream_data.py#L430)）与 `reference_phase`（[`stream_model.py:227`](file:///simai_transformer/src/maichart/composer/stream_model.py#L227)）仍只存最小标量相位 `0`，导致对称双押跨 `8→1` 时 `movement` 算错；2）天然可旋转的 `v/p/q/s/z/w/V` 滑条被排除在旋转外。 |
| **2) 连续 16 步因果解码 + 共享音频** | **85% 已落实**：`StreamDataset` 按 `H=64, K=16` 切片，同歌共享 `encode_audio` | **选时未接局部音频**：`action_context_at` 在选定时间后读取了 `cache['local']`，但选时头 `delta_head(hidden)`（[`stream_model.py:259`](../../src/maichart/composer/stream_model.py#L259)）只靠 `hidden` 盲猜 141 个 `delta`，没有读取候选落点 `beat + delta` 处的 `cache['local']`！ |
| **3) 显式 `WAIT` 步 + 时间窗均衡** | **85% 已落实**：`build_steps` 注入 2 拍 `WAIT`，窗口总权重设为 `K / len(indices)` | **空窗与梯度裁剪副作用**：1）稀疏窗（仅 1~2 步）产生单步 `weight = 8.0~16.0` 的梯度尖峰，被 `clip_grad_norm_(1.0)` 压制，抵消了与密集窗（`weight = 0.16`）的平衡；2）`plan_logits` 的休止预测未接回 `kind_logits[WAIT]`。 |
| **4) 14 维频带/瞬态 4 拍相对变化** | **已计算但仅在整拍可见**：[`stream_data.py:210-220`](../../src/maichart/composer/stream_data.py#L210-L220) 正确计算了 14 维 4 拍落差并拼入 `238` 维特征 | 因 `near_audio` 只读 `offsets = [0,1,2,3,4,5,6,7,8]` 整拍点，半拍/四分之一拍处的瞬态对比未被选时头直接读取。 |
| **5) 逐键真实负担抑制高速同键复读** | **仅做表面**：[`stream_model.py:162-171`](../../src/maichart/composer/stream_model.py#L162-L171) `_prefix_load` 只算了 `1s / 4s` 衰减并投影进 `x` | 1）缺少 `0.15s`（16 分连打尺度）与逐键最近间隔；2）只加在选时前的 `x` 上，**未在确定 `target_second` 后对 `event_phase_logits` 的 8 个键位施加逐键负荷偏置**。 |
| **6) 两阶段预训练（先语法后配乐）** | **仅表面开关**：`--symbolic-pretrain-epochs` 将 `cache` 置零 | 置零期间 `near_audio` / `action_fusion` / `decoder` 偏置向零输入漂移，切回音频阶段时无门控缓冲，且 WebUI `event_set_api.py` 未暴露或默认关闭。 |

---

### 2. 因果泄漏、Phase/Shape/Detail 歧义、Padding/WAIT/EOS、窗口权重与闭环重试核查

#### (1) 因果性核查（通过，无未来泄漏）
- [`stream_data.py:172`](../../src/maichart/composer/stream_data.py#L172) 将列 107 的歌曲进度分母从“原谱末音符拍数”修正为 `beat_at_second(clock, duration)`（音频总长对应拍数），消除了末音符位置泄漏。
- [`stream_data.py:425-427`](../../src/maichart/composer/stream_data.py#L425-L427) 中 `SustainStateIndex.at` 使用 `cutoff = step.start + 1e-12`（当上一步为 `NOTE`）或 `step.start`（当上一步为 `WAIT`），严格只包含 `step.start` 之前已提交的动作。
- [`stream_model.py:258-274`](../../src/maichart/composer/stream_model.py#L258-L274) 中，`kind_logits` 与 `delta_logits` 完全由 `hidden`（仅含 `step_beat / step_second`）算出，`target_second` 仅在选时之后用于 `action_context_at`。

#### (2) 对称配置跨 `8→1` 边界的 `history_phase` / `movement` 伪跳变（必须修复）
- **源码定位**：[`event_set_codec.py:83-89`](../../src/maichart/composer/event_set_codec.py#L83-L89) 与 [`stream_model.py:218-230`](../../src/maichart/composer/stream_model.py#L218-L230)。
- **复现过程**：
  考虑最基础的对向双押顺时针旋转：`3/7 → 4/8 → 1/5 → 2/6`。
  1. `3/7` 的等价相位集为 `{2, 6}`，`codec.encode` 取 `min` 得到 `phase = 2`；
  2. `4/8` 的等价相位集为 `{3, 7}`，`codec.encode` 取 `min` 得到 `phase = 3`（此时 `movement = (3 - 2) % 8 = +1`，正确）；
  3. 到了 `1/5`，等价相位集为 `{0, 4}`，`codec.encode` 取 `min` 得到 **`phase = 0`（而不是顺延的 `4`）**！
  4. 于是 [`stream_model.py:229`](../../src/maichart/composer/stream_model.py#L229) 算出的 `movement = (0 - 3) % 8 = 5`（即 **`-3` 格跳变**）！紧接着在预测 `2/6` 时，`reference_phase` 变成了 `0`；而在生成端 [`stream_model.py:332-333`](../../src/maichart/composer/stream_model.py#L332-L333)，即便模型采样出了 `phase = 4`，存入 `StreamInferenceContext._append_step`（[`stream_data.py:531`](../../src/maichart/composer/stream_data.py#L531)）时再次调用 `self.codec.encode(step.text)`，又把 `event['phase']` 强行洗回了 `0`！
- **修复方式（极简 10 行）**：
  - 在 `EventSetCodec.encode(note_text, *, reference_phase=None)`（以及序列构建/推理 `_append_step`）中：当 `reference_phase is not None` 且 `phase_mask` 有多个等价相位时，选择 `phase_mask` 中**离 `reference_phase` 正向/圆周距离最近的等价相位**（例如 `old_phase = 3` 时，在 `{0, 4}` 中选 `4`）；
  - 同时在 `StreamComposer.forward` 中，历史相位嵌入改用 `history_phase_mask` 归一化均值（`phase_mask_float @ self.phase_embedding.weight`），让对称双押的历史嵌入天然具备旋转对称群不变性！

#### (3) `detail_heads` 缺少 `phase/shape` 条件与同拍槽间耦合（影响双押修饰与双滑条）
- **源码定位**：[`stream_model.py:110, 311-312, 365-369`](../../src/maichart/composer/stream_model.py#L365-L369)。
- **问题**：
  1. `head(ctx)` 输入的 `ctx` 是纯 `action_context`，**不包含刚选定的 `shape` 和 `phase`**！当模型为 `shape` 选择滑条终点或长按时值时，它不知道这个滑条是从 1 号键还是 5 号键出发（无法避开特定键位上的持续动作冲突）。
  2. 同一个 `shape` 的多个 `slot`（如双押 `1b/5b` 的两个 Break 修饰，或双滑条 `1-4[4:1]/5-8[4:1]` 的两个时值）是**完全独立**从 `head_0(ctx)` 和 `head_1(ctx)` 采样的！这会导致原本成对出现的 `1b/5b` 在采样时有 $2p(1-p)$ 的概率变成单边 Break `1b/5`，或双滑条出现不对称时值。
- **修复方式**：
  - 将细节上下文定义为 `detail_ctx = ctx + self.shape_embedding(shape) + (phase_mask_weights @ self.phase_embedding.weight)`；
  - 对 `slot > 0`，将前一个槽位 `slot - 1` 的 detail logit 分布（或嵌入）通过一个轻量线性层加到当前槽位的输入上（训练时 teacher-forcing 前一槽位 `target[:, slot - 1]`，生成时传入刚采样的 `details[slot - 1]`）。

#### (4) 窗口权重 `K / len(indices)` 与梯度裁剪 `clip_grad_norm_(1.0)` 的隐式抵消
- **源码定位**：[`stream_data.py:338, 402`](../../src/maichart/composer/stream_data.py#L338) 与 [`stream_train.py:273`](../../src/maichart/composer/stream_train.py#L273)。
- **问题**：
  - 在 8 秒时间窗内，高密段有 ~64~96 个步，被切成 4~6 个 chunk，每个步的 `window_weight = 16 / 96 ≈ 0.17`；
  - 安静段或尾段 8 秒窗内可能只有 1~4 个 `WAIT`/`EOS` 步，只生成 1 个短 chunk，每个步的 `window_weight = 16 / 1 = 16.0`（或 `4.0`），相差高达 **$24\times \sim 96\times$**！
  - 当 DataLoader 抽到含稀疏窗的微批次时，单步 `16.0` 的权重会产生巨大梯度范数，随即被 [`stream_train.py:273`](../../src/maichart/composer/stream_train.py#L273) 的 `clip_grad_norm_(model.parameters(), 1.0)` 强行压回 `1.0`；而抽到全密集窗微批次时，梯度范数小，几乎不被裁剪。结果**本应等权的静段在裁剪后实际有效步长反而缩水**。
- **修复方式**：
  - 对 `window_weight = self.targets / len(indices)` 增加合理上下界：`window_weight = min(4.0, max(0.25, self.targets / len(indices)))`；
  - 或者在构建样本时，不把仅有 4 个 `WAIT` 步的稀疏窗截断成长度为 4 的残缺 target，而是允许跨相邻安静窗拼接满 `K=16` 个目标步，使每个 batch 的有效目标步数与权重方差保持平稳。

#### (5) 生成与恢复中的 `sample_guarded_event` 盲抽失败导致强制跳空 2 拍（`forced_waits`）
- **源码定位**：[`stream_sampling.py:38-50`](../../src/maichart/composer/stream_sampling.py#L38-L50) 与 [`stream_generate.py:145-159`](../../src/maichart/composer/stream_generate.py#L145-L159)。
- **问题**：
  - `sample_guarded_event` 在固定 `nxt` 拍位上调用 `model.decode_event` 独立重抽 8 次。当模型对某个相位（如 `Tap(3)`）置信度高达 95%，而 3 号键恰好在 `nxt` 时刻处于 `min_same_head_ms`（50ms 冷却）或 `intercept_tap_on_slide` 保护期内时，8 次随机采样会**重复抽中同一个违例的 `Tap(3)` 并连续失败 8 次**！
  - 一旦 8 次失败，[`stream_generate.py:157`](../../src/maichart/composer/stream_generate.py#L157) 立即执行 `kind = WAIT; nxt = cursor + 2; forced_waits += 1`，**在原本应该密集出音的乐句中间硬生生挖出一个 2 拍的大空洞**！
- **修复方式（无放回相位遍历，零额外前向开销）**：
  - 在 `decode_event` / `sample_guarded_event` 中，计算一次 `shape_logits` 和 `phase_logits`；若首选 `(shape, phase, details)` 被 `guard.accepts` 拒绝，**直接在当前 `shape` 下按 `phase_logits` 概率从高到低无放回遍历其余 7 个 `phase`**（仅当 8 个相位全部冲突时才尝试下一个 `shape` 或次优 `delta`），绝不因单键冷却而盲目跳过 2 拍！

---

### 3. 全曲共享音频与历史时钟能否维持常规走位与音乐对应？哪些分支仍浪费？

#### (1) 为什么当前选时头（`delta_head`）无法精准踩住 8 分/16 分/三连音瞬态？（最高优必改项）
请看 [`stream_model.py:248-259`](../../src/maichart/composer/stream_model.py#L248-L259) 的实际数据流：
```python
offsets = torch.linspace(0, 8, 9, device=x.device)
nearby = self._read_audio(cache, beat[..., None]+offsets, inverse, clock='beats')
x = x + self.near_audio(nearby.flatten(-2))
...
hidden = self.decoder(x, cache['global'][inverse], ...)
kind_logits, delta_logits = self.kind_head(hidden).float(), self.delta_head(hidden).float()
```
- `offsets = torch.linspace(0, 8, 9)` **只读取了 `beat + 0, 1, 2, 3, 4, 5, 6, 7, 8` 共 9 个整拍位置**！
- 而 `cache['local']` 的局部卷积 `Conv1d(d, d, 5, padding=2)` 在 `10ms` 网格上视野只有 `50ms`（在 150 BPM 下仅 `0.125` 拍）。
- 这意味着：**当模型站在第 `0` 拍，决定下一个音符是落在 `+1/4` 拍（`100ms`）、`+1/3` 拍（`133ms`）、`+1/2` 拍（`200ms`）还是 `+3/4` 拍（`300ms`）时，`x` 和 `hidden` 根本没有读取 `0` 到 `1` 拍之间任何一帧的 `cache['local']`！**
- 选时头 `delta_head(hidden)` 是在**完全看不见各候选拍位局部声学特征**的情况下盲猜 141 个 `delta` 类！
- **具体修改方案（Candidate-Audio Cross-Scoring，仅增加 ~15 行代码）**：
  1. 将 `near_audio` 改为读取未来 2 拍内的 8 个四分之一拍网格点 + 未来 2~8 拍的整拍点（如 `offsets = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0, 8.0]`，共 13 个点）；
  2. **最关键一步**：在计算 `delta_logits` 时，直接用 `self._read_audio(cache, beat[..., None] + self.delta_values, inverse, clock='beats')` 取出所有候选拍位 `beat + delta_values` 处的局部音频表示 `cand_music`（在 `loss_mask` 有效步或全步上查询，`B=8, T=80, D=141` 仅 $9\times 10^4$ 次一维索引，GPU 耗时 `< 0.3ms`），通过双线性/内积打分：
     ```python
     # hidden: [B, T, d], cand_music: [B, T, num_deltas, d]
     delta_logits = self.delta_head(hidden) + (
         self.delta_query(hidden)[:, :, None, :] * self.delta_audio_key(cand_music)
     ).sum(-1) / math.sqrt(self.config.d_model)
     ```
     这样，每一个候选间隔（无论是 `1/4`、`1/6`、`1/8` 还是 `3/16`）都能**直接对准该精确拍位上的 10ms 声学特征与 5ms 瞬态峰值**打分！

#### (2) `_clock` 的无理数周期 Bug 与 `decoder` 序列相对位置缺失
- **源码定位**：[`stream_model.py:123-126, 237-256`](../../src/maichart/composer/stream_model.py#L123-L126)。
- **问题**：
  1. `_clock` 中 `beats.sin(), beats.cos()` 漏乘了 $2\pi$，导致正余弦周期是 $2\pi \approx 6.283$ 拍，与音乐节拍完全脱节！
  2. `x` 中没有序列步位置编码（`step_index`），且当歌曲进行到第 200 拍时，`log1p(200.0) = 5.3033` 与 `log1p(200.5) = 5.3058` 仅相差 `0.0025`，`self.decoder` 的自注意力很难区分哪个 token 是刚发生的 $t-1$ 步、哪个是 $t-4$ 步。
- **具体修改方案**：
  - 将 `_clock` 的节拍相位改为真正的多尺度音乐周期特征：对 $p \in (1/4, 1/2, 1, 2, 4)$ 拍计算 $\sin(2\pi \cdot \text{beats} / p)$ 与 $\cos(2\pi \cdot \text{beats} / p)$；
  - 在 `x` 进入 `self.decoder` 前，加入相对于序列起点的步位置嵌入（或因果相对位置偏置 / RoPE），并在 `movement` 旁显式加入 `prev_delta_embedding`（上一步与上上一步的间隔相等指示器 `history_delta[:, t] == history_delta[:, t-1]`），让模型极易保持稳定的 8 分/16 分律动。

#### (3) `plan_head` 在 10ms 格上的标签失真与前向断路
- **源码定位**：[`stream_data.py:224-240`](../../src/maichart/composer/stream_data.py#L224-L240) 与 [`stream_model.py:281-285, 379-387`](../../src/maichart/composer/stream_model.py#L379-L387)。
- **问题**：
  1. 在 `10ms` 格上统计 `events == 0`，导致哪怕在每秒 10 颗音的高潮段，也有 90% 的格 `blank == 1`；且 `rates = F.softplus(logits[..., 1:])` 没有乘上格宽 `audio_plan_exposure`，导致当格宽从 `50ms` 变为 `10ms` 时，初始 `softplus(0) = 0.693` 相当于预测每秒 **69.3 个事件**！
  2. `plan_logits` 在模型内部没有连向 `kind_head`（`WAIT` vs `NOTE`）或 `delta_head`，训练出的空白/密度预测在默认生成时完全闲置。
- **具体修改方案**：
  - 在 `_audio_plan`（[`stream_data.py:224`](../../src/maichart/composer/stream_data.py#L224)）中，将 `blank` 标签定义为**以当前格为中心的 `0.5s`（或 2 拍）窗口内无任何起音**（真正的局部休止标签！），并将 `rates = F.softplus(logits[..., 1:]) * batch['audio_plan_exposure'][..., None]` 归一化为**每秒事件率/头率**；
  - 在 `StreamComposer.forward` 中，读取当前拍到未来 2 拍内的平均 `blank` 预测概率（`detach()` 防止主损失冲掉规划器），作为 `kind_logits[..., WAIT]` 的直接先验增益！

---

### 4. 哪些建议现在即可低风险完成？哪些应先做小数据实验？

#### A. 现在即可低风险直接落地（零重训架构风险、纯修补断路与提速）

| 优先级 | 修改项 | 涉及文件 | 预期效果与代价 |
| :--- | :--- | :--- | :--- |
| **P0-1** | **修复 `_clock` 的 $2\pi$ 节拍周期 + 补齐序列步位置编码** | `stream_model.py` (`_clock`, `__init__`, `forward`) | **零风险**：消除 $6.283$ 拍无理数周期错位，让解码器立即分辨 $t-1, t-2$ 步与 $1/4, 1/2, 1, 4$ 拍相位 |
| **P0-2** | **为 `delta_head` 接通各候选 `beat + delta` 的局部音频打分** | `stream_model.py` (`forward`) | **极高收益**：让 `10ms` 音频银行和 `5ms` 瞬态峰真正参与 `1/4, 1/3, 1/2` 拍细分选时；GPU 开销增加 `< 1%` |
| **P0-3** | **修复对称配置跨 `8→1` 边界的 `history_phase` / `reference_phase` 最小旋转对齐** | `event_set_codec.py` (`encode`) & `stream_data.py` | **零风险，不改词表**：让 `3/7 → 4/8 → 1/5 → 2/6` 全程保持 `movement = +1`，根除 `8→1` 边界的 `-3` 伪跳变 |
| **P0-4** | **放开 `_rotation_spec` 对 `v, p, q, s, z, w` 滑条的旋转支持 + 回退事件保留首个按键相位** | `event_set_codec.py:54` & `stream_model.py:220` | **1 行改动**：`v, p, q, s, z, w` 在 Simai 中严格满足旋转等变，且有第 72 行原生解析器兜底，预计可将旋转覆盖率从 `85.95%` 显著提升至 `93%+`，大幅减少 `literal` GRU 回退 |
| **P0-5** | **`sample_guarded_event` 无放回遍历 8 个相位 + 逐键 `0.15s` 快速负荷偏置** | `stream_sampling.py`, `stream_model.py` | 消除因单键冷却导致的连续 8 次盲抽失败与强制 2 拍跳空（`forced_waits`），并让相位头直接感知 16 分同键连打负荷 |
| **P0-6** | **扩大 `StreamDataset._banks` / `_states` 缓存并按曲预计算静态 `action_state`** | `stream_data.py:381, 410, 426` | **大幅降低 CPU 数据准备耗时**：按 `audio_cache_gb` 管理缓存而非硬编码 `4` 首歌，避免每轮/每次验证/每次恢复重跑 `5ms` 波形扫描与 `SustainStateIndex` |
| **P0-7** | **修复 `_audio_plan` 的 `10ms` 曝光尺度与 `blank` 窗口定义，并补齐 `event_set_api.py` 漏传参数** | `stream_data.py`, `stream_model.py`, `event_set_api.py` | 让 `plan_head` 预测真正的每秒密度与半秒休止，并接入 `WAIT` 先验；补齐 WebUI 的 `plan_weight` / `audio_grid_seconds` 透传 |

#### B. 应放在小数据（50~80 谱）实验对照明确收益后再定型的项目
1. **`condition` 随机掩码（`condition_mask_prob = 0.25`）与两阶段预训练（`symbolic_pretrain_epochs`）**：在 60 谱小数据集上对比 `0 轮预训练` vs `3 轮带音频门控预热的符号预训练`，确认是否加速常规走位收敛而不损失起音对齐。
2. **`detail_heads` 槽间自回归耦合 vs 独立分类**：在小数据实验中验证对双押 Break（`1b/5b`）和双滑条同步时值的一致性提升。
3. **历史短句记忆显式调用（方向 E）**：待上述 P0 修复在小数据上验证单段 8~16 步已能稳定写出干净走位后，再决定是否需要在 64 步因果自注意力之外单独加短句归档记忆。

---

## 二、针对少数据流式的“更简洁、更强”改进（不加惩罚补丁）

当前 `stream_model.py` 在预测下一拍动作时，预测的是**相对于上一拍位置的增量 `Δphase = (phase - reference_phase) % 8`**（一阶位置差分，即“速度” $v_t$）。  
但在人类写谱中：
- **顺时针流** `1 → 2 → 3 → 4` 是 $v_{t-2}=+1, v_{t-1}=+1 \implies v_t = +1$（**二阶差分为 0：$v_t = v_{t-1}$**）；
- **逆时针流** `8 → 7 → 6 → 5` 是 $v_{t-2}=-1, v_{t-1}=-1 \implies v_t = -1$（**二阶差分为 0：$v_t = v_{t-1}$**）；
- **单双押平移** `1/5 → 2/6 → 3/7` 同样是 **$v_t = v_{t-1}$**；
- **左右交互（Trill）** `1 → 2 → 1 → 2`（或 `3 → 6 → 3 → 6`）是 $v_{t-1} = -v_{t-2} \implies$ **往返振荡（$v_t = -v_{t-1}$）**！

**为什么一阶 `phase_head(context)` 在少数据下仍然不够省力？**  
因为对于一阶头而言，“顺时针继续走（`+1`）”和“逆时针继续走（`-1`）”是输出层的两个不同类别，它必须从 `hidden` 中先解码出上一步是 `+1` 还是 `-1`，再去激活对应的输出维度。

### 极简二阶动量头（Velocity-Equivariant Phase Head，仅需 6 行代码，零惩罚）
在 `event_phase_logits`（[`stream_model.py:193-197`](../../src/maichart/composer/stream_model.py#L193-L197)）中，除了传入上一拍的方位 `reference_phase`（$\theta_{t-1}$），再传入上一拍的实际旋转步进 `reference_velocity`（$v_{t-1} = (\theta_{t-1} - \theta_{t-2}) \bmod 8$）：
- 模型的 3 维动量门 `momentum_head(context)` 直接输出三个标量 logit：
  1. `keep_velocity`：直接给 $\theta_t = (\theta_{t-1} + v_{t-1}) \bmod 8$ 加分（**无论当前是顺时针 `+1`、逆时针 `-1` 还是跳格 `+2`，统统共享同一个参数！**）；
  2. `reverse_velocity`：直接给 $\theta_t = (\theta_{t-1} - v_{t-1}) \bmod 8$ 加分（**所有方向的交互/折返 `A → B → A → B` 统统共享同一个参数！**）；
  3. `stay_same`：给 $\theta_t = \theta_{t-1}$ 加分（纵连，天然受负荷特征抑制）。
- 再与原有的 `relative`（8 维）和 `absolute_phase_bias`（8 维）相加：
  $$\text{logit}(\theta_t) = s_{\text{rel}}(\theta_t - \theta_{t-1}) + s_{\text{mom}}(\text{ continue } v_{t-1} \text{ / reverse } -v_{t-1}) + b_{\text{abs}}(\theta_t) + s_{\text{load}}(\theta_t)$$

**为什么这比任何防复读惩罚或短句 DSL 都强大？**
- 它不禁止任何走位，也不需要手写短句编译器；
- 但它把全语料中所有顺时针流、所有逆时针流、所有跨度平移整合到了**同一个标量 logit `keep_velocity`** 上，把所有按键上的交互（`1-8-1-8`、`2-7-2-7`、`3-6-3-6`）整合到了**同一个标量 logit `reverse_velocity`** 上！哪怕只有 30 张训练谱，模型也能在几百步内立刻学出极其干净、连贯的顺逆流与交互配置。

---

## 三、P0 改动逐文件具体落地方案（供代码端直接核对实施）

### 1. `src/maichart/composer/event_set_codec.py`
1. **第 54 行放开等变滑条模式**：
   ```python
   # '-', 'v', 'p', 'q', 's', 'z', 'w' 均为单终点且在 C8 旋转下符号严格不变
   if note.note_type == 'slide' and (match['pattern'] not in {'-', 'v', 'p', 'q', 's', 'z', 'w'} or len(match['end_position']) != 1):
       return None
   ```
2. **`encode` 支持参考相位就近对齐（消除对称多押跨 `8→1` 伪跳变）**：
   - 在 `encode(self, note_text, *, reference_phase=None)` 中，计算 `mask = self.phase_equivalents(sid, phase, detail_ids)` 后，若 `reference_phase is not None` 且 `any(mask)`：
     在所有 `p for p, ok in enumerate(mask) if ok` 中，选择使最小圆周距离 `min((p - reference_phase) % 8, (reference_phase - p) % 8)`（距离相同时优先正向 `(p - reference_phase) % 8`）最小的 `p` 作为返回的 `phase`；
   - 若 `fallback=True`（`sid == 0`），从 `atoms` 解析首个含 1~8 按键的 head 作为 `reference_phase` 追踪用的物理方位（不再一律填 `0`）。

### 2. `src/maichart/composer/stream_model.py`
1. **修复 `_clock` 的节拍正余弦周期（第 123–126 行）**：
   ```python
   @staticmethod
   def _clock(seconds, beats):
       b = beats.float()
       freqs = b.new_tensor([2. * math.pi / p for p in (0.25, 0.5, 1.0, 2.0, 4.0)])
       ang = b[..., None] * freqs
       return torch.cat((
           torch.stack((torch.sign(seconds)*torch.log1p(seconds.abs()),
                        torch.sign(beats)*torch.log1p(beats.abs())), -1).float(),
           ang.sin(), ang.cos()
       ), -1)  # 共 2 + 10 = 12 维（相应将 audio_clock 改为 Linear(12, d)，cursor_clock 改为 Linear(16, d)）
   ```
2. **选时头 `delta_head` 接入候选拍位局部音频打分（第 248–274 行）**：
   - 新增 `self.delta_query = nn.Linear(d, 64)` 与 `self.delta_audio_key = nn.Linear(d, 64, bias=False)`（零初始化 `delta_query` 以保证初始等价）；
   - 对每个时间步 `beat`，查询 `cand_beats = (beat[..., None] + self.delta_values[None, None, :]).clamp_max(end_beat[:, None, None])` 处的局部音频 `cand_audio = self._read_audio(cache, cand_beats, inverse, clock='beats')`，将内积分数加到 `delta_logits` 上。
3. **加入二阶旋转动量（`keep_velocity` / `reverse_velocity`）与落点时刻逐键快速负荷偏置（第 193–197 行）**：
   - 在 `event_phase_logits` 中，传入 `reference_phase`、上一步旋转量 `reference_velocity = (reference_phase - old_phase) % 8` 以及在 `target_second` 处按 `(0.15, 0.5, 2.0)` 秒衰减计算的 8 键负荷，让相位预测同时具备二阶走位惯性与高速同键防复读能力。
4. **`detail_heads` 注入 `shape + phase` 条件与前序槽位嵌入（第 110, 311, 368 行）**：
   - `ctx_detail = ctx + self.shape_embedding(shape) + phase_emb`，消除细节与绝对方位的脱节。

### 3. `src/maichart/composer/stream_data.py` & `stream_sampling.py`
1. **修复 `_audio_plan` 的 `blank` 窗口与曝光归一化（`stream_data.py:224-240` & `stream_model.py:380-387`）**：
   - `blank` 改为判断 `[t - 0.25s, t + 0.25s]`（0.5 秒窗）内是否无事件；Poisson 率预测乘上格宽 `audio_plan_exposure`；并在 `StreamComposer.forward` 中将未来 2 拍的平均 `blank` 预测温和注入 `kind_logits[..., WAIT]`。
2. **扩大 `_banks` / `_states` 缓存并缓存静态序列状态（`stream_data.py:381, 410, 426`）**：
   - 将 `_banks` 缓存上限按 `audio_cache_gb` 设为可容纳至少 64~128 首歌（或按字节预算淘汰），`_states` 缓存全部训练曲目的静态 `SustainStateIndex`，彻底消除训练/验证循环中的重复 CPU 构建。
3. **`sample_guarded_event` 无放回相位回退（`stream_sampling.py:38-50`）**：
   - 当某 `(shape, phase, details)` 未通过 `guard.accepts` 时，先按 `phase_logits` 降序尝试其余 7 个相位，避免触发 `forced_waits` 硬跳 2 拍。
