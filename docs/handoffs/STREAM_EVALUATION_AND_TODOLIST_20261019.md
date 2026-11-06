# `chart_stream_v1` 深度实证评估与架构演进 TodoList

> **生成时间**：2026-10-19
> **评估基准**：`src/maichart/stream/` 当前代码库 + `checkpoints/Prism_stream_v9/stream_last.pt`（Epoch 46, Step 8050, `v7` 语义）+ `E:\后训练（官谱）\training_manifest.json` 实证抽样测量
> **核心背景**：当前 `chart_stream_v1` 从早期的分段/两阶段（`event` / `form` 系列）演进为全因果单流自回归（`<at> -> P:* -> <note> -> N:* -> <end>`）。早期分段架构天然具备“整段能量视野”与“整段配置协同”，而迁移到单流架构后，部分历史包袱与单流短视性叠加，导致了训练监督偏差与生成层面的“段落能量失明”、“毫无道理的摆键”等问题。

---

## 一、 核心问题实证评估与重要性分级

根据指示，我们对前 4 项训练/监督隐患进行了基于 `Prism_stream_v9` 权重与真实训练批次的 **CPU 实证量化测量**，并对第 5–7 项架构与生成问题结合“单流 vs 分段历史包袱”进行了深入设计评估；第 8、9 项已按舞萌 DX 谱面设计惯例归档排除。

### 重要性总览矩阵

| 编号 | 问题域 | 实证严重度 | 是否导致崩溃/灾难 | 真实影响范围与实证结论 | 优先级 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **问题 2** | `alignment_loss` 错位监督与跨层梯度劫持 | **极高 (P0)** | 非数值崩溃，但造成**事件边界隐状态梯度被 8.4 倍无效梯度主导** | 在 `<bos>/<end>` 位置，`0.15 * alignment_loss` 对 `hidden` 的梯度范数是全部主损失之和的 **8.404 倍**，余弦相似度 `-0.0025`（正交冲突），直接干扰同位置的 `plan_head` 与 `<at>/<eos>` 决策 | **P0** |
| **问题 4** | `_head_labels` 漏掉 Touch / 重复 Chord；`masked_note_inputs` 8 事件连坐清零 | **极高 (P0)** | 确定性代码缺陷，直接损失 **100% Touch 意图通道** 与 **~8.7% 有效监督 token** | `intent_projection` 第 3、6 列余弦相似度高达 **0.8980**（完全退化为重复特征），Touch 意图为 0；8 事件整段清零使模型失去“掩码后紧邻事件恢复”的监督信号 | **P0** |
| **问题 1** | `GroundedContext.local` 缺失相对时间偏移 $\Delta t$ | **高 (P1)** | 非灾难性失明（靠节拍相位保底），但导致**局部瞬态严重模糊** | $\pm 2\text{s}$（32 节点）窗口内，仅 **7.4%** 注意力权重落在当前事件 $\pm 50\text{ms}$ 内，**92.6%** 弥散在前后 2 秒（过去 35.9%、未来 56.7%），整拍周期相位发生混叠 | **P1** |
| **问题 3** | `cycle_exit_targets` (`v7`) 形状破环位置计算偏差 | **中高 (P1)** | 不会训崩（仍为真实标签），但 **84.7% 的新增破环加权打偏** | `v7` 新增的形状破环样本中，**84.7% (692/817)** 因 `expected = cells[i-period]` 首键位数字不同而落在了 `N:1..8` 键位数字上，未能加权到真正的结构修饰符（`N:h`/`N:/`/`N:-`） | **P1** |
| **问题 7** | 段落能量失明（`curve_head` 断联 + `TypeGuidance` 全曲静态均值） | **极高 (P0·架构)** | 单流架构核心短板 | 早期分段架构可整段规划密度，单流解码器丧失未来小节能量视野，且 `TypeGuidance` 强制全曲静态比例，抹平副歌/间奏反差 | **P0** |
| **问题 6** | 毫无道理的摆键（缺乏 $D_8$ 圆环几何 + 纯 Tap 段动作状态全零） | **极高 (P0·架构)** | 单流生成核心体验痛点 | 实测官谱中 **62.9%** 的事件查询 `query_actions` 为**全零向量**（无活跃 Hold/Slide 时完全失忆），且缺乏八角圆环相对位移与 $D_8$ 对称性 | **P0** |
| **问题 5** | 生成引导失衡（单候选绕过 + 4 候选同质化重采样） | **高 (P1·推理)** | 推理期配置控制失效 | `candidates == 1` 或显存守卫触发时 `local_chords` 与物理守卫完全失效；4 次同分布采样在高频段极易坍缩为 4 个单押 Tap | **P1** |
| **问题 8** | Touch/TouchHold 未计入双手容量上限 | **排除 (不修改)** | 符合官谱/自制谱习惯 | 舞萌 DX 中 Touch 判定区极大、可擦可蹭，常作为堆物量或调味品，保持现状 | **不处理** |
| **问题 9** | Slide 等待期手部分配与同键连打（Jack） | **排除 (不修改)** | 属于游戏本身设计范畴 | 官谱与自制谱广泛存在启动后换手及特定曲风纵连，无需在底层强行收紧 | **不处理** |

---

## 二、 逐项深度评估与技术方案设计

### 1. 问题 1 评估：`GroundedContext.local` 局部声学窗口缺失相对时间 $\Delta t$

- **代码定位**：[grounded_model.py](../../src/maichart/stream/grounded_model.py#L40-L68)
- **慎重评估：是否会造成灾难性后果？**
  - **结论**：**不会导致训练崩溃或完全丧失节奏感，但会导致局部声学瞬态（鼓点、重音、切分音）发生“跨拍相位混叠”，属于限制定位上限的高危瓶颈。**
  - **原因分析**：`detail_audio`（120 维）内部包含 `sin(2*pi*beat)`、`cos(2*pi*beat)` 与小节相位特征，`clock`（20 维）也包含 `sub` 与 `bar_phase`。因此 `local_query(clock) @ local_key(detail_audio)` 能够匹配“相同拍内相位”的节点。
  - **实证数据**：在 `Prism_stream_v9` 上对真实批次测量 `GroundedContext.local` 的 32 邻域注意力权重：
    - 距离当前时刻 $|\Delta t| \le 50\text{ms}$ 的中心音频节点仅获得 **7.4%** 的注意力权重；
    - 过去窗口（$\Delta t < -50\text{ms}$）分走了 **35.9%**，未来窗口（$\Delta t > +50\text{ms}$）分走了 **56.7%**，最大单节点平均权重仅 `0.1259`。
  - **病理机制**：因为正余弦节拍相位每 1 拍（或 1 小节）严格周期重复，在 $\pm 2\text{s}$（约 $\pm 4\sim 8$ 拍）窗口内，前 2 拍、前 1 拍、当前拍、后 1 拍、后 2 拍的半拍/整拍节点具有**完全相同的节拍相位特征**！由于 `local()` 计算完 `sample_times - seconds[..., None]` 掩码后直接丢弃了相对时间差，模型无法区分“当前这一拍的鼓点”与“前后第 $1, 2, 3$ 拍同相位的鼓点”，退化成了 $\pm 2\text{s}$ 的同相位均值池化。
- **修复方案**：
  - 复用 `GroundedContext.retrieve()` 的成熟设计，计算局部邻域的相对时间与相对节拍特征 $\Delta \tau = (\Delta t / 2.0, |\Delta t| / 2.0, \text{sign}(\Delta t))$，通过零初始化线性层 `local_distance = nn.Linear(3, key, bias=False)` 和 `local_value_distance = nn.Linear(3, width, bias=False)` 分别注入 `keys` 与 `neighbors`（零初始化保证无缝兼容现有 checkpoint 继续微调）。

---

### 2. 问题 2 评估：`GroundedContext.alignment` 监督错位与梯度劫持

- **代码定位**：
  - [data.py](../../src/maichart/stream/data.py#L422-L433)（`alignment_event_indices` 选择 `target == codec.at_id` 的行）
  - [grounded_model.py](../../src/maichart/stream/grounded_model.py#L130-L144)（`alignment` 方法）
  - [objectives.py](../../src/maichart/stream/objectives.py#L32-L45)（`alignment_loss`）
- **慎重评估：是否会造成灾难性后果？**
  - **结论**：**不会弄坏 `N:*` 键位内部的自回归语法，但在事件边界（`<bos>` / `<end>`）上造成了严重的梯度劫持（Gradient Hijacking），直接损害 `plan_head`（事件意图规划）与下一事件起拍预测！**
  - **实证数据（`Prism_stream_v9` 真实批次反向传播测量）**：
    - `alignment_loss` 原始值高达 **`7.3307`**（加权后 `1.0996`，占总 Loss `1.818` 的 **60.5%**），8192 分类准确率仅 `11.39%`。
    - 在 `<bos>` / `<end>` 位置（即 `alignment_event_indices` 所在位置，同时也正是 [model.py](../../src/maichart/stream/model.py#L203-L209) 中 `plan_head(hidden)` 提取事件级 `intent` 并广播给整个事件的边界位置）：
      - 主任务全部损失（`token_loss + plan_loss + decision_loss + event_balance_loss + cycle_exit_loss`）对 `hidden` 的梯度 L2 范数均值为 **`0.000068`**；
      - 仅 `0.15 * alignment_loss` 一项对同一位置 `hidden` 的梯度 L2 范数均值高达 **`0.000569`**；
      - **梯度范数比值：`||grad_align|| / ||grad_main|| = 8.404`（对齐损失梯度是主损失梯度的 8.4 倍！）**；
      - **梯度余弦相似度：`-0.0025`（完全正交甚至微弱反向）**。
  - **病理机制**：
    1. **因果错位**：在 `target[i] == <at>` 的位置，解码器输入是上一事件的 `<end>`，此时 `CausalClock` 还停留在上一事件时刻 $t_{k-1}$，模型还没看到第 $k$ 个事件的 `P:*` 位置 token，却被强迫用 `8.4` 倍的梯度去 8192 个全曲音频节点里盲猜第 $k$ 个事件的时间戳。
    2. **与 `plan_head` 争抢同一边界隐状态**：[model.py](../../src/maichart/stream/model.py#L206-L209) 恰好也在 `decoder_input == <bos> | <end>` 的位置读取 `plan = self.plan_head(hidden)` 并通过 `intent_projection(intent)` 注入后续所有 token。`alignment_loss` 在该位置施加 8.4 倍的正交梯度，直接污染了事件级 `plan` 意图的隐状态表征。
- **修复方案**：
  1. 将 `alignment_event_indices` 从 `target == codec.at_id`（事件开始前）移至 **`target == codec.note_id`（即 `P:*` 位置编码已完整输入、`<note>` 即将生成音符的时刻）**，此时因果时钟已更新至当前事件真实拍号 $b_k$。
  2. 将 `alignment_key` 的输入从原始 `batch['detail_audio']` 改为共享 `bank['projected']`（或对 `hidden` 侧做梯度缩放），使对齐监督真正训练声学表征与位置解码的对齐，而非训练一个推理期完全不用的旁路头。

---

### 3. 问题 3 评估：`cycle_exit_targets` (`v7`) 形状破环目标偏移

- **代码定位**：[continuation.py](../../src/maichart/stream/continuation.py#L30-L69)
- **慎重评估：是否会造成灾难性后果？**
  - **结论**：**不属于灾难性错误（不会注入错误标签），但导致 `v7` 新增的结构破环监督有 84.7% 落在了错误的 token 位置（键位数字而非形态修饰符），使结构破环加权形同虚设。**
  - **实证数据（12 张官谱 6,201 个事件全量统计）**：
    - `v6` 精确 cell 重复破环命中 `575` 次；`v7` 新增的 `shapes`（形态元组）重复破环额外命中 **`817` 次**。
    - 在这 `817` 次新增的形态破环中，由于第 63 行仍使用 `expected = cells[i - period]`（上周期的具体 token 序列，如 `('N:7',)`）与当前 `current`（如 `('N:2', 'N:h', ...)`）逐字符比对，第 0 个字符（键位数字 `'N:7'` vs `'N:2'`）几乎必然不同，导致 `divergence` 恒为 `0`。
    - 统计显示：`v7` 新增的 817 个破环目标中，**692 个（84.7%）落在了首键位数字 `N:1..8` 上**，落到真正改变形态的 `N:h`（33 次）、`N:/`（30 次）、`N:V`（13 次）、`N:b`（6 次）、`N:-`（4 次）等修饰符上的合计仅占 **15.3%**。
- **修复方案**：
  - 当由第二个循环（`shapes` 形态周期破环，而非 `cells` 精确重复破环）触发时，定位当前事件中**第一个非数字的形态修饰符 token**（如 `N:/`, `N:h`, `N:b`, `N:x`, Slide 符号, Touch 区域符号 `N:A..E`）或长度变化处的 `<end>` 作为 `cycle_exit` 监督目标，精准强化模型在“连打转长条/滑键/多押”时的结构切换能力。

---

### 4. 问题 4 评估：`_head_labels` 维度重复与 `masked_note_inputs` 8 事件连坐清零

- **代码定位**：
  - [data.py](../../src/maichart/stream/data.py#L82-L97)（`_head_labels`）
  - [grounding.py](../../src/maichart/stream/grounding.py#L183-L200)（`masked_note_inputs`）
- **影响范围量化评估**：
  1. **`_head_labels` 重复 `chord`、漏掉 `touch` 的真实影响范围**：
     - 在 [model.py](../../src/maichart/stream/model.py#L203-L209) 和 [model.py](../../src/maichart/stream/model.py#L243-L249) 中，`plan_head(hidden)` 输出的 8 维向量不仅用于计算 `plan_loss`，还通过 **`hidden = hidden + self.grounding.intent_projection(intent)` 直接加回解码器隐状态，指导当前事件所有 `P:*` 和 `N:*` 的生成**！
     - **实证测量**：在 `Prism_stream_v9` 中，检查 `model.grounding.intent_projection.weight`（形状 `[384, 8]`）各列范数与余弦相似度：
       - 8 列 L2 范数分别为：`[1.0875, 0.8031, 0.8389, 0.7543 (chord1), 0.8493 (hold), 1.0617 (slide), 0.7405 (chord2_dup), 0.3061 (break)]`；
       - 第 3 列（`chord1`）与第 6 列（`chord2_dup`）的**余弦相似度高达 `0.8980`**！
     - 这证明模型在 `intent_projection` 中浪费了整整一个通道学习重复的 `chord` 投影，同时导致 **Touch 音符（抽样官谱中占 4.6% 事件，在 DX 谱面与高密度段落中占比达 10%~25%）在事件级意图规划与 `intent_projection` 中彻底失明**。
  2. **`masked_note_inputs` 整段 8 事件 `quality = 0` 的真实影响范围**：
     - 当 `prefix_note_dropout = 0.10` 且 `span_events = 8` 时，每次采样命中会把连续 8 个事件的**全部 token**（包含未掩码的 `<at>`, `P:*`, `<note>` 以及 `<end>`）的 `target_quality` 设为 `0`。
     - **量化损耗**：平均抹除了全数据集约 **8.5% ~ 9.5%** 的有效监督 token（含大量完好无损的 `P:*` 节奏位置监督）。
     - **更隐蔽的训练机理损伤**：`masked_note_inputs` 的设计初衷是“当历史/前缀中出现 `<mask>` 时，让模型学会利用声学与上下文从掩码中恢复，避免滚雪球复读”。但代码将连续 8 个事件的 `quality` 全部清零，意味着第 1 个事件被 mask 后，紧随其后的第 2~8 个事件虽然在自注意力前缀里看到了 `<mask>`，但它们的梯度被 `quality = 0` 完全屏蔽了！模型**永远学不会“在被 mask 后的紧邻下一拍立刻恢复正常写谱”**，只能学到 8 个事件之后的远距离恢复。
- **修复方案**：
  1. `_head_labels` 第 4 项修正为 `float(counts[3] > 0)`（Touch 标志位），恢复 8 维计划向量 `[interval, density, density_delta, chord, hold, slide, touch, break]` 的完整语义。
  2. 在 `masked_note_inputs` 中：**仅将被替换为 `<mask>` 的 `N:*` 音符位（以及该被掩码事件的 `<end>`）的 `target_quality` 置 0**，保留 `P:*` 位置监督；或者将掩码跨度缩小为 `1~2` 个事件并保留紧邻后续事件的 `target_quality > 0`，使模型真正获得“紧邻事件从 `<mask>` 恢复”的梯度信号。

---

### 5. 问题 5 平衡方案设计：生成期引导（Guidance）与候选搜索的平衡重构

- **历史背景与现状矛盾**：
  - 在 `grounded_inference.py` 与 `configuration_search.py` 中，为了防止引导权重过大破坏模型原生分布，当前逻辑将配置搜索绑定在了 `configuration_candidates > 1 and style_strength > 0` 上，且仅从模型分布中独立温度采样 `N=4` 条候选。
  - 这导致两个极端：要么**引导完全被绕过**（当 `candidates == 1`、长序列触发显存回退、或只开 `local_chords` 未开 `style_strength` 时）；要么**在模型概率高度集中于单押 Tap 时，4 次独立采样全是单押 Tap**，事后重排序选不出多押、长条或滑键。
- **“平衡态”解决方案（先验轻推 + 多样化候选提案 + 始终生效的硬约束守卫）**：
  1. **解耦“安全/局部和弦约束”与 `style_strength`**：
     - 将条件评估拆分为两层：**物理与局部条件层**（`physical_guard`、`local_chords`）与**全局风格分布层**（`TypeGuidance`）。即使 `candidates == 1` 或 `style_strength == 0`，单步解码时的增量物理守卫（防止手部冲突）和首修饰符的轻微先验校准也必须始终工作。
  2. **“先验轻推（Logit Nudge）+ 后验重排（Rerank）”双阶段平衡机制**：
     - 为什么 4 个候选抽不到 Hold/Slide/Chord？因为在生成第一个修饰符（如 `N:1` 之后的 `N:/`、`N:h`、`N:-` 或 `<end>`）时，模型原生 logit 可能给 `<end>`（单押 Tap）0.92 的概率，给 `N:/` 0.04 的概率。
     - **平衡策略**：在 `configuration_search` 采样 4 个候选时，不在所有 token 上强行加权（避免生成畸形语法），而是采用 **“1 个贪心/低温原生态候选 + 3 个带轻量级首修饰符先验偏置（Logit Nudge, 强度 $\alpha \in [0.15, 0.35]$）的多样化提案候选”**。
     - 这样既保证了第 1 个候选 100% 忠于模型原生最高概率分布（当模型非常确定时不会被强行扭曲），又保证了其余 3 个候选能够探索出“双押”、“Hold”、“Slide”等合法分支供评分器在模型 logprob 与风格/运动成本之间做公平竞争。

---

### 6. 问题 6 全新方法评估：在单流框架下解决“毫无道理的摆键”与八角几何缺失

- **为什么早期分段模型不明显、而当前单流模型“毫无道理地乱摆键”？**
  1. **纯 Tap 段落的“动作状态彻底失忆”**：实测显示，即使在 Master 官谱中，也有 **62.9% 的事件查询 `query_actions` 是全零向量**！因为当前 `action_summaries` 只记录**尚未结束的 Hold 和 Slide**。一旦进入纯 Tap 交互、散打、回旋楼梯段落，`query_actions` 全是 `0`，接地路由器（Router）里的 `active_actions` 分支彻底失效，模型只能靠稀疏的 token 历史去猜上一个键打在哪。
  2. **缺乏八角圆环空间几何（$D_8$ 群）与左右手运动惯性**：在 token 词表中，`N:1` 到 `N:8` 只是 8 个离散 ID，模型并不知道 `1` 和 `8` 相邻、`1` 和 `5` 对称，也不知道玩家是用左手（通常负责 `5, 6, 7, 8` 侧）还是右手（通常负责 `1, 2, 3, 4` 侧）交替击打。
  3. **单流逐事件短视**：早期分段模型一次看到一整段的位置与配置，而单流每次只采样一个事件，没有考虑“当前拍落键位置与上一拍、下一拍是否构成顺手的弧线/交互/对称”。
- **全新解决方案设计（三位于一体，完全契合 `chart_stream_v1` 架构）**：
  1. **方法 A：构建“全时段手部运动学状态机”（`KinematicHandState`，终结 62.9% 的全零状态）**
     - 在 `grounding.py` 与 `grounded_inference.py` 中，不再让 `query_actions`（50 维）在无长条时全零，而是利用其中的空闲维度（或扩展动作摘要）实时编码**因果手部运动学轨迹（Causal Hand Kinematics）**：
       - **8 键位指数衰减余温场（8 维）**：记录最近 $K=4$ 个事件在 `1..8` 号键上的击打余温 $\sum_{m=1}^4 \gamma^{m} \mathbf{1}_{\text{lane}}$ 及时间衰减 $\exp(-\Delta t / 0.25\text{s})$，让模型在纯 Tap 段也清晰感知“刚刚哪几个键刚被按过”。
       - **圆环相对位移与旋转动量（6 维）**：计算上一拍到上上拍的圆环有向角步长 $\Delta \theta \in [-4, +4]$（编码为 $\cos(\frac{2\pi \Delta\theta}{8}), \sin(\frac{2\pi \Delta\theta}{8})$）、顺/逆时针回旋惯性（流向一致性）、以及是否发生同键纵连（Jack 计数）。
       - **左右半区平衡与手部交替相位（4 维）**：统计最近事件在右半区（`1..4`）与左半区（`5..8`）的分布重心与左右交替（Alternation）奇偶相位。
     - **效果**：无论是否有长条，100% 的事件查询都携带鲜活的“刚打过哪个键、正在顺时针还是逆时针流转、该换左手还是右手”的运动学状态！
  2. **方法 B：$D_8$ 二面体群几何等变增广（8 旋转 $\times$ 2 镜像）与相对键位编码**
     - 舞萌 DX 的 8 个外圈键与 `A/B/D/E` 传感器严格服从正八边形二面体群 $D_8$（16 种对称变换）。
     - 在 `StreamDataset` 训练采样时引入 **在线 $D_8$ 对称变换增广**（含 Simai 滑键轨迹符号精确映射：镜像下 `< <-> >`、`p <-> q`、`s <-> z`、`pp <-> qq`，旋转下全部键位 $\equiv (x - 1 + r) \pmod 8 + 1$）。一份官谱走位（如 `1-2-3-4` 顺时针楼梯）自动教会模型全部 8 个起点、顺逆时针 16 种对称走位，彻底消除高频键位偏置与罕见角度乱跳。
  3. **方法 C：候选搜索中的“运动学流畅度评分”（`KinematicFlowCost`）**
     - 在 `configuration_search.py` 对候选音符配置打分时，加入轻量级运动学流畅度项：根据当前节拍间隔 $\Delta b$，奖励合理的左右手交替（如 16 分音符交互/小幅顺滑位移 $|\Delta \theta| \in \{0, 1, 2\}$），对无音乐重音支撑的极端反手大跨度乱跳施加温和软惩罚。

---

### 7. 问题 7 架构级方案设计：打通 Planner 与 Decoder 的“段落能量视野”

- **历史背景与现状病因**：
  - 早期 `event` / `form` 系列是“先看整段规划位置，再整段填音符”，因此天然具备段落级能量感。
  - 在当前 `chart_stream_v1` 中：
    1. 编码器虽然有 `curve_head(encoded)` 预测全曲 9 维负载与音符类型曲线（`[density, tap, hold, slide, touch, break, chord, sustain, complexity]`），但它在 [model.py](../../src/maichart/stream/model.py#L179-L210) 中**只算了 `curve_loss` 就被彻底丢弃**，完全没有输入给解码器！
    2. `plan_head` 只预测当前单拍的瞬时值，缺乏“未来 1~4 小节”的段落级视野。
    3. 推理期 `TypeGuidance` 把全曲条件向量当作每一拍的恒定目标比例，把 Intro/间奏与副歌强行拉平。
- **Planner-to-Decoder 架构升级方案（在单流框架内恢复“整段考虑配置”的能力）**：
  1. **在 `GroundedContext` 中引入多尺度段落前瞻窗口（`PhraseHorizonConditioning`）**：
     - 在 [model.py](../../src/maichart/stream/model.py) 与 [grounded_model.py](../../src/maichart/stream/grounded_model.py) 中，将编码器输出的 `curve.sigmoid()`（9 维动态负载曲线）传入 `GroundedContext`。
     - 对于每个解码查询时刻 $t_q$，不仅提取当前时刻 $t_q$ 的曲线值，还沿时间轴池化三个尺度的段落能量包络：
       - **瞬时窗口** $[t_q - 0.25\text{s}, t_q + 0.25\text{s}]$（当前拍能量与音符类型倾向）
       - **乐句前瞻窗口（1 小节 / +2.0s）** $[t_q, t_q + 2.0\text{s}]$（即将到来的乐句密度均值与峰值）
       - **段落前瞻窗口（4 小节 / +8.0s）** $[t_q, t_q + 8.0\text{s}]$（当前处于低潮间奏、爬升 Build-up 还是高潮副歌）
     - 将这 $3 \times 9 = 27$ 维**多尺度段落能量包络**通过零初始化线性层 `phrase_plan_projection = nn.Linear(27, width, bias=False)` 注入 `GroundedContext.input()` 与 `GroundedContext.fuse()`。
     - **训练策略**：训练时以 50% 概率使用真实 `curve_target`（带微量高斯噪声）、50% 概率使用 `curve.detach().sigmoid()`，使解码器在训练和推理时都能严格服从局部段落能量曲线！
  2. **推理期 `TypeGuidance` 升级为“局部窗口动态配额”（`WindowedDynamicTypeGuidance`）**：
     - 在 [type_guidance.py](../../src/maichart/form/type_guidance.py) 与 [grounded_inference.py](../../src/maichart/stream/grounded_inference.py) 中，不再使用全曲静态比例对比全曲累计计数，而是以当前时刻 $t_q$ 周围局部窗口（如最近 4 小节）内 `curve_head` 预测的局部类型占比作为动态目标：
       - 在间奏/慢速段，`curve_head` 预测低密度、高 Hold/Slide 占比，`TypeGuidance` 自动引导生成长条与舒缓配置；
       - 在副歌高潮段，`curve_head` 预测高密度、高 Tap/Chord/Break 占比，`TypeGuidance` 自动放开高频交互与多押！

---

## 三、 分阶段实施 TodoList

> **执行原则**：先修补无需改动模型参数形状的训练监督与数据层缺陷（Phase 1），再升级 Planner 段落能量闭环与手部运动学接地架构（Phase 2，通过零初始化投影层保持与现有 `Prism_stream_v9` 权重的平滑微调兼容），最后升级推理期平衡引导与走位搜索（Phase 3）。

### Phase 1：核心训练监督与数据层修复（P0/P1，零参数形状变更，立即可修）
- [x] **Todo 1.1 [P0 - 问题 4a]**：修复 [data.py](../../src/maichart/stream/data.py#L96) 中 `_head_labels` 的维度错误，将第 4 个返回值从重复的 `chord` 修正为 `float(counts[3] + counts[4] > 0)`（Touch/TouchHold 标志位），并更新单元测试校验 8 维 `plan_labels` 语义 `[interval, density, density_delta, chord, slide, hold, touch, break]`。
- [x] **Todo 1.2 [P0 - 问题 4b]**：修复 [grounding.py](../../src/maichart/stream/grounding.py#L183-L200) 中 `masked_note_inputs` 的连坐清零逻辑：仅将自身 `decoder_input == <mask>` 的目标位置的 `quality` 置为 `0`，保留 `<at>`、`P:*` 位置 token、首音符 token 及后续未掩码事件的 `target_quality`（掩码损失比例从 `7.3%` 降至 `3.69%`），并同步遮蔽被掩码事件的手部运动学特征。
- [x] **Todo 1.3 [P0 - 问题 2]**：修复 [grounded_model.py](../../src/maichart/stream/grounded_model.py) 与 [objectives.py](../../src/maichart/stream/objectives.py) 中的 `alignment_loss`：将原本全曲 8192 分类（无位置编码）重构为围绕当前查询时刻的 128 节点因果局部指针（带 `alignment_distance` 相对时间编码），覆盖率 `99.1%`，初始 `alignment_loss` 从 `6.17` 降至 `3.72`。
- [x] **Todo 1.4 [P1 - 问题 3]**：修复 [continuation.py](../../src/maichart/stream/continuation.py) 中 `cycle_exit_targets` 的两处缺陷：（1）排除 `<end>` 与 `[...]` 时值括号对 `shapes` 形态集合的污染（修复 `<end>` 中 `'>'` 导致全员误判为 Slide 的 Bug）；（2）新增 `_structure_classes` 将键位数字 `1..8` 归并为同一 `'lane'` 类，使形状周期破环精确指向首个结构修饰符 token（`N:/`, `N:h`, `N:-`, `N:b`, `<end>`）。

### Phase 2：单流框架下的“段落能量视野”与“手部运动学几何”架构升级（P0/P1，支持旧权重无缝升级）
- [x] **Todo 2.1 [P1 - 问题 1]**：在 [grounded_model.py](../../src/maichart/stream/grounded_model.py) 的 `GroundedContext.local()` 中注入 9 维相对时间与节拍相位特征 `relative_features`（通过零初始化线性层 `local_distance_key` / `local_distance_value` 注入 `keys` 与 `neighbors`），消除 $\pm 2\text{s}$ 窗口内的跨拍节拍相位混叠。
- [x] **Todo 2.2 [P0 - 问题 7a]**：打通 Planner (`curve_head`) 到 Decoder (`GroundedContext`) 的多尺度段落能量视野：
  - 在 `GroundedContext.phrase()` 中基于查询时间戳 $t_q$ 提取 `(-1, 1)s`、`+(0, 4)s`、`+(0, 12)s`、`(-12, 0)s` 四尺度 `curve` 包络特征（`PHRASE_DIM = 16`），通过零初始化投影层 `phrase_projection` 融合至 `local` 上下文（训练与 [grounded_inference.py](../../src/maichart/stream/grounded_inference.py) 保持严格一致）；
  - 在 [grounded_model.py](../../src/maichart/stream/grounded_model.py) 的 `UPGRADE_PARAMETERS` 与 [checkpoint.py](../../src/maichart/stream/checkpoint.py) 中注册新增分支键名，实测 `Prism_stream_v9` 可直接无损加载（初始输出保持 bit-identical，且首次反向传播时 5 个新增投影分支均获得有效梯度）。
- [x] **Todo 2.3 [P0 - 问题 6a]**：在 [symmetry.py](../../src/maichart/stream/symmetry.py) 中实现 `kinematic_rows`（`KINEMATIC_DIM = 24`：8 键位指数余温衰减、上拍键位分布、对数拍距、圆环有向角位移 $\sin\Delta\theta/\cos\Delta\theta$、同键连打计数、多押规模与纯 Touch 标志），通过零初始化 `kinematic_projection` 注入 `queries['action']`，将非零动作查询占比从 `34.5%` 提升至 `99.8%`。
- [x] **Todo 2.4 [P0 - 问题 6b]**：在 [symmetry.py](../../src/maichart/stream/symmetry.py) 与 [data.py](../../src/maichart/stream/data.py) 中实现官方街机四向对称增广 `TRANSFORMS = ('identity', 'mirror_lr', 'mirror_ud', 'rotate_180')`（排除破坏人体左右手习惯的 $45^\circ/90^\circ$ 任意旋转），同步重映射 Simai 键位（`1..8`, `D1..D8`, `E1..E8`, `C`）、滑键轨迹符号（`< <-> >`, `p <-> q`, `s <-> z`）与 50 维持续状态张量 `transform_state`。

### Phase 3：生成期平衡引导、动态段落配额与运动学走位搜索（P1，纯推理增强，无需重训即可生效）
- [x] **Todo 3.1 [P0 - 问题 7b]**：新建 [section_profile.py](../../src/maichart/stream/section_profile.py)（`SectionTypeProfile`）并升级 [type_guidance.py](../../src/maichart/form/type_guidance.py) 与 [decoding.py](../../src/maichart/stream/decoding.py)，利用编码器预测的 4 通道负载曲线动态缩放局部目标音符类型比例（`--section-type-strength`，默认 `1.0`），恢复前奏、间奏与副歌的段落类型反差。
- [x] **Todo 3.2 & 3.3 [P1/P0 - 问题 5 & 6c]**：重构 [configuration_search.py](../../src/maichart/stream/configuration_search.py) 与 [decoding.py](../../src/maichart/stream/decoding.py) 的候选提案与引导机制：
  - 采用防御性混合提案分布（Row 0 原生态采样 + Rows $1..K-1$ 注入包含风格、局部和弦与 `PhysicalFeasibilityGuard` 物理走位代价的前缀势函数 `event_bias.biases`），并在 `resample_terminal` 中通过平衡启发式对数密度修正 `_mixture_correction` 严格保持目标分布无偏；
  - 当 `candidates == 1` 或逐 token 回退采样时直接施加 `event_bias.biases`，确保 `local_chords`、`physical` 与 `style` 永不静默失效。
