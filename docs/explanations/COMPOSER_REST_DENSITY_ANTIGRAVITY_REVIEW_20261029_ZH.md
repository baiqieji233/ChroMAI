# Composer `song_paired_v3` (`v24`) 休止出音、高速双押纵连与宏观分类瓶颈审查报告（可落地定稿）

> **审查范围**：只读代码审查（对应 `song_paired_v3` / `v24` 第 3 轮 4086 步产出 `9ce605238260`）。
> **核心目标**：针对「休止段密集出音」「190 BPM 下 48.63s 处 `5/6, 5/6, 5/7, 5/7`（79ms 双押连打）」「`action_macro_ce = 2.265` 居高不下」以及「前向 8 拍视野与 128 事件历史」，结合最新确认的修改方向（复用 `workload` 事件率构建带无事件尾质量的生存分布 + 均匀探针 Poisson 计数监督、逐按钮多尺度累计负担经共享小网络对所有 slot 类内偏置、8 拍 `future_horizon` 前移参与选时），给出**直接可落地的实现方案、隐藏陷阱修复与冗余删减清单**。

---

## 一、 先给结论：对已确认方向的评估与必补缺口

1. **生存分布直接复用 `workload` 事件率 + 保留无事件尾质量（$\neq \text{EOS}$）+ 均匀时间探针补 Poisson 计数监督**：**方向完全正确且非常优雅**。
   - **必须同时补上的 3 个代码级缺口**（否则仅靠训练端生存分布，推理期仍会在休止段被锁死）：
     - **缺口 A（训练端空隙零计数）**：除了均匀时间探针的 Poisson 计数监督外，**必须把每个真实样本的被跳过空隙 $(t_{\text{prev}}, t_{\text{target}}^-)$ 直接作为 $N=0$ 的精确空区间**加入累积风险损失 $\mathcal{L}_{\text{gap}} = \hat{\Lambda}(t_{\text{prev}}, t_{\text{target}}^-)$。因为事件锚定采样中 98% 样本位于密集段，若不显式惩罚真实休止跳步中间的 $\hat{\Lambda}$，长休止样本的梯度占比依然不足。
     - **缺口 B（推理端 `generate.py` 休止锁死）**：
       1. [`generate.py:618-625`](../../src/maichart/composer/generate.py#L618-L625) 的 `_filter_rhythm_and_active_span_candidates` 只要看到 `plan_values` 中任意一格 `heads >= 1.0 & rest < 0.55` 且无 `heads < 0.5 & rest >= 0.55`，就判定 `span_is_active = True`，**硬屏蔽所有 $\Delta b \ge 6$ 拍或 $\Delta t \ge 1.8\text{s}$ 的候选**（直接把跨休止长跳封死）。
       2. [`generate.py:370-399`](../../src/maichart/composer/generate.py#L370-L399) 中 `bin_temperature` 硬编码为 `1.0`（完全无视 `pointer_temperature = 0.05`），且 `bin_min_p = 0.04` 会在休止段一旦误落一粒音后将远端桶 `Bin 4/5` 永久剪枝锁死。
       3. [`curve.py:160-230, 386-390`](../../src/maichart/composer/curve.py#L160-L230) 中 `generate_learned_profile` 生成的 `rest_regions` 恒为空列表 `[]` 且 `active_gate` 下限为 `0.15`，导致 `points == reference_points` 时 `_neutral()` 直接返回 `True` 透传失效。
2. **逐按钮累计负担 + 共享小网络 + 所有 slot 类内偏置**：**方向精准命中根因**。
   - 现有 123 维 `head_timing`（[`events.py:114-135`](../../src/maichart/composer/events.py#L114-L135)）只有最近一次间隔，无累计击打/双押负荷，且 [`events.py:138-152`](../../src/maichart/composer/events.py#L138-L152) 的 `enrich_head_timing` 是**从未被调用的死代码**。
   - **关键补充**：在 `ConservedGroupedActionHead` 中，Slot 0 直接预测可能包含多键的宏观组（如双押组），或者多 Slot 依次选键；因此共享小网络产出的 8 键偏置 $\delta^{\text{fatigue}}_{c, \text{kind}, b}$ 不仅要在所有 slot 上按类内（`tap/hold/slide` 分别守恒）偏置单键，还要对**双押加权负担（Chord-Strain：某键作为双押/多押成员时的连打次数）**单独统计；同时修复 [`generate.py:688-747`](../../src/maichart/composer/generate.py#L688-L747) 中 `_official_tail_repetition_bias` 只按整事件位掩码全等（`e[run] == e[0]`）判定的漏洞（`5/6 -> 5/7` 会使 `run` 归零，漏掉 5 号键 4 连击）。
3. **8 拍 `future_horizon` 的位置与结构改造**：
   - 现有 `future_horizon` 在 [`grounded_model.py:777`](../../src/maichart/composer/grounded_model.py#L777) 位于选时 `full_logits`（line 754）**之后**，确实只影响配置而不参与选时！
   - 更深层问题是：现有 `FutureAudioHorizon`（[`future_horizon.py:59-126`](../../src/maichart/composer/future_horizon.py#L59-L126)）是在 **96 个候选之间做 $O(C^2)$ 注意力**。而 96 个候选有 75% 挤在 `[0, 4]` 拍内，对于排在 `+4` 拍以后的候选，其后方 8 拍内几乎没有密集候选可读（视野失明）。
   - **落地建议**：将前向 8 拍声学窗口改为**从全曲声学/事件率底座按每个候选后方固定节拍偏移点（如 $+\{0.25, 0.5, 1, 2, 3, 4, 6, 8\}$ 拍）读取固定前向探针**，并**前移到 `full_logits` 之前**，同时喂给生存选时头与动作配置头，直接删掉 $96 \times 96$ 的候选间 `FutureAudioHorizon` 与 `future_gap_evidence`。
4. **`action_macro_ce = 2.265` 的三大根因与根治**：
   - **根因 1（代数抵消导致 `action_head` 零宏观梯度）**：在 [`conserved_actions.py:293-299`](../../src/maichart/composer/conserved_actions.py#L293-L299) 中，`within_log_probs = detail_scores - detail_group_lse` 把 `base_logits = self.action_head(contexts) + F.linear(contexts, factor_weight)` 在每个宏观组内的均值**完全减掉抵消**，宏观概率 100% 只来自 `macro_head`；而 `self.action_head`（`width -> 18000`）白算了全词表投影却对 `macro_ce` 零贡献！
   - **根因 2（后置 4 道全词表 `logsumexp` 截流与算力浪费）**：[`grounded_model.py:906-931`](../../src/maichart/composer/grounded_model.py#L906-L931) 在 `conserved_action_head` 之后串行跑了 `_same_event_logits`、`_cross_event_logits`、`cooperative_geometry`、`_echo_logits` 共 **4 次 $\times$ 3 类 = 12 次 18,000 维 gather/logsumexp/scatter**，其实它们调整的全是 8 键宏观/几何偏置！
   - **根因 3（`sorted(atoms)` 字典序造成的 Slot 0/1 严重不对称与 `QualityContext` 尾事件复读）**：[`data.py:489-490`](../../src/maichart/composer/data.py#L489-L490) 对同事件多音符按 ASCII 字典序排序（`atoms = tuple(sorted(atoms))`），导致任何双押 `(i, j)`（$i < j$）中 **Slot 0 永远只能是小编号键（1..7），Slot 1 永远只能是大编号键（2..8）**，与旋转对称的几何表冲突；同时推理期 `quality_context_strength = 1.5` 放大了最近历史事件的符号拷贝。

---

## 二、 四个模块的具体可落地设计方案

### 1. 复用 `workload` 事件率的生存/风险（Hazard-Survival）选时与 Poisson 计数监督

#### (1) 全曲连续事件率底座与离散区间积分
- 令全曲底座在细时间网格（直接复用 `AcousticContext` / `Workload` 的帧网格或 1/4 拍网格，记步长 $\Delta \tau$）上输出非负事件率（强度函数）：
  $$\hat{\lambda}(t) = \text{softplus}\bigl(f_{\text{rate}}(t)\bigr) + \varepsilon, \quad \varepsilon = 10^{-4}\text{ s}^{-1}$$
  其中 $\hat{\lambda}(t)$ 直接对应 `workload` 中的 `events_per_second`（注意：是**事件到达率**而非头数率，双押计为 1 个事件，避免双押段虚增跳步风险率）。
- 預计算累积风险函数（Cumulative Hazard，前缀和即可 $O(1)$ 查询任意区间积分）：
  $$\hat{\Lambda}(a, b) = \int_a^b \hat{\lambda}(\tau)\,d\tau$$

#### (2) 候选集 + 无事件尾质量（Tail Survival $\neq \text{EOS}$）的统一生存分布
对于当前事件时刻 $t_{\text{prev}}$ 与有效候选集合 $t_1 < t_2 < \dots < t_C$（设最大候选视野边界为 $t_{\text{horizon}} = \max_c t_c$）：
- 每个候选 $c$ 的**生存先验对数质量**为：
  $$\ell^{\text{surv}}_c = \log \hat{\lambda}(t_c) + \log \Delta t_c - \hat{\Lambda}(t_{\text{prev}}, t_c)$$
  其中 $\Delta t_c$ 为候选 $c$ 代表的局部有效时间宽度（或相邻候选中点间距，防止候选网格疏密不均造成密集区总质量虚高——**这一点极其关键**，因为 `[0, 2]` 拍有 60 个候选而 `[8, 64]` 拍只有几个候选，不乘 $\Delta t_c$ 会导致近端积分重复计入 60 次！）。
- **无事件尾质量（Tail Survival Mass，代表「直到 $t_{\text{horizon}}$ 仍未发生事件」，绝不等于全曲结束 `EOS`）**：
  $$\ell^{\text{tail}} = -\hat{\Lambda}(t_{\text{prev}}, t_{\text{horizon}})$$
- 与模型细粒度指针 logit $s^{\text{fine}}_c$ 融合方式：
  $$\text{logit}_c = s^{\text{fine}}_c + \alpha_{\text{surv}} \bigl(\log \hat{\lambda}(t_c) - \hat{\Lambda}(t_{\text{prev}}, t_c)\bigr)$$
  并在桶级（`bin_head`）显式注入各桶覆盖区间的生存积分特征 $\bigl[\hat{\Lambda}(t_{\text{prev}}, b_{\text{start}}), \hat{\Lambda}(b_{\text{start}}, b_{\text{end}}), \exp(-\hat{\Lambda}(t_{\text{prev}}, b_{\text{start}}))\bigr]$。
  - **物理效果**：
    - 当处于 **8~16 拍休止段**时，$\hat{\lambda}(t) \approx 0$ covers `[0, 8]` 拍，近端候选因 $\log \hat{\lambda}(t_c) \to -\infty$ 被自然压低，而跨过休止段后的第一个重音候选 $t_{c^*}$ 满足 $\hat{\Lambda}(t_{\text{prev}}, t_{c^*}) \approx 0$ 且 $\hat{\lambda}(t_{c^*}) \gg 0$，其生存先验直接胜出！
    - 当处于**密集乐段**时，$\hat{\Lambda}(t_{\text{prev}}, t_c)$ 随距离线性暴涨，$-\hat{\Lambda}(t_{\text{prev}}, t_c)$ 自动指数级压制跨有音区间的长跳（天然替代脆弱的硬规则 `span_is_active`）。

#### (3) 双重训练监督：均匀时间探针 Poisson 计数 + 真实事件空隙零计数
1. **均匀时间探针 Poisson 计数损失（覆盖全曲含长休止）**：
   - 在每首歌上按多尺度窗口长度 $W \in \{0.5\text{s}, 2.0\text{s}, 6.0\text{s}\}$ 均匀/随机采样 $M$ 个时间区间 $[a_m, b_m]$，统计真实事件个数 $N_m \in \mathbb{Z}_{\ge 0}$。
   - 预测区间积分 $\hat{\Lambda}_m = \hat{\Lambda}(a_m, b_m)$，施加精确 **Poisson NLL**：
     $$\mathcal{L}_{\text{poisson}} = \frac{1}{M} \sum_{m=1}^M \frac{1}{b_m - a_m} \Bigl(\hat{\Lambda}_m - N_m \log(\hat{\Lambda}_m + 10^{-6})\Bigr)$$
2. **真实相邻事件空隙零计数损失 + 尾存活性 BCE**：
   - 对每个训练样本的真实上一事件 $t_{\text{prev}}$ 与真实下一事件 $t_{\text{target}}$，开区间 $(t_{\text{prev}}, t_{\text{target}} - \delta)$ 内真实事件数严格为 $0$，直接监督：
     $$\mathcal{L}_{\text{empty\_gap}} = \hat{\Lambda}(t_{\text{prev}}, \max(t_{\text{prev}}, t_{\text{target}} - \delta))$$
   - 对任意探测截断点 $\tau_{\text{probe}} \in (t_{\text{prev}}, t_{\text{horizon}})$，定义尾事件指示变量 $y_{\text{tail}} = \mathbb{I}(t_{\text{target}} > \tau_{\text{probe}})$，用生存概率 $S(\tau_{\text{probe}}) = \exp(-\hat{\Lambda}(t_{\text{prev}}, \tau_{\text{probe}}))$ 直接做二元交叉熵监督。

#### (4) 同步修复 `generate.py` 与 `curve.py` 的 3 处休止拦截 Bug
1. **修复 [`generate.py:618-625`](../../src/maichart/composer/generate.py#L618-L625) `_filter_rhythm_and_active_span_candidates`**：
   - 删除「只要区间内没有 `heads < 0.5 & rest >= 0.55` 就一刀切封杀 $\ge 6$ 拍 / $\ge 1.8\text{s}$ 候选」的逻辑；改为基于区间累积风险 $\hat{\Lambda}(t_{\text{prev}}, t_c)$：仅当 $\hat{\Lambda}(t_{\text{prev}}, t_c) > \tau_{\text{active}}$（例如区间内预期事件数 $> 4.5$ 个）**且**近端存在合法候选时才过滤超长跳步。
2. **修复 [`generate.py:370-399`](../../src/maichart/composer/generate.py#L370-L399) `_pointer_bin_settings`**：
   - 将 `bin_temperature` 与 `pointer_temperature` 联动（例如 `bin_temperature = np.clip(0.35 + 0.65 * pointer_temperature, 0.15, 1.0)`），不再固定 `1.0`；
   - 当近端 4 拍累积期望事件数 $\hat{\Lambda}(t_{\text{prev}}, t_{\text{prev}} + 4\text{b}) < 0.8$ 时，不对远端桶 `Bin 4/5` 施加 `bin_min_p = 0.04` 剪枝。
3. **修复 [`curve.py:160-230, 386-390`](../../src/maichart/composer/curve.py#L160-L230)**：
   - 在 `generate_learned_profile` 中，将 `plan_rest >= 0.60` 且预测事件率 `< 0.35/s` 的连续窗口写入 `rest_regions`，并将 `active_gate` 下限从 `0.15` 降为 `0.02`；
   - 在 `_neutral()` 中，当 `self.mode == 'follow'` 且 `self.strength > 0` 时，即使 `points == reference_points` 也允许执行 `plan_rest` 门控压制，不再直接透传。

---

### 2. 逐按钮累计负担（Per-Button Cumulative Load）+ 共享小网络 + 全 Slot 类内偏置

#### (1) 为什么 `5/6, 5/6, 5/7, 5/7`（190 BPM 16 分双押 4 连 = 79ms/次）会漏网？
- 现有 `head_timing` 只记每个头最后一次出现的 $\Delta t$，不知道 5 号键在过去 `240ms` 内已经被连续打了 3 次、更不知道这 3 次全都是**双押（Chord）**！
- 现有 `_official_tail_repetition_bias`（[`generate.py:701-734`](../../src/maichart/composer/generate.py#L701-L734)）用整事件位掩码 `e[run] == e[0]` 判断重复：`5/6` 切到 `5/7` 时整事件不等，`run` 立刻重置为 1；且窗口饱和规则 `limit = 4` 对前 4 击惩罚为 0。

#### (2) 逐按钮多尺度负担特征设计（8 个外圈键 $b \in \{0..7\}$ 独立记账，双押逐键计）
对每个候选时刻 $t_c$ 和每个按钮 $b \in \{0..7\}$，从已提交历史前缀中提取 $d_{\text{btn}} = 14$ 维向量 $u_{c,b}$：
1. **最近一次击打间隔（3 维）**：$\text{seen}_b \in \{0,1\}$, $\log(1 + \Delta t_{c,b}/0.08)$, $\log(1 + \Delta \text{beat}_{c,b}/0.25)$；
2. **多时间尺度指数衰减击打计数（4 维）**：
   $$C_{c,b}^{(\tau)} = \sum_{k \in \text{hits}(b)} \exp\left(-\frac{t_c - t_k}{\tau}\right), \quad \tau \in \{0.08\text{s}, 0.18\text{s}, 0.45\text{s}, 1.2\text{s}\}$$
3. **多时间尺度双押/多押连带负荷（Chord-Strain Load，3 维，专治 `5/6, 5/6, 5/7, 5/7`）**：
   $$S_{c,b}^{(\tau)} = \sum_{k \in \text{hits}(b)} \mathbb{I}(m_k \ge 2) \cdot \exp\left(-\frac{t_c - t_k}{\tau}\right), \quad \tau \in \{0.10\text{s}, 0.25\text{s}, 0.60\text{s}\}$$
   其中 $m_k$ 是历史事件 $k$ 的总按键数（单点 $m_k=1$ 不产生双押疲劳，双押 $m_k=2$ 令两个按键同时累加 $S_{c,b}$！）；
4. **连续同键击打游程（Run-Length，2 维）**：按钮 $b$ 在最近连续多少个事件中从未断过（$\text{streak}_b / 4.0$），以及连续作为双押成员未断过的次数（$\text{chord\_streak}_b / 2.0$）；
5. **相邻键同手连带负荷（2 维）**：$(C_{c, (b-1)\bmod 8}^{(0.18\text{s})} + C_{c, (b+1)\bmod 8}^{(0.18\text{s})}) / 2$ 与其双押分量。

#### (3) 共享小网络与全 Slot 类内（Intra-Kind）守恒偏置
- **共享小网络（跨 8 个按钮共享权重，保持旋转等变性）**：
  - 输入：$[u_{c,b}, g_{\text{music}}(t_c), e_{\text{slot}}]$，其中 $g_{\text{music}}(t_c) \in \mathbb{R}^{16}$ 为当前候选时刻的局部声学冲击力/能量与预测工作量摘要，$e_{\text{slot}} \in \mathbb{R}^4$ 为当前解码槽位指示（让网络知道当前是在选第 1 粒音还是在给双押补第 2 粒音！）。
  - 结构：`Linear(14 + 16 + 4 -> 32) -> SiLU -> Linear(32 -> 3)`，输出每个按钮在 `tap / hold / slide` 三种动作类别上的偏置 $\delta^{\text{btn}}_{n, \text{kind}, b} \in \mathbb{R}^{[N, 3, 8]}$（外乘 `tanh` 门控限幅在 $\pm 3.5$ logit 内）。
- **注入方式（对所有 Slot 生效，且在每个动作类别内居中/守恒）**：
  - 对每个 `kind` $\in \{0, 1, 2\}$，先在 8 个键上做零均值化：
    $$\tilde{\delta}^{\text{btn}}_{n, k, b} = \delta^{\text{btn}}_{n, k, b} - \frac{1}{8}\sum_{b'=0}^7 \delta^{\text{btn}}_{n, k, b'}$$
    这样它**只重分配同类动作在 8 个键上的相对概率，绝不改变 `STOP` 概率或 `tap/hold/slide` 的宏观比例**！
  - 直接加到 `ConservedGroupedActionHead` 的宏观组 logit `macro_raw`（每个组按其包含的 `(kind, head_lane)` 查表累加）以及桶内 `detail_scores` 上，对 **Slot 0、Slot 1、Slot 2 全程生效**。
- **同步加固推理端 `_official_tail_repetition_bias`（[`generate.py:688-747`](../../src/maichart/composer/generate.py#L688-L747)）**：
  - 将规则 1 的整事件全等 `while run < n and e[run] == e[0]:` 增补一条**逐键连续游程统计**：对每个按钮 $b \in \{0..7\}$ 统计最近连续包含按钮 $b$ 的事件数 $\text{lane\_run}_b$，以及最近 `0.35s` 内按钮 $b$ 参与双押的次数 $\text{chord\_hits}_b$；当 $\Delta t < 0.11\text{s}$ 且 $\text{chord\_hits}_b \ge 2$ 时，对继续包含按钮 $b$ 的双押续写施加平滑负偏置。

---

### 3. 根治 `action_macro_ce = 2.265`：消除梯度抵消、合并 18k 词表重归一化与解除字典序偏置

#### (1) 修复 `ConservedGroupedActionHead` 中 `action_head` 的代数抵消与未归一化残差
- **现状问题**（[`conserved_actions.py:270-299`](../../src/maichart/composer/conserved_actions.py#L270-L299)）：
  1. `full_logits = self.action_head(contexts) + F.linear(contexts, factor_weight)` 先算了 `[N, 18000]`；
  2. 随后 `within_log_probs = detail_scores - detail_group_lse.gather(-1, expanded)` 把 `full_logits` 在每个宏观组上的 `logsumexp` **完全减掉**！这意味着对于单原子组（如普通单点 `'1'..'8'` 和 `STOP`），`within_log_probs == 0` 恒成立，`self.action_head` 算出的 logit 被 100% 抹零，梯度恒为 0！
  3. 同时 line 279 的 `macro_raw = self.macro_head(normed) + F.linear(slot_context.float(), macro_factor_weight)` 把 `LayerNorm` 后的输出与**未经 LayerNorm** 的 `slot_context` 线性投影直接相加，导致尺度失衡。
- **落地方案**：
  1. 令 `macro_factor_weight` 同样作用在 `normed = self.macro_norm(slot_context)` 上；
  2. 如果保留 `base_logits` 的组内聚合信息，应在减去 `macro_delta[..., :1]` 之前将 `0.25 * base_group_lse.detach()`（或可学习的低秩组投影）并入 `macro_raw`，或者干脆**删除 `self.action_head = nn.Linear(width, vocab_size)` 这个 18,000 维巨型全连接层**（组间完全由 `macro_head(G)` + `macro_factors` 决定，组内细节完全由 `detail_head` + `factor_weight(353维)` 决定，显存与投影耗时立减 40%，且彻底消除冗余参数！）。

#### (2) 将 4 道后置全词表 `logsumexp` 合并下沉到宏观组/键位空间
- **现状问题**（[`grounded_model.py:906-931`](../../src/maichart/composer/grounded_model.py#L906-L931)）：
  `conserved_action_head` 跑完后，又依次调用 `_same_event_logits`、`_cross_event_logits`、`cooperative_geometry`、`_echo_logits`。这 4 个函数每个都对 `kind in range(3)` 做 18,000 维 `index_select -> logsumexp -> index_copy`，产生 12 道全词表重归一化！
- **落地方案**：
  - 这 4 个模块产出的本质上全都是 **按 `(kind, head_lane, end_lane)` 索引的低维几何/回声/防撞偏置**（加上上面新增的 **逐按钮累计负担偏置 $\tilde{\delta}^{\text{btn}}_{n, \text{kind}, b}$**）。
  - 将这 5 项偏置在进入 18,000 维展开**之前**，先统一汇总为一个张量：
    $$\Delta^{\text{lane}}_{n, \text{kind}, b} = \delta^{\text{same}} + \delta^{\text{cross}} + \delta^{\text{coop}} + \delta^{\text{echo}} + \tilde{\delta}^{\text{btn}}$$
  - 直接在 **`G` 维宏观组空间（`G ~ 260`）** 按 `kind` 做 **1 次** `logsumexp` 守恒归一化并加到 `target_group_lse` 上（仅 `end_lane` 项留在 `detail_scores` 中在桶内归一化）。
  - **收益**：12 次 18,000 维 `logsumexp` 直接缩减为 1 次 260 维 `logsumexp`，梯度直接回传到宏观分类，不再被 4 层后置重归一化截流。

#### (3) 修复 `data.py` 字典序 `sorted(atoms)` 造成的 Slot 0 键位偏置与 `QualityContext` 复读
- **现状问题**：
  - [`data.py:489-490`](../../src/maichart/composer/data.py#L489-L490) 中 `atoms = tuple(sorted(atoms))` 使双押 `(1, 8)` 永远先排 `'1'` 后排 `'8'`，导致 Slot 0 的经验分布严重偏向 `'1', '2', '3'`，而 `_cross_event_logits` 和 `_LocalLeadInContext` 又是按旋转对称设计的；更糟的是，生成期 Slot 0 一旦采样了 `'5'`，[`generate.py:843`](../../src/maichart/composer/generate.py#L843) 的 `canonical_order` 就会把 `'1'..'4'` 全部屏蔽，逼迫 Slot 1 只能在 `'6', '7', '8'` 中选（直接促成了 `5/6, 5/6, 5/7, 5/7`！）。
- **落地方案（轻量无损改法）**：
  1. **训练期同事件多槽对称化（Symmetric Slot Permutation）**：训练构造多槽样本时，对同一事件内的多个非 `STOP` 原子，以 50% 概率保持规范序、50% 概率逆序（或随机排列），并在 `slot >= 1` 的前序已选掩码 `selected_lanes` 中屏蔽已选键位；推理期**移除 `generate.py:843` 的字典序硬截断 `mask[1:lookup[selected[-1]] + 1] = False`**，只保留物理合法性约束（`EventGuard` 同键防重与最大头数限制）。这样 Slot 0 在 8 个键上恢复完全旋转对称，且 Slot 0 选 `'5'` 后 Slot 1 依然可以自由选 `'1'..'4'`！
  2. **修复 `audio_action_logits` 的分组冲突**（[`grounded_model.py:937`](../../src/maichart/composer/grounded_model.py#L937)）：`audio_action_logits` 的训练目标是位置无关的 `audio_macro_group_id`（[`objectives.py:447`](../../src/maichart/composer/objectives.py#L447)），不要再让它复用按位置分组归一化的 `self.conserved_action_head`，直接用一个简单的 `nn.Linear(width, num_audio_groups)` 预测即可。
  3. **给 `QualityContext` 历史符号拷贝加防复读门控**（[`quality_context.py:218-233`](../../src/maichart/composer/quality_context.py#L218-L233)）：在 `QualityContext` 注意力读取 `history_memory` 时，将最近 2 个历史事件（$\Delta \text{beat} < 1.0$）在 `old_symbol` 检索中的注意力权重掩码或衰减（近端由 `_LocalLeadInContext` 和逐按钮负担负责，远端乐句重现才由 `QualityContext` 负责），防止 `quality_context_strength = 1.5` 时直接把上一拍的 `5/6` 原样拷贝过来。

---

### 4. 前向 8 拍固定声学窗口（前移至选时之前）与 128 事件历史去重

1. **8 拍 `future_horizon` 前移并用固定节拍探针替代 $O(C^2)$ 候选间注意力**：
   - 将前向 8 拍读取从 [`grounded_model.py:777`](../../src/maichart/composer/grounded_model.py#L777) **前移至 line 715（`audio_context = audio_features + symbolic` 合成处、选时 `full_logits` 之前）**。
   - **不再在 96 个稀疏不均的候选之间做 `[B, 4, 96, 96]` 注意力**，而是借鉴 `event_ctx` 的固定节拍前瞻思想：对每个候选时刻 $t_c$，按当前 BPM 下的 **10 个固定前向节拍偏移** $\delta_k \in \{0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0\}$ beats，直接从全曲声学底座与 `workload` 事件率曲线插值读取 `[B, C, 10, D_probe]`，经一个轻量深度可分离卷积/MLP 压成 `[B, C, D]` 后并入 `audio_context`。
   - **双重收益**：
     1. 选时头（`pointer` + `bin_head`）能直接看到每个候选后方 8 拍的精确声学包络与事件率走势；
     2. 远端候选（如跨休止后的候选）也能拥有完整的 8 拍前向视野，同时彻底省掉 `[B, 4, 96, 96]` 的 `FutureAudioHorizon` 与 `[B, 96, 96]` 的 `future_gap_evidence`。
2. **清理 128 事件后向历史中的死算子**：
   - 在 [`grounded_model.py:684-692, 779`](../../src/maichart/composer/grounded_model.py#L684-L692) 中，当 `song_paired=True` 时，模型先跑了一遍 `history_encoder` + `history_attention` 算出 `legacy_history_attended` 加进 `symbolic`，然后又在 line 779 的 `_action_history_context` 里把它**原样减掉**！
   - 在 `song_paired_v3` 下应直接跳过 `legacy_history_attended` 的前向计算（或仅保留给选时门控使用而不做加后再减的无效操作）。

---

## 三、 可直接删除的冗余清单（减负提速）

| 序号 | 冗余模块 / 路径 | 所在文件与行号 | 删除/替换方式与算力收益 |
| :--- | :--- | :--- | :--- |
| 1 | `enrich_head_timing` 死代码与旧 123 维线性投影 | [`events.py:138-152`](../../src/maichart/composer/events.py#L138-L152), [`grounded_model.py:471`](../../src/maichart/composer/grounded_model.py#L471) | 替换为新的「逐按钮累计负担 + 共享小网络」模块，不再混入全局 `symbolic`。 |
| 2 | `FutureAudioHorizon` ($O(C^2)$ 候选间注意力) 与 `future_gap_evidence` ($O(C^2)$ 矩阵) | [`future_horizon.py:59-126`](../../src/maichart/composer/future_horizon.py#L59-L126), [`future_gap_evidence.py:10-41`](../../src/maichart/composer/future_gap_evidence.py#L10-L41) | 统一替换为前移至选时前的「固定 10 点前向 8 拍声学+事件率探针」，消除两个 $96 \times 96$ 中间张量。 |
| 3 | 4 道串行全词表 `for kind in range(3): logsumexp` | [`grounded_model.py:906-931, 948-989`](../../src/maichart/composer/grounded_model.py#L906-L931) | 将 `_same_event`、`_cross_event`、`cooperative_geometry`、`_echo` 与逐按钮负担合并，在 `G` 维宏观群空间做 1 次 `kind` 守恒归一化。 |
| 4 | `audio_action_logits` 复用 `conserved_action_head` | [`grounded_model.py:935-937`](../../src/maichart/composer/grounded_model.py#L935-L937) | 改为独立 `nn.Linear(width, num_audio_groups)`，省去整套 18,000 维展开与二次 `group_logsumexp`。 |
| 5 | `song_paired` 下算完又减掉的 `legacy_history_attended` | [`grounded_model.py:684-692, 779`](../../src/maichart/composer/grounded_model.py#L684-L692) | 当 `song_paired=True` 时不再对 128 步历史重复跑旧 `history_encoder` + `history_attention`。 |

---

## 四、 可证伪验证方案（Falsifiable Verification）

在启动 fresh 训练前及训练第 1~3 轮，通过以下 4 项定量测试直接检验假设是否成立：

1. **休止段生存抑制与跳步测试（验证问题 1）**：
   - **方法**：构造合成/真实 8 拍与 16 拍纯静音/环境音休止片段（$\hat{\lambda}(t) < 0.2\text{ s}^{-1}$），对比开启生存先验 + 修复 `_filter_rhythm_and_active_span_candidates` / `_pointer_bin_settings` 前后的候选分布。
   - **通过标准**：在休止起点 $t_{\text{prev}}$ 处，近端 `[0, 4]` 拍内所有候选的合计概率质量 $< 10\%$，跨休止后的首个重音候选（或无事件尾质量 $S_{\text{tail}}$）占据主质量；在 `9ce605238260` 同曲同参数（`pointer_temperature=0.05`）下，休止段误出音数量从密集连打降为 $0\sim 1$。
2. **190 BPM 高速双押疲劳压制测试（验证问题 2）**：
   - **方法**：固定前缀为 190 BPM 下已连续出现两次 `5/6`（间隔 79ms），输入第 3 步（+79ms）与第 4 步（+158ms）候选，检查所有包含按钮 `5` 或 `6` 的双押（包括换另一键的 `5/7`、`5/8`、`4/6`）相对于换手配置（如 `1/2`、`2/3`、`3/4`）或单点的对数几率差。
   - **通过标准**：即使从 `5/6` 切换到 `5/7`，按钮 `5` 的累计负担项 $\tilde{\delta}^{\text{btn}}_{\text{tap}, 5}$ 仍单调下降 $\ge 1.5$ logit；生成结果中 `< 90ms` 的同键参演双押连续游程 $\le 2$。
3. **宏观分类梯度与对称性测试（验证问题 3）**：
   - **方法**：
     1. 检查单步反传中 `macro_head`、合并后的键位偏置头以及 `detail` 因子表对 `action_macro_ce` 的梯度范数；
     2. 在对称上下文输入下，测试 Slot 0 对 8 个外圈键 `1..8` 的先验边际分布方差，以及 Slot 0 选中 `'5'` 后 Slot 1 对 `'1'..'4'` 的可达性。
   - **通过标准**：Slot 0 selected `'5'` 后，Slot 1 选择 `'1'..'4'` 的概率不再被 `canonical_order` 归零；相同轮次（Epoch 3）下验证集 `action_macro_ce` 从 `2.265` 显著下降至 `< 1.85`。
4. **吞吐量与显存回归测试（验证工程效率）**：
   - **通过标准**：合并 12 次 18k `logsumexp` 与替换 $O(C^2)$ `FutureAudioHorizon` 后，单步训练耗时较 `v24` 持平或降低 $\ge 8\%$，峰值显存不增加。
