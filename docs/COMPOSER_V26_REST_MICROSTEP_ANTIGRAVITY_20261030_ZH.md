# Composer `v26`（`eebc6ec0f191` / `8dd5ebe8bc12`）休止出音、16分连打与32分交互根因核查与修复方案

> **审查范围**：基于只读代码核查（[`event_survival.py`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py)、[`grounded_model.py`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py)、[`button_load.py`](file:///e:/simai_transformer/src/maichart/composer/button_load.py)、[`conserved_actions.py`](file:///e:/simai_transformer/src/maichart/composer/conserved_actions.py)、[`generate.py`](file:///e:/simai_transformer/src/maichart/composer/generate.py)、[`candidates.py`](file:///e:/simai_transformer/src/maichart/composer/candidates.py)、[`curve.py`](file:///e:/simai_transformer/src/maichart/composer/curve.py)、[`acoustics.py`](file:///e:/simai_transformer/src/maichart/composer/acoustics.py)、[`data.py`](file:///e:/simai_transformer/src/maichart/composer/data.py)、[`train.py`](file:///e:/simai_transformer/src/maichart/composer/train.py)）及 `v26` 第 10 轮两份真实生成产物（[eebc6ec0f191](file:///e:/simai_transformer/webui_workspace/generated/eebc6ec0f191/maidata.txt)、[8dd5ebe8bc12](file:///e:/simai_transformer/webui_workspace/generated/8dd5ebe8bc12/maidata.txt) 及其 `.composer.json` 报告）。
> **核心原则**：不归因于“训练轮数不足”，不引入任何新大模块，完全复用现有组件，严格区分**可直接修复（推理/采样/连线即改即生效）**与**必须训练（损失/网络结构微调）**。

---

## 一、 简短结论（四问逐条定性）

1. **问 1（Survival midpoint cell 积分、shape/history 与 tail 同归一、0.25s 计数率 vs 稀疏 cell 瞬时率）**：**全部命中，且存在两个致命断路**。
   - **断路 A（`tail` 被 `shape + correction` 指数稀释且无监督）**：[`event_survival.py:26-34`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py#L26-L34) 中 $\sum_c \exp(\text{base}_c) + \exp(\text{tail}) \equiv 1$ 原本严格守恒；但 [`grounded_model.py:789-792`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L789-L792) 把零均值的 `shape + correction`（峰值 $+1.5\sim +2.0$）加在 96 个候选上、而 `tail` 加 `0` 后一起做 `log_softmax`。由 Jensen 不等式，$\sum_c \exp(\text{base}_c + \text{shape}_c + \text{correction}_c)$ 被放大 $3\sim 6$ 倍，把 `tail` 概率硬生生压扁（两份报告中 730/742 个事件 `survival_skip_count` 均只有 **1 次**！）；且 [`train.py:425`](file:///e:/simai_transformer/src/maichart/composer/train.py#L425) 只 gather 真实候选下标，`no_event_log_prob` 在主损失中**零直接监督**。
   - **断路 B（0.5s/2s/8s 宽窗特征预测 0.25s 计数率 + 稀疏 midpoint cell 跨休止边界失真）**：`workload` 的输入 [`AcousticContext.at()`](file:///e:/simai_transformer/src/maichart/composer/acoustics.py#L83-L95) 最窄窗口也有 `0.5s`（另有 `2s/8s`），1~3 秒休止内仍混入前后高能乐段，导致休止段预测事件率下不去；而远端候选（如 $+8, +12, +16$ 拍）的 midpoint cell 宽达 $0.6\sim 2.5\text{s}$，用休止结束重音点上的高事件率 `rates[c]` 乘整个左半空隙宽度，会把**休止后的重音误算成休止期内的累积风险 $\Lambda$**，反而指数抑制跨休止跳步！
2. **问 2（F23 负担只类内重排、尾部软惩罚仅 slot 0 且取消 canonical 后第二槽绕过）**：**完全属实，生成谱中有直接物证**。
   - **Slot 1 绕过实锤**：[`generate.py:910-911`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L910-L911) 中 `if slot==0 and anti_rep_bias is not None:` 只在 `slot 0` 加惩罚。取消 `canonical_order` 后，双押直接通过交换槽位顺序逃逸惩罚——见 [eebc6ec0f191/maidata.txt](file:///e:/simai_transformer/webui_workspace/generated/eebc6ec0f191/maidata.txt) 第 9 行 `{16}5/3,3/5`、第 20 行 `{16}6/5,5/6`、第 26 行 `{8}3/5,{16}5/3,3/6`、第 108 行 `8/2,2/8`！
   - **16 分连打免罚漏洞**：[`_official_tail_repetition_bias`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L686-L722) 的 `OFFICIAL_RUN_TAILS['repeat'] = (3, 5)` 对第 2、第 3 次连续同键击打惩罚恒为 **0**，且不区分时间间隔是 `2 拍` 还是 `1/4 拍（79ms）`，导致 `{16}3,3`、`{16}6,6`、`{16}4/3,4/3` 畅通无阻。
   - **F23 不影响选时与 Slot 1 的 `STOP` 概率**：[`CompactGroupedActionHead`](file:///e:/simai_transformer/src/maichart/composer/conserved_actions.py#L390-L396) 强制保持 `tap` 类总质量守恒，即便在 79ms 的 16 分步上某键刚打过双押，F23 也**不能提高 Slot 1 的 `STOP` 概率（把双押降级为单点）**，也**不能压低指针选 16 分微步的概率**。
3. **问 3（`grid_family_coherent_v2` 只控绝对相位，无法避免无音乐支持的 32 分）**：**完全命中，且它反而在保护 `{32}` 的噪声概率不被低温削弱**。
   - [`candidates.py:329, 349`](file:///e:/simai_transformer/src/maichart/composer/candidates.py#L329-L350) 将 `denom=8`（`{32}`, `level=3`）与 `{8}`、`{16}` 同归入 `_BINARY_LEVELS = (1, 2, 3)`。在占全曲 98% 的 `binary/neutral` 状态下，`level=3` **从不被标记为 `switch`**！
   - 更致命的是：[`_grid_level_conserved_within`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L296-L315) 在 `Bin 0`（$\le 0.5$ 拍）内**按 `T=1.0` 锁定每个 grid level 的总质量**，`pointer_temperature=0.3` 只在 `level=3` 内部（`+1/8` 与 `+3/8` 之间）降温，**完全不会压低 `level=3`（32分）相对于 `{16}`/`{8}` 的概率**！一旦踩上 `+1/8` 拍，下一步为回到 `{16}`/`{8}` 网格又必须再走 `+1/8` 拍，从而在两份谱面中制造了 20+ 处孤立的 `{32}6,7,{16}6`、`{32}3,4,{16}3` 伪 32 分双连交互。
4. **问 4（未编辑 `follow` 曲线 `_neutral()` 透传是否让 `rest` 预测永不影响生成）**：**完全命中，`plan_rest` 实际上处于“全链路悬空”状态**。
   - 在 [eebc6ec0f191/maidata.composer.json](file:///e:/simai_transformer/webui_workspace/generated/eebc6ec0f191/maidata.composer.json#L6098-L6105) 和 [8dd5ebe8bc12/maidata.composer.json](file:///e:/simai_transformer/webui_workspace/generated/8dd5ebe8bc12/maidata.composer.json#L4720-L4726) 中，`coarse_audio_plan` 在前奏（`0.63s`）和尾奏（`154s~160s`）其实已经**准确预测出 `rest = 0.84 ~ 1.00`**！
   - 但因为：(a) [`curve.py:386`](file:///e:/simai_transformer/src/maichart/composer/curve.py#L386) `_neutral()` 为 `True`，`build_window_overrides` 直接跳过，未把 `plan_rest` 折算进 `workload_override`；(b) [`grounded_model.py:785`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L785) 的 `event_survival` 只读 `decision_workload[..., 0]`，**完全不读 `candidate_plan[..., 5]`（`rest`）**；(c) [`generate.py:596`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L596) 因 `learned_rest=True` 直接跳过了 `plan_values` 休止检查。结果模型学出的 `plan_rest = 0.88` 对选时事件率的影响为 **0**，导致 `8dd5ebe8bc12` 在 `0.59s`（`target_head_rate = 0.0`, `plan_rest = 0.883`）处直接冒出 `{32}B2/B3`！

---

## 二、 根因核对与最小可复用修复方案

### 1. Survival 选时、休止边界与 `plan_rest` 联动修复

#### 根因细节核对
1. **`normalize_survival` 中的 `tail` 稀释**：
   - 现有代码（[`grounded_model.py:791-792`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L791-L792)）：
     ```python
     audio_logits,audio_tail=normalize_survival(base+shape,tail)
     full_logits,full_tail=normalize_survival(base+shape,tail,correction)
     ```
   - `base` 和 `tail` 满足 $\log(\sum_c e^{\text{base}_c} + e^{\text{tail}}) = 0$，即总事件质量 $P_{\text{event}} = 1 - e^{\text{tail}} = \sum_c e^{\text{base}_c}$。
   - 当把 `shape` 和 `correction` 直接加到 `base` 上再与 `tail` 拼接做 `log_softmax` 时，$\sum_c e^{\text{base}_c + \text{shape}_c + \text{correction}_c}$ 膨胀了 $\exp(\Delta_{\text{lse}})$ 倍，破坏了事件率积分决定的窗口总事件概率 $1 - e^{\text{tail}}$！
2. **稀疏 Midpoint Cell 的右半区间误用终点率 + 左端跨休止泄露**：
   - 在 [`event_survival.py:21-25`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py#L21-L25) 中，第 $c$ 个候选代表区间 $[L_c, R_c] = [\frac{t_{c-1}+t_c}{2}, \frac{t_c+t_{c+1}}{2}]$，现有代码用单点 $r_c \cdot (R_c - L_c)$ 作为该区间风险，并把整个 $h_c = r_c (R_c - L_c)$ 计入候选 $t_c$ 之前的累积衰减或落点质量。
   - 当 $t_{c-1}$ 在休止内（低率 $r_{c-1}$）、$t_c$ 在休止结束的进入重音上（高率 $r_c$）时，区间 $[t_{c-1}, t_c]$ 的真实积分风险应由两端梯形均值 $\frac{1}{2}(r_{c-1} + r_c)(t_c - t_{c-1})$ 决定，而候选 $c$ 自身的精细落点窗口不应把远端稀疏间隔（如 $4\text{b} \to 8\text{b}$ 的 $2$ 拍半宽）全部当成“落在 $t_c$ 这一拍”的瞬时落点质量！
3. **`candidate_plan`（含 `plan_rest`）未接入 `survival` 的有效事件率**：
   - `workload` 只看局部 `AcousticContext.at()`（`0.5s/2s/8s` 均值），而 `planner`（`SongContext`）看全曲上下文并显式预测了 `candidate_plan[..., 0]`（`log_event_rate`）和 `candidate_plan[..., 5]`（`rest` 概率）。

#### 修复方案（复用现有张量，零新模块）
- **【可直接修复 + 训练同步受益】A1. 保持 `tail` 生存质量守恒（Conditioned Shape Normalization）**：
  修改 [`event_survival.py:30-34`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py#L30-L34) 的 `normalize_survival(first, tail, correction=None)`：
  - `first`（即 `base`）与 `tail` 已经定义了精确的窗口内总事件对数概率 $\log P(\text{event in window}) = \text{logsumexp}_c(\text{base}_c)$ 与无事件尾概率 $\text{tail}$。
  - `shape` 与 `correction` 应当**仅在窗口内有效候选中重分配条件概率（Conserved Within-Window Redistribution）**，绝不能挤占 `tail` 的概率质量：
    ```python
    def normalize_survival(base, tail, delta=None, valid=None):
        # 1. 先由纯事件率积分确定窗口内总事件 log-mass 与 tail log-mass
        joint_base = torch.cat((base, tail[:, None]), -1).log_softmax(-1)
        event_mass_lse = joint_base[:, :-1].logsumexp(-1, keepdim=True)
        tail_log_prob = joint_base[:, -1]
        if delta is None:
            return joint_base[:, :-1], tail_log_prob
        # 2. shape + correction 只在窗口内候选中守恒重分配，不改变 event_mass_lse 与 tail_log_prob
        shifted = joint_base[:, :-1] + delta
        if valid is not None:
            shifted = shifted.masked_fill(~valid, -1e4)
        within = shifted - shifted.logsumexp(-1, keepdim=True)
        return event_mass_lse + within, tail_log_prob
    ```
  - **效果**：无论 `shape` 和 `correction` 多尖锐，`tail` 永远严格等于 $\exp(-\hat{\Lambda}(t_{\text{prev}}, t_{\max}))$！
- **【可直接修复 + 训练同步受益】A2. 用梯形区间积分算累积风险 $\Lambda(t_{\text{prev}}, t_c)$，限制远端稀疏候选的虚拟宽度增益**：
  在 [`event_survival.py:10-28`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py#L10-L28) 中：
  - 对相邻有效候选间的时间段 $\Delta t_c = t_c - t_{c-1}$，用两端梯形率 $\bar{r}_c = \frac{1}{2}(r_{c-1} + r_c)$ 计算跨区间风险 $\Delta \Lambda_c = \bar{r}_c \Delta t_c$，从而到达 $t_c$ 前的累积风险为严格分段线性积分 $\Lambda_{<c} = \sum_{j < c} \Delta \Lambda_j$（休止结束重音 $r_c$ 不会污染整段前序静音）；
  - 候选 $c$ 自身的代表宽度 `exposure_c` 对远端稀疏跳步做上限截断（例如 `cell_exposure = exposure.clamp_max(0.35)` 用于计算该点落点发生质量 `1 - exp(-rates * cell_exposure)`，而完整未截断的 $\sum \Delta \Lambda_j$ 仍用于累积生存衰减和 `tail`），防止 $+16$ 拍处的单个稀疏锚点仅凭 `2.5s` 的 Voronoi 单元宽度在非休止段吸走过多概率。
- **【可直接修复 + 训练同步受益】A3. 将 `candidate_plan` 的事件率与 `rest` 门控并入 `survival` 的有效 `rates`**：
  在 [`grounded_model.py:785`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L785) 计算 `survival` 的 `rates` 时，复用 [`curve.py:158-163`](file:///e:/simai_transformer/src/maichart/composer/curve.py#L158-L163) 已有的融合公式（让全曲 `SongContext` 的 `plan_event_rate` 与 `plan_rest` 直接参与有效事件率）：
  ```python
  wl_rate = torch.expm1(decision_workload[:, :count, 0].float().clamp(0., 8.))
  plan_rate = torch.expm1(candidate_plan[:, :count, 0].float().clamp(0., 5.)).detach()
  plan_rest = candidate_plan[:, :count, 5].float().clamp(0., 1.).detach()
  # 当 plan_rest 高（如 >0.5）时，直接二次门控压低有效事件率
  rest_gate = (1.0 - 0.85 * plan_rest).clamp(0.02, 1.0)
  rates = (0.65 * wl_rate + 0.35 * plan_rate) * rest_gate
  ```
  - **效果**：无需改动任何网络层，直接把 `v26` 已经学准的 `plan_rest`（在前奏/休止/尾奏高达 `0.85~1.0`）接通到 `event_survival`！
- **【可直接修复】A4. 修复 [`curve.py:386-435`](file:///e:/simai_transformer/src/maichart/composer/curve.py#L386-L435) 的未编辑 `follow` 曲线透传**：
  - 在 [`curve.py:161`](file:///e:/simai_transformer/src/maichart/composer/curve.py#L161) 中，当 `plan_rest >= 0.75` 且 `plan_head_rate < 0.35` 时，允许 `active_gate` 随 `(1 - plan_rest)^2` 降到 `0.0`（并在 `generate_learned_profile` 中将连续 `>= 1.0s` 满足 `plan_rest >= 0.78` 的区间写入 `rest_regions`）；
  - 在 `build_window_overrides` 中，即便 `points == reference_points`，只要 `self.mode in {'follow', 'hard', 'strict'}` 且存在低目标密度点（如 `target_rates < 0.25`），仍对 `model_workload` 的高估噪声做下压。

---

### 2. 16 分单键/双押成员连打修复（Slot 1 绕过、时间尺度盲区与跨类 STOP 调节）

#### 根因细节核对
1. **Slot 1 绕过尾部惩罚**：[`generate.py:910`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L910) 和 [`generate.py:985`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L985) 写了 `if slot==0 and anti_rep_bias is not None:`。取消 `canonical_order` 后，Slot 0 选搭档键，Slot 1 零惩罚选刚打过的键（形成 `5/3 -> 3/5`、`6/5 -> 5/6`）。
2. **`_official_tail_repetition_bias` 对 $\le 1/4$ 拍（`{16}`/`{32}`）高速连打无短间隔感知**：
   - `OFFICIAL_RUN_TAILS['repeat'] = (3, 5)` 意味着连续第 2、第 3 次击打同一按键（`run=1, 2`）惩罚为 **0**，哪怕两次击打仅相隔 `79ms`（190 BPM 下的 `{16}`）！而在 `legacy_v1`（[`generate.py:798-800`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L798-L800)）中原本有专门的 `(beat - evt_beats[0]) <= Fraction(1, 4)` 短间隔单键/双押连打阻尼。
3. **F23 (`PerButtonLoadV2`) 类内守恒无法在高速步上抑制双押（Slot 1 `STOP`）或抑制微步选时**：
   - [`button_load.py:84-88`](file:///e:/simai_transformer/src/maichart/composer/button_load.py#L84-L88) 的 `run` 特征不区分事件间隔是 `4 拍` 还是 `79ms`；
   - [`conserved_actions.py:395`](file:///e:/simai_transformer/src/maichart/composer/conserved_actions.py#L395) 强制 `tap` 类总概率守恒，使 `PerButtonLoadV2` 在 Slot 1 上无论按钮多累都不能提高 `STOP` 的概率。

#### 修复方案
- **【可直接修复（生成与恢复采样立即生效）】B1. 将 `anti_rep_bias` 扩展到所有 `slot >= 0`，并补回 $\le 1/4$ 拍（或 $\le 110\text{ms}$）高速同键/双押成员阻尼**：
  1. 在 [`generate.py:910`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L910) 与 [`generate.py:985`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L985) 中，去掉 `slot == 0` 限制，改为对所有槽位生效（注意：`anti_rep_bias[0] == 0.0`，即 `STOP` 原子不受负惩罚！因此当 `slot >= 1` 上重复键被惩罚时，不仅换键概率上升，`STOP` 概率也会自然上升，直接把不合理的 16 分双押连打柔和降级为单点！）：
     ```python
     if anti_rep_bias is not None:
         values = values + anti_rep_bias
     ```
  2. 在 [`_official_tail_repetition_bias`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L712-L723) 的 `per_lane_runs` 分支中，加入**短步长（$\Delta b \le 1/4$ 拍）游程加速计数**：
     - 若当前步满足 `(beat - t[0]) <= Fraction(1, 4)`：
       - 若上一事件是双押（`int(e[0]).bit_count() >= 2`）或当前已在 $\le 1/4$ 拍下连续击打该键（`run >= 2`），则有效游程计为 `effective_run = run + 2`（直接触发 `moderate = 1.65` / `strong = 2.35` 阻尼）；
       - 即使 `run == 1`（刚在 $\le 1/4$ 拍前打过一次该键），对**双押成员复读**（即上一事件包含该键且上一事件是双押，或者当前正在解码 `slot >= 1` 试图凑双押）施加 `-1.15` log-odds 阻尼！
     - 这将一举消灭 `{16}5/3,3/5`、`{16}6/5,5/6`、`{16}4/3,4/3`、`{16}6,6/4,4` 以及 `{16}6,6` 四连打，同时完全保留官谱正常的 8 分双押跳拍与主歌正常单键双点！
- **【必须训练（小改现有头，不加新模块）】B2. 让 `PerButtonLoadV2` 输出第 4 维「槽位 `STOP`/负荷标量」并把汇总负担送入 `history_pointer`**：
  1. 在 [`button_load.py:68-89`](file:///e:/simai_transformer/src/maichart/composer/button_load.py#L68-L89) 中，将 `run` 乘上短间隔门控 $\exp(-\Delta t_{\text{last}} / 0.25\text{s})$（使 4 拍前的同键不再算作高速连打游程）；
  2. 在 [`CompactGroupedActionHead.forward`](file:///e:/simai_transformer/src/maichart/composer/conserved_actions.py#L373-L396) 中，计算 `PerButtonLoadV2` 未居中前的 8 键最大/平均过载度 `overload = scores_raw.mean(-1)`，允许其在 `slot >= 1` 时作为标量加到 `macro[..., 0]`（`STOP` 组）上，让模型学会“当刚打完高速双押时，下一拍减少双押头数”；
  3. 在 [`grounded_model.py:757`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L757) 中，将候选时刻的 8 键最高短期负荷 `load_features[..., [3, 10, 14]].amax(dim=-2)`（3 维标量）拼入 `history_pointer`，让选时头也能感知当前双手是否处于刚打完双押的 80ms 极限冷却期。

---

### 3. 无音乐支持的 `{32}`（32 分交互）根因与修复

#### 根因细节核对
1. **为什么 `grid_family_coherent_v2` 会放行甚至保护 `{32}`？**
   - 在 [`candidates.py:329, 349`](file:///e:/simai_transformer/src/maichart/composer/candidates.py#L329-L350) 中，`level = 3`（`denom = 8`, `{32}`）属于 `_BINARY_LEVELS = (1, 2, 3)`。当处于 `binary` 或 `neutral` 状态时，`switch_levels[3]` 为 `False`。
   - 在 [`generate.py:296-315`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L296-L315) 的 `_grid_level_conserved_within` 中，桶内各 grid level 的总质量 `masses[level]` 是用 **`T = 1.0` 的 `base = (values - values.max()).softmax(-1)`** 算出的！只要 `Bin 0` 内两个 `{32}` 候选（`+1/8, +3/8`）在 `T = 1.0` 下的合计概率达到最高 level 的 `8%`（`level_min_p = 0.08`），`masses[3]` 就会被**原封不动保留**，完全不受 `pointer_temperature = 0.3` 的压制！
   - 一旦采样落到 `1/8` 拍，下一步为了回到常规 `{16}`/`{8}` 拍点，又必须走 `+1/8` 拍（`1/8 -> 2/8`），于是必然成对产出 `{32}6,7,{16}6`、`{32}3,4,{16}3`！

#### 修复方案（纯采样器/规则修复，无需重训立竿见影）
- **【可直接修复】C1. 在 `_grid_level_conserved_within` 中将 `level = 3`（`{32}`, `denom = 8`）设为需受温度调控的微步细分级（除非当前已在 `{32}` 游程中）**：
  - 修改 [`candidates.py:371-383`](file:///e:/simai_transformer/src/maichart/composer/candidates.py#L371-L383) 或 [`generate.py:298-308`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L298-L308)：
    - 仅保留 `{4}` (`level=0`)、`{8}` (`level=1`)、`{16}` (`level=2`) 作为二进制基础免降温骨架；
    - 当 `prev.denominator != 8`（即当前**不在** 32 分流水中，而是要从 4/8/16 分**新发起**一段 32 分！）时，将 `level = 3` (`denom = 8`) 的进入质量按 `pointer_temperature` 与主流二进制层一起做低温锐化（或要求 `level = 3` 的单点峰值概率超过更严门槛 `level_min_p_32 = 0.22` 且局部声学瞬态 `attack` / `fine_note_ratio` 达到支持阈值），否则在 `prev.denominator != 8` 时过滤掉新发起的 `denom == 8` 候选；
    - 反之，一旦已经合法进入 `prev.denominator == 8`，允许下一步无缝走 `+1/8` 拍回到 `denom <= 4` 的主网格（绝不卡死在 `1/8` 相位上）。
  - **效果**：真正有强烈音乐爆发/高置信度的 32 分扫键得以保留，而因 `Bin 0` 内 `T=1.0` 尾部噪声守恒而随机蹦出的孤立 `{32}6,7,{16}6` 将 100% 消失！

---

### 4. 可直接修复 vs 必须训练 清单汇总

| 优先级 | 修复项 | 涉及文件与位置 | 类别 | 预期解决的问题 |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | **A1. `normalize_survival` 保持 `tail` 独立守恒**（`shape+correction` 仅在候选内做 `logsumexp` 零和重分配，不稀释 `tail`） | [`event_survival.py:30-34`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py#L30-L34), [`grounded_model.py:791-792`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L791-L792) | **可直接修复**（且继续训练更准） | 立即恢复 `no_event_log_prob`（`tail`）应有的概率质量，解除 `shape/correction` 对休止尾概率的 $3\sim 6$ 倍挤压。 |
| **P0** | **A3 & A4. 将 `candidate_plan`（含 `plan_rest`）接入 `survival` 有效事件率 + 修复 `curve.py` 低密度门控** | [`grounded_model.py:785`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L785), [`curve.py:160-163, 386-425`](file:///e:/simai_transformer/src/maichart/composer/curve.py#L160-L163) | **可直接修复** | 让模型已学出的 `plan_rest = 0.85~1.0` 直接压低休止段 `rates`，根治前奏 `0.59s` 冒 `{32}B2/B3` 与安静段密集出音。 |
| **P0** | **B1. `_sample_actions` 全槽位施加 `anti_rep_bias` + 补齐 $\le 1/4$ 拍（`{16}`）同键与双押成员阻尼** | [`generate.py:712-723, 910, 985`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L712-L723) | **可直接修复** | 彻底堵死取消 `canonical_order` 后 Slot 1 绕过惩罚的漏洞，消除 `{16}5/3,3/5`、`{16}6/5,5/6`、`{16}3,3` 高速复读。 |
| **P0** | **C1. 限制 `prev.denominator != 8` 时 `Bin 0` 内 `level=3` (`{32}`) 的 `T=1` 无条件质量保护** | [`generate.py:296-315, 579-592`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L296-L315) | **可直接修复** | 消除无音乐支撑时因 `T=1` 层级守恒而随机触发的 `{32}6,7,{16}6` 伪 32 分交互。 |
| **P1** | **A2. `event_survival` 改用两端梯形率算累积风险 $\Lambda_{<c}$，并对远端稀疏候选的落点宽度设上限** | [`event_survival.py:18-28`](file:///e:/simai_transformer/src/maichart/composer/event_survival.py#L18-L28) | **可直接修复 + 续训校准** | 消除休止结束重音向休止内部反向泄露风险率，使跨休止跳步落点准确落在重音拍上。 |
| **P2** | **B2. `PerButtonLoadV2` 增加短间隔门控游程，并在 `slot >= 1` 赋能 `STOP` 组偏置及选时微步输入** | [`button_load.py:84-120`](file:///e:/simai_transformer/src/maichart/composer/button_load.py#L84-L120), [`conserved_actions.py:389-397`](file:///e:/simai_transformer/src/maichart/composer/conserved_actions.py#L389-L397) | **必须训练**（可在修完 P0/P1 验证生成后并入下一阶段训练） | 让模型内生学会在高速双押后主动降密或收束为单点，减少对采样期尾部规则的依赖。 |
