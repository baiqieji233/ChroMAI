# Composer V33 深度复审与根因修复方案（基于最新修复实现的校准与复核）

> 完成补记：下文行号对应独立复核时的源码，最终请按函数名定位。浮点一拍边界已修复并通过回归；延续概率固定上限没有采纳。实施、双种子结果、性能成本与残留问题以[最终交接](COMPOSER_ABSOLUTE_ONSET_AND_EVENT_FAMILY_REPAIR_20261030_ZH.md)为准。

- **审查日期**：2026-10-30（第二轮校准与新实现走查）
- **审查范围**：`event_set_stream_v1` 最新工作区源码（[`stream_model.py`](../src/maichart/composer/stream_model.py)、[`stream_timing.py`](../src/maichart/composer/stream_timing.py)、[`stream_actions.py`](../src/maichart/composer/stream_actions.py)、[`stream_literal.py`](../src/maichart/composer/stream_literal.py)、[`stream_sampling.py`](../src/maichart/composer/stream_sampling.py)、[`stream_proposal.py`](../src/maichart/composer/stream_proposal.py)、[`stream_generate.py`](../src/maichart/composer/stream_generate.py)、[`stream_data.py`](../src/maichart/composer/stream_data.py)、[`stream_train.py`](../src/maichart/composer/stream_train.py)、[`muri/runtime.py`](../src/maichart/generation/muri/runtime.py)）
- **审查目标**：
  1. 修正上一版文档中与最新源码不符的 3 处表述；
  2. 对当前已落地的五大修复（`prenorm_independent_v2`、`phase_gap_v2` 22 桶 + 区间预期事件数、`sqrt_window_v2`、`audio_prior_transition_v3` 共享延续率 $\rho$ 混合、`type_blocks_v1` 字面量类型分块 + `STOP` hazard）做逐行数学与代码走查，指出剩余的**直接 Bug 与关键风险点**。

---

## 一、 前版文档的 3 处事实校准说明

1. **`literal_head_bits` 已覆盖全部 33 个 Touch 传感器（共 41 位）**：
   - 在 [`stream_model.py:324-342`](../src/maichart/composer/stream_model.py#L324-L342) 中，`literal_head_bits` 通过 `head_pad_indices_for_policy(event_metadata(text), 'separate_pads_v2')` 构建，已完整覆盖 `0..7`（8 个外圈键）与 `8..40`（33 个 Touch 传感器 `A1..E8, C`），并非仅覆盖外圈数字键。
2. **MiaCode 运行时默认 `excludeTouchFromMultiTouch=True`，不按 Touch 圆相交判 `MultiTouch`**：
   - 在 [`muri/runtime.py:301`](../src/maichart/generation/muri/runtime.py#L301) 中，`check_multi_touch` 默认参数为 `excludeTouchFromMultiTouch: bool = True`（第 307 行 `if excludeTouchFromMultiTouch and note.type == "touch": continue`），因此纯 Touch 传感器之间的圆形判定区相交不会触发 `MultiTouch`。此前生成中的 `MultiTouch` 冲突主要来自旧字面量解码抽出的异常多原子/TouchHold 与场上活跃 Hold/Slide/TouchHold 的跨动作同时占用。
3. **`sample_event_family` 对正温度始终保持 $T=1$ 采样，且 `preserve_sampled_family` 固定重试是防退化机制**：
   - 在 [`stream_model.py:1039-1042`](../src/maichart/composer/stream_model.py#L1039-L1042) 中，`sample_event_family` 仅在 `temperature <= 0` 时走贪心 `argmax`，对任意 `temperature > 0` 均使用 `scores.float().softmax(-1)`（即保持 $T=1$ 校准分布，未用 `0.65` 锐化语义族）。
   - 同时，[`sample_guarded_event`](../src/maichart/composer/stream_sampling.py#L68-L75) 在同一时间点的重试中保持 `preserve_sampled_family=True`，是为了防止当常规 Tap/双押因局部键位冲突失败时，反复重抽语义族把条件接受率偏置到更容易通过守卫的 Touch 上。

---

## 二、 最新落地五大修复模块的走查结论

当前工作区已落地的五项核心改动在架构与主干契约上已基本闭环：
- **Decoder 深层坍缩修复（[`prenorm_independent_v2`](../src/maichart/composer/stream_model.py#L186-L199)）**：8 层 Post-LN 在随机初始化下将 7 个不同前缀的 `centered RMS` 从 `3.18` 压到 `1.96e-6`（`pairwise cosine = 1.0`），彻底解释了为何 `v32` 的 `hidden` 对历史前缀失明；改为 `norm_first=True`（Pre-LN）+ `decoder.norm`（Final LayerNorm，初始化为单位增益 `ones`）+ 逐层独立初始化后，残差主干梯度与前缀区分度直接恢复，且父 checkpoint 权重可无损迁移。
- **选时 22 动态桶与区间预期事件数（[`phase_gap_v2`](../src/maichart/composer/stream_timing.py#L131-L198)）**：将 $0 < \Delta b < 1$（纯拍内细分移动，桶 `0..10`）与 $1 \le \Delta b \le 2$（整拍及跨拍移动，桶 `11..21`）拆开，彻底解决了“当 `cursor` 在整拍 $\phi=0$ 时，$\Delta b=1$ 整拍跳步混入拍内 `grid(1)` 窃取高先验”的病灶；同时接入了 `candidate_expected_counts`（区间积分预期事件数 $\hat{n}_c$）。
- **温和窗口加权（[`sqrt_window_v2`](../src/maichart/composer/stream_data.py#L379-L388)）**：将 8 秒窗口权重从极端反比 `K / M` 改为 $\text{clamp}(\sqrt{K / M}, 0.5, 2.0)$，消除了 16 分音符密集 Tap 段与稀疏段之间 16 倍的单步梯度悬殊。
- **字面量类型分块与独立 `STOP` 风险率（[`type_blocks_v1`](../src/maichart/composer/stream_model.py#L1102-L1168)）**：按 `(Tap, Hold, Slide, Touch, TouchHold)` 5 个子块依次生成，每块首原子必出（`slot=0`），`slot >= 1` 由独立二元 `literal_stop_head` 在 $T=1$ 下决定是否停止（[`stream_model.py:1158-1160`](../src/maichart/composer/stream_model.py#L1158-L1160) 中 `hazard = torch.stack((stop, 0))` 选 `0` 即 `break`，其停止概率 $\sigma(\text{stop})$ 与 [`stream_model.py:1130`](../src/maichart/composer/stream_model.py#L1130) 的 `BCEWithLogits(stop, target.eq(0))` 及 [`stream_model.py:1100`](../src/maichart/composer/stream_model.py#L1100) 的先验偏置 $\ln\frac{\text{blocks}+0.5}{\text{extras}+0.5}$ 完全等价），且 [`stream_loss`](../src/maichart/composer/stream_model.py#L1452-L1455) 与 [`collate_stream`](../src/maichart/composer/stream_data.py#L734-L737) 已完成接线。

在此基础上，我们排查出 **1 处确定的浮点数边界 Bug**、**1 处 `audio_prior_transition_v3` 的动力学放大风险** 以及 **2 处边界细节**，建议在启动正式训练前核对：

---

## 三、 剩余直接 Bug 与关键风险点（优先级排序）

### 1. [直接 Bug — 1 行即可修复] [`stream_timing.py:133`](../src/maichart/composer/stream_timing.py#L133) 中 `gaps.ge(1.)` 的 IEEE-754 `float64` 1-ULP 舍入错桶

请核对 [`stream_timing.py:123-133`](../src/maichart/composer/stream_timing.py#L123-L133)：
```python
def candidates(self, cursor):
    origin = cursor.floor()[..., None]
    phase = self.phases.to(cursor)
    anchored = origin + phase
    first = anchored + (anchored <= cursor[..., None] + 1e-9).to(cursor)
    beats = torch.cat((cursor[..., None].expand(*cursor.shape, 2), first, first + 1), -1)
    return beats, (beats - cursor[..., None]).clamp_min(0)

def candidate_groups(self, gaps):
    if self.group_policy == 'phase_only_v1': return self.group_ids
    return self.group_ids + gaps.ge(1.).long() * len(GRID_FAMILIES)
```
- **触发条件**：当 `cursor` 位于非二进分母的有理数拍位（例如三连音/六连音相位 $\phi \in \{1/3, 2/3, 1/6, 5/6, 1/12, \dots\}$，以 `cursor = float(Fraction(2, 3)) = 0.6666666666666666` 为例）时：
  - 对于与 `cursor` **同相位**的候选点（$\phi = 2/3$），`anchored == cursor`，因此 `first = anchored + 1.0`。
  - 精确有理数意义下，该候选点的步长是整整 $1$ 拍（`step.end - step.start == Fraction(1) >= 1`），因此在 [`_initialize_structured_heads`](../src/maichart/composer/stream_train.py#L332) 和 [`timing_group`](../src/maichart/composer/stream_timing.py#L32) 中，它被严格归入**后 11 个桶（`>= 1` 拍组，`group + 11`）**。
  - **但在 IEEE-754 `float64` 下**：
    $$\text{first} - \text{cursor} = (0.6666666666666666 + 1.0) - 0.6666666666666666 = 1.6666666666666665 - 0.6666666666666666 = 0.9999999999999999 < 1.0$$
    差了整整 1 个 ULP（$1.11 \times 10^{-16}$）！
  - 结果：`gaps.ge(1.)` 返回 **`False` (`0`)**，把这颗原本属于 `gap = 1` 拍的候选点错误地塞进了**前 11 个桶（`< 1` 拍拍内组，`group + 0`）**！在整数拍 `origin >= 2` 时，約有 **25%～42% 的非二进相位**在同相位 $+1$ 拍处都会触发 `gap = 0.9999999999999998 < 1.0` 的错桶。
- **修复方式**：由于 `phase_catalog` 中任意两个不同相位的最小距离 $\ge 1/768 \approx 1.3 \times 10^{-3} \gg 10^{-7}$（且第 127 行已使用 `1e-9` 容差），只需将 [`stream_timing.py:133`](../src/maichart/composer/stream_timing.py#L133) 改为：
  ```python
  return self.group_ids + gaps.ge(1. - 1e-7).long() * len(GRID_FAMILIES)
  ```
  即可保证 `candidate_groups` 与精确 `Fraction` 100% 一致。

---

### 2. [关键机理风险点] `audio_prior_transition_v3` 中无类型条件的共享 $\rho$ 对稀有类（Touch）游程长度的几何放大

请看 [`stream_actions.py:55-68`](../src/maichart/composer/stream_actions.py#L55-L68) 与 [`stream_model.py:953-965`](../src/maichart/composer/stream_model.py#L953-L965) 的平稳转移混合核：
$$P_{\text{full}}(k \mid \text{last}=i) = \rho \, \mathbf{1}[k = i] + (1 - \rho) \, P_{\text{audio}}(k), \qquad \rho = \sigma\big(\text{family\_continuity\_head}(\Delta \text{music}, \text{music}\odot\text{prev\_music}, \ln(1+\Delta t))\big)$$
这套设计有一个非常漂亮的代数性质：在固定 $P_{\text{audio}}$ 且 $\rho$ 与上一时刻类型 $i$ 无关时，转移矩阵 $K = \rho I + (1-\rho) \mathbf{1} p_{\text{audio}}^\top$ 的**无限步平稳边际分布严格等于 $p_{\text{audio}}$**（$p_{\text{audio}}^\top K = p_{\text{audio}}^\top$）。

但需要高度警惕它在**有限步极大似然训练与自回归局部游程（Run-Length）**上的一个不对等效应：
1. **训练阶段 $\rho$ 的梯度由高频类 `Tap -> Tap` 主导**：
   - 在真实谱面中，全部相邻 `NOTE` 对里有 **~75%–80% 都是 `Tap -> Tap`（$y = i_{\text{prev}} = 1$）**。
   - 在 75% 权重的 `event_family_full_loss` 下，由于 `detach_audio=True`，损失 $-\ln\big(\rho \, \mathbf{1}[y = i_{\text{prev}}] + (1-\rho) P_{\text{audio}}(y)\big)$ 只对 $\rho$ 求导：
     $$\frac{\partial \mathcal{L}_{\text{full}}}{\partial \rho} = -\frac{\mathbf{1}[y = i_{\text{prev}}] - P_{\text{audio}}(y)}{P_{\text{full}}(y)}$$
   - 在任意两个相邻音声学特征相近的常规快交互点上，因为真实标签绝大多数满足 $y = i_{\text{prev}}$（`Tap` 接 `Tap`），极大似然最优解会把该音乐相似度和短间隔下的共享 $\rho$ 推高到这些点上所有类型的经验同类型保持率（通常 **$\rho \approx 0.45 \sim 0.65$**）。
2. **生成阶段一旦抽中 1 次 `Touch`，共享的高 $\rho$ 会无差别地赋予 `Touch -> Touch` 同等强度的自保持率**：
   - 尽管平稳边际 $P(\text{Touch})$ 仍等于 $P_{\text{audio}}(\text{Touch})$（因为它通过减少 `Touch` 的新进入率 $(1-\rho) P_{\text{audio}}(\text{Touch})$ 来平衡更长的连段），但**一旦某一步采样进入了 `Touch`（或因某处音频瞬态触发了 1 个 `Touch`），下一步继续出 `Touch` 的条件概率会从 $P_{\text{audio}}(\text{Touch}) \approx 0.06$ 直接跳升到**：
     $$P(\text{Touch}_{t} \mid \text{Touch}_{t-1}) = \rho + (1 - \rho) P_{\text{audio}}(\text{Touch}) \approx 0.45 + 0.55 \times 0.06 \approx 0.48$$
   - 这意味着稀有类一旦触发，其局部游程从独立采样变成了几何分布（可能连出 3～6 个连续单点 Touch，而官谱里单点 Touch 往往是点缀性的 1～2 个，除非是专门的 Touch 划擦段）。
3. **验证与低成本预案**：
   - 在接下来的对等短训验证中，重点观察生成指标中的 **`touch_only_rate` 与 `max_touch_only_run` 的比值**：如果 `touch_only_rate` 已经降回 ~6%–10%，但仍偶发长度为 5～10 的连续单点 `Touch` 串，说明正是共享高 $\rho$ 把单次 Touch 放大成了局部游程。
   - 若出现该现象，无需推翻混合核结构，只需对 `continuity_logits` 加一个温和上界（例如 $\rho = \rho_{\max} \sigma(\dots)$，如 $\rho_{\max} = 0.35$），或让延续分支仅在当前音频本身也具备最低限度同族支持（例如与 $P_{\text{audio}}(i_{\text{prev}})$ 的软门控耦合）时激活。

---

### 3. [边界细节 1] [`stream_model.py:934-935`](../src/maichart/composer/stream_model.py#L934-L935) 中 `family_context_at` 通过 `audio_cache['beats']` 反插值计算 `beats` 的尾端截断

在 [`stream_model.py:934-935, 979-980`](../src/maichart/composer/stream_model.py#L934-L935) 中：
```python
beat_cache = {**audio_cache, 'family_beat_values': audio_cache['beats'][..., None]}
beats = self._read_audio(beat_cache, event_seconds, song_inverse, field='family_beat_values').squeeze(-1)
...
gap_beats = (event_beats.double() - batch['step_beat'].double()).clamp_min(0)
```
- 在 10ms 网格下，曲中任意位置的线性插值误差 $< 10^{-4}$ 拍，完全足够；
- 唯一需要注意的是：当 `event_seconds` 落在全曲最后一个采样网格点之后（即 `audio_cache['seconds'][-1] < event_seconds <= audio_end_sec` 的最后 `<10ms` 尾端）时，`_read_audio` 会将索引 clamp 到 `lengths - 1`，此时插值出的 `beats` 会被截断在 `audio_cache['beats'][-1]`。由于调用 `family_context_at` 的三处地方（`forward` 的 `batch['target_beat']`、`stream_generate.py:211` 的 `beat`、`stream_recovery.py:240` 的 `beat`）其实都已知精确的目标拍位，若可选传入 `event_beats=None`，就能连最后 10ms 尾端的插值截断也一并消除。

---

### 4. [边界细节 2] [`guard.py:140`](../src/maichart/composer/guard.py#L140) 对 `miacode_static:` 同键 Hold 未释放重叠（`miacode_Overlap`）的放行

在 [`guard.py:139-141`](../src/maichart/composer/guard.py#L139-L141) 中：
```python
if issue.severity == 'hard' and (not enforced_only or
    (issue.action == 'enforce' and (issue.evidence.startswith('miacode_runtime:')
     or (self.intercept_tap_on_slide and issue.conflict_type == 'miacode_TapOnSlide')))):
```
- 此前实测谱 [`final_43375136/maidata.composer.json`](../.scratch/absolute_semantic_repair_20261030/final_43375136/maidata.composer.json#L484-L511) 之所以在通过 `EventGuard` 后仍有 `remaining_hard = 2`（同键 `7h[16:3]` 尚未结束又在 7 号键打 `7`），是因为 MiaCode 将同键 Hold 期间的按键重叠标记为 `evidence = "miacode_static:..."`、`conflict_type = "miacode_Overlap"`，而第 140 行只拦截了 `miacode_runtime:`。
- 若希望导出谱的 `remaining_hard` 严格归零，可在第 140 行补充 `or issue.conflict_type == 'miacode_Overlap'`。

---

## 四、 总结判断

当前已落地的五项修改（特别是 **Pre-LN + Final LN 解决 8 层 Post-LN 特征坍缩**、**22 桶动态拍内/跨拍选时分组 + 区间预期事件数**、**`type_blocks_v1` 字面量类型分块 + 独立 $T=1$ `STOP` hazard** 以及 **`audio_prior_transition_v3` 补齐节拍时钟/步长/plan 率特征**）已经全部精准命中了此前造成“漏采 `1/4` 拍 Tap、单步多 Touch 原子堆积、深层历史失效、自由历史头吞噬声学先验”的真正根因。

在启动 GPU 短训前，**最值得立刻顺手修掉的只有 1 处**：
- 将 [`src/maichart/composer/stream_timing.py:133`](../src/maichart/composer/stream_timing.py#L133) 的 `gaps.ge(1.)` 改为 `gaps.ge(1. - 1e-7)`，消除非二进相位在 `float64` 下同相位 $+1$ 拍差 1 ULP 掉进 `< 1` 拍桶的问题。
其余链路（包括 `stream_loss` 中的 `literal_block_nll` 与 `collate_stream` 中的 `pack_literal_type_blocks`）均已核对闭环，可直接进入实验验证。
