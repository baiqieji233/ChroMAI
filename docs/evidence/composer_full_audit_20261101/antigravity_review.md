Analyzed COMPOSER_ORDERED_PHRASE_REPAIR_20261031_ZH.md:L1-122
Analyzed COMPOSER_DEEP_CAUSE_AUDIT_20261031_ZH.md:L1-95
Analyzed stream_phrase.py:L1-549
Analyzed stream_phrase_model.py:L1-298
Analyzed stream_quality.py:L1-396
Analyzed stream_search.py:L1-357
Analyzed stream_duration.py:L1-149
Analyzed stream_model.py:L1-450
Analyzed stream_model.py:L650-1000
Analyzed stream_model.py:L1000-1600
Analyzed stream_model.py:L1600-1921
Analyzed stream_actions.py:L1-141
Analyzed stream_sampling.py:L1-195
Analyzed stream_proposal.py:L1-69
Analyzed stream_generate.py:L1-526
Analyzed stream_train.py:L1-350
Analyzed stream_recovery.py:L1-250
Analyzed stream_recovery.py:L250-320
Analyzed stream_model.py:L1010-1085
Analyzed stream_model.py:L1085-1155
Analyzed stream_model.py:L1155-1260
Analyzed stream_model.py:L1720-1790
Analyzed stream_model.py:L1680-1730
Analyzed stream_model.py:L1260-1370
Analyzed stream_model.py:L1502-1671
Analyzed stream_transition.py:L1-150

以下为基于当前工作区 `HEAD 0ff1f5a`（`v38` 全链路：`ordered_phrase_v2` + `audio_state_phrase_v4` + `joint_event_rejection_v2` + `music_exit_v3` + `interval_v1`）的**只读深度审计报告**。本报告严格区分**已证实源码事实/断路**、**架构统计捷径与设计上限**以及**最小算力高收益改进方案**。

---

## 一、 已证实的源码级断路与机制缺陷（Proven Code Facts）

在回答四个顶层架构问题前，先列出当前代码链路中**直接削弱或架空已实现模块**的 7 处精确断路：

### 1. `OrderedPhraseContext` 双塔对比学习存在“时长/BPM/密度哈希”统计捷径与梯度塌缩
* **源码证据**：[`stream_phrase_model.py:41-77`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L41-L77)、[`stream_phrase_model.py:249-275`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L249-L275)。
* **机制事实**：
  1. `ordered_phrase_auxiliary_loss`（[L262-L274](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L262-L274)）用 `prediction = self.ordered.predict(intent)` 与 `labels = self.ordered.encode_sequences(...)` 做 batch 内去重后的 InfoNCE 匹配（`match_loss`）与余弦损失（`cosine_loss`）。注意在 `cosine` 中 `labels.detach()` 截断了梯度（[L270](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L270)），而在 `match_loss` 中 `labels` **带有完整梯度**（[L271](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L271)）。
  2. 查询侧 `intent` 只看当前乐句的声学 `anchor`、整曲固定 `style` 和历史参考；而目标侧 `encode_sequences`（[L57-L70](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L57-L70)）直接输入了**绝对秒数跨度 `relative_seconds`、绝对拍数跨度 `relative_beats`、相邻间隔 `gap_beats / gap_seconds`**。
  3. 同一个 batch 内的 `U` 个唯一乐句来自不同 BPM 的曲目、不同乐句长度（`2, 3, 4, 6, 8` 拍）和不同音符密度。双塔 InfoNCE 只需要让 `sequence_gru` 把 `(乐句总秒数, 乐句总拍数, 平均音符密度)` 编码进 `labels`，就能在 batch 内轻松取得极低的 `match_loss` 和 `cosine_loss`，**完全不需要在 `labels` 中保留精细的键位几何、手顺轨迹或形态序列**。

### 2. 32 事件有序短句被压成单一句末 `128d` 静态向量，导致“配置呼应 = 0%”
* **源码证据**：[`stream_phrase_model.py:76`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L76)、[`stream_phrase_model.py:115-122`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L115-L122)、[`stream_phrase_model.py:191-212`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L191-L212)。
* **机制事实**：
  1. `encode_sequences` 将最多 32 个有序事件跑完 `sequence_gru` 后，只取最后一个时间步 `pooled = sequence[batch_index, last]`（[L76](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L76)），并在 `CausalPhraseContext.forward`（[L208-L210](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L208-L210)）中与 4 个参考段及 `null_reference` 加权求和为**整个乐句恒定不变的单一 128 维向量 `intent`**。
  2. 在乐句内部第 $t$ 步解码时，`actor_residual`（[L115-L122](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L115-L122)）仅把静态 `intent` 与当前已写直方图 `phrase_progress`、剩余拍比 `remaining` 拼接。**解码器在当前乐句的相对拍位（如第 1.5 拍）根本无法按相对拍位对齐检索参考乐句在第 1.5 拍做了什么动作**。这直接解释了为何 [`COMPOSER_DEEP_CAUSE_AUDIT_20261031_ZH.md:71-75`](file:///e:/simai_transformer/docs/COMPOSER_DEEP_CAUSE_AUDIT_20261031_ZH.md#L71-L75) 中 `ordered_phrase_v2` 启用后**配置呼应仍为 `0% / 0%`**。

### 3. `OrderedPhraseContext` 对对称形态与 `shape == 0`（Literal 复合动作）存在相位表征坍缩
* **源码证据**：[`stream_phrase.py:347`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase.py#L347)、[`stream_phrase_model.py:53-54`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L53-L54)。
* **机制事实**：
  1. `encode_sequences` 计算事件相位表征时使用 `phase_values = (phase_mask @ model.phase_embedding.weight) / phase_mask.sum(-1)`（[L53-L54](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L53-L54)），且**完全没有输入相邻事件间的相位旋转量（`movement`）或实际按键坐标**。
  2. 对于 `shape == 0`（约占官谱 12% 的所有 literal 复合 Hold/Slide/多 Touch 事件），编码器的 `phase_mask` 恒为 `[True, False, ...]`（相位恒为 0），因此 `phase_values` 恒等于常数 `phase_embedding[0]`，有序短句编码器完全看不见 literal 事件的真实按键方位。

### 4. `music_exit_v3` 的 `template_keep` 在训练与生成两端直接架空 `audio_state_phrase_v4`
* **源码证据**：[`stream_model.py:1118-1126`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1118-L1126)、[`stream_model.py:1210-1217`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1210-L1217)、[`stream_model.py:1320-1326`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1320-L1326)、[`stream_model.py:1337-1353`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1337-L1353)。
* **机制事实**：
  1. **生成侧**：在 [`sample_event_intent`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1337-L1353) 中，模型**先**在 `mask`（保持上一拍的 `template_reference_shape`）与 `~mask`（切换模板）之间做二元决策 `group`。一旦 `group == 0`（保持模板），`routes` 被掩码到单一 `reference_shape`，其语义类型族 `family` **被 100% 强制锁死为上一事件的 family**，`audio_state_phrase_v4` 在该候选时刻计算出的 31 维声学-短句类型分布被完全旁路。
  2. **训练侧**：在 [`event_route_log_probs`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1318-L1323) 中，先用 `grouped_event_log_probs` 把每个类型族的概率校准为 `family_logits`（即 `audio_state_phrase_v4` 的 `full_logits`），但紧接着第 1323 行的 `two_group_log_probs(..., template_keep_logit)` 用模板保持概率覆盖了 `reference_shape` 与其余所有路由的相对质量，从而**改变了 31 个类型族的边际分布**（[`stream_model.py:1119`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1119) 重新用 `grouped_logsumexp(event_routes, ...)` 覆写了 `output['event_family_logits']`）。
  3. 更隐蔽的是：在 [`family_context_at`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1210-L1215) 中，当 `event_family_audio_weight > 0`（默认 `0.25`）时，`proposal = audio_logits.detach() + correction[..., :31]` **把 `audio_logits` 截断了梯度**！这意味着主干 `family_nll`（权重 `1 - alpha = 0.75`）对 `audio_event_family_head / family_audio_projection / family_rhythm_projection` 的梯度为 **0**，75% 的类型族梯度只流向 `family_state_phrase` 残差和 `continuity_head`。

### 5. `DurationAudioScorer`（`interval_v1`）仅覆盖单段 Regular 动作，且忽略候选时移对 `ordered_residual` 的影响
* **源码证据**：[`stream_duration.py:39-61`](file:///e:/simai_transformer/src/maichart/composer/stream_duration.py#L39-L61)、[`stream_model.py:653-671`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L653-L671)、[`stream_model.py:991-994`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L991-L994)。
* **机制事实**：
  1. `DurationAudioScorer`（[`stream_duration.py:42-60`](file:///e:/simai_transformer/src/maichart/composer/stream_duration.py#L42-L60)）只注册 `shape > 0` 且 `len(slide_segment_keys) == 1` 的单段常规 Hold/Slide（`literal_candidates_supported: False`）。所有 `shape == 0`（多段折线 Slide、同起点双 Slide、含 TouchHold 复合动作等）**完全没有 `duration_audio` 持续区间声学读取**。
  2. 在推理与恢复采样时，`hidden` 中的 `ordered_residual`（[`stream_phrase_model.py:115-122`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L115-L122)）是在游标 `step_beat` 处按当时的 `remaining` 计算并加到 `hidden` 上的；当 [`action_context_for_delta`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L653-L671) 评估不同候选时刻 `event_seconds` 时，**并未按候选落点拍位更新 `phrase_output` 的 `remaining` 或跨乐句边界切换 `phrase_intent`**（若 `step_beat` 在上一乐句末尾而 `step_beat + delta` 跨入下一乐句，`action_context` 和 `family_state_phrase` 读到的仍是上一乐句的 `phrase_intent`）。

### 6. `stream_search.py` 与 `stream_quality.py` 的四处评估/搜索断路
* **源码证据**：[`stream_search.py:211-212`](file:///e:/simai_transformer/src/maichart/composer/stream_search.py#L211-L212)、[`stream_search.py:264-265`](file:///e:/simai_transformer/src/maichart/composer/stream_search.py#L264-L265)、[`stream_quality.py:347-382`](file:///e:/simai_transformer/src/maichart/composer/stream_quality.py#L347-L382)。
* **机制事实**：
  1. **默认单候选零干预**：`phrase_candidates` 默认为 `1`（[`stream_generate.py:40`](file:///e:/simai_transformer/src/maichart/composer/stream_generate.py#L40)），此时 `PhraseQualityScorer.score_phrase` 的 `score_effect` 为 `'diagnostic_only_single_candidate'`，双手舒适度束搜索 `_trajectory_comfort` 对生成结果**零干预**。
  2. **多候选（`phrase_candidates > 1`）窗口不对称**：在 [`stream_search.py:264-265`](file:///e:/simai_transformer/src/maichart/composer/stream_search.py#L264-L265) 中，`candidate_index == 0` 保留上一轮未提交的后 4 拍 `retained`（只新生成 `[start+4, start+8]`），而 `candidate_index >= 1` 丢弃 `retained` 从 `start` 重新生成整整 8 拍 `[start, start+8]`，但两者却在同一个 `[start, stop]` 窗口上比拼 `quality['total']`。
  3. **固定 4/8 拍搜索网格与动态声学乐句边界错相**：`search_stream` 固定按 `commit=4.0, write=8.0` 拍步进，完全未对齐 `music_phrase_layout(bank).boundaries`（`2, 3, 4, 6, 8` 拍可变边界，[`stream_phrase.py:174-202`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase.py#L174-L202)）。
  4. **评分器不看远端重现段且只比 5 维类型直方图**：[`stream_quality.py:350-377`](file:///e:/simai_transformer/src/maichart/composer/stream_quality.py#L350-L377) 的 `reference_points` 只取紧邻前一段 `[start - span, start]`，**从不读取 `PhraseLayout.references` 中的远端声学重现段（`far_top1 / far_top2`）**；且 `similarity` 仅比较 5 维音符类型边际占比（权重仅 `0.025`），完全不评估节奏对齐或几何配置呼应。

---

## 二、 四大底层假设挑战与深层原因剖析

### 问题 1：整段手顺与声学呼应究竟在哪里被训练约束？loss 快速下降是否只学到统计捷径？
* **核心判断**：**是的。当前训练目标中，“整段手顺（多步双手人体工学）”和“跨段配置呼应（相对拍位几何重现/镜像/旋转）”实际上没有任何显式梯度约束；loss 快速下降主要来自局部一阶转移与粗粒度统计捷径。**
* **具体机制拆解**：
  1. **整段手顺在训练期是“开环单步 Teacher Forcing”**：
     训练损失 [`stream_loss`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1673-L1917) 严格是逐时刻单步交叉熵之和。相位头 [`event_phase_logits`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1395-L1425) 虽然接收了 `reference_phase`、`motion_distribution` 和 `button_load`（4 窗口按键近期负荷），但**模型在训练时从未因为“第 $t$ 步选了顺手相位 $p_t$ 却导致第 $t+2, t+3$ 步双手交叉拧麻花或大幅度反手”而受到任何多步联合惩罚**。[`stream_quality.py:136-210`](file:///e:/simai_transformer/src/maichart/composer/stream_quality.py#L136-L210) 中的双手状态机 `_trajectory_comfort` 只存在于推理端诊断，训练端完全不可见。
  2. **声学呼应被“静态池化 + 邻段局部惯性”短路**：
     在 [`stream_phrase.py:238-246`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase.py#L238-L246) 中，每个乐句的 4 个参考段恒为 `[segment-1, segment-2, far_top1, far_top2]`。在训练时，紧邻前一段 `segment-1` 与当前段共享完全相同的曲风、BPM 与局部密度惯性，且 [`CausalPhraseContext.forward`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L201-L209) 对 `phrase_reference_weights` 没有任何选段监督。模型最省力的降 loss 捷径就是：**用 `null_reference` 或 `segment-1` 预测当前段的平均类型占比与音符密度**（这就能把 `phrase_loss` 和 `family_loss` 快速压低），而完全放弃学习“当 `far_top1` 声学相似度高时，按相对拍位复刻或镜像 `far_top1` 的 shape/phase 序列”。

### 问题 2：有序短句的压缩、多解、检索、目标错配，是否仍让创作退化？
* **核心判断**：**是的，当前 `ordered_phrase_v2` 存在严重的“信息瓶颈 + 一对多均值模糊 + 坐标错配”三重退化。**
* **具体机制拆解**：
  1. **压缩退化（32 事件 $\to$ 单一 `128d` 句末向量）**：如前文断路 2 所述，把 4～8 拍内多达 32 个复杂事件压缩成单一句末向量 `pooled`，抹平了乐句内部的**时间相位结构（Temporal-Beat Alignment）**。声学呼应的本质是“在相对拍位 $\tau \in [0, 1)$ 处重现对应的节奏型与几何动作”，而不是整段共享一个静态偏置向量。
  2. **多解模糊（One-to-Many 平均化塌缩）**：同一段音乐既可以写顺时针流转、也可以写逆时针镜像、还可以写对称双押。`CausalPhraseContext` 让 `intent` 仅凭当前段音频 `anchor`、全局 `style` 和历史参考去线性预测目标段的 `target_phrase_summary`（L1 loss）和 `target_phrase_sequence`（余弦/对比 loss）。由于 `intent` 看不到当前段真正选择的起始破局动作，面对“同一音乐多种合法写谱方案（尤其是旋转/左右镜像等价类）”时，确定性回归预测 `self.ordered.predict(intent)` 只能收敛到**所有可能写法的条件期望（模糊均值）**。
  3. **检索与目标错配（绝对相位 vs 相对运动）**：官谱在重复副歌时，往往采用**镜像（左右翻转）或旋转平移（整体旋转 $+k$ 键）**而非死板复刻绝对键位。但 `encode_sequences`（[`stream_phrase_model.py:53-54`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py#L53-L54)）编码的是**绝对 `phase_mask`** 而非**旋转/镜像不变的相对运动序列（`delta_phase`、手间张角、流向变化）**，导致镜像重现的两段副歌在编码空间里反而互相冲突。

### 问题 3：生成端评分、物理状态、类型条件分解、强度曲线是否还有新的断路？
除前文已列出的断路 4、5、6 外，还有三处具体机制缺口：
1. **物理状态 `STATE50` 在 `StatePhraseFamilyResidual` 中被过度池化为 9 个全局标量**：
   在 [`stream_actions.py:107-130`](file:///e:/simai_transformer/src/maichart/composer/stream_actions.py#L107-L130) 中，`candidate_state` 将 `STATE50` 推进到候选时刻后，对所有活跃槽位做了均值/最大值池化，只输出 9 个标量（4 类占比、活跃总数、`remaining` 均值/最大值、`waiting` 均值、`until` 均值）。它抹去了**空间方位冲突信息**（例如当前正按着哪个键的 Hold、哪条 Slide 轨迹正在横穿屏幕中央），而真正需要候选时刻空间占用状态的几何解码器 [`action_context_at`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L615-L651) 读到的 `action_state` 仍然是**游标时刻 `step_second` 未经时间推进到 `event_seconds` 的旧状态**！
2. **`GreedyDecisionFrontier`（0 温确定性重放）中的冗余单支节点**：
   在 [`stream_sampling.py:126-129`](file:///e:/simai_transformer/src/maichart/composer/stream_sampling.py#L126-L129) 与 [`stream_model.py:1568-1571`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1568-L1571) 中，`sample_event_intent` 已经调用 `frontier.choose` 选定了 `family_override`，但进入 `decode_event` 后第 1569 行在 `routes` 已被掩码到单一 `family_override` 的情况下，又对 `masses`（此时 31 维中只有 1 维有限）调用了一次 `choose(masses)`，往决策树里塞入了一个无分支的冗余层。
3. **强度曲线 `_Curve` 在多候选搜索与 `WAIT` 偏置上的尺度不对称**：
   在 [`stream_generate.py:285, 299-301`](file:///e:/simai_transformer/src/maichart/composer/stream_generate.py#L285) 中，`curve.bias`（对数事件率比值）被同时加到了 `kinds[NOTE]`（一次加在游标处）和所有 `timing` 候选落点上；但当 `timing` 全部进入 `rest_mask` 触发强制 `WAIT`（`cursor + 2` 拍）时，若休止区不足 2 拍，固定 2 拍的 `WAIT` 会直接跳过休止区尾部的出音点。

---

## 三、 不指数增加算力的最小高收益演进路线（Minimal High-Return Roadmap）

遵循“不推翻现有流式主干、不增加指数级搜索算力、精准打通手顺与呼应闭环”的原则，建议在 `v38` 评估后按以下 **4 个最小改动包（均可保持单次前向与 $O(T)$ 复杂度）** 推进：

### 1. 消除有序短句的“统计捷径”与“单向量瓶颈”：相对运动编码 + 拍位对齐检索（Beat-Aligned Phrase Cross-Read）
* **改动点**（[`stream_phrase.py`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase.py)、[`stream_phrase_model.py`](file:///e:/simai_transformer/src/maichart/composer/stream_phrase_model.py)）：
  1. **剥离时长/BPM 捷径**：从 `encode_sequences` 输入中移除绝对秒数 `relative_seconds / gap_seconds`，只保留**归一化相对拍位进度 $\tau = (b - b_{\text{start}}) / (b_{\text{end}} - b_{\text{start}}) \in [0, 1]$**、`gap_beats`、以及**相邻事件间的相对相位差 $\Delta \text{phase} \in \{0..7\}$ 与手间张角（chord span）**（对旋转/镜像具有不变性或协变性）。同时对 Literal 事件（`shape == 0`）从 `atoms` 提取真实按键重心相位，不再恒填相位 0。
  2. **按相对拍位 $\tau$ 插值读取参考短句（零额外序列长度开销）**：
     不要只取 `sequence_gru` 的句末向量 `pooled`！将每个乐句的 `sequence_gru` 隐藏状态按相对拍位 $\tau \in [0, 1]$ 池化到固定的 **$K=8$ 个等分拍位槽（shape: `[segment, 8, 128]`）**。在解码第 $t$ 步（候选落点拍位为 $b_{\text{cand}}$）时，直接按候选落点在当前乐句中的相对进度 $\tau_{\text{cand}} \in [0, 1]$ 索引参考乐句对应的拍位槽特征。这样无需展开 32 事件 Cross-Attention，就能以 $O(1)$ 开销让当前第 1.5 拍精准看到参考副歌在第 1.5 拍的节奏与动作形态！

### 2. 理顺 `music_exit_v3` 与 `audio_state_phrase_v4` 的层次关系，并把 `STATE50` 时间推进下沉到 `action_context`
* **改动点**（[`stream_model.py`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py)、[`stream_actions.py`](file:///e:/simai_transformer/src/maichart/composer/stream_actions.py)）：
  1. **先定语义类型族 `family`，再在同族内决定 `template_keep`**：
     将 `template_keep` 限制在已选定的 `family` 内部（仅当 `family == family(template_reference_shape)` 时才允许触发 `template_keep`），彻底杜绝 `template_keep` 跨类型族强行锁死上一拍 family、架空 `audio_state_phrase_v4` 的问题；同时移除 [`stream_model.py:1213`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L1213) 中阻断 75% 梯度的 `audio_logits.detach()`。
  2. **候选时刻 `STATE50` 空间推进共享给 `action_context_at`**：
     把 [`StatePhraseFamilyResidual.candidate_state`](file:///e:/simai_transformer/src/maichart/composer/stream_actions.py#L107-L130) 中将 `remaining / until` 减去 `gap = event_seconds - step_second` 并清零已结束动作的逻辑，直接应用到 [`action_context_at`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py#L622-L645) 读取的 32 槽 `STATE50` 上，让几何解码器（`shape / phase / detail`）看到的是**候选时刻 `event_seconds` 真实存活的按键与 Slide 占用**，而非游标处的过期占用。

### 3. 训练期引入零搜索开销的“可微手顺势能（Ergonomic Flow Margin Loss）”
* **改动点**（[`stream_model.py`](file:///e:/simai_transformer/src/maichart/composer/stream_model.py)）：
  * 不需要在训练期跑昂贵的束搜索：对于每个 `NOTE` 步，模型已经输出了 8 个相位的分布 $P(\text{phase} \mid \text{context})$。利用前 2 个真实音符的按键位置 $(p_{t-2}, p_{t-1})$ 与时间间隔 $(\Delta t_{t-1}, \Delta t_t)$，可预先向量化计算出当前 8 个候选相位各自的**物理转向角速度与同手连击代价 $C_{\text{erg}}(p_{t-2}, p_{t-1}, p) \in \mathbb{R}^8$**，在训练损失中加入一项轻量级的期望人体工学惩罚 $\lambda_{\text{erg}} \sum_{p=0}^7 P(p) \cdot \max(0, C_{\text{erg}}(p) - C_{\text{erg}}(p^*))$（仅惩罚比官谱真实相位 $p^*$ 更别扭的相位）。计算量仅为 `[B, T, 8]` 的张量点积（训练耗时增加 `< 0.5%`），却能直接把多步手顺约束注入 `phase_head`。

### 4. 修复生成端 `stream_search.py` 与 `stream_quality.py` 的对齐与评分闭环
* **改动点**（[`stream_search.py`](file:///e:/simai_transformer/src/maichart/composer/stream_search.py)、[`stream_quality.py`](file:///e:/simai_transformer/src/maichart/composer/stream_quality.py)）：
  1. 让 `search_stream` 的提交边界对齐 `music_phrase_layout(bank).boundaries`，并消除 `candidate 0`（保留 `retained`）与 `candidate 1..3`（丢弃 `retained`）的起点不对称。
  2. 在 `PhraseQualityScorer.score_phrase` 中，直接读取 `layout.references[segment]` 的远端高相似声学参考段（`far_top1`），并加入**拍位对齐的节奏/相对旋转运动相似度**评分项，使多候选搜索真正奖励前后乐句的声学与配置呼应。
