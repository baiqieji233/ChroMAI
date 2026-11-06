# ChartComposer 全架构隐患排查与改进点清单（证据分级修订版）

日期：2026-10-22（修订版 v2）\
审查范围：`src/maichart/composer/` 全模块及关联的 `form/`、`stream/`、`generation/`、`webui/` 接口

---

## 一、分级准则与修订说明

1. **证据四列化**：本清单将所有模块排查项统一拆分为 **代码事实**、**产物观察**、**机制假设**、**验证方法与实测数据** 四列，严格区分“代码客观实现”、“实测验证结论”与“待 A/B 实验验证的机制假设”。
2. **P0 评级收敛**：
   - **P0（已证实必现缺陷）**：仅用于**有明确代码路径、已被实测数据复现、且必然导致谱面物理/语法错误**的问题（共 2 项：**G-01 跨事件 Slide 轨迹/多手冲突漏检**、**C-02 跨分母体系相对拍距累加导致绝对拍分母 48 `{192}` 跨小节漂移**）。
   - **P1（高价值假设/实测部分证实）**：有实测证据支持其局部影响（如 6ms `attack` 通道插值抬高、4 维段落零占位特征、缺乏推理闭环密度反馈），但对最终谱面质量的主导权重需通过 A/B 对照实验确认。
   - **P2（表征局限/工程优化）**：不构成训练推理错位或直接错误，但属于可进一步提升上限的表征或资源管理改进点。
3. **撤回过激硬规则与未经证实的量化承诺**：
   - 撤回上一版“无需重训即可消除 80% 无理”的未证实百分比承诺；
   - 撤回 `<90ms 禁双押`（190 BPM 下合法 `{16}` 步长即 `78.9ms`）、一刀切封杀 `{48}`/`{64}`、弱音必舍及“每条活动 Slide 无条件占死一只手”等会破坏合法谱面表达的硬规则，统一改为**基于项目内置 `MiaCode` 冲突分析器（[`detect_chart_conflicts`](../../src/maichart/Virtual_Player/conflict_detector.py#L430)）的精确硬冲突拒绝 + 软评分诊断**。

---

## 二、模块级隐患全量排查（按模块与修订后证据等级）

### 1. [`src/maichart/composer/candidates.py`](../../src/maichart/composer/candidates.py)（候选网格、特征构造与音频插值）

| 编号 | 级别 | 位置 | 代码事实 | 产物观察 | 机制假设 | 验证方法、实测数据与最小改进 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **C-01** | **P1 高价值实测发现** | [`composer_audio_nodes`](../../src/maichart/composer/candidates.py#L253-L259)<br>[`interp_candidate_audio`](../../src/maichart/composer/candidates.py#L224-L250) | `song.audio.times` 来自 `extract_rational_audio`（`balanced_evidence_v1`，含瞬态峰、频谱通量峰及首尾节点，8192 点），`interp_candidate_audio` 在相邻节点间对 118 维特征做一维线性插值。 | 安静段落（低于全曲 RMS 中位数）仍出现 `258/576`（`44.8%`）的 $\le 1/4$ 拍（`{16}` 及更密）步长。 | 稀疏节点间的线性插值是否把峰间空白拉成了高能量平台，或导致 `sin/cos` 节拍相位几何塌缩？ | **实测结论（已修正原过度推断）**：<br>1. `8192` 个节点平均间距仅 `19.62ms`（中位数 `18.62ms`，P90 `31.92ms`），因此 **`rms`（`MAE=0.0143`）、`mel`（`MAE=0.0395`）与 `sin/cos` 节拍相位（误差 `0.0200`）均未发生严重失真**；<br>2. **真正失真的是 6ms 窄窗脉冲特征 `attack`（列 `92, 98, 99`）**：全曲 `{16}` 拍点 `attack_mae = 0.2868`；在最低 25% 安静段，`attack` 中位数被线性连线虚高 **2.69 倍**（`0.2340` vs 真实 `0.0871`）；在偏离节点 `>15ms` 的 `{16}` 拍点上，**`35.82%` 的点原生 `attack < 0.05` 却被插值抬高到 `> 0.15`**。<br>**最小改进**：对 `attack` 通道改用最近邻节点距离衰减 $a_{\text{near}} \exp(-\Delta t_{\text{node}} / \tau)$（$\tau \approx 12\text{ms}$）或显式拼接 $\Delta t_{\text{nearest\_node}}$，并对周期 `sin/cos` 改为在 `beat_floats` 上直接解析计算。 |
| **C-02** | **P0 已证实必现缺陷** | [`_FINE_OFFSETS`](../../src/maichart/composer/candidates.py#L16-L24)<br>[`candidate_beats`](../../src/maichart/composer/candidates.py#L177) | 候选拍点通过纯相对加法 `previous_beat + offset` 生成，`_FINE_OFFSETS` 同时包含三连音系（分母 `3, 6, 12`）与二进制系（分母 `2, 4, 8, 16`）偏移，未检查累加后的绝对拍位分母 `cand.denominator`。 | [`2023bb6c2fec`](../../webui_workspace/generated/2023bb6c2fec/maidata.txt) 中有 **132 个事件（占全谱 13.04%）** 处于绝对拍分母 `48`（编译为连续 15.25 小节的 `{192}` 与 `{96}`）。 | 当 `previous_beat` 落在三连音相位（如 `799/3`）时，一旦采样到一个二进制偏移（如 `+1/16`），绝对拍位即变为 `12787/48`，此后连续踩正常的 `1/2`（`{8}`）或 `1/4`（`{16}`）步长也会永远卡在 `/48` 相位上无法脱身。 | **实测结论（100% 复现）**：<br>1. 全量训练集 956,579 个事件中，`beat.denominator == 48` 仅出现 **252 次（0.026%）**，而 `2023bb6c2fec` 单谱即高达 **13.04%（高出语料基线 500 倍）**；<br>2. 语料中合法存在 `delta_beat` 分母 `12`（`0.41%`）与 `16`（`0.20%`），直接删掉相对分母 `12, 16` 会损失 `0.60%` 召回率（新增 5,758 个 miss）。<br>**最小改进**：**不封杀合法的 `{48}`/`{64}` 相对步长**，而是在候选掩码中增加**跨系相位兼容约束（Phase Compatibility Mask）**：禁止三连音相位与深层二进制相位混加产生 `cand.denominator in {24, 48}`；若当前已处于三连音网格（分母含因子 3），仅允许选择能维持三连音网格或落回整数/半拍网格（`cand.denominator in {1, 2, 3, 4, 6, 8, 12}`）的候选。 |
| **C-03** | **P1 已证实特征缺失** | [`composer_audio_nodes`](../../src/maichart/composer/candidates.py#L286) | 第 286 行 `np.zeros((n, 4))` 固定填入 4 列全零特征；但需注意模型仍能看到候选局部声学特征、`progress`、节拍相位与拍号信息。 | 安静前奏/过渡段与高潮段之间的拍密对比度不足。 | 缺少显式的“当前局部能量相对于全曲中位数的比值”及 `song.regions` 段落角色特征，可能削弱模型对主歌/副歌宏观密度落差的把控。 | **验证与最小改进**：保持 118 维接口不变，将 `np.zeros((n, 4))` 替换为局部 2s/8s 窗口 RMS 与全曲中位数之比及 `song.regions` 边界距离，通过重训前后安静段/高潮段 NPS 比值进行 A/B 验证。 |
| **C-04** | **P2 表征增强假设** | [`candidate_features`](../../src/maichart/composer/candidates.py#L186-L221) | 候选几何特征为 5 维连续值（`delta_beat`, `delta_sec`, `cand_progress`, `sin/cos(2π * cand_beat)`），未显式编码候选相对偏移的分音分母类别或候选点与 `previous` 的声学差分。 | 指针在 `{8}`、`{12}`、`{16}` 之间的切换有时不够干脆。 | 显式提供 `offset.denominator` 嵌入以及 `audio(cand) - audio(prev)` 差分特征，可能帮助指针网络更直接识别节奏网格身份与新发声事件。 | **验证与最小改进**：作为新训模型的可选特征扩展项，对比验证集 `pointer_top1_accuracy`。 |

---

### 2. [`src/maichart/composer/model.py`](../../src/maichart/composer/model.py)（网络架构与表征融合）

| 编号 | 级别 | 位置 | 代码事实 | 产物观察 | 机制假设 | 验证方法、实测数据与最小改进 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **M-01** | **P1 架构局限（已修正归因）** | [`ChartComposer.__init__`](../../src/maichart/composer/model.py#L51)<br>[`decode_candidate`](../../src/maichart/composer/model.py#L125-L126) | `action_head` 并行输出 `8 × 18454` 的槽位 logits，模型内部未用已采样的 `slot 0` 条件化 `slot 1`；但推理端 [`_sample_actions()`](../../src/maichart/composer/generate.py#L76-L113) 会逐槽采样并用 `grammar.heads` 过滤同拍重复起始键。 | `2023bb6c2fec` 全谱共检出 28 处 `MiaCode` 冲突（25 处硬冲突，3 处软冲突）。 | 原报告假设“槽位并行独立是同拍碰撞无理的主因”。 | **实测结论（已推翻原主因假设）**：<br>逐条运行 `MiaCode` 诊断证明，**`2023bb6c2fec` 的 28 处冲突中有 27 处是跨事件 Slide 生命周期冲突**（后续事件踩中前序未结束 Slide 的轨迹或导致 3~4 手并发），同拍双押内部仅 1 处 Slide 终点重合（`idx=230: 5-8[8:1]/8>3[8:1]`）被判定为允许的链式滑键。<br>**最小改进**：无需急于将 `action_head` 大改为自回归解码器（降为 P2 远期演进）；当前只需在 `_sample_actions()` 中补充组合级 `MiaCode` 校验即可。 |
| **M-02** | **P2 表征局限（已修正计算）** | [`ChartComposer.forward`](../../src/maichart/composer/model.py#L64)<br>[`_select_candidate_offsets`](../../src/maichart/composer/candidates.py#L64-L99) | `audio_summary` 对 96 个候选隐状态做等权平均。**修正候选分布计算**：在 `_select_candidate_offsets(False, 136, 96)` 中，**48 个候选在 `(0, 2]` 拍内，61 个在 `(0, 4]` 拍内，第 84 个候选（index 83）已到达 `12` 拍，最大到 `16` 拍**（原报告“前 84 个都在 0~4 拍”计算有误）。 | 局部高密连打偶发超出乐句边界。 | 2 拍内候选占了等权平均的 50%（48/96），可能使 `audio_summary` 略偏近场，但远场 `4~16` 拍仍占 35 个候选（36.5%）。 | **验证与最小改进**：不属于 P0/P1 缺陷。后续若重训可尝试按候选时间跨度加权池化，并对比验证集损失。 |
| **M-03** | **P2 表征局限（已修正对齐判断）** | [`ChartComposer.forward`](../../src/maichart/composer/model.py#L101-L107)<br>[`prefix_action_state`](../../src/maichart/stream/context.py#L8-L35)<br>[`generate.py`](../../src/maichart/composer/generate.py#L192) | **训练与推理时刻完全对齐（修正原错误判断）**：训练时 `prefix_action_state(..., event_index)` 内部明确取 `points[committed_events - 1]`（即上一已提交事件 `previous`），推理时 `generate.py:192` 同样在 `previous` 处取状态，**两边不存在时刻错位**。真正的局限在于：96 个不同未来时刻的候选共享同一个 `previous` 时刻的状态向量，没有按每个候选时刻 `cand` 分别提供持续状态。 | 持续长音/滑键期间偶发对何时释放判断不准。 | 缺少每个候选时刻 `cand` 独立的 `active_hands(cand)` 特征，使模型需靠自身隐式推算 Hold/Slide 何时结束。 | **验证与最小改进**：推理端可直接通过增量 `MiaCode` 冲突检测精确拦截未释放冲突；训练端后续可在 `candidate_features` 中补充每个候选时刻的剩余占用时长特征。 |
| **M-04** | **P1 推理闭环缺失** | [`ChartComposer.forward`](../../src/maichart/composer/model.py#L103-L113)<br>[`train.py`](../../src/maichart/composer/train.py#L96) | `future_head` 在训练时已监督未来间隔与头密度等辅助目标（`future_loss * 0.2`），但在推理时其输出未被读取，也不参与对指针或动作分布的闭环校准。 | 生成谱面在局部陷入 `{16}` 连打时缺乏自我纠偏机制。 | 辅助头学到的未来节奏/密度预期若能在推理期用于软评分或监控，可抑制失控连打。 | **验证与最小改进**：在推理期读取 `future_head` 预测值或引入滚动 NPS 账本进行软偏置实验。 |

---

### 3. [`src/maichart/composer/data.py`](../../src/maichart/composer/data.py) 与 [`audio_bank.py`](../../src/maichart/composer/audio_bank.py)（数据集、词表与缓存）

| 编号 | 级别 | 位置 | 代码事实 | 产物观察 | 机制假设 | 验证方法、实测数据与最小改进 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **D-01** | **P1 词表长尾稀释（已修正 `[1:0]` 定性）** | [`ComposerDataset.__init__`](../../src/maichart/composer/data.py#L80-L90) | 1. 词表大小 `18,454`，未对等价分数时值（如 `[8:2]` 与 `[4:1]`）或 BPM 特定字面量做归约，导致 Slide/Hold 变体占据绝大多数词表项；<br>2. **修正对 `h[1:0]` 的定性**：全量语料统计显示，`4h[1:0]`（243 次）、`5h[1:0]`（238 次）、`3hx[1:0]` 等零时值 Hold 在训练集中共出现 **1,210 次（覆盖 30 种原子）**，属于官谱/社区谱常用的“点按式短 Hold”合法惯用记谱法，并非解析器错误。 | 生成谱面中底排单键 `"4"`、`"5"`、`"3"`、`"6"` 合计占单键事件的 **62.1%**，Slide 种类偏少。 | 18,454 维扁平词表将同一几何路径的 Slide 分散到数百个不同时值 ID 上，导致单类 Slide logit 在采样时竞争不过高度集中的裸 Tap ID。 | **验证与最小改进**：<br>1. 保留合法的 `h[1:0]` 惯用法，不在生成端盲目一刀切封杀；<br>2. 后续构建词表时可先做有理数时值规范化（或拆分为“动作几何 + 时值”双头），观察 Slide 生成比例与键位熵的变化。 |
| **D-02** | **P2 样本分布特性（已补充全库统计）** | [`ComposerDataset.__init__`](../../src/maichart/composer/data.py#L157-L166) | 训练按事件等权采样，这是事件级自回归模型的自然定义。 | `2023bb6c2fec` 中 `1/4` 拍（`{16}`）占比达 `43.6%`，而 `1/2` 拍（`{8}`）占 `30.8%`、`1` 拍（`{4}`）占 `12.6%`。 | 原报告假设“训练集被 `{16}` 样本 16 倍垄断”。 | **实测结论（已修正原假设）**：<br>全量训练集 954,784 个步长的真实分布为：**`1/2`（`{8}`）占 `37.31%`（第 1 位），`1`（`{4}`）占 `25.24%`（第 2 位），`1/4`（`{16}`）仅占 `17.58%`（第 3 位）**！说明训练集整体并未被 `{16}` 垄断；生成谱面 `{16}` 飙升至 `43.6%` 主要来自推理期 `attack` 插值虚高与自回归历史连打正反馈。暂不需要激进修改样本时长权重。 |
| **D-03** | **P2 工程健壮性** | [`ComposerDataset._song_features`](../../src/maichart/composer/data.py#L403-L434)<br>[`load_bank`](../../src/maichart/composer/audio_bank.py#L81-L86) | `load_bank` 返回 `mmap_mode='r'` 的 `np.memmap`，`_feature_cache` 按虚拟字节数预算可同时持有数百个打开的 `memmap` 对象。 | 长时间训练稳定，但持有的文件映射句柄较多。 | 限制 `_feature_cache` 最大条目数（如 `128`）即可控制句柄数量。 | **最小改进**：可选工程优化，为 `_feature_cache` 增加条目上限。 |

---

### 4. [`src/maichart/composer/generate.py`](../../src/maichart/composer/generate.py)（推理循环、合法性过滤与导出）

| 编号 | 级别 | 位置 | 代码事实 | 产物观察 | 机制假设 | 验证方法、实测数据与最小改进 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **G-01** | **P0 已证实必现缺陷** | [`_active_heads`](../../src/maichart/composer/generate.py#L116-L135)<br>[`_sample_actions`](../../src/maichart/composer/generate.py#L76-L113) | `_active_heads` 仅过滤 `kind in {'hold', 'touch_hold'}` 且 `head_index >= 0` 的占用键，完全不检查前序未结束 `slide`/`wifi` 的滑行轨迹面板碰撞与并发总手数；同时项目已具备完备的 [`detect_chart_conflicts`](../../src/maichart/Virtual_Player/conflict_detector.py#L430)（`MiaCode` 虚拟玩家物理分析器），但生成循环未调用它。 | 实测 5 份最新生成谱面（除低温纯单键谱外）全部存在 `MiaCode` 硬冲突：<br>- [`2023bb6c2fec`](../../webui_workspace/generated/2023bb6c2fec/maidata.txt)：**28 处冲突（25 处硬冲突：`TapOnSlide: 15`, `MultiTouch: 10`, `Overlap: 1`；3 处软冲突：`SlideHeadTap: 2`）**；<br>- [`8b9012135d4e`](../../webui_workspace/generated/8b9012135d4e/maidata.txt)：10 处冲突（8 硬：`TapOnSlide: 5`, `MultiTouch: 3`）；<br>- [`1717d960548c`](../../webui_workspace/generated/1717d960548c/maidata.txt)：4 处冲突（1 硬：`MultiTouch: 1`）；<br>- [`e95941550d1c`](../../webui_workspace/generated/e95941550d1c/maidata.txt)：46 处冲突（31 硬：`TapOnSlide: 15`, `MultiTouch: 14`, `Overlap: 2`）。 | 生成端缺乏对在途 Slide 轨迹（`TapOnSlide`）与并发手数上限（`MultiTouch > 2`）的校验，是导致全部 25 处硬无理的直接原因。 | **实测结论（100% 证实）**：<br>无需发明“每条活动 Slide 无条件占死一只手”或 `<90ms 禁双押` 等粗糙经验规则！直接使用项目现成的 `MiaCode` 物理引擎 [`detect_chart_conflicts`](../../src/maichart/Virtual_Player/conflict_detector.py#L430) 在候选事件提交前对最近窗口（如最近 4 秒内的活动事件 + 当前候选动作）做增量校验：<br>- 若触发**硬冲突**（`TapOnSlide`、`MultiTouch`、`Overlap`），则拒绝该动作组合并重采样次优原子；<br>- 若触发**软冲突**（`SlideHeadTap`），仅记录诊断或做温和 logit 扣分，不硬性一刀切封杀。 |
| **G-02** | **P2 代码整洁度** | [`generate.py`](../../src/maichart/composer/generate.py#L18-L21) | 导入了 `detail_queries` 与 `phrase_features` 但未在函数体内使用。 | 无运行时错误，但表明乐句规划特征尚未接入 Composer 推理。 | 属于演进过程中的残留导入。 | **最小改进**：清理未使用导入，或在后续实验接入段落特征时使用。 |
| **G-03** | **P1 采样超参耦合** | [`generate.py`](../../src/maichart/composer/generate.py#L202-L214) | 位置指针（96 选 1）与动作槽位（18,454 选 1）共用单一 `--temperature`。 | 当 `temperature=0.85`（`2023bb6c2fec`）时，指针易在微弱 확률下踩入跨系分音触发 `{192}` 漂移；当 `temperature=0.45`（`5604cf3bf378`）时，虽然 `MiaCode` 硬冲突降为 `0`，但动作完全坍缩为 `0` 双押且全谱仅 `20` 种 unique 文本（几乎全是单键 Tap）。 | 位置决策需要较低温度/核采样以锁定主节奏网格，而动作决策在 18,454 维长尾词表上需要适度温度与重复惩罚以维持键位多样性。 | **验证与最小改进**：提供独立的 `pointer_temperature` 与 `action_temperature` 参数，通过 A/B 对比生成谱面的网格规整度与 `unique_texts` / Slide 比例。 |

---

### 5. [`src/maichart/composer/train.py`](../../src/maichart/composer/train.py) 与 [`checkpoint.py`](../../src/maichart/composer/checkpoint.py)（训练目标与检查点保存）

| 编号 | 级别 | 位置 | 代码事实 | 产物观察 | 机制假设 | 验证方法、实测数据与最小改进 |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **T-01** | **P1 留白校准与自回归暴露偏差（已修正表述）** | [`_loss`](../../src/maichart/composer/train.py#L85-L111) | **修正原绝对化表述**：`pointer_loss` 使用 96 候选交叉熵（`F.cross_entropy`），**非目标候选已作为 Softmax 分母竞争项被压低，并非“完全没有留白监督”**；且 `future_head` 已提供局部间隔与头密度的辅助监督（权重 `0.2`）。真正缺少的是：训练期 `history` 100% 来自真实谱面（Teacher Forcing），模型未经历“自己在安静段连续误采 `{16}` 后如何刹车”的自由生成反馈。 | 推理一旦在安静段踩出 1~2 个 `{16}`（尤其受 `attack` 插值抬高诱导），历史上下文即被 `{16}` 填满，容易顺着自回归惯性连续生成整串 `{16}`。 | 对 `history_features` 的时间步长施加适度随机掩码（Scheduled History Dropout），或对安静段被跳过的近场 `{16}` 候选增加温和的校准项，可能增强模型对音频真实瞬态的依赖、削弱盲目连打惯性。 | **验证与最小改进**：作为训练期 A/B 实验项，对比开启/关闭 `history_step_dropout` 后安静段（RMS < P50）的 `{16}` 生成占比。 |
| **T-02** | **P2 内存峰值防护** | [`_cpu_snapshot`](../../src/maichart/composer/checkpoint.py#L10-L42)<br>[`save`](../../src/maichart/composer/checkpoint.py#L49-L61) | `save()` 在 CPU 内存构造模型与优化器状态的完整副本（约 1 GiB）。 | 常规训练正常，仅在宿主机可用提交内存极低（`< 1.5 GiB`）时存在瞬时压力。 | 保存前主动检查并收缩 `_feature_cache` 可进一步提升极端低内存环境下的稳健性。 | **最小改进**：可选工程加固。 |

---

## 三、基于实测证据的最小改动方案与 A/B 评估指标

遵循“**先用确定性规则消除已证实的物理冲突与跨系分母漂移，不确定项通过软评分或 A/B 实验验证，绝不一刀切压平谱面风格**”的原则，建议按以下最小闭环推进：

### 1. 生成端最小确定性修复（仅针对已证实的 2 项 P0 缺陷，无需重训）
1. **接入 `MiaCode` 增量硬冲突拦截（解决 G-01）**：
   - 在 [`generate.py`](../../src/maichart/composer/generate.py) 采样当前拍点的动作组合时，调用 [`detect_chart_conflicts`](../../src/maichart/Virtual_Player/conflict_detector.py#L430) 检查新增动作是否与当前未结束的 Slide/Hold 构成 **硬冲突（`TapOnSlide`、`MultiTouch`、`Overlap`）**；
   - 若构成硬冲突，仅屏蔽引发冲突的具体动作原子并从当前分布中重采次优动作；**不把活动 Slide 一刀切硬算作占死一只手，也不对 `<90ms` 的高速步长一刀切禁双押**。
2. **接入跨系相位兼容掩码（解决 C-02 `{192}` 连续漂移）**：
   - **保留 `_FINE_OFFSETS` 中全部合法相对偏移（不删除 `{48}`、`{64}`）**，保证训练集 `99.50%` 的候选召回率不受损；
   - 在生成端 `cand_mask` 中，仅屏蔽会导致**绝对拍位分母落入跨系污染集合 `cand.denominator in {24, 48}`** 的候选，并在当前处于三连音网格（`previous.denominator % 3 == 0`）时引导候选在 1~2 拍内回归主拍网格。
3. **解耦 `pointer_temperature` 与 `action_temperature`（验证 G-03）**：
   - 允许分别调节位置温度与动作温度，避免低温下动作坍缩为纯底排单键。

### 2. 候选音频特征最小修正实验（针对实测证实的 C-01 `attack` 插值虚高）
- 在不改变其余 115 维特征的前提下，针对列 `92, 98, 99` 的 6ms 脉冲 `attack` 通道，测试“基于最近邻节点距离的指数衰减插值”对比现行“线性插值”，统计安静段（RMS < P25）`{16}` 步长占比是否显著回落。

### 3. 标准化 A/B 验证指标面板
每次改动均在同一组固定种子测试曲目上输出以下量化指标进行对照：
1. **物理合法性指标**：`MiaCode` 硬冲突总数（`TapOnSlide` + `MultiTouch` + `Overlap`，目标：`0`）与软冲突数（`SlideHeadTap`，仅作诊断参考）；
2. **节奏网格健康度**：绝对拍分母为 `24` 或 `48`（`{96}`/`{192}`）的事件占比（目标：接近语料基线 `< 0.05%`）；
3. **音乐情绪贴合度**：高能量段（RMS > P75）与安静段（RMS < P25）的平均音符密度之比（`NPS_loud / NPS_quiet`），以及安静段 `{16}` 步长占比；
4. **动作与键位丰富度**：全谱 `unique_texts` 数量、8 键分布熵（抑制 `3, 4, 5, 6` 底排垄断）、Slide/Hold/Chord 事件占比。
