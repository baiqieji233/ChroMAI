# `chart_stream_v1` 近期架构演进与技术方案说明文档（供跨智能体审核）

> **生成时间**：2026-10-20
> **训练语义合同版本**：`chart_stream_training_v11_flight_kinematics`（定义于 [src/maichart/stream/data.py](../../src/maichart/stream/data.py#L36)）
> **文档目的**：完整阐述近期从 `v9` 经 `v10` 至 `v11` 的全套问题诊断、架构修改、已识别并主动拔除的“补丁型/污染型”设计，以及端到端（训练 $\rightarrow$ 检查点 $\rightarrow$ 推理搜索 $\rightarrow$ 物理与风格守卫）的技术实现细节，供其他智能体独立审核。

> **状态说明（2026-10-20）**：本文记录历史 v11 设计，后续实施见 `docs/handoffs/STREAM_V12_QUALITY_FIXES_20261020.md`、`docs/handoffs/STREAM_BLANK_GAPS_HANDOFF_20261020.md`。当前合同为 `chart_stream_training_v13_bounded_cycle_exit`；plan 相位通道（1:3）已改为对称不变的相对手部移动相位，旧 `plan_dim=8` 检查点由 loader 显式扩展到 10 维。本文中关于 `stream/grammar.py` / `CharNoteAutomaton` 的描述不代表当前实际代码路径。

---

## 一、 核心背景与设计演进主线

`chart_stream_v1` 采用全因果单流自回归架构，将舞萌 DX（Simai）谱面编码为扁平事件流：
$$\texttt{<bos>} \rightarrow \underbrace{\texttt{<at>} \rightarrow \texttt{P:*} \rightarrow \texttt{<note>}}_{\text{位置阶段 (Position Phase)}} \rightarrow \underbrace{\texttt{N:*} \dots \texttt{N:*} \rightarrow \texttt{<end>}}_{\text{配置阶段 (Configuration Phase)}} \rightarrow \dots \rightarrow \texttt{<eos>}$$

在近期迭代中，我们针对单流架构在长程生成与复杂手法上的核心短板，完成了两阶段根因治理与一轮彻底的“反补丁/去污染”架构净化：

1. **第一阶段（`v10`）：根治节奏复读循环与局部声学视野坍缩**
   - 解决单流解码器在预测 `P:*` 拍位时“看不见未来一整拍音频”以及通过 `CausalClock` 的 $\Delta\text{beat}$ 通道直接抄袭上一小节节奏型的问题。
2. **第二阶段（`v11`）：建立持续动作（Hold/Slide）在途运动学感知与一笔画物理闭环**
   - 解决模型与物理守卫只知道“Slide 何时发起”、不知道“Slide 正在划向何处、手何时到达终点键”的失忆问题，赋予模型对在途轨迹（In-Flight Trajectories）、一笔画交接（One-Stroke Hand-off）与同起点多滑（`1-4*7[8:1]`）的原生感知与约束。
3. **第三阶段（`v11` 架构净化）：彻底拔除具有多模态污染与校准失真风险的冗余机制**
   - 经严格审视，主动**全量移除**了 `blueprint`（`<note>` 处 16 维具体键位蓝图预测与门控残差回注）和 `structure_weights`（`<note>` 内部修饰符 token 损失加权），杜绝自回归采样下的“奇美拉拼接（Chimera Pollution）”与单步条件概率校准失真。

---

## 二、 核心技术方案详解

### 1. 65 维因果在途运动学特征（`KINEMATIC_DIM = 65`）

- **代码定位**：
  - 特征计算核心：[`kinematic_rows`](../../src/maichart/stream/symmetry.py#L140-L249)、[`_effective_lanes`](../../src/maichart/stream/symmetry.py#L99-L115)
  - 模型接入层：[`GroundedContext.input`](../../src/maichart/stream/grounded_model.py#L228-L250)、[`GroundedContext.fuse`](../../src/maichart/stream/grounded_model.py#L252-L265)
  - 推理状态机：[`GroundedInference._kinematic`](../../src/maichart/stream/grounded_inference.py#L103-L115)
  - 训练数据集组装：[`StreamDataset._grounded_item`](../../src/maichart/stream/data.py#L469-L570)
- **问题根因**：
  - 旧版运动学特征仅记录历史事件在触发时刻 $t_{\text{start}}$ 按下的静态键位 `lanes`（36 维）或已结束长条的释放近因（44 维）。
  - 当谱面中存在跨拍 Slide 时，后续事件处于 $t \in (t_{\text{start}}, t_{\text{end}})$ 期间，模型输入端完全不知道“有一条 Slide 正在途、起点手是否仍在等待、终点将落向哪个键位、还有多久到达”。同时，对于折线滑条或连滑（如 `1-5-2`），用原始数字集合 `{1, 2, 5}` 计算手位中心会严重偏离真实落手点 `{2}`。
- **65 维通道严格定义**（对于第 $i$ 个事件时刻 $t_i$，仅统计满足因果律的历史已提交事件 $j < i$）：

| 维度区间 | 维数 | 物理含义与计算方式 |
| :--- | :---: | :--- |
| `0:8` | 8 | 上一事件 $i-1$ 的活跃键位 one-hot 掩码 |
| `8:16` | 8 | 上上事件 $i-2$ 的活跃键位 one-hot 掩码 |
| `16:24` | 8 | 8 个键位自上次按下以来的指数时间衰减近因 $\exp(-\Delta t / 1.5)$ |
| `24:32` | 8 | 几何步进与动量特征：上一事件角度 $(\sin\theta, \cos\theta)$、步进角差 $(\sin\Delta\theta, \cos\Delta\theta, |\Delta\theta|/\pi)$、时间间隔、活跃度、旋转动量 |
| `32:36` | 4 | 当前小节首事件（Bar-start Anchor）的手位参考角与相对偏角 |
| `36:44` | 8 | **已完成持续动作释放近因**：满足 $t_{\text{end}} \le t_i$ 的历史 Hold/Slide 在终点键 `end_lane` 上的指数衰减 $\exp(-(t_i - t_{\text{end}}) / 1.5)$ |
| `44:52` | 8 | **在途 Slide 目的地热力（In-Flight Destination Urgency）**：对所有满足 $t_{\text{start}} \le t_i < t_{\text{end}}$ 且 `kind == 'slide'` 的在途轨迹，在 `end_lane` 上标记紧迫度 $1.0 - 0.5 \cdot \text{clamp}(\frac{t_{\text{end}} - t_i}{\max(t_{\text{end}} - t_{\text{start}}, 10^{-3})}, 0, 1)$（刚发起时 `0.5`，临近到达终点时线性升至 `1.0`） |
| `52:60` | 8 | **在途 Slide 起点状态（In-Flight Origin Occupancy）**：对在途 Slide，在 `start_lane` 上标记状态（若处于星星头等待期 $t_i < t_{\text{trace}}$ 则为 `1.0`；若已进入划动期 $t_{\text{trace}} \le t_i < t_{\text{end}}$ 则为 `0.5`） |
| `60:62` | 2 | **在途全局摘要**：`[min(在途条数, 4)/4.0, clamp(最近到达剩余秒数 / 2.0, 0, 1)]`（无在途 Slide 时剩余秒数通道置 `0.0`） |
| `62:65` | 3 | **有效落手步进向量（Effective Hand Step）**：先通过 [`_effective_lanes`](../../src/maichart/stream/symmetry.py#L99-L115) 将每个事件的键位集合修正为 `(set(lanes) - slide_heads) | slide_ends`（例如 `1-5-2` 的有效落手点为 `{2}` 而非 `{1,2,5}`），再计算有效手位角度的步进 $(\sin\Delta\theta_{\text{eff}}, \cos\Delta\theta_{\text{eff}}, |\Delta\theta_{\text{eff}}|/\pi)$ |

- **模型注入路径与因果对齐**：
  1. **双路径注入**：在 [`GroundedContext.input`](../../src/maichart/stream/grounded_model.py#L238-L246) 中，`action = self.kinematic_projection(kinematics)` 被直接加到进入 `self.decoder` 之前的 token 嵌入上，使所有自注意力层均能根据在途轨迹规划下一拍；同时在 [`GroundedContext.fuse`](../../src/maichart/stream/grounded_model.py#L252-L265) 中再次与解码器输出融合。
  2. **掩码防泄露**：在 [`StreamDataset._grounded_item`](../../src/maichart/stream/data.py#L522-L532) 中，凡被 `history_note_dropout` 或 `prefix_note_dropout` 遮罩的事件，其 `lanes[event]` 与 `trajectories[event]` 均同步置空后再计算 `kinematic_rows`，严防模型通过运动学通道偷看被遮罩的音符。
  3. **镜像增广缓存加速**：在 [`StreamDataset`](../../src/maichart/stream/data.py#L329-L380) 中使用 `(chart_index, transform)` 字典缓存整谱镜像变换后的 `_oriented_lanes` 与 `_oriented_trajectories`，避免训练时每个切片重复遍历整谱。
  4. **旧权重零填充平滑升级**：[`checkpoint.py`](../../src/maichart/stream/checkpoint.py#L70-L81) 支持从 36 维或 44 维旧检查点加载，自动将新增的 `kinematic_projection.weight` 列初始化为 `0`，确保微调起点与旧模型输出完全等价。

---

### 2. 物理可行性守卫（`PhysicalFeasibilityGuard`）与一笔画闭环

- **代码定位**：[src/maichart/form/physical_guard.py](../../src/maichart/form/physical_guard.py)
- **核心设计**：
  - 将双手状态从瞬时点重构为持续动作轨迹 [`_SustainTrack`](../../src/maichart/form/physical_guard.py#L51-L97)：
    - **Hold**：在 $[t_{\text{start}}, t_{\text{end}})$ 期间将手锁定在 `start_lane` 上，不可移动去按其他键。
    - **Slide**：在 $[t_{\text{start}}, t_{\text{trace}})$ 星星头等待期手位于 `start_lane`；在 $[t_{\text{trace}}, t_{\text{end}})$ 划动期手沿参考路径 `path_lanes` 连续插值移动；在 $t \ge t_{\text{end}}$ 结束后，手的位置更新为 **`end_lane`**（而非留在起点）。
  - **三项关键物理规则**：
    1. **一笔画交接豁免（One-Stroke Hand-off）**：当一只手刚完成或即将完成 Slide（$t_{\text{end}} - \epsilon \le t$），且当前事件需要在 `end_lane` 上触发新动作时，`end_lane in group` 判定为顺畅的一笔画交接，不计远距离跳手惩罚。
    2. **启动拍同步伴奏 Tap 豁免**：在 Slide 轨迹开始划动的瞬时（$|t - t_{\text{trace}}| \le \frac{1}{90}\text{s}$），允许该手顺带触发同侧伴奏 Tap，但**严禁同时绑定新的 Hold 或另一条 Slide**（`rebinding = True` 时不豁免，防止单手同时跟两条持续轨迹）。
    3. **中途弃滑惩罚（`ABANDON_EXCESS`）**：若在 Slide 尚未到达终点（$t < t_{\text{end}} - \epsilon$）且不满足启动拍豁免时，强行抽调正在划动的手去按其他键或绑定新轨迹，按剩余划动比例施加惩罚：
       $$\text{cost}_{\text{abandon}} = \text{ABANDON\_EXCESS} \times \left(0.5 + 0.5 \cdot \frac{t_{\text{end}} - t}{\max(t_{\text{end}} - t_{\text{start}}, 0.1)}\right)$$

---

### 3. 同起点多条 Slide 语法边界（当前实现）

- **代码定位**：[src/maichart/generation/event_cli.py](../../src/maichart/generation/event_cli.py#L261-L330)
- **核心修改**：
  - 舞萌 Simai 语法中，`*` 用于连接同一星星头发起的多条 Slide（如 `1-4[8:1]*-6[8:1]` 或简写 `1-4*7[8:1]`）。
  - 当前没有独立的 `CharNoteAutomaton` 文件。chart_stream 在共享 `_NoteGrammar` 上启用 `max_star_slides=2`，过滤超过上限的 atom 并重建前缀索引；普通 `/` 多押不受影响。旧 event/song_form 路径默认保持 legacy atom catalog，CLI 可显式传入正数 `--max-star-slides` 做诊断。

---

### 4. 对称不变宏观意图（`plan_head`）与 `TypeGuidance` 协同

- **代码定位**：
  - 标签构造：[`plan_labels`](../../src/maichart/stream/symmetry.py#L252-L280)（`PLAN_DIM = 10`）
  - 模型预测与注入：[`GroundedContext.plan_head`](../../src/maichart/stream/grounded_model.py#L223)、[`ChartStreamTransformer.decode`](../../src/maichart/stream/model.py#L140-L195)
  - 生成期门控：[`TypeGuidance.update_plan_intent`](../../src/maichart/form/type_guidance.py#L140-L165)
- **核心设计**：
  - 在每个事件边界（`<bos>` 与 `<end>`），`plan_head` 预测下一事件的 10 维宏观属性：
    $$\text{plan} = [\Delta b_{\text{norm}}, \sin|\Delta\theta_{\text{hand}}|, \cos|\Delta\theta_{\text{hand}}|, \mathbb{I}_{\text{chord}}, \mathbb{I}_{\text{slide}}, \mathbb{I}_{\text{hold}}, \mathbb{I}_{\text{touch}}, \text{density}, \mathbb{I}_{\text{break}}, \mathbb{I}_{\text{ex}}]$$
  - **为什么 `plan_head` 是安全的**：相对手部角位移取绝对值，因此第 2、3 维在当前四种镜像增广下保持不变；其余结构维度也严格不变，不会在对称增广下发生正负样本自我抵消。
  - **单向保守门控**：在生成期，[`TypeGuidance`](../../src/maichart/form/type_guidance.py#L140-L165) 读取模型自身的 `plan_head` 概率 $p_{\text{intent}}$，仅用于**衰减**不合时宜的全局比例补偿（乘子 $\text{gate} = 1 - s + s \cdot (0.25 + 0.75 p_{\text{intent}}) \in [0.25, 1.0]$），绝不在模型认为不该放 Slide/Break 的拍位强行塞入复杂音符。

---

### 5. 节奏去回声与宽视野局部声学（`v10` 基底保留）

- **代码定位**：
  - 宽视野多尺度局部声学采样：[`GroundedContext.local`](../../src/maichart/stream/grounded_model.py#L97-L120)
  - 训练期时钟间隔去回声：[`masked_clock_inputs`](../../src/maichart/stream/grounding.py#L210-L245)
  - 节奏循环退出监督：[`cycle_exit_targets`](../../src/maichart/stream/continuation.py#L30-L62)
  - 生成期节奏循环惩罚与拍位提案排序：[`rhythm_cycle_cost`](../../src/maichart/stream/timing.py#L39-L83)、[`compose_stream`](../../src/maichart/stream/decoding.py#L370-L385)、[`search_position`](../../src/maichart/form/rational_position_search.py#L58-L60)
- **要点**：
  1. `GroundedContext.local` 的 32 个局部采样偏移覆盖完整的 `[-0.25 * local_audio_seconds, +local_audio_seconds]`（默认 `[-0.5s, +2.0s]`），中心 $\pm 4$ 步保持单帧精细分辨率以捕捉瞬态鼓点，外围指数/线性步进延伸至未来 `+2.0s`，解决高密网格下 `[-16..+15]` 仅能看清未来 `0.23s` 的视野坍缩问题。
  2. 训练期 [`masked_clock_inputs`](../../src/maichart/stream/grounding.py#L210-L245) 对被遮罩事件及随机比例事件清零 `CausalClock` 的 `delta / delta_sec`（`15:17` 维），迫使模型依赖声学重音而非复制上一小节同相位的 $\Delta\text{beat}$。
  3. [`rhythm_cycle_cost`](../../src/maichart/stream/timing.py#L39-L83) 在生成期仅对连续重复超过 2 轮且跨度超过 2 小节的**非均匀复合节奏型**施加有界软惩罚，对均匀 4/8/16 分音流与正常两小节呼应保持 0 惩罚。

---

## 三、 主动废弃与排除的方案（防“补丁叠补丁”重点说明）

审核智能体请重点关注以下三项**已被我们识别出隐患并彻底清除/排除**的设计，切勿在后续迭代中重新引入：

### 1. 已彻底拔除：`blueprint`（`<note>` 处 16 维键位蓝图预测与门控残差）
- **原设计初衷**：在 `<note>` 位置预测当前事件的 16 维具体键位（`head_lanes[0:8]` + `end_lanes[8:16]`），并通过 `blueprint_projection(sigmoid(logits))` 加回 `<note>` 之后所有 `N:*` 字符的隐藏状态。
- **为什么在 `chart_stream_v1` 下是严重污染源（已全量删除）**：
  1. **多模态对称均值导致“奇美拉拼接（Chimera Pollution）”**：在特定音乐拍点与对称前文下，左手滑条 `1-5[8:1]` 与右手镜像滑条 `8-4[8:1]` 往往具有近似相等的合理性。由于 `<note>` 时尚未采样第一个键位数字，`blueprint_head` 用逐键位独立 BCE 训练出的最优解是两者的均值（起点 `1, 8` 各 `0.5`，终点 `4, 5` 各 `0.5`）。当自回归第 1 步采样选择了 `N:8` 后，注入后续步骤的静态 `blueprint` 向量仍然在终点 `5` 上保留高激活，直接把 `8-4` 污染成跨场斜穿的奇美拉滑条 `8-5`！
  2. **`BLUEPRINT_POS_WEIGHT` 破坏条件概率校准**：为对抗 8 键位稀疏性而引入的正样本加权（`4.0`），会将未归一化激活放大至 $0.75\sim 0.85$，在 `<note>` 内部持续压制 `<end>` token，诱发冗余多押与畸形尾缀。
  3. **架构冗余**：当前事件前文（含 65 维在途运动学）已完整存在于 Transformer K/V 缓存中，自回归逐步生成（采样 `N:8` $\rightarrow$ 选路径 `N:-` $\rightarrow$ 选终点 `N:4`）天然能够在已知首键位的条件下精确消歧，根本不需要在 `<note>` 处做跨模态平均的密集键位预测。

### 2. 已彻底拔除：`structure_weights` / `structure_focus`（`<note>` 内修饰符损失加权）
- **原设计初衷**：在计算交叉熵损失时，对 `<note>` 内部的修饰符 token（`/`、`h`、`-`、`*`、`(`、`[` 等）乘以 `1.0 + structure_focus` 的额外权重。
- **为什么有害（已全量删除）**：
  1. [`objectives.py`](../../src/maichart/stream/objectives.py#L20-L50) 中已经有严谨的 [`event_balanced_loss`](../../src/maichart/stream/objectives.py#L20-L50)，将每个事件的拍位损失与配置损失按事件均衡归一化。
  2. 在单个事件内部人为放大 `/`、`h`、`-` 的 token 权重，而保持 `<end>` 权重为 `1.0`，本质上是篡改了单步条件概率 $P(y_t \mid y_{<t})$ 的最大似然估计，导致生成期在刚输出键位数字 `N:1` 后，`<end>`（单押 Tap）的 logit 被系统性压低，破坏了模型原生的物量与题型校准。

### 3. 经评估明确不修改：Touch 音符（`A..E`）不纳入物理手位跟踪与局部多押计数
- **技术理由**：舞萌 DX 中 Touch 判定区域宽大且高度重叠，玩家常采用手掌擦滑、顺手拍击等非标准指法，官谱与自制谱也广泛将 Touch 作为视觉烟花或高潮段物量调味。若强行将 Touch 纳入 `PhysicalFeasibilityGuard` 的双手点位追踪或 `local_chords` 约束，会产生大量虚假的“跨屏瞬移惩罚”，扭曲外圈 1–8 实体按键的真实物理轨迹建模。

---

## 四、 跨智能体审核清单（Review Checklist）

建议审核智能体按以下清单核验代码实现：

1. **训练—推理运动学逐位一致性核验**：
   - 检查 [`StreamDataset._grounded_item`](../../src/maichart/stream/data.py#L510-L550) 与 [`GroundedInference._kinematic`](../../src/maichart/stream/grounded_inference.py#L103-L115) 是否调用完全相同的 [`kinematic_rows`](../../src/maichart/stream/symmetry.py#L140-L249) 与 [`NoteEvent.from_simai(...).sustain_trajectories(...)`](../../src/maichart/stream/reference.py#L420-L448)。
   - 检查 [`ChartStreamTransformer.decode`](../../src/maichart/stream/model.py#L140-L195)（训练并行前向）与 [`ChartStreamTransformer.decode_step`](../../src/maichart/stream/model.py#L197-L237)（推理单步缓存前向）是否输出逐位一致的 logits。
2. **因果性与掩码无泄露核验**：
   - 检查 [`kinematic_rows`](../../src/maichart/stream/symmetry.py#L168-L215) 处理第 `index` 个事件时，是否仅遍历 `0 .. index - 1` 的已提交事件，且以 `now = event_seconds[index]` 判断历史轨迹是处于已完成（`end_sec <= now`）还是在途（`start_sec <= now < end_sec`）。
   - 检查 [`StreamDataset._grounded_item`](../../src/maichart/stream/data.py#L522-L532) 是否对所有被 `masked_note_inputs` 遮蔽的事件清空了 `trajectories[event]`。
3. **`blueprint` 与 `structure_weights` 清理彻底性核验**：
   - 确认 `src/maichart/stream/` 与 `src/maichart/webui/` 中已无任何残留的 `blueprint` 预测头或 `structure_weights` 加权逻辑（仅 [`checkpoint.py`](../../src/maichart/stream/checkpoint.py#L57-L68) 的 `obsolete` 兼容忽略列表保留旧键名以防加载旧存档报错）。
4. **物理守卫边界条件核验**：
   - 检查 [`PhysicalFeasibilityGuard`](../../src/maichart/form/physical_guard.py#L160-L249) 对一笔画终点交接、启动拍伴奏 Tap（`not rebinding`）以及中途弃滑（`ABANDON_EXCESS`）的成本计算是否自洽。
5. **一键运行自动化测试验证**：
   ```powershell
   pytest tests/test_stream_grounding.py tests/test_stream_sustain_intent.py tests/test_stream_continuation.py tests/test_stream_geometry_and_sections.py tests/test_stream_configuration_search.py tests/test_stream_generation_integrity.py tests/test_stream_quality_constraints.py tests/test_stream_training_contract.py tests/test_form_physical_guard.py tests/test_form_audit_edge_cases.py -q
   node --test tests/test_stream_generation_ui.js tests/test_task_polling.js
   ```
