# 舞萌DX Composer 架构 / 动作表征 / 训练目标链路 代码级审查

> **历史独立审查原稿，保留供追溯。** 本文对应2026-10-25的输入与检查点，后续已修改持续状态、跨拍关系、Touch采样、音乐结构和验证链路。原文关于“不可学”“不可能”的判断不作为本轮审核结论；文中`chart_composer_v2`也是原文称呼，实际主线模型族为`chart_composer_v1`。当前实现、已验证变化和剩余问题请先读[当前更新说明](COMPOSER_CURRENT_CHANGES_20261027_ZH.md)。

审查时间：2026-10-25 19:5x（Asia/Shanghai）
审查对象：`src/maichart/composer/*`（audio_grounded_v3 / chart_composer_v2 家族），并对标 `src/maichart/stream/*`、`src/maichart/generation/sustain_state.py`
审查方式：源码逐行核对 + 只读探针（`.scratch/review_probe_20261025.py`、`review_params_20261025.py`、`review_ckpt_20261025.py`、`review_incremental_cache*.py`、`review_chart_stats_20261025.py`）+ 现役 checkpoint 元数据 + 已生成谱面统计
**未修改任何训练进程、权重或采样配置。**

---

## 0. 结论摘要

| 维度 | 判定 | 一句话依据 |
|---|---|---|
| 时间/音频对齐（"音对得上"） | **可学会** | `candidate_logits` 在 96 候选中 top-1 = 65.1%（checkpoint metrics） |
| 音符类型/键位偏好（"像谱面"） | **可学会，但坍缩到底排** | 最新生成谱 4/5/3/6 占按键头 58.3%；`"4"`×83、`"5"`×63 |
| **手顺 / 左右手 / 人体工学** | **结构上不可学** | 全链路无任何一手/左右手/终端手位表征；`action_state` 被 `mean(0)` 压成单向量；滑键终点手位在 RELEASE≈16.7ms 后从状态里彻底消失 |
| 圆周对称 / 镜像 | **无** | 等变分量只有一个 `Embedding(8)`，且只在同拍内生效；composer 数据管线 0 增强（`stream/symmetry.py` 现成能力未接） |
| 绝赞 / EX 账本 | **不存在** | 4 张生成谱 break 数 = 1 / 2 / 1 / 16；无 ledger、无 condition 通道、无 guard、无后处理 |
| 训练目标是否在教"好谱" | **否** | 历史/符号分支只买到 0.40 nats（action）与 0.09 nats（pointer）；27% 的模型提案被硬冲突守卫拒绝后重采样 |

**总评：** 这是一个"音频 → 事件时刻 + 音符串"的条件生成器，在"节奏对齐 + 音符合法性"上已经可用；但**舞萌谱面质量的第一性指标（手顺）在本架构中既没有输入表征、也没有监督信号、也没有推理约束**。它不会、也不可能学会"手顺自然、引导清晰、无拧麻花"——目前这层质量完全由外部硬规则（MiaCode 冲突判定 + 拒绝重采样）在兜底，而兜底成功率只有约 73%。作为"资深谱师"判断：当前产物属于 **"合法但无手感的底排连打谱"**（level 14.5 出现 728 事件、149 种唯一文本、1 个 break、0 个 EX），离"高质量"缺的不是调参，而是**表征层**。

---

## 1. 关键新发现（本次审查新增，均可复现）

### 1.1 `SustainStateIndex.at()` 的 1-ULP 边界 bug（P0，静默丢状态）

`src/maichart/generation/sustain_state.py:137`

```python
if (onset > boundary if include_current else onset >= boundary - _EPS):
    continue
```

`include_current=True` 分支用的是**严格大于且无 epsilon**，而 else 分支带 `- _EPS`。编译后的谱面时间与 `seconds_at_beat()` 的浮点累加路径不同，会在最后 1 ULP 上不一致，于是"恰好在边界起拍的动作"被丢掉。

实测（`.scratch/review_incremental_cache3_20261025.py`，190BPM、1/4 拍间距、4 拍 Hold）：**47 个边界里 36 个（77%）丢失了刚刚起拍的那个动作**；单例复核显示 `marker.second = 0.4736842105263158` 而 `boundary = 0.47368421052631576`，差 5.55e-17。

后果链：`stream/context.py:13` → `data.py:514` → `data.py:515-517 mean(0)` → 模型输入。也就是**训练样本里最新、且恰恰是人体工学上最重要的那个动作（上一个事件刚起的 Hold/Slide）在 77% 的样本中不存在**；而生成端走的是 `IncrementalState`（`stream/grounded_inference.py:22-37`，逐格 `parse_markers(..., first=second)`），起拍时间与边界完全相等，因此**永远包含**。这是训练/推理表征不一致 + 系统性丢失最relevant证据的双重缺陷。修法：改成 `onset > boundary + _EPS`。

### 1.2 现役 checkpoint 仍在用有 bug 的 `legacy_digit_v1` 锚定

`checkpoints/Prism_composer_v10/composer_last.pt` 的 `model_config.history_anchor_policy = "legacy_digit_v1"`（epoch 3 / 11,124 步）。而 `action_factors.py:163-167` 的字符扫描实测量化如下（`.scratch/review_probe_20261025.py`）：

| atom | legacy_digit_v1 锚定 | parsed_heads_v2 锚定 |
|---|---|---|
| `15`（紧凑双押） | **`1`**（丢掉 5） | {1:0.5, 5:0.5} ✅ |
| `48` | **`4`**（丢掉 8） | {4:0.5, 8:0.5} ✅ |
| `D3` | **`4`**（fallback） | {3:1.0}（半格偏移，见 3.4） |
| `C`（Center Touch） | **`4`** | {}（无外圈角度）✅ |
| `A1` / `B5` / `E7` | `1` / `5` / `7`（数字扫描） | 同（A/B 可接受，D/E 见 3.4） |
| `1-5[8:1]` | `1`（只有头） | `1`（只有头） |

`grounded_model.py:102` 里 anchors 权重 **0.75**（`.75*anchors+.25*event_embedding+factors`），是 memory 里最大的一项。所以现役模型对**每一个紧凑双押都只记住最低位按键**，对 Center Touch 一律记成 4 号键——这不是"抽象"，是**错误的手位先验**。

### 1.3 生成报告的 `mean_audio_full_pointer_js` 只有 0.014–0.036

4 张最新生成谱（`webui_workspace/generated/{89ddafd2504f,63ed1a8ede4a,4417e4aa2672,baff795569b5}`，均 level 14.5）：

| dir | events | notes | **breaks** | ex | slides | holds | touches | unique_cells | pointer_js | fallback/checks |
|---|---|---|---|---|---|---|---|---|---|---|
| 89ddafd2504f | 728 | 858 | **1** | 0 | 118 | 70 | 91 | 148 | 0.0152 | 268/996 |
| 63ed1a8ede4a | 555 | 650 | **2** | 0 | 56 | 77 | 118 | 121 | 0.0163 | 77/633 |
| 4417e4aa2672 | 557 | 663 | **1** | 3 | 107 | 58 | 46 | 148 | 0.0142 | 179/736 |
| baff795569b5 | 818 | 1026 | **16** | 0 | 110 | 79 | 17 | 134 | 0.0358 | 183/1001 |

三个致命读数：
1. **`pointer_js ≈ 0.015`**：完整（含符号历史）指针分布与纯音频指针分布的 JS 散度只有 1.5%——**整个符号/历史通路对"下一个音在哪"几乎没有影响**。这条诊断是项目自己算的（`generate.py:513-515`、`635`），等于架构自己承认历史分支已死。
2. **`fallback` ≈ 27%**：模型的提案有 27% 被判为 `new_miacode_runtime_hard_conflict` 后丢弃重采样（`guard.py:97-99`、`generate.py:547-572`）。也就是说物理合法性**没有学进网络**，靠拒绝采样补。
3. **break = 1/2/1/16**：同一 level、不同种子之间 break 数波动 16 倍，说明 break 完全由采样噪声决定（详见 §4）。

---

## 2. 人体工学与左右手状态缺失（P0，架构级）

### 2.1 全链路盘点：模型到底能知道哪些"身体"信息

`AudioGroundedComposer.forward`（`grounded_model.py:60-162`）的全部输入里，与"手"有关的只有 4 条通道：

| # | 通道 | 维度 | 语义 | 致命缺陷 |
|---|---|---|---|---|
| A | `action_state` | **50**（单向量） | 活跃持续动作快照 | `mean(0)` 坍塌；见 2.2 |
| B | `head_timing` | **99** = 33×3 | 33 个键头顶的"多久没按" | **只记录 head，不记录滑键终点、不记录松手位置**；见 2.3 |
| C | `last_heads` → `action_relation` | 8 → 384 | 上一事件的**按键头**位掩码 | 头位，不是手位；且是 0 初始化的 `Linear(8,384)`（3,072 参数，占全模型 0.008%） |
| D | history atom embeddings | 128×384 | 历史原子 | 漂移 30%（`history_dropout`）+ 15% 全零（`symbolic_dropout`） |

**没有第 5 条。** 全仓库 `grep hand_tracking|kinematic|HAND_DIM` 在 `composer/` 下命中数为 **0**；`stream/hand_tracking.py` 的 24 维手部假设（`HAND_DIM=24`，含同手连打 run、换手 switched、移动速度代价、不确定性熵）与 `stream/symmetry.py:198-368` 的 65 维 `kinematic_rows`（含 `36:44` 松手衰减、`44:52` 滑键目的地、`52:60` 起点相位、`62:65` 实际跨步角）**一个都没被 composer 使用**。`grounded_model.py:1` 的 docstring 自称 "ordered listening, hierarchical time, shared actions"，确实——它只有听、时间、动作，**没有手**。

### 2.2 `active.mean(0)`：一个物理上不存在的状态向量

三处同一写法：
- `composer/data.py:514-517`（训练）
- `composer/generate.py:504-506`（自由生成）
- `composer/recovery.py:128-130`（rollout 恢复）

```python
action, mask, _ = prefix_action_state(...)   # [n_active, 50]
active = action[~mask]
table['action_states'][event_index] = active.mean(0)   # [50]
```

`SustainStateIndex.at()`（`sustain_state.py:158-169`）精心按 `(deadline, parse_order, line, col)` **排序**并逐行 `_encode_state`，编码了 4 类 one-hot + 33 键位 + 8 终端键位 + waiting + 剩余秒 + 到 trace 秒 + progress + ex（`_TYPE_OFFSETS`..`_EX_OFFSET`，`sustain_state.py:26-33`）。然后 `mean(0)` 把**顺序信息和绑定关系全部扔掉**（均值是 order-invariant，之前的排序白做）。

实测（`.scratch/review_probe_20261025.py`：一拍上同时存在 `1-5[4:1]` 与 `2h[2:1]`）：

```
state_rows = 2   types = ['slide','hold']
mean_type_vector = [0.5, 0.0, 0.5, 0.0]        # 半个 Hold + 半个 Slide，不是任何合法动作
per_row_pad_bits = [1.0, 1.0]  → mean_pad_bits = 1.0   # A2 上 0.5 只手、轨迹上 0.5 只手
per_row_terminal_lanes = [[5], [2]] → mean_terminal_lanes = [2, 5]（两者都是 1.0）
mean_state_nonzero = 8  vs  max_single_state_nonzero = 5   # 均值反而"更丰富"
```

数学后果（分通道）：

1. **类型通道**：n≥2 时得到 4 维概率单纯形上的内点，可以落在任意多个真实状态凸包的**外部**（如 0.33/0.33/0.33），模型必须用非线性把"混合"再拆回"n 个确定动作"，而 n 本身已经不可知。
2. **33 键位通道**：从"0/1 占用"退化为"占有率"。`0.5` 这个数在物理上无意义（手只有"在/不在"），却被写进状态；且**同一键位被两个动作共用时又会被加回 1.0**，于是"一只 Hold 占 A2"与"两只动作各占一半"在输入上不可区分。
3. **8 终端键位通道**：求和恒等于 n，但"哪条 Slide 落在哪个终点"这一 binding 丢失。手顺推理本质就是 binding（"哪只手被钉在 5 号键上、还要多久"），均值恰好摧毁它。
4. **剩余秒/进度**：`_unit_log_seconds` = `log1p(clamp(s,0,60))/log1p(60)`（`sustain_state.py:25,60-62`）是**凹函数**，由 Jensen 不等式 `mean(log(1+s_i)) ≤ log(1+mean(s_i))` → **均值系统性低估最长剩余时间**。数值例：两条动作 remaining = 0.2s 与 3.0s，均值特征 = (0.1823+1.3863)/2/4.1109 = **0.1909**，反解 expm1(0.1909×4.1109)−1 = **1.19s**；而实际是"一条 0.2s、一条 3.0s"。模型会把"还要钉 3 秒的手"读成"1.19 秒后就自由"，从而**提前占用被钉住的键位**——这正好对应生成端要靠 `_active_constraints` 硬挡的那类错误。progress 同理：10% 与 90% 的滑动平均成 50%。

**额外证据：** `MAX_STATES=32`（`generation/action_state_spec.py:5`）设计上允许 32 条并发状态，`STATE_DIM=50` 每行；而 `ComposerConfig.action_dim = 50`（`model.py:41`）并在 checkpoint 校验里被硬性锁定（`checkpoint.py:190-195`）。**即：表征层的容量（32×50）在契约层就被压成了 1×50**，且改这个字段会触发 checkpoint 不兼容——这是一个被写进契约的架构缺陷，不是遗漏。

### 2.3 "划完 `1-5[8:1]`，手停在 5 号键上"——五条通道逐条被否

这是本次审查最核心的一个案例。用户的手在 slide 结束时停在 **5**；我们逐条检查模型能否知道：

| 通道 | 代码位置 | 在 t=sliding 中 | 在 t=slide 结束后 |
|---|---|---|---|
| A `action_state`（终端键位 8 通道） | `sustain_state.py:287-302`（`_terminal_lanes`） | ✅ 有 5，但被 `mean(0)` 与其它动作混掉 | ❌ **整行被删**：`sustain_state.py:146,152-153` 在 `end+RELEASE ≤ boundary` 时 `continue`；`RELEASE = 3/180 = 16.7ms`（`generation/muri/model.py:9`）；`IncrementalState.append` 同样在 `end+RELEASE < second` 时物理删除（`stream/grounded_inference.py:29`） |
| B `head_timing`（99 维） | `events.py:74-85` | ❌ 记的是 `A1` 的按下时刻 | ❌ 只记 `A1`（见下） |
| C `last_heads`（8 维） | `action_factors.py:96-98` → `grounded_model.py:121-122` | ❌ `head_lane_mask` 只由 `isinstance(note.position,int)` 生成，slide 的 position = **头**（`notes.py:204` `position=int(first_segment["position"])`） | ❌ 仍然是 1 |
| D history 原子因子 | `action_factors.py:107-111` 的 `relative_endpoint` | ✅ 相对终点（`(arg-current)%8`）在因子里，模型**理论上**可自行合成 1+4=5 | ✅ 同上，但需跨 128 事件 attention 自行推断；且 30% 位置被 drop、15% 样本全零 |
| E 锚定嵌入 | `action_factors.py:195-198` | ❌ 只读 `isinstance(position,int)` = 头 | ❌ 1 |

**实测复核**（探针）：`1-5[4:1]/2h[2:1]` 之后 4 拍到下一个事件的边界，`at()` 返回 **0 行状态**、`post_terminal_lanes = []`。也就是**滑键结束 16.7ms 后，"手在 5 号键"这件事在整个模型输入里彻底不存在**——只剩 D 通道的隐式几何，而这需要模型自己在 192/384 维空间里做"头 + 相对偏移 = 终端"的向量加法，而它的注意力只有 3/6 层、且 15% 的训练样本把历史整体清零。

顺带一个语义不对称：**训练时 `action_state` 取的是"上一个事件时刻"的快照**（`stream/context.py:13` → `at(points[ev-1].beat, include_current=True)`），而**候选占用是另一套时间**：`_active_constraints`（`generate.py:276-297`）在**候选时刻**独立重算。于是模型看到的世界与硬掩码看到的世界不是同一个时刻，`action_state.remaining_seconds` 相对候选时间是"过期"的，模型必须自己做时间外推。

### 2.4 `head_timing` 只覆盖 33 个"键头"，且 `enrich_head_timing` 未接线

- `events.py:17-18`：`HEAD_PADS = A1..A8,B1..B8,D1..D8,E1..E8,C` = 33；`HEAD_TIMING_DIM = 3*33 = 99`。
- `events.py:49-54`：pads 由 `heads = {str(n.position)}` 派生 → **slide 只有头**；`head_timing_features`（`74-85`）对每个 pad 只给 `[seen, log1p(age_sec), log1p(age_beat)]`。
- 数据端 `data.py:400-423`：`seen_seconds/seen_beats` 只在 `meta.head_pad_indices` 上更新（`422-423`）→ 同样只有头。
- 生成端 `generate.py:582-583`：`for pad in event_metadata(text).head_pad_indices: last_head_seconds[pad]=...` → 只有头。
- `enrich_head_timing`（`events.py:88-101`，把 99 维映射成"短/拍/长"三档 recency）**全仓库只有 1 个测试引用**（`tests/test_composer_audio_grounded_v3.py:289-291`），生产路径 0 调用。

也就是说：**"多久没按过某键"有了，"手现在在哪 / 手什么时候离开某键 / 划键落点"完全没有。** 33 个键头里也没有"松手"事件——一个 Hold 结束时不会写任何通道，一个 Slide 结束时也不会。这正是用户直觉的核心：`head_timing` 回答的是"这个键多久没被敲"，而不是"这只手在哪"。

### 2.5 训练时"物理状态"和"谱面语义"被同一个 dropout 一起丢掉

`grounded_model.py:123-134`：

```python
if action_state is not None: symbolic = symbolic + self.action_projection(action_state)[:,None]
symbolic = symbolic + self.head_timing_projection(head_timing.to(hidden))
...
symbolic = symbolic * symbolic_keep[:,None]      # 15% 的样本：整体置零
```

`symbolic_keep` 以 `symbolic_dropout=0.15` 概率整批置零（`grounded_model.py:87-90`，`model.py:49`）。设计意图是"模态 dropout，强迫音频分支独立成立"（注释在 `model.py:206-208`），但它把**物理事实**（此刻手被钉在哪、多久没按）与**谱面语义**（作者写了什么历史）当成同一类可丢弃信息。再叠加 `history_dropout=0.3`（`grounded_model.py:106-109`，逐位置随机遮）与 `ledger_dropout=0.5`（`127-128`），实际效果是：**唯一那点手部证据有 15% 概率整条消失**，而推理时 `symbolic_keep≡1`（`grounded_model.py:89`）。这会主动压掉模型对状态通道的依赖——与 §1.3 观测到的 `pointer_js≈0.015` 完全一致。

### 2.6 生成端的"手"约束同样是坏的

- `_active_constraints`（`generate.py:276-297`）只把 `hold`/`touch_hold` 的源键位放进 `held`；**Slide 轨迹明确不算占用**（注释 `295-296`），并且 `active_hands = 0` 是**死变量**（`283` 初始化，`297` 原样返回，中间从未自增）。
- `base_playable = [len(held) < 2 ...]`（`452`）→ 语义是"同时最多 2 个 Hold"，既不是"手数"也不是"指头数"。**1 Hold + 1 Slide 在飞时**，该规则认为还可再补一个 2 键双押（4 个肢体），只能靠 MiaCode 硬冲突兜。
- `_sample_actions(..., max_heads=2)`（`generate.py:235,247`）是**硬上限 2 个 pad/事件**：既禁止了合法的 3 键（maimai 有），也没有对"1 手跨 4 个键"这类真正的拧麻花做任何建模。
- 结果：27% 的提案被拒（§1.3）。**模型没有学会物理，系统只是把它拒掉。**

### 2.7 小结（谱师视角）

舞萌的"手顺"可以被分解成：① 上一动作结束时双手的位置；② 每个动作由哪只手承担；③ 由 ①+② 决定的可达性；④ 由重复性/对称性决定的"好打"。当前架构：① 只剩 16.7ms 的 Slide 终端 + 被均值污染的 33 键位；② **完全不存在**（无 hand id、无左右、无交替统计）；③ 退化成一个 2-Hold 计数硬规则；④ 靠 `repetition.py`（**未接入 composer**：`grep RepetitionGuard composer/` = 0 命中）。**这不是"训练不够"，是"没有可学的变量"。**

---

## 3. 圆周几何对称性与键位偏置（P1）

### 3.1 零等变、零增强

`composer/` 下 `grep -E "symmetry|mirror|transform_text|rotate_180"` = **0 命中**。而 `stream/` 早就有全套：`symmetry.py:17-45`（4 种机台镜像 + 形状/边界映射表）、`49-74`（`transform_text`，`[...]` 内时值原样复制）、`97-109`（`transform_state`，50 维 sustain 状态的键位置换）、`179-185`（轨迹变换），并在 `stream/data.py:180-204` 用 `_mirror_ok`（`_counts` 一致 + `parse_markers` 无错）做**逐谱/逐格合法性校验**，`symmetry_augmentation ∈ {none, mirror_lr, mirror4}`（`stream/data.py:216,220`）。composer 侧本可原样复用，却完全没有接。

模型层面唯一的**旋转等变**分量是 `CompositionalActions.relative_factors = nn.Embedding(8, width)`（`action_factors.py:172-173`）——索引是圆周相对位移 `delta=(lanes[None,:]-lanes[:,None])%8`（`261`）。它在数学上确实等变，但：

- 零初始化（`173`）；
- 只在 `relative_logits` 里用，而 `relative_logits` 只在**同拍前缀**上生效（见 3.3）；
- 输出过 `F.linear(current, heads)`，`heads` 是**绝对**键位掩码（`259,264`）→ 等变性在最后一步被破坏。

其他所有键位表征都是**绝对类别**：`head_lane_mask` 被编码成 2^8 个类别里的**一个 ID**（`action_factors.py:96-97, 142-153`）；`head_timing` 逐 pad 绝对（`events.py:17`）；`action_state` 逐 pad 绝对（`sustain_state.py:284-289`）。**同一个 C 大三和弦，8 个旋转版本是 8 个互不相关的 embedding 行。**

### 3.2 后果是可测量的：底排坍缩 + Slide 终点偏置

最新一张（`89ddafd2504f`）实测（`review_chart_stats_20261025.py`）：

- 按键头直方图：`4:133, 5:120, 3:114, 2:95, 1:85, 6:80, 7:76, 8:64` → **3/4/5/6 合计 58.3%**（四张图 50.1%–66.7%）。
- 文本频次：`"4"×83, "5"×63, "3"×59, "2"×44, "6"×43`；759 格里只有 **149 种唯一文本**；`max_consecutive_same_text=4`。
- Slide **终点**分布：`{1:21, 2:4, 3:5, 4:27, 5:31, 6:20, 7:2, 8:9}` → 5/4/1 占 79/119，7/2/8 几乎消失。**终点偏置比起点更严重**，因为终点只能通过 `relative_endpoint` 因子间接表达（`action_factors.py:69-72`），却没有任何绝对位置的等变约束去纠正它。

对谱师的含义：这不是"风格"，是"模型只会用最省事的两只拇指位置"。官谱里 1↔8 的镜像对称段落、外圈→内圈的螺旋，在当前表征下都是 8 套独立参数，各自只能靠数据量硬记。

### 3.3 `relative_logits` 在 slot 0 恒为 0（起因 + 实测）

`grounded_model.py:182-193`：

```python
heads = self.action_factors.atom_head_lanes[action_targets].to(contexts)
relation = (heads.cumsum(-2) - heads) > 0            # 只看本事件内的前缀
...
full_logits = full_logits + self.action_factors.relative_logits(contexts, relation)
```

`relative_logits`（`action_factors.py:238-264`）里 `previous = last_head_mask`，`previous/previous.sum(-1).clamp_min(1)`（`258`）。slot 0 的 relation 全 False → 分母 `clamp_min(1)` → 全 0 → `current=einsum(...)=0` → `F.linear(0, heads)=0`（无 bias）。

实测：`relative_logits(...)[:,0].abs().max() = 0.0`（精确 0），`[:,1].abs().max() = 1.4356`。（`action_slot` 更彻底：`grounded_model.py:212-213` 用 `if selected_ids:` 直接跳过调用。）

**双重问题：**
1. slot 0 是事件的**第一个原子、也是最重要的决策**（`_sample_actions` 的 slot 0 决定头和主导类型），而它恰好拿不到任何相对几何项。
2. 即便 slot>0，`last_head_mask` 也只是**本事件内**已选原子的头位，**不含上一个事件**的键位/滑键终端。所以"从上一个事件的手位出发能不能打到这个键"这条最关键的跨拍关系，在任何地方都没有被显式喂进去——除了 `last_heads`（8 维头位）经 `action_relation`（3,072 参数、零初始化、0.008% 参数量）进 `action_context`（`grounded_model.py:151`）。

### 3.4 锚定策略的几何错误（含 parsed 策略）

`action_factors.py:184-199` 的 `parsed_heads_v2`：`elif position != 'C' and len(position)==2 and position[1] in '12345678': lanes.add(int(position[1]))`。

- **A/B 环**：`generation/muri/model.py:19` 中 A/B 与按键同角度（`-67.5 + (k-1)*45`，而 `symmetry.py:22` 的按键角 `(k-0.5)*45` 顺时针等价）→ 按数字映射是**可接受的粗角度**。
- **D/E 环**：`muri/model.py:19` 对 D/E 用 `-90` 偏移，`symmetry.py:24` 注释明确 "D/E sensors sit on the boundaries between buttons (D1 at the top)"，且 `_BOUNDARY`（`symmetry.py:31-36`）的镜像映射 `(9-k)%8+1` 与 `_BUTTON` 的 `9-k` **不同**（边界编号与按键编号差一格）。因此 `D3 → lane 3` 是**系统性半格（22.5°）错误**。
- **C**：parsed 策略已正确置空；legacy 仍回退 4。
- **Slide**：两个策略都只给头。

建议：把 D/E 的位次换成 `boundary(k)` 的等效按键角（或直接给 D/E 单独的 ring embedding，不与按键头共享），并把 Slide 的锚定改成 `0.5*head + 0.5*terminal`（终点可由 `relative_endpoint` 还原）。

---

## 4. 18,454 词表与多槽自回归解码（P1）

### 4.1 容量分配：37% 参数在两张 18,454 行的表里，手部逻辑占 0.03%

现役 checkpoint 的 state tensor 共 **38,325,395**（`review_ckpt_params_20261025.py`）：

| 模块 | 参数量 | 占比 |
|---|---|---|
| `audio_encoder`（6 层） | 10,646,784 | 27.8% |
| `history_encoder`（6 层） | 10,646,784 | 27.8% |
| `action_head`（18454×384） | 7,104,790 | 18.5% |
| `event_embedding`（18454×384） | 7,086,336 | 18.5% |
| `action_factors`（10 张因子表） | 1,311,788 | 3.4% |
| `history_attention` | 591,360 | 1.5% |
| `candidate/slot/audio_fusion`（各 148,608） | 445,824 | 1.2% |
| `pointer` | 148,225 | 0.39% |
| **`action_relation`（8→384，唯一的"上一事件键位"通路）** | **3,072** | **0.008%** |
| **`history_pointer`/`history_bin`/`bin_head`** | 384×3 | 0.003% |
| **`placement_head`（唯一预测 break 的辅助头）** | **400** | **0.001%** |
| `workload_head`+`planner`+`lens` | 88,682 | 0.23% |

结论：**92.6% 的参数用于"读 96 个候选"和"查 18,454 个原子"，真正决定"手顺/密度/break"的机制合计 < 0.3%。** 不是"模型太小"，而是**参数放错了地方**。

### 4.2 break / ex / firework 是怎么预测的：没有任何账本

全链路检查（`grep is_break|'b'|firework` in composer）：

1. **词表层**：`b`/`x`/`f` 是原子字符串的一部分，`atom_catalog`（`events.py:121-125`）直接收集原始字符串 → `1b`、`1x`、`1bf`、`1h[4:1]b` 各占一行（18,453 行里的大多数是 slide 的"几何×时值"组合）。
2. **因子层**：`break_ex` 是一个 **4 类**因子（`action_factors.py:103`：`int(is_break) | 2*int(ex)`），贡献一条 `Embedding(5,384)`（现役 5×384=1,920 参数）的**加性 logit**。它对所有 break 原子共享一个方向 → 只能表达"上下文沿某一维投影高时多打 break"，**无法表达位置、段落、配额**。
3. **辅助头**：`PlacementTargets` 有 `break_ex` 通道（`audio_lens.py:22,153`），266 个参数的单层 `Linear(80→5)`（`grounded_model.py:40`），预测"参考谱在 ±25ms 内有没有 break/EX"，然后 `.sigmoid().detach()` 加进 `audio_context`（`grounded_model.py:85`）。它是**逐时刻存在性**，不是数量。
4. **ledger**：`LEDGER_DIM=6`（`workload.py:19-23`）= 事件率 1s/4s + 头率 1s/4s + 距上次秒数 + 历史跨度。**没有 break 计数、没有"本段还剩几个 break"、没有"距上次 break 多久"。**
5. **plan**：`PLAN_CHANNELS=6`（`planning.py:10`）= event/head/hold/slide/touch/rest。**没有 break。**
6. **condition**：`CONDITION_KEYS = level + 7 radar + 8 style`（`generation/conditions.py:12-14`），radar 里没有 break/firework 项 → **用户无法要求"这段高潮多放 break"**。
7. **生成约束**：`_sample_actions`（`generate.py:235-273`）只做拨头数/占用/共键/词典序约束，**对 b/x/f 无任何约束或计数**；`guard.py` 只判 MiaCode 硬冲突；`process_chart_points`（`generate.py:600-615`）也没有 break 规则。

**唯一的"全局性"是那个 1,920 参数的共享偏置。** 所以 break 数纯粹由采样温度与词表频率决定——4 张图的 1/2/1/16 就是证据。作为对比：官谱 level 14 的 break 通常在 4–20 个且集中在副歌重音/段落收尾/最后一押；当前系统既无"重音证据"输入（`AcousticContext` 只有 26 维均值：6 动力学 + 8 频带 + 12 其他，`acoustics.py:103-106`；无段落/副歌/能量基线分割），也无"配额"控制，`candidates.py:408` 里预留的 4 列段落上下文还是 `np.zeros((n,4))`。

### 4.3 多槽解码的四个具体缺陷

1. **STOP 权重被压低**：`train.py:272` `slot_weights = torch.where(action_targets.ne(0), 1., .35)`。每个样本的最后一个槽一定是 STOP（`data.py:850-856` 裁到 `max(nonzero)+1`），而它只拿 0.35 的权重 → 模型被系统性训练成"倾向再加一个原子"。生成端靠 `mask_np[0]=slot>0`（STOP 不能占 slot 0）和 `max_heads=2` 才没炸。修法：STOP 权重 = 1.0，或直接改成"长度分类 + 原子"分解。
2. **slot 顺序是词典序**：`data.py:412-413` `atoms = tuple(sorted(atoms))`（`MUSIC_PROFILES` 含 grounded），生成端 `_sample_actions(..., canonical_order=music_v2)`（`generate.py:554`）用 `mask_np[1:lookup[selected[-1]]+1]=False`（`249-250`）复现同一顺序。于是 **slot 0 永远是该事件里字符串最小的原子**（`'1...'`/`'A...'` 优先）。这是纯粹的字符串偏置，被写进了 teacher forcing 的分词顺序里。
3. **截断会互相挤掉等价时值**：`_category_truncate_actions`（`generate.py:184-210`）按 tap/hold/slide/A..E 分组，组内 `top_p=.85, min_p=.08, max_k=16`。`[8:2]` 与 `[4:1]` 是**不同词表行、相同因子类别**（`action_factors.py:44-47` 的 `_duration_key` 会归一成 beats），二者在 16 个名额里互相分流，可能**同时被截掉**；而裸 `4` 因为频次极高稳定占位。这与文档里已记录的"Slide 概率被长尾稀释"是同一机制的推理侧版本。
4. **拒绝采样没有重要性修正**：`generate.py:547-572` 每次被 `guard.accepts` 拒绝就重新采样（最多 6 次），并把 `blocked_first` 永久加入禁用（`567`）。被接受的样本是从**截断+多次条件化**的分布里抽的，没有 `p_accept` 权重补偿 → 最终谱面分布 ≠ 模型分布，且系统性地偏向"容易通过守卫"的原子（即底排单键）。**这是 §3.2 观测到的 58% 底排占比的第二个成因。**

### 4.4 为什么 `workload / candidate_plan / placement / trailing` 全部 `.detach()`，以及代价

三条 detach：

```python
# grounded_model.py:83
audio_context = audio_context + curve_strength*(self.workload_projection(decision_workload[:,:count].detach())
                                               + self.plan_projection(candidate_plan.detach()))
# grounded_model.py:85
audio_context = audio_context + self.placement_projection(placement[:,:count].sigmoid().detach())
# grounded_model.py:131
difference = trailing[:,:count].detach() - ledger[...,:4].float().clamp_min(0).log1p()
```

设计意图在注释里（`model.py:245-246`）："让 chart-calibrated head 只拿自己的回归梯度，别被 token CE 变成一条不透明的隐藏通道"。**这个动机是对的**（避免辅助头退化成隐通道），但有三个明确代价：

1. **辅助头不可能"为了生成好看"而修正**。`workload` 的标签是**作者谱面自己的密度**（`workload.py:102-226` `ExactWorkloadQuery`），`placement` 的标签是**参考谱在 ±25ms 内有没有音**（`audio_lens.py:158-167`）。它们本质上是在做"用音频复现这张谱的统计量"，而这条通路对最终决策没有梯度回传 → 辅助头只在"模仿统计量"上优化，从不被要求"下一个决策要自洽"。生成时变成自指循环：模型预测自己将写出的密度，再据此偏置。
2. **信息被压成 5/6/6/5/4 维**。`workload_projection: Linear(5→384)`（1,920 参数）、`plan_projection: Linear(6→384)`（2,304）、`placement_projection: Linear(5→384)`（1,920）。**全部"谱面级"知识进入决策时只剩 20 个标量、rank-5/6 的子空间。** 一个 14.5 级谱的段落密度形状、break 分布、slide 密度，无法由 5 个标量承载。
3. **"极浅"是准确描述**：
   - `workload_head = Sequential(Linear(120→192), GELU, Linear(192→5))`（`model.py:119-120`，24,197 参数）——**逐帧 MLP，没有时间上下文**，输入还是 `.5/2/8s` 三尺度均值 + 全局均值（`acoustics.py:123-134`），也就是说它看不到 onset 序列，只看到平滑能量。
   - `placement_head = Linear(80→5)`（`grounded_model.py:40`，400 参数）——**单层线性**。
   - `planner = Linear(120→64) + 3×Conv1d(64,64,3,dil)`（`planning.py:62-75`，45,190 参数），在 **4 拍网格**上（`planning.py:21-26`）。
   - `lens = 3×深度可分离 Conv + Linear(31→1) 池化`（`audio_lens.py:73-127`，19,295 参数），池化是**无约束线性**（`103`），把 31 个时间偏移压成 1 个标量/通道。
   - `trailing_head = Sequential(Linear(68→192),GELU,Linear(192→4))`（`model.py:130-131`，13,056）。

   合计 88,682 + 400 参数 ≈ 0.23% 的容量，去承担"密度规划 / 段落占用 / 趋势"三件事。

---

## 5. 训练目标、采样器与 Recovery（P1）

### 5.1 `_loss` 逐项审计（`train.py:250-347`）

总目标（`339`）：

```
total = pointer + action + .05*eos + .2*future + .4*workload + .05*curve_slope
      + .15*(audio_pointer + audio_action) + .2*trailing + .15*placement + .2*plan
```

| 项 | 代码 | 问题 |
|---|---|---|
| `pointer` | `255-257` | CE over 96；按 `sample_weight*transition_weight*pointer_weight` 归一（`253-254`）→ 见 5.2 |
| `action` | `266-275` | 只在**目标候选**上 teacher-forcing（`candidate_indices`，`train.py:371`）→ 负候选从不见动作标签；STOP 只 0.35（`272`） |
| `audio_action` | `276-285` | 纯音频路 + 同 prefix（`grounded_model.py:183-187` 给两条路都加了 prefix）→ 与主导通路的差异只剩 0.40 nats（见 5.5） |
| `eos` | `286-291` | `pos_weight` 上限 64，权重 0.05 → 谱尾靠音频结束（`generate.py:595` 阈值 0.98） |
| `future` | `292-293` | 目标含未来 4 事件统计（`data.py:433-454`），只作辅助、不回灌（`model.py:222-226` 注释） |
| `workload/slope/trailing/placement/plan` | `294-338` | 全部"复现参考谱统计"，且对主决策 `.detach()`（§4.4） |

**没有的项**（这是关键）：没有任何
- 手顺/同手连打/换手/跨体/三手的损失或标签；
- break/EX 的计数或位置一致性损失；
- 密度/段落形状与条件的匹配损失（workload 只是"复现参考谱"，不是"匹配用户 radar"）；
- 重复/单调惩罚（`stream/repetition.py` 未接）；
- 镜像一致性损失（同一谱的两镜像版本应给等价似然）。

### 5.2 `SongLocalBatchSampler`：批内归一化把"图级质量权重"消掉了

`train.py:32-89`，默认 `sample_locality=4`、`batch_size=64`（现役 checkpoint args）。行为：每批固定 **4 首歌 × 每歌 16 个样本**（`70`：`take = max(1, batch_size//locality)`），歌内顺序 `torch.randperm`（`59`）。

`_loss` 的 `reduce`（`253-254`）：`(values*factors).sum()/factors.sum().clamp_min(1e-6)`。

数学后果：若整批权重同乘 c，损失不变 → **"这张谱质量 3.0、那张 1.0"的绝对标度被除掉了，只剩批内相对比例**。而一批只有 4 首歌 → 质量加权实际是"在这 4 首歌内做归一化"，跨批/跨歌的质量信息完全丢失。同时 16 个同歌样本共享一个 `sample_weight`，梯度噪声被单歌风格主导（`--sample-locality` 的 help 文本 `train.py:210-211` 自己也写了 "increases same-song gradient correlation"）。建议：改成按**全局/滑动窗口**的权重和归一（或直接 sum / global_count），并把 `sample_locality` 降到 2 或改用全局 shuffle + 缓存预热。

### 5.3 Recovery：只监督 pointer，`action_weight=0`——它教不会"恢复手位"

代码路径逐一核对：

- `train.py:688-707`：每 100 步，跑 `recovery_batch`，然后
  ```python
  recovery_output = _forward_batch(model, recovered, decode_actions=False)   # ← 动作头不投影
  per_sample = F.cross_entropy(recovery_output['candidate_logits'], recovered['candidate_target'])
  recovery_loss = (per_sample*importance).sum()/importance.sum()*args.recovery_weight
  ```
  **只有 pointer CE。**
- `recovery.py:165`：`'action_weight': torch.tensor(0.)`，`'pointer_weight': 1.`。
- `recovery.py:221-224`：把所有 `*_target`/`*_weight`（除 candidate_target/pointer_weight/sample_weight/transition_weight）**清零**，包括 `placement_target`、`planner_target`、`action_target`、`future_target`、`eos_target`、`trailing_target`。
- `recovery.py:128-130`：状态同样 `live.mean(0)`。
- `recovery.py:1-7` 的 docstring 自己承认："The reference continuation is not an optimal recovery oracle. Only its time is supervised."

**为什么这教不会恢复手位**（用户的直觉完全正确）：
1. rollout 走偏 2–4 个事件后，模型面对的是**自己写出来的手位分布**；而监督信号只有"下一个参考音在第几号候选（时间）"。
2. 要让参数更新去修正"手位"，梯度必须经过**动作头/状态编码器**并把"当前手位 → 好手位"的映射改对。`decode_actions=False` 直接切断了动作头的梯度；`action_weight=0` 即使投影也无效。
3. 更根本的是 **target 只存在于时间轴**：`target_index` 是"生成末尾之后第一个作者事件"（`recovery.py:193-196`），然后要求 `target.beat in candidates` 且 `candidate_mask[selected]==False`（`211-215`）。它连"应该打哪个键"都不提供，**没有任何变量承载"手应该去哪里"**。
4. 丢弃率不低：`batch_from_indices`（`258-279`）按 `invalid_rollout_length / changed_authored_prefix / no_reference_continuation / unknown_reference_span / illegal_generated_prefix / illegal_reference_continuation / unreachable_reference_time / masked_reference_time` 逐条丢弃，只统计不干预。

**建议（高收益）**：把 recovery 改成"在自生成状态上监督**完整下一决策**"（时间 + 原子 + STOP），并对"参考谱在该状态下的动作"做 CE；同时把 `decode_actions=True`、`action_weight=1`、`pointer_weight=1` 灌进去。再进一步：把 27% 的守卫拒绝当作**价值头（legality critic）**标签——这是现成的、免费的负样本，可以让模型在采样前学会"这个提案非法"，而不是靠事后拒绝。

### 5.4 没有验证集，也没有手顺指标

- `train.py` 全程无 hold-out（resume 靠 `dataset_fingerprint`，不是 split）。
- `diagnose.py` 是唯一的"评估"：
  - `interventions`（`54-123`）只测 **`candidate_logits` 的 JS** 与 **slot 0 的 atom JS**（`119`：`[...,0,:]`）；对 slot≥1、chord 组成、手位、break、slide 终点**完全无感**。
  - `phase_recovery`（`31-41`）是纯几何可达性，不是学到的恢复能力。
  - `omit_last_committed_event`（`151-199`）只报 `expected_next_second`，并显式写 `'supervised_recovery_loss': None`（`248`）。

也就是说 **"手顺变好还是变坏"在现有工具链下不可观测**——这解释了为什么这类缺陷能长期存在。

### 5.5 现役模型的实测数字（最有力的证据）

`checkpoints/Prism_composer_v10/composer_last.pt`，epoch 3，11,124 optimizer steps，`metrics`：

| 指标 | 全上下文 | 纯音频 | 差值 |
|---|---|---|---|
| pointer CE | 1.036 | 1.128 | **0.092** |
| action CE | 2.654 | 3.045 | **0.391** |

解读：
- 动作头在 18,454 类上的 CE = 2.654（均匀基线 ln 18454 = 9.82，折合 perplexity ≈ 14.2），而**把整条符号历史/状态链路加上只买到 0.39 nats**（≈ 1.5× odds）。
- 指针：96 候选 top-1 = 65.1%，而历史只买到 0.092 nats。
- 生成时 `mean_audio_full_pointer_js = 0.014–0.036`（§1.3）。

三处独立测量一致指向：**模型学到的几乎全是"音频→时刻/类型"的先验映射，谱面上下文（包括所有手部信息）对决策的边际贡献接近噪声。** 对谱师而言这意味着：**它不是在"接着写"，而是在"每拍重新猜"**——这正是"手顺"不可能出现的直接原因。

---

## 6. 可直接修的 bug 清单

| # | 位置 | 问题 | 影响 | 修法 |
|---|---|---|---|---|
| B1 | `generation/sustain_state.py:137` | `onset > boundary` 无 epsilon，1 ULP 丢边界动作（实测 36/47） | 训练态在 77% 的边界丢失刚起拍动作；与生成端不一致 | `onset > boundary + _EPS` |
| B2 | `composer/data.py:515-517`、`generate.py:505-506`、`recovery.py:129-130` | `active.mean(0)` | 状态绑定/顺序/计数全丢；Jensen 低估剩余时间 | 送 `[K,50]` 行 + mask，模型侧做 permutation-invariant 编码 |
| B3 | `action_factors.py:163-167` | 首数字符扫描 + fallback 4 | `15→1`、`48→4`、`C→4`、`D3→4`；现役模型正在用 | 用解析器真实头位；C 置空；Slide 加终端 |
| B4 | `action_factors.py:195-196` | D/E 边界传感器按数字当按键 | 半格（22.5°）系统偏置 | 用 `symmetry._BOUNDARY` 的等效角，或独立 ring embedding |
| B5 | `grounded_model.py:182-193` | `relation` 只含同拍前缀 | slot 0 相对 logits 恒 0；跨拍手位关系缺失 | 追加 `last_heads`（含终端键位）到 relation |
| B6 | `grounded_model.py:123-134` | 物理状态被 `symbolic_keep` 15% 整条丢弃 | 削弱唯一的身体证据 | 只对谱面语义做 dropout，状态/head_timing 不受 `symbolic_keep` 影响 |
| B7 | `generate.py:283,297` | `active_hands` 恒 0 死变量；Slide 不算占用；`len(held)<2` | 物理约束退化 | 接入真实手数/轨迹占用，或交由合法性 critic |
| B8 | `train.py:272` | STOP 槽权重 0.35 | 系统性偏向"再加一个原子" | 设为 1.0 或改长度头 |
| B9 | `data.py:412-413` + `generate.py:249-250,554` | 词典序 slot 顺序 | slot 0 恒为最小字符串原子 | 用"头数→键位→类型"的音乐序，或改为集合式解码 |
| B10 | `stream/grounded_inference.py:29` + `sustain_state.py:146,152-153` | RELEASE=16.7ms 后立即物理删除 | 松手位置零记忆 | 增加 release 通道（借鉴 `symmetry.py:36-44`，0.5s 衰减） |
| B11 | `stream/grounded_inference.py:29-35` + `sustain_state.py:201-218,229-241` | 缓存键 `(id(markers), len(markers))` | list 重建后 `id` 可能复用 → 命中陈旧 bucket（未被本次实测复现，但是明确的健全性隐患） | 改用单调递增 `version` 计数 |
| B12 | `grounded_model.py:35` | `super().__init__(replace(config, profile='music_context_v2'))` | 继承分支按"另一个 profile"建模块；未来加分支会静默漏建 | 显式传入 `head_mode` 标志 |

---

## 7. 修复路线（按 收益/代价 排序）

**P0（不修就不可能学到手顺）**
1. **状态集合编码器**（替 `mean(0)`）：`[B,K,50]+mask → 2 层小型 Transformer → 与 `audio_context` 做 cross-attn`。`action_state_spec.py` 已有 `MAX_STATES=32`，只需在 composer 侧恢复集合语义。**契约变更**：`action_dim`/`INPUT_CONTRACT` 需升版（`checkpoint.py:190-199`），旧权重只能 finetune。
2. **修 B1/B3/B4**（1–2 行级改动，无需改契约；`parsed_heads_v2` 已经在现网代码里，只需切过去）。
3. **加"松手/终端"通道**：从 `SustainStateIndex` 额外输出 `released[8]`（`exp(-age/0.5)`）与 `terminal[8]`、`origin[8]`，直接复用 `symmetry.kinematic_rows` 的 `36:65` 设计。这一步即使不做 1，也能让"划完 1-5 手在 5"第一次进入模型。

**P1（决定"像不像好谱"）**
4. **接手部假设**：`HandHistory.rows()` 已经在 `stream/data.py:710-722` 被验证过是因果、有界、可增量的，直接给 composer 喂 `HAND_DIM=24`（或截断到 12：同手 run、switched、两步跨距、滑动中数）。无需作者标签，纯 latent。
5. **镜像增强 + 等变化**：接 `symmetry.transform_text` + `_mirror_ok`（`stream/data.py:180-204`）做 `mirror_lr`/`rotate_180`/`mirror_ud` 增强，并用 `transform_state` 同步变换 `action_state`；把 `head_lane_mask` 的类别 embedding 换成 `sin/cos(kθ)` Fourier 特征 + 相对偏移，让镜像成为**参数共享**而不只是数据翻倍。预期直接压低 58% 的底排比例。
6. **break/EX 账本**：`LEDGER_DIM` 6→10（加 break_1s/4s、distance_since_break、budget_remaining）；`PLAN_CHANNELS`/`WORKLOAD_CHANNELS` 加 break 率；condition 加 `break_bias`；生成端对 `b` 做预算门控（软约束先上，硬约束后上）。
7. **词表结构化**：把 `action_head`（7.1M）降级/冻结，让 `CompositionalActions`（1.3M）成为主头（几何×时值分解），并把 `_duration_key` 的等价归约（`[8:2]≡[4:1]`）前移到词表层。这一条能同时解决"长尾稀释"与"截断互挤"。

**P2（训练/评估闭环）**
8. **Recovery 升级**：自生成状态上监督完整下一决策（时间+原子+STOP），`decode_actions=True`；把守卫拒绝当作 legality critic 的负样本。
9. **Loss 归一化**：改成全局权重和（或滑动窗口），不要批内归一；`sample_locality` 降到 2。
10. **诊断补齐**：`diagnose.py` 增加手顺指标（同键连打率、跨体率、3 手率、slide 终点直方图、break 数、unique-cell 比、`P(atom|prefix)` 与 `P(atom|audio)` 的 KL）；`interventions` 至少要测 slot 0 以外的槽位与 chord 组成。

---

## 8. 最终判断

**能学会的部分**：音量/攻击点→时刻（65.1% top-1 / 96）、音符类型分布、大体段落密度。这些已经在现役 checkpoint 上体现。

**学不会的部分（结构决定，非训练量问题）**：
- **手顺**——没有左右手变量、没有终端手位、没有可达性损失；唯一的状态被均值摧毁，且在 77% 的训练边界上压根不包含刚发生的动作。
- **对称美感**——8 个键位是 8 套独立参数，镜像不共享；实测底排 50–67%、slide 终点 79% 集中在前三个键。
- **绝赞编排**——没有账本、没有重音证据、没有配额；实测 break = 1/2/1/16。
- **段落变化**——谱面级信息进决策时只剩 20 个标量（rank-5/6 子空间），且被 `.detach()` 切断。

**量化差距**：18,454 类动作上的历史边际增益 0.39 nats、指针 0.09 nats、生成时 JS 0.015 —— 这三个数字合起来说明：**当前网络实质上是一个"音频条件的一阶马尔可夫猜拍器"，而不是一个"接着写谱的谱师"。** 要让它成为后者，必须先让它"知道手在哪"（§7 P0 的 1 与 3），再让它"在意手顺"（§7 P1 的 4 与 8）。在这两件事完成之前，任何温度/阈值/采样策略的调整都只是在给一个没有手感的模型换噪声。
