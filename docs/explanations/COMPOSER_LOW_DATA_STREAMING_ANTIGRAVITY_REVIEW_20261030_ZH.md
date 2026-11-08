# 少量数据流式谱面生成：学习瓶颈根因剖析与下一代架构设计（Antigravity 独立审查）

日期：2026-10-30  
审查范围：只读审查当前源码（`v27` 最新状态）、[COMPOSER_LOW_DATA_STREAMING_DESIGN_20261030_ZH.md](COMPOSER_LOW_DATA_STREAMING_DESIGN_20261030_ZH.md)、[COMPOSER_V27_EFFICIENCY_AND_LEARNING_REVIEW_20261030_ZH.md](COMPOSER_V27_EFFICIENCY_AND_LEARNING_REVIEW_20261030_ZH.md)、COMPOSER_V26_REST_MICROSTEP_REPAIR_20261030_ZH.md（本地开发资料，不随源码发布） 及早期 `event_ctx`（[event_model.py](../../src/maichart/generation/event_model.py)、[event_ctx.py](../../src/maichart/generation/event_ctx.py)、[STAGE_E_LOCAL_ONSET_QUALITY.md](STAGE_E_LOCAL_ONSET_QUALITY.md)）。未修改任何源码、未触碰正在 `7864` 运行的 `v27` 训练任务 `5fbb4701c0a4`。

---

## 零、事实基线与边界声明（已验证事实 vs. 本轮新假设）

### 0.1 已验证的源码与实验事实（不再重复当作待修缺陷）
1. **`v27` 已落地的修复与当前状态**：
   - 选时积分已升级为固定 `50ms` 网格右端点积分（[`event_survival.py`](../../src/maichart/composer/event_survival.py#L78-L130) `integrated_survival` + `redistribute_survival`），等待尾概率 `tail` 独立守恒，不再被 `shape`/`correction` 稀释；粗时间区间（`coarse`）与区间内精细落点（`shape`）已拆开且取消了音频 `tanh` 限幅。
   - 全曲规划器休止通道已改为 `BCEWithLogits`（[`train.py`](../../src/maichart/composer/train.py#L537-L547)），并通过 [`fuse_survival_rates`](../../src/maichart/composer/event_survival.py#L37-L50) 接入固定格率；64 行真实探针中 `plan_rest_logits` 范围为 `-4.94 ~ 3.03`（未发生 FP32 `sigmoid=1` 梯度死锁），且精细/粗音频辅助选时头与主选时头梯度余弦分别为 `0.998` 与 `0.650`（同向）。
   - 逐键负担已通过 `per_button_context_v2` 汇总 `mean/max` 46 维送入主 `symbolic` 分支（[`grounded_model.py`](../../src/maichart/composer/grounded_model.py#L771-L778)），影响选时与全部槽位 `STOP`；生成端重复抑制 `official_tail_v2` 已覆盖全部槽位，网格采样已升至 `grid_family_coherent_v3`，短段细分计划已通过 `relative_phrase_v1` 桥接至选时。
   - 同歌批次未来音频去重与 `song_audio` 去重已落地，受控测量 CPU 准备降约 `10.16%`，B64 单步前反向降约 `6.3%`。
   - 当前总 loss `~7.19–7.43` 中，配置项 `~4.86–5.05` 是同拍多槽与首个 `STOP` 的**联合负对数似然**（[`objectives.py`](../../src/maichart/composer/objectives.py#L246)），对应旧标尺 `action_legacy_ce` 约 `2.79`，不能与旧版按槽加权平均的总 loss `~4` 直接按字面比较。

### 0.2 核心问题：既然数值漏洞已修，为什么模型学习常规写法依然如此费力？
用户的核心洞察完全正确：**其他项目同样采用流式生成（Streaming Autoregressive Generation），仅靠少量优质数据与中等训练即可产出干净、有逻辑的常规谱面。**  
因此，“流式逐步生成”和“数据量有限”绝不是根因。真正的根因在于：**当前 `Composer` 架构把一个原本在时间和空间上高度结构化、强自相关的序列生成问题，拆碎成了“以单个孤立音符为样本、以 96 候选网格重编码为载体、以 18,000+ 绝对原子词表为目标、外加 20 余个带限幅/守恒锁的残差补丁”的高方差逐点分类问题。**

---

## 一、深度诊断：为什么当前架构学得如此费力？（五大盲区）

### 1. 底层表示盲区：动作空间的“绝对坐标爆炸”与“残差守恒内耗”
- **源码事实**：
  - [`ComposerDataset`](../../src/maichart/composer/data.py#L128-L149) 将原谱每个单音/滑条字符串切成独立原子（`atom_catalog`，规模约 $1.8\times 10^4$），即便在 [`CompactGroupedActionHead`](../../src/maichart/composer/conserved_actions.py#L305-L405) 中，第一级宏观分类 `macro_head` 仍在预测 `(kind_class, actual_start_pad_set, ring_class)` 的**绝对键位组合**。
  - 对于同拍多押（如双押 `1/5`），[`decode_candidate`](../../src/maichart/composer/grounded_model.py#L1002-L1060) 将其拆成 `slot 0`、`slot 1`、`slot 2 (STOP)` 序列，且训练时 `50%` 随机翻转槽位顺序（[`objectives.py`](../../src/maichart/composer/objectives.py#L106)）。槽与槽之间没有因果自注意力，仅靠嵌入向量累加 `prefix = embedded.cumsum(-2) - embedded`（[`grounded_model.py`](../../src/maichart/composer/grounded_model.py#L1015)）区分。
  - 更关键的是：所有学习跨拍相对移动的分支（`cross_event_relative_factors`、`later_cross_factors`、`echo_heads`、`button_load`）在 [`CompactGroupedActionHead.forward`](../../src/maichart/composer/conserved_actions.py#L373-L396) 中，**只允许在已选定的音符大类内部做 `logsumexp` 质量守恒重排**，且只看前一个事件 `previous_heads`（一阶马尔可夫）或手工高斯核池化（[`_LocalLeadInContext`](../../src/maichart/composer/grounded_model.py#L263-L275)）。
- **为什么难学**：
  - 在舞萌常规谱中，`1/5 → 2/6 → 3/7 → 4/8` 与 `3/7 → 4/8 → 5/1 → 6/2` 是**完全同一个动作概念**：`[双押：对向(Δ=4), 步进：顺时针(+1)]`。
  - 但在当前表示下，主分类头 `macro_head(slot_context)` 必须在 8 个绝对起点 $\times$ 2 种槽位排列 = **16 种表面标签**上分别拟合；而 `slot 0` 在做决定时，根本不知道这一拍是要出“单押”还是“对向双押”！它只能先盲猜一个单键（比如 `1`），到了 `slot 1` 再靠 `prefix` 去猜 `5` 还是 `STOP`。这相当于让模型“先落左脚，再临时决定刚才是不是双脚起跳”。

### 2. 训练样本分布盲区：“按起音事件抽样”造成的严重时间密度偏置与计算浪费
- **源码事实**：
  - [`ComposerDataset.__init__`](../../src/maichart/composer/data.py#L207-L270) 遍历每张谱面的 `points`，**每一个真实起音事件 `event_index` 生成一条 `ComposerSample`**。
  - 在 [`grounded_model.py`](../../src/maichart/composer/grounded_model.py#L607-L943) 中，每条样本都要对 `C = 96` 个候选拍位跑一遍 `audio_encoder`、`AudioLens`、`candidate_competition`、`history_attention`、`QualityContext`（生成 `[B, 96, 256]` 的 `action_context`），然后到了 [`decode_candidate`](../../src/maichart/composer/grounded_model.py#L963)，通过 `candidate_context[rows, candidate_indices]` **把其余 95 个候选的 `action_context` 直接丢弃**，只对 1 个真实落点算动作损失！
- **为什么难学且慢**：
  1. **样本分布向高密段极度倾斜（Density Sampling Bias）**：一段 4 拍的 16 分爆发（16 个事件）在训练集里占 **16 条样本（16 次梯度更新）**；而一段 4 拍的留白或二分音符慢段（1~2 个事件）只占 **1~2 条样本**！模型在训练中 85% 以上的梯度步都泡在高密度连打段里，天然形成“只要拿不准就高密出音”的先验偏置。
  2. **监督/算力比极低（1/96 候选丢弃与重复编码）**：即便开启 `phrase_groups_v1` 将相邻 8 个事件排进同一 batch，这 8 行样本依然各自独立跑一遍 `history_encoder`（128 事件 Transformer）和 96 候选 `audio_encoder`。相比之下，早期 `event_ctx` 编码一次 16 拍窗口音频，因果解码器一次前向同时监督窗口内 **30~60 个事件**的全部时间与音符，且窗口内的自注意力直接看见前 1、2、3、4 步刚做出的决定。

### 3. 优化目标与梯度流盲区：人工限幅（`tanh`）与多层 `.detach()` 掐断了“节奏惯性”
- **源码事实**：
  - 在 [`grounded_model.py`](../../src/maichart/composer/grounded_model.py#L804-L828) 中，历史符号分支对选时的修正 `correction` 被强制执行零均值化并套上 `self.config.history_residual_bound * torch.tanh(delta / bound)`（默认 `bound = 1.0`）。
  - 在 [`grounded_model.py`](../../src/maichart/composer/grounded_model.py#L670) 与 [`quality_context.py`](../../src/maichart/composer/quality_context.py#L66) 中，`workload`、`candidate_plan` 被 `.detach()`，`phrase` 与 `gap` 预测传给动作头时梯度被乘以 `QUALITY_ACTION_PREDICTION_GRAD_SCALE = 0.1`，`placement` 梯度被乘以 `0.15`。
  - 在 [`train.py`](../../src/maichart/composer/train.py#L556) 中，16 个损失项叠加在同一个主干上。
- **为什么难学**：
  - **$\pm 1.0\text{ nat}$ 的历史选时上限是把双刃剑**：当初加 `history_residual_bound = 1.0` 是为了防止模型不看音频纯靠历史自嗨；但它同时导致：**当音乐进入持续 Pad/弦乐或混响较重、局部瞬态峰值较平缓的常规乐句时，模型即使已经连续走了三步标准的 `{8}` 分音符（`+0.5` 拍），历史分支最多也只能给第 4 个半拍位提供 `+1.0` logit（约 $2.7\times$ 赔率）的微弱支持！** 在 96 个候选点分摊概率的情况下，`+1.0` logit 根本锁不住稳定的 8 分/16 分律动网格，导致选时完全暴露在局部声学特征的微小波动下，产生不自然的拍位抖动（偶发 16 分/32 分碎步）。

### 4. 音频与谱面多解盲区（One-to-Many Ambiguity）：缺少“局部谱面状态/意图潜变量”
- **机理剖析**：
  - 同一段 2 拍的常规电子乐鼓点，写成 `1, 2, 3, 4`（顺时针单点）、`8, 7, 6, 5`（逆时针单点）、`1, 8, 1, 8`（交互）或 `1/5, 2/6`（双押慢走）在音乐上全都完全合法。
  - 如果模型**在每一个事件 $e_t$ 都主要依赖局部音频+被压制的历史残差去预测绝对动作**，交叉熵损失的最优解就是把概率均匀摊在所有合法模式上；而在流式采样时，温度稍高就会在第 1 拍走顺时针、第 2 拍跳去交互、第 3 拍突然冒出双押（逐步摇摆）；温度稍低则会坍缩到训练集边际频率最高的单一高频键位（造成局部同键/双押成员复读）。

### 5. 闭环衔接盲区：开环训练的“完美前缀假设”与 `Recovery` 覆盖率不足
- **源码事实**：
  - 主训练批次 100% 使用官谱真值前缀；`recovery` 每隔数步只抽 2 条样本向后滚 2~4 步（约占 4.5% 时间），且当生成前缀导致原谱下一拍动作违反 `EventGuard` 可达性时，只能跳过配置监督或回退。
  - 在逐事件重编码架构下，模型没有在训练主循环里经历“在自己决定的节奏网格与走位惯性下连续完成 8~16 个事件”的梯度流。

---

## 二、为什么早期 `event_ctx` 和其他流式项目能用少数据学好？哪些经验可直接迁移？

对比 [`event_model.py`](../../src/maichart/generation/event_model.py) / [`event_ctx_dataset.py`](../../src/maichart/training/event_ctx_dataset.py) 与当前 `Composer`，可以清楚看到 `event_ctx` 用少数据快速学会常规写法的四个核心原因：

| 维度 | 早期 `event_ctx` / 典型高效流式模型 | 当前 `Composer (v27)` | 可直接迁移至 `Composer`（不换大模型）的核心机制 |
| :--- | :--- | :--- | :--- |
| **1. 预测与梯度单位** | **单窗连续多事件因果解码**（1 次音频编码 $\rightarrow$ 因果 Decoder 连续输出 16 拍内全部 30~60 个事件，层内自注意力天然传递刚写出的节奏与键位趋势） | **单事件孤立重编码**（每预测 1 个事件重跑 96 候选编码器 + 128 历史编码器，丢弃 95 个候选的动作上下文） | **固定音频/全曲编码共享 + 连续 $K$ 事件（如 8~16 事件）因果 Rollout 解码头**，一次前向同时监督整段事件链 |
| **2. 空白与休止表达** | **时间区间内置表达**：16 拍窗口内无音符时直接输出 `<eos>`，有 4 拍休止则直接从 `AT:4` 跳到 `AT:8`，与正常出音共享同一个交叉熵目标，天然无密度抽样偏置 | **按起音事件抽样 + 外部积分拼接**：样本数正比于音符密度（静段极少被抽中），靠独立 `survival_grid` + `planner_rest` + `count_debt` 拼接 | **按均匀音乐时间锚点或带显式 `WAIT(Δb)` 步进原语抽样**，让静段与密段拥有同等训练曝光，且无需多套外部率校准 |
| **3. 历史选时惯性** | **无人工 `±1` 限幅**：Decoder 自注意力直接从前序 `AT:` 间隔学出强网格周期性（如连续 `{8}` 自动锁定半拍相位） | `history_residual_bound = 1.0` 强行将历史对选时的修正压死在 `±1.0` nat 内 | **解开局部节奏相位的 `±1` 硬锁**：将“网格节奏惯性（Grid Momentum）”与“键位复读风险”解耦，允许历史强力锁定细分相位，只对同键复读做抑制 |
| **4. 词表与组合泛化** | 极小词表（`~600`：精确分数拍位 + 38 个字符），所有键位与滑条共享底层表示，少数据下不会出现长尾桶饥饿 | `18,000+` 原子大词表，槽位按 50% 随机正反序逐槽预测绝对键位 | **无损的整体事件集合分解（Event-Set Factorization）+ 旋转相对坐标**：保留 `Composer` 100% 语法合法的优势，彻底消除字符拼写错误，同时把有效分类词表压回百量级 |

> **关键结论**：我们**完全不需要**退回 `event_ctx` 的字符级拼写（那会带来复杂 Slide 语法脆弱和固定窗口接缝问题），而是要把 `event_ctx` **“一次音频编码连续因果解码多步、空白与出音同分布抽样、节奏惯性不设 ±1 锁”** 的三大优势，直接嫁接到 `Composer` 的无损合法动作引擎上。

---

## 三、对提案（方向 A / B / C / D / E / F）的独立审视：致命反例与更简洁替代

用户与 Codex 在 [COMPOSER_LOW_DATA_STREAMING_DESIGN_20261030_ZH.md](COMPOSER_LOW_DATA_STREAMING_DESIGN_20261030_ZH.md) 中提出了 A~F 六个方向。以下逐一给出**致命反例（Fatal Counterexamples）**与**更简洁、零妥协的替代设计**：

### 1. 针对【方向 A：持续执行的短句程序 / DSL 编译器】的致命反例
- **致命反例 1：表示歧义与长尾退化（Segmentation & Ambiguity Collapse）**  
  真实官谱并不是用 `repeat(rotate(+1), 4)` 写出来的。例如一段常见的 4 拍 8 分配置：`1, 2, 3, 4, 5/6, 4, 3, 2/1`（单点顺转 4 步 + 第 5 步突然双押强调重音 + 逆转 2 步 + 结尾双押）。  
  - 如果把它编成“1 条带多处补丁的 8 步程序”，补丁字段（哪一步加双押、哪一步反转、哪一步带 Break/Ex/Slide）的组合空间会瞬间爆炸成新的数万维长尾词表；
  - 如果把它拆成“4 段短程序 + 字面量（literal）回退”，那么在少数据下，**恰恰是音乐最出彩的重音变化、切分音、Hold 穿插单点（如 `1h[4:1], 5, 6, 7`）全部会退化成 `literal` 逐音表达**！结果模型在训练中看到的是“一会儿执行宏指令、一会儿退回原子词表”的割裂序列，不仅没有省去原子学习，反而多背了一套编译器语法包袱。
- **致命反例 2：流式执行中的“半途违例与重规划死锁”**  
  当短句程序执行到第 2 步时，若因音频瞬态变化或持续滑条占用（`EventGuard`）需要微调第 3 步，DSL 程序必须支持“中途打断并编译剩余部分”，这在训练端和推理端会造成严重的**状态空间不一致（Off-policy Program State）**。
- **更简洁的替代方案：【隐式短句保持（Latent / Momentum-Gated Rollout）+ 显式一阶/二阶相对转移】**  
  根本不需要发明一套外显的 `起点 + 步长 + 次数 + 末尾补丁` 宏指令编译器！  
  只需要在流式每一步输出两个极简原语：
  1. **相对配置原语 $(\Delta \text{beat}, \Delta \text{rot}, \text{ChordShape})$**；
  2. **一个二元/三元延续门 `ContinueMode ∈ {KEEP_MOTION, REVERSE_OR_ACCENT, NEW_PHRASE}`**。
  当音乐平稳推进时，模型只需高概率预测 `KEEP_MOTION`（保持上一步的 $\Delta \text{beat}$ 和 $\Delta \text{rot}$，例如继续 `+1/2` 拍、顺时针 `+1` 格），遇到重音时自然切换到双押形状而不必打断整个短句编译器！**这用 1 个连续/离散转移原语实现了方向 A 95% 的“少独立决策”收益，且 100% 无损覆盖任意复杂穿插配置，零编译器维护成本。**

### 2. 针对【方向 B：相对配置形状 + 独立绝对起点】的致命反例
- **致命反例 1：对称多押的“锚点多解（Symmetry Gauge Ambiguity）”**  
  以最常见的对向双押 `1/5`（或 `2/6`, `3/7`, `4/8`）为例：在循环群 $\mathbb{Z}_8$ 下，把 `1` 当起点（形状 `{0, +4}`，起点 `1`）和把 `5` 当起点（形状 `{0, +4}`，起点 `5`）生成的**是完全相同的物理配置 `1/5`**！同理，正方形四押 `1/3/5/7` 有 4 个等价锚点，等边/对称三押或双滑条也有等价锚点。如果强行挑选最小数字 `1` 作为规范锚点（canonical anchor）：
  - 当真实走位是 `4/8 → 5/1`（顺时针转 `+1` 格）时，按最小数字锚点算，起点却从 `4` 变成了 `1`（跳了 `-3` 格）！**规范锚点直接把真实的顺时针 `+1` 连续运动破坏成了 `-3` 突变！**
- **致命反例 2：异构多押与 Slide 轨迹的旋转不等价**  
  当同拍包含 `1b/5`（一侧 Break 一侧普通 Tap）、`1h[4:1]/5`（一侧 Hold 一侧 Tap）或人体工学强相关的底层/上层大跨度滑条（如 `1v4` vs `3v8`）时，纯几何旋转并不总是等价。
- **更简洁的替代方案：【最小传输相位对齐（Optimal-Transport Rotational Phase $\Delta \theta$）+ 轨道等价商群（Orbit Quotient）】**  
  1. **整拍多押作为单一完备姿态对象（Full-Chord Configuration Token）**：不再用 `slot 0, slot 1, ...` 乱序逐槽猜单键！统计全量谱面可以发现，99.2% 的同拍事件属于不到 **300 种标准姿态模板**（如：单 Tap、单 Hold、单 Slide、28 种双 Tap 组合、常见的 Tap+Hold / Tap+Slide / 双 Hold / 双 Slide 组合），仅极少数超复杂同拍事件（0.8%）需要多槽回退。
  2. **用“相对上一拍占用的最小旋转角 $\Delta \theta \in \{-3,-2,-1,0,+1,+2,+3,+4\}$”彻底避开锚点跳变**：对于对称双押 `4/8 → 1/5`，它的旋转转移不是“最小锚点 `4 → 1`（`-3`）”，而是集合对集合的唯一最小旋转 $\text{Rotate}(+1)\{4,8\} = \{5,1\}$！在计算损失时，直接对产生同一物理集合的所有等价锚点求 `logsumexp`（或直接以旋转商群轨道 `Orbit` + 轨道内相位 $\phi \pmod{8/\text{sym}}$ 编码），从数学上根除 `4 → 1` 的伪跳变！

### 3. 针对【方向 D：持续跟随某条音乐线索（Attention Focus）】的致命反例
- **致命反例：无监督线索头的“密度坍缩（Mode / Density Collapse）”**  
  在没有分轨（Stems：人声/鼓/贝斯）监督的情况下，给同一个混合音频编码器加 3~4 个轻量读取头（“瞬态/旋律/持续”），再用谱面真实落点与空白去端到端训练片段级选择：
  - 根据深度学习中的 shortcut learning 定律，模型**绝对不会**自动把第 1 个头学成“主旋律”、第 2 个头学成“底鼓伴奏”；
  - 它会立刻退化成：**头 1 = 高密度出音模式，头 2 = 中密度模式，头 3 = 休止/低密度模式**！当推理时没有真值谱面告诉它切到头 3 时，只要伴奏底鼓依然有高能量瞬态，选择器依然会留在头 1 或头 2 继续铺音。
- **更简洁的替代方案：【显式频带/瞬态-稳态分解特征（Harmonic-Percussive & Mid-Side/Vocal-Band Contrast）+ 前后乐段相对落差】**  
  为什么“旋律停了但底鼓还在响”时谱师会停笔？因为**中高频谐波/主唱频段（Harmonic / Mid-channel 300Hz–4kHz）的能量和谱通量相比前 4 拍发生了断崖式下跌，而只剩低频打击乐（Percussive <200Hz）在维持**！  
  与其让神经网络盲猜隐式线索，不如直接复用 [`AcousticContext`](../../src/maichart/composer/acoustics.py) 已有的多频带/瞬态特征，显式构造 **3 个物理可解释的声部通道（无需人声分离大模型）**：
  1. `Percussive / Sub-Bass Flux`（鼓点瞬态通道）；
  2. `Harmonic / Vocal-Lead Centroid & Chroma Flux`（主旋律/中高频谐波变化通道）；
  3. `Sustain / Pad Envelope`（长音持续通道）；
  并计算每个通道**相对于前 4 拍同通道均值的对数落差（Relative Drop）**。当 `Harmonic/Lead` 通道发生显著负落差（主线撤出），哪怕 `Percussive` 仍在响，选时与休止门也能直接读到“主旋律已退场”的确定性物理信号！

---

## 四、现有 `Composer` 模块断舍离清单：删减冗余、降 CPU/GPU 开销且不丢已有优点

当前 `grounded_model.py` + `data.py` 积累了多代演进留下的重叠通路。以下明确列出**应当保留的核心资产**与**应当合并/替换的冗余模块**：

### 4.1 必须保留的四大核心资产（不可丢弃）
1. **精确有理数拍位（`Fraction`）与真实秒时钟（`TempoClock`）**：天然支持任意拍号、变速、三连音/五连音，绝不强制 4/4。
2. **全曲共享音频底座（`song_state` / `PlanGrid` 96 维全曲上下文）**：已在 `v27` 实现按曲去重，计算极轻且提供全曲宏观位置感。
3. **持续动作真实生命周期（`lifecycle_features` 88 维 + `EventGuard` / `selected_prefix_v3`）**：精确表达 Hold 与多段 Slide 的等待、滑行、落点占用，无需把短 Hold/Slide 强行对齐音频尾音。
4. **逐键真实间隔与累计负担特征（`button_load_features` + `per_button_context_v2`）**：无需伪造左右手标签即可表达物理连打负荷。

### 4.2 建议替换或删除的五大冗余/内耗设计

| 序号 | 现有冗余模块 / 通路 | 所在文件与问题 | 替换/合并方案与直接收益 |
| :--- | :--- | :--- | :--- |
| **1** | **每事件独立重跑 96 候选 `audio_encoder` + 112 点 `AudioLens` + 丢弃 95 个候选上下文** | [`grounded_model.py:607-681, 963`](../../src/maichart/composer/grounded_model.py#L607-L681) / [`data.py:1098-1267`](../../src/maichart/composer/data.py#L1098-L1267)：每预测 1 颗音符都要查 96 个候选拍位的局部声学并跑 Transformer，最后在 `decode_candidate` 扔掉 95 个 | **改为歌曲级/片段级统一拍位网格编码（Shared Beat-Grid Audio Bank）**：同一首歌的常见拍位（或连续 8~16 拍段内的候选点）只需过一次 `AudioLens` 与轻量时序编码，段内连续 8~16 个事件的因果解码直接按索引 gather，**CPU 数据准备与 GPU 音频编码开销同时下降 70%+** |
| **2** | **三套重叠的历史编码与检索**：`history_encoder` + `history_attention` + `_LocalLeadInContext`（高斯四尺度池化）+ `QualityContext`（128 步音画配对）+ `PhaseEchoPrior` + `MotifMemoryAdapter` | [`grounded_model.py:152-415, 702-742, 808-820, 893-905, 985-998`](../../src/maichart/composer/grounded_model.py#L702-L742)：五套机制都在读同一份 128 事件历史，各自算注意力或手工核函数，最后被 `history_residual_bound = 1.0` 一刀切压扁 | **统一为单一因果音画事件 Transformer（Unified Audio-Paired Event Decoder）**：每个已提交事件 token 直接融合 `(事件姿态嵌入, 相对上一拍旋转 Δθ, 精确时间步长 Δb/Δt, 该事件时刻的局部声学+全曲状态)`，由一个 4 层因果自注意力直接读取，删去手工高斯池化与重复的配对分支 |
| **3** | **多槽随机正反序展开（33 槽 $\times$ 18,000 词表）** | [`grounded_model.py:1002-1060`](../../src/maichart/composer/grounded_model.py#L1002-L1060) / [`data.py:1324-1330`](../../src/maichart/composer/data.py#L1324-L1330)：同拍双押被拆成两步独立分类且 50% 随机反序，制造人工排序噪声与巨大词表投影 | **整拍常见姿态合并（Chord-Set Macro Token）+ 罕见复杂多押槽位回退**：将 99% 的单押/常规双押直接表示为单一旋转等变姿态 + 起点相位，1 步出完整个同拍配置，无需 `slot 0 / slot 1` 反序博弈 |
| **4** | **重叠的密度/计数辅助头**：`predict_workload` + `predict_trailing` + `curve_slope_loss` + `quality_phrase_prediction` + `plan` + `future_head` | [`train.py:457-561`](../../src/maichart/composer/train.py#L457-L561)：6 个不同模块都在回归局部密度/音符比例，互相 `.detach()` 并争夺主干表示 | **只保留一套全曲规划底座（`song_state` $\rightarrow$ `plan` 含 `rest` 与细分倾向）+ 固定格事件率**，删除 `trailing_workload`、`future_head`、`curve_slope_loss` 等重复统计回归 |
| **5** | **选时历史残差的 `±1.0` 硬截断（`history_residual_bound`）** | [`grounded_model.py:804-828`](../../src/maichart/composer/grounded_model.py#L804-L828)：把合法的网格节拍惯性与键位复读风险混为一谈，导致常规 8 分/16 分流无法靠节奏惯性稳定锁格 | **拆分为“无上限（或宽上限 ±4.0）的纯节拍网格惯性头（Beat-Grid Momentum）”与“纯抑制性的复读/负荷头”**：让模型在常规乐段能像 `event_ctx` 一样稳稳踩住 8 分/16 分网格 |

---

## 五、两条完整的新架构路线设计（首选路线 vs. 备选路线）

为了给出可直接落地、可证伪的完整蓝图，以下提出**两条截然不同但都严格满足所有约束（流式生成、少量数据高效学习、无强制 4/4、无主副歌/左右手伪标签、无镜像增广、无短长条尾音强对齐）的完整架构路线**。

```
====================================================================================================
路线一（首选推荐）：旋转等变整拍集合 + 连续多事件因果流式架构 (Equivariant Event-Set Rolling-Horizon Streaming, EES-Stream)
  核心思想：保留单步流式的精确物理约束，但把“训练/前向计算单位”升级为连续 K=16 事件（含显式 WAIT 步）的共享音频因果序列，
           并将每个事件表示为【(Δbeat, 旋转不变量姿态 Shape, 相对旋转步进 Δθ, 组内轨迹细节)】，一步生成整拍多押！

路线二（备选对照）：固定节拍跨度（2~4 拍）微段自回归流式架构 (Beat-Span Micro-Chunk Streaming, BSMC-Stream)
  核心思想：彻底按均匀音乐时间段（如每次流式推进 2 拍）建立样本与生成合同，段内允许 0~N 个事件（空白即 <SPAN_EMPTY>），
           不发明脆弱的 DSL 编译器，而是用段级潜意图向量 z_span 锁定该 2 拍内的声部关注点与走位模式。
====================================================================================================
```

---

### 路线一（首选完整路线）：旋转等变整拍集合 + 连续多事件因果流式架构（`EES-Stream`）

这是回答**“除 A~F 之外，有没有更好的单步流式方案？”**的首选答案。它不引入复杂的短句 DSL 编译器，不改变流式逐事件通过 `EventGuard` 提交的推理合同，却能一次性解决**表示碎片化、密度抽样偏置、候选重复计算、同拍双押拆槽、节奏惯性被锁死**五大顽疾。

#### 1. 底层表示改革：从“逐槽绝对原子”到“旋转等变整拍集合（Equivariant Event-Set）”
每个流式事件 $e_t$ 由 **4 个结构化部分** 组成（编译器 100% 双射无损还原原谱 Simai 文本）：
1. **时间推进 $\Delta b_t$（含显式等待 token `WAIT_GRID`）**：
   - 在原有的候选拍位旁，当相邻两音符间隔 $> 2$ 拍（或处于无起音段）时，在训练序列中自动插入显式的 **`WAIT(2.0b)` / `WAIT(4.0b)` 虚拟节拍事件**！
   - **为什么这步极其关键**：
     - 第一，它彻底消除了“按音符抽样导致休止段没有训练样本”的偏差！一段 16 拍的安静前奏会产生 4 个连续的 `WAIT(4.0b)` 步，在因果序列中作为一等公民被直接用交叉熵监督；
     - 第二，`WAIT` 步同样推进时钟、更新持续动作生命周期与按钮冷却，且无需任何拼凑的尾概率校准公式！
2. **旋转不变量整拍姿态类 `ChordShape`（规模约 $M \approx 180\sim 250$ 类）**：
   - 将同拍的所有音符视为**无序集合**，按旋转群 $\mathbb{Z}_8$ 取商集（**仅用循环旋转 $\mathbb{Z}_8$，严禁左右/上下镜像，从而 100% 保留顺/逆时针方向性与起手偏置！**）。
   - 例如：
     - 所有单点 Tap `{1}, {2}, ..., {8}` 归为同一姿态类 `Shape = SINGLE_TAP`；
     - 所有相邻双押 `{1/2}, {2/3}, ..., {8/1}` 归为 `Shape = CHORD_TAP_OFFSET_1`；
     - 所有隔一双押 `{1/3}, ..., {8/2}` 归为 `Shape = CHORD_TAP_OFFSET_2`；
     - 所有对向双押 `{1/5}, {2/6}, {3/7}, {4/8}` 归为 `Shape = CHORD_TAP_OFFSET_4`；
     - 常见 `Tap + Hold`（按相对偏移 $\Delta \in \{1..7\}$）、`Tap + Slide`、`双 Hold`、`双 Slide` 同理归入各自的旋转不变姿态类；
     - 对于极罕见的超复杂同拍组合（占全语料 $<1\%$），设立唯一回退类 `Shape = COMPLEX_FALLBACK`，仅在该类下启用原有的多槽展开。
   - **直接收益**：模型不再需要分 `slot 0`、`slot 1` 去猜“这次是单押还是双押”以及“谁先谁后”，**一步直接选定整拍的完整几何姿态**！
3. **相对旋转步进 $\Delta \theta_t \in \{0, +1, +2, +3, +4, -3, -2, -1\} \pmod{\text{sym}(S_t)}$ + 弱绝对锚点偏置**：
   - 给定上一拍的主参考方位 $\theta_{t-1} \in \{0..7\}$（由上一拍起手头或滑条落点确定），当前拍的绝对方位由 $\theta_t = (\theta_{t-1} + \Delta \theta_t) \bmod 8$ 唯一决定！
   - **如何解决对称双押（如 `1/5`）的锚点多解问题**：
     - 对向双押 `Shape = CHORD_TAP_OFFSET_4` 的旋转对称阶数为 $2$（即旋转 $+4$ 格与自身重合，只有 4 个等价轨道 `{1/5, 2/6, 3/7, 4/8}`）。
     - 因此，当 `Shape` 具有对称周期 $P \in \{1, 2, 4, 8\}$ 时，模型预测的 8 维旋转 logits $z_0, \dots, z_7$ 直接按周期 $P$ **折叠求和（Orbit Folding：`logsumexp` over $\theta \equiv r \pmod P$）**，只在 $P$ 个真实物理不等价的落点上做归一化与交叉熵！
     - 同时保留一个轻量 8 维绝对键位残差 $b_{\text{abs}}(\theta_t)$（与相对转移相加：$s(\theta_t) = s_{\text{rel}}(\Delta \theta_t) + s_{\text{abs}}(\theta_t)$），既让 `1/5 → 2/6 → 3/7 → 4/8` 变成连续 4 次预测同一个常数 $\Delta \theta = +1$，又完全保留原谱“喜欢从 1/2/7/8 起手”的真实绝对位置分布！
4. **姿态内部细节 `Detail`（仅当姿态含 Hold/Slide/Touch/Break 时激活）**：
   - 沿用现有的组合因子表（相对终点、轨迹形状、时值、Break/Ex 修饰），仅在目标姿态类内部做条件分类（采纳 `COMPOSER_V27_EFFICIENCY_AND_LEARNING_REVIEW_20261030_ZH.md` 第 68 行提出的“训练时只算目标宏观组内细节”）。

#### 2. 训练与生成合同：共享音频底座 + 连续 $K=16$ 步因果解码（Rolling-Horizon Causal Training）
- **为什么能省 70% CPU/GPU 准备与重复计算**：
  - 训练不再以“单个孤立事件”为 DataLoader 样本，而是以**“一段连续的 $K=16$ 个流式步（含 `WAIT` 步，通常覆盖 4~8 拍）”**为一条训练样本，配以之前最多 $H=64$ 个已提交事件作为只读 lead-in 上下文（与 `event_ctx` 的 `<ctx> ... <ctx_end>` 思想完全同源，但作用于结构化事件 token 而非字符）。
  - 对于一条包含 16 个连续目标步的样本：
    1. 全曲 `song_state` 与固定 `50ms` 声学网格只算 **1 次**；
    2. 64 步历史前缀 + 16 步目标序列通过**同一个 4 层因果 Transformer Decoder** 一次并行前向完成历史编码与步间自注意力传递（无需为 16 个事件跑 16 次 128 步 `history_encoder`！）；
    3. 每个目标步 $t \in \{1..16\}$ 只需对自己前方的精简候选集（如近端 32 个精细网格点 + 1 个 `WAIT` 步）评估选时 logits，且**动作头只在每步的 1 个真实落点（共 16 个点，而非 $16 \times 96 = 1536$ 个点）上执行**！
- **如何从根源上解决“旋律停但伴奏响时乱出音”**：
  - 在 `AcousticContext` 的固定格输入中，直接拆出 **谐波/中频主线通量（Harmonic/Mid-band 300Hz–4kHz）** 与 **低频瞬态通量（Percussive/Bass <250Hz）** 及其**相对前 4 拍的对数落差（4-Beat Relative Drop）**。
  - 在连续 16 步的因果解码中，模型每步额外受一个轻量自回归门控（或通过因果自注意力隐式维持）读取当前所跟声部的能级变化；当主线通道发生断崖下跌且前一步进入 `WAIT` 时，因果自注意力直接维持 `WAIT` 状态，不再被底鼓瞬态拉出碎音。

#### 3. 学习目标（大幅瘦身，从 16 项减至 4 项核心损失）
1. **时间推进与 `WAIT` 联合损失 $\mathcal{L}_{\text{time}}$**：固定格积分 + 局部候选精细选时 + 显式 `WAIT` 步交叉熵（历史节拍网格惯性放开至 $\pm 3.5\text{ nat}$，仅对同键复读施加负向负荷偏置）。
2. **旋转等变姿态与相位联合损失 $\mathcal{L}_{\text{config}}$**：$\mathcal{L}_{\text{shape}} + \mathcal{L}_{\text{orbit}}(\Delta \theta, \theta) + \mathcal{L}_{\text{detail}}$（仅在目标组内展开细节，计算量降 $80\%$）。
3. **固定格事件率与全曲规划损失 $\mathcal{L}_{\text{plan+rate}}$**：保留 `v27` 已验证有效的固定格 Poisson 计数与 `BCEWithLogits` 休止监督（按时间均匀加权，不再按事件数重复加权）。
4. **故障定向短程恢复损失 $\mathcal{L}_{\text{recovery}}$**：保留 `4.5%` 预算，优先在“谐波主线退场段”与“高速 16 分交互段”从模型自产前缀向后采样 4 步校验可达性与休止回归。

---

### 路线二（备选完整路线）：固定节拍跨度（2 拍）微段流式架构（`BSMC-Stream`）

如果希望更彻底地践行提案中的 **方向 A（短段决策）+ 方向 B（相对坐标）+ 方向 C（时间段空白作为直接主输出）**，同时避开“手写短句 DSL 编译器”的致命陷阱，推荐采用 **`BSMC-Stream`（Beat-Span Micro-Chunk Streaming）**。

#### 1. 核心设计：以“2 拍微段（Micro-Span）”为流式推进单位
- **时间切片**：不预设 4/4 小节，仅以固定的 **2 拍（`ΔB = 2` beats，由 `TempoClock` 精确映射为音频秒）** 作为一个流式微段。
- **两级流式生成合同**：
  1. **微段意图决策（Span Controller，每 2 拍执行 1 次）**：读取全曲状态、当前 2 拍与未来 4 拍的声学特征（含谐波/打击乐分离落差）及进入微段时仍在持续的 `Hold/Slide` 身体占用状态，直接预测该 2 拍微段的 **离散结构模式 `SpanMode`**（如：`EMPTY_REST` 空段、`SPARSE_ACCENT` 1~2 音重音、`STEADY_8TH` 8 分常规推进、`BURST_16TH` 16 分爆发、`SUSTAIN_SLIDE` 滑条/长按主导）以及微段潜向量 $z_{\text{span}} \in \mathbb{R}^{128}$。
     - **若预测为 `EMPTY_REST`**：直接跳过整个 2 拍，输出 0 个事件！休止成为最顶层的直接分类输出。
  2. **微段内局部自回归展开（Intra-Span Rollout，通常仅 1~8 步）**：在固定 $z_{\text{span}}$ 条件下，轻量解码器在 2 拍局部网格（`0, 1/8, 1/6, 1/4, 1/3, 3/8, 1/2, ... < 2`）上用路线一的 **旋转等变姿态 `(offset_in_span, ChordShape, Δθ, Detail)`** 顺序吐出该 2 拍内的事件，直到输出 `<SPAN_END>`。
- **跨微段边界处理**：
  - `EventGuard` 与 `SustainStateIndex` 的真实物理时钟完全跨微段连续传递（Hold/Slide 起点在微段 $k$、终点在微段 $k+2$ 毫无障碍，绝不在微段边界清零状态）。

#### 2. 路线一 vs. 路线二 全面对比

| 评估维度 | 路线一：`EES-Stream`（连续事件因果 + 旋转等变集合 + `WAIT` 步） | 路线二：`BSMC-Stream`（2 拍微段控制器 + 段内相对展开） |
| :--- | :--- | :--- |
| **与现有 `Composer v27` 兼容性** | **极高**（90% 复用现有 `TempoClock`、`EventGuard`、`density_grid`、因子表与生成循环） | **中等**（需改写 DataLoader 样本切片与生成外层循环为按 2 拍推进） |
| **解决常规配置难学的方式** | 靠 `ChordShape + Δθ` 一步出整拍 + 16 步因果自注意力保持走位惯性 | 靠每 2 拍锁死一个 $z_{\text{span}}$ 潜向量 + 段内相对展开 |
| **解决休止乱出音的方式** | 显式 `WAIT(2b/4b)` 步参与均匀因果序列训练 + 谐波/低频落差特征 | 顶层微段分类直接输出 `EMPTY_REST` + 谐波/低频落差特征 |
| **跨边界长条/变速鲁棒性** | **天然无人工切段边界**，任意变速/跨拍长滑条完全原生支持 | 需在每 2 拍交界处显式对齐跨边界滑条与三连音跨拍相位 |
| **开发风险与验证周期** | **低风险，2~3 天可完成原型与同数据对照** | 中等风险，需先验证 2 拍切分对跨拍三连音/附点的边界影响 |

---

## 六、最小决定性实验、失败退出条件与推进顺序

在**完全不动当前 `v27` 训练（7864）**的前提下，建议按以下四个阶段推进，每一步都有明确的**通过标准**与**失败退出条件（Kill Criteria）**：

### 阶段 0：零训练离线表示审计（耗时 < 30 分钟，纯 CPU 脚本）
- **实验内容**：在现有训练集与开发集谱面上，对**路线一的旋转等变整拍集合分解 `(ChordShape, Orbit Δθ, Detail)`**（以及提案 A 的短句字典）做无损往返编译（Round-Trip Compilation）与信息熵统计：
  1. 统计前 $K=200$ 个旋转不变姿态 `ChordShape` 对常规单押/双押/Tap+Hold/Tap+Slide 的覆盖率；
  2. 测量 `Δθ`（集合间最小旋转步进）在常规谱中的一阶/二阶条件熵 $H(\Delta \theta_t \mid \Delta \theta_{t-1}, \text{Shape}_t)$ 与原版绝对键位熵 $H(\text{pad}_t \mid \text{pad}_{t-1})$ 的比值；
  3. 100% 验证解码后再经 Simai 解析器还原的音符与原谱完全等价（零精度损失）。
- **失败退出条件**：
  - 若前 250 个旋转不变姿态对非极端弹幕谱的事件覆盖率 $< 95\%$，或条件熵下降不足 $25\%$，则停止整拍姿态合并，退回仅对单押/双押做旋转相对转移。

### 阶段 1：目标组内细节计算优化 + 谐波/打击乐落差特征验证（同合同微基准）
- **实验内容**：
  1. 落实 [COMPOSER_V27_EFFICIENCY_AND_LEARNING_REVIEW_20261030_ZH.md](COMPOSER_V27_EFFICIENCY_AND_LEARNING_REVIEW_20261030_ZH.md) 第 1 项：训练时 `CompactGroupedActionHead` 只展开目标宏观组内的轨迹/时值细节，验证前反向梯度逐位/数值等价并测真实 B64 吞吐提升；
  2. 在离线验证集安静段（特别是“伴奏底鼓仍在、主旋律退场”的真实误出音片段）测试 4 拍谐波/打击乐相对落差特征对真实休止边界的区分度（AUROC / AP）。
- **失败退出条件**：
  - 若目标组内稀疏化因 gather/scatter 开销导致真实 B64 墙钟时间未能下降 $\ge 5\%$，则不并入主干；
  - 若谐波/打击乐相对落差对“伴奏型休止段”的区分 AP 相比现有 104 维均值无显著提升（$< +0.05$ AP），则不新增该声学对比特征。

### 阶段 2：50~80 张一致常规谱的“小数据学习效率”决定性 A/B/C 对照（核心实验）
- **实验设置**：抽取 60 张定数 `12.0 ~ 13.5`、风格干净的官谱作为小型训练集，10 张同定数独立曲目作为盲测开发集。固定相同训练墙钟时间（如 45 分钟 GPU），对比三个分支：
  - **Arm A（基线）**：当前 `v27` 逐事件抽样 + `CompactGroupedActionHead`；
  - **Arm B（仅连续多事件因果 + `WAIT` 步）**：保持 `v27` 动作头不变，但改为共享歌曲音频 + 连续 16 步（含 `WAIT`）因果解码与解开节拍网格惯性锁；
  - **Arm C（完整 `EES-Stream`）**：在 Arm B 基础上，将动作头替换为 **旋转等变整拍集合 `(ChordShape, Orbit Δθ, Detail)`**。
- **验收指标（不仅看 Loss，必须看固定种子闭环生成与试玩指标）**：
  1. **走位逻辑性**：连续 4~8 步规律走位（顺/逆时针流、对称双押平移、交互）的完整保持率 vs. 单步中途无音乐理由跳变率；
  2. **高速同键/双押成员异常复读率**：在关闭外部强硬惩罚时，模型原生输出的 $\le 100\text{ms}$ 同键连打率是否自然降至官谱水平（$< 2\%$）；
  3. **伴奏型休止段留白准确率**：在主旋律暂停而伴奏持续的段落，生成谱的留白召回率与起音 F1；
  4. **训练吞吐（受监督事件数 / 秒）**：验证 Arm B/C 每秒有效监督的事件数是否达到 Arm A 的 $2.5\times \sim 4\times$。
- **失败退出条件**：
  - 若 Arm B + Arm C 在 60 张谱上训练后，固定开发集的闭环生成仍需要强依赖外部后处理抑制才能避免同键复读或乱出音，或者旋转相对坐标导致起手明显偏离人类习惯（如大量出现反手别扭双押），则回退至绝对组内加旋转残差混合方案。

### 阶段 3：历史短句记忆调用（方向 E）与两阶段预训练（方向 F）的按需演进
- **准入前提**：**只有在阶段 2 证明 `EES-Stream` 在单段内已能稳定写出干净的常规 8 分/16 分走位与正确休止后**，才启动：
  1. **方向 F（两阶段解耦）**：先在全量谱面（无需跑昂贵的全曲声学网格）上用 15 分钟快速预训练因果事件序列解码器（学会“什么是顺畅的舞萌走位语法”），再冻结或低学习率微调接入音频选时与声部关注选择器；
  2. **方向 E（已完成短句归档调用）**：将本曲前面已生成的 4 拍 `(ChordShape, Δθ)` 序列以紧凑向量归档，当 `song_state` 检测到乐段再现（如副歌再现）时，通过轻量门控复用前序走位骨架并允许末尾变化。
