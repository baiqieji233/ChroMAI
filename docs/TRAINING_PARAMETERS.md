# event_v1 训练参数手册

面向 **event_v1 事件模型**的训练参数参考：每个参数做什么、对结果/速度/显存的影响、推荐值，
以及在当前设备（RTX 5070 Ti Laptop 12 GB / 31 GB RAM）上如何按数据规模选型、避免过拟合、
以及如何做"多次抽卡"式实验。

文中标注 **[实测]** 的数字来自本项目的真实运行记录，标注 **[估算]** 的是按规则外推。

> **相关文档**：[金标集与谱面验收](GOLD_AND_ACCEPTANCE.md) —— 如何用认可的高质量谱面同时当
> 训练组与验收标准，以及音频结构特征（相对位置 / 相对强度）的完整说明。

---

## 0. 特征契约（结构特征改动后）

| 项 | 值 | 说明 |
|---|---|---|
| `INPUT_DIM` | **108** | 98 声学 + 10 时序上下文 |
| 声学新增列 | `relative_rms` / `intensity_slope` / `harmonic_novelty` | 相对强度显式化（1.0 = 全曲中位响度；新颖度峰值=段落变化） |
| 上下文新增列 | `beat_progress` / `bar_phase_sin,cos` / `phrase_phase_sin,cos` | 音乐位置显式化（拍进度、小节相位、乐句相位） |
| 特征版本 | `event-audio-v3` | 参与缓存键，升级会**自动作废**旧缓存 |
| 全曲记忆帧数 | **256**（`GLOBAL_MEMORY_FRAMES`，CLI `--global-frames`） | 160 s 歌曲 ≈ 每 2 拍 1 个记忆 token；越大越能"回头看"相似段落，代价是编码器注意力变长 |

**影响**：`input_dim` 属数据合同 → 旧 checkpoint（100 维）**不能续训/微调**，
但**不需要重新预处理**（音频特征缓存按特征配置自动失效并重建）。

---

## 1. 快速推荐（照抄这一节即可）

### 1.1 按数据规模选模型档位

有效批量统一为 16，便于横向比较；显存不够时**减半 batch、加倍梯度累积**，数学等价。

| 档 | d_model | 层数 | 头数 | ffn | 参数量 | batch | 累积 | 适用数据量 |
|---|---|---|---|---|---|---|---|---|
| A 保守 | 256 | 8 | 8 | 1024 | 15.7M **[实测]** | 16 | 1 | < 1500 首 |
| B 推荐 | 320 | 8 | 8 | 1280 | ~24M **[估算]** | 8 | 2 | 1500–3000 首 |
| C 进阶 | 384 | 10 | 8 | 1536 | 42.9M **[实测]** | 4 | 4 | 3000–5000 首 |
| D 上限 | 512 | 10 | 8 | 2048 | 75.6M **[实测]** | 2 | 8 | > 5000 首 |

显存约束（实测锚点：`d256/L8/ffn1024 + batch 12 + 768 tokens = 7.4 GB`）：

```
batch_size × (d_model / 256)² × (num_layers / 8) ≤ 12     # 序列长度 768 时
```

### 1.2 其余参数（C 档 / 3000 首场景）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| 验证 manifest | **必填** | 最好直接填**金标验证集**（见金标文档）；没有它早停/best/学习率下降全部失效 |
| epochs | 30 | 由早停提前结束 |
| patience | 3–5 | 抽卡要"及时止损" |
| lr | 2e-4（C/D 档可 3e-4；金标 `finetune` 阶段 2e-5） | 由 `ReduceLROnPlateau(factor 0.5, patience 3)` 自动衰减 |
| max_tokens | **与 segment_beats 配套**（见 §3.4） | 例如 S=16 → 640；S=32 → 1024；S=64 → 1024 |
| segment_beats | **16**（从 64 下调，见 §3.7） | 单窗口 token 需求下降后预算才够，空白问题从结构上消失 |
| dropout | 0.1 | 过拟合时升到 0.15 |
| weight_decay | 0.01 | 过拟合时升到 0.02–0.05 |
| condition_dropout | 事件 fresh 默认 0 | 条件随机丢弃；resume / finetune 继承所选 checkpoint，旧模型可能为 0.15；与模型 dropout、context_dropout 独立 |
| 全曲记忆帧数 | 256（默认） | 想更强结构感知可加到 512，代价是训练变慢 |
| save_every | 2–3 | 抽卡时多留中间档 |
| 音频缓存上限 | 4 GB（≈1300 首） | 见 §3.6 |
| 随机种子 | 留空 | 每次运行自动抽新种子 |

---

## 2. 参数逐项说明

### 2.1 数据合同参数（改动会作废已有 checkpoint）

这些值写入 checkpoint 的 `data_contract`，**resume/finetune 时会逐项比对，不一致直接拒绝**
（`train_events.py:141`、`event_checkpoints.py:42`）：

| 参数 | 影响 | 改动代价 |
|---|---|---|
| `segment_beats` | 每个训练片段切多少拍 | 必须从头训练；**不需要重新预处理** |
| `max_tokens` | 位置网格与预算上限 | 必须从头训练；**不需要重新预处理** |
| `sample_rate` / `hop_length` / `feature_profile` / `input_dim` | 音频特征本身 | 必须重新预处理（缓存全部失效） |
| `vocabulary_signature` / `dataset_signature` | 词表与数据集指纹 | 换数据集自然改变 |

> 关键澄清：`segment_beats` 与 `max_tokens` 只影响**训练时如何切分**，不影响磁盘上的音频特征缓存。
> 所以调整它们只要重训，不必重跑几小时的预处理。

### 2.2 模型结构

| 参数 | 默认 | 作用与影响 |
|---|---|---|
| `d_model` | 256 | 表示宽度。**对参数量与显存影响最大**（参数量约正比 d²）。数据量大时优先加它。 |
| `num_layers` | 4 | 深度。决定能建模多长的依赖；本任务里深度比宽度更缺。 |
| `nhead` | 4 | 注意力头数，**必须整除 `d_model`**（256/8=32、384/8=48、512/8=64）。 |
| `dim_feedforward` | 1024 | 前馈层宽度，惯例取 4×`d_model`。 |
| `dropout` | 0.1 | 训练期随机丢弃。防过拟合的第一道闸；数据少→升到 0.15–0.2。 |

### 2.3 优化

| 参数 | 默认 | 作用与影响 |
|---|---|---|
| `lr` | 2e-4（finetune 2e-5） | 学习率。太大 loss 抖动/发散，太小收敛慢。**抽卡实验请固定它**。 |
| `weight_decay` | 0.01 | AdamW 的 L2 正则。防过拟合的第二道闸，0.02–0.05 是常用区间。 |
| `batch_size` | 8 | 每批片段数。小→梯度噪声大；大→显存吃紧且泛化可能变差。每批按**批内最长样本**补齐。 |
| `grad_accumulation` | 1 | 累积 N 批才更新一次，等效放大批量。显存不足时用它换有效批量。 |
| AMP（`--no-amp` 关闭） | 开 | float16 混合精度：省显存、提速，不改变数学目标。NaN 时才关。 |
| `epochs` | 20 | 训练轮数上限；有验证集时会被早停截断。 |
| `patience` | 10 | 连续 N 轮验证 loss 不改善就停。**只在填了验证 manifest 时生效**，否则是死代码。 |
| `save_every` | 5 | 每 N 轮额外存 `event_epoch_NNN.pt`。中断/后段过拟合时的退路。 |
| `seed` | 42 → **留空＝随机** | 固定权重初始化 + 数据打乱顺序。见 §4。 |

### 2.4 条件与生成相关

| 参数 | 默认 | 作用与影响 |
|---|---|---|
| `condition_dropout` | 事件 fresh 默认 0 | 额外随机丢弃目标派生控制，供 CFG 使用；已有模型按条件合同继承，可能为 0.15。此项对应 `condition_params.dropout` / `--condition-dropout`，与模型内部 `dropout`、上下文 `context_dropout` 独立。 |
| 难度/等级条件 | 由 manifest 决定 | 训练时用谱面**作者标注的 BPM**与等级，与检测器无关。 |

### 2.5 运行环境

| 参数 | 默认 | 作用与影响 |
|---|---|---|
| `device` | auto | cuda/cpu。CPU 训练在本规模下不实用。 |
| `音频缓存上限` | 4 GB | 常驻内存缓存预算，见 §3.6。 |

---

## 3. 实测基准与规模外推

### 3.1 当前数据集实测

| 指标 | 数值 |
|---|---|
| 歌曲 / 难度 / 可用文档 | 1003 / 3555 / 1072 |
| 训练片段 | 6403（`max_tokens=768`）→ 6603（`max_tokens=1024`） |
| tokens/epoch | 2,838,898 → 3,006,174 |
| 片段长度分位（1024 档） | p25 353 / p50 459 / p75 567 / p90 669 / p95 730 / p99 849 / max 1018 |
| 因超长被丢弃的片段 | 216（768 档）→ 16（1024 档） |
| 每 epoch 耗时 | **3.12 分钟**（epoch 10→20 实测，534 batch，batch 12） |
| 显存占用 | 7.4 GB / 12 GB（d256/L8/ff1024，batch 12） |
| tokens / 参数 / epoch | 0.181 |

### 3.2 每 epoch 时间外推 **[估算]**

以"算力≈参数量、tokens≈歌曲数"缩放：

| 数据量 | A 档 (15.7M) | B 档 (~24M) | C 档 (42.9M) |
|---|---|---|---|
| 2000 首 | 6 min | 9 min | 17 min |
| 3000 首 | 9 min | 14 min | 25 min |
| 5000 首 | 16 min | 24 min | 42 min |

### 3.3 抽卡的总时间预算 **[估算]**

| 场景 | 每次抽卡（30 epoch） | 4 次抽卡 |
|---|---|---|
| 3000 首 + C 档 | ~12.5 h | ~2 天 |
| 3000 首 + B 档 | ~7 h | ~28 h |
| 2000 首 + B 档 | ~4.7 h | ~19 h |

### 3.4 `segment_beats` 与 `max_tokens` 必须配套 **[实测]**

两者单位不同（拍 vs token），但被每拍 token 数严格耦合。**同一份语料**（1072 篇可用谱面）实测：

| segment_beats | 训练窗口数 | 窗口 token p50 | p95 | p99 | max | 建议 max_tokens | 覆盖率 |
|---|---|---|---|---|---|---|---|
| 8 | 49,087 | 63 | 106 | 132 | 300 | 512 | 100% |
| **16** | **24,805** | **122** | **202** | **244** | **518** | **640** | **100%** |
| 32 | 12,684 | 238 | 387 | 460 | 817 | 1024 | 100% |
| 64 | 6,603 | 459 | 730 | 849 | 1018 | 1024 | 100% |
| 256（曾用） | 2,072 | 1505 | 2497 | 2918 | 3783 | 2560–3072 | **只有 51.6%** |

规则：**`max_tokens ≥ 该 segment_beats 下窗口 token 需求的 p99`**。

> 为什么重要：填小了，超预算的窗口会在训练时被**静默丢弃**（丢的恰好是最密集的那部分），
> 生成时则**截断**——剩余拍数永久空白，于是出现"每隔一段时间一大段空白"。
> 256 拍 + 1536 token 就是这种情况：48.4% 的训练窗口被丢，生成长空白达 32 秒。

补齐是按**批内最长样本**做的，所以把 `max_tokens` 提到覆盖 p99 几乎不增加计算量（约 2–5%）。

### 3.5 数据量与参数量配比

参考点：当前 0.181 tokens/参数/epoch。经验上累计 tokens 达到参数量的 10–20 倍较健康
（30 epoch × 0.181 ≈ 5.4 倍，仍偏数据饥渴）。因此：

- 数据 < 1500 首：**别加参数**，先加数据。
- 数据 1500–3000 首：B 档是甜点。
- 数据 > 3000 首：才值得上 C/D 档。

### 3.6 音频缓存上限（内存约束）**[实测]**

每首歌的特征约 **2.9 MB 常驻内存**（98 列 float32 × 帧数，加上 times），磁盘 npz 压缩后 2.2 MB。
旧实现是**无上限字典**，会随语料线性增长：

| 歌曲数 | 常驻内存 | 说明 |
|---|---|---|
| 1000 | ~2.9 GB | 当前规模 |
| 3000 | ~8.7 GB | 需要提高上限或接受淘汰 |
| 5000 | ~14.5 GB | 31 GB 机器上很危险 |

现在的实现是**按字节计量的 LRU**：超预算时淘汰最久未用的一首，被淘汰的歌从 npz 缓存重新读取
（**不重新分析音频**，代价只是 I/O）。填 `0` 关闭缓存；填 `-1` 恢复无上限的旧行为。

### 3.7 为什么把 `segment_beats` 从 64 降到 16

切细的收益是**结构性的**，不只是数字好看：

| | S=64 | S=16 |
|---|---|---|
| 窗口 token p99 | 849 | **244** |
| 可用 `max_tokens` | 1024 | **640** |
| 单个窗口失败时的空白上限 | 19.7 秒 | **4.9 秒** |
| 窗口数（516 拍歌曲） | 8 | 32（**生成时间更长**） |
| 每轮梯度更新次数（batch 12） | 550 | 2067（**需同步放大 batch**） |

三条要点：

1. **空白上限随窗口线性缩短**：一个窗口完全失败时留下的洞最多就是该窗口长度。
   拉长窗口不会减少总空白比例（窗口数×单次损失守恒），只会让单次空白更**长**。
2. **必须同步放大 `batch_size`**：窗口数变多后每轮更新次数暴涨、每次更新的 token 数骤减，
   等价于把有效批量缩小了 4 倍。S=16 建议 batch 32–48（保持"每次更新 ≈ 5500 token"）。
3. **不要低于 16 拍**：一个窗口是模型能看到的全部内容，8 拍 = 2 小节，学不到 4/8 小节的乐句；
   且生成时间与窗口数成正比（S=16 时一首歌约 6–10 分钟）。
   另外窗口数变多会放大数据加载开销（当前 `num_workers=0`），若 GPU 利用率掉下来需要开多进程加载。

### 3.8 音频特征预提取（并行）**[实测]**

特征提取原先发生在训练的**第一个 epoch** 里，由数据加载器逐首计算，期间 GPU 空等。
现在改为训练开始前用进程池一次性算完：

| | 串行 | 4 进程 | 8 进程 |
|---|---|---|---|
| 每首耗时 | 1.39 s | 0.64 s | 0.61 s |
| 1003 首总耗时 | **23.3 min** | 10.7 min | **10.2 min** |
| 加速比 | 1.00× | 2.17× | 2.28× |

**为什么只有 2.2×？** 这是本机的 CPU 上限，不是代码问题：用纯 numpy 负载做同样的多进程基准，
本机（32 逻辑核、笔记本功耗限制）也只到 **4 进程 2.00× / 8 进程 2.46×**。
提取的 2.17×/2.28× 已经达到该上限的约 **93%**。因此 4 进程与 8 进程耗时几乎相同，
用 4 进程更省内存。

**配置方式**：

- WebUI：训练区「F. 运行环境」→「音频特征预提取进程数」（自动 / 关闭 / 2 / 4 / 8 / 16）
- 命令行：`--prewarm-workers auto|off|N`（默认 `auto` = min(8, 核数−2)）
- 单独跑：`python -m maichart.training.prewarm_audio <manifest> --cache-dir <cache> --workers auto`

**安全性**：缓存按「音频内容哈希 + 特征配置」寻址，已算过的歌直接跳过（重复执行零成本）；
失败的歌只记录不中断（训练读到时才会暴露）。并发写入使用唯一临时文件名 + 原子替换，
两个进程同时命中同一首歌也不会写坏缓存。

### 3.9 跨窗口模型族 `event_ctx_v1`（窗口互相衔接）

`event_v1` 的一个窗口是**孤立**的：它不知道自己前面写了什么、后面是什么音乐，所以两个窗口的接缝处
经常出现"重复上一句""在小节中间重新起句"。新族在完全相同的基础上补三件事：

| | `event_v1` | `event_ctx_v1` |
|---|---|---|
| 输入 | 只有本窗口的音符 | 本窗口 + **前置上下文块**（前面 `context_beats` 拍的音符，用 `<ctx>`/`<ctx_end>` 包住） |
| 音频 | 恰好是本窗口 | 本窗口 **+ 前后各 `audio_margin_beats` 拍**（音乐是全曲已知的，"往后看"合法） |
| 窗口步进 | `segment_beats` | `segment_beats` − `overlap_beats`（>0 时为滚动重叠扫描） |
| 损失 | 窗口全部 token | **只算窗口部分**，前置上下文被掩码排除（它是给定条件，不是预测目标） |
| 词表 | 共享特殊符 | 共享 + 2 个控制符 → **checkpoint 与 event_v1 互不通用** |

**为什么"往后看"是允许的**：音频在生成时整首都在手上，只有**音符**是因果的。
所以模型能提前知道"下一段音乐变激烈了"——这正是你设想的"注意后一段可能是什么样"。

**参数怎么选**（WebUI 训练区 C 组下方，选 `event_ctx_v1` 后出现）：

| 参数 | 默认 | 建议 |
|---|---|---|
| 前置上下文拍数 | 8 | 8（= 半个窗口）。填 0 就退化成 event_v1 |
| 前后音频余量 | 4 | 4；显存/速度紧张时改 2 |
| 上下文 token 预算 | 192 | 192；若大量窗口因超预算被跳过就调小，或把 `max_tokens` 调大 |
| 滚动重叠拍数 | 0 | 先用 0。想让接缝更平滑可试 4–8（代价：窗口数成倍增加） |

> `max_tokens` 现在要同时装下「上下文 + 窗口」，所以比 event_v1 更宽裕：
> S=16 建议 `max_tokens=640`（窗口 p99 244 + 上下文 ~40）。

**命令行**：

```powershell
$env:PYTHONPATH="E:\simai_transformer\src"
# 训练
python -m maichart.training.train_event_ctx --manifest <train.json> --validation-manifest <val.json> `
  --checkpoint-dir checkpoints\ctx --segment-beats 16 --max-tokens 640 `
  --context-beats 8 --audio-margin-beats 4 --context-tokens 192 --overlap-beats 0
# 生成（会自动补上同名的上下文参数）
python -m maichart.generation.event_ctx_cli --checkpoint checkpoints\ctx\event_best.pt `
  --audio song.mp3 --output out\maidata.txt --gold-stats <gold_stats.json>
```

**实测（机制验证，非质量结论）**：2 步玩具模型 5 个窗口生成时报告的上下文 token 数为
`[0, 24, 37, 17, 34]`（首窗口无前置，其余都带上了自己前面写过的音符）；同一命令加
`--context-beats 0` 后全部为 0。两族共用一个训练循环、checkpoint 格式与验收集成，
只有数据集与 `target_schema` 不同。

---

## 4. 随机种子与"抽卡"

### 4.1 机制

- **种子留空 → 服务端每次运行抽一个新种子**（`server.py::_resolve_seed`，用 `secrets`）。
- **填整数 → 完全可复现**。
- 实际使用的种子会出现在三个地方：任务面板（`seed=…（本次随机）`）、训练日志首行、以及
  checkpoint 的 `args` 字段；生成时还会写进 `quality.json` 的 `seed` / `candidate_seed`。
- `resume` 会从 checkpoint 恢复 RNG 状态，此时 `--seed` 不起作用。**抽卡必须用 `fresh`。**

### 4.2 协议

1. 固定全部超参，只让种子随机（清空种子框）。
2. 每次抽卡用**独立输出目录**：`checkpoints/event_v1/draw_a`、`draw_b`…
3. 跑 4–6 次。
4. 用 `metrics.jsonl` 的 **val_loss**（不是 train loss）与 token_acc 排序。
5. 取 top-2 真正生成谱面试听——离线指标只能排序。
6. 记下赢家的 seed，以后填回即可复现。

---

## 5. 过拟合判据

| 现象 | 判断 | 动作 |
|---|---|---|
| train↓ 且 val↓，gap 小 | 还在学 | 加 epochs |
| train↓ 但 val↑ 连续 ≥3 epoch | **过拟合** | patience→3；dropout +0.05；weight_decay→0.03；仍不行降一档模型 |
| train、val 都高且平 | **欠拟合** | 升一档模型；加 epochs；dropout→0.1 |
| val 与 train 差距一直很大 | 数据问题 | 检查划分是否按歌曲隔离、有无重复歌曲 |

⚠️ **没有验证 manifest 时，`event_best.pt` 是按训练 loss 选的，不具备泛化含义**，
上面这些判断也无从做起。

---

## 6. WebUI ↔ 命令行对照

| WebUI 字段 | CLI 参数 | WebUI 字段 | CLI 参数 |
|---|---|---|---|
| 训练 epochs | `--epochs` | 学习率 lr | `--lr` |
| batch size | `--batch-size` | 梯度累积 | `--grad-accumulation` |
| 最大事件 tokens | `--max-tokens` | segment beats | `--segment-beats` |
| d_model | `--d-model` | 注意力头数 | `--nhead` |
| 层数 | `--num-layers` | 前馈层宽度 | `--dim-feedforward` |
| dropout | `--dropout` | 权重衰减 | `--weight-decay` |
| 早停耐心 | `--patience` | 每隔多少 epoch 保存 | `--save-every` |
| 训练随机种子 | `--seed` | 音频缓存上限 | `--audio-cache-gb` |
| 验证 manifest | `--validation-manifest` | checkpoint 输出目录 | `--checkpoint-dir` |
| （无界面项） | `--global-frames` | 生成时验收 | `--gold-stats` |
| 模型族 `event_ctx_v1` | `python -m maichart.training.train_event_ctx` | 生成（跨窗口） | `python -m maichart.generation.event_ctx_cli` |
| 跨窗口：前置上下文拍数 | `--context-beats` | 跨窗口：前后音频余量 | `--audio-margin-beats` |
| 跨窗口：上下文 token 预算 | `--context-tokens` | 跨窗口：滚动重叠拍数 | `--overlap-beats` |

直接命令行训练示例：

```powershell
$env:PYTHONPATH="E:\simai_transformer\src"
python -m maichart.training.train_events `
  --manifest  webui_workspace\datasets\<数据集>\training_manifest.json `
  --validation-manifest webui_workspace\datasets\<数据集>\val_manifest.json `
  --cache-dir webui_workspace\datasets\<数据集>\cache `
  --checkpoint-dir checkpoints\event_v1\draw_a `
  --epochs 30 --patience 4 --batch-size 8 --grad-accumulation 2 `
  --max-tokens 1024 --segment-beats 64 --d-model 320 --nhead 8 --num-layers 8 `
  --dim-feedforward 1280 --dropout 0.1 --weight-decay 0.01 --lr 0.0002 `
  --save-every 2 --audio-cache-gb 6 --seed 12345
```

---

## 7. 产物与进度查看

| 文件 | 内容 |
|---|---|
| `metrics.jsonl` | **每 epoch 一行** JSON：loss、token_accuracy、position_accuracy、lr、wall_time |
| `event_last.pt` | 每 epoch 重写；`LastWriteTime` 即最近一次 epoch 完成时刻 |
| `event_best.pt` | 按选择指标保存的最好模型（有验证集时才有泛化含义） |
| `event_epoch_NNN.pt` | 每 `save_every` 轮一个永久档 |
| `quality.json`（生成） | 生成诊断：`generation_status`、`note_warnings`、`seed`、`candidate_seed` |

```powershell
Get-Content checkpoints\event_v1\metrics.jsonl -Tail 3
Get-Item    checkpoints\event_v1\event_last.pt | Select-Object LastWriteTime
```

---

## 8. 常见误用

1. **不填验证集** → 早停/最优模型/学习率衰减全部失效，等于盲训。
2. **生成时把 `max_tokens` 填得比 checkpoint 上限大** → 会被服务端自动夹到上限（并告警）；
   想让上限变高只能用更大的 `max_tokens` **重新训练**。
3. **抽卡时换了超参** → 结果不可比。抽卡只变种子。
4. **resume 期待换种子** → resume 恢复 RNG 状态，种子不起作用。
5. **以为 `max_tokens` 越大越好** → 它只是上限，实际计算量由批内最长样本决定。
6. **`save_every=0`** → 只剩 last/best，训练中断后无法回到中间档。
