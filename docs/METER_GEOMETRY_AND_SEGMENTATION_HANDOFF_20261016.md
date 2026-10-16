# 小节相位几何与物理可行性：交接文档

日期：2026-10-16
分支：`codex/song-form-rebuild`
基线：`a3493d3`（本文件涉及的全部改动均未提交）

本文件面向接下来接手 WebUI 与相关工作的其他智能体。内容分四部分：
1. 分段机制的现状与评估（回答"分段依据从哪来、是否有问题"）
2. 窗口内绝对拍位问题的确认与修复（回答"是否存在该瓶颈"）
3. 物理可行性软约束与无理审计（前一轮已完成）
4. 交接事项：WebUI 待办、验证状态、风险与未做项

---

## 1. 分段机制

### 1.1 实际生效的路径

生产代码只有**一条**分段路径：`organize_song()` → `segment_music()`
（[data.py:226-233](src/maichart/form/data.py#L226)，[segmentation.py:245](src/maichart/form/segmentation.py#L245)）。

`timing.py` 里的 `build_spans()`（4 小节 / 16 拍定长窗口）**只被测试调用**，不在生产路径上。
上一轮讨论中"16 拍窗口"的说法对当前架构并不成立 —— 当前窗口由**秒**决定，不是拍。

### 1.2 两阶段结构

`segment_music()` 先找音乐区域（`MusicRegion`），再按算力上限切块（`MetricSpan`）：

**第一阶段：音频内容边界**

`_audio_novelty()` 在 1.0s 与 2.0s 两个尺度上比较左右上下文，加权为
`0.45·chroma + 0.35·timbre + 0.20·rhythm`，再取
`0.65·max(尺度) + 0.35·min(尺度)`，经 0.25s 平滑后按 **85 分位** 归一化并裁剪到 [0,1]。

随后：
- 取 `novelty ≥ 0.65` 的局部极大值作为候选；
- 贪心抑制：保留 2.0s 软最小间距内更强的那个，且距离首尾至少 2.0s；
- 通过 `_nearest_grid(prefer_bar=True)` 吸附 —— **有拍号时吸附到最近小节线**，无拍号时向下取整到整数拍；
- 与 `meter_change` 边界、`song_start`/`song_end` 合并，再执行一次 2.0s 软最小间距过滤。

**第二阶段：算力切块**

任何长于 `max_seconds`（默认 14.0s）的区域由 `_computational_chunks()` 切开，
优先落在小节线上（`resource_cap`），仅在极慢速下退化为小数拍。
`audit['resource_cap_is_musical_boundary'] = False` 明确声明这不是音乐边界。

### 1.3 评估：设计合理，但有一个真实缺陷

**合理的部分**（不要改动）：
- 边界来自内容对比，不是固定网格。这正是 `song_form_v2` 相对旧架构的核心改进。
- 算力上限与音乐边界在数据结构上分离（`regions` vs `spans`，`span_region_indices` 映射）。
- 未知拍号时不发明小节线，只记录四分之一拍相位。

**真实缺陷：拍号从不被推断。**

`meters` 只有两个来源：`--meter-map` 命令行参数，或每首歌的 `music_structure.json`
（[cli.py:68,184](src/maichart/form/cli.py#L68)）。全仓库**没有任何**拍号检测代码
（`grep -rn "detect_meter|infer_meter"` 无结果）。

后果链条：无拍号 → `meter_map` 为空 → `_nearest_grid` 退化为 `floor(beat)` →
`_metric_span` 写出 `numerator=None, beats_per_bar=None, known=False` →
`metrical_features()` 只填四分之一拍相位，第 2-7 列全零。

也就是说：**对一首没有手工提供 `music_structure.json` 的非 4/4 歌曲，系统既不知道小节在哪，也不知道拍号。**
你提到旧框架因为无法识别 12/24 分与非 4/4 拍而被推倒重来；分段这一层已经解决了"不假设 4/4"，
但**下游的拍位表示层此前又把 4/4 假设加了回来**（见第 2 节）。

### 1.4 更优雅的方案（已实现，见第 2 节）

与其改动分段，不如**切断分段与节奏表示之间的耦合**。
`exact_fraction_v2` 做的就是这件事：小节相位来自拍号表（`MeterChange.beat` 按定义是小节线），
未提供拍号时退化为**锚定在绝对拍 0** 的四拍平滑基，并置 `meter_known=0` 让模型自行决定信任程度。

这样：
- 有 `music_structure.json` 的歌曲获得真实小节相位；
- 没有的歌曲仍然获得**跨分段一致**的坐标（这本身已消除最严重的缺陷）；
- 分段算法完全不用改，缓存不失效。

**后续可选项（未做）**：从音频自相关/谱通量周期推断拍号，写入 `music_structure.json` 的缺失字段。
这会改变 `structure_sha256` 缓存身份，需要重建准备缓存，因此本轮未做。

---

## 2. 窗口内绝对拍位问题

### 2.1 你的朋友的判断是对的，而且比描述更严重

原 `_coordinate_row(beat, origin)`（[rhythm_geometry.py:19](src/maichart/form/rhythm_geometry.py#L19)）
的 8 个通道全部由 `delta = beat - origin` 导出，其中 `origin` 是**分段起点**：

```
0: delta / 64                       距离分段起点的带符号线性量
1: copysign(log1p(|delta|)/scale)   距离分段起点的带符号对数量
2: |delta| / 64                     距离分段起点的绝对值
3: frac = delta - floor(delta)      拍内相位
4: sin(2π·frac)                     拍内相位
5: cos(2π·frac)                     拍内相位
6: sin(0.5π·delta)                  四拍谐波，锚在 delta = 0
7: cos(0.5π·delta)                  四拍谐波，锚在 delta = 0
```

实测：一个 **24 分音符落在第 1 拍**（绝对拍 7/6）与**落在第 2 拍**（绝对拍 13/6），
通道 3/4/5 **完全相同**（都是 `1/6`），只有通道 0/1/2（距离分段起点的量）和
通道 6/7（四拍谐波的两个相位）不同。

更严重的问题在通道 6/7 的**锚点**：`sin(0.5π·delta)` 的周期是 4 拍，相位零点是 `delta = 0`，
也就是**分段起点** —— 一个由秒数算力上限决定的、与音乐无关的位置。

于是产生三个后果（实测确认）：

- **同一音乐位置在不同分段里得到不同坐标**（最严重）。分段起点由 14s 算力上限决定，
  与音乐无关；起点一变，同一音符的 8 个通道全部改变。
- **周期被硬编码为 4 拍**。四拍谐波只在标准 4/4 下碰巧等于小节相位；
  3/4、7/8 或任何变速段里它都不是小节线，而拍号表完全没有参与。
- 连带效应：拍 1 与拍 5（相邻两小节的同一位置）在通道 3-7 上重合。
  注意这一点在 v2 中是**刻意保留**的正确行为（见 2.2），
  v1 的问题不在于"重合"，而在于它重合**所依据的锚点是任意的**。

这与架构初衷（不假设 4/4、不假设二进制分音）直接矛盾。

### 2.2 修复：`exact_fraction_v2`

新增 profile，13 通道（[rhythm_geometry.py](src/maichart/form/rhythm_geometry.py)）：

| 通道 | 含义 | 是否依赖分段 |
|---:|---|---|
| 0 | `delta / 64` | 是（局部尺度） |
| 1 | 带符号 `log1p(\|delta\|)/scale` | 是（局部尺度） |
| 2 | 拍内相位 `beat - floor(beat)` | **否，由绝对拍导出** |
| 3,4 | `sin/cos(2π·拍内相位)` | 否 |
| 5 | **小节相位** `(beat - meter.beat) mod beats_per_bar) / beats_per_bar` | 否 |
| 6,7 | `sin/cos(2π·小节相位)` | 否 |
| 8 | 组相位（拍号 grouping） | 否 |
| 9,10 | `sin/cos(2π·组相位)` | 否 |
| 11 | `meter_known` | 否 |
| 12 | `grouping_known` | 否 |

关键点：
- 拍内相位改为从**绝对拍**取 `beat - floor(beat)`。若仍从 `delta` 取，
  一个起点落在拍中的分段会让整张网格平移。
- 小节相位以 `MeterChange.beat` 为原点 —— 该字段按定义就是小节线。
- 无拍号时退化为锚定在**绝对拍 0** 的四拍先验，并置 `meter_known=0`。
  这仍然消除了"随分段浮动"，同时用标志位告诉模型不要当真。

实测（`causal_prefix_geometry`，4/4，origin=0）：

```
24 分在第 1 拍 (7/6):  拍内相位 0.166667  小节相位 0.291667  sin/cos (±0.966, -0.259)
24 分在第 2 拍 (13/6): 拍内相位 0.166667  小节相位 0.541667  sin/cos (∓0.259, -0.966)
```

两者**共享拍内相位**（都是 24 分），**在小节相位上区分**（拍 1 与拍 2）——
正是你朋友描述的"两个类别"，但类别轴是小节位置而非窗口距离。

跨分段与跨小节不变性：
- 拍 16 起的 24 分（绝对 33/2），分段起点 14 与 16 给**完全相同的通道 2-12**；
- 拍 0.5 与拍 8.5 的 24 分给**完全相同的通道 2-12**（小节相位每小节重复）。

### 2.3 改动的文件与理由

| 文件 | 改动 |
|---|---|
| [timing.py](src/maichart/form/timing.py) | 新增 `meter_phase()`（单次测量）与 `MeterPhaseIndex`（带活跃拍号窗口缓存的查询器） |
| [rhythm_geometry.py](src/maichart/form/rhythm_geometry.py) | 新增 `GEOMETRY_DIM_V2=13`、`GEOMETRY_PROFILES`、`geometry_dim()`、`_coordinate_row_v2()`；`coordinate_features`/`CausalGeometry`/`causal_prefix_geometry`/`event_timing_targets`/`timing_targets` 全部接受 `profile=` 与 `meters=` |
| [model.py](src/maichart/form/model.py) | `FormConfig.geometry_dim` 属性；三处硬编码 `8` 改为 `config.geometry_dim` |
| [checkpoint.py](src/maichart/form/checkpoint.py) | 新增 `RATIONAL_METER_GEOMETRY_INPUT_CONTRACT`（`song_form_inputs_v4`）、`_geometry_contract()`、`V1_TO_V2_CHANNELS`、`migrate_geometry_projection()`；`load_finetune_model()` 支持 v1→v2 通道迁移 |
| [data.py](src/maichart/form/data.py) | 所有几何构造点传入 `profile` 与 `meters`；新增 `result['_meters']` 供解码期复用 |
| [rhythm_decode.py](src/maichart/form/rhythm_decode.py) | `DecodeGeometry(model, codec, origin, meters=...)`，复用单个 `MeterPhaseIndex` |
| [generate.py:250](src/maichart/form/generate.py#L250) / [rational_generate.py:233](src/maichart/form/rational_generate.py#L233) | 解码期传入同一份拍号表 |

### 2.4 契约与迁移

- **v1 checkpoint 完全不受影响**：`exact_fraction_v1` 的 8 通道行为逐位未变，`load()` 按原合同加载。
- **v1 → v2 可迁移**：`load_finetune_model()` 用 `V1_TO_V2_CHANNELS = ((0,0),(1,1),(3,2),(4,3),(5,4))`
  复制语义相同的通道，新的小节/组相位通道从零开始学。已训练的模型不会白跑。
- **v2 → v1 被拒绝**（`cannot shrink a geometry contract`），因为会丢弃已学通道。
- **resume 永远继承** checkpoint 的 profile；fintune 需要显式指定更宽的契约。
- **fresh 的 `auto` 现在选 v2**（此前选 v1）。这是本次唯一改变默认行为的地方。

### 2.5 未验证的一项（需要你决定）

v2 需要拍号表才有真实小节相位。当前**没有任何自动拍号推断**，
所以一首没有 `music_structure.json` 的歌会退化到"绝对拍 0 锚定的四拍先验"。
这已经比 v1 好（跨分段一致），但**不等于识别 3/4、7/8**。

如果你希望在没有手工拍号时也能拿到真实小节相位，需要新增音频拍号推断并写入
`music_structure.json`。这会改变 `structure_sha256`，**导致准备缓存失效并需要重建**。
本轮没有做，因为你说过不要启动数据任务。

---

## 3. 物理可行性软约束与无理审计（前一轮）

详见 [PHYSICAL_FEASIBILITY_AND_MURI_AUDIT_20261016.md](docs/PHYSICAL_FEASIBILITY_AND_MURI_AUDIT_20261016.md)。摘要：

- 新增 [physical_guard.py](src/maichart/form/physical_guard.py)：解码期因果占用 + 双手移动约束，
  **只加有界惩罚、从不删除候选**（硬禁止会产生空候选集，这是项目已踩过的坑）。
  占用 8.0 / 移动 4.0 / 上限 12.0，乘以强度。触摸不参与（无实体按键）。
- 新增生成后无理审计：`generate_rational_chart` 调用 `analyze_maidata`，
  结果写入 `.form.json` 的 `muri_audit` 并打印 `form_muri_audit`。只读、异常安全、单曲约 0.43s。
- 实测：占用冲突 0.5-0.9% 事件，移动超速 0.5-1.3%；
  `fdee2c332443` 的 18 条诊断中 **MultiTouch 占 13 条**，是最主要的可玩性问题来源。
- CLI/WebUI：`--physical-feasibility-strength` / 「物理可行性软约束」，默认 0，建议先用 1.0。

---

## 4. 交接事项

### 4.1 WebUI 待办（交给其他智能体）

已完成并测试：
- 「精确节奏几何」下拉新增 `exact_fraction_v2（小节相位，推荐）` 选项；
- `songFormRhythmGeometryUiState()` 重写为基于 profile 排序的状态机
  （`none < v1 < v2`），支持 v1→v2 前向迁移、拒绝收窄、resume 锁定；
- 帮助文案按 profile 分支说明。

待办：
1. **物理可行性控件与几何下拉的样式统一** —— 上一轮已把「物理可行性软约束」做成
   与其他控件一致的数字输入，但如果你希望改成下拉/滑条，`formPhysicalFeasibilityStrength`
   是唯一入口（`app.js` 两处 payload + `generation_options` 块）。
2. **`muri_audit` 的展示** —— 后端已写入 `.form.json`，前端尚未渲染。
   建议在生成结果区显示诊断计数（尤其 MultiTouch），并在谱面预览里定位到对应音符。
3. **`docs/QUALITY_SPEED_REVIEW_20261015.md` §7 列出的过时文档** 尚未更新。

### 4.2 验证状态

- 完整 Python 套件：**2080 通过，12 跳过**（本次改动前为 2069；新增 11 项）。
- 全部 14 个 JS 套件通过。
- 新增/更新的测试：
  - [test_rhythm_geometry.py](tests/test_rhythm_geometry.py)：9 项 v2 测试，
    含"24 分在第 1 拍 vs 第 2 拍"的缺陷对照、跨分段不变性、跨小节重复、因果性与 rewind 重放。
  - [test_form_rhythm_geometry_contract.py](tests/test_form_rhythm_geometry_contract.py)：
    v1→v2 通道迁移守恒、收窄拒绝、fresh/resume/finetune 的 profile 解析。
  - [test_rhythm_geometry_training_ui.js](tests/test_rhythm_geometry_training_ui.js)：UI 状态机。

### 4.3 已知风险与未做项

1. **v2 的第一层通道 0/1 仍然是分段相对的**。这是刻意的（局部尺度信息有用），
   但意味着模型看到的不是纯粹的"音乐内在"表示。若后续发现模型依赖这两通道做
   不合理的推断，可考虑再压缩其幅度。
2. **本轮没有启动任何训练或数据任务**。v2 未被任何 checkpoint 训练过；
   要生效需要 fresh 训练或从 v1 finetune 迁移。
3. **没有自动拍号推断**（见 2.5）。
4. **全部改动未提交**。当前工作树 20 个文件修改 + 3 个新增文件。
5. **物理守卫的双手模型是代表点近似**，不是身体模拟；报告中 `playable_verified` 恒为 `false`。

### 4.4 建议的下一步顺序

1. 用 v2 跑一次训练（见 §5 的参数建议）。
2. 把 `muri_audit` 的 MultiTouch 逐条落到谱面位置上，判断哪些是合法双押被误判 ——
   这决定该调物理约束强度还是该调训练。
3. 若需要真正的非 4/4 支持，再考虑音频拍号推断（代价：准备缓存重建）。

---

## 5. 拍位均衡损失（本轮新增）

### 5.1 问题

原 token 交叉熵对每个 token 取平均，而一个事件的开销由**拼写长度**决定。
实测一份普通谱面的 token 组成：

```
position_digit   31.3%      note_char   30.1%
structure        26.5%      note_marker 12.0%
每个事件的拍位 token 数: [1, 3, 3, 3, 3, 3, 1, 5, 3, 1]   (最大 5, 均值 2.6)
```

即 `123/128` 这类长分母会被监督得比一位整数重 **1.92 倍**，
训练会偏向拼写长的事件。这正是你最初报告"超大分音"的那一类行为。

### 5.2 修复

`position_event_loss()`（`rhythm_geometry.py`）先对每个事件的拍位拼写取平均，
再对事件取平均，让一个事件算一份监督。**只重组拍位拼写**，音符文本仍按原 token 损失学习。

**该函数此前存在形状错误、从未被调用过**：`boundary` 用 `(B,)` 的输入切片与 `(B, L)`
的 target 比较，把事件计数器广播成 `(B, L)`，逐事件分组被静默破坏。
已修复，并新增 4 项测试与一个独立参考实现交叉验证（含批内独立性与 padding 不泄漏）。

### 5.3 接线

| 位置 | 说明 |
|---|---|
| `--position-event-loss-weight` | 新增训练参数；fresh v2 默认 **0.25**，resume 继承原值（未记录为 0），v1 固定 0 |
| `train.py` | `layout_loss(..., position_event_weight=)`；新增 `position_event_aux_loss` 指标与 `position_event` 能力计数 |
| `form_api.py` | `RESUME_ARGUMENT_KEYS` 与 values 块加入该字段 |
| `index.html` | 新增「拍位均衡损失权重」控件 |

resume 不叠加新目标：改变已保存检查点的损失尺度会让优化器状态失配。

---

## 6. 训练参数建议

现有最佳两次运行（只读检查自 `checkpoints/`）：

| | Prism_Re_v4 | Prism_Re_v5 |
|---|---|---|
| d_model / layers | 320 / 6 | 384 / 8 |
| batch × accum | 4 × 8 | 4 × 8 |
| loader_workers | 4 | 0 |
| 到达 epoch / steps | 6 / 7698 | 4 / 21992 |
| 几何合同 | exact_fraction_v1 | exact_fraction_v1 |

数据集：`manifests/training_manifest.json` 共 **1525 首**（难度槽 2-5 各约 1430+）。

**推荐做法：从现有 v1 检查点 finetune 迁移到 v2，而不是从头训练。**
理由：`load_finetune_model()` 会按 `V1_TO_V2_CHANNELS` 复制语义相同的 5 个通道，
已学会的布局能力全部保留，只有小节/组相位通道从零开始学。
从头训练会丢掉这些，而 v2 的修复不需要那样做。

```
python -m maichart.form.train --manifest manifests/training_manifest.json --checkpoint-dir checkpoints/Prism_Re_v6 --init-checkpoint checkpoints/Prism_Re_v5/form_last.pt --model-family song_form_v2 --rhythm-geometry-profile exact_fraction_v2 --position-event-loss-weight 0.25 --rhythm-loss-weight 0.1 --event-decision-loss-weight 0.25 --d-model 384 --layers 8 --nhead 8 --max-tokens 2048 --max-memory-tokens 640 --batch-size 4 --grad-accumulation 8 --lr 0.0002 --weight-decay 0.01 --dropout 0.1 --epochs 4 --loader-workers 2 --device cuda
```

要点：
- **`--model-family song_form_v2`**：默认已是它，但迁移时显式写出更安全。
- **`--checkpoint-dir` 必须是新目录**：训练器拒绝覆盖来源模型。
- **`--epochs 4`**：迁移只需让新通道收敛，不需要重跑全程。
- **`--loader-workers 2`**：v4 用 4、v5 用 0。数据准备是主要瓶颈，
  2 是兼顾内存与吞吐的中间值；显存不足就降到 0。
- **`--lr 0.0002` 与 v5 一致**：迁移时不需要更小的学习率，
  因为新通道是零初始化，梯度不会冲击已学权重。
- 若显存吃紧，先降 `--batch-size` 到 2 并把 `--grad-accumulation` 提到 16
  （有效批大小不变）。

如果想从头训练（例如想同时改变数据构成），把 `--init-checkpoint` 去掉并
把 `--epochs` 提到 10。

