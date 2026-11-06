# 前向物理约束与无理审计接入（2026-10-16）

本记录说明本轮为 song_form_v2 生成侧做的改动、依据和实际测量。**没有修改训练代码、损失函数或检查点合同**，正在运行的训练（`Prism_Re_v4`，epoch 4 layout_revision）不受影响，也不需要停止以获得这些改动。

## 1. 判断依据：缺口在哪里

项目里物理/无理层早已齐全，但**全部只在生成结束后当检查器运行**：

| 模块 | 作用 | 改动前的使用方式 |
|---|---|---|
| `generation/muri/runtime.py` | 180Hz 动作与 Slide 模拟器（移植自 MiaCode 6573efb） | 仅生成后 |
| `generation/muri/static.py` | 静态无理引用 | 仅生成后 |
| `generation/maimai_physics.py` | 转盘几何、左右手可达 | 仅被检查器调用 |
| `generation/sustain_state.py` | 精确算出某拍点被占用的 pad | 仅作训练输入特征 |
| `form/local_load.py` | 解码期因果多押约束 | 已在解码循环内 |

解码循环里唯一的前向约束是 `LocalChordGuard`，它只统计"同一瞬间押了几个键"，**完全不知道 Hold / Slide 会把键位占住一段时间**。因此模型写完 `1h[4:4]` 后立刻在同一键位写 Tap，此前没有任何机制阻止。

这就是被提议的"前向约束"，也是可行的原因：需要的因果原语（已提交前缀、精确拍点时钟、转盘几何）都已存在，且 `LocalChordGuard` 已经验证了接线路径（`event(beat)` 打分 → `commit(text, beat)` 更新状态 → 通过 `TypeGuidance._EventBias` 注入 logits）。

## 2. 新增：解码期因果物理约束

新文件 `src/maichart/form/physical_guard.py`，类 `PhysicalFeasibilityGuard`。约束两类事实，**只使用已提交事件**，因此在每一步解码都可查询，无需知道未来：

- **占用**：Hold 在其时长内占用自身键位；Slide 占用头、尾及路径键位。窗口内的后续音符按不到该键；原生检查器把这一族报告为 `Overlap` / `TapOnSlide`。
- **移动**：跟踪两只手的代表位置（取该手按键集合的 medoid），由 `MAX_HAND_SPEED = 10.0`（归一化台面单位/秒，A 环半径为 1.0，横穿全台 2.0 单位需 0.2 秒）计算所需速度。单只手一次覆盖的键位跨度上限为 `MAX_HAND_SPAN = 0.90`（相邻键 0.765，次近 1.414），因此五指多押会被拆到两只手上，而不是当作一只宽手。

**关键设计：软约束，永不删除候选。** 惩罚有界（占用 8.0、移动 4.0、合计上限 12.0，再乘强度），只加到候选 logits 上。理由是这个项目已经踩过空候选集的坑——`_NoteGrammar.head_closure` 的注释记录了"可达但无法闭合"状态会变成空 mask 并导致硬报错。硬禁止会把一个可疑音符变成解码失败，同时用黑名单代替模型判断。强度 0 完全关闭该约束。

惩罚经 `_EventBias` 的**势差**注入（与多押项相同的望远镜式结构）：只有当续写之间仍有差异时才起引导作用，一旦所有续写共享同一惩罚即自动抵消。

### 接线点

| 文件 | 改动 |
|---|---|
| `form/physical_guard.py` | 新增模块（占用、双手移动、有界惩罚、报告） |
| `form/type_guidance.py` | `TypeGuidance(physical_strength=...)`；`_EventBias._physical_potential`；`commit` 转发；报告新增 `physical_feasibility` |
| `form/rational_generate.py` | `generate_rational_chart(physical_strength=...)`；`event_bias` 注册拍点时钟；新增无理审计 |
| `form/cli.py` | `--physical-feasibility-strength`；校验；写入生效设置 |
| `webui/form_api.py` | 参数透传；`form_muri_audit` 日志前缀分发 |
| `webui/static/index.html` | 控件「物理可行性软约束」，默认 0 |
| `webui/static/app.js` | 两处 payload 与状态字段 |

## 3. 新增：生成后无理审计

在同一方法论的另外一半——既然已有模拟器，就应当让它"玩"模型刚写出的谱。`generate_rational_chart` 现在把成品重新编译后送入 `generation/miacode_python.analyze_maidata`，结果写入 `.form.json` 的 `muri_audit` 并打印 `form_muri_audit` 行。

审计**只读、不修改谱面**，且异常绝不导致生成失败（失败时记为 `applied: false` 并给出原因）。单曲实测耗时 0.43 秒。

## 4. 在真实产出上的测量

用 `.scratch/physical_guard_replay.py` 回放已生成的真实谱面，统计护栏在解码时会上报什么：

| 谱面 | 事件数 | 占用冲突 | 移动超速 |
|---|---:|---:|---:|
| `fdee2c332443` | 378 | 2（0.53%） | 5（1.32%） |
| `501f78e35b62` | 363 | 2（0.55%） | 2（0.55%） |
| `2a9665ec6a16` | 436 | 4（0.92%） | — |

也就是说，当前模型约每 100–200 个音符会写出一个**物理上不可能**的音符（键位仍被 Hold 占用），另有约 0.5–1.3% 的事件要求人手移动快于人手上限。数量不大，但每一个都是玩家会直接感到"这是无理"的硬伤，值得在解码期消除。

同一批谱面的无理审计给出更宏观的画面，例如 `fdee2c332443`：18 条诊断（MultiTouch 13、Overlap 2、TapOnSlide 1、SlideHeadTap 1、SlideTooFast 1），另有 11 条静态引用。**MultiTouch 是最主要的可玩性问题来源**，这也是把审计接进报告的价值——此前生成报告完全没有这一项，无法判断改动是否有效。

## 5. 验证

- 新增 `tests/test_form_physical_guard.py`，14 项通过。
- 完整 Python 回归：**2069 通过，12 跳过**（本轮改动前为 2039 通过）。
- 全部 JS 套件通过（含新控件负载）。

本轮测试发现并修复了三个真实缺陷（均由新测试或回放发现，非推测）：

1. `MAX_HAND_SPAN` 在重写模块时丢失，`_within_span` 直接 `NameError`。
2. `_physical_potential` 把 8 宽布尔掩码传给只接受键位号的标量入口，元素被逐位当成键位（索引 4 的 `True` 变成 1 号键）——已改用矩阵入口。
3. `assign_hands` 会把相距 2.0 单位的和弦塞给**一只手**，因为未放置的手代价为 0，且当时不检查跨度；已改为跨度优先、其次移动、再次均衡的排序。

另修复一处先前遗留问题：`tests/test_training_input_payload.js` 的注入器未提供 `songFormRationalGridDivisions`，导致 4 项测试长期失败（经 git stash 确认与本次改动无关）。已在注入前缀中补上，现 16 项全通过。

## 6. 建议的默认值

`physical_feasibility_strength` 默认 **0**（保持既有行为）。建议试验值 **1.0**，无理明显偏多时提高到 2.0–3.0。它与既有的「近似常见分音容差」「拍位分母上限」互补：那两项管**时间是否合理**，这一项管**手能不能做到**。

## 7. 未做与不确定

- **没有**加入训练侧物理损失。训练数据来自官谱，本身物理可行；编解码能力薄弱才会产出不可行音符，而这属于解码/采样问题，生成期约束能直接覆盖。若要进入训练，需改动损失合同并重新训练，本轮未做。
- 双手位置是代表点，不是身体模拟；`playable_verified` 在报告中恒为 `false`，审计是参考意见而非认证。
- Slide 占用目前按其头键与目标键在时长内计算，未逐段展开扫过路径；精确扫掠仍由生成后的 MiaCode 检查器负责。
- 仍过时的文档列表见 `docs/explanations/QUALITY_SPEED_REVIEW_20261015.md` 第 7 节。
