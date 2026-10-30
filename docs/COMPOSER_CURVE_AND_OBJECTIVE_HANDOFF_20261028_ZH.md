# Composer 曲线控制、训练指标与后续修复交接（2026-10-28）

这份文档接续 `docs/astra_review_handoff.md`。旧文档记录的是上一轮实现意图，其中部分表述与实际代码不一致。接手时以当前源码、本文的验证记录为准。

## 本轮范围

用户明确要求本轮修复曲线接线与控制、零时值 Hold、训练目标记录、损失计算中的多余开销。提前结束和候选竞争模块只提供方案，留给下一位修改。没有改训练数据，没有覆盖检查点，也没有启动、停止或恢复正式训练。

接手基线是 `4dd2c84`，其上已有另一位协作者的大批未提交改动。本轮是在这些改动上继续修复，不能把全部工作区差异算成本轮新增。主界面是 `src/maichart/webui/static/index.html` 和 `app.js`，独立 Composer 页面已弃用。

## 项目当前怎么生成一张谱

1. `composer/data.py` 把真实谱转成事件和候选拍位，为音频、历史、负荷等分支准备训练输入。
2. `grounded_model.py` 的音频编码器先处理候选附近的声音；AudioLens 提供更细的瞬态信息；选时分支决定下一个事件放在哪里。
3. 配置分支结合音频、已写出的谱面、占用状态与历史统计，依次决定同一拍的音符，再以 STOP 结束当前事件。整曲 EOS 与这个 STOP 是两件事。
4. Planner 预测较长时间块的事件率、音符率与休止，Workload 预测局部负荷。前者六通道是事件率、音符率、Hold、Slide、Touch、休止；后者五通道没有休止。前两项保存为 `log1p(rate)`。历史尾迹四通道是过去 1 秒和 4 秒的事件率、音符率，不是 Touch/Slide 比例。
5. `generate.py` 完成采样和物理冲突检查。`guard.py` 判断一个提议能不能提交。人工曲线通过 `curve.py` 调整规划输入，不直接替代音频选时器。
6. `train.py` 总 loss 是多项任务的加权和。音频配置辅助任务最近从具体音符预测改成宏观分组预测；数值下降不能直接算作模型质量提升，必须看同口径分项和固定音频产出。

## 本轮修复的意图

### 曲线应当是可理解的编辑器

原实现有两个直接接线错误：将数值 `0.5` 传给只接受 `full/trim` 的命令参数；强度滑块发送的字段没有被控制器使用。原生模型还有一个名为 `curve_strength` 的参数，它控制模型自己的负荷条件。人工曲线强度必须单独传递，不能在关闭人工控制时顺便关闭原模型能力。

此前控制器还在未经编辑的曲线上把事件率从 1 次/秒推到约 1.87 次/秒，尽管音符率仍是 4 个/秒。原因是它硬写了事件率等于音符率的 0.78 倍，改变了同拍组合。本轮移除这个固定比例，让音符密度与同拍组合分别受控。

设计约定：

- 强弱曲线控制局部音符密度。模型预测出的原始曲线和基准随 JSON 保存，未编辑的预测曲线不会重新改写各分支原有预测。
- “同拍物量倍率”单独调节平均每个事件有多少音符，默认 1。在总音符量不变时，倍率增大让事件更疏、每次音符更多；倍率减小则相反，最低一个事件一个音符。原有按钮物理上限仍由可玩性规则负责，不拿它充当包括 Touch 的物量上限。
- 休止区表示这段时间不开始新音符；此前已开始的长音可以经过。强度为 0 时，包括休止在内的人工曲线控制全部关闭。
- `full` 按整首音频时间对齐；`trim` 把整条曲线映射到去掉首尾静音后的有效区间，休止区域使用同一映射。
- 预测时传递实际等级、条件文件、BPM、起始偏移、变 BPM 表和拍号。编辑/保存不能丢失模型预测基准。

这些都是控制行为的修复，不是对模型音乐理解能力的重新训练。正式质量验收仍要固定歌曲、条件和种子比较。

这里的“未编辑保持原样”仅指 Planner、Workload、尾迹预测和默认同拍预算不被重新标定。已有的曲线参与 EOS 判断逻辑仍存在，不能据此宣称启用曲线与关闭曲线会逐字生成同一张谱。

主界面使用方法：先选择 Composer 检查点、歌曲、等级和节拍设置，再到曲线编辑页生成预测；按需拖动曲线、标记休止并保存，随后生成。以下均使用主界面名称：

| 控件 | 含义与默认使用方法 |
|---|---|
| 强度曲线指导强度 | 默认 0.65；调节人工编辑的影响。设为 0 关闭人工曲线、同拍倍率和休止标记。 |
| 学习负荷曲线引导强度 | 原模型负荷条件的强度，与上面的人工控制独立。不要为了关闭人工曲线而改此项。 |
| 曲线时间范围 | 默认“整段音频”，沿用原时间；“对齐有效音乐段”把曲线和休止一起缩放到去掉首尾静音后的范围。 |
| 同拍物量倍率 | 默认 1 保留原模型组合；例如 0.75 偏少同拍、1.25 偏多同拍。是指导目标，不保证每个事件都达到指定平均数。 |

曲线纵轴是音符密度参考，不是最终谱面定数，也不是对音乐难度的人工评级。代码采用 `bounded_odds_v1` 映射，保存基准速率和上限，避免旧映射把高峰都压成 1 后无法还原。控制器使用原预测作为编辑参照，避免不同时间窗口的预测互相覆盖。只有 `hard/strict` 模式下用户明确把同拍倍率调到 1 以下，才对实际选中事件增加包含 Touch 的总音符头预算；默认不增加硬限制或随机抽样。

### 零时值 Hold

`1h`、`1h[16:0]`、`Ch[16:0]` 是项目已接受的表示。本轮移除专门拒绝它们的规则：按有起音、无持续占用处理，保留原文本和词表身份。重复键位、同时同键、与既有长音冲突仍走正常检查。没有把所有 Hold 改写成 Tap，也没有放宽错误语法。

### 训练指标与开销

新配方必须在检查点、训练启动和验证结果中明确记录。特别要区分“完整音符分类的 CE”和“宏观分组的 CE”；后者任务更简单，不能直接与过去的约 5.68 比较。旧检查点续训若目标配方改变，应明确打印变化，而不是沿用旧标签造成误判。

实际版本字段是 `training_objective_recipe.version = composer_objective_recipe_v1_precision_audio_macro`。记录音频分组、Gaussian 参数和权重、placement 回传比例、恢复 EOS、竞争模块是否实际启用。旧配方未记录或发生变化时打印 `composer_objective_change`，没有新增禁止恢复训练的门槛。

比较历史音频动作损失：旧 `event_balanced_v1` 对照 `audio_action_atom_legacy_ce`；旧 joint 策略对照 `audio_action_atom_joint_nll`。这两项复用已算出的分组和组内信息，不额外执行一次完整词表 CE。样本、词表和权重仍须一致。`audio_action_atom_first_slot_ce` 与 `audio_action_atom_nonstop_ce` 可以分别观察首槽和真实音符预测，不能直接拿来替代旧总损失。

组数在模型构建时已经知道，不应每个 batch 再从 GPU 张量取最大值并调用 `.item()`。宏观辅助目标也不应先执行完整词表 CE 然后丢弃。本轮只消除这些重复工作，不改现有高斯监督的数学定义。

曲线的速率变换也已保留在原设备上，不再为了调整数值把模型预测整体从 GPU 取回 CPU 再传回。尾迹曲线使用固定 32 点窗口积分近似，计算量随候选数线性增加；没有引入递归搜索或额外模型前向。没有用本轮回归测试宣称新的整轮训练加速比例。

## 已完成接力修复一：长音尾端与 EOS 新音符头需求解耦（注释 1）

**已在 `src/maichart/composer/generate.py` 完成修复并单测验收。**

### 原问题根因

原 `generate.py` 以 `action_tail_sec = max(beat_sec, committed_action_end_sec)` 作为未来活动搜索起点，且设置 `near_audio_end = (action_tail_sec >= end_sec - 2.5)` 直接跳过所有未来活动检查；同时在无活动时用 `beat_sec < end_sec - 6.0`（`before_outro_window`）硬否决 EOS。
这导致：
1. 当谱面在 153 秒放置一条延伸到 160.327 秒的长 Hold（音频有效结束 160.737 秒），而 Planner 在 156 秒仍预测明显活动、EOS 概率为 0.999 时，`action_tail_sec` 落入末尾 2.5 秒窗口，跳过后续检查直接停谱，遮住另一只手在 156 秒应继续打出的新音符。
2. 当歌曲尾奏在 `end_sec - 6.0` 之前就已经真正安静时，硬窗口否决又会强行拒绝合理的自然尾奏结束。

### 实际修改方案

1. **严格拆开两套时间职责**：
   - `last_head_sec`（`beat_sec`）：唯一用于判断后面是否还应出现**新的音符头**。
   - `last_action_end_sec`（`committed_action_end_sec`）：仅用于键位持续占用、物理边界检查和 `.composer.json` 报告，绝不参与未来新音符需求搜索，也不会触发跳过检查。
2. **提取独立可单测的 [`evaluate_eos_proposal`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L563-L630) 与更新 [`_has_future_planned_activity`](file:///e:/simai_transformer/src/maichart/composer/generate.py#L525-L560)**：
   - 当 `last_head_sec >= end_sec` 或 `has_future_candidates=False` 时，以 `audio_range_exhausted` 正常结束。
   - 当模型提出 EOS（`eos_probability >= eos_threshold`）时，始终从 `last_head_sec + 0.25s` 到 `end_sec` 检查手工强度曲线（`curve_controller.has_future_activity`）、Planner（`plan_values`）与物量预测（`song_workload`，并遵守激活曲线的 `rest_mask`）。
   - 移除 `near_audio_end` 长音尾短路与 `before_outro_window` 6 秒硬否决：只要后续仍有手工曲线、Planner 或物量活动即否决 EOS（记录 `curve_activity_remaining` / `planner_activity_remaining` / `workload_activity_remaining`）；若后续确实无活动，则允许安静尾奏自然接受 `learned_eos`。
   - 中间休止段通过 `not any(playable)` 推进搜索游标 `previous = candidates[0]` 跨过，同时完整保留已提交谱面历史。
3. **五项核心验收单测**（见 [`tests/test_composer_astra_priority_fixes.py`](file:///e:/simai_transformer/tests/test_composer_astra_priority_fixes.py#L72-L162)）：
   - 长 Hold 跨过后续有音区（最后音符头 153s、Hold 尾 160.327s、音频结束 160.737s、156s 规划有活动、EOS=0.999）→ 必须否决 EOS（`planner_activity_remaining`）。
   - 真正安静的尾奏（150s 起无曲线/规划/物量活动，距 `end_sec=160.737s` 超过 10s）→ 正常接受 `learned_eos`，不强行塞满到 `end_sec`。
   - 中段静音（60s–75s）后 80s 恢复活动 → 在 60s 否决 EOS。
   - 手工曲线后半段仍有活动（即使长音尾已接近 `end_sec`）→ 否决 EOS（`curve_activity_remaining`）。
   - 无剩余合法候选时 → 以 `audio_range_exhausted` 自然结束。

## 已完成接力修复二：候选竞争模块的无效邻居隔离与邻距加权（注释 4）

**已在 [`src/maichart/composer/grounded_model.py`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L35-L141) 完成修复并单测验收。**

### 原问题根因

1. 原 `_CandidateCompetition` 虽对 `normed` 乘了 `mask_f`，但 `left`、`right`、`p_left`、`p_right` 未检查邻居自身是否有效（`left_valid` / `right_valid`）。当无效邻居的 `placement_probs` 或特征变化时，`prominence = p - 0.5 * (p_left + p_right)` 和邻居差分会把无效位置混入边缘有效候选的计算；一旦 `out_proj` 学出非零权重就会污染有效候选得分。
2. 候选数组下标相邻不等于时间/节拍等距（`1/16` 与 `1/12` 拍紧邻，而远距稀疏候选可相隔数拍），且原 `workload_audio[..., :28]` 只是固定秒窗的前 28 维切片，不含显式邻距信息。

### 实际修改方案

1. **显式有效邻居掩码与零泄漏保护**：
   - 计算 `left_valid[:, 1:] = valid_b[:, 1:] & valid_b[:, :-1]` 与 `right_valid[:, :-1] = valid_b[:, :-1] & valid_b[:, 1:]`。
   - 对输入 `audio_context`、`workload_audio`、`placement_probs`、`candidate_time` 先用 `torch.where(mask, ...)` 屏蔽无效候选，再仅在 `left_valid` / `right_valid` 为真时计算 `diff_left`、`diff_right` 与 `prominence` 邻居均值。
   - 当某有效候选无有效左/右邻居（`w_sum <= 1e-6`）时，其邻居差分与 `prominence` 严格为 `0`，绝不凭空造峰。
2. **显式拍距/秒距特征与指数距离衰减权重**：
   - 从 `candidate_features[..., audio_dim:]` 传入候选时间/节拍特征 `candidate_time`（包含 `delta_beats` 与 `delta_sec`），计算相邻有效候选的真实拍距 `gap_beats` 与秒距 `gap_sec`。
   - 按 `w = exp(-gap_beats / 0.25)` 对 `diff_left`、`diff_right` 及 `prominence` 中的邻居均值做距离加权归一化：紧邻细网格（如 `1/16` vs `1/12` 拍）强竞争，跨拍稀疏候选平滑衰减。
   - 保持 `in_proj` 输入维度 `width * 3 + 33` 与 `out_proj` 零初始化残差契约不变（兼容全部现有 `v3` 检查点）：其中 28 维统计特征由 14 维短窗-中窗局部对比（`0.5s - 2.0s` 动态与频带）、6 维短窗动态及 8 维显式邻距几何特征（`log1p(gap_beats_left/right)`、`log1p(gap_sec_left/right)`、`w_left/right`、`left/right_valid`）组成。
3. **非零 `out_proj` 不变性单测**（见 [`tests/test_composer_astra_priority_fixes.py`](file:///e:/simai_transformer/tests/test_composer_astra_priority_fixes.py#L233-L273)）：
   - 将 `out_proj` 权重与偏置设为随机非零值后，大幅改动所有无效候选（`~valid`）的 `audio_context`、`workload_audio`、`placement_probs` 与 `candidate_time`，验证所有有效候选输出逐元素完全不变（`atol=0, rtol=0`），且孤立有效候选（无有效邻居）不会因自身 `placement_probs` 变化而凭空产生显著度峰。

## 其余待办修复与后续验证说明

### 1. 起音中心 `PlacementTargets.proximity` 逐通道高斯对齐（已修复）

- **问题**：[`PlacementTargets.query`](file:///e:/simai_transformer/src/maichart/composer/audio_lens.py#L159-L178) 合并 `±25ms` 内所有音符类型的占用标志，而原 [`PlacementTargets.proximity`](file:///e:/simai_transformer/src/maichart/composer/audio_lens.py#L180-L215) 仅取最近单个事件的 `flags[nearest_idx]`，导致附近同时存在 Tap 与 Hold 时可能出现 Hold 占用为 1、但 Hold 高斯中心目标为 0 的矛盾。
- **修复**：`PlacementTargets.proximity` 现枚举每个查询点 `±25ms` 窗口内的全部可信事件，对 5 个通道分别取最大高斯核响应 $\max_i \exp(-\frac{1}{2}(\Delta t_i / \sigma)^2) \cdot \text{flag}_{i,c}$，保证每个非零占用通道都对齐到该通道最近的真实起音中心。
- **后续架构演进方向**：当前占用 BCE 与高斯居中项仍作用于同一组 `placement` logits；后续若需进一步解耦，可在保持轻量前提下尝试独立的标量起音中心头或邻近候选排序目标。

### 2. 恢复构建器 `RecoveryBuilder` 可达性统计累加与三阶段细分（已修复）

- **问题**：原 [`RecoveryBuilder.build`](file:///e:/simai_transformer/src/maichart/composer/recovery.py#L221-L315) 在每次调用开头重置 `self.playability_unreachable_references = 0`，导致批级诊断只能保留最后一条样本的值。
- **修复**：移除 `build()` 内的清零操作，使 `playability_unreachable_references` 在 builder 生命周期内正确累加；同时新增 `reference_reachability_counts`（并在 `batch_from_episodes` 诊断中输出 `reference_reachability`），显式区分 `generator_unreachable`（产生器未产出）、`prefilter_masked`（预过滤屏蔽）、`playability_filtered`（生成期可玩性过滤屏蔽）与 `reachable`（完全可达）四个阶段。

### 3. 指标与质量验收

继续保留宏观/细节 CE 分解，但比较时明确数据、掩码、STOP 权重和目标版本一致。闭环谱面至少检查提前结束、实际拍位/相邻拍距分母、同键/ABA/ABAB 重复、Touch 位置和同拍数量。文本 `{12}`、`{24}` 的出现次数不是三分音事件比例。

对这些指标的改进不能直接等同于音乐质量提升。固定歌曲、条件、种子和人工听谱仍不可省略；但明确的实现缺陷应先修复，不必等待主观评谱才能动手。

## 本轮最终实现与统一测试

实现入口：

- `src/maichart/composer/curve.py`、`generate.py`：独立人工强度、时间映射、参考预测、事件/物量分配、总头预算，以及解耦长音尾的 `evaluate_eos_proposal` / `_has_future_planned_activity`。
- `src/maichart/composer/grounded_model.py`：`_CandidateCompetition` 显式有效邻居掩码、拍距/秒距特征与指数距离加权。
- `src/maichart/composer/audio_lens.py`、`recovery.py`：`PlacementTargets.proximity` 逐通道最大高斯响应；`RecoveryBuilder` 跨样本累加 `playability_unreachable_references` 与分阶段 `reference_reachability` 诊断。
- `src/maichart/webui/composer_api.py`、`server.py`、`static/app.js`、`static/index.html`：主界面控件、快照、预测条件和真实 CLI 参数传递。接口版本为 `composer_generation_controls_v12_independent_manual_curve`。
- `src/maichart/composer/guard.py`：零 Hold 放行及无持续占用。
- `src/maichart/composer/objectives.py`、`conserved_actions.py`、`train.py`、`checkpoint.py`：缓存组数、跳过废弃计算、可比指标和训练配方。

新增与扩展行为测试位于 `tests/test_composer_astra_priority_fixes.py`、`tests/test_composer_curve_control_contract.py`、`test_composer_curve_webui_contract.py`、`test_composer_curve_controls.js`、`test_composer_zero_hold_guard.py`、`test_composer_objective_recipe.py`。

最终统一测试：

- Python：**942 passed，14 skipped**，约 65 秒。范围包括 Composer、EOS 五场景与长音尾解耦、候选竞争非零权重无效邻居隔离、逐通道高斯对齐、恢复可达性统计累加、曲线接口/生成、条件复用、旧 event 曲线契约、方言零 Hold 和预取；使用项目 Python 3.12，隐藏 CUDA，以免占用正式训练资源。
- JavaScript：**162 项全部通过**，执行了 `tests` 下全部 `.js` 测试。
- 真实主页面烟雾检查：在无头 Edge 加载完整 HTML/JS，模拟后端接收生成请求。人工强度 0、原生强度 1.3、`trim`、同拍倍率 0.75 和原预测基准全部保留，无页面 JS 错误。该检查没有真的启动生成；后端参数到实际 CLI 解析、预测条件路由由 Python 测试另行覆盖。
- 损失测试同时覆盖完整/打包槽位的数值与梯度一致、宏观路径只调用一次分组 CE、禁止热路径调用 `.item()`、新指标与完整原子损失一致。
- 初次统一测试发现旧接口版本断言、旧 JS 测试夹具没有跟随新增依赖；已更新夹具，并修复真实服务状态/前端提交检查仍停留在 v11 的遗漏，最终以上测试通过。

复现命令（PowerShell，环境变量只影响此终端及其子进程）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:PYTHONUTF8='1'
& .scratch/python312_runtime/Scripts/python.exe -m pytest tests -k 'composer or intensity_curve or curve_condition or form_curve_condition or event_cli_curve_contract or dialect_shorthand or layout_prefetch' -q -p no:cacheprovider
$tests = (Get-ChildItem tests -Filter '*.js').FullName
& node --test @tests
```

本轮不涉及正式权重训练或真实歌曲的音乐质量验收。后台旧 WebUI 进程不会自动重载所有 Python 模块；使用新控件需要加载更新后的服务，不要为此打断仍需保留的训练监控进程。

---

## 2026-10-29 后续更新：异常 `{32}`/`{48}` 切分、大段空白与历史自感知（对比 `event_ctx`）修复

针对 `4a69b3df555b/maidata.txt` 中出现的 `{32}` 二连抽搐（`1/8` 拍 + `3/8` 拍）、`{48}` 跨网格微步长、`[32:3]` 畸形 Hold 时值、曲中/曲尾大段空白，以及模型局部前导历史自感知能力（对比 `event_ctx`）的完整根因排查、修复思路与架构实现，详见专篇交接文档：
- [`docs/COMPOSER_RHYTHM_BLANK_AND_HISTORY_HANDOFF_20261029_ZH.md`](file:///e:/simai_transformer/docs/COMPOSER_RHYTHM_BLANK_AND_HISTORY_HANDOFF_20261029_ZH.md)

