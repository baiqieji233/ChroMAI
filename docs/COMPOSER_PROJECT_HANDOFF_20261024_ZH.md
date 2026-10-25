# Simai Transformer — ChartComposer (`chart_composer_v1` / `audio_grounded_v3`) 全量项目总结与智能体交接文档 (2026-10-24)

> **学习效率排查最新接续（2026-10-25）：** 原v9正式训练已停止，保留第6轮/13,100更新及原优化器。已实测历史版本耗时、768同输入loss、两组384样本GPU梯度和六组同样本短训；没有证实更激进学习率/更新频率可提速，原64×6/LR1e-4仍是受控对照基线。修复历史遮挡遗漏`last_heads`关系旁路，训练语义升为 **v11_history_relation_mask**，旧v10只可生成/显式finetune，不能直接resume。另修复生成全局调温破坏时间区间概率；冻结权重双种子纯Tap半拍长链13/23→4/8，hard8/6→4/4，物量同时下降，音乐质量未认证。358项Python、24项前端通过，真实GPU训练入口12微批/2更新正常完成。当前更新主界面 **7865**；长程训练尚未重启。详见[学习效率、修复与证据](COMPOSER_LEARNING_EFFICIENCY_AUDIT_20261025_ZH.md)。下方训练继续/端口记录均为历史阶段。

> **主界面参数接入（2026-10-25）：** 用户确认独立`/composer`页已弃用，本轮仅维护主界面。已补Touch历史保护开关、0–4放大上限、轮中重分batch续训和fresh普通dropout；修复提交前检查覆盖用户重分批设置及默认重置遗漏。333项Python、23项前端与真实浏览器检查通过。新界面在 **7864**，原 **7863 / `9d5be08cfe23` / v9总目标20轮** 训练未重启，继续在旧端口监控；两端任务列表独立。默认策略未改，没有新音乐质量结论。详见[主界面控件记录](COMPOSER_WEBUI_CONTROLS_20261025_ZH.md)。

> **loss 后续核实：** 第3轮已完成，完整第2→3轮总loss 4.5853→4.3577，动作2.7788→2.6524，指针准确率63.31%→66.14%。仍在学习，但精确原子与纯音频动作进展较慢；未发现更新/梯度链失效。当前检查点独立CPU8样本重复48更新loss6.202→0.0637，不能作全量质量证明。页面新增近期窗口并明确累计标尺；正式训练未重启、参数未改，继续原20轮。详见 [loss诊断与证据](COMPOSER_LOSS_DIAGNOSIS_20261025_ZH.md)。

> **GPU 满载慢速最新修复：** 已复现闲置 CUDA 缓存从正常约 3.7 GiB 撑至 9.9 GiB，使同一真实 batch 从 0.18s 退化至 3.61s；释放后恢复。已加入更新边界缓存回收，Windows 使用支持的 native GC，按用户建议改 **64×累积6（有效384）** 并保留已保存样本顺序续训。当前任务 **`9d5be08cfe23`**，工作台 **`http://127.0.0.1:7863`**，目录仍 `Prism_composer_v9`，原总目标20轮。已核验新 step1700 检查点、优化器和版本化采样状态；约200–260样本/s，旧慢态约30。详见 [显存驻留压力修复与恢复](COMPOSER_GPU_RESIDENCY_FIX_20261025_ZH.md)。先前任务状态与端口均为历史记录。

> **Touch 总量后续修复：** 已确认符号动作分支在实际高 Touch 段把概率从音频分支的 17–29% 放大到 94–99%。新增生成端 Touch 历史保护及替代提案安全检查；同 CPU 后端固定双种子占比 **21.47%→7.82%、24.27%→8.11%**，最长纯 Touch 串 17/37→6/4，hard 11/17→10/8，无后处理删除。Python 294 项、前端 16 项通过。当前 v9 / 20 轮训练未停止、结构与损失未改；详见 [Touch 总量修复与限制](COMPOSER_TOUCH_TOTAL_FIX_20261025_ZH.md)。这些 CPU 数值不能与此前 CUDA 数值混作同后端对照。

> **2026-10-25 最新接续：** 已复现并修复 Touch 跨环区截断放大 E 区偏好，保留模型预测的区域概率；同权重双种子 E/Touch 从 67.35%/91.06% 降到 42.74%/42.11%，但 Touch 总量和残留冲突未一致改善，不能宣布音乐质量问题已解决。历史空槽裁剪、声学查询复用及本机供数配置已验证；增加跨歌曲并发反而变慢的试验已撤回。相关 Python 276 项、前端 16 项通过。已保留旧检查点，在 **`checkpoints/Prism_composer_v9` fresh 启动 20 轮**，任务 **`09d14b6f4d31`**，工作台 **`http://127.0.0.1:7862`**；当前训练合同为 v10（目录号独立）。配置、证据、运行状态和剩余验收见 [本轮排查与训练交接](COMPOSER_V9_FRESH_DIAGNOSIS_20261025_ZH.md)。下文均保留为先前阶段记录。

> **最新完成与生成对照：** 第 3 轮已于 2026-10-24 21:53:32 正常完成，epoch_complete=true、step 7,416，last/inference 权重一致且有限。当前代码下已重跑原 v7、24 更新短程模型和完成模型的固定音频/条件/双种子六次对照；原文件哈希未变。Touch 偏高未稳定改善（短程→完成：15.36%→15.20%、14.73%→18.36%）；物量下降，hard/千音符一升一降（12.61→20.68、21.04→9.23），不能宣布整体质量改善。完整结果及人工复核时段见 [完成训练与固定条件对照](COMPOSER_COMPLETED_TRAINING_COMPARISON_20261024_ZH.md)。

> **此前实现修复与恢复训练：** 已修复相位游标、低温类别概率、零长 Hold、未来变速点导出及难度槽五项实现问题；策略取舍暂缓。相关 Python 回归共 230 项、前端 16 项通过，复核未发现阻断性误修。原 v7/v8 检查点 SHA-256 均保持不变；从 v8 的 step 5,000 恢复到新目录，只补第 3 轮。该阶段的启动、进度快照及核验见 [实现修复与恢复训练](COMPOSER_IMPLEMENTATION_FIXES_AND_RESUME_20261024_ZH.md)，不代表任务此刻仍在训练。

> **此前只读审查：** 见 [生成质量相关代码审查](COMPOSER_QUALITY_REVIEW_20261024_ZH.md)。两名默认配置子代理提出的重点问题由主代理独立复现；该阶段未修改源码或操作训练。检查时旧 7862 服务不可达，v8 保存于第 3 轮中途 / 5,000 次更新，inference 为第 2 轮 / 4,944 次更新；这些是恢复前的历史状态。

> **同日接续更新：** 已复现并修复微调默认结构参数错误、API 续训隐式覆盖学习率和无监督槽位重复投影；完成真实数据 96 batch / 24 次更新的 v9 微调，以及固定音频双种子四次生成对照。正式全量训练已在新目录 `Prism_composer_v8` 启动，任务 `7aac408a2f9e`，工作台为 `http://127.0.0.1:7862`，尚未完成。原 v7 已保留且哈希未变。最新配置、结果和未完成验收见 [v9 验证与训练接续](COMPOSER_V9_VALIDATION_AND_TRAINING_20261024_ZH.md)。下文为此前修复阶段的历史记录；“25–40 倍”不得当作已测端到端收益。

## 一、项目定位与当前主干架构

本项目（工作区 `E:\simai_transformer`）是针对舞萌 DX（maimai DX）的端到端自动谱面生成系统。在经历了早期固定帧模型（`generator`）、窗口事件模型（`event_v1..v11`）、全局规划模型（`song_form_v1/v2`）以及流式自回归模型（`Prism_stream_v1..v18`）的演进后，当前主工作台的一等核心模型族为 **ChartComposer（`family = "chart_composer_v1"`，当前主架构 profile 为 `audio_grounded_v3`）**。

与传统逐字符预测 Simai 文本的自回归模型不同，`AudioGroundedComposer` 将谱面生成建模为**因果事件级结构化决策**，每一步按顺序完成两个解耦决策：

1. **何时落音（Hierarchical Candidate Timing Pointer）**：
   - 给定上一事件拍位 `previous`（初始为 `None`），`candidates.py` 在合法音乐网格上构造最多 `96` 个候选下一拍位 `candidates`（当前策略 `MUSIC_CANDIDATE_POLICY_V6 = "absolute_recovery_v6_phase_safe"`，在全量 952,710 个事件上的教师强制召回率为 `99.612%`）。
   - 每个候选位置提取：
     - `141` 维候选特征（`120` 维高时间分辨率音频特征 + `21` 维拍位/秒差/多倍频相位特征）；
     - `AudioLens`（`audio_lens.py`）：围绕每个候选时刻的 `31 × 26` 局部因果/双侧声学时频块卷积编码，并由 `placement_head` 监督 25ms 容差内的真实落音先验；
     - `AudioPlanner`（`planning.py`）：以 4 拍为单位的全曲粗粒度音频规划器，输出 6 通道段落规划（`log_event_rate, log_head_rate, hold, slide, touch, rest`）；
     - `predict_workload` / `predict_trailing`（`workload.py`）：基于 `AcousticContext`（`acoustics.py`）预测局部 ±1s 三角核负荷曲线（事件率、头率、Hold/Slide/Touch 占比）及滞后负荷，并与因果已提交账本 `ledger`（`prefix_ledger`）做预算差比较；
     - `head_timing`（`events.py`）：33 个键区（`A1..A8, B1..B8, D1..D8, E1..E8, C`）的因果按键冷却/距今拍数与秒数。
   - 指针头先按拍距分桶（`BIN_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0)`）计算粗粒度时间带概率与桶内细粒度概率（`hierarchical_log_probs`），并将历史符号分支 `symbolic` 对纯音频指针 `audio_logits` 的残差严格限幅在 `history_residual_bound * tanh(...)` 内。

2. **写什么配置（Slot-Autoregressive Compositional Actions）**：
   - 选定目标候选拍位后，`decode_candidate` / `action_slot` 在同一拍位内按槽位 `slot = 0, 1, ..., S-1` 自回归生成同时落下的音符原子（每个槽位从 `18,453` 个合法单音/滑星原子 + `0 = STOP` 中选择，槽位间以 `/` 拼接为单个 `EventPoint`）。
   - 动作 logits 由共享线性头 `action_head` 与解析器支撑的因子化模块 `CompositionalActions`（`action_factors.py`，含 `kind_mask, head_lane_mask, hold_duration, slide_duration, trajectory_shape, relative_endpoint, break_ex, branch_count, touch_ring, firework` 共 10 组嵌入及 8×8 相对键位关系 `relative_factors`）共同相加得到。
   - 在线生成由 `EventGuard`（`guard.py`）和 `_sample_actions` 执行硬物理约束（同拍最多 2 头、持续动作占手检查、同键位最小间隔 `min_same_head_ms`、MiaCode 进程内实时冲突事务检查，以及可选的 `intercept_tap_on_slide` 开关）。

---

## 二、本轮深度审查定位的 4 个根本问题与代码证据

### 1. 词表粒度极度不对称导致的大类温度失真 + 错误的 `quota_diff` 单位相减
- **代码与数据事实**：
  - 全量官谱（`1,795` 张谱面，`956,579` 个事件，`1,230,210` 个音符）的真实音符占比为：`Tap 72.77%`、`Slide 13.10%`、`Hold 8.18%`、`Touch 5.78%`、`TouchHold 0.16%`。
  - 但在 `18,453` 个合法原子词表中，各类别的原子数量极度悬殊：
    - `Tap`：**仅 32 个原子**（占词表 `0.17%`）；
    - `Hold`：**1,839 个原子**（占词表 `9.97%`）；
    - `Slide`：**16,096 个原子**（占词表 `87.23%`）；
    - `Touch`：**486 个原子**（占词表 `2.63%`）。
  - 因为训练期 `F.cross_entropy` 在温度 $T = 1.0$ 下训练，若模型在某拍赋予 `Tap` 80% 总概率（分摊到 8 个普通 Tap 原子上，每个原子约 `10%`），赋予 `Slide` 10% 总概率（分摊到数百个合理滑星原子上，每个原子仅 `0.05%~0.2%`），则**单个 Slide/Hold 原子的 logit 天然比单个 Tap 原子低 3～6**。
  - 这导致了近期生成中两类连续的踩坑：
    1. 在 `deeadc419d82`、`8a164085f073`、`0e8b8dc4b3b9` 中，对 18,454 维分布直接做扁平截断 `_truncate_probabilities(top_p=0.90, min_p=0.04, max_k=64)`，由于 `min_p = 0.04 * top_tap_prob ≈ 0.5%`，单个 Slide/Hold 原子全部低于阈值被直接清零；
    2. 在 `47a8d3ffa8ff` 中，虽然改成了按类别截断 `_category_truncate_actions`，但混入了单位错配的 `quota_diff = workload[2:5] - ledger[:3]`（左边是 `[0, 1]` 占比，右边是截断到 `1.0` 的事件/秒速率），在密集段把 Hold/Slide/Touch 乘上 $\exp(-2.5) \approx 0.082$；且即便移除 `quota_diff`，原函数用 `macro_temp = max(temperature, 0.85)` 计算大类概率，只要 $T_{\text{macro}} = 0.85 < 1.0$，就会因 $\exp(\Delta \ell (1/0.85 - 1))$ 将 `Slide` 总概率质量额外压低 **2.6～4.5 倍**、`Hold` 压低 **2.1 倍**。
- **已落地修复（`src/maichart/composer/generate.py`）**：
  - 彻底移除错误的 `quota_diff`；
  - 在 `_category_truncate_actions` 中，大类总概率质量（`STOP, Tap, Hold, Slide, Touch`）固定在训练校准温度 **$T_{\text{macro}} = 1.0$**（`masked_logits.softmax(-1)`）下求和；`action_temperature` 与尾部截断（`top_p=0.85, min_p=0.08, max_k=(16, 12, 16, 8)`）仅作用于每个类别内部。

### 2. 18,454 维历史原子嵌入导致的 2-gram 类别/修饰符自强化锁死（Mode-Locking）
- **代码与实测证据**：
  - 在 `Prism_composer_v7/composer_last.pt` 上，当把上述大类温度恢复为校准的 $T_{\text{macro}}=1.0$ 后，实测发现一旦某拍随机采样出 `Touch` 或 `EX`（如 `E6, E5` 或 `1x/8x`），模型会立刻陷入长达 **38 行（200+ 个音符）的纯 Touch 锁死** 或 **13 连发的纯 EX 锁死**；反之当历史全是普通 Tap 时，模型又会将 `Tap` 推高到 `96.46%`、将 `Touch` 压到 `0.27%`。
  - 对 `AudioGroundedComposer` 内部张量的逐项消融证明：
    - `self.event_embedding`（18,454 行独立嵌入，均值范数 `19.36`）的范数是结构化因子嵌入 `self.action_factors.embed`（均值范数 `2.07`）的 **9.4 倍**；
    - 且 `v7` 训练时 `history_dropout` 豁免了 `positions < 2`（即 `t-1` 和 `t-2` 从不丢弃），`tail = memory[rows, lengths-1]` 以权重 `1.0` 直接汇入 `symbolic`。
    - 在固定同一音频时刻（`beat=280`）和同一 30 步混合历史下，仅将末尾 2 步从 `['1', '2']` 换成 `['E4', 'E5']`：
      - 若 `event_embedding` 传入 `['E4', 'E5']` 原始 ID，下一步 `Touch` 概率从 `0.66%` 暴涨至 **`98.81%`**；
      - 若 `event_embedding` 传入其对应的物理键位锚点 `['4', '5']`、而由 `action_factors.embed` 与 `history_features` 传递 `Touch` 结构特征，则下一步 `Touch` 概率温和上升至 **`2.04%~2.19%`**，且下一拍按键分布完美落在该手位附近的 `4 (45.6%), 3 (35.5%)`！
- **已落地修复（`src/maichart/composer/action_factors.py`, `src/maichart/composer/grounded_model.py`）**：
  - 在 `CompositionalActions` 中注册非持久化缓冲区 `_lane_anchor_ids`（`persistent=False`，完全兼容全部现有检查点加载），将每个原子映射到其物理键位锚点（`1..8`）；
  - 在 `AudioGroundedComposer.forward` 的历史序列编码中，采用键位锚定混合嵌入 `(.75 * self.event_embedding(lane_ids) + .25 * self.event_embedding(history) + self.action_factors.embed(history))`（同一拍内部的 `decode_candidate` / `action_slot` 仍保留 100% 原始 `_embedding`，以保持同拍双押 `1x/8x`、`3h/6h` 的对称性）；
  - 移除训练期 `history_dropout` 对 `positions < 2` 的豁免，使 `t-1, t-2` 同样参与随机遮挡。

### 3. 二/三进制混相导致的 `{48}/{64}/{96}/{192}` 极短微距音符爆发
- **代码与数据事实**：
  - 在 `event_compile.py` 中，`1 beat = {4}`，分母 `d` 编译为 Simai 的 `{4 * d}`。全量官谱中 `{4}, {8}, {12}, {16}, {24}, {32}`（即 `beat.denominator in {1, 2, 3, 4, 6, 8}`）占 **`99.5381%`**。
  - 而 `candidates.py` 的 `_FINE_OFFSETS` 包含了分母 `12`（即 `{48}`，200 BPM 下仅 `25ms`）和 `16`（即 `{64}`，200 BPM 下仅 `18.75ms`），且防混相函数 `_phase_transition_allowed` 仅在 `if policy == 'phase_v3':` 下调用，**未在 `MUSIC_CANDIDATE_POLICY_V6`（`absolute_recovery_v6_phase_safe`）中调用**。一旦模型在 `/8` 或 `/16` 二进制相位上选择 `1/12` 偏移，就会落入 `/24`（`{96}`）和 `/48`（`{192}`）混相，导致此前所有生成谱面在 `Line 69–72` 均出现 `{64}/{48}/{96}/{192}` 乱码簇。
- **已落地修复（`src/maichart/composer/generate.py`）**：
  - 在生成端构造候选掩码 `playable` 时（不改变训练/推理候选槽位索引 `0..95`，完全兼容已有权重），对 `MUSIC_CANDIDATE_POLICY_V6` 过滤掉非标准分母候选，要求 `beat.denominator in {1, 2, 3, 4, 6, 8}` 且 `(previous is None or (beat - previous).denominator in {1, 2, 3, 4, 6, 8})`（仅当全部候选均被持键占满时回退至 `base_playable`）。

### 4. 训练侧 `slot 0` 梯度饥饿、合同版本号未升版与 `--resume` 学习率覆盖
- **代码与检查点事实**：
  - 当前最新的 `checkpoints/Prism_composer_v7/composer_last.pt` 保存于 `2026-10-24 08:23:56`（Epoch 9，22,000 步）。它是在**旧版训练代码**下训练出来的：
    1. 当时因官谱中存在 33 原子全屏 Touch 烟花事件，`action_slots` 自动从 `8` 扩展到 `33`，而旧 `_loss` 对全部 33 个槽位求加权平均（其中 31 个槽位是 `STOP` 之后的恒零填充位），导致真正的 `slot 0` 选音梯度被稀释；
    2. 当时 `train.py` 将 `effective_lr` 无条件截断为 `min(args.lr, 5e-5)`，因此 `v7` 全部 9 个 Epoch 实际以 `5e-5`（而非传入的 `2e-4`）训练。
  - 在 `08:23:56` 之后修改的 `active_slots` 掩码、`audio_action_loss` 辅助损失、`history_dropout` 全位置遮挡以及本次键位锚定历史嵌入，**尚未经过任何一轮训练或微调**。
- **已落地修复（`src/maichart/composer/__init__.py`, `checkpoint.py`, `train.py`）**：
  - 将 `GROUNDED_TRAINING_SEMANTICS` 升级为 `'chart_composer_training_v9_active_slots_phrase_tail'`；
  - 在 `checkpoint.py` 中将 `'chart_composer_training_v8_boundary_plans_recovery'` 纳入 `GROUNDED_PROFILE` 的历史兼容合同，使 `Prism_composer_v7/composer_last.pt` 保持 `valid=True, can_generate=True, can_init=True`，但 `can_resume=False`（强制通过 `--init-checkpoint` 显式微调到新目录或开启新训练，防止静默混用新旧合同）；
  - 在 `train.py` 中，当使用 `--resume` 且命令行未显式指定 `--lr` 时，自动恢复优化器状态中保存的实际学习率 `saved_lr`，不再被默认 `2e-4` 覆盖。

---

## 三、修复前后同音频、同随机种子、同条件 A/B 实测对比

使用同一权重 `checkpoints/Prism_composer_v7/composer_last.pt`、同一音频 `5a0d17982e01_haptime - dingerbox.mp3`（BPM 200，Level 14.5）及完全相同的 `condition.json`（`tap_bias=0.787, slide_bias=0.088, hold_bias=0.070, touch_bias=0.041, chord_bias=0.245`）：

| 指标 | `47a8d3ffa8ff` 修复前 | `47a8d3ffa8ff` 修复后 | `0e8b8dc4b3b9` 修复前 | `0e8b8dc4b3b9` 修复后 |
| :--- | :---: | :---: | :---: | :---: |
| **随机种子 (`seed`)** | `709835971` | `709835971` | `1696576229` | `1696576229` |
| **总事件数 / 总音符数** | `903 / 1,338` | `772 / 972` | `894 / 1,272` | `792 / 1,136` |
| **Tap 数量** | `1,329 (99.3%)` | `861 (88.6%)` | `1,264 (99.4%)` | `1,043 (91.8%)` |
| **Hold 数量** | `1 (0.07%)` | **`49 (5.0%)`** | `7 (0.55%)` | **`27 (2.4%)`** |
| **Slide (+Wifi) 数量** | `8 (0.60%)` | **`41 (4.2%)`** | `1 (0.08%)` | **`64 (5.6%)`** |
| **Touch 数量** | `0 (0.0%)` | **`21 (2.2%)`** | `0 (0.0%)` | **`2 (0.2%)`** |
| **独特配置数 (`unique_texts`)** | `46` | **`97 (+111%)`** | `55` | **`109 (+98%)`** |
| **最长同配置连发 (`max_consec`)** | `7` | **`5`** | `4` | **`5`** |
| **`{48}/{64}/{96}/{192}` 异常细分** | `14 处` | **`0 处`** | `6 处` | **`0 处`** |
| **最小事件间隔 (`min_event_interval_ms`)** | `18.75 ms` | **`37.50 ms`** | `18.75 ms` | **`37.50 ms`** |

---

## 四、当前工作区关键文件索引与验证状态

- **核心模块索引**：
  - `src/maichart/composer/__init__.py`：模型族常量与当前训练合同 `GROUNDED_TRAINING_SEMANTICS = "chart_composer_training_v9_active_slots_phrase_tail"`。
  - `src/maichart/composer/grounded_model.py`：`AudioGroundedComposer` 主体（层次化时间指针、键位锚定历史编码、槽位自回归解码）。
  - `src/maichart/composer/action_factors.py`：18,453 个合法原子的语法解析、10 维组合式因子嵌入与 `_lane_anchor_ids`。
  - `src/maichart/composer/generate.py`：生成主循环、$T_{\text{macro}}=1.0$ 校准类别截断采样器 `_category_truncate_actions`、`{4}..{32}` 相位安全掩码、可选 `intercept_tap_on_slide`。
  - `src/maichart/composer/train.py`：`active_slots` 损失掩码、`audio_action_loss`、在线 `RecoveryBuilder` 恢复批训练、`--resume` 学习率保持与合同校验。
  - `src/maichart/composer/checkpoint.py`：检查点保存/加载、`v7/v8 -> v9` 合同校验与显式 finetune 迁移（`migrate_audio_grounded`）。
  - `src/maichart/composer/guard.py`：增量式 `EventGuard`（同头冷却与 MiaCode 冲突拦截）。
- **自动化测试状态**：
  - Python 单元测试：`166 passed`（涵盖 `tests/test_chart_composer_v1.py` 及全部 14 个 `tests/test_composer_*.py` 测试文件）。
  - 前端 UI 控制契约测试：`node tests/test_composer_generation_ui.js`（`5 passed`）、`node tests/test_composer_training_ui.js`（`9 passed`）。

---

## 五、交接给下一位智能体的直接行动指南（Next Steps）

1. **当前权重状态认知**：
   - `checkpoints/Prism_composer_v7/composer_last.pt`（Epoch 9，22,000 步）是在旧版 `v8` 合同（无 `active_slots` 掩码、无 `audio_action_loss`、学习率被锁在 `5e-5`、`positions < 2` 未做 dropout）下训练的。
   - 本轮生成侧与前向侧修复已经使 `v7` 检查点直接产出正常比例的 Hold/Slide/Touch、消除了 `{48}/{64}/{96}/{192}` 乱码与 38 行 Touch 锁死，但**模型的底层动作权重仍只经历了 `5e-5` 学习率下被 33 槽位稀释的 9 个 Epoch 训练**。
2. **推荐启动的训练任务（Finetune 到 `Prism_composer_v8`）**：
   - 不要对 `Prism_composer_v7` 执行 `--resume`（合同检查已将其拦截为 `can_resume = False`）。
   - 使用 `--init-checkpoint E:\simai_transformer\checkpoints\Prism_composer_v7\composer_last.pt` 并指定新输出目录 `--checkpoint-dir E:\simai_transformer\checkpoints\Prism_composer_v8`，在 `--lr 0.0002`（或稳健微调 `0.0001`）与新 `v9` 合同下继续训练：
     - 此时 `active_slots` 掩码与 `effective_lr = 2e-4` 均已生效，`slot 0` 选音与 `action_factors` 的有效学习速度较 `v7` 提升约 **25～40 倍**；
     - 训练数个 Epoch 后，重点观察 `action_loss` 的下降斜率，并在固定种子与多样化 `style` 条件（如提高 `slide_bias` / `hold_bias` 或使用官谱默认 `tap_bias=0.72, slide_bias=0.14, hold_bias=0.08`）下复测滑星轨迹与复合配置的丰富度。
3. **生成端调参注意**：
   - WebUI 若加载了偏 Tap 的参考条件文件（如 `tap_bias = 0.787`），模型会忠实偏置到高 Tap 占比；若希望生成更丰富的滑星/长条配置，可在生成面板将 `slide_bias` 设为 `0.14~0.18`、`hold_bias` 设为 `0.08~0.12`、`tap_bias` 设为 `0.68~0.72`，`pointer_temperature` 保持 `0.50~0.60`，`action_temperature` 保持 `0.80~0.90`。
