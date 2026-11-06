# 训练与生成根因修复进度

更新时间：2026-02-13
状态：**训练前暂停**。用户要求先完成代码修改，等其调整训练/生成参数后再继续；本轮未启动训练、GPU 实验或真实音频生成。

## 本轮目标

直接修复音频与谱面不同步、静音区错误产出、难度控制错配和训练/生成契约漂移。旧 checkpoint 不再作为兼容目标，新架构允许破坏性变更并要求重新构建缓存、重新训练。

## 已完成

- 数据集难度槽与等级范围可配置：`min/max_difficulty_index`、`min/max_level`。
- 难度过滤、窗口参数、条件 mask/default 纳入 dataset signature 和训练 contract。
- 条件 dropout 只 mask target-derived controls，保留 level，使用 contract default。
- 逐窗音频强度因子应用于训练和生成的 `note_density/peak`，不再只在推理时改变条件。
- 训练 scheduler/early stopping 在学习率下降后重新开始观察期。
- 生成 CFG null 保留 level，其他维度使用训练 mask default。
- top-p 改为在 temperature 缩放后计算 nucleus。
- 生成 difficulty 元数据不再固定写死为 5。
- 模型条件改为独立 encoder memory token，并新增条件类型嵌入。
- `max_audio_tokens` 默认从 512 提高到 1024，旧 checkpoint 不兼容。
- CFG null 保留 level，其他控制使用训练默认值。
- 生成窗口支持读取 family contract 的 overlap，使用与训练相同的 stride。
- 新增此进度文档，持续记录根因和验收项。

## 本轮验证

- `py_compile`：训练入口、生成入口、事件模型均通过。
- 定向测试：`83 passed`（模型、生成窗口、条件契约、训练流程）。
- 额外最终 smoke：`28 passed`（模型、事件窗口、训练 smoke）。
- 未运行训练、GPU benchmark 或真实音频生成，遵照用户要求停在训练前。

## 仍需完成

- 完整检查并补全所有 family contract 字段，尤其 audio margin、overlap、difficulty 与 condition policy。
- 最终 audible span 需要在 merge/compile 前实施硬边界过滤，并保证 chart `first`/负 offset 的时间映射一致。
- 完成 best-of-N 候选排序和质量报告。
- 修复低难度样本的 dataset 审计/UI 计数，并检查不同槽位解析质量与覆盖数。
- 统一 CFG default 从 checkpoint `condition_params` 传入，而不是固定 0.5。
- 真实训练和生成：等用户调整参数后再继续。


## 第二轮已完成

- merge 后增加最终 audible span 硬过滤；即使窗口级 trim 漏掉事件，编译前也会再次按音频活动区过滤。
- 质量报告记录 `boundary_trim.span_seconds` 和 `removed_after_merge`。
- 修复 CFG null 分支被重置为空的 bug；guidance>1 不再向 encoder 传入 None 条件。
- 增加候选质量分数 `candidate_score`，综合语法、警告、音频跟随和请求条件偏差。
- 生成窗口继续使用训练 contract 的 overlap stride。
- `candidate_score` 在生成结束后转换为 `candidate_rank` / `candidate_best` 并更新质量报告。
- CFG null 的 `condition_default` 从 checkpoint contract 传入，不再硬编码 0.5。
- 修复候选质量报告 rank 路径映射 bug；覆盖候选文件名与报告文件名不同的情况。
- 最终全量回归：`586 passed, 11 skipped`；三个修改模块通过 `py_compile`。


## 结构规划器与变 BPM 处理（当前轮）

- 变 BPM 继续使用完整 tempo map 积分 beat phase/local BPM，不把固定秒长直接当作固定音乐拍长。
- 明确 planner 的 density target 使用 tempo map 积分后的绝对秒长，按 notes/second 归一化；beat-normalized rhythm 只作为节奏形态辅助分析，不替代舞萌实际谱面密度。
- Event model 新增轻量 `plan_head`，从 encoder memory 预测 5 维窗口规划：事件预算、密度代理、和弦比例、滑条比例、尾部事件位置。
- EventDataset/EventCtxDataset 生成 `plan_target`，训练 token CE 外加 0.1 权重 planner MSE；推理默认仍返回原 logits。
- 旧 checkpoint 缺少 planner head 参数时允许兼容加载；新训练会学习 planner head。
- 结构 smoke 验证：`34 passed`（事件音频、tempo map、模型、窗口数据集、训练 smoke）。

本轮未启动真实训练或真实生成。下一步应通过短跑观察 planner loss、窗口密度 CV、静音区和段落响应，再决定是否把 rhythm channels 正式升级为新的输入 contract。

- 检查确认生成 merge 已使用 contract-derived `overlap_beats`，不会退回 CLI 默认值。
- `python -m maichart.training.train_events --help` 已确认新增参数可见：`max-audio-tokens`、`prewarm-workers`、难度范围和 overlap。
- overlap smoke：`segment=16, overlap=8` 生成滚动窗口并通过。
- 当前已到达真实训练/生成阶段；按用户要求暂停，等待训练参数调整，不自动启动训练或真实音频生成。

## 旧 checkpoint 兼容（修复 generation_status=failed）

**现象**：WebUI 生成报 `Error(s) in loading state_dict for EventConditionedTransformer: Missing key(s) in state_dict: "condition_type_embedding"`。

**根因**：条件注入从「逐帧常量偏置」改成「独立 encoder memory token」时新增了 `condition_type_embedding` 参数。旧 checkpoint 没有这个参数，`strict=True` 加载必然失败。

**修复**：新增 `EventTransformerConfig.condition_fusion`（`memory_token` / `per_frame_bias`）与 `EventConditionedTransformer.load_weights_compatible()`：

- state_dict 缺少 `condition_type_embedding` → 自动切到 `per_frame_bias`，即该 checkpoint 训练时的结构；
- 缺这个参数之外还有任何缺失/多余键 → 仍然硬报错，避免“结构不同却静默加载”；
- 结构模式写入 `model_config`，所以旧结构 finetune 后再保存，下次仍按旧结构加载（否则会因参数被序列化而误判为新结构）；
- `load_event_model`（生成路径）与 `train_events`（resume/finetune）都走这条路径，并在日志打印使用哪种融合方式。

**验证证据**：

- 用你失败时用的 `checkpoints/event_v7/event_epoch_021.pt`：加载成功，`condition_fusion=per_frame_bias`，memory 长度 = 帧数 + global（未追加 token），`decode_step` 正常。
- 真实短生成（20s 片段，CPU，参数对齐你那次运行）：4 个窗口、121 事件、全部 closed、退出码 0，产出 `&inote_5=` 谱面 156 个音符单元；日志显示 `audio_bounds music=2.00-19.98s`，第 4 窗口 `trimmed_out_of_bounds=14`，说明边界裁剪也在工作。
- 新增 6 个回归测试覆盖：旧结构加载、旧结构保存后重载仍为旧结构、新结构保持 condition token、真实结构不匹配仍报错、未知融合模式被拒绝。
- 全量测试：`594 passed, 11 skipped`。

**注意**：你那次运行的参数里带了 `--no-audio-bounds-trim`（音频边界裁剪被关闭），同时 `--audio-bounds-lead 2`、`--silence-threshold 0.02`、`--silence-window-share 0`。如果目标包含“音频结束后不要有音符”，需要在生成面板把「音频边界裁剪」重新勾选。

## 当前暂停

未启动训练或 GPU benchmark。为定位并验证本次报错，只在 `.scratch/` 下用旧 checkpoint 对一段 20 秒片段做了一次 CPU 短生成（不写入你的输出目录），未触碰你的正式产物。下一步需要在用户调整训练参数后执行。候选最终 rank 尚未改成两阶段写出，当前每个候选报告都有可比较的 `candidate_score`，不会改变现有输出接口。

1. 由用户确认/调整难度档、等级范围、`max_audio_tokens`、context/audio margin、overlap、batch/显存预算。
2. 构建新 dataset audit 和新 checkpoint contract；旧 checkpoint 不复用。
3. 短跑训练 1-2 epoch 验证 checkpoint 保存/恢复与条件分布。
4. 用一首有静音前导、安静起音、尾部静音的真实音频生成，检查首尾时间误差、音频强度跟随、不同 level 输出差异。
5. 指标满足标准后再决定是否长训。

## 验收标准

1. 全量测试通过。
2. 短跑训练能生成包含新 model/data contract 的 checkpoint。
3. 生成质量报告包含 audible span、窗口跳过、开放窗口、条件和难度信息。
4. 真实音频生成中首音不早于可听起点容差，末音不晚于可听终点容差。
5. 低/高 difficulty 请求在固定音频上产生可测的密度差异。
