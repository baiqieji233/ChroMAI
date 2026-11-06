# Composer 已复现实现修复与恢复训练（2026-10-24）

> **接续完成：** 恢复任务已正常完成第 3 轮 / step 7,416，last 和 inference 权重一致。固定音频/条件/双种子六次生成对照也已完成；Touch 偏高仍在，残留冲突表现因种子而异。最新结果见 [完成训练与固定条件对照](COMPOSER_COMPLETED_TRAINING_COMPARISON_20261024_ZH.md)。下文保留修复及恢复阶段的历史证据。

本轮按要求只修复已复现的实现问题，保留原检查点，完成回归和误修复核后尝试恢复训练。此前审查依据见 [生成质量相关代码审查](COMPOSER_QUALITY_REVIEW_20261024_ZH.md)；本文件是修复后的接续记录。

## 1. 修复范围

本轮模型源码仅修改 `src/maichart/composer/generate.py`，新增回归文件 `tests/test_composer_generation_regressions.py`。训练目标、数据构建、词表和模型参数形状均未修改。

| 已复现问题 | 修复 | 核实边界 |
| --- | --- | --- |
| 搜索游标污染相位保护 | 新增 `_candidate_playability`，实际生成调用从 `history` 读取最后已提交拍位作为相位基点；搜索游标仍只负责推进候选搜索 | 保留原分母白名单及无安全候选时的既有回退策略。新测试强制耗尽实际生成循环的重试，覆盖有历史和空历史两条路径 |
| 低正温度抹掉整类概率 | 每类内部先减去该类最大 logit，再除温度并 softmax；大类质量仍按 T=1 汇总 | 温度 0 仍走原贪心路径，截断参数未改；低至 `1e-12` 的正温度保持测试中的 Slide 类别质量，STOP-only 和缺失类别也正确 |
| 合法零时长 Hold 被文本特判屏蔽 | 删除 `[1:0]` 等字符串导致头尺寸清零的特殊逻辑，使用解析得到的真实头计数 | 两种等价零长 Hold 均能采样；占用、两头容量和同键冷却仍生效。主代理实测同键 5 ms 重发均被拒绝 |
| 未来 tempo 锚点令早停导出失败 | 导出末拍按音符和所有保留 tempo 锚点的最大值取 `ceil(max)+1` | 保留输入变速表；恒速情况下保持原尾拍行为。不改变 EOS 或事件上限，不尝试修复未经验证的长动作尾部问题 |
| condition 的难度槽未用于导出 | 将 `condition.difficulty` 传给编译器，并在加载权重前检查范围 1–6 | 默认槽 5 和显式槽 6 均经过真实生成、TXT 回读和报告一致性验证；非法槽位不会先加载权重 |

导出行为需要特别说明：如果提前结束写音符，但输入仍有后段变速点，保留这些点会使 E 延后并留下空尾；输入变速点位于音频范围之后时也一样。E 延后不表示相应音乐段落已经生成了音符。本轮没有新增裁剪变速表的策略。

## 2. 明确暂缓的策略与假设

以下内容没有修改：

- 绝对拍位和相邻拍距的分母白名单，以及它对 `1/8 → 1/3` 等转换的限制。
- 同权重批次的质量权重归一化；`sample_locality` 与质量加权如何配合，仍待实际权重分布实验。
- 恢复训练的 T=1 时间采样及其与部署默认温度的差异。
- EOS 阈值、`max_heads=2`、组合失败后的首原子封禁、MiaCode 各类报告/拦截策略。
- Touch 占比、物量、音乐段落呼应和长动作尾是否被播放器截断等尚未归因的问题。

因此不能把本轮修复解释为已完成音乐性或全部可玩性验收，也不能称所有相位回退均被取消。

## 3. 回归与误修复核

| 检查 | 结果 |
| --- | --- |
| 全部 Composer Python 回归 | **195 passed**，34.68 秒 |
| 补充编译/导出回归 | **35 passed**，包含 event_compile、form_cli_export_timing 和 playability_export |
| 前端生成/训练/曲线入口 | **5 / 9 / 2 passed**，共 16 项 |
| 编译检查 | `compileall` 通过 |
| 空白/补丁检查 | `git diff --check` 通过 |
| 源码范围检查 | 与上轮审查的 SHA-256 对比，Composer 源文件中仅 `generate.py` 改变 |
| 子代理误修复核 | 未发现阻断性误修或范围扩大；主代理独立运行并核实回归与物理冷却探针 |

新增的 17 项回归包括：低正温度大类质量、STOP-only、普通温度条件分布、零长 Hold 的可达性与占用限制、原相位规则及回退保持、真实生成循环重试耗尽、未来变速点早停导出、默认/指定难度槽、非法难度槽提前失败和恒速尾拍保持。

补充导出测试出现一条现有 PyTorch nested-tensor 优化提示，没有测试失败。本轮没有为消除该提示修改模型结构。

上述验证未发现误修，但它们不代替歌曲试听或人工试玩。本轮没有为了对照重新生成完整歌曲。

## 4. 原检查点保护与恢复依据

原目录保持不写入：

- `checkpoints/Prism_composer_v7/composer_last.pt`
- `checkpoints/Prism_composer_v8/composer_last.pt`
- `checkpoints/Prism_composer_v8/composer_inference.pt`

原文件路径、大小和 SHA-256 已记录于 `.scratch/composer_impl_fixes_20261024/checkpoint_provenance.json`。本轮从 v8 的 **训练文件**恢复，不使用第 2 轮的 inference 文件替代它。

恢复前实际校验：v9 合同，`valid=true`、`can_resume=true`；第 3 轮、5,000 次更新、`batch_in_epoch=224`、`epoch_complete=false`。模型权重和优化器浮点状态全部为有限值，优化器学习率为 `1e-4`。第 224 batch 正好位于梯度累积窗口边界。

旧日志虽记录过第 475 batch / 5,062 次更新，但未保存的更新无法从日志恢复；本轮以有效文件中的 **5,000 步**为起点。

## 5. 恢复任务

- 工作台：[http://127.0.0.1:7862](http://127.0.0.1:7862)，已用修复后的源码新建服务。
- 任务 ID：`ee3b216288c4`。
- 来源：原 `Prism_composer_v8/composer_last.pt`。
- 新输出目录：`checkpoints/Prism_composer_v8_resume_20261024`，不覆盖原目录。
- 请求只使用 `resume`，清掉旧的 smoke `init_checkpoint`；未显式传 `lr`，由优化器恢复 `1e-4`。
- `epochs=1` 的实际范围是 `start_epoch=end_epoch=3`，只补完第 3 轮，不是另加三轮。
- 保持 batch 96、梯度累积 4、BF16、`max_steps=0`、`limit_charts=0`、seed 0、locality 4、样本线程 4、预取 2，以及原恢复训练和辅助损失设置。
- 训练器已重建完整数据并通过原数据指纹检查：1,787 张可用图、949,015 个样本；运行时确认第 3 轮和 `effective_lr=0.0001`。
- 已确认实际恢复更新：第 3 轮第 **500 / 9,886 batch、step 5,069**，`resume_skipped_batches=224`，状态 running、无任务错误。该数字为本次取样快照，不是最终进度。
- 后续取样已到第 875 batch / **step 5,162**；step 5,100 的恢复 rollout 接受 2 个样本、加权 loss 约 0.05619。完整新日志未发现 traceback、非有限批次或非有限损失提示。
- 此时约 2.43 batch/s；取数累计 20.42 秒、传输 0.87 秒、计算 91.10 秒，CUDA 峰值分配约 4.538 GiB。当前阶段尚未到下一次 500 更新的定期保存点。
- 正式日志：`webui_workspace/logs/20261024_203852_composer_train_ee3b216288c4.log`。

从保存位置应跳过本轮已完成的 224 个 batch，由第 225 batch 继续；若没有跳过无效更新窗口，剩余 9,662 个 batch、约 2,416 次更新，预计结束于 step 7,416。训练完成与否以真实任务状态和新保存文件为准。

这属于合同兼容的修复后续训。优化器、RNG 和批次位置得到恢复，但生成 helper 也被恢复 rollout 复用，修复后的合法原子与数值采样可能改变后续提案，因此不承诺与旧代码执行轨迹逐位一致。

## 6. 接续材料与剩余工作

`.scratch/composer_impl_fixes_20261024/` 中保存：

- `checkpoint_provenance.json`：原文件身份。
- `checkpoint_validation.json`：权重/优化器有限值与零长 Hold 冷却检查。
- `resume_request.json`：实际恢复请求；没有 `lr` 和 `init_checkpoint`。
- `resume_task.json`、`resume_observation.json`：任务入口和运行观察。
- `resume_confirmed.json`：实际更新已超过 5,000 步，以及三个原检查点恢复后 SHA-256 均未改变的核验。

第 3 轮完成及固定单曲双种子对照现已核验，见上方接续报告。仍需更多歌曲、条件干预及人工试听/试玩。审查中的策略取舍与证据不足假设继续保留为待研究项，不能混入已修复列表。
