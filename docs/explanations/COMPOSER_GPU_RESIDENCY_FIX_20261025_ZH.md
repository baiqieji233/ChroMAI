# GPU 满载却训练变慢：显存驻留压力修复（2026-10-25）

本记录接续 [Touch 总量修复](COMPOSER_TOUCH_TOTAL_FIX_20261025_ZH.md)。用户允许必要时重启，并建议 batch 从 96 降到 64。本轮已保留有效检查点、定位并复现性能悬崖，改为 **64 × 梯度累积 6**，继续原 v9 的总 20 轮目标。

## 1. 原因与证据

原任务 `09d14b6f4d31` 的日志有明确拐点：

| 位置 | 近期速度 | PyTorch reserved | peak allocated |
| --- | --- | --- | --- |
| batch 2750 前后 | 约 2.5–2.9 batch/s | 约 8.03 GiB | 约 4.3 GiB |
| batch 2800 | 约 0.228 batch/s | **9.93 GiB** | **4.64 GiB** |
| 随后至 batch 5475 | 约 0.23–0.34 batch/s | 持续 9.93 GiB | 持续 4.64 GiB |

慢速区间每 25 batch 约耗时 80 秒，供数等待仅约 0.03 秒，主要时间在 GPU 执行路径。并行 CPU 生成测试结束很久后仍持续缓慢，因此不能把长期退化归咎于那些测试。也没有伴随更新计数异常、非有限值或 loss 爆炸；loss 是本轮累计均值，不代表最近一个窗口的误差。

Windows 性能计数器在停止前记录训练进程 dedicated 约 10.01 GiB、shared 约 0.63 GiB。此时 GPU 利用率 100%，但 PyTorch 当前活跃张量仅约 0.71 GiB、全程峰值约 4.64 GiB。大量保留的闲置 CUDA 块仍占用驱动侧显存预算，形成显存驻留压力；满载不等于张量运算高效。

另一个确定问题：原日志明确出现 **`expandable_segments not supported on this platform`**。现有 Windows / PyTorch 2.7.0+cu128 环境并未获得原启动设置期望的保护。

### 同模型、同真实 batch 的 GPU 隔离复现

停止原训练、释放其上下文后，使用保留权重、相同 96 样本 batch、BF16 做对照，不更新模型权重：

| 状态 | 每 batch 稳定均值 | reserved |
| --- | --- | --- |
| 干净缓存 | **0.180s** | 3.66 GiB |
| 仅增加闲置缓存压力 | **3.609s** | 9.90 GiB |
| `empty_cache` 后 | **0.345s** | 3.67 GiB |

清理当时 reserved 从 9.90 降至 0.53 GiB，活跃分配仍约 0.51 GiB，耗时约 0.081s。这直接复现了缓存压力导致的巨大退化，并验证释放闲置块有效；缓存清理后的冷分配/驱动恢复也会带来短期额外耗时。

该压力实验中的 `inactive_split_bytes` 仅约 0.019 GiB，说明不需要不可释放碎片就能重现。原进程没有记录该细项，不能把它全部 9 GiB 闲置保留量都直接称作“碎片”。WDDM 驻留迁移是由计数器与隔离实验支持的机制解释，本轮没有做逐页驱动追踪。

## 2. 保留的修复

1. Composer 训练接入已有 `CudaCacheGuard`，在**完整 optimizer 更新且梯度释放之后**检查闲置 CUDA 缓存。高水位取设备显存的 70%；还要求闲置至少 0.5 GiB 且占 reserved 的 20%，清理后有 8 次更新冷却。不是每个 batch 清空，也不删除活跃张量、修改模型参数或消耗 RNG。
2. Windows 服务默认改用支持的 native allocator `garbage_collection_threshold:0.7`；其他平台保留原默认。显式用户环境配置仍优先。新任务运行时已确认实际配置为该值，不再依赖不受支持的 expandable segments。
3. 按本次授权把微批从 96 降至 **64**，累积从 4 改为 **6**，常规有效 batch 仍为 **384**。学习率继续为 `1e-4`，模型、损失、质量标注、数据集与恢复采样策略未改。
4. 日志补充活跃/保留/闲置缓存、driver free、inactive split、回收次数与事件耗时，避免再仅看利用率或早期 16 个 profiling 样本。

此前短跑验证覆盖了计算与梯度等价，却没有捕捉到长程缓存积累；本轮补上了持续的显存约束和分配观测。

## 3. 改 batch 时如何保留已保存的进度

原检查点为第 1 轮、step 1300、batch 5200，对应已保存 **499,200** 个样本。Windows 取消接口会强制停止，不会临时多存一次；当时最后约 93 次未落盘更新需要从此补算，不从随机权重重训。

单纯把旧 batch 游标按 96/64 换算后改用新的歌曲采样器，会改变样本顺序：`SongLocalBatchSampler` 的组间交错取决于 batch 大小。为防止重漏，本轮新增显式 `--rebatch-resume`：

- 从保存的 epoch 起始 loader RNG 重建**原来的 96 采样器**；将同一索引序列重新分成 64，跳过已保存的样本前缀。
- 要求新旧有效 batch 相同、旧新均为 `max_steps=0`，且位于完整 optimizer 更新及样本边界。缺少 RNG 或游标不一致时拒绝。
- 当前未完成 epoch 保存 `sampler_batch_size=96` 和真实 `sample_in_epoch`；后续普通 64 续训继续该序列，进入下一 epoch 后才使用 64 采样器。
- 未完成的重分批检查点使用版本化 loader RNG 字典。当前 reader 校验并解包；旧 reader 会在训练更新前明确失败，避免默默按错误顺序续训。完成 epoch 后恢复普通 Tensor 格式，生成不读取该训练状态。

CPU 三段真实训练回归覆盖：旧 batch 中断 → 重分 batch 中断 → 普通续训并进入下一 epoch。逐样本验证无重漏、剩余顺序与 RNG 一致、尾批游标正确。相同有效 batch **不保证与原 96 微批的数值轨迹逐位相同**：微批权重归一化和 dropout 张量尺寸会改变。

## 4. 当前任务与恢复验证

- 工作台：**`http://127.0.0.1:7863`**。
- 当前任务：**`9d5be08cfe23`**；目录仍为 **`checkpoints/Prism_composer_v9`**。
- 总目标仍为 epoch 20，当前从 epoch 1 中途继续，未追加成额外 20 轮。
- 原慢速任务已取消；`653bd0df6406` 为首次 64 验证任务，随后补齐了版本化采样状态防误读保护。最终任务从其完整 step 1600 检查点继续，权重和 Adam 状态保留。
- 截至 **08:33 北京时间**：当前任务 running；实测加载的正式检查点为 **step 1700 / batch 10200 / sample 652800**，`valid=true`、`can_resume=true`，模型和优化器全部有限，版本化采样状态可正常恢复。
- 一次实时快照为 step 1795、batch 10775；活跃约 0.69 GiB、reserved 4.20 GiB、峰值约 2.99 GiB、driver free 约 6.43 GiB。
- **08:40 最新状态**：第1轮 batch **12,150 / 14,829**、step **2,025**，running、无错误；约208样本/s，reserved约4.23 GiB、driver free约6.40 GiB。状态快照保存为 `task_at_handoff.json`。
- 恢复阶段约 **200–260 样本/s**，原慢速稳态约 **30 样本/s**。batch 不同，比较样本吞吐比比较 batch/s 更合适。仍需观察后续曲目与 epoch，不能把初期恢复速度当作整个 20 轮的保证。

### 为什么正常情况下 epoch 仍约一小时

全量每轮有 **949,015 个事件样本**。batch 96 是 9,886 次前后向；batch 64 是 **14,829** 次。每个样本还包含候选音频、历史和大词表动作预测，并不是每首歌仅做一次推理。

按目前约 200–260 样本/s，完整一轮约 **61–79 分钟**；这是测算，会随数据与机器负载变化。按先前约 30 样本/s 的慢态换算则接近 9 小时。梯度累积只改变更新频率，不会省去每个样本的前后向。当前第 1 轮已经处理一部分，剩余时间应看新任务 ETA，而不是把此前慢速耗时算成新速度。

## 5. 检查与证据

Composer/预取/CUDA 缓存/重分批 Python 回归 **300 passed**，前端 **16 passed**；`compileall`、`git diff --check` 通过。两名默认配置子代理参与日志和续训审查，工具未返回实际模型名称。

证据目录：[`.scratch/composer_gpu_slow_20261025`](../../.scratch/composer_gpu_slow_20261025/)：

- `preserved/composer_last.pt`、`preserved.json`：原 v9 step 1300 的完整、有限且有哈希的备份。
- `gpu_cache_probe.json` / `.log`：同 GPU、同模型、同真实 batch 的压力与释放对照。
- `throughput_series.json`：逐窗口速度、显存及供数/计算耗时。
- `resume_request.json`、`final_resume_request.json`、`final_resume_task.json`：64×6、原总 epoch 20 的续训参数与谱系。
- `preserved_b64/composer_tensor_state.pt`：首次 64 运行的原始对照快照，仅供诊断、必须使用当前 reader；`preserved_b64/composer_last.pt` 为等权重/等优化器的版本化恢复入口。没有并发改写活动训练检查点。
- `final_verification.json`：正式检查点与优化器数值、旧六个检查点哈希不变、采样游标、源码哈希及实时状态。
- `regressions_final.log`、`rebatch_final_tests.log`、`node_tests.log`：回归结果。

原 v7/v8 等模型继续保留；本轮没有更改生成端 Touch 保护或把吞吐恢复视为音乐质量改善。
