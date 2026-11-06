# Composer CUDA 内存分配排查与主机内存保护交接文档（中文版）

原文：[COMPOSER_CUDA_ALLOCATION_HANDOFF_20261022.md](COMPOSER_CUDA_ALLOCATION_HANDOFF_20261022.md)\
日期：2026-10-22

## 问题报告与证据

任务 `e79e578258df` 在首个 batch 的 `pointer` 附近执行 BF16 cuBLAS GEMM 时失败。请求配置为：batch 32、宽度 384、双编码器各 6 层、FFN 1536、历史 128、候选 96、8 动作槽位、词表 18,454、fused AdamW、预取 2 个 batch，模型参数量 85,499,706。

仅凭原始 `pointer` 报错栈无法定位首个失败的异步 CUDA 操作。在开启 `CUDA_LAUNCH_BLOCKING=1` 复现时，错误实际上更早发生在历史编码器 FFN dropout 的 20 MiB 显存分配处——当时 CUDA 报告仍有 9.52 GiB 空闲显存，Torch 仅分配了 1.12 GiB。因此这并非 batch 32 超出独立显存容量所致。

实机环境：RTX 5070 Ti Laptop（12,227 MiB），驱动 581.15，Torch 2.7.0+cu128，CUDA 12.8（含 `sm_120` 架构支持）。单独运行同尺寸 BF16 前向/反向传播在 1.42 GiB 峰值显存下成功完成。

在未修改代码前使用真实数据复现时，任务完成了两次优化器更新，期间观察到宿主机状态：可用物理内存（physical RAM）从 2.275 GiB 跌至 1.100 GiB，Windows 可用提交内存（free commit）从 3.084 GiB 跌至 0.990 GiB。因此，**Windows 宿主机内存/页面文件提交额度（commit）压力是高度可疑的诱因**。目前无证据表明需要替换 BF16 GEMM 或缩减模型容量。

## 变更内容

- 将修改前的原始训练/数据模块备份至 `backups/composer_cuda_before_20261022_153936.zip`。
- 新增基于 Windows `GlobalMemoryStatusEx` 的宿主机内存监测：可用物理内存、可用提交额度、总物理内存与总提交上限（非 Windows 平台保持原缓存策略）。
- Composer 特征 LRU 缓存现在最多每秒检查一次内存余量：其实际生效容量绝不超过用户请求容量，并在必要时自动收缩以保留约 **2 GiB 物理内存余量**和 **4 GiB 提交额度余量**。当系统内存恢复时，缓存预算可随之恢复。
- 将实际生效的缓存预算（`effective_budget_bytes`）与配置预算分开报告。在极端宿主机内存压力下，生效预算可降至 0（以更多磁盘 I/O 换取系统内存安全）。
- 启动时若宿主机内存余量过低会输出 `composer_warning`；运行时与进度日志包含 `host_memory`；batch 循环中若发生任何 CUDA 错误会输出 `composer_cuda_failure`（含 Torch/CUDA 版本、显存分配计数与宿主机内存快照），并通过异常链保留原始堆栈。不对已损坏的 CUDA 上下文做自动重试或静默降级精度/batch。
- 修复了一处限步运行（`max_steps`）缺陷：DataLoader 的 batch sampler 现在会在 `max_steps` 处终止，防止后台预取线程继续加载超出步数上限的无用批次。

模型参数、权重键名、检查点 schema、特征、监督目标、损失权重与优化器算法均保持不变。

## 集中验证

修改后，`test_composer_runtime_memory.py` 与 `test_chart_composer_v1.py` 共 19 项测试全部通过，覆盖模拟宿主机内存压力驱逐、容量上限、有界迭代、诊断异常链、Windows 实时内存快照、CPU 训练/检查点/生成及精确断点续训。

在真实数据（947,683 样本）上使用原模型配置与异步 CUDA（`CUDA_LAUNCH_BLOCKING=0`）完成了 8 步 batch-32 BF16 fused-AdamW 更新并成功保存检查点（峰值分配显存 2.068 GiB，保留显存 2.252 GiB；当时宿主机空闲 commit 约 1.85 GiB、空闲物理内存约 1.31 GiB，特征 LRU 按预期停止驻留新数组）。

## 后续运行建议

释放宿主机内存后即可重试原模型与 batch 配置：请关闭不必要的占内存应用/服务，并确保系统页面文件（pagefile）充足，无需将全局精度降为 FP32 或缩小模型。
