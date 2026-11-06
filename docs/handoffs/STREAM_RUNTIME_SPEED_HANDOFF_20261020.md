# chart_stream 训练与推理效率交接

日期：2026-10-20。分支：`codex/v11-quality-fixes-20261020`。本轮保留此前质量修复和工作区未提交内容；没有修改权重、删除缓存、启动正式训练或提交 Git。

> 后续实际训练低利用率调查、CPU 流水线提速及阶段归档见 `STREAM_TRAINING_CPU_BOTTLENECK_HANDOFF_20261020.md`；新增执行方式默认保持普通执行。本页保留前次测量与当时服务状态。

## 已确认的原因

1. 最新任务 `04c96199a306` 使用 384 维、6 层、FFN 1536、batch 8、累积 4、BF16，共 26,360,654 参数。第 2 轮前 300 batch：主机墙钟 80.687 秒，等数据 1.435 秒，传输主机计时 14.617 秒，计算主机计时 63.930 秒；近期 3.4–3.9 batch/s。此时主要瓶颈已经不是等待数据。主机分项会承担异步 CUDA 排队时间，不能当作纯 DMA 或 GPU 核心时间。
2. 设备为 RTX 5070 Ti Laptop、12 GiB，PyTorch `2.7.0+cu128`，cuDNN 9.7.1。该 wheel 明确报告 `Torch was not compiled with flash attention`。默认实际使用 CUTLASS efficient attention，短剖析中其反向约占 CUDA 核心时间 41%；单批约 2,666 次 CUDA 内核启动，也存在 CPU 调度开销。
3. 训练输入按几十个字段分别锁页、分配和传输；部分时间字段在到 GPU 后才转 FP32。
4. 生成桥接器把 CPU 已知的 token 从 CUDA 读回，时钟行未变也重复上传；边界诊断和候选分数分别多次读回。因果逐步决策本身也限制了并行度。
5. 每个局部曲线查询重建整曲积分，并分别执行八次窗口端点搜索。训练已经传 FP32 时间，它又无条件升到 FP64。
6. 最新启动预热耗时 536.015 秒。即使缓存完整，仍会重新组装全部 14,142 个训练样本，包括当轮遮罩、规划标签和档案。
7. 最新驻留日志 loaded=0、memory_skips=20546，主机 RAM 保留量设置为 16 GiB。配置 18 GiB 驻留预算不等于实际已驻留 18 GiB；此时仍依赖映射缓存与运行期 LRU。本轮保留用户设置的内存保护量。

## 训练执行修改

- `stream/runtime.py::attention_backend` 支持 auto / cudnn / efficient。自动仅在 Windows、Blackwell 或更新设备、cuDNN≥9.7、缺少 Flash 内核且 PyTorch 支持优先级配置时优先尝试 cuDNN；其余环境保留 PyTorch 默认。其他 SDPA 实现仍可作为选择回退。硬件/软件升级后应重新比较速度。
- WebUI **“注意力计算后端”**已接入 HTML、payload、CLI 转发、checkpoint 回填与恢复默认。当前设备选“自动（按设备选择）”即可；CPU 只能选自动。能力标记 `stream_runtime_optimization=packed_transfer_sdpa_v1`，前端拒绝向旧后端静默提交新选项。
- `training_batch.py::pack_batch` 在后台 collator 跳过未使用字段、完成原有时间 FP32 转换、按 dtype 放入少量 pinned slabs；每字段偏移保持 256 字节对齐。GPU 每个 slab 复制一次，再恢复字段视图。每批存储独立，不在 DMA 未完成时复用。同步 CPU 路径保留，Fraction/token 表示不转换。
- `StreamBatchPrefetch` 接入后台批处理转换，字节统计包括 slab 填充。
- decoder 显式声明已有三角掩码的 `tgt_is_causal=True`，避免自动比较掩码。
- 日志周期抽样一批 CUDA events，WebUI 显示**“抽样 batch CUDA 流：传输 / 前后向”**。只在日志点等待完成。这是流经过时间，含流内等待，不是核心忙碌百分比；前后向项不含优化器更新。保留原主机计时用于对照。

## 缓存预热修改

- `MappedInputCache.validate_existing()` 使用正式读取相同的身份、校验和、张量和有限值检查。
- `warm_input_cache()` 对唯一确定性输入各验证一次，完整缓存不再重新组装样本；缺失或损坏时调用原构建路径。没有放松检查，没有改变缓存 schema/identity，没有删除旧文件。
- 日志新增 `validated_artifacts` 与 `rebuilt_samples`。小数据验证完整缓存构建次数为 0，缺失/损坏条目能重建。未重跑全量 1787 张谱面，因此不能给出全量启动节省秒数。

## 生成执行修改

- 新增 `stream/attention.py`，仅 stream 使用：交叉注意力查询只投影 Q，固定 memory 首次只投影 K/V；保留原参数和 KV cache 布局。每 step 的 additive mask 各层复用。没有修改 event_ctx/song_form 的共享 helper。
- `GroundedInference` 使用已知 CPU 前缀 token 更新时钟；未提供前缀的原生调用仍可读取 token。时钟 numpy 行未变时复用已上传张量。
- phrase 四个窗口的八个端点一次搜索与 gather，生成初始化时缓存整曲累计积分。训练保留传入时间精度，原生 FP64 调用仍保留 FP64；池化规则与窗口范围不变。
- 细节音频全曲均值在 bank 初始化时缓存，避免每次档案检索重算。
- 合并边界 logits、plan 与诊断读回，合并候选选择与概率读回，保留原抽样法及 RNG 抽样次数。
- 保留完整 Fraction 搜索、候选预算、安全检查和质量分支。没有把生成默认改成 BF16：本机短测中逐 token autocast BF16 反而比 FP32 慢。

## 性能证据及范围

固定合成输入：384 维 / 6 层 / FFN 1536、batch 4、1024 decoder tokens、8192 detail nodes、BF16 前后向；不包含真实数据准备、优化器更新和保存。

| 测量 | 修改前 | 修改后 | 范围 |
| --- | --- | --- | --- |
| 前后向中位时间 | 111.223 ms | 102.856 ms（原默认后端） | 约 8% 吞吐提升，存在短测噪声 |
| 前后向，优先 cuDNN | 111.223 ms | 78.333 ms | 约 1.42 倍吞吐，不代表整轮同比提速 |
| 固定形状传输 | 1.614 ms | 1.468 ms | 约 10% 吞吐提升，不含后台 CPU 打包成本 |
| 同进程交替对照：40 token 缓存解码 | 165.706 ms | 151.588 ms | 约 9% 吞吐提升；末尾 logits 最大差异 2.384e-7 |
| 同进程交替对照：40 次 phrase 查询 | 149.832 ms | 19.040 ms | 约 7.9 倍；样例结果差异 0，不是整曲生成倍数 |

短基准峰值 allocated 约 1.53–1.58 GiB，是合成形状的进程张量峰值，不是实际 batch 8 全训练显存预算。顺序解码、CPU 语法及检查器仍可能让生成 GPU 不满载，不能只看任务管理器占用图验收速度。

原始报告位于 `E:\simai_transformer_backups\runtime_speed_before_20261020`：`baseline.json`、`optimized_auto.json`、`optimized_cudnn.json`、`inference_comparison.json`。注意 `optimized_auto.json` 在 auto 规则改为本机优先 cuDNN **之前**记录，代表原 PyTorch 默认后端；最终本机 auto 按已测的 cudnn 策略运行。

## 验证、兼容及后续

- 第一阶段定向回归 219 passed，包含因果性、训练/缓存等价、完整解码、时钟、质量目标和 API 合同。
- 后续缓存/时钟改动定向回归 42 passed，与前述重叠，不相加。
- 最终新增用例已覆盖 packed 传输、顺序预取、旧/新投影与 cache 等价、独立 phrase 参考、缓存损坏重建、时钟张量复用。cuDNN/efficient BF16 前后向及 padding 掩码对照 2 passed；新增 padding 用例最初使用不匹配的 FP32 bias，修正为有效 BF16 bias 后通过，生产计算无需因此修改。
- WebUI Node 最终 12 passed；`git diff --check` 通过。未进行全量测试、正式训练或整曲质量验收。
- 模型参数、标签和训练语义仍为 `chart_stream_training_v13_bounded_cycle_exit`。已有 v13 权重可按原规则 resume；更早 v12 仍遵循前次交接，使用 finetune 与新目录。后端可能改变舍入与 dropout 随机序列，不承诺逐位复现。
- 建议先保持 batch 8 / 梯度累积 4、数值精度自动、注意力计算后端自动、预取 2 / workers 4；再依据实际 batch/s 和 CUDA 抽样决定是否调整 batch。
- 源码备份在 `E:\simai_transformer_backups\runtime_speed_before_20261020\stream`。此前空白修复交接仍为 `docs/handoffs/STREAM_BLANK_GAPS_HANDOFF_20261020.md`。

## 服务状态：需要手动重启

7860 重载前已确认无活动训练、生成、解压，当前服务 PID 11224。自动审批拒绝了停止并重启服务的命令，仅返回 `blocked by policy`，因此没有重试绕过。**7860 后端尚未加载本轮变更，需要用户手动重启 WebUI 并刷新浏览器。**新前端的能力检查会提示旧后端尚未加载速度优化。

7861 本来未运行，本轮没有启动；正式训练保持停止。服务状态与 PID 可能随后变化，应重新核对。没有自动重启训练任务。
