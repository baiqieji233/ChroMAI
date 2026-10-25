# Composer GPU供给与训练效率优化（2026-10-26）

## 当前运行与保留点

- 任务`4f37670023b0`，主界面`http://127.0.0.1:7870/`，目录`checkpoints/Prism_composer_v11`。
- 从原任务`3000c12ed5d9`的第1轮/2,000更新恢复，已经验证第2,500更新完整检查点保存成功。
- batch64、累积4、有效batch256、BF16、fused AdamW、学习率2e-4、目标第80轮保持不变。运行参数改为**CPU算子线程2、采样准备线程1、预取2**；sample_locality仍4，未改变样本顺序策略。
- 第2,000步权重和完整优化器备份：`checkpoints/Prism_composer_v11_preserved_20261026_step2000/composer_last.pt`，SHA-256：`0521c6e5c301249b665013fe70fd34093df5e548fb2f7d81690b74273cb7d7a1`。
- 第2,500步校验：模型配置和dataset fingerprint与备份相同，batch_in_epoch=10000、sample_in_epoch=640000、sampler_batch_size=64、can_resume=true。恢复起点sample cursor=512000；后续500更新正好消耗128000个样本。

## 原因

原配置16个PyTorch CPU线程、4个sample workers。停止前1,125个microbatch耗时389.797秒，约184.71样本/s；data等待96.276秒、compute主机计时287.275秒、传输3.478秒。约四分之一时间在等数据，预取队列经常为空。

compute包括CPU发射和同步，不是纯GPU内核时间。同一真实batch已驻留GPU时，microbatch中位约133–138ms；原正式运行compute约255ms/microbatch，CPU准备与训练线程并行时存在额外调度竞争。总CPU占用约1.9个核，并非全机CPU耗尽；很多工作是Python和小数组操作，多线程不自动带来有效并行。本轮未随意去掉同音频准备锁。

原每次更新检查全部state tensors，逐Tensor执行isfinite/all，产生大量小kernel和CPU发射。当前模型274个Tensor原检查约21.4ms；261个浮点Tensor使用批量L-infinity norm检测约0.73ms。检查本身是必需的，本轮优化执行方式而没有取消检查。

原可用主存约4.1GiB，音频cache实际预算降至约2.1GiB，配置12GiB不能绕过安全余量。未观察到E盘饱和或分页风暴。GPU处于P0，采样时无当前温度降频；100W软件功耗上限在繁忙时会触发，但不能解释CPU等待和空队列。未修改硬件功耗、温控或缓存安全预算。

## 修改与等价验证

1. `_nonfinite_tensors`按device/dtype批量计算浮点L-infinity norm，保留异常检查和原名称顺序。L-infinity不做平方，避免大有限值因平方溢出误判；空Tensor、整数/bool及不支持的dtype/layout/device有明确处理或原谓词fallback。
2. `AcousticContext.trailing/interval`合并重复积分和插值查询，保留逐元素浮点算式；song table缓存duration，避免热路径反复读取lazy metadata。
3. 新增CLI `--cpu-threads`和主界面“CPU算子线程”，范围0–256；0保留环境/PyTorch设置。resume未显式指定时继承保存值，仍可明确调整，不属于模型合同。主界面同时回填保存的sample workers，避免优化后的checkpoint恢复时丢掉运行配置。

安全检查覆盖FP16/BF16/FP32/FP64、NaN/正负Inf、最大有限值、长向量首中末、空Tensor、非连续、complex和混合设备。相同checkpoint、优化器状态及RNG的完整4microbatch更新，在确定性验证模式下loss、全部梯度和更新后参数逐值一致；确定性模式没有设为生产默认。普通CUDA本来存在部分backward非确定性，另有old-old控制。

单独安全检查替换，正常执行模式old/new交替各3轮、每轮16microbatch，平均2.2231→2.1061秒，约快5.56%。CPU查询改动的边界与4chart/256sample全部字段exact；有效重复中，16线程/4workers热准备中位0.894→0.857秒，2/1为0.644→0.589秒。一次包含bank重建的测量已标明并重测，未计入收益。

## 端到端与实际续训

相同8张谱、相同采样索引、batch64×累积4，CPU预取与GPU共同执行；各场景两个独立进程，每次32microbatch，以下为剔除首4个后的28个平均。各场景采用相同父权重及一致的新建优化器，仅用于吞吐诊断，不冒充正式续训轨迹。

| 场景 | 时间 | 吞吐 | 相对基线 |
| --- | ---: | ---: | ---: |
| 原代码，16线程/4workers | 8.383s | 213.77样本/s | — |
| 优化代码，16/4 | 7.945s | 225.55样本/s | +5.51% |
| 优化代码，2/1 | 5.703s | 314.22样本/s | +46.99% |

正式任务使用原优化器和原样本游标恢复，已连续完成500更新并保存第2,500步。恢复后约2,025microbatch累计约259.75样本/s，原观察窗口约184.71，现场窗口约快40.6%。两个生产窗口歌曲不同，重启后缓存和主存状态也不同；受控对照以上表为准，不保证整轮或80轮都获得相同比例提升。

GPU仍有间歇空闲：恢复后16次采样约4%–59%，平均32.25%。不能宣称已满载，也不能拿优化前单次14%/60%直接相除。较少的无用检查kernel和更快的吞吐，比单独追求利用率数字更有意义。

恢复后loss累计窗口重新开始，不能把显示值下降解释为学习质量改善。本轮训练目标、模型结构、学习率没有改变。

## 验证和证据

- Python相关回归549通过、1项需要本机数据开关而默认跳过；该真实数据等价项已另行显式运行通过。
- Node回归32通过；真实Edge检查CPU线程继承、编辑、提交前重检保留及实际运行线程回显。
- 第2,000步备份哈希未变；第2,500步模型配置/数据指纹相同，完整优化器和游标可恢复。
- 后续继续观察更长区间data等待、主存余量、allocated/reserved峰值及每轮耗时，不据短测盲目增大batch或移除检查。

本机证据目录`.scratch/composer_gpu_efficiency_20261026/`：`pre_pause_task.json`、`preserved.json`、`runtime_summary.json`、`finite_training_ab.json`、`finite_training_deterministic.json`、`input_optimization_summary.json`、`e2e_summary.json`、`regression.log`、`cpu_threads_browser_result.json`、`resume_request.json`、`live_resume_latest.json`、`resumed_checkpoint_validation.json`。模型、音频、日志和基准批次不上传GitHub。
