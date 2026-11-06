# v14 双手特征与训练流水线效率交接

日期：2026-10-21。工作分支：`codex/v11-quality-fixes-20261020`。

## 故障与边界

用户已停止 v17 训练。最新任务 `d68d2c7e2466` 使用 batch16 / 累积2，早期只有约 0.73–0.87 batch/s；数据队列等待不长，但 CPU 准备的 Python 工作会争用主线程调度。系统提交内存接近上限、驻留不足也是需要控制的因素，不能把所有开销归到 GPU 运算或磁盘带宽。

修改前源码、脚本、测试与文档备份：`E:\simai_transformer_backups\v14_speed_before_20261021`，基线 Git `1fbd365`。未启动正式训练、未改权重或删除缓存；没有使用子智能体。

## 阶段工作

- [x] 核对停止状态并备份。
- [x] 缓存按键分组几何，去除反复 dataclass 相等比较。
- [x] 被 MASK 重置的双手历史不再求解无效原谱前缀；长 Slide 时钟单独保留。
- [x] 不再在训练查询 LRU 保留未使用的拍位文本副本；磁盘缓存与精确事件不变。
- [x] 阶段完成后验证特征/标签一致性、既有梯度和因果遮罩、生成缓存路径。
- [x] 以真实 v14 + batch16 + 并发预取做完整短流水线对比，记录边界。
- [x] 完成交接；提交号由本文件所在 Git 提交确定。

## 不改变的训练内容

保留 v14 motor_profile、8 条潜在分手上限、连续单手扫键与交替双手解释、精确 Fraction、所有 loss/遮罩/RNG 规则和用户设置的 batch。优化目标是执行和内存，不通过关闭双手特征或缩短历史提高速度。

## 已确认的原因

1. 双手状态搜索此前在 Python 中按事件、分手候选、按键分组多重循环。预取线程与主线程共享 GIL；预取队列看似已有批次，CUDA 调度仍可被抢占，因此“数据等待很小、计算主机计时很大”不能证明纯 GPU 算力不足。停止任务日志中采样 CUDA 计时也明确是 stream elapsed，不是 kernel busy。
2. 随机抽到后半首的样本时，旧实现先求解整段原谱历史，再应用会清空双手状态的 MASK；大量已被重置的信息白算。现在从第一条查询之前最近的重置后开始，长 Slide 的持续时间仍从完整已提交前缀获得。
3. 分组几何、dataclass 相等比较、重复查询的最终特征读出都有多余工作。现在缓存有限按键组合的几何和角度，使用已去重状态的对象身份选择，在同一查询调用内复用 `(committed_count, second)` 相同的结果。
4. 旧任务确实命中了磁盘缓存，`rebuilt=0`；命中不等于文件不被打开或不再读页。仅约 1.54 GiB 输入成功驻留，剩余输入走映射/LRU。不要把磁盘活动直接归咎于校验失败。
5. 查询 LRU 保留了无人读取的拍位字符串，且驻留池和回退 LRU 的预算独立叠加。此前观察到 Windows 提交量约 50.3/50.9 GiB：物理可用内存不等于仍有提交容量，可能触发换页或分配失败。

## 原生后端与边界

新增 `stream/native_hand_tracking.py`，使用可选 Numba，将最重的双手候选搜索和回放编译为释放 GIL 的 CPU 循环。使用 float64、`fastmath=False`，保留原来的去重、稳定排序、8 条候选和多解保留顺序；最终 24 维特征仍调用既有 `features()`。

- 训练 CLI：`--hand-backend auto|python|native`，默认 auto。
- WebUI：**双手特征计算后端**，对应“自动（可用时原生加速）”“Python（兼容／对照）”“原生 CPU（需要 Numba）”。
- 训练在启动数据线程前预热 JIT，并以 `stream_prepare.stage=hand_backend` 和 `stream_runtime.hand_backend` 报告有效后端。auto 不可用时明确报告 fallback；显式 native 不可用时给出错误。
- 可选依赖：`python -m pip install -e ".[training-speed]"`。本设备已安装 Numba 0.68.0，本次有效后端为 native，使用编译缓存的预热约 0.92 秒；首次编译会更久。
- 推理保持增量 Python HandHistory，得到几何缓存和重复读出的通用优化；不在生成时为不断增长的历史重建原生歌曲数组。本次没有测整曲生成加速比例。

## 内存及磁盘处理

- 训练映射查询不再把未使用的拍位文本长期留在 LRU；原磁盘元数据、谱面 Fraction、音频查询中心及 float 精度不变，无需清空缓存。
- 对驻留池持有的相同缓冲区，查询/结构 LRU 只持引用，不再重复计入张量字节。条目自身的小量 Python 元数据仍有内存开销。
- 回退查询预算按实际驻留量扣除，重复准备使用原配置预算，避免逐次扣减；驻留预载不会再次复制、累计计费同一个 key。
- stream 驻留预载在 Windows 上同时看物理可用内存和提交余量，继续尊重用户的系统预留。Form 原有物理内存检查默认行为不变。
- 全驻留状态回归已断言取样不重新打开输入文件。部分驻留仍可能读盘；不承诺硬盘始终零访问。

## 实测结果

脚本：`scripts/benchmark_stream_v14_speed.py`。结果原件：`E:\simai_transformer_backups\v14_speed_before_20261021\v14_native_pipeline_comparison.json`。

本机 RTX 5070 Ti Laptop 12 GiB，PyTorch 2.7.0+cu128，真实训练集前 32 张谱面抽样，384 维/6 层/FFN1536，batch16、累积2、BF16、融合 AdamW、cuDNN 优先、预取2/线程4；每侧冷态和热态各16批次。先准备相同确定性磁盘输入，两侧均无缓存重建。冻结备份中的双手实现与 `_mapped_query` 作为 before，其余公共流水线使用当前版本；不是两个独立完整旧/新环境的全轮实验。两侧重新建立内存查询、结构及双手状态缓存。

| 指标 | before | after（native） |
| --- | ---: | ---: |
| 冷态批次/秒 | 0.868 | 1.681 |
| 冷态样本/秒 | 13.89 | 26.89 |
| 热态批次/秒 | 1.386 | 1.860 |
| 热态样本/秒 | 22.18 | 29.76 |
| 热态16批次耗时 | 11.54 s | 8.60 s |
| 热态预取样本准备累计时间（含并发） | 32.49 s | 21.87 s |
| 无预取、固定输入计算中位数/批次 | 0.291 s | 0.270 s |
| 峰值已分配 CUDA 显存 | 5.984 GiB | 5.982 GiB |

热态吞吐提高 **34.2%**，相应短测耗时降低 **25.5%**。冷态吞吐提高约93.6%，包括少做无效双手前缀的收益。已抽查的真实批次所有字段张量完全一致（`rtol=atol=0`）。GPU 无数据准备时仍更快，说明尚存在 CPU 准备/调度开销；没有承诺 GPU 持续100%或全数据集也能保持同一比例。更早的首轮 before 冷测包含160次缓存重建，已不用那一轮作为最终对比。

## 对训练结果的影响

不改模型结构、标签、有效 batch、损失、数据指纹、训练语义、输入遮罩、随机增广及随机数调用顺序。本次是执行等价优化；没有通过降低音频节点数、删掉双手特征或缩短历史换速度。原生与 Python 特征、完整数据批次均有零容差比对，包含随机遮罩、跨 epoch 和镜像。CUDA 本身的非确定性仍不保证多次训练权重逐位相同；测试也不构成新模型成品音乐质量的证明。

既有 v14 checkpoint 无需因本次后端切换而新建模型族；是否允许 resume 仍由既有数据/训练合同校验决定。更老合同仍使用 finetune 和新目录，不绕过校验。

## 验证范围

完成最后一阶段修改后集中执行：

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_stream_v14_speed.py tests/test_stream_causal_motor_v14.py tests/test_stream_grounding.py tests/test_stream_pipeline.py tests/test_stream_residency.py tests/test_stream_training_speed.py tests/test_stream_training_contract.py tests/test_stream_runtime_optimization.py
node --test tests/test_stream_generation_ui.js
```

Python **120 passed**，Node **12 passed**。覆盖原始分手参考实现、长Slide/乱序查询/MASK/重复查询、原生数值状态、输入批次/数据指纹、驻留别名及重复预载、提交内存约束、可选后端回退、遮罩和梯度、缓存式生成注意力及界面选项传递。未运行全仓测试、全轮训练或整曲生成。短测速只创建临时随机模型，不读取、覆盖或保存用户 checkpoint。

## 使用建议与服务状态

1. 先重启7860 WebUI并刷新页面。检查时它仍报告旧的 `bounded_training_graph_v2`；源码现在是 `native_hand_pipeline_v3`。新前端会拒绝向旧后端静默提交，需用户手动重启服务后使用。没有停止服务或自动恢复正式训练。
2. **双手特征计算后端**选“自动”即可；日志必须显示 `effective=native` 才表示原生加速启用。想明确避免回退可选“原生 CPU”。
3. 可以先保留**批大小16／梯度累积2、训练预取批次数2、训练数据准备线程4、CPU算子线程0（自动）**，这是本次直接测过的组合。没有证据支持盲目增到8个数据线程或不断加预取。
4. **输入驻留方式**优先自动；**输入驻留内存上限（GiB）**是上限，不会预占。原16 GiB上限并未实际驻留满；本次不建议为了提速一味扩大预算或降低系统预留。先关闭不需要的内存大户，观察实际 loaded/total、memory_skips 和系统提交余量，再决定是否调整。
5. 正式训练启动后，以一段稳定的样本/秒、缓存重建数、提交内存和近期 batch/秒判断；任务管理器单个GPU图百分比只做辅助。

## 文件索引

- `stream/hand_tracking.py`、`stream/native_hand_tracking.py`：等价双手后端和回放。
- `stream/symmetry.py`：有限按键组合角度缓存。
- `stream/data.py`、`stream/input_cache.py`、`form/audio_residency.py`：内存与驻留预算。
- `stream/train.py`、WebUI server/API/static：后端选择、日志和运行能力合同。
- `tests/test_stream_v14_speed.py`、`tests/test_stream_generation_ui.js`：本次新增/补充回归。
- `scripts/benchmark_stream_v14_speed.py`：短流水线复现实验。

未使用子智能体；`.workbuddy/` 不纳入提交。
