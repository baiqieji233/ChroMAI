# Composer (`chart_composer_v1` / `audio_grounded_v3`) 训练性能与 GPU 利用率排查与优化交接文档

> 后续复审：第一轮优化及 Astra 复审见 [性能复审与继续优化](COMPOSER_PERFORMANCE_FOLLOWUP_20261028_ZH.md)。本文档已追加 **第二轮针对“训练时 GPU 占用率周期性拉升后掉到 0% 及硬盘读写开销大”的根因定位与修复记录**。

> 最新接续：[缓存预热复审与重复音频计算优化](COMPOSER_UNIQUE_AUDIO_PERFORMANCE_20261028_ZH.md) 保留本文全部优化，新增精确查询的批内共享、跨缓存目录预热修正，以及 ABBA/BAAB 两轮测量和完整回归结果。

**日期**：2026-10-28  
**目标硬件**：NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM) + Windows 11  
**基准任务**：`chart_composer_v1`（`audio_grounded_v3` 架构，`batch_size=96`，`grad_accumulation=6`）  
**状态**：已完成第一轮 + 第二轮全链路性能优化：
- **Cold Epoch 1（首次访问各歌曲，含每 ~23 batch 的 4 曲轮换）**：单 batch 耗时 **`206.8 ms`（`464.2 samples/s`）**，step 1 之后主线程最大数据等待 `max_data_wait = 1.0 ms`，预取队列始终保持充盈（`min_buffered = 59 MB`），彻底消除周期性掉 0%。
- **Warm Epoch 2+（歌曲表已缓存）**：单 batch 耗时 **`162.4 ms`（`591.2 samples/s`）**，平均 `data_sec = 3.0 ms/batch`（`max_data_wait = 1.7 ms`）。
- **全量缓存预热完成**：`prism_composer_v19` 训练集（2,666 张谱）与验证集（63 张谱）共缺失的 234 个音频节点库与 231 个 `BeatThis` 结构分析库已全部通过 CUDA 预热落盘（`465/465` 100% 就绪）。

---

## 一、 第二轮排查：为什么训练中 GPU 周期性拉升后掉到 0%，且伴随大量磁盘读写？

通过对 `webui_workspace/logs/20261028_185038_composer_train_d0a7689df8e4.log` 的逐窗口剖析与 CPU/I/O 采样，定位到导致 GPU 周期性掉 0% 与磁盘读写风暴的 **4 个深层根因**：

### 1. 后训练新谱触发训练循环内 CPU 单线程惰性 `BeatThis` 推理 + 音频建库（导致 10–25 秒级 GPU 完全掉 0% 与磁盘集中写入）
- **现象**：日志从 `batch=2825` 到 `3625` 期间，每当 `bank_builds` 增加 `1` 或 `2` 时，单窗口（25 batches）的 `data_sec` 从 `1.3s` 暴涨至 **`9.3s–25.4s`**，期间 GPU 掉到 **0%** 长达 10–25 秒，同时磁盘发生数十 MB 的 `.npz` 读取与 `.npy` 写入。
- **根因**：
  - 新纳入的后训练谱面（`datasets/train_post` 与 `datasets/train_high_ds`）尚未建立 `composer_audio_banks` 与 `composer_structure_v6`（`BeatThis`）磁盘缓存。
  - 当 `ComposerDataset.__getitem__` 在训练途中首次抽到这些歌曲时，[`beat_this_frames`](file:///e:/simai_transformer/src/maichart/composer/structure_backend.py#L128) 默认在 **`device='cpu'`** 上且受限于 `--cpu-threads 2` 运行 `BeatThis` RoFormer 神经网络，单首歌曲阻塞唯一的后台取样线程 **10–13 秒**（而同一首歌在 CUDA 上仅需 **`0.4s`**，相差 **25–30 倍**）。
  - 同时，[`structure._build`](file:///e:/simai_transformer/src/maichart/composer/structure.py#L199) 在未缓存的 `_LazyAudio`（`store.budget == 0`）上连续访问 `.times`、`.features`、`.duration_sec` 等属性，导致同一个 15 MB `.npz` 压缩包被重复读取并 `zlib` 解压 6 次。
- **修复**：
  - 在 [`src/maichart/composer/train.py`](file:///e:/simai_transformer/src/maichart/composer/train.py#L203-L300) 中新增 [`_prewarm_missing_song_banks`](file:///e:/simai_transformer/src/maichart/composer/train.py#L203-L300)，在正式进入 Epoch 1 训练循环前自动扫描训练集与验证集缺失的音频节点库和 `BeatThis` 结构库，自动挂载 `MAICHART_BEAT_THIS_DEVICE=cuda` 一次性在 GPU 上高速补齐所有缺失缓存，随后卸载 `BeatThis` 模型并清空显存缓存（当 100% 命中时扫描仅需 `0.12s`）。
  - 在 [`src/maichart/composer/structure.py`](file:///e:/simai_transformer/src/maichart/composer/structure.py#L199-L203) 的 `_build` 入口处一次性物化 `_LazyAudio`，避免建库时重复解压 `.npz`。
  - 在 [`src/maichart/webui/server.py`](file:///e:/simai_transformer/src/maichart/webui/server.py#L714-L725) 中接入 `audio_and_structure_banks` / `audio_banks` / `structure_banks` 预热阶段进度显示。

### 2. 每首歌轮换时触发 5 次完整 `.npz`（`~100 MB`）磁盘读取与 `zlib` 解压 + 488 字节步幅 `mmap` 随机缺页中断
- **现象**：即便磁盘缓存全部存在，在 Epoch 1 每隔约 23 个 batch（`SongLocalBatchSampler` 轮换 4 首新歌时），CPU 耗时与磁盘读取仍会出现明显尖峰。
- **根因**：
  1. **隐式 5 次 `.npz` 磁盘解压**：由于 `ComposerDataset` 将底层 `FormDataset` 的 `audio_cache_gb` 设为 `0.25 GB`（实际 `store.budget == 0`），`song.audio` 是每次属性访问都会重新调用 `load_rational_audio(cache_path)` 的 `_LazyAudio`。在 [`_song_table_unlocked`](file:///e:/simai_transformer/src/maichart/composer/data.py#L461) 访问 `song.audio.duration_sec`（1 次）以及在 [`structure._identity`](file:///e:/simai_transformer/src/maichart/composer/structure.py#L139) 访问 `audio.source`、`audio.metadata`、`audio.sample_rate`、`audio.hop_length`（4 次）时，**每首新歌仅为了获取元数据就从磁盘读取并完整 `zlib.decompress` 了 5 次 15–20 MB 的 `.npz` 特征数组（每首歌白白解压 ~100 MB）**！
  2. **488 字节步幅 `np.memmap` 随机二分查找引发数千次 4KB 磁盘缺页中断**：`composer_audio_nodes_native_v3` 的 `.npy` 文件采用结构化 dtype `[('values', '<f4', (120,)), ('times', '<f8')]`（单行步幅 488 字节，单文件 ~14 MB）。此前 [`ComposerDataset.__getitem__`](file:///e:/simai_transformer/src/maichart/composer/data.py#L834) 先在步幅 488 字节的 `mmap` 视图 `times` 上运行 `AcousticContext` 的 3,000 次 `np.searchsorted`，在 `E:` 盘上触发数千次随机 4KB 缺页中断（耗时 `~30 ms/song`），之后才提取连续数组 `_raw_times_contig`；且 [`LensBank.__init__`](file:///e:/simai_transformer/src/maichart/composer/audio_lens.py#L34) 对已校验的缓存库重复执行全表 `np.isfinite` 扫描与 11 MB 临时数组拷贝。
- **修复**：
  - 重写 [`structure._identity`](file:///e:/simai_transformer/src/maichart/composer/structure.py#L139-L176) 与 [`ComposerDataset._audio_duration_sec`](file:///e:/simai_transformer/src/maichart/composer/data.py#L426-L445)，仅从 `.npz` ZIP 目录中读取 200 字节的 `metadata` 头部或复用 `_song_audio_ends`，**彻底消灭每首歌 5 次 `.npz` 大数组解压（0 次 `.npz` 特征读取）**。
  - 在 [`ComposerDataset.__getitem__`](file:///e:/simai_transformer/src/maichart/composer/data.py#L833-L845) 与 [`_grounded_features`](file:///e:/simai_transformer/src/maichart/composer/data.py#L980-L999) 中，**先**顺序提取连续时间轴 `raw_times_contig`，再传入 `AcousticContext` 与 [`LensBank(..., validated_bank=True)`](file:///e:/simai_transformer/src/maichart/composer/audio_lens.py#L34-L59)，将冷盘构建耗时从 `32.4 ms/song` 压降至 **`14.5 ms/song`**，并省去每首歌 11 MB 临时内存分配。

### 3. `_song_table_unlocked` 与 `structure_features` 的 CPU 计算热点
- **根因**：
  - [`_song_table_unlocked`](file:///e:/simai_transformer/src/maichart/composer/data.py#L451) 对每首歌 ~550 个事件逐个调用 `np.r_`、`np.diff`、`np.median`、`views.future_mask` 与 `views.window_weight`（内部每次调用 `np.broadcast_arrays` 达 ~1,700 次/曲），导致单曲建表耗时 **`74 ms/song`**（4 曲同时轮换即卡顿 `~300 ms`）。
  - [`structure.py`](file:///e:/simai_transformer/src/maichart/composer/structure.py#L517-L685) 中每个样本重复计算段落锚点的 `log1p`、三角函数和未排序的 `_LOCAL_OFFSETS` 二分查找。
- **修复**：
  - 在 [`src/maichart/composer/event_views.py`](file:///e:/simai_transformer/src/maichart/composer/event_views.py#L103-L154) 中为 `window_weight` 与 `future_mask` 增加标量快速路径与无拷贝掩码。
  - 在 [`src/maichart/composer/data.py`](file:///e:/simai_transformer/src/maichart/composer/data.py#L451-L590) 中预计算全曲 `all_gaps` 与 `transition_weights` 快速路径，将 `_song_table_unlocked` 从 `74 ms/song` 降至 **`18.6 ms/song`（提速 4.0x）**；同时优化 `ComposerDataset.__init__` 候选覆盖率扫描循环，复用 `views.seconds` 与单 BPM 线性算术替代 143 万次 `Fraction` 拍钟转换。
  - 在 [`src/maichart/composer/structure.py`](file:///e:/simai_transformer/src/maichart/composer/structure.py#L517-L685) 中预计算 `2 * num_segs` 锚点状态表、使用二倍角公式计算 `(4., 8., 16.)` 小节相位，并对 `_LOCAL_OFFSETS` 排序查询。
  - 在 [`src/maichart/composer/events.py`](file:///e:/simai_transformer/src/maichart/composer/events.py#L110-L133) 中仅对 `seen` 激活键位计算 `log1p`，在 [`src/maichart/composer/prefix_state.py`](file:///e:/simai_transformer/src/maichart/composer/prefix_state.py#L23-L47) 中消除单拍位查询的冗余零张量分配。

### 4. `pause_workers()` 门控范围过宽导致后台工作线程时间预算不足（每跑完一个 batch 队列见底掉 0%）
- **根因**：
  - 第一轮优化中，`batches.pause_workers()` 包裹了 `H2D 传输 + materialize + _forward_batch + _loss` 全过程（约 `62 ms/batch`），导致后台唯一的取样线程每 batch 只有 `backward` 的 `~98 ms` 可以工作。
  - 在 Epoch 1 冷启动阶段，当准备一个 batch 需要 `~140 ms` 时，`98 ms` 的可用窗口不足以在下一个 `next(batches)` 前产出新 batch，导致预取队列 `buffered_tensor_bytes` 跌至 `0 MB`，每次 `backward` 结束后 GPU 都必须停下来等待 CPU 补齐剩余的 `40–50 ms`，表现为 GPU 占用率高频锯齿状掉 0%。
- **修复**：
  - 在 [`src/maichart/composer/train.py`](file:///e:/simai_transformer/src/maichart/composer/train.py#L891-L892) 中，当开启 `prefetch_batches` 时将 `sys.setswitchinterval(0.0005)` 从默认 `5 ms` 收紧至 `0.5 ms`（消除长时切片抢占）。
  - 在 [`src/maichart/composer/train.py`](file:///e:/simai_transformer/src/maichart/composer/train.py#L1169-L1180) 中，将 `batches.pause_workers()` / `batches.resume_workers()` 精确收窄至仅包裹 **`H2D 传输 + audio_materializer.materialize`（仅 `~11 ms`）**，在进入 `_forward_batch` 前立即恢复后台工作线程。这样后台线程在每个 batch 周期内拥有 `forward + loss + backward` 共 **`~150 ms`** 的充裕时间窗口，而主线程在 `_forward_batch` 期间得益于 `0.5 ms` GIL 切换间隔仍保持极高发射速率。

---

## 二、 第一轮核心瓶颈诊断与修改回顾

1. **GIL 门控的 CPU/GPU 异步重叠调度**：[`src/maichart/form/layout_prefetch.py`](file:///e:/simai_transformer/src/maichart/form/layout_prefetch.py#L46-L157) 与 [`src/maichart/composer/train.py`](file:///e:/simai_transformer/src/maichart/composer/train.py#L1169-L1180)。
2. **GPU 侧音频物化器算子合并与零拷贝切片**：[`src/maichart/composer/device_audio.py`](file:///e:/simai_transformer/src/maichart/composer/device_audio.py#L14-L230)。
3. **消除 `AudioGroundedComposer._encode_planner` 中的 GPU-CPU Stream Sync**：[`src/maichart/composer/grounded_model.py`](file:///e:/simai_transformer/src/maichart/composer/grounded_model.py#L121-L265)。
4. **Astra 复审与增强**：[`src/maichart/composer/metric_totals.py`](file:///e:/simai_transformer/src/maichart/composer/metric_totals.py) 向量化指标累加、历史有效宽度预裁剪、按类型批量查找起音中心、修复 `_DeviceBank.nbytes` 独立 storage 显存统计（详见 [性能复审与继续优化](COMPOSER_PERFORMANCE_FOLLOWUP_20261028_ZH.md)）。

---

## 三、 第二轮实测性能结果（RTX 5070 Ti Laptop GPU）

测试配置：真实 `checkpoints/prism_composer_v19/data_split/train_manifest.json` 数据集，`audio_grounded_v3`，`batch_size=96`，`grad_accumulation=6`，`prefetch_batches=8`，`sample_workers=1`，`cpu_threads=2`，`bf16`。连续测量 **36 个 batches（3,456 个样本，跨越多次 4 曲轮换）**：

| 指标 | 第二轮修复前 (`d0a7689df8e4.log` 实测) | **第二轮修复后：Cold Epoch 1（首次访问歌曲）** | **第二轮修复后：Warm Epoch 2+（歌曲表已缓存）** |
|---|---:|---:|---:|
| **单 Batch 总耗时 (`ms/batch`)** | `248 ms – 1,216 ms`（新曲轮换/建库时剧烈抖动） | **`206.8 ms`** | **`162.4 ms`** |
| **训练吞吐量 (`samples/sec`)** | `79 – 387 samples/s` | **`464.2 samples/s`** | **`591.2 samples/s`** |
| **`data_sec`（主线程等待数据耗时）** | `52 ms – 1,016 ms/batch`（每次等待期间 GPU 掉 0%） | **`13.2 ms/batch`（Step 1 后最大等待仅 `1.0 ms`）** | **`3.0 ms/batch`（Step 1 后最大等待仅 `1.7 ms`）** |
| **预取队列最低水位 (`buffered_tensor_bytes`)** | `0 MB`（频繁抽干） | **`59 MB`（全程保持 6–7 个 batch 充盈）** | **全程无饥饿等待** |
| **未缓存歌曲 `BeatThis` + 音频建库停顿** | `10,000 – 25,400 ms / 25 batches`（CPU 惰性建库） | **`0 ms`（训练前已由 CUDA 一次性预热完毕）** | **`0 ms`** |

---

## 四、 回归测试结果

- **Composer 与预取专项回归测试**：`tests/test_chart_composer_v1.py`、`tests/test_composer_audio_bank.py`、`tests/test_composer_device_audio.py`、`tests/test_composer_structure.py`、`tests/test_composer_structure_enhanced.py`、`tests/test_composer_event_views.py`、`tests/test_composer_rebatch_resume.py`、`tests/test_form_layout_prefetch.py`、`tests/test_webui_fields.py` 全部通过。
- **JavaScript 回归测试**：`162 pass, 0 fail`。
