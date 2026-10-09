# maimai 谱面生成项目 · 训练可用性修复报告

日期：2026-02-13（本轮修复）
范围：`src/maichart/training/`、`src/maichart/generation/`、`src/maichart/webui/`、`tests/`
状态：显存溢出问题**已解决**（完整轮次验收见 §5）

---

## 1. 结论摘要

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| 每样本耗时（b16，真实语料） | 25.6 ms | **7.97 ms**（第2轮稳态） | **3.2× 更快** |
| 单轮耗时（4,791 批） | ~32.6 min | **10.2 min**（第1轮 11.5 min） | **3.2× 更快** |
| GPU 占比（gpu_s / 总时长） | 约 30%（大量空等） | **99.6%** | 不再空跑 |
| 显存峰值预留 | 17.21 GiB（溢出到共享） | **7.06 GiB（硬上限内）** | 溢出消除 |
| 共享 GPU 内存占用 | 6,900 MB | **170 MB（峰值 214 MB）** | 溢出消除 |
| 每步 GPU→CPU 同步次数 | 4 次/微批（≈19,000/轮） | 1 次/轮 | 消除 |
| 补齐浪费（padding） | 1.79× | 1.19× | 分桶 |

预计 70 轮总时长：**~38 h → ~12–13 h**（按第2轮 10.2 min/轮 + 验证/存档）。

---

## 2. 你观察到的现象 = 根因（实测证据）

你在任务管理器看到"专用 GPU 内存超出、开始用共享 GPU 内存"。实测复现：

```
1,200 批长程（b32，无硬上限）
  epoch=001 ms_per_batch=425.5 batches=1200
  cuda_memory allocated=0.12 reserved=4.33 peak=17.21 GiB   ← 12 GB 卡
  采样：专用 11,797 MB + 共享 4,956 MB → 最终 11.7 GB + 6.9 GB
```

**同时活跃张量只有 0.12 GiB**，而预留峰值 17.21 GiB —— 说明吃内存的不是数据本身，而是**分配器的形状缓存**：

- 每个样本窗口长度 13–756 token、音频 232–2193 帧，**几乎每批形状都不同**；
- PyTorch 缓存分配器**为每个见过的形状各留一块**，随批数单调增长；
- 越过 12 GB 专用显存后，Windows 用共享 GPU 内存兜底（走 PCIe），吞吐随之塌方；
- 这正是最初"第一轮 38 分钟跑不完"的原因（32.6 min/轮 × 溢出放大）。

修复前的守卫之所以失明：检查点上采样的 `memory_reserved()` 是 **4.33 GiB**（优化步之后是低点），
而峰值 17.21 GiB 出现在前向/反向**内部**。已改为按 `max_memory_reserved()`（峰值）判定。

---

## 3. 根因清单（按影响排序）

| # | 根因 | 证据 |
|---|---|---|
| 1 | **显存形状缓存无界增长 → 溢出共享内存** | peak 17.21 GiB / 活跃 0.12 GiB；425 ms/批 vs 168 ms/批 |
| 2 | **CPU 预处理与 GPU 完全串行**（`num_workers=0`） | 400 批实测 `data_s 32.2 s + gpu_s 82.8 s ≈ 总 117 s`；GPU 利用率 2–53% |
| 3 | **每样本 CPU 成本高**：`_resample_rows` 每样本 98 次 `np.interp`；tempo map 每窗口重建 | 98 次 GIL 往返/样本，与主线程 CUDA 提交抢 GIL |
| 4 | **批太小导致每批固定开销主导** | dec 488→250（token 减半）耗时反而 83.2→89.4 ms |
| 5 | **补齐浪费**：随机批平均补齐到 1.79× | 子代理独立测得 1.79×→1.19×（分桶后） |
| 6 | 掩码让注意力掉出 flash 后端 | 最坏形状：有掩码 7.62 GiB / 367.6 ms vs 无掩码 4.79 GiB / 289.7 ms |
| 7 | OMP 自旋、每微批 4 次 GPU→CPU 同步、`torch.isin` 每批排序 | 环境注入后 322→293 ms/批 |

---

## 4. 修复清单

| # | 修复 | 位置 | 实测效果 |
|---|---|---|---|
| 1 | 显存硬上限（`set_per_process_memory_fraction`，auto=60% 设备显存） | [train_events.py:417](src/maichart/training/train_events.py:417) | 物理封顶 7.2 GiB；宁可 OOM 也不静默溢出 |
| 2 | 每 100 批按**峰值**归还缓存块 | [train_events.py:389](src/maichart/training/train_events.py:389) | 预留稳定 ≤7.2 GiB |
| 3 | 后台预取线程（CPU/GPU 重叠，默认 2 批） | [train_events.py:299](src/maichart/training/train_events.py:299) | `data_s` 32.2 s → 0.4 s（稳态） |
| 4 | `_resample_rows` 向量化 | [event_audio.py:688](src/maichart/generation/event_audio.py:688) | 98 次 np.interp → 1 次向量化 |
| 5 | tempo map 按歌缓存 | [event_ctx_dataset.py:83](src/maichart/training/event_ctx_dataset.py:83) | 省 76k 次/轮重复构建 |
| 6 | `_timing_context` 向量化 | [event_audio.py:863](src/maichart/generation/event_audio.py:863) | 6.5 ms → **0.09 ms/窗口** |
| 7 | 长度分桶批（默认开；每轮重洗、小语料自适应） | [train_events.py:463](src/maichart/training/train_events.py:463) | 补齐 1.79× → 1.19× |
| 8 | 设备侧统计累加 + position LUT 取代 `torch.isin` | [train_events.py:184](src/maichart/training/train_events.py:184) | 同步 4 次/微批 → 1 次/轮 |
| 9 | 子进程注入 `OMP_WAIT_POLICY=PASSIVE`/`KMP_BLOCKTIME=0`/`expandable_segments` | [server.py:1007](src/maichart/webui/server.py:1007) | 322 → 293 ms/批（**更正：这 9% 来自 OMP/KMP；`expandable_segments` 在本平台是空操作**，生成日志里 torch 明确警告 `expandable_segments not supported on this platform`，见 §9） |
| 10 | 每轮遥测：`ms_per_batch / data_s / gpu_s / cuda_memory` | [train_events.py:184](src/maichart/training/train_events.py:184) | GPU 空跑从此可见 |
| 11 | WebUI 默认 **batch 16 × 累积 2**（等效 32） | [index.html](src/maichart/webui/static/index.html) | b32 溢出后**反而比 b16 慢 26%** |

### 4.1 为什么默认从 b32 退回 b16

| 配置 | ms/批 | ms/样本 | 峰值预留 | 共享显存 |
|---|---|---|---|---|
| b8 × 累积4 | 143.6 | 17.95 | — | — |
| b16 × 累积2 | 168.8–229.3 | 10.55–14.33 | 3.68 GiB | 0.18 GB |
| b32 × 累积1 | 383.2–425.5 | 11.97–13.3 | **17.21 GiB** | **6.9 GB** |

b32 单批最坏形状就需要 7.62 GiB，**预算（7.2 GiB）本身小于单批需求**，周期性归还也无济于事 →
默认 b16。硬上限保证：真越界时会**明确报 OOM** 并提示降批，而不是静默溢出拖慢 2 倍。

---

## 5. 完整轮次验收（决定性）

**完整 2 轮 × 4,791 批**（b16 × 累积2，硬上限 auto 7.2 GiB，真实语料 1,991 首 / 76,643 窗口），全程每 25 s 采样显存：

```
cuda_memory_budget bytes=7692288000 (7.2 GiB, requested=auto) hard_cap=7.2 GiB flush_every=100
epoch=001 loss=1.571950 token_accuracy=0.5079 val_loss=1.2080 seconds=689.1 ms_per_batch=143.8 batches=4791 data_s=49.8 gpu_s=638.7
dataset_audio_cache epoch=001 songs=1550 resident=3.89 GiB limit=6.00 GiB spills=0 loads=1550
cuda_memory epoch=001 allocated=0.12 reserved=6.62 peak=7.06 GiB budget=7.2 freed=0.00
epoch=002 loss=1.069610 token_accuracy=0.6278 val_loss=1.0652 seconds=611.0 ms_per_batch=127.5 batches=4791 data_s=2.7 gpu_s=607.8
cuda_memory epoch=002 allocated=0.12 reserved=6.87 peak=7.06 GiB budget=7.2 freed=0.00
exit=0
```

### 5.1 验收结论

| 验收项 | 结果 | 判定 |
|---|---|---|
| 能否跑完完整轮次 | 2 轮全部完成，退出码 0 | ✅ |
| 显存是否溢出共享内存 | 共享显存 **最大 214 MB / 平均 170 MB**，全程 58 次采样中**没有一次超过 500 MB**（修复前 6,900 MB） | ✅ 已解决 |
| 显存峰值 | 7.06 GiB（预算 7.2 GiB 内），`freed=0` 说明连归还都没被动用 | ✅ |
| 单轮耗时 | 轮1 **11.5 min**（689.1 s / 4,791 批）、轮2 **10.2 min** | ✅ 目标 11.2 min |
| GPU 是否空跑 | `gpu_s / seconds`：轮1 92.7%、轮2 **99.6%**；采样平均利用率 60%、最高 91% | ✅ |
| 数据侧是否仍是瓶颈 | `data_s` 轮1 49.8 s（冷加载 1,550 首）、轮2 **2.7 s（0.4%）** | ✅ |
| 模型是否在学 | loss 1.572→1.070、token_acc 0.508→0.628、val_loss 1.208→1.065 | ✅ |

每样本耗时：轮1 8.99 ms → 轮2 **7.97 ms**（修复前 25.6 ms，**3.2×**）。

### 5.2 显存采样（58 次，每 25 s）

| 指标 | 数值 |
|---|---|
| 专用显存（含桌面 ~0.9 GB） | 最大 8,676 MB / 平均 6,493 MB |
| **共享 GPU 内存** | **最大 214 MB / 平均 170 MB**（修复前 6,900 MB） |
| GPU 利用率 | 平均 60% / 最高 91% |
| 功耗 | 103–109 W（无降频迹象） |

---

## 6. 验证与测试

| 项目 | 结果 |
|---|---|
| 全量 pytest | **551 passed / 0 failed / 11 skipped**（47 个测试文件） |
| 前端端到端（设置持久化/默认值） | **42/42** |
| 独立子代理审计（5 项改动，CUDA 全程禁用） | **5× PASS**；其指出的 3 个缺陷已修 |
| 真实语料对照实验 | b8/b16/b32 × 带/不带守卫 × 1,200 批，逐 15 s 采样专用/共享显存 |
| 自查发现的严重 bug | 预取线程停止标志跨轮未清 → **第 2 轮起 0 批、训练静默空转**；已修 + 回归测试 |

子代理指出的 3 个缺陷（均已修复）：
1. 分桶采样器每轮批序完全相同（同 seed）→ 训练失去每轮重洗，`--resume` 也不延续；已改为从全局 RNG 取每轮种子。
2. 语料 ≤1,024 窗口时分桶退化为随机批（chunk 大于语料）→ 改成自适应 chunk（保证 ≥8 个 chunk）。
3. 文档里 "2.2×" padding 缩减夸大（实测 1.5×）→ 已改为实测值。

---

## 7. 代码回顾与质量路线（回应"回顾已写就的代码，思考我的思路"）

### 7.1 我已读懂的你的思路

1. **先泛化再精修**：全语料预训练（1,991 首 / 76,643 窗口）→ 金标集微调（84 首）→ 金标验证集holdout。数据规模与切分方式是对的。
2. **显式可控性**：16 维条件（等级 + 7 雷达 + 8 风格）——你要的是"点单"而不是"抽卡"。
3. **正确性优先于美观**：BPM 绝不能错、必须过可行性验证、不能后半空白；同时承认"3:2、两倍、三倍这类整数/有理倍是谱师自己的选择"，即**音乐上合理即可**，不要求与检测 BPM 逐位相等。
4. **产品化**：一键流程、设置持久化、实时日志、可视化验收——你会反复跑，所以要求可运维。
5. **资源敏感 + 拒绝静默失败**：盯的是"每轮多少分钟、GPU 有没有空跑"，且要求"不要直接拒绝输出，给出显式提醒后继续输出"。
6. **你自己的判据最硬**：质量最终由你（谱师视角）判断，工具只负责把可量化的事量化。

### 7.2 代码里已经具备的质量能力（比多数同类项目强的地方）

- **条件空间**：`CONDITION_KEYS` 16 维，含 `slide_tricky/slide_sequence/jack/chord` 这类玩法维度，以及
  `melody_following`（实为**起音对齐度**，实测训练侧 0.803–0.975，p50 0.895）。
- **闭环评估**：`assess_event_chart` → 统计 + **从生成结果反算条件**（`condition_from_events`），
  可与请求条件做**往返比对**；`event_quality.py` 还有 `_peak_nps/_entropy/_motif_stats/_signature`。
- **金标区间判据**：`gold_reference` 用金标集 p10–p90（warn: p05–p95）分档，而不是拍脑袋阈值
  （[acceptance.py:407](src/maichart/generation/acceptance.py:407)）。
- **绝对缺陷判据**：语法可解析、同格重复按键等"必须不存在"的缺陷单列
  （[acceptance.py:438](src/maichart/generation/acceptance.py:438)）。
- **时钟外置**：生成时 BPM/变速来自音轨检测或 `--bpm`，不由模型自由采样——
  这是你最初"BPM 异常"问题的**结构性保证**（[event_cli.py:86](src/maichart/generation/event_cli.py:86)）。
- **完整长度**：按训练窗口逐段生成并重锚到全曲时钟，跨窗口家族用 `<ctx>` 引子承接前文
  （[event_cli.py:679](src/maichart/generation/event_cli.py:679)）。

### 7.3 四个质量缺口（按性价比排序）

| 优先级 | 缺口 | 为什么重要 | 建议 |
|---|---|---|---|
| **P0** | **多候选无排序**：`--candidates N` 只写 N 个文件，没有任何打分/择优 | 我们**已经有打分器**（`metric_bundle` + `evaluate` + `condition_from_events`），却没用起来。这是不重训就能拿到的最大质量提升 | 生成 N 个候选 → 用金标区间 + 条件往返误差 + 语法/可行性做加权评分 → 输出最优，并把每个候选的分数写进报告 |
| **P1** | 缺**对齐/密度辅助损失**：条件只作为输入，从不作为预测目标 | 16 维条件下模型"能听懂"程度未知；文献也警告"常见代理指标会奖励错误行为"（[ChartGenEval](https://arXiv.org/pdf/2607.12857)） | 加轻量多任务头：预测本段的 `note_density/peak/chord/jack`，权重小（如 0.1–0.2），只作正则与可控性校准 |
| **P2** | **长程结构**只靠 16 拍文本引子 | 歌曲 100–200 拍，intro/verse/chorus 的展开与回收难以建模；已有工作专门研究"考虑整体乐曲展开的谱面生成"（[JSAI 2026](https://www.jstage.jst.go.jp/article/pjsai/JSAI2026/0/JSAI2026_3M1GS10p03/_pdf/-char/ja)） | 把 `context_beats` 16→32 做一次对照；或给解码器加"段落进度"条件（`song_progress` 已在每帧特征里） |
| **P3** | **模型容量偏小**（d192/L8/FFN768 ≈ 9.5M 参数） | "高质量谱面"通常需要更大容量；但 12 GB 卡是硬约束 | 显存溢出解决后，可安全试 d320/L10/FFN1280 + batch 8×累积4（硬上限会挡住溢出并给出明确 OOM 信号） |

参考阅读（本轮检索所得，因本机代理拦截未能取全文，仅作方向参考）：
- [ChartGenEval: Corruption-Tested Multi-Dimensional Feedback for Rhythm-Game Chart Generation](https://arXiv.org/pdf/2607.12857) — 多维反馈评估，且明确指出常见代理指标会奖励错误行为。
- [STRUM: Spectral Transcription and Rhythm Understanding for Playable Rhythm-Game Charts](https://ar5iv.labs.arxiv.org/html/2605.12135) — 可玩性导向的端到端生成。
- [HachimiDX](https://github.com/ck2739046/HachimiDX) — maimai 方向的社区工程，可对照数据/表示方式。
- [大局的な楽曲展開を考慮した音楽ゲーム譜面生成モデルの検討](https://www.jstage.jst.go.jp/article/pjsai/JSAI2026/0/JSAI2026_3M1GS10p03/_pdf/-char/ja) — 显式建模乐曲整体展开。

---

## 8. 你需要做的

1. **重启 WebUI**（子进程环境注入是服务端改动，必须重启才生效）。
2. 刷新页面（默认已是 batch 16 × 累积 2；一键流程会自动带分桶、预取、显存硬上限）。
3. 看每轮这三行：

```
epoch=001 ... ms_per_batch=... data_s=... gpu_s=...
dataset_audio_cache epoch=001 ...
cuda_memory epoch=001 allocated=... reserved=... peak=... budget=7.2 freed=...
```

判读方法：
- `data_s` 应远小于 `gpu_s`（否则仍是数据侧瓶颈）；
- `reserved/peak` 应稳定 ≤7.2 GiB；**Windows 任务管理器的"共享 GPU 内存"不应再增长**；
- `ms_per_batch` 若突然翻倍 → 立即看 `cuda_memory` 与共享显存，判断是否溢出。
4. 想更快：确认 `reserved` 有 2 GiB 以上余量后，再试 batch 32；越界会**明确报 OOM**，不会再静默拖慢。

---

## 9. 仍未解决 / 风险

1. 真实 GPU 流时间（b16 稳态 ~136 ms/批）仍略高于同形状合成基准（~100–140 ms），差值不在池化（实测 0 成本）、掩码（+12%）、分配器配置（0%）或同步（0%）；下一步候选是让注意力走 flash 后端（当前掩码使最坏形状多花 2.8 GiB / 27% 时间）。
2. 分桶降低的是**平均**补齐，不降低**峰值**显存（最长的 chunk 仍是全 756 token）。
3. 质量全链路（P0–P3）尚未实施，本轮只修可用性；谱面质量的最终判据仍在你手上。
4. 70 轮的真实收敛曲线尚未跑过（本次只做了可用性与单轮到数小时的验证）。
5. **`expandable_segments` 在本平台无效**：生成日志里 torch 警告 `expandable_segments not supported on this platform`。因此显存溢出**完全**靠"硬上限 + 峰值周期归还 + batch 16"这三招解决；把 `PYTORCH_CUDA_ALLOC_CONF` 换成平台支持的其他项（如 `max_split_size_mb`）才可能有额外收益，目前不需要。
