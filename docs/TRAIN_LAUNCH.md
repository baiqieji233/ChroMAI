# 训练启动说明 · event_v7 续训（按你的原参数）

启动时间：2026-02-13 00:29:21
进程：**PID 55608**（独立进程；我的会话结束它继续跑）
启动方式：**在你上次的中断点上续训**（`--resume checkpoints\event_v7\event_last.pt`）
起点：你已完成 **epoch 1**（train loss 1.5103 / val loss 1.1223 / token acc 0.6143），从 **epoch 2** 继续
终点：**epochs 70**（早停 patience 5 会提前结束）

---

## 1. 参数地址（你要的"参数在哪"）

| 项目 | 路径 |
|---|---|
| **逐字可复现的命令** | `.scratch\bpmfix\launch_argv_v7_resume.txt` |
| **checkpoint 输出/续训源** | **`checkpoints\event_v7\`**（`event_last.pt` 续训 / `event_best.pt` 最终生成用 / `event_epoch_NNN.pt` 每 3 轮） |
| 本次运行日志 | `webui_workspace\logs\20261007_002921_train_t002921.log` |
| 错误流 | `webui_workspace\logs\20261007_002921_train_t002921.err.log` |
| 日志元数据（GUI 历史列表读取） | 同目录同名 `.meta.json` |
| 每轮指标 JSONL | `checkpoints\event_v7\metrics.jsonl` |
| 训练集清单 | `webui_workspace\datasets\full训练集\pretrain_manifest.json`（1,991 首 / 76,643 窗口） |
| 验证集清单 | `webui_workspace\datasets\full训练集\gold_val_manifest.json`（52 首 / 2,278 窗口） |
| 音频特征缓存 | `webui_workspace\datasets\full训练集\cache\` |

---

## 2. 参数与你原命令的逐项对照

```
python -m maichart.training.train_event_ctx \
  --manifest "...\full训练集\pretrain_manifest.json" \
  --validation-manifest "...\full训练集\gold_val_manifest.json" \
  --cache-dir "...\full训练集\cache" \
  --checkpoint-dir "checkpoints\event_v7" \
  --epochs 70 --batch-size 16 --grad-accumulation 2 \
  --segment-beats 16 --max-tokens 1024 \
  --d-model 192 --nhead 8 --num-layers 8 --dim-feedforward 768 \
  --dropout 0.15 --condition-dropout 0.15 --weight-decay 0.02 --lr 0.0002 \
  --condition-mask-prob 0.5 \
  --context-beats 16 --audio-margin-beats 8 --context-tokens 320 --overlap-beats 0 \
  --slide-oversample 2 \
  --device cuda --save-every 3 --patience 5 --audio-cache-gb 6.0 \
  --prewarm-workers 16 --seed 2059561939 \
  --resume "checkpoints\event_v7\event_last.pt"        ← 唯一新增：从中断点继续
```

启动日志已确认生效：`max_tokens=1024`、`batch=16 accumulation=2`、`lr=0.0002`、
`dropout=0.15 weight_decay=0.02`、`segment_beats=16`、`d_model=192 layers=8 heads=8`、
`seed=2059561939`、`parameters=9496821`、`target_schema=event_v1`。

### 2.1 与你原命令完全一致的部分（逐项核对）

| 参数 | 你的值 | 本次 | 一致 |
|---|---|---|---|
| `--epochs` | 70 | 70 | ✅ |
| `--batch-size` | 16 | 16 | ✅ |
| `--grad-accumulation` | 2 | 2 | ✅ |
| `--segment-beats` | 16 | 16 | ✅ |
| **`--max-tokens`** | **1024** | **1024** | ✅ |
| `--d-model / --nhead / --num-layers / --dim-feedforward` | 192 / 8 / 8 / 768 | 同 | ✅ |
| `--dropout / --condition-dropout / --weight-decay / --lr` | 0.15 / 0.15 / 0.02 / 2e-4 | 同 | ✅ |
| `--condition-mask-prob` | 0.5 | 0.5 | ✅ |
| `--context-beats / --audio-margin-beats / --context-tokens / --overlap-beats` | 16 / 8 / 320 / 0 | 同 | ✅ |
| `--slide-oversample` | 2 | 2 | ✅ |
| `--device / --save-every / --patience / --audio-cache-gb` | cuda / 3 / 5 / 6.0 | 同 | ✅ |
| `--prewarm-workers` | 16 | 16 | ✅ |
| `--seed` | 2059561939 | 2059561939 | ✅ |
| 训练集 / 验证集 | full训练集 pretrain + gold_val | 同 | ✅ |

### 2.2 本次额外叠加的修复（只影响"跑得快不快/会不会溢出"，不改变优化数学）

| 修复 | 默认值 | 为什么不影响训练结果 |
|---|---|---|
| 长度分桶批 | 开启（`--no-length-bucketing` 可关） | 只改一批里放哪些窗口；样本集合、损失、梯度语义不变 |
| 后台预取线程 | `--prefetch-batches 2` | 只改数据准备的线程位置 |
| 显存硬上限 | `--cuda-memory-budget-gb auto` = **7.2 GiB** | 只限制分配器缓存；越界报 OOM 而不是静默溢出 |
| 缓存周期归还 | `--cuda-cache-flush-batches 100` | 只归还空闲缓存块 |
| 环境注入 | `OMP_WAIT_POLICY=PASSIVE`、`KMP_BLOCKTIME=0`、`expandable_segments` | 线程/分配器行为 |
| 设备侧统计 + position LUT | 内置 | 指标数值等价（子代理已逐位验证），只是不再每微批同步 4 次 |

### 2.3 关键：契约逐字段相等，所以能安全续训

| 契约键 | 你的 event_v7 | 当前环境 | |
|---|---|---|---|
| `target_schema` | `event_ctx_v1` | `event_ctx_v1` | ✅ |
| `max_tokens` | 1024 | 1024 | ✅ |
| `segment_beats` | 16 | 16 | ✅ |
| `input_dim` / `feature_profile` | 108 / `event_audio_v1` | 同 | ✅ |
| `sample_rate` / `hop_length` | 22050 / 441 | 同 | ✅ |
| `dataset_signature` | `9b7126e4bb6c85b5…` | 同 | ✅ |
| `vocabulary_signature` | `3da06d0ddb31971c…` | 同 | ✅ |
| `family_params` | `audio_margin=8, context_beats=16, ctx_tokens=320, overlap=0, dropout=0.15, slide_oversample=2` | 同 | ✅ |
| `condition_params` | `{"default":0.5,"mask_prob":0.5}` | 同 | ✅ |

---

## 3. 怎么盯这次训练

读日志（必须带 `-Encoding UTF8`，否则中文路径显示乱码）：

```powershell
Get-Content "E:\simai_transformer\webui_workspace\logs\20261007_002921_train_t002921.log" -Encoding UTF8 -Tail 30
```

每轮三行：

```
epoch=002 loss=... token_accuracy=... val_loss=... seconds=... ms_per_batch=... batches=4791 data_s=... gpu_s=...
dataset_audio_cache epoch=002 songs=... resident=... GiB limit=6.00 GiB spills=0 loads=...
cuda_memory epoch=002 allocated=... reserved=... peak=... GiB budget=7.2 freed=...
```

判读表：

| 现象 | 含义 | 处理 |
|---|---|---|
| 首行 `epoch=002` | 续训成功，从你已完成的第 1 轮继续 | 正常 |
| `data_s` 接近甚至大于 `gpu_s` | 数据侧瓶颈，GPU 在空等 | 检查磁盘/杀毒；确认分桶与预取开启 |
| `peak` 逼近 7.2 GiB | 缓存顶到硬上限 | 正常；若报 OOM 把 batch 降到 12/8（累积相应调大） |
| `ms_per_batch` 突然翻倍 | 疑似溢出到共享显存 | 看任务管理器"共享 GPU 内存"是否增长（应稳定 ~200 MB 以下） |
| `spills` 增长 | 音频缓存不足 | 提高 `--audio-cache-gb`（31 GB 内存可给 10–12） |
| 健康耗时 | 第 2 轮起 **~10.2–10.8 min/轮** | 明显超出按上表排查 |

---

## 4. 中断与再次续训

```powershell
python -m maichart.training.train_event_ctx `
  --manifest "E:\simai_transformer\webui_workspace\datasets\full训练集\pretrain_manifest.json" `
  --validation-manifest "E:\simai_transformer\webui_workspace\datasets\full训练集\gold_val_manifest.json" `
  --cache-dir "E:\simai_transformer\webui_workspace\datasets\full训练集\cache" `
  --checkpoint-dir "E:\simai_transformer\checkpoints\event_v7" `
  --epochs 70 --batch-size 16 --grad-accumulation 2 --segment-beats 16 --max-tokens 1024 `
  --d-model 192 --nhead 8 --num-layers 8 --dim-feedforward 768 `
  --dropout 0.15 --condition-dropout 0.15 --weight-decay 0.02 --lr 0.0002 `
  --condition-mask-prob 0.5 --context-beats 16 --audio-margin-beats 8 --context-tokens 320 `
  --overlap-beats 0 --slide-oversample 2 --device cuda --save-every 3 --patience 5 `
  --audio-cache-gb 6.0 --prewarm-workers 16 --seed 2059561939 `
  --resume "E:\simai_transformer\checkpoints\event_v7\event_last.pt"
```

`batch-size` / `grad-accumulation` 不属于数据契约，可随时调整；`segment-beats / max-tokens /
context-* / slide-oversample / condition-*` 属于契约，续训必须与原值一致（否则会被
`compare_event_contract` 明确拒绝，这是有意保护）。

---

## 5. 生成谱面（训练结束后）

用 `checkpoints\event_v7\event_best.pt`（按验证 loss 选择）走「生成」面板或
`python -m maichart.generation.event_ctx_cli`。面板默认已就绪：temperature 0.65、top-p 0.9、
重复惩罚 1.0、引导 1.0、窗口 32、等级 13.5、雷达/风格取语料中位数。

---

## 6. 附：本次为对照而跑、现已停止的 event_v8

为验证修复效果，我先按同参数跑过一版 `event_v8`（epochs 80 / prewarm 8，其余相同），已完成 1 轮
（train loss 1.5265 / val loss 1.2023 / 135.6 ms/批），**其 checkpoint 仍保留在 `checkpoints\event_v8\`**，
可作为对照或备用初始化；本次正式训练不使用它。
