# event_v1 测试用例与实测记录

本文档记录事件级模型最小可复现实验。测试分为两层：

- **工程正确性**：语法、时间、数据隔离、checkpoint 和失败报告必须正确。
- **模型质量**：需要足够训练后，再检查音乐对齐、难度、手感和人工试玩。本页的 smoke run 不证明这些质量目标。

## 1. 自动化测试

在项目根目录运行：

```powershell
python -m pytest -q
```

本次实测结果：

```text
258 passed, 11 skipped
```

事件管线专项测试位于：

```text
tests/test_event_pipeline.py
```

它覆盖以下场景：

1. `EventTempo` 对象能够进入音频上下文，BPM 变化按当前 tempo 段积分；
2. `first` 非零时，音频起点与 chart beat 不一致不会导致训练崩溃；
3. 一个坏难度只被跳过，不会污染同一 manifest 中的好难度；
4. `difficulty_index < 4` 在读取缺失 `lv` 前被过滤；
5. `4-6` 这类多字符 note 不会被生成器截断；
6. 生成器必须输出 `<eos>`，超过 `max_tokens` 会明确失败；
7. 生成失败会写 `.quality.json` 与 `.error.json`，不会写一个假的 maidata；
8. 生成成功时会同时写 maidata、ChartIR 和 quality report。

`tests/test_stage3_roundtrip.py` 另外覆盖 exact-event 与旧 CLI 的兼容：包含
`{28}` 产生的 `8/7` beat 时，`parse --timing`、`parse --notes`、`parse --ir`、
`validate` 和 `stats` 会自动避开旧的整数 `1920 tick` 限制。非整数 beat 的
派生 `tick` 会保留为 `null`，不会为了兼容旧接口而把 note 时间量化掉。

生成时还会优先读取 checkpoint 的 `atom_catalog`。它是训练数据中出现过的完整
Simai note atom 集合，解码器只允许进入该集合的合法前缀，并在
`<event_end>` 前调用 note parser。这不是把 atom 变成不可组合的大词：模型仍然
逐字符预测；它是生成阶段的语法安全网，用来避免训练不足时输出未闭合括号、随机
Slide 路径或 directive 混入 note。`/` 同拍组合会按 atom 分段处理，因此 `1/2`
仍然可以生成。

## 2. 训练集事件审计

海鲜杯训练划分的审计命令：

```powershell
python -c "import json; from pathlib import Path; from maichart.training.event_dataset import EventDataset; d=EventDataset('webui_workspace/seafoodcup2_splits/train_manifest.json', cache_dir='webui_workspace/seafoodcup2_event_cache', segment_beats=8, max_tokens=512, eager_audio=False); Path('webui_workspace/seafoodcup2_event_audit.json').write_text(json.dumps(d.audit(), ensure_ascii=False, indent=2), encoding='utf-8')"
```

实测结果：

```text
accepted_documents = 68
skipped_documents  = 51
accepted_segments  = 3345
invalid_chart      = 35
unsupported_syntax = 16
```

验证划分实测结果：

```text
accepted_documents = 8
skipped_documents  = 7
accepted_segments  = 430
```

跳过的文档会保留在 audit JSON 的 `skipped` 中，并带有 `song_id`、难度、阶段、原因和短错误信息。不会输出完整的超大诊断对象，也不会因为单个文档异常卡死整个任务。

## 3. 真实音频训练冒烟

下面的命令使用海鲜杯真实音频，只跑两个训练 batch：

```powershell
python -m maichart.training.train_events `
  --manifest webui_workspace/seafoodcup2_splits/train_manifest.json `
  --cache-dir webui_workspace/seafoodcup2_event_cache `
  --checkpoint-dir webui_workspace/seafoodcup2_event_smoke `
  --epochs 1 `
  --batch-size 1 `
  --segment-beats 8 `
  --max-tokens 512 `
  --d-model 64 `
  --nhead 4 `
  --num-layers 1 `
  --dim-feedforward 128 `
  --grad-accumulation 1 `
  --device auto `
  --no-amp `
  --max-steps 2
```

实测结果：

```text
device = cuda
parameters = 229619
epoch = 001
loss = 6.310582
```

产物目录：

```text
webui_workspace/seafoodcup2_event_smoke/
  event_last.pt
  event_best.pt
  event_epoch_001.pt
  dataset_audit.json
  metrics.jsonl
```

此命令只证明 MP3 特征、数据集、模型 forward、反向传播和 checkpoint 保存连通。两步训练后的 token accuracy 为 0 是正常现象，不能据此判断模型结构不可行。

## 4. 带验证集的冒烟

为了快速检查 validation metric，可限制训练和验证各一个 batch：

```powershell
python -m maichart.training.train_events `
  --manifest webui_workspace/seafoodcup2_splits/train_manifest.json `
  --validation-manifest webui_workspace/seafoodcup2_splits/val_manifest.json `
  --cache-dir webui_workspace/seafoodcup2_event_cache `
  --checkpoint-dir webui_workspace/seafoodcup2_event_validation_smoke `
  --epochs 1 --batch-size 1 --segment-beats 8 --max-tokens 512 `
  --d-model 64 --nhead 4 --num-layers 1 --dim-feedforward 128 `
  --grad-accumulation 1 --device auto --no-amp `
  --max-steps 1 --validation-max-steps 1
```

实测结果：

```text
train loss = 6.355863
validation loss = 5.720019817352295
```

`--validation-max-steps` 只适合冒烟，正式实验必须设为 `0`，让验证集完整跑完。

## 5. 断点继续

使用上一节生成的 checkpoint：

```powershell
python -m maichart.training.train_events `
  --manifest webui_workspace/seafoodcup2_splits/train_manifest.json `
  --cache-dir webui_workspace/seafoodcup2_event_cache `
  --checkpoint-dir webui_workspace/seafoodcup2_resume_smoke `
  --resume webui_workspace/seafoodcup2_event_validation_smoke/event_last.pt `
  --epochs 1 --batch-size 1 --segment-beats 8 --max-tokens 512 `
  --grad-accumulation 1 --device auto --no-amp --max-steps 1
```

实测结果为 `epoch=002`，说明模型权重、数据合同、optimizer、scheduler、AMP scaler 和 RNG 状态能够恢复。若使用不同内容的新数据集，应使用 `--init-checkpoint` 做 event_v1 微调，而不是 `--resume`。

## 6. 真实 MP3 生成

选取海鲜杯原始目录中的任一 `track.mp3`：

```powershell
$audio = (Get-ChildItem webui_workspace/seafoodcup2_raw -Recurse -File -Filter *.mp3 | Select-Object -First 1).FullName
python -m maichart.generation.event_cli `
  --checkpoint webui_workspace/seafoodcup2_event_smoke/event_last.pt `
  --audio "$audio" `
  --output webui_workspace/seafoodcup2_event_smoke/generated/sample_01_maidata.txt `
  --output-ir webui_workspace/seafoodcup2_event_smoke/generated/sample_01.chart.json `
  --quality-report webui_workspace/seafoodcup2_event_smoke/generated/sample_01.quality.json `
  --level 13 --device auto --max-tokens 128 --seed 7
```

两步训练的第一次 greedy 实测结果是：

```text
generation_status=failed
decoder reached max_tokens before emitting <eos>
```

只会写：

```text
sample_01_maidata.error.json
sample_01.quality.json
```

不会写一个看似成功的 `maidata.txt`。这是正确行为，因为当前 checkpoint 尚未学会完整事件序列。

加入 checkpoint atom grammar 后，使用相同 MP3 和 seed 的受约束随机生成已经得到：

```text
generation_status = success
syntax_valid = true
events = 5
notes = 7
fatal_errors = []
```

产物位于：

```text
webui_workspace/seafoodcup2_event_smoke/generated/
  sample_01_constrained_final_maidata.txt
  sample_01_constrained_final.chart.json
  sample_01_constrained_final.quality.json
```

这份结果只表示语法和结构安全，不表示音乐贴合或等级 13 可玩；该 smoke
checkpoint 只训练了两个 batch，生成密度明显不足，仍应归类为工程验证产物。

随机采样是手动开启的实验选项：

```powershell
... --sample --temperature 1.3 --max-tokens 256
```

在加入 atom grammar 之前，随机采样可能产生不符合 Simai 语义的随机 note 字符组合；
现在这些字符在解码阶段就会被屏蔽，或者在严格编译阶段使候选失败，不会落盘为一份
看似成功的乱码谱面。该结果再次说明：语法约束、事件状态机和质量报告可以防止假成功，
但无法替代足够的训练数据和人工验收。

## 7. 多候选测试

多候选仍然是手动 opt-in。本次使用同一首 MP3、`event_last.pt`、`--candidates 3`、
`--sample --temperature 1.3` 实测：

```text
candidate 1: success, syntax_valid=true, events=5, notes=7,  density=0.05 NPS
candidate 2: success, syntax_valid=true, events=5, notes=12, density=0.08 NPS
candidate 3: success, syntax_valid=true, events=4, notes=8,  density=0.05 NPS
```

三份文件都通过了 `maichart validate --strict`。产物保存在：

```text
webui_workspace/seafoodcup2_event_smoke/generated/test_cases/
```

这些候选仍全部是 `quality_status=needs_review`，因为 checkpoint 只训练了两个
batch；候选数只增加搜索样本，不会自动补密度、修正等级或证明可玩。

## 8. 判断标准

一个生成实验只有同时满足以下条件，才可以进入人工试听/试玩：

1. 进程返回码为 `0`；
2. maidata、ChartIR、quality report 三者都存在；
3. `quality.json` 的 `generation_status` 为 `success`；
4. `syntax_valid` 为 `true`，且没有 fatal error；
5. 用 `maichart validate` 和兼容编辑器再次读取；
6. 人工检查 BPM、首拍、重复段、Slide 头、Slide 时长、密度峰值和手部负担；
7. 最终在目标游戏或兼容模拟器中试玩。

`playable_verified` 当前始终是 `false`，因为 gameplay harness 暂未纳入本阶段，不能把模型输出自动标成“可玩”。
