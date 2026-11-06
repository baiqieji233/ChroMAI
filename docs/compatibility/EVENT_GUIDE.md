# event_ctx_v1 事件模型指南

`event_ctx_v1` 是当前 WebUI 的默认训练与生成主路径。它在基础 `event_v1` 事件序列上加入跨窗口上下文，和旧的 fixed-frame generator 不是同一个模型族。两者不能共享 checkpoint，也不能把旧模型权重直接解释为 event 模型。

## 先确认边界

- checkpoint schema 必须是 `maichart-event-generator-checkpoint-v1` 才能走 event 生成 CLI。
- 旧 `generator_last.pt`、旧 `maichart-conditioned-generator-checkpoint-v2` 或纯权重文件只走 `maichart.generation.cli`；它们是旧固定帧基线，不会被自动升级成 event 模型。
- 旧 checkpoint 不能用于 event 的 `resume` 或 `finetune`。要使用当前主路径，必须重新准备 event manifest 并从头训练 `event_ctx_v1`。
- 一百余张谱面适合验证预处理、训练、保存、加载、生成流程和小样本实验，不足以证明泛化、难度控制或生成质量。
- 生成结果即使能被解析器读取，也不等于音乐对齐、难度达标、手感可玩或质量合格。必须查看 `quality.json`，再进行人工检查和试玩。

## Event 预处理

event 预处理不需要旧 fixed-frame 的 `frame_labels`、`alignment_reports` 或 `division`。它自行读取原始 `maidata.txt` 与音频，建立事件训练所需的缓存和 manifest：

```powershell
python -m maichart.training.prepare_events RAW_ROOT `
  --cache-dir PATH `
  -o MANIFEST `
  --sample-rate 22050
```

可选参数：`--force` 重建已有 event 缓存；`--limit N` 只处理前 N 个样本，用于冒烟测试。`--sample-rate` 的默认值是 `22050`；文档中的完整写法是 `--sample-rate 22050`。Windows PowerShell 的反引号只是换行符，也可以把命令写成一行。

输出仍使用公共 manifest schema：

```json
{
  "schema": "maichart-training-manifest-v1",
  "preprocessing": {
    "target_schema": "event_v1",
    "feature_profile": "event_audio_v2_rhythm",
    "sample_rate": 22050
  }
}
```

每个 song 应包含可追溯的 `maidata_path` 和 `audio_path`。每个 difficulty 的 `usable_for_training` 必须理解为 event 预处理器的结果；旧 fixed-frame 的这个字段不能作为 event 训练是否拒绝样本的唯一依据。event 训练 CLI 会重新读取 manifest 指向的精确 source，并重新执行自己的 parser/QC。

旧基线仍使用原命令：

```powershell
python -m maichart.cli preprocess build RAW_ROOT `
  --cache-dir PATH -o MANIFEST `
  --division 16 --sample-rate 22050
```

WebUI 的模型族选择决定预处理入口：

| 模型族 | WebUI 预处理命令 | 需要 fixed-frame / alignment |
| --- | --- | --- |
| `event_ctx_v1` | `maichart.training.prepare_events` | 不需要 |
| `event_v1` | `maichart.training.prepare_events` | 不需要 |
| `legacy_frame_v2` | `maichart.cli preprocess build` | 旧路径按原合同需要 |

## Event 训练命令

```powershell
python -m maichart.training.train_events `
  --manifest PATH `
  --validation-manifest PATH `
  --checkpoint-dir PATH `
  --epochs 20 `
  --batch-size 2 `
  --segment-beats 16 `
  --d-model 256 `
  --nhead 4 `
  --num-layers 4 `
  --lr 0.0002 `
  --grad-accumulation 1 `
  --device auto `
  --seed 42 `
  --save-every 5
```

冒烟实验可以额外使用 `--max-steps N` 和 `--validation-max-steps N` 限制训练/验证批数；正式训练必须去掉这两个限制。完整的真实 MP3 测试记录见 `docs/explanations/EVENT_TEST_CASES.md`。

`--validation-manifest` 可以省略。训练和验证应按歌曲划分，不能把同一首歌的不同难度分到两边。长路径可以直接传 Windows 绝对路径，例如 `E:\simai_transformer\manifests\event_train.json`；WebUI 的多行路径框不会截断长路径。

断点与微调：

```powershell
# 继续完整训练状态
python -m maichart.training.train_events ... --resume checkpoints/event_v1/event_last.pt

# 只加载同一 event schema 的权重并重新开始优化器
python -m maichart.training.train_events ... --init-checkpoint checkpoints/event_v1/event_best.pt
```

`--resume` 和 `--init-checkpoint` 不能同时使用。两者都拒绝旧 fixed-frame schema；`finetune` 不是格式转换工具。

标准输出文件名：`event_last.pt`、`event_best.pt`、`event_epoch_NNN.pt`。`event_best.pt` 只有在训练程序有可比较的验证指标时才应理解为“验证指标较好”的版本；它不代表已通过 gameplay 或人工质量验收。

## Checkpoint 合同

WebUI 的 checkpoint 检查会用 `torch.load(..., map_location="cpu")` 读取并检查 `schema`。它不修改 shared `training/checkpoints.py`，只在 WebUI route 做 event 字段适配。

event checkpoint 至少应能明确报告：

```text
schema = maichart-event-generator-checkpoint-v1
model_config
data_contract.target_schema = event_v1
```

训练与验证时，event 只关心 event 模型合同、音频 profile、source 和自身 QC；不要向 event UI 暴露 fixed-frame 的 presence/button threshold，也不要要求 `division` 才能生成 event 模型。

## Event 生成命令

```powershell
python -m maichart.generation.event_cli `
  --checkpoint PATH `
  --audio AUDIO_PATH `
  --output OUTPUT_PATH `
  --output-ir OUTPUT_IR_PATH `
  --condition-file CONDITION_JSON `
  --first 0 `
  --guidance-scale 1 `
  --candidates 1 `
  --seed 42 `
  --device auto `
  --title "Generated maimai chart" `
  --artist "" `
  --chart-level "" `
  --designer AI `
  --tempo-mode auto `
  --temperature 0.8 `
  --max-tokens 2048
```

可选项：`--bpm` 配合 `--tempo-mode constant` 时填写单一 BPM；`auto` 可以留空。`--sample` 显式开启随机采样，默认不启用。`--candidates` 默认 `1`，大于 `1` 是手动 opt-in，不会自动排序、修复难度或筛选可玩性。`--audio` 可以是上传后得到的本机路径，也可以是直接粘贴的绝对路径。

WebUI 会先按 schema 选择 `event_cli` 或旧 `generation.cli`。如果 checkpoint schema 未知，生成会被拒绝，不会猜测模型类型。

event checkpoint 如果包含训练时保存的 `atom_catalog`，生成器会将它作为字符级
解码的语法安全网：只有能继续组成训练数据中合法 note atom 的字符前缀才会被允许，
只有现有 Simai note parser 能完整识别的 note 才能关闭为 `<event_end>`。这不会把
整张谱面或整种 note 当成不可组合的大词，`/` 同拍组合和 `*` Slide 分支仍然保留
组合式生成能力。旧的、没有 `atom_catalog` 的 event checkpoint 会退回较宽松的
字符约束，但生成失败仍不会被当成成功谱面。

chart_stream 的 note grammar 会将同一 note cell 内的 `*` 同头 Slide 分支限制为最多 2 条；
普通 event/song_form 路径默认保持旧 atom catalog 行为。普通 `/` 同拍多押不受此上限影响。
需要诊断旧 event atom catalog 时，可在 CLI 显式使用正数 `--max-star-slides`。

## 质量报告与验收

event 生成任务应产生输出谱面、ChartIR 和 sidecar `quality.json`。WebUI 会显示报告是否存在，并展示诊断字段；报告缺失时，任务不能被当作质量检查完成。

如果解码器没有在限制内完成事件序列，或严格 Simai 编译失败，CLI 会返回非零状态并写出 `.quality.json` 与 `.error.json`，不会写入一个假谱面。音频浮点时长只用于终点 `E` 的有界有理化；note 事件位置仍使用精确 Fraction。

`maichart validate`、`maichart parse --timing/--notes/--ir` 和 `maichart stats`
会在旧解析器报告“beat 无法映射到整数 tick”时自动切换到 exact-event 解析器。
这只是兼容入口，不会把事件重新量化；因此有理 beat 的派生 `tick` 可能是 `null`。

最低检查流程：

```powershell
maichart validate OUTPUT_PATH --difficulty 5
maichart parse OUTPUT_PATH --difficulty 5 --ir
```

这两步只检查语法和可解析性。还必须检查：

1. `quality.json` 中的事件数量、时间范围、异常和诊断警告；
2. 首拍、BPM/变速与音频是否对齐；
3. 事件密度、同时按键、Hold、Slide、Touch 是否符合目标歌曲和手部负担；
4. 目标等级只是元数据和软条件，不能当作真实难度测量；
5. 至少在兼容编辑器中打开，并对训练外歌曲人工试玩。

“语法通过”只能说明文件能被当前解析器读取，不能说明谱面可用。条件强度、温度和候选数都是实验旋钮，不是质量保证。
