# WebUI 新手使用说明

WebUI 是本地运行的训练和生成界面。它不需要用户编写代码，但仍然需要准备 Simai 谱面、音频和 Python 依赖。页面中的长路径使用多行文本框，Windows 路径可以直接粘贴。

多训练集管理入口是顶部的「训练素材」：勾选多个来源参与联合训练，进入单个来源查看标题、文件夹、谱师和质量标注；支持排序与输入页码跳转。新增素材后使用已有 checkpoint 做 finetune 的操作见 [多训练集管理与加强训练](../explanations/MULTI_DATASET_TRAINING.md)。

## 当前主路径：song_form_v1

当前主工作台默认使用 `song_form_v1`。它使用全曲 blueprint、段落关联记忆、真实 Radar/Style 条件监督和局部负荷曲线，不使用旧 event 的固定窗口、Slide/Touch 过采样或 `condition_dropout` 合同。

主页面仍可从“模型族”切换到 `event_ctx_v1`、`event_v1` 或 `legacy_frame_v2`。切换后训练、生成、强度曲线和高级参数会恢复对应模型族的设置；训练素材、质量权重、checkpoint 路径、输出和日志流程共用。

song_form 的正式训练建议使用完整已准备清单，模型宽度 256、4 层、8 头、token 2048、记忆 token 512、batch 4、梯度累积 4、学习率 0.0002、Dropout 0.1、权重衰减 0.01、20–30 epochs。训练控制中的 Radar/Style 条件监督、局部负荷损失和段落关联损失只在新合同 checkpoint 中有效；早期短训 checkpoint 会在页面显示能力未校准。

旧 event 兼容路径说明如下。

当前页面默认选择 `event_ctx_v1`，预处理按钮会运行：

```powershell
python -m maichart.training.prepare_events RAW_ROOT --cache-dir PATH -o MANIFEST --sample-rate 22050
```

event manifest 仍是 `maichart-training-manifest-v1`，`preprocessing` 记录 `target_schema=event_v1` 和 `feature_profile=event_audio_v2_rhythm`；选择 `event_ctx_v1` 时，模型族由训练与生成入口决定，manifest 合同保持兼容。event 只需要 source 中的 `maidata.txt` 和音频，不需要 fixed-frame 的 frame division、frame labels 或 alignment reports。song 应保留 `maidata_path`、`audio_path`；event 训练会重新读取精确 source 并执行自己的 parser/QC，不会把旧 fixed-frame 的 `usable_for_training=false` 当作拒绝依据。

选择 **旧固定帧基线** 后，页面才使用原来的 `maichart.cli preprocess build`、`train_generator` 和 `generation.cli`。这是显式对照实验，不是 event 模型的兼容模式。旧 checkpoint 不能直接用于 event；请重新训练。

完整的 event CLI、checkpoint schema、续训限制、质量报告和一百余张谱面的实验边界见 [EVENT_GUIDE.md](EVENT_GUIDE.md)。

## 安装与启动

在项目目录打开 PowerShell，首次启动会自动安装依赖：

```powershell
START_WEBUI.bat --no-browser --port 7860
```

已有环境只检查不启动：`START_WEBUI.bat --check`。

启动服务：

```powershell
START_WEBUI.bat --no-browser --port 7860
```

打开 `http://127.0.0.1:7860/`。服务只监听本机。训练、预处理或生成任务运行期间不要关闭启动服务的窗口。

## 第一次实验推荐配置

你当前计划使用一百余张谱面进行可行性测试，建议先使用 event 主路径：

- 模型族：`event_ctx_v1`；
- event 音频特征方案：`event_audio_v2_rhythm`；
- 采样率：22050；
- 训练方式：`fresh`；
- batch size：1 或 2；
- segment beats：16；
- gradient accumulation：4；
- d_model：256；
- Transformer 层数：4；
- 设备：自动；
- 首次 epoch：1。

先证明流程能够稳定完成，再增加训练轮数。一百余张谱面只适合流程和小样本行为测试，不能证明泛化、难度控制或质量。首次可把 `--limit` 设为 1 或 2，再用 1 epoch 验证保存和加载。

## 准备训练集

最小 ZIP 结构如下：

```text
dataset.zip
├─ song_001/
│  ├─ maidata.txt
│  └─ track.wav
└─ song_002/
   ├─ maidata.txt
   └─ track.mp3
```

每首歌目录需要一个 `maidata.txt` 和一个同级音频。Simai 文档真正必须配对的是 `lv_N` 与 `inote_N`；`wholebpm`、`des_N`、`E` 等字段可以缺少。缺少或留空 `lv_N`，或缺少对应 `inote_N`，会使该难度不可用。

训练 ZIP 不需要 style、radar 或 condition 文件。style 和 radar 在处理/训练阶段自动从谱面标签派生，生成时再由页面输入目标值。

### 页面操作

1. 上传 ZIP，等待“上传和解压”完成。
2. 确认页面自动填写的原始目录、缓存目录和 manifest 路径属于这次上传。上传完成不代表已经生成 manifest。
3. 保持模型族 `event_ctx_v1`，点击“预处理训练集”。event 路径会调用 `prepare_events`，不需要 division、frame labels 或 alignment 缓存。
4. 等右侧任务完成后点击“检查训练集状态”。event 会显示 source 与自身 QC 诊断，最终训练仍会重新解析 source。
5. 若结果显示处理错误，查看红色“处理错误”区域。格式异常的当前文档会被跳过，其余文档仍继续处理；修复后重新预处理即可。

上传新 ZIP 或切换已有训练集后，旧的 manifest 检查状态会自动失效。异步检查返回较晚时也不会覆盖当前新选择的数据。

## 三种训练方式

### 从头训练 fresh

选择“从头训练（fresh）”。不填写已有模型 checkpoint。`epochs` 是总训练轮数，默认学习率为 `2e-4`。

### 从断点继续 resume

选择“从断点继续（resume）”，上传或粘贴 checkpoint 路径，点击“检查 checkpoint”。这里的 `epochs` 是**本次额外轮数**：原 checkpoint 在 epoch 20，本次填 5，就表示继续训练 5 轮。

resume 要求 checkpoint 保存了模型、optimizer、scheduler、AMP 和 epoch 等完整状态。页面会把学习率固定为 `2e-4`，不会重置优化器。旧版纯模型权重不能 resume，应改用 finetune。

### 已有模型加强训练 finetune

选择“已有模型加强训练（finetune）”。它只加载已有模型权重，优化器重新开始，默认学习率为 `2e-5`，默认勾选“重置优化器状态”。`epochs` 同样是本次额外轮数，建议先填写 1–5。

已有模型模式会从 checkpoint 自动采用模型结构。请把 checkpoint 输出目录改成新目录，例如：

```text
原模型：checkpoints/generator
加强训练：checkpoints/finetune_100_basic
```

不要覆盖原模型目录，这样才能比较加强训练前后的结果，也能随时回退。

## checkpoint 检查

点击“检查 checkpoint”会向后端发送：

```json
{
  "checkpoint_path": "完整 checkpoint 路径",
  "mode": "resume 或 finetune",
  "manifest": "当前训练 manifest 路径"
}
```

页面预期收到：

```json
{
  "valid": true,
  "can_resume": true,
  "can_init": true,
  "epoch": 20,
  "global_step": 1000,
  "model_config": {},
  "data_contract": {},
  "warnings": [],
  "errors": []
}
```

event 检查内容包括 schema、模型结构、event target schema、音频 profile 和训练状态；旧基线检查内容包括模型输入维度、层数、division、采样率、音频特征方案和 note vocabulary。`valid=false` 或当前模式能力为 false 时，不要开始训练。

## 训练请求中的重要参数

页面会显式发送以下字段：

- `training_mode`：`fresh`、`resume` 或 `finetune`；
- `training_checkpoint`：已有模型路径，fresh 为 null；
- `lr`：fresh 默认 `2e-4`，finetune 默认 `2e-5`，resume 页面固定 `2e-4`；
- `save_every`：保存中间 checkpoint 的间隔；
- `reset_optimizer`：finetune 默认 true；
- `model_family`：默认 `event_ctx_v1`，旧基线必须显式选择 `legacy_frame_v2`；
- `validation_manifest`：event 可选，训练与验证应按歌曲划分；
- `segment_beats`：event 默认 `16`，不使用 `segment_frames`；
- `feature_profile`：event 为 `event_audio_v2_rhythm`，旧基线为 `basic_v1`；
- `condition_dropout`：默认 `0.15`。

后端应根据 `training_mode` 区分初始化、完整恢复和权重微调。前端不会把已有模型模式伪装成 fresh。

## 生成谱面与约束力度

生成区域需要音频、模型 checkpoint 和输出路径。BPM 留空时使用自动音频分析；需要手动覆盖时才填写 tempo map。tempo map 中的 beat 是谱面 beat，不是浮点秒。

目标等级、Radar、Style 和 `guidance_scale` 都是软约束：

- `guidance_scale=0`：基本不强调条件；
- `1`：默认值，兼容旧模型；
- `1–2`：加强目标条件；
- `2–3`：强条件实验，可能降低音乐贴合度或结构自然度。

旧模型没有条件强化能力时，保持 1。高强度不能保证生成结果严格达到等级，也不能凭空产生模型没有学会的长程重复、细碎节奏或 Slide 几何。候选数量默认 1，改大后才手动开启多候选；当前不自动排序，也不运行 gameplay harness。

## 预处理参数能否混用

多份产物可以同时保留，但不能无条件混合：

| 参数 | 建议 |
| --- | --- |
| `division` | 只属于旧 fixed-frame 基线；event 不使用 |
| `sample_rate` | 保持一致；不同值建议单独实验 |
| `feature_profile` | `event_audio_v2_rhythm` 与旧 `basic_v1` 分开训练，不能直接混用 checkpoint |
| cache 目录 | 可以不同，只要每个 manifest 指向自己的缓存 |
| batch、epoch、学习率 | 属于训练配置，可以在不同实验中变化 |

如需合并多个 manifest，先确认数据合同一致、歌曲没有重复，并按歌曲划分训练/验证/测试，避免同一歌曲的不同难度泄漏到不同集合。

## 看到错误时怎么做

- `training manifest 不存在`：先执行预处理，上传 ZIP 本身不会产生 manifest。
- `No usable training difficulties found`：检查 `lv_N` 与 `inote_N` 是否成对、音频和缓存是否存在，再重新预处理。
- `Malformed meter directive` 或其他单个文档解析错误：该文档应出现在处理错误列表并被跳过；修复后再预处理，不应让整个任务卡死。
- `duplicate-slide-head`：检查是否误写成 `4-6[8:1]/4-7[8:1]`；独立 `/` 会在同一位置产生两个 Slide 头，同头分支应使用 `*`。
- checkpoint 不兼容：查看 `errors` 中的字段，确认没有 event schema 与旧 fixed-frame schema 混用；旧模型需要重新训练 event。
- CUDA 不可用：改为自动或 CPU，先确认 PyTorch 和显卡驱动安装正常。

## 最低验收标准

一次可行性实验至少应做到：

1. 一百余张谱面中有一部分能被 event 预处理并显示 source/QC 诊断。
2. 训练 1 个 epoch 不报错，loss 为有限值，产生 `event_last.pt`。
3. 能用 event checkpoint 对未参与训练的歌曲生成 maidata、ChartIR 和 `quality.json`。
4. 生成文本可以被项目解析器读取，没有致命的重复 Slide 头或时间轴错误。
5. 至少人工检查一首训练歌曲和一首未训练歌曲，记录节奏对齐、密度、重复段落、Slide 选择与可玩性；语法通过不等于质量通过。
