# maimai AI Chart Lab 新手完整手册

这份手册面向第一次使用训练程序的人。你只需要会复制文件、打开
PowerShell，并能看懂基本的 Simai 谱面；不需要先学会 Python 编程。

本项目是研究型基线，当前 WebUI 默认使用 `event_ctx_v1` 跨窗口事件模型和
`event_audio_v2_rhythm` 音频特征。它可以尝试把
音频转换为一张可解析的 Simai 谱面，并提供条件控制，但不是官方谱面编辑器，
也不能保证生成结果一定好玩。每张谱面都必须查看质量诊断、解析、人工检查和试玩。
旧 fixed-frame 模型属于显式对照路径，不会自动转换成 event 模型。

推荐的完整顺序是：

```text
安装依赖 -> 准备 ZIP -> 上传并预处理 -> 检查 manifest -> 小规模训练
-> 确认 checkpoint -> 上传音频生成 -> validate -> 在编辑器中人工检查和试玩
```

## 一、先认识几个名词

- **项目目录**：以下命令都应在本项目根目录执行；路径可以是任意盘符。
- **原始训练集**：每首歌的 `maidata.txt` 和对应音频组成的目录。
- **预处理**：event 路径读取 Simai 和音频，生成 event manifest；旧基线才生成 frame labels 和对齐报告。
- **manifest**：预处理产生的 JSON 清单，训练程序通过它找到所有缓存。
- **checkpoint**：event 训练保存 `event_last.pt`、`event_best.pt` 和 `event_epoch_NNN.pt`；旧基线通常叫 `generator_last.pt`。
- **WebUI**：浏览器页面。它只是把相同的预处理、训练和生成程序包装成按钮。

## 二、安装依赖

### 1. 打开 PowerShell

在 Windows 开始菜单搜索 **PowerShell** 并打开。每次输入命令后按 Enter。

```powershell
cd <项目目录>
```

### 2. 确认 Python

```powershell
py --version
```

建议使用 Python 3.10 或更高版本。若系统提示找不到 `py`，需要先安装
Python，并在安装器中勾选把 Python 加入 PATH。

### 3. 安装项目和可选依赖

```powershell
START_WEBUI.bat --no-browser --port 7860
```

首次启动会在项目目录创建独立 `.venv`，并安装测试工具、音频分析库和训练需要的
PyTorch 等依赖。若网络中断，重新执行同一条命令即可。安装完成后可用
`START_WEBUI.bat --check` 只检查环境而不启动服务。

检查 PyTorch 是否能看到显卡：

```powershell
py -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

显示 `CUDA: True` 表示可以使用 NVIDIA GPU。你的 12 GB 显存设备建议从
`d_model=256`、4 层、batch size 1 或 2 开始。如果显示 `False`，程序仍可用
CPU 运行，但音频分析和训练会慢很多；不要在 WebUI 中强制选择 CUDA。

## 三、启动和关闭 WebUI

在项目目录的 PowerShell 中启动：

```powershell
START_WEBUI.bat --no-browser --port 7860
```

看到服务启动提示后，在浏览器打开：

```text
http://127.0.0.1:7860/
```

这个地址只允许本机访问。浏览器可以关闭，但训练任务属于服务器进程；若要
停止服务，回到启动它的 PowerShell 窗口按 `Ctrl+C`。不要为了停止 WebUI
去关闭 Tailscale 或其他网络进程。

如果提示 7860 已被占用，可以换端口：

```powershell
START_WEBUI.bat --no-browser --port 7861
```

然后打开 `http://127.0.0.1:7861/`。

## 四、Simai 最小知识

训练程序不是直接按浮点秒保存 note。它读取 Simai 的节拍结构，并在预处理时
把 note 对齐到项目使用的 beat frame 网格；生成时再把整数 beat/frame 编译回
Simai。BPM 改变的是 beat 到真实秒数的换算，不会把谱面的逻辑顺序打乱。

一个最小 `maidata.txt` 大致如下：

```text
&title=示例歌曲
&artist=示例作者
&first=0
&wholebpm=180
&lv_5=13+
&des_5=Chart Designer
&inote_5=(180){4}1,2/6,3h[4:1],4-8[4:1],A1,5b,E
```

常见字段和符号：

- `&title`、`&artist`：歌曲信息。
- `&first`：谱面 beat 0 相对于音频的偏移，单位是秒；不确定时不要随意改。
- `&wholebpm`：初始 BPM。局部 BPM 可以在谱面正文中用 `(120)` 之类的指令写出。
- `&lv_5`、`&des_5`：第 5 难度（通常对应 Master）的等级和谱师。
- `&inote_5`：第 5 难度正文。
- `,`：读完当前时间点后推进到下一个时间点；因此谱面必须按 Simai 的顺序线性读取。
- `{4}`、`{8}`、`{16}`：后续逗号步进使用的节拍切分。`{16}` 比 `{4}` 更细。
- `1`：普通 Tap；`5b`：Break Tap。
- `1/5`：同一时间点的两个 note。
- `1h[4:1]`：从 1 开始的 Hold，长度由方括号中的 Simai 时值表示。
- `1-5[4:1]`：从 1 到 5 的 Slide 示例。
- `A1`：Touch 区域示例。
- `(120)`：从当前位置起使用 120 BPM；例如 `(120)8`。
- `E`：谱面结束。

以上只是本项目解析器使用的最小示例。复杂 Slide、多段轨迹、Touch Hold 和
特殊修饰符请以你的实际编辑器语法和项目 `tests` 中的解析示例为准。

训练识别难度时，真正必须配对的是 `&lv_N` 与 `&inote_N`。`&title`、`&artist`、
`&wholebpm`、`&first`、`&des_N` 以及 `E` 都不是训练集硬性必需字段。没有
`wholebpm` 时程序会使用默认 BPM，并结合谱面中的局部 BPM 或音频分析继续处理；
没有 `des_N` 时谱师名为空即可。注意：缺少或留空 `lv_N`，或者缺少对应的
`inote_N`，才会使该难度不能被正确识别；`lv_N=?` 仍会作为未知等级保留。

## 五、训练集 ZIP 应该怎么放

最简单的 ZIP 结构如下：

```text
dataset.zip
├─ song_001/
│  ├─ maidata.txt
│  └─ track.wav
└─ song_002/
   ├─ maidata.txt
   └─ track.mp3
```

每首歌使用一个独立目录，并且同一目录内必须有：

1. 一个叫 `maidata.txt` 的 Simai 文件；
2. 一个与它并列的音频文件。支持的常见扩展名包括 wav、mp3、flac、ogg、m4a、
   aac、opus、wma 和 mp4。

ZIP 内可以再有一层总目录，但上传后应确认 WebUI 自动填出的 **原始训练集目录**
的下一层确实能看到 `song_001`、`song_002` 这些歌曲目录。不要把一个 ZIP 再套
进另一个 ZIP，也不要只上传若干散落的 maidata 文件。

项目自带可运行的示范：

- `examples/maimai_training_demo.zip`：可以直接上传的 ZIP；
- `examples/training_zip_demo/README.md`：示范目录和命令；
- `examples/training_zip_demo/raw_dataset/`：已经展开的目录。

训练集 ZIP **不需要** style、radar 或 condition 文件。当前设计会在训练时从
frame labels 在内存中推导 Radar/Style，再把你在生成界面输入的目标向量交给模型。
因此改变 style 推导代码不需要重新给 ZIP 增加 style 字段。需要注意的是，本基线
中的 `melody_following` 和 `pattern_reuse` 目前仍是弱监督/默认控制，不能把它们
理解成已经解决了旋律识别和长程复用。

## 六、用 WebUI 准备训练数据

### 1. 上传 ZIP

打开页面后，按左侧从上到下操作。在 **准备训练集并开始训练** 中点击上传，选择
`examples/maimai_training_demo.zip` 或你自己的 ZIP。上传完成后，**原始训练集目录**
会自动填写；也可以不上传 ZIP，直接把本机目录完整路径粘贴到这个文本框。

**上传成功不等于准备完成。** ZIP 上传只负责解压，缓存与训练清单由下一步生成。
看到清单路径已经填入时，它可能还是一个将要生成的文件，因此现在不能直接训练。

如果已经上传过，无需再上传。刷新页面后，从 **重新选择已上传的训练集** 中选择
对应的文件名与上传编号。每份 ZIP 使用独立目录，避免新谱面读取上一份缓存。

### 2. 设置预处理参数

第一次建议使用：

```text
模型族           event_ctx_v1
event profile    event_audio_v2_rhythm
采样率           22050
预处理样本上限   1 或 2
```

event 路径不使用 fixed-frame division，也不要求 frame labels 或 alignment reports。
第一次只验证流程时设置样本上限可以缩短时间；正式训练时清空它。

点击 **2. 预处理训练集**，等待右侧任务状态变成完成。完成后会自动检查清单。
不要在任务运行时关闭服务，也不用重复上传 ZIP 或重复点击预处理。上传 ZIP 时，
预处理产物都保存在本次解压目录内，实际路径已经自动填好：

```text
event_cache/
├─ audio_features/
├─ chart_ir/
└─ event manifests / QC reports
event_manifest.json
```

如果更改采样率，建议使用新的 event cache 和 manifest 路径，或者勾选**强制重建缓存**，避免把旧缓存和新参数混在一起。旧 fixed-frame 的 cache 不能当作 event cache。

### 3. 检查 manifest

点击 **检查训练集状态**。页面会显示：

- 歌曲数和难度数；
- event 会显示 `event QC 可用`，但最终训练仍由 `train_events` 重新读取 source 并 QC；旧 fixed-frame 的 `usable_for_training=false` 不会单独阻断 event；
- 音频状态，例如 `ok`、`missing`、`failed` 或 `dependency_missing`；
- 主要过滤原因，例如解析错误、音频不可读或对齐失败；
- event 预处理时记录的 `target_schema=event_ctx_v1`、`feature_profile=event_audio_v2_rhythm` 和 sample rate。

如果显示“尚未生成训练清单”，说明还没有完成第 2 步，点击 **2. 预处理训练集**。
如果显示“正在预处理”，继续等待右侧完成即可。如果显示失败或中断，查看日志，
修复后重新执行预处理。`missing_level` 表示对应的 `&lv_N` 缺失或为空，应填写
实际难度；不要凭空填一个等级只为通过检测。

`duplicate-slide-head` 表示同一时刻出现两个相同键位的独立 Slide 头，例如
`4-6[8:1]/4-7[8:1]`；如果本意是同头分支，应在源谱面中改成
`4-6[8:1]*-7[8:1]` 后强制重建缓存。`^` 最短弧语法本身是可识别的。

event 训练不需要 `include_unusable`、frame labels 或 alignment。若 source 路径缺失或 event 自己的 parser/QC 失败，应修复原始数据；不要用旧 fixed-frame 标志伪造 event 可用性。

## 七、开始训练

训练区现在显式提供三种方式：**从头训练（fresh）**、**从断点继续（resume）**和
**已有模型加强训练（finetune）**。event 首次实验请选择 fresh。resume 恢复同一
event schema 的完整训练状态；finetune 只加载同一 event schema 的权重。旧 fixed-frame
checkpoint 不能用于 event 的 resume 或 finetune，必须重新训练。后二者的 epochs 都
表示本次追加轮数。选择后请上传 checkpoint 或粘贴路径，再点击**检查 checkpoint**。

finetune 的默认学习率是 `2e-5`，建议先追加 1--5 个 epoch，并使用新的输出目录。
完整步骤和兼容性说明见[继续训练与 checkpoint 使用说明](CONTINUE_TRAINING.md)。

### 1. 第一次只做冒烟测试

在训练区填写或保留以下值：

```text
epochs                 1
batch size             1
segment beats          16
gradient accumulation  1 或 4
d_model                32（只做极小流程验证）
Transformer 层数       1
设备                   自动
```

点击 **开始训练**。成功后，右侧下载结果中应出现 `event_last.pt`，可能还有
`event_best.pt` 和 `event_epoch_NNN.pt`。这一步只
证明 manifest、缓存、模型和保存逻辑互相接通，不代表模型已经学会制谱。

### 2. 正式起步参数

确认冒烟测试成功后，可以先使用：

```text
epochs                 20
batch size             1 或 2
segment beats          16
gradient accumulation  4 或 8
d_model                256
Transformer 层数       4
设备                   自动
```

12 GB 显存不足时，优先减小 batch size、segment beats 或 d_model；显存足够且
数据量较大时再增加 epochs。训练过程中右侧会显示最后几行日志；发生错误时，
请同时记录 manifest 路径、设备、参数和错误末尾内容。

### 3. 命令行等价操作

不使用 WebUI 时，可以在项目目录执行：

```powershell
py -m maichart.training.prepare_events examples/training_zip_demo/raw_dataset `
  --cache-dir examples/training_zip_demo/event_cache `
  -o examples/training_zip_demo/event_manifest.json `
  --sample-rate 22050 --limit 2

py -m maichart.training.train_events `
  --manifest examples/training_zip_demo/event_manifest.json `
  --checkpoint-dir checkpoints/event_v1 `
  --epochs 1 --batch-size 1 --segment-beats 16 `
  --d-model 32 --nhead 4 --num-layers 1 --device auto
```

旧 fixed-frame 基线仍可按 [docs/generation_v02.md](generation_v02.md) 的旧命令运行，但必须显式选择 `legacy_frame_v2`。旧模型需要重新训练才能进入 event 主路径。

## 八、用训练好的模型生成谱面

当前 WebUI 默认是 `event_ctx_v1`，event 音频 profile 是 `event_audio_v2_rhythm`，只接受
`maichart-event-generator-checkpoint-v1`。生成区的 **条件强度 guidance scale** 是软约束
倍率：保持 1 为默认；1--2 更强调目标；2--3 仅用于实验。它不能保证目标等级或
Radar/Style 严格达到要求。旧 checkpoint 不要上传到 event 训练或生成区；WebUI
会按 schema 选择旧 `generation.cli`，但旧结果必须明确标成基线。

### 1. 导入输入

在 **生成谱面** 区：

1. 上传音频，或粘贴音频的完整路径。event 模型必须使用音频，不读取旧 `audio_features.json` 输入。
2. 上传训练产生的 `event_best.pt` 或 `event_last.pt`，或粘贴 checkpoint 完整路径。
3. 选择目标等级，例如 `13.5`。
4. 调整 Radar 和 Style。所有滑条和数字输入值都在 `0` 到 `1` 之间；数值越大
   表示更强的对应倾向，不等于官方机台的原始雷达公式。
5. 首次生成保持 `first=0`、`tempo-mode=auto`、temperature `0.8`、候选数量 `1`，不要勾选随机采样。

### 2. BPM 和变速

- BPM 留空：程序自动分析音频的局部 tempo，并把检测到的变化换算成谱面 beat
  位置，必要时在输出中写入 `(BPM)`。
- 填入 BPM：强制把整首歌按单一 BPM 解释。只有你确定自动检测不适合时才这样做。
- `first`：音频中 chart beat 0 的秒数；它不是 note 的绝对秒时间。
- event 没有 `division`、presence threshold 或 button threshold 控件；不要把旧 fixed-frame 参数填进 event 请求。
- Tempo map JSON 是旧 fixed-frame 基线的手动覆盖项。event 使用 `tempo-mode auto` 或 `constant`，constant 时必须填写 BPM。例如：

```json
[
  {"beat": "0", "bpm": 180},
  {"beat": "32", "bpm": 150}
]
```

这里的 `beat` 是谱面 beat，不是秒数。WebUI 已经提供了较高的多行输入框，
不要把复杂 JSON 压成难以检查的一行。手动 tempo map 会覆盖音频自动检测。

### 3. 候选数量

默认候选数量是 1。把它改成 2 或更大，才会手动开启多个随机候选；当前版本
不会自动进行可玩性检查、目标可行性修复或候选排序，需要你逐个下载和人工比较。

点击 **生成谱面** 后，在右侧任务状态等待完成。下载链接会显示实际文件名，通常
包括 `maidata.txt` 和对应的 ChartIR JSON。

## 九、验证和人工检查

先在 PowerShell 中进入项目目录，然后执行：

```powershell
maichart validate webui_workspace/generated/maidata.txt --difficulty 5
maichart parse webui_workspace/generated/maidata.txt --difficulty 5 --ir
```

验证结果没有 errors 只代表格式和解析基本正确，不代表谱面好玩。还要检查：

1. 音乐和第一拍是否对齐；
2. BPM 变化处是否仍然贴合音乐；
3. 重复段落是否有合理的重复或变奏；
4. 钢琴、鼓点和细碎节奏是否被过度压成 Tap，或被不必要地变成 Slide；
5. Hold、Slide 的长度和方向是否合理；
6. 同时按键、Jack、Touch 是否超出目标难度的手部负担；
7. 在兼容的 Simai/Majdata/MiaCode 工具中打开并实际试玩。

## 十、不同预处理参数能否一起训练

可以保留多份实验产物，但不要直接把不同参数生成的缓存放进同一 manifest。
建议如下：

| 参数 | 是否建议混合 | 原因 |
| --- | --- | --- |
| `division` | 仅旧基线 | event 不使用 fixed-frame division。 |
| `sample_rate` | 不建议 | 音频特征的时间/频率表示可能不同。 |
| cache 目录 | 可以不同 | 目录只是保存位置，但 manifest 必须指向正确缓存。 |
| 训练 epochs、batch、模型宽度 | 可以不同 | 它们是训练配置，不是数据格式。 |

若确实要合并，先保证数据的 division 和 sample rate 兼容，并确认没有重复
`song_id`，再执行：

```powershell
py tools/merge_training_manifests.py `
  manifests/a.json manifests/b.json `
  -o manifests/merged.json
```

推荐按歌曲划分训练、验证和测试集，不要把同一首歌的不同难度同时放到不同
集合，否则结果会被歌曲泄漏高估。项目的 split 工具和 `docs/generation_v02.md`
提供了进一步说明。

## 十一、常见错误

### `No usable training difficulties found`（旧 fixed-frame 基线）

先点击 **检查训练集状态**。对于旧 fixed-frame 基线，通常是没有成功执行预处理、manifest 路径填错，或
所有难度都被 QC 过滤。查看主要过滤原因，修好原始 `maidata.txt` 或音频后重新
预处理。不要一看到错误就勾选 `include_unusable`。

### `dependency_missing`

重新执行：

```powershell
py -m pip install -e ".[audio,training]"
```

然后重新预处理。旧的失败 manifest 不会自动变好。

### `audio_unreadable`、`missing`、`failed`

确认音频和 `maidata.txt` 在同一个歌曲目录，文件扩展名受支持，文件没有损坏，
并且路径没有因为 ZIP 多套一层而指错。修复后重新预处理。

### `unknown_tokens` 或 `parse_coverage_below_1`

先用 `maichart validate` 查看具体位置。当前程序支持常见的紧凑同时按键、拍号
标记、修饰符顺序变体和部分复杂 Slide。解析覆盖率达到 80% 的谱面可能仍会被
保留为 `partial_parse`，但应查看 manifest 中的未知 token 数量。确认每个训练
难度都有成对的 `&lv_N=`、`&inote_N=`；`E` 不是硬性必需字段。

### `Missing audio_features`

说明缓存不完整。不要只复制 manifest；要让 manifest 指向包含
`audio_features`、`frame_labels`、`chart_ir` 和 alignment 的完整 cache，或重新
执行预处理。训练不能仅凭 maidata 文本完成。

### `CUDA requested but unavailable`

把 WebUI 的设备改为 **自动** 或 **CPU**，并再次确认 `torch.cuda.is_available()`。
如果显卡驱动、CUDA 版 PyTorch 或环境安装不完整，先修复依赖再强制使用 CUDA。

### checkpoint 路径错误

确认训练任务已完成，并使用右侧显示的下载文件或实际生成的
`checkpoints/event_v1/event_last.pt`。上传 checkpoint 后不要只选择文件，
还要确认模型路径框已经自动填入。

### ZIP 上传后预处理找不到歌曲

检查 WebUI 自动填入的原始训练集目录。它的下一层应该能看到歌曲目录，而不是
另一个 ZIP 文件或只含一个总目录的错误路径。必要时展开 ZIP，直接把展开后的
目录路径粘贴进去。

如果 ZIP 来自 Majdata/MiaCode 等编辑器，里面可能包含 `.miacode`、`.autosave`
或其他备份目录。程序会跳过隐藏和自动保存目录，并优先使用包含音频的真实歌曲
目录；仍建议清理明显的备份文件，减少重复样本。

## 十二、实验记录建议

每次 event 生成至少记录：歌曲名、checkpoint schema、训练 manifest、sample rate、
目标等级、Radar/Style JSON、BPM 或 tempo map、随机种子、验证输出和人工试玩意见。
这样才能判断是模型能力、音频对齐、节拍量化还是条件控制导致了结果变化。
