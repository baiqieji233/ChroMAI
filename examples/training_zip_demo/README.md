# ChroMAI 训练目录与 ZIP 导入示例（Composer v2）

本目录与配套压缩包 [`examples/maimai_training_demo.zip`](../maimai_training_demo.zip) 用于演示 **ChroMAI / Composer v2** 的原始谱面目录结构与 WebUI ZIP 上传解压流程。

```text
training_demo/
  song_001/
    maidata.txt
    track.wav
  song_002/
    maidata.txt
    track.wav
```

> **重要提示：**
> 1. 本示例包内的 `song_001`（26 个音符时间点）与 `song_002`（27 个音符时间点）仅含短合成测试音频与简短结构谱面，用于验证目录组织与导入解压路径。由于 `Composer v2` 数据预处理要求每张入训谱面至少包含 **50 个音符**，直接预处理本示例包会提示 `note_count_too_low`（可用谱面数为 0）。实际训练请准备完整谱面。
> 2. 每个歌曲子目录下须包含 `maidata.txt` 与同级音频文件，且音频文件**必须命名为 `track.mp3`、`track.wav` 或 `track.ogg`**（`.flac` 或其他文件名在训练预处理时会被跳过）。
> 3. `Composer v2` 在训练阶段直接依据真实谱面与音频计算条件和表征，无需旧版固定帧（frame label）字段。

## 安装依赖

```bash
python -m pip install -e ".[audio,training,composer-v2-music]"
```

## 使用方式

- **WebUI 导入**：在「训练模型 → Step 01 · 选择训练素材 → 导入或预处理新的素材」上传 `examples/maimai_training_demo.zip`（或填写本机原始目录路径），点击 **「预处理当前目录」** 生成 `manifests/training_manifest.json`。
- **命令行生成清单**：

```bash
python -m maichart.training.prepare_events examples/training_zip_demo/raw_dataset \
  --cache-dir cache/chart_composer_v2 \
  -o manifests/training_manifest.json
```
