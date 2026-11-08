# 在另一台设备继续训练 Composer v2

当前开发分支：`codex/composer-v2-20261106`。

## 取得代码和环境

```powershell
git clone --branch codex/composer-v2-20261106 https://github.com/baiqieji233/simai_transformer.git
cd simai_transformer
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[composer-v2-music]"
```

GPU训练需要可用的CUDA版PyTorch。安装后确认 `torch.cuda.is_available()` 为真；具体环境配置见[环境说明](README.md)。用 `.\.venv\Scripts\python.exe -m maichart.webui.server --host 127.0.0.1 --port 7860` 启动主界面。

## 另外搬运哪些文件

启动脚本默认自动检测NVIDIA显卡和驱动。已有CUDA环境通过实际小型矩阵运算后保留；CPU版或尚未安装PyTorch时，自动查询官方索引，选择当前Python／系统有配套wheel、且适合显卡架构和驱动的torch与torchaudio。优先保留已有公开版本，其次优先项目验证过的2.7.1；找不到时才选择符合条件的其他稳定配套版本。不会修改NVIDIA驱动，也不要求自行安装CUDA Toolkit。

直接运行 `START_WEBUI.bat` 可完成检查、必要安装和启动，无需手填下载源。`START_WEBUI.bat --cuda --check`只检查GPU部署；`--cpu`明确使用CPU，`--cuda`要求GPU可用。`--no-install`和`--check`均不会安装。没有NVIDIA设备时，默认允许CPU环境；检测到显卡但无法确认架构或找到匹配包时会明确停止。显存不足、设备被隐藏等错误不会触发自动重装。

选择过程参考[PyTorch官方安装索引](https://pytorch.org/get-started/previous-versions/)和[NVIDIA兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。为避免PTX等功能的驱动限制，自动选择不超过驱动报告的CUDA上限；混合显卡需所有已检测架构都满足条件。安装只用官方CUDA索引的精确配套版本，随后补齐项目依赖并在新进程中重查，失败不会循环尝试大批版本。

部署后先运行 `START_WEBUI.bat --check`，或使用启动解释器运行 `python -m maichart.webui.server --check`。检查包含真实导入、PyTorch／torchaudio版本及构建匹配、音频重采样和MERT所需类，不会下载模型或启动服务。双击脚本正常启动时可补装；直接命令行启动遇到缺失时会退出并显示修复参数。只维护基础界面时可显式使用服务参数 `--dependency-profile basic`，该模式不证明MERT环境可用。

Git仓库包含源码、配置示例、测试和文档。以下本机材料没有上传：

- 要继续的训练目录，尤其 `composer_v2_last.pt`。严格续训需要包含优化器和随机状态的完整检查点，`composer_v2_inference.pt`只适合生成或初始化微调。
- 该任务使用的训练、开发集清单，以及清单引用的音频、谱面和质量标注。保留任务目录中的 `data_split` 和实际使用的来源清单。
- 可选搬运 `cache/chart_composer_v2`，用于避免重新准备音频特征。MERT模型下载缓存与音频特征缓存是不同材料，缺少时仍可能需要下载模型。

正在写入的检查点不要直接复制。选择已经保存完成的点，复制后核对两端SHA256；需要精确保留当前进度时，先正常停止任务再搬运。

## 选择继续方式

**严格续训／resume**：保留原数据内容、模型结构、批大小、梯度累积、学习率计划等合同。主界面选择原模型族 `chart_composer_v2`，检查完整检查点，再选继续训练。用原任务的参数和清单，不要把新设备的默认值当成原任务参数。更换GPU可能带来数值差异，续训成功不代表逐位复现旧设备。

**加强训练／finetune**：需要改数据、批大小或启用独立起音参考时，使用新输出目录。可以从原v6完整点或推理点初始化，但不会继承原优化器进度。

独立声学起音参考在 `auto` 模式下：新训练和连续模型加强训练（`finetune`）默认启用 `phase_audio_v1`，严格续训（`resume`）继承原检查点设置。旧检查点不会在不训练的情况下自动获得该分支，需通过 `finetune` 在新输出目录训练启用。[操作说明](COMPOSER_V2_ONSET_REFERENCE_IMPLEMENTATION_20261106_ZH.md)

## 处理设备间路径差异

优先保留清单原始内容，使用路径前缀映射定位音频和谱面。例如新建本地 `path_map.json`：

```json
{
  "E:/全量官谱": "D:/datasets/全量官谱",
  "E:/后训练（官谱）": "D:/datasets/后训练（官谱）",
  "E:/simai_transformer": "D:/projects/simai_transformer"
}
```

启动训练进程或WebUI之前设置：

```powershell
$env:MAICHART_PATH_MAP = (Resolve-Path .\path_map.json).Path
```

映射支持最长前缀优先，不能代替缺失的数据，也不会放宽内容及训练合同校验。主界面中的检查点、清单、输出和缓存路径仍应填写新设备上的实际位置。若保存的任务参数还指向旧设备路径，应逐项核对；不要删除合同字段来强行续训。

Windows迁往Colab另见[Colab迁移说明](COLAB_MIGRATION_20261102_ZH.md)。该文档涉及历史状态，执行命令与参数仍以当前源码和待搬运检查点为准。
