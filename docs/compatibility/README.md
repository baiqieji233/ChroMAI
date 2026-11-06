# 环境与兼容性

这里集中保存启动环境选项、旧模型的参数和检查点迁移说明。新用户的安装、训练和生成流程见[项目 README](../../README.md)。

## 运行环境

[Windows 启动说明](WINDOWS_QUICKSTART.md)覆盖 `START_WEBUI.bat`、端口、指定 Python、项目虚拟环境和环境检查。通常用项目自己的 Python 3.12 环境。虚拟环境仍需其创建时使用的 Python，卸载那个版本后应重新创建环境并安装依赖。

| 设置 | 用途 |
| --- | --- |
| `--check` | 检查当前环境及项目的 MiaCode 资源，不启动 WebUI。 |
| `--cpu`／`--cuda` | 前者让服务及其任务使用 CPU，后者要求当前环境能访问 CUDA。 |
| `--port 7870` | 改用指定的本地端口。 |
| `--no-browser` | 启动服务后手动打开终端中的地址。 |
| `MAICHART_PYTHON`／`.webui-python` | 指定要用的 Python 可执行文件。环境变量优先。 |
| `MAICHART_VENV` | 指定项目虚拟环境目录，用于自动选择环境的流程。 |
| `MAICHART_TORCH_INDEX_URL` | 启动器安装缺失依赖时附加的 PyTorch 包索引。 |

PyTorch 和 torchaudio 需要在同一 Python 环境中使用相配的构建。CUDA 是否可用取决于这个环境中的 PyTorch 构建和 NVIDIA 驱动。安装适合该环境的包后，可用 `.\START_WEBUI.bat --check --cuda` 验证。

一键启动器会检查基础依赖。默认 MERT 路线还需要在同一 Python 环境安装 `.[composer-v2-music]`，如需 Beat This 节拍检测可另装 `.[music-structure]`。MERT 和其他资源的来源见[第三方声明](../../THIRD_PARTY_NOTICES.md)。

在全局 Python 中安装项目也可以启动。`python -m maichart.webui.server` 使用当前 Python 运行模块，`maichart-gui` 是安装项目时注册的命令。遇到 `No module named 'maichart'` 或命令找不到时，先核对安装和启动用的是同一个 Python。已换系统 Python 时，先看启动窗口打印的实际路径。

## 检查点与旧模型

训练和生成都在主工作台操作。用 **检查生成 checkpoint 类型** 确认模型族及能力，训练时从 **模型族** 选择相应入口。

| 检查点 | 使用方式 |
| --- | --- |
| `chart_composer_v2`，`continuous_events_v1` | 当前连续事件路线。参数见 [Composer v2 使用说明](../explanations/COMPOSER_V2_WEBUI_ZH.md)。 |
| `chart_composer_v2`，`window_v1` | 早期 v2。按已保存的窗口序列生成，使用自己的词表和状态。进入连续路线应新建 fresh 训练。 |
| `chart_composer_v2`，`level_only_v1` | 只支持等级条件。兼容的 v2 权重可在新目录加强训练，学习 `radar_style_v1`。 |
| `chart_composer_v1` | 旧 Composer 的候选动作和整拍集合架构，依检查点显示对应控件。 |
| `chart_stream_v1`、`song_form_v1/v2`、`event_v1`、`event_ctx_v1`、`legacy_frame_v2` | 选择对应模型族入口，沿用各自的检查点及数据格式。 |

文件名里的实验编号不代表模型族。续训需要完整状态并通过训练合同和数据检查。只保存权重的推理文件用于生成，兼容时也可用作新阶段初始化。跨模型族或跨时间编码应新建训练目录，保留原点供生成或比较。

## 旧版操作与参数

以下材料保留各自发布时的界面、参数和实验值。以当前检查点实际提供的能力为准。

- 早期 v2 的[窗口版路线](../explanations/COMPOSER_V2_20261104_ZH.md)与[窗口重试、条件和节拍设置](COMPOSER_V2_GENERATION_AND_CONTROLS_20261104_ZH.md)。
- Composer v1 的[旧生成参数和条件文件](COMPOSER_V1_PARAMETERS_ZH.md)、[候选拍位与动作温度](COMPOSER_GENERATION_TEMPERATURE_CONTROLS_20261023_ZH.md)、[整拍集合参数](COMPOSER_STREAM_PARAMETER_GUIDE_20261031_ZH.md)与[训练参数分组](COMPOSER_TRAINING_UI_GUIDE_20261029_ZH.md)。
- 早期 [event 模型指南](EVENT_GUIDE.md)、[event 训练参数](TRAINING_PARAMETERS.md)和[固定帧生成](generation_v02.md)。
- Song Form [使用指南](SONG_FORM_V1_GUIDE.md)与[续训迁移](SONG_FORM_RESUME_MIGRATION_20261014.md)。
- [早期训练方式](CONTINUE_TRAINING.md)、[可搬移副本](PORTABLE_COPY.md)和 [Colab 迁移记录](COLAB_MIGRATION_20261102_ZH.md)。
