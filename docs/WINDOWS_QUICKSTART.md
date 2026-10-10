# Windows 一键部署与启动

双击根目录 **START_WEBUI.bat**。需要先安装 64 位 Python 3.11；也支持 3.10–3.13。脚本会优先复用项目 `.venv`，然后尝试本机的 `D:\maimai_ai\venv`；两者都不存在时才创建新的环境。首次部署需要联网安装音频、训练与测试依赖，之后直接复用已安装环境，不会每次重新下载。

启动脚本根据显卡选择 CPU 或 CUDA 依赖。当前 CUDA 配置复现原项目验证过的 PyTorch 2.7.0 + CUDA 12.8；若 NVIDIA 驱动不支持，脚本会提示错误，需升级驱动或明确选择 CPU。普通 Python WebUI 不需要 Node.js；运行全部测试和可选浏览器验证时需要 Node.js。

启动窗口显示实际访问地址并打开浏览器。默认从 7860 开始找空闲端口，避免与另一份项目服务冲突。服务始终从本目录源码加载，关闭窗口或 Ctrl+C 即停止服务。启动不会自动训练或生成。

常用命令（在项目目录的 CMD 中执行）：

```bat
START_WEBUI.bat --cpu
START_WEBUI.bat --cuda
START_WEBUI.bat --port 7870
START_WEBUI.bat --check
START_WEBUI.bat --no-browser
START_WEBUI.bat --reinstall
```

`--check` 只检查现有环境，不安装、不启动服务；未创建虚拟环境时检查已安装 Python 的依赖。`--no-install` 用当前可用环境启动而不安装，适合已有完整依赖的电脑；正常双击仍使用独立 `.venv`。自动调用时可设置 `MAICHART_NO_PAUSE=1` 取消 BAT 结束时暂停。

可用 `MAICHART_PYTHON` 指定 Python 可执行文件，或设置 `MAICHART_VENV` 指定虚拟环境目录；也可在项目根目录建立被 Git 忽略的 `.webui-python` 文件，写入 Python 可执行文件路径。`MAICHART_PYTHON` 和 `.webui-python` 优先级高于自动选择。还可通过 `MAICHART_TORCH_INDEX_URL` 设置兼容的 PyTorch 下载源。脚本锁定原项目验证过的 torch 版本；源中须包含该版本。

```bat
set "MAICHART_PYTHON=C:\Path\To\Python311\python.exe"
START_WEBUI.bat --check
```

复制时未包含训练集、checkpoint、缓存或运行中间文件。训练前在 WebUI 导入数据，生成前导入自己保存的兼容 checkpoint 和歌曲；小型示例 ZIP 仅用于验证流程。原项目仍在 E 盘，需要在同机使用其数据时，可在 WebUI 填写原数据目录。

运行测试：

```bat
.venv\Scripts\python.exe -m pytest -q
```

历史记录与保留/排除范围见 [精简副本说明](PORTABLE_COPY.md)。
