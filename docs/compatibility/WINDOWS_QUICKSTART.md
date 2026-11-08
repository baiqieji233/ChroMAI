# Windows 一键部署与启动

双击根目录 **START_WEBUI.bat**。建议使用 64 位 Python 3.12。未指定 Python 时，脚本优先复用项目 `.venv`，再尝试通过 Windows `py` 启动器找到 Python 并创建环境。启动脚本的版本检查允许 `3.10 ≤ Python < 3.16`，还会检查实际依赖能否导入。更新系统 Python 后，需要用可运行的环境重新安装项目和依赖。

使用 Composer v2 默认的 MERT 音频路线时，在启动窗口显示的 Python 环境安装 `.[composer-v2-music]`。安装示例见[项目 README](../../README.md#启动-webui)。

项目此前验证过 PyTorch 2.7.0 + CUDA 12.8。当前脚本按 `pyproject.toml` 安装依赖，并未锁定torch版本；`--cuda`检查现有torch能否访问GPU，不会自动把CPU版替换成CUDA版。需要复现既有CUDA环境时，先在项目虚拟环境中安装对应版本，再运行检查。普通 Python WebUI 不需要 Node.js；运行全部测试和可选浏览器验证时需要 Node.js。

启动窗口显示实际访问地址并打开浏览器。默认端口为7860；端口被占用时脚本会报错，请通过`--port`明确换端口。服务始终从本目录源码加载，关闭窗口或 Ctrl+C 即停止服务。启动不会自动训练或生成。

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

可用 `MAICHART_PYTHON` 指定 Python 可执行文件，或设置 `MAICHART_VENV` 指定虚拟环境目录；也可在项目根目录建立被 Git 忽略的 `.webui-python` 文件，写入 Python 可执行文件路径。`MAICHART_PYTHON` 和 `.webui-python` 优先级高于自动选择。还可通过 `MAICHART_TORCH_INDEX_URL` 添加PyTorch下载源；添加源本身不锁定版本。

```bat
set "MAICHART_PYTHON=C:\Path\To\Python311\python.exe"
START_WEBUI.bat --check
```

## PowerShell启动、端口与命令入口

在项目根目录执行，下例使用7860端口；把数字改成7871即可换端口：

```powershell
.\START_WEBUI.bat --port 7860
```

环境安装完成后，也可以明确指定项目Python或命令入口，无需激活环境：

```powershell
.\.venv\Scripts\python.exe -m maichart.webui.server --host 127.0.0.1 --port 7860
.\.venv\Scripts\maichart-gui.exe --host 127.0.0.1 --port 7860
```

上面两行是等价的启动方式，选择一行执行即可。`-m`表示由指定Python运行模块；`maichart-gui`是安装项目时按`pyproject.toml`生成的命令，最终同样调用`maichart.webui.server:main`。

全局环境也能启动，但项目及依赖必须安装到那个全局Python。项目采用`src`布局，仅切换到根目录不能让另一个Python自动找到包。`No module named 'maichart'`通常表示选错环境或尚未安装项目；找不到`maichart-gui`通常表示命令所在的Scripts目录不在PATH，或原Python已被卸载。

升级 Python 后，先检查 `.venv\pyvenv.cfg` 及 `.\START_WEBUI.bat --check` 的输出。如果虚拟环境依赖已卸载的解释器，需要创建新环境并重新安装依赖。选择 Python 时，也要确认 PyTorch、torchaudio 和音频库有适配的安装包。本机历史恢复记录（本地开发资料，不随源码发布）

运行测试：

```bat
.venv\Scripts\python.exe -m pytest -q
```

历史记录与保留/排除范围见 [精简副本说明](PORTABLE_COPY.md)。
