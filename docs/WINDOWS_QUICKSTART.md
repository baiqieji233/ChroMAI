# Windows 一键部署与启动

双击根目录 **START_WEBUI.bat**。建议使用64位 Python 3.11或3.12；启动脚本原有自动选择范围为3.10–3.13。脚本会优先复用项目 `.venv`；没有可用环境时通过 Windows `py` 启动器寻找受支持的 Python 并创建新的环境。首次部署需要联网安装音频、训练与测试依赖，之后直接复用已安装环境，不会每次重新下载。更新系统Python不会自动迁移旧环境中的依赖或命令入口。

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

升级Python后，先检查`.venv\pyvenv.cfg`及`.\START_WEBUI.bat --check`的输出。若虚拟环境依赖已卸载的解释器，需要备份旧环境后重新创建并安装依赖。不要仅放宽启动脚本版本限制：还需确认torch、音频库等提供对应Python版本的安装包。例如PyTorch 2.7没有Python 3.15的Windows安装包，无法通过原样复制3.11的依赖恢复。

### 本工作区2026-10-26恢复记录

系统Python 3.11已卸载，系统默认版本为3.15。已另建项目专用Python 3.12.14环境，安装PyTorch 2.7.0+cu128，并保留旧环境的NumPy 2.1.2、SciPy 1.17.1、librosa 0.11.0等主要库版本。系统3.15没有改动。

本机实测根目录环境中的Python写文件失败；更换启动器类型仍可复现，不能认定为标准venv启动器故障。相同Python与依赖在专用目录中写入正常，底层权限原因尚未确定。目前通过本机忽略文件`.webui-python`指定`E:\simai_transformer\.scratch\python312_runtime\Scripts\python.exe`。**该目录现在是正在使用的运行环境，不应按普通临时文件删除。** 后续定位环境时先读取`.webui-python`，不要假定它位于默认`.venv`。

日常仍使用`.\START_WEBUI.bat --port 7860`。已验证脚本的CUDA检查、主页面、真实文件上传、保留检查点的检查入口，以及Composer回归544通过/1跳过；这不代表新一轮长程训练或音乐质量验收。原环境及尝试目录归档在`.scratch`，原检查点没有覆盖。

复制时未包含训练集、checkpoint、缓存或运行中间文件。训练前在 WebUI 导入数据，生成前导入自己保存的兼容 checkpoint 和歌曲；小型示例 ZIP 仅用于验证流程。原项目仍在 E 盘，需要在同机使用其数据时，可在 WebUI 填写原数据目录。

运行测试：

```bat
.venv\Scripts\python.exe -m pytest -q
```

历史记录与保留/排除范围见 [精简副本说明](PORTABLE_COPY.md)。
