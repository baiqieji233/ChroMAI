# Windows 本机运行环境恢复记录

2026-10-26，系统 Python 3.11 已卸载，系统默认版本为 3.15。本机另建项目专用 Python 3.12.14 环境，安装 PyTorch 2.7.0+cu128、NumPy 2.1.2、SciPy 1.17.1 和 librosa 0.11.0。系统 3.15 保持原样。

当时根目录环境中的 Python 写文件失败，更换启动器类型后仍可复现。相同 Python 与依赖在专用目录中写入正常，底层权限原因尚未确定。本机通过忽略文件 `.webui-python` 指定 `E:\simai_transformer\.scratch\python312_runtime\Scripts\python.exe`。接手时先读该配置，确认运行环境所在目录再清理临时文件。

当时使用 `.\START_WEBUI.bat --port 7860`。CUDA 检查、主页面、真实文件上传、检查点检查入口已验证，Composer 回归 544 项通过、1 项跳过。原环境及尝试目录保存在 `.scratch`，原检查点得到保留。

精简副本未包含训练集、检查点、缓存或运行中间文件。副本可在主工作台导入素材或填写本机路径。[Windows 启动说明](../compatibility/WINDOWS_QUICKSTART.md) · [精简副本说明](../compatibility/PORTABLE_COPY.md)
