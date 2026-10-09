# 精简副本说明（2026-10-09）

来源：`E:\simai_transformer`。目标：`D:\simai_transformer`。

保留完整 `src`（含 WebUI 与校准 JSON）、`docs`、`tests` 和测试素材、`tools`、`configs`、项目配置与第三方说明，以及两首合成音频的小型演示数据及 ZIP。

正式训练集、checkpoint、备份、缓存、历史 manifest、上传素材、日志、条件文件、生成结果、曲线中间文件、虚拟环境、Python 字节码与 node_modules 均未复制。首次启动和实际使用时会重新创建运行目录；需要训练数据或模型时，在 WebUI 中导入或指定路径。

当前可重复运行的浏览器检查从临时目录整理到 `tools/browser_qa`：连续等级与偏好、曲线密度与休止、框选休止显示、时间缩放与多点编辑。安装方法见该目录 README。历史一次性探针未保留。

为使精简副本独立工作，图表对比工具改用本项目相对路径；两条旧固定帧 smoke 测试改为从保留的演示数据在系统临时目录生成 manifest 和缓存，不再依赖旧项目的产物。测试生成的模型也在系统临时目录，不随副本交付。

验证：在 D 盘源码下执行全量测试，721 passed、12 skipped；四组 Edge 浏览器验证通过。已确认导入模块与 WebUI `PROJECT_ROOT` 均指向 D 盘副本，而非 E 盘已安装的可编辑包。

双击根目录 `START_WEBUI.bat` 部署并启动，详细说明见 [Windows 一键启动](WINDOWS_QUICKSTART.md)。历史文档中的 E 盘路径、旧日志和旧实验结果保留作记录，指向的产物不包含在精简副本中。
