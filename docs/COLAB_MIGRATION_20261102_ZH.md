# Composer v43 迁移到 Colab Pro / Pro+

本轮准备的是续训副本：原始训练/验证清单、谱面、音频、结构与质量标注保持原字节，完整检查点带优化器、随机状态和游标。本机训练继续使用原目录。云端另存输出，不自动覆盖本机。

## 文件和运行步骤

- `notebooks/Composer_Colab.ipynb`：Colab 启动笔记本。
- `tools/export_colab.py`：在本机导出私有迁移包，默认输入 `checkpoints/prism_composer_v43/composer_last.pt`，输出 `dist/colab/composer_colab.tar.gz` 和 `.sha256`。
- `tools/colab_resume.py`：读取检查点保存的训练参数、执行路径映射并启动训练。

本机导出命令（PowerShell）：

```powershell
& ./.scratch/python312_runtime/Scripts/python.exe tools/export_colab.py --dry-run
& ./.scratch/python312_runtime/Scripts/python.exe tools/export_colab.py
```

如默认导出已存在，显式提供新的 `--output`，避免覆盖旧迁移包。

将迁移包和 SHA256 文件放到 Drive 的 `Composer_Colab_20261102` 目录，在 Colab 打开笔记本。选择支持原生 BF16 的 GPU（如 A100、L4）和高 RAM，建议至少 32 GiB。运行第一格时完成个人 Drive 授权，然后按格执行。笔记本会核对压缩包 SHA256，解压到 Colab 本地磁盘，安装项目所需音频依赖，沿用 Colab 自带 CUDA PyTorch。

首次先续训 100 次更新，确认数据内容验证通过、吞吐和内存合适、Drive 备份成功后，再运行正式训练格。沿用 16×4、BF16、固定学习率 0.0005、总目标 50 轮及原有学习/生成验证设置。默认独立数据进程为 0，可在有足够实测内存余量时比较 1。每 100 次更新保存并同步备份完整检查点和推理权重到独立 Drive 目录；运行时重建后从该目录恢复。训练用本地锁及本地磁盘，避免将数据读取和训练锁放到 Drive 挂载盘。

## 路径兼容和校验

仅在设置 `MAICHART_PATH_MAP` 时，新增 `material_path` 才将原 Windows 绝对路径前缀映射到 Linux 的材料目录；未配置时保留原行为。原始 JSON 不重写，原始权重和训练状态不改写。训练器原有的内容指纹、数据/词表合同、优化器和批游标校验继续执行。若重新构建的监督内容受跨平台依赖差异影响，原校验可能拒绝续训，应调查差异，不能绕过校验或把微调冒称完整续训。

原始音频、谱面约 8.43 GiB；本轮包加上源码、完整检查点及可复用的音频 NPZ 缓存，未压缩合计约 14.32 GiB。排除路径相关的准备缓存、流式银行、波形映射和 SQLite 数据库，由 Colab 重建。压缩包实际大小取决于打包结果。云端需同时容纳压缩包和解压内容；Drive 另需迁移包、完整/推理检查点和日志空间。

## 验证范围

本机路径映射、原始清单字节不变、标注回退、训练入口和已有数据身份相关测试共 62 项通过，启动笔记本所有 Python 代码格编译通过。尚未在 Colab 完成真实训练，因此不宣称云端续训已成功或已有加速倍数。

本机 GPU 是 RTX 5070 Ti Laptop。模型存在 CPU 数据准备和主内存瓶颈，不能只按显卡型号预测收益。应在同一执行中比较稳定训练窗口的有效目标数/秒，剔除首次准备和轮末验证。不同设备及数值库不保证逐位一致。

Colab 官方说明 GPU 类型、可用内存、使用配额和运行时限制会动态变化，付费也不保证某种硬件：[Colab FAQ](https://research.google.com/colaboratory/faq.html)。突然断线后，尚未同步的更新需重跑。

