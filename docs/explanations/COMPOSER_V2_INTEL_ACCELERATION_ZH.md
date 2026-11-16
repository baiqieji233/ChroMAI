# ChroMAI · Intel 核显与 NPU 加速

**ChroMAI（Composer v2）** 支持将音乐特征提取、写谱解码和训练分别配置在不同设备上执行。已有检查点可直接使用，不需要重新训练或转换检查点文件。

## 安装

在运行 WebUI 的同一个 Python 环境中安装可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[intel-acceleration]"
```

OpenVINO 需要正常工作的 Intel 显卡或 NPU 驱动。缺少 OpenVINO 时，自动模式保留原来的 PyTorch 路径。

## 推理

生成页的“音乐特征设备”和“写谱加速引擎”默认自动选择。命令行示例：

```powershell
.\.venv\Scripts\python.exe -m maichart.composer_v2.generate `
  --checkpoint checkpoints/<模型名>/composer_v2_best.pt `
  --audio path/to/track.mp3 --output outputs/maidata.txt `
  --device cpu --audio-device auto --inference-backend auto
```

- 音乐特征自动优先使用当前 CUDA 设备，其次 OpenVINO Intel GPU、NPU，最后 CPU。
- MERT 只导出卷积前端、位置卷积和前八个 Transformer 块；输出取在第八块之后、最终 LayerNorm 之前，与现有缓存语义一致。
- 核显和 NPU 处理常规完整片段，首尾变长片段使用原始 CPU 计算。这样避免为每首歌不同的尾段长度额外编译 GPU 内核，也不会用补零改变时间归一化。
- 写谱自动比较 PyTorch、OpenVINO CPU 和可用的 Intel GPU。只采用数值检查通过且至少快 10% 的后端。自动 CPU 线程数也会测速；手动指定 `--cpu-threads` 时尊重指定值。
- 写谱 GPU 使用 FP16，CPU 使用 FP32。词元采样、语法约束、物理检查、候选比较和回退仍使用现有逻辑。浮点差异可能使相同种子的采样结果不同。
- 上下文预填充、多词元查询和不适用的查询保持 PyTorch；单词元加速返回新增 K/V，继续使用可回退的历史缓存。GPU 历史缓冲区按长度分档并屏蔽空位。
- 模型图、编译结果和自动选择保存在 `--cache-dir` 下，按权重、导出版本、OpenVINO/PyTorch 版本及硬件配置区分。首次导出和编译会增加等待时间。
- 自动后端失败会报告原因并回退 PyTorch。显式指定的写谱后端失败会报错。特征加速失败会报告并回退 CPU，原始 MERT 提取失败仍然报错。

强制 NPU 特征提取：`--audio-device openvino-npu`。完整 CPU 对照：`--device cpu --audio-device cpu --inference-backend torch`。手动选择 Intel GPU 解码：`--device cpu --inference-backend openvino-gpu`。

同一首歌已有音频特征缓存时，不会再次运行 MERT；这时总体收益主要取决于写谱解码。短音频的首次调用可能因加载、导出和编译而更慢。阶段吞吐提升不代表整曲生成有相同倍数的提升。

在 Core Ultra 5 225H / Arc 130T 的实测参考中：约 147 秒音频的 MERT 特征提取，CPU 耗时 40.5 秒，核显与 CPU 边界片段组合耗时 8.1 秒（约 5 倍，复用已有编译缓存，含模型加载）；特征相对 L2 差异约 0.116%。写谱解码自动测速选择了 PyTorch CPU；原生 XPU 写谱推理较慢，未加入自动生成设备。

## 训练

`--prepare-device auto` 复用上述 MERT 加速与缓存；也可指定 `openvino-gpu` 或 `openvino-npu`。特征提取属于冻结编码器的离线计算，不会改变主模型的反向传播设备。

主模型支持 `--device xpu`，通过原生 PyTorch 使用 Intel 核显训练。需要另外安装 XPU 版 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/xpu
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

验证可用后，训练页可选择“Intel 核显（XPU）”。自动设备顺序为 CUDA、XPU、CPU；XPU 使用 BF16 或 FP32，FP16 训练仍只开放给 CUDA。音频特征插值在 CPU 按原有双精度规则完成后传输，兼容不同客户端驱动的 FP64 能力。布尔条件和分组查询改用等价的 `index_select`，规避部分 XPU 高级索引内核错误。

严格续训的自动设备继承原检查点设备，不会自动切换 CPU/XPU；跨设备加强训练应使用新输出目录进行 `finetune`。XPU 检查点保存并恢复 XPU 随机状态。

在启用起音与强度曲线、128 token、batch 1 的小批量实测参考中，预热后 CPU FP32 约 0.74 秒/步，XPU BF16 约 0.41 秒/步（不同硬件与规模下加速倍数会有所差异）。

## 验证

```powershell
.\.venv\Scripts\python.exe tools/check_intel_acceleration.py
```

检查包括自动设备选择、融合计算与原始解码的一致性、历史回退、GPU 空位屏蔽、OpenVINO CPU 实际执行，以及 NPU 边界片段不补零的约束。生成报告的 `acceleration` 字段和 `composer_v2_acceleration` 日志记录实际选择、测速和数值误差。

参考：[OpenVINO NPU](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)、[PyTorch Intel GPU](https://docs.pytorch.org/docs/2.15/notes/get_start_xpu.html)。
