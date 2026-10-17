# chart_stream_v1 生成修复与交接（2026-10-17）

本轮处理生成界面设备选择、设备解析、checkpoint 加载和连续解码中的确定缺陷。分支为 `codex/chart-stream-v1`，开始时 HEAD 为 `cd2a502`。未启动训练，未停止现有训练或 7860 / 7861 服务。模型参数名称、尺寸和训练损失保持原合同；本轮不代表谱面音乐质量验收。

## 用户可见的修改

- “生成谱面 → 导入歌曲与模型”直接显示“生成设备 device”，可选自动、CUDA GPU、CPU，所有模型族共用。
- 生成 checkpoint 按自身 schema 检查，独立于训练页选中的模型族。原先训练页默认 song_form_v2 时会把 stream checkpoint 送进 form 检查接口，导致错误识别。
- stream 生成传递随机种子、零温度和手动 tempo map；留空 BPM 时使用音频分析结果。
- 每项生成任务分别保存条件、tempo、meter 文件和输出，避免并发任务覆盖条件或谱面。日志中的 effective_settings 包括实际传给子进程的文件。
- 能力声明不再宣称 stream 已接入旧模型的自动无理检测／修复，也不再宣称已有独立音频位置对齐监督。当前训练实际使用 token CE、事件意图辅助目标和曲线目标。软物理引导仍保留，不能等同于自动修复或可玩性保证。
- 页面说明明确 EOS 由模型决定；不承诺强制写满音频。

## 确认并修复的生成缺陷

| 问题 | 修复及边界 |
| --- | --- |
| `auto` 作为 torch storage 设备传给加载／张量操作 | `stream/runtime.py` 统一解析，输出明确的 CPU 或 CUDA 编号；不可用的 CUDA 给出错误，不静默换设备。 |
| 整份 checkpoint 的 Adam 状态占用推理显存 | CPU 读取整个 payload，加载权重后仅把模型搬到选定设备；生成和曲线推理释放其余 payload。 |
| 加载时宽泛忽略未知参数 | 仅兼容早期未使用的 memory_projection、plan_projection、event_gate 参数；其他缺失／多余权重严格报错，并检查输入合同。 |
| 配置温度为 0 除零 | 所有语法状态支持贪心选择，零温度不会进入 softmax 除法。 |
| 事件上限计入尚未写完的音符 | 仅提交 `<end>` 已闭合的完整事件，在事件边界检查上限；预算结束明确标记 incomplete。 |
| 首拍 0 被严格下界排除 | 首个合法拍位允许 0，并计算正／负 first 对音频范围的影响。 |
| 上下文上限切断正常配置或越过位置嵌入范围 | 在已提交事件边界轮换历史；若当前配置跨越资源边界，保存已采样前缀并在新上下文重放。只有单个配置本身无法容纳才报错，不截掉音符、不补休止、不发分段 EOS。 |
| 历史分支缺少训练时的位置嵌入 | 全量与缓存解码共用 prepare_decode_memory，历史 token、位置嵌入和 padding mask 相同。 |
| 缓存交叉注意力与后续 memory/mask 不一致 | 同一缓存生命周期使用固定历史与 memory；重建上下文或改写分数后重置并重放缓存。 |
| `2/4` 输出变成 `1/2`，上下文仍保留旧写法 | Fraction 规范化后同步重写并重放 decoder 前缀，不移动实际拍位。 |
| 物理引导未注册实际时间 | 事件取得真实拍位后，注册 BPM / first 对应的秒数，再计算引导；只提交完整配置。 |
| memory retrieval 限制与 WebUI 范围矛盾 | 检索槽数接受配置范围，仍受模型槽数和有界档案控制。 |
| 跨上下文重复运行整曲音频 Encoder | 分离不可变音频编码与动态历史／持续动作拼接，音频 Encoder 每次生成仅调用一次。 |
| 生成／曲线分析使用旧音频证据策略 | 统一使用训练的 balanced_evidence_v1 与 native_measured_v2；报告保存实际策略名称。 |
| 到音频尾部已无合法下一事件时继续数字搜索 | 预检可达性，区分正常音频范围结束与位置限制耗尽；不伪造下一音符。 |

本轮没有把音符拍位重新绑定到固定时间帧，也没有增加固定常见分音白名单。保留既有可配置分母、间隔及最小间距约束。

## 验证结果

统一修改完成后执行相关回归；实际页面检查发现 checkpoint 路由及运行中后端缓存问题后，针对新增修改补验。

- Python：36 项通过。覆盖真实微型模型前向／反向、全量与缓存解码的一致性、带 padding 的历史、多次上下文轮换、配置跨边界重放、温度 0、首拍 0、完整事件预算、Fraction 规范化、实际物理时钟、checkpoint 兼容和 HTTP 参数传输。
- Node：56 项通过。包含设备常显、所有选项、种子／零温度、跨模型族 checkpoint 识别，以及既有 form 生命周期、前端参数和任务轮询回归。
- 浏览器：Microsoft Edge 实际渲染页面并点击生成；自动／CUDA／CPU 选项可见且正确进入请求。没有前端 pageerror。
- 新后端 CPU 实际生成：使用 4 秒音频与 `.scratch/chart_stream_e2e/checkpoint4/stream_last.pt`，种子 17、两种温度均为 0，事件预算 4；实际提交 2 个事件，由模型 EOS 结束。生成文件经项目自身解析器重解析，事件数与报告一致。此 checkpoint 为微型短训权重，不能用此结果判断音乐质量。
- 早期 stream checkpoint：checkpoint3 实际 CPU 加载与生成成功，包含对早期遗留参数的兼容验证。
- 曲线分析：自动设备在隔离 CPU 环境中解析为 CPU，得到 6 个严格递增控制点，覆盖 0 到音频终点，包含音频 SHA256，并确认两项音频策略与训练一致。
- 测试没有使用 GPU。未进行完整训练、全库测试、生产规模推理测速或音乐／可玩性验收。

主要回归命令：

```powershell
python -m pytest -q tests/test_stream_generation_integrity.py tests/test_chart_stream_v1.py tests/test_model_compatibility.py tests/test_form_checkpoint_legacy.py tests/test_generator_checkpoints.py
node --test tests/test_stream_generation_ui.js tests/test_form_ui_lifecycle.js tests/test_song_form_evidence_ui.js tests/test_rational_generation_ui.js tests/test_training_input_payload.js tests/test_task_polling.js
```

CPU 测试设置 `CUDA_VISIBLE_DEVICES=-1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。PyTorch 的 norm_first/nested_tensor 警告仍存在，相关测试均通过；本轮没有为消除此提示改变模型数学结构。

## 现有进程与生效范围

7861 的静态页面直接读取当前文件，刷新后设备控件和 checkpoint 检查路由已生效。当前后台进程仍缓存较早的 stream_api：真实浏览器试跑中，选择 CPU 后能够调用新的生成子进程，但请求的种子 17 被旧 API 忽略成 0，新增事件预算也未转发。旧结果没有本轮新增的 task.result.device / actual_output 等字段。这是运行进程缓存差异，不能把源文件修改当成后台已重新加载的证据。

为保持现有服务不受影响，完整新 API 链路在临时独立端口 64433 验证。临时服务器设置自动退出。没有重启 7860 / 7861；主端口后台新增接线需要重启对应 WebUI 进程后加载。先前自动审批未允许重启 7861，本轮未绕过该限制。

本地验证证据位于 `.scratch/stream_generation_fix_20261017/`：browser_result.json、verification_result.json、generation_device.png，以及隔离任务输出。该目录不提交到 Git。暂未上传 GitHub。

## 后续判断的边界

这些修复解决了解码上下文、设备和接线错误，不能证明模型已经理解音乐。模型仍可能提前 EOS、忽略细节音频、生成不合理密度或配置；当前不存在额外的音频位置对齐损失。事件意图头也不等同于可在生成时独立操控的未来草稿。以后改训练目标或推理规划需单独论证，不能把本次工程回归通过当成这些能力已完成。

无理自动修复尚未接入 stream 生成入口；本轮已修正能力声明，保留软物理引导，不宣称已验收可玩。训练样本切块、质量权重、持续记忆学习以及长曲生成效率没有在本轮完成全量审计，不能据本轮结论声称整个新架构已无隐患。
