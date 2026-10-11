# 内嵌 MiaCode 无理检测

生成侧现在直接链接 MiaCode 项目的判定区和 Muri 分析核心。生成时不启动 MiaCode 编辑器、不创建外部子进程，也不要求用户填写可执行文件路径。

## 包含的检测

内嵌核心沿用固定提交 `6573efb611056bccd046dd17bc2f4cceda78bf88`，包含：

- Hold / Tap / Touch 的有序判定区与叠键检查；
- 星星头的辅助判定；
- Slide 和 Wifi 的轨迹、关键区域与提前完成检查；
- Touch 联动与多押形成的动作检查；
- MiaCode 的 `Muri` 与 `Warning` 分级及原始行列号证据。

它仍然是谱面判定参考，不能替代玩家实际游玩认证。

## 构建与部署

第一次部署运行：

```powershell
python tools/build_muri.py
```

脚本会校验 `native/miacode/UPSTREAM.json` 中的源码哈希，使用本机 Qt 6/CMake 构建 `.native/miacode/runtime/maichart_muri_v2.dll`（ABI2，支持可调静态撞尾范围），并把 Slide 参考数据嵌入 DLL。旧 DLL 保留，当前机器已构建新版；其他部署需运行构建脚本。没有匹配的 DLL 时，WebUI 在提交生成前会明确报错，不会静默跳过检测。

## WebUI 行为

“启用无理检测”后，生成命令自动使用 `--conflict-engine miacode`。可选模式仍是：

- `仅检测`：保留结果，写入质量报告和日志；
- `局部修复`：冲突窗口重新采样；若仍有可定位的 Muri，只删除明确对应的音符，并重新检测，只有冲突数量下降才接受修改；
- `拒绝生成`：存在剩余“参与处理”的硬 Muri 或解析失败时拒绝候选；“仅报告”的物理 Muri 保留在报告中。

生成页还可以按 `MultiTouch`、`Overlap`、`TapOnSlide`、`SlideHeadTap`、`SlideTooFast` 分别选择忽略、仅报告或参与处理。`150/200/250 ms` 撞尾档位只调整静态邻近参考：150 更宽松，250 更谨慎，不修改机台 Perfect 判定。预设“保留写法／均衡”中 MultiTouch 默认仅报告，若希望多押参与修复，需要在高级选项中改为“参与处理”。

当前窗口处理失败时可回溯最多 1、2 或 4 个窗口；“自动”会根据证据最多回溯 2 个。全曲预算限制事务次数。开启前置长音符修复后，系统会用反事实重检确认该 Hold/Slide 真正造成硬冲突，才会缩短简单 Hold，或将符合条件的简单 Slide 简化为起点星头 Tap；复杂 Slide/Wifi、EX 和无法确认因果的动作不会被改写。

一次回溯会重建修改点之后的上下文与密度统计，不直接复用旧前缀。结果必须降低参与处理的 Muri，且重生成区间至少保留原候选 80% 的起音头；不满足则恢复原候选。最多检查 8 个活跃长音符，必要时检查前 4 个中的两两组合，因此允许“两个长 Hold 合起来占手”的有限归因，但不承诺任意规模的全曲最优修复。全曲跨窗预算 0 同时关闭回溯和前置时值修复。修复模式为“仅检测／拒绝”时，前置修改关闭。

“单 note 修复策略”是生成后候选搜索：保持起音时间，尝试相邻 Tap/Touch、精确 Hold 缩短和简单 Slide 终点移动，然后重新运行 MiaCode。只有硬 Muri 减少且没有引入新硬 Muri 才接受；它不会修改模型或声称模型拥有单 note 填空能力。

日志会写出 `playability_engine`、每个窗口的 `playability_check`、`playability_issue`、`playability_retry`、`playability_cross_window`、`playability_cross_window_done`、`playability_note_edit`、`playability_repair`、修复复检摘要 `playability_repair_done` 和最终的 `playability_final`。质量报告中的 `playability.reference`、`coverage.engine`、`remaining_details`、`cross_window` 和 `repairs` 保存同样的证据。

检测只需要开启原来的“启用无理检测”。新 CLI 默认使用内嵌核心，`--conflict-engine static` 仅保留给旧规则对照。`event_v1`、`event_ctx_v1` 和旧版 planner 都可使用，不改变模型权重或训练合同。

## 2026-10-11 实际谱面复查

对 `webui_workspace/generated/.history/maidata-1791679433581635400/maidata.txt`（SHA-256 `9630678dc1d82f3477503ca39d7913d77118e34b17bc631e74d63535ce59aa9d`）执行独立复查，原文件保持不变：

- 检出 30 条合并后的诊断：10 条叠键、3 条撞尾（包含 Warning）、13 条动作多押、4 条 Slide 提前完成。
- 局部删除修复 13 个明确关联的音符，每次修改均通过同一核心复检。
- 剩余 12 条 MultiTouch/Muri 和 1 条 TapOnSlide/Warning，未宣称该谱已完全可玩。MultiTouch 不在没有明确受影响音符的情况下直接删谱；生成时它会触发当前窗口重试。
- 此次检测加修复约 4.48 秒，使用 CPU，无外部进程或编辑器。
- 完整日志、报告和单独的修复副本保存在 `.scratch/miacode_native_review/`。

复查现有谱面可使用：

```powershell
python tools/check_playability.py <maidata.txt> --mode local --output-dir <单独的输出目录>
```

原始 MiaCode 行列号对应内部检测文本（每个逗号配置单独一行），不是导出的整小节行号；集成日志另外提供精确拍位、秒数和原始音符文本来定位。紧凑和弦如 `12b` 在检测时等价展开为 `1b/2b`，防止上游宽松解析遗漏此类写法；原谱文本不因检测而改写。

旧版本的 `runtime_reference_mode` 请求仍可被读取，但会迁移到内嵌核心；旧的 MiaCode.exe 路径永远不会被执行。

## 验证记录

本轮完整测试为 `1087 passed, 11 skipped`；另有 8 项 Node 行为测试通过。原生核心测试实际加载本机 DLL，覆盖 Hold 尾部接音、星星头错位、复合 Slide/Wifi、Touch 组、跨窗口重试、删改单个和弦子音符、变速与 first 偏移、旧外部调用参数迁移、日志输出和三个生成模式。测试同时拦截子进程创建，确认内嵌检测没有启动外部程序。

后续分类策略、跨窗与单音符微调增量按用户要求只作短功能验证：相关 Python 测试 `135 passed`，前端参数行为 `5 passed`。覆盖源窗口重生成后新前缀、失败回滚、同窗双长 Hold 归因、默认旧参数、邻键/时值候选、类别与 WebUI 参数。没有再次跑完整训练或大批生成；真实音乐表达与游玩体验由用户和玩家进一步测试。

原有三个测试桩缺失 checkpoint schema/数据合同，已补齐其模拟元数据，保持真实 checkpoint 的校验规则不变。本轮不修改训练权重，不启动训练。
