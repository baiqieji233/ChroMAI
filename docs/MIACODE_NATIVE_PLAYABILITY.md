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

脚本会校验 `native/miacode/UPSTREAM.json` 中的源码哈希，使用本机 Qt 6/CMake 构建 `.native/miacode/runtime/maichart_muri.dll`，并把 Slide 参考数据嵌入 DLL。没有 DLL 时，WebUI 在提交生成前会明确报错，不会退回到一个看似成功但没有 MiaCode 检查的流程。

## WebUI 行为

“启用无理检测”后，生成命令自动使用 `--conflict-engine miacode`。可选模式仍是：

- `仅检测`：保留结果，写入质量报告和日志；
- `局部修复`：冲突窗口重新采样；若仍有可定位的 Muri，只删除明确对应的音符，并重新检测，只有冲突数量下降才接受修改；
- `拒绝生成`：存在剩余硬 Muri 或解析失败时拒绝候选。

日志会写出 `playability_engine`、每个窗口的 `playability_check`、`playability_issue`、`playability_retry`、`playability_repair`、修复复检摘要 `playability_repair_done` 和最终的 `playability_final`。质量报告中的 `playability.reference`、`coverage.engine`、`remaining_details` 和 `repairs` 保存同样的证据。

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

原有三个测试桩缺失 checkpoint schema/数据合同，已补齐其模拟元数据，保持真实 checkpoint 的校验规则不变。本轮不修改训练权重，不启动训练。
