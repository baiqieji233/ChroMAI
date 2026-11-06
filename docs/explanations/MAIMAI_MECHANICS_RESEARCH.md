# 判定区核对与检测边界（2026-10-10）

本页纠正此前 Goal/P0/P1 报告中未经验证的机制描述。核对对象是 MiaCode 的开源参考实现，**不是 SEGA 官方判定规范或实机测试结论**。

## 固定来源

参考版本固定为 [MiaCode 6573efb](https://github.com/Team-MiaCode/MiaCode/tree/6573efb611056bccd046dd17bc2f4cceda78bf88)：

- [MuriConfig.h](https://github.com/Team-MiaCode/MiaCode/blob/6573efb611056bccd046dd17bc2f4cceda78bf88/src/common/MuriConfig.h)：区域编号、几何与时基。
- [MuriAnalyzer.cpp](https://github.com/Team-MiaCode/MiaCode/blob/6573efb611056bccd046dd17bc2f4cceda78bf88/src/tools/muri/MuriAnalyzer.cpp)：Touch 相邻关系。
- [MuriRuntimeModelBuilder.cpp](https://github.com/Team-MiaCode/MiaCode/blob/6573efb611056bccd046dd17bc2f4cceda78bf88/src/tools/muri/MuriRuntimeModelBuilder.cpp) 与 [MuriSimpleNoteJudge.cpp](https://github.com/Team-MiaCode/MiaCode/blob/6573efb611056bccd046dd17bc2f4cceda78bf88/src/tools/muri/MuriSimpleNoteJudge.cpp)：相邻 Touch 分组和 0.51×组大小的判定门槛。
- [MURI_DETECTION_SPEC.md](https://github.com/Team-MiaCode/MiaCode/blob/6573efb611056bccd046dd17bc2f4cceda78bf88/docs/specs/muri/MURI_DETECTION_SPEC.md)：静态参考与运行时结果的区别，保护音符与告警等级。

已移植区域常量、相邻关系和分组门槛，MIT 版权声明见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。未引入素材或完整桌面程序。

## 关键纠正

1. 玩家视角下，1 位于正上顺时针 22.5°，1/8 是水平上边。1–4 在右侧、5–8 在左侧；这只是初始手位倾向，不能锁死手部。
2. C 是中心，A/B/D/E 各有 1–8 编号，并不是 A–E 五个同心圆。`A4` 不能截断成 `A`，也不能统一与按钮 1 比较。
3. 同刻相邻 Touch 组成连通判定组；参考实现中组内已判定数量达到 `ceil(0.51×组大小)` 后，会处理其余成员。不能把每个 Touch 计成一只手，不能将整圈 Touch 填成一整块持续占用区域。
4. Slide 的星头起音、等待、起滑、经过判定区、最后区域完成是不同事件。普通时值默认等待一拍；显式 `[等待秒##持续秒]` 单独处理。持续时间经过变 BPM 时按 tempo map 积分。
5. `2-6` 是穿过中心的直线，绝非“半转时顺逆时针不明”。弧线 `<`/`>` 的方向还受起点方位影响，不能一概规定 `<` 逆时针。
6. 星星图像到达某个按钮位置，不自动证明它会抢掉 Tap 的 Perfect。需要判定区进入时机、先后顺序、可判定窗和玩家输入；旧报告中“3 条硬冲突全部人工确认”为无依据结论，撤回。
7. 显式 `1/1-5[...]` 存在两个头部输入；`1-5[...]*-4[...]` 才是一个头部的并行分支。二者不能共用“同刻 Tap 一概放行”规则。
8. 交互连段或快速左右手交替本身是正常配置；没有可靠手部动作分配之前，不把任意 100ms/135° 阈值当作人体极限。

## 当前检测承诺

检测范围是整首合并后的 EventPoints，绝对秒保留 first 和变 BPM，跨窗口持续动作会参与；此前“仅检测段内”描述不符合原调用点，真正补充的是窗口重试与修复后的再次检测。

- `hard`：静态同 pad 的明确同时头重叠或持续输入内部重叠；保护音符、Touch 组可联动与边界时间会降为提示。这里仍是指定静态输入假设下的冲突，不是实机全 Perfect 的数学证明。
- Hold 起点前后 `6/180 s` 的同 pad 交叠只作为 soft 人工检查，依据本地 MiaCode 静态 overlap 容差；严格的 Hold 内部重叠仍按原规则检查。soft 边缘项受 `detect_soft_conflicts` 开关控制，不参与 reject 或 local repair。
- `soft`：Slide 参考区可能提前触发后续输入、Touch 覆盖路径、需要人工检查的多按键配置。不会自动删除或触发严格拒绝。
- `partial`：无法完整解析，或复杂 Slide/Wifi/连续路径不在当前几何子集内。报告明确列出原始配置和原因，不伪装成“通过”。

`success` 表示当前导出策略允许继续，不表示实机可玩性通过。合法的复杂 Slide 仅因轨迹不在静态覆盖范围内，不会变成 hard 或被 `reject` 自动拒绝；覆盖状态仍为 `partial`。`reject` 针对已检测出的 hard 冲突和致命解析错误。`local` 可以修复来源明确的独立 hard 冲突，同时保留未支持路径及 `partial` 标记，不声称整谱问题全部消除。

直线判定区交点使用线段与圆的交点；弧线参考足迹使用 720 个轨迹比例采样点。两者只是传感器参考足迹。复杂路径、有序判定区提前清空和完整双手动作搜索仍需后续移植/外部工具核验。

局部修复只删除有精确来源索引、确认被判为 hard 的普通 Tap/Touch 子音符，保留混合和弦的其它成员、长条和 Slide 分支。修复前后分别输出文件，并再次检测。默认只报告，不改谱。
