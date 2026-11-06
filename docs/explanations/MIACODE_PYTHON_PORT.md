# MiaCode 无理检测 Python 迁移报告

日期：2026-10-12。

## 结果与使用

生成侧已将原来的 `ctypes → MiaCode DLL → Qt` 调用替换为项目内的 Python 实现。直接使用原来的 WebUI 或 CLI 即可，无需重新训练、转换 checkpoint、编译 DLL 或安装 Qt。更新后需要重启常驻 WebUI 服务并刷新网页。

保留 `--conflict-engine miacode`、`miacode_native` 导入路径、`analyze_inote` / `analyze_maidata` / `ensure_available` 接口，以及现有类别策略和修复参数。旧的外部可执行文件参数仍按此前规则迁移，不执行该文件。新报告如实标记 `engine=miacode_python`、`implementation=python`、`dynamic_dependency=false`；现有 `conflict_engine=miacode` 策略值保持不变。

## 实现范围

- `muri/parser.py`：解析原始 Simai、BPM、时值、分支与连续 Slide，保留行列位置及星头、联动标记。
- `muri/static.py`：移植叠键、撞尾和星头的静态参考检查，保持 0–250 ms 范围与 Muri/Warning 分级。
- `muri/runtime.py`：按 180 Hz 时基处理按下/释放、持续 Hold、Touch 联动、动作合并和多押；Touch 不直接按音符数量算多押。
- `muri/slide.py`：处理 Slide/Wifi 有序判定区、跳区、释放和提前完成，并识别最后判定区是否由其他动作触发。
- `miacode_conflicts.py`：继续把检测结果映射回具体音符，原有窗口重试、来源回溯、前置 Hold/Slide 修复及单音符候选复检沿用这些结果。

这不是退回旧 `static` 检测器的近似规则；原来的 `--conflict-engine static` 仍是单独的旧规则选项。Python 版使用固定上游提交 `6573efb611056bccd046dd17bc2f4cceda78bf88` 的数值路径表，涵盖 608 个 Slide 形状条目和 8 个 Wifi 条目。数据压缩后约 4.59 MB，包内保存来源、校验值与许可范围，无运行时下载。

## 验证

保留旧 ABI3 DLL 作为开发时的独立对照，正常生成和快照回归不使用它。

1. 45 组配置 × 4 个撞尾阈值（0/100/200/250 ms）× 2 个时间偏移，共 **360 组对照全部一致**。比较解析是否成功、五类诊断及等级、音符定位、发生时间、静态原因位置/时间差和统计摘要；浮点时刻允许 2 微秒差异。
2. 三份完整历史谱面（750、807、666 个参考引擎计数音符）与 DLL 对照一致，解析错误均为 0。此次 Python 检测约 0.12–0.22 秒/谱，属于本机这三份样本的观测，不是通用性能保证。
3. 相关 Python 回归 **528 项通过**（527 项组合回归，随后 API 移除检测器桩并新增状态测试，44 项 API 回归通过）；WebUI JavaScript 语法检查与 **6 项参数行为测试通过**。覆盖 CLI 三种处理模式、API 传参、旧外部路径迁移、单音符修复、跨窗长 Hold 与双 Hold 归因、日志和界面说明。
4. 检测测试拦截 DLL 加载和进程启动，并把旧库位置替换为不存在的路径，确认 Python 路径可独立检测。所有 616 个打包路径条目均可解析。另已成功构建 wheel，并在独立解包目录、无旧检测库且拦截动态库/进程调用的条件下完成检测；一键启动脚本的 `--check --no-install` 检查通过。

固定对照用例与预期结果保存在 `tests/fixtures/miacode_python_reference.json`。本机完整对照报告位于 `.scratch/muri_python_port/parity.json` 和 `parity.charts.json`；这些临时报告不提交仓库。

复查命令：

```powershell
python -m pytest tests/test_miacode_python.py tests/test_miacode_native.py tests/test_cross_window_repair.py -q
python tools/check_playability.py <maidata.txt> --mode local --output-dir <独立目录>
```

仅开发者需要重新核对原生版本时，可使用 `tools/build_muri.py` 与 `tools/compare_muri_python.py`。不需要用户为正常生成安装这些构建依赖。

## 界面与来源说明

WebUI 已显示“MiaCode · Python”，解释无需 DLL/Qt，继续保留对 [MiaCode 团队与贡献者](https://github.com/Team-MiaCode/MiaCode) 的鸣谢及 GitHub 链接。源码移植适用上游 MIT 声明；路径数值资源保留上游非商业分发范围，详见 `THIRD_PARTY_NOTICES.md` 和包内 `reference/LICENSE_SCOPE.md`。

对照通过表示这些样本的检测与修复行为得到保留，不代表所有 Simai 扩展语法都等价，也不是实机 Perfect 或人类双手可玩性认证。未支持的输入会显式报告解析问题，不应把零条报警当作通过。此次没有启动训练或修改模型权重。
