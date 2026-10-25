# Composer 负荷控制与 WebUI 集成交接文档（中文版）

原文：[COMPOSER_WORKLOAD_WEBUI_20261023.md](file:///e:/simai_transformer/docs/COMPOSER_WORKLOAD_WEBUI_20261023.md)  
日期：2026-10-23

## 范围

本次修改将 Composer 学习负荷（learned workload）控件与诊断指标集成到主工作台（main workbench）中。它不改变其他模型族，也不会把音频参考曲线编辑器接入 Composer 生成流程。独立的 Composer 页面仍保持在主工作流之外。

## 文件级变更

1. `src/maichart/webui/static/index.html`：新增仅针对 Composer 显示的数值控件 `composerCurveStrength`（范围 0～2，默认值 1）。原有的设备、温度、随机种子和条件控件保持共享不变。音频分析区的提示文案现在明确区分“仅供查看的音频参考曲线”与“保存在生成报告中的学习负荷曲线”。
2. `src/maichart/webui/static/app.js`：将该新字段与其他生成参数一并持久化保存，并作为 `curve_strength` 转发给后端。Composer 训练面板新增显示 `workload_loss`、`curve_slope_loss` 和 `workload_rate_mae`；其中 MAE 标签明确标注为 `log1p` 密度单位，不格式化为百分比，也不代表音频还原度得分。
3. `src/maichart/webui/composer_api.py`：校验 `curve_strength` 为 `[0, 2]` 区间内的有限数值，转发 `--curve-strength` 参数，并在任务快照中记录实际生效值。非法输入会在创建任务前直接拒绝；未传该字段的 API 调用默认取 1。
4. Composer 生成结果摘要会报告是否包含学习负荷曲线、曲线时间点数量、引导强度、门控检查次数（guard checks）、拒绝次数、跳过候选数以及停止原因。完整的 `times_sec`、`event_rate`、`head_rate` 和类型占比数组仍保留在可下载的 `.composer.json` 中，且不会阻塞谱面导出。
5. 结果摘要与完成提示现在统计全部 `playability.remaining_hard`，而不再仅统计 `remaining_actionable_hard`。因此，即使没有可自动修复的冲突，仅作报告（report-only）的硬冲突依然对用户可见；部分检测覆盖率不足的情况也会提示需要人工复核。
6. 生成合同升级为 `composer_generation_controls_v5_workload_sampling`。主界面在提交新控件前会拒绝旧版后台服务，防止旧进程静默忽略新参数。主实现必须在 `server.py` 的 `/api/status` 中发布相同合同并重启 WebUI 服务。

## 已准备的测试

- `tests/test_composer_generation_controls.py`：显式与默认引导强度转发、生效设置记录、越界/非有限输入拒绝。
- `tests/test_composer_generation_ui.js`：控件范围与默认值、请求载荷、服务合同校验、缺失曲线提示、引导强度为 0 时的文案、门控计数以及仅作报告的硬冲突展示。
- `tests/test_composer_training_ui.js`：负荷与斜率损失数值展示，以及不带百分号的 `log1p` 密度误差格式。

本 UI 实现阶段未执行测试或 GPU 任务；请在 Composer 核心修改完成后统一进行最终验证。

## 局限说明

该曲线反映的是模型学习到的谱面负荷，而非音乐质量的客观度量。剩余硬冲突数为 0 仅代表检测器结果，不等同于人工试玩认证。旧检查点可能不具备负荷预测能力，界面会对此做明确提示。
