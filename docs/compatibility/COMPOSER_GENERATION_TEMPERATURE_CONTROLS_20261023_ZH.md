# Composer 生成温度解耦控制说明（中文版）

原文：[COMPOSER_GENERATION_TEMPERATURE_CONTROLS_20261023.md](COMPOSER_GENERATION_TEMPERATURE_CONTROLS_20261023.md)\
日期：2026-10-23

Composer 生成端现在除原有的 `temperature` 字段外，还支持独立的 `pointer_temperature`（候选拍位温度）和 `action_temperature`（动作配置温度）。若任一新参数未指定，则自动继承 `temperature`，从而保证旧版 API 客户端与命令行调用行为完全兼容。WebUI 与后端合同标识为 `composer_generation_controls_v5_workload_sampling`，启动前会拒绝旧版 v4 服务。

候选拍位采样器使用 `pointer_temperature`，单事件动作槽位采样器使用 `action_temperature`。实际生效的数值会写入运行时日志、任务设置快照及 `.composer.json` 生成报告。主 WebUI 与独立 Composer 页面均暴露了这两个控件，并提供保守默认值（拍位温度 `0.55`，动作温度 `0.85`）；同时已纳入生成参数持久化列表，刷新页面后仍会保留。结果摘要仅当两个实际生效温度同时为 `0` 时才标记为贪心解码（greedy decoding）。

本改动仅影响推理期采样，不改变训练目标或检查点 schema。
