# 文档导航

项目当前主路线为 **Composer v2（`chart_composer_v2`）**：基于连续事件序列（`continuous_events_v1`）、75 Hz MERT 音乐特征与原生 100 Hz 频谱缓存、支持拍内声学起音参考与自动变速校准的 maimai 谱面生成与训练系统。

## 快速上手与使用指南

- [项目总览与 3 分钟快速上手](../README.md)
- [Composer v2 WebUI 与参数完整说明](explanations/COMPOSER_V2_WEBUI_ZH.md)
- [在另一台设备部署或继续训练 Composer v2](explanations/COMPOSER_V2_OTHER_DEVICE_20261107_ZH.md)

## Composer v2 核心机制与专项说明

- [Composer v2 架构与命令行说明](explanations/COMPOSER_V2_20261104_ZH.md)
- [连续事件序列训练与生成机制](explanations/COMPOSER_V2_CONTINUOUS_EVENTS_20261104_ZH.md)
- [自动变速识别与全曲节拍校准](explanations/COMPOSER_V2_VARIABLE_TEMPO_20261108_ZH.md)
- [拍内独立声学起音参考](explanations/COMPOSER_V2_ONSET_REFERENCE_IMPLEMENTATION_20261106_ZH.md)
- [可选 Slide 撞尾门控与清理](explanations/COMPOSER_V2_OPTIONAL_TAIL_GUARD_20261107_ZH.md)
- [生成页谱面预览播放器（MiaView Web）](explanations/MIAVIEW_GENERATION_PREVIEW_20261105_ZH.md)
- [多数据集管理与混合训练](explanations/MULTI_DATASET_TRAINING.md)
- [MiaCode Python 无理检测实现](explanations/MIACODE_PYTHON_PORT.md)
- [maimai 判定与谱面机制参考](explanations/MAIMAI_MECHANICS_RESEARCH.md)

## 本地开发资料

旧路线说明、交接记录与实验审计仅保留在维护者本地，不随发布源码上传。本导航只提供随源码发布的使用和技术说明。

## 许可证与资源范围

项目源码许可证见 [LICENSE](../LICENSE)，第三方资源及模型权重的适用范围见 [许可范围说明](../LICENSE_SCOPE.md) 和 [第三方声明](../THIRD_PARTY_NOTICES.md)。
