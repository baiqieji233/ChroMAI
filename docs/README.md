# 使用说明导航

本项目当前主路线为 **Composer v2（`chart_composer_v2`）**：基于连续事件序列（`continuous_events_v1`）、75 Hz MERT 音乐特征与原生 100 Hz 频谱缓存、支持拍内声学起音参考与自动变速校准的 maimai 谱面生成与训练系统。

## 用户指南与参数手册

- **[项目总览与 3 分钟快速上手](../README.md)**：环境安装、双击一键启动、已有 `.pt` 模型生成与命令行速查。
- **[Composer v2 主界面（WebUI）与参数完整说明](explanations/COMPOSER_V2_WEBUI_ZH.md)**：WebUI 各工作区操作流程、生成与训练参数详解、Radar / Style 偏好微调、多数据集素材管理与跨设备路径配置。
- **[采音与候选评估](explanations/COMPOSER_V2_QUALITY_EVALUATION_ZH.md)**：主界面评估入口、采音检查参数、候选概率记录与选优偏向分析。
- **[Intel 核显与 NPU 加速](explanations/COMPOSER_V2_INTEL_ACCELERATION_ZH.md)**：MERT 加速、写谱后端自动测速、编译缓存与 XPU 训练。
- **[成对音乐参考与衔接示范](explanations/COMPOSER_V2_PAIRED_MUSIC_ZH.md)**：新音乐输入的用途、旧模型微调接入、人工确认前文衔接的操作。

## 许可证与第三方资源说明

- [源码许可证（MIT）](../LICENSE)
- [许可范围说明](../LICENSE_SCOPE.md)
- [第三方组件与预训练模型声明](../THIRD_PARTY_NOTICES.md)
