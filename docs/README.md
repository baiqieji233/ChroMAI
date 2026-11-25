# ChroMAI 使用说明导航

- [Composer v2 架构、训练与连续事件生成](explanations/CHROMAI_COMPOSER_V2_ARCHITECTURE_AND_TIMING_ZH.md)
- [ChroMAI 原理与技术（早期版本）](explanations/CHROMAI_PRINCIPLES_AND_TECHNOLOGY_ZH.md)
- [ChroMAI 与 Gemai 的设计对照](explanations/CHROMAI_GEMAI_DESIGN_COMPARISON_ZH.md)


**ChroMAI** 的当前主路线为 **Composer v2（`chart_composer_v2`）**：基于连续事件序列（`continuous_events_v1`）、75 Hz MERT 音乐特征与原生 100 Hz 频谱缓存、支持拍内声学起音参考与自动变速校准的 maimai DX 谱面生成与训练系统。

## 用户指南与参数手册

- [ChroMAI 怎样学谱、怎样写谱](explanations/COMPOSER_V2_FOR_CHARTERS_ZH.md) 用流程图说明歌曲记忆、近期配置与相似段参考怎样参与训练和连续写谱。
- [盘面走势、Slide 轨迹与局部变速](explanations/COMPOSER_V2_GEOMETRY_AND_TIMING_ZH.md) 介绍近期改动、迁移方式及两轮对照结果。
- [编排构思学习](explanations/COMPOSER_V2_PHRASE_INTENT_ZH.md) 说明模型怎样从真人乐句借鉴写法，以及这项实验在真实续写中表现如何。
- **[项目总览与 3 分钟快速上手](../README.md)**：环境安装、双击一键启动、已有 `.pt` 模型生成、训练素材准备与命令行速查。
- **[Composer v2 技术架构与目录说明](explanations/COMPOSER_V2_TECHNICAL_OVERVIEW_ZH.md)**：模型如何读取音乐和历史、训练目标、缓存边界、主要技术目录与质量排查思路。
- **[Composer v2 主界面（WebUI）与参数完整说明](explanations/COMPOSER_V2_WEBUI_ZH.md)**：WebUI 各工作区操作流程、生成与训练参数详解、Radar / Style 偏好微调、多数据集素材管理与跨设备路径配置。
- **[Composer v2 短程微调](explanations/COMPOSER_V2_SHORT_EXPERIMENTS_ZH.md)**：学习率下降曲线、衰减步数和生成前文对照实验。
- **[强度曲线与局部疏密编辑](explanations/COMPOSER_V2_INTENSITY_ZH.md)**：可学习事件率与同拍物量曲线的工作原理，以及生成页音频定位与曲线编辑器的用法。
- **[短段试排、音乐上下文与片段反馈](explanations/COMPOSER_V2_PHRASE_AND_MUSIC_ZH.md)**：短段草稿比较、音频编码扩展、全曲音乐摘要与片段评价／微调入口。
- **[成对音乐参考与衔接示范](explanations/COMPOSER_V2_PAIRED_MUSIC_ZH.md)**：成对音乐对比与前方采音读取的启用方式，以及人工确认前文衔接的操作步骤。
- **[完整谱面审阅与反馈](explanations/COMPOSER_V2_PREVIEW_ZH.md)**：在线谱面播放器、时间轴选区、A/B 对比试听与「收集训练示范」的详细说明。
- **[候选库与片段审阅](explanations/COMPOSER_V2_CANDIDATE_POOL_ZH.md)**：建立待审候选库、查看自动检查提示、逐段确认可学习内容并估算学习份额。
- **[审阅、标注和片段包操作指南](explanations/COMPOSER_V2_ANNOTATION_GUIDE_ZH.md)**：按界面顺序处理检查结果、审核素材与生成成品、导出与导入片段包并用于后续微调。
- [动作走向、候选选优与整谱评价](explanations/COMPOSER_V2_MOTION_AND_REVIEW_ZH.md)说明旧模型怎么用新选优、新策略怎么微调，以及怎样验证实际效果。
- **[采音与候选评估](explanations/COMPOSER_V2_QUALITY_EVALUATION_ZH.md)**：主界面评估入口、采音检查参数、候选概率记录与选优偏向分析。
- **[Intel 核显与 NPU 加速](explanations/COMPOSER_V2_INTEL_ACCELERATION_ZH.md)**：OpenVINO MERT 加速、写谱后端自动测速、编译缓存与 Intel XPU 训练。

## 许可证与第三方资源说明

- [源码许可证（MIT）](../LICENSE)
- [许可范围说明](../LICENSE_SCOPE.md)
- [第三方组件与预训练模型声明](../THIRD_PARTY_NOTICES.md)
