# Composer v1 旧生成参数

这里保留旧 README 中用于 `chart_composer_v1` 的参数参考，主要适用于 `audio_grounded_v3`。主工作台会按检查点能力显示控件。整拍集合架构见[整拍集合参数](COMPOSER_STREAM_PARAMETER_GUIDE_20261031_ZH.md)，当前 Composer v2 见 [Composer v2 使用说明](../explanations/COMPOSER_V2_WEBUI_ZH.md)。

## 采样与配置

| 旧控件 | 含义 |
| --- | --- |
| 候选拍位温度 | 控制下一事件的落点。旧教程用 `0.55` 作对照起点。 |
| 动作配置温度 | 控制键位、修饰与时值的采样。旧教程用 `0.85` 作对照起点。 |
| 动作类别温度／动作类别 min-p | 调整各音符大类的采样集中度和低概率截断。旧参考 `1`／`0` 保留默认类别分布。 |
| 抑制 Touch 历史自强化／Touch 历史放大上限 | 限制已生成前文额外抬高 Touch 概率，受检查点及采样方式限制。上限值描述赔率变化，实际 Touch 比例以生成结果为准。 |
| Touch 组合策略 | 开放模式将按钮预算与 Touch 分开计算，旧模式可复现此前的计数方式。 |
| 选时历史权重／动作历史权重 | 分别调节前文对选时与配置的作用。降低可能削弱跨拍关联。 |
| 动机记忆强度 | 调节已训练动机分支的影响。没有训练该能力时不生效。 |
| 失败重试策略 | 按检查点自动选择，或指定该模型支持的失败前缀重试策略。 |
| 同键最短间隔／拦截 Slide 撞尾（TapOnSlide） | 旧路线的局部冲突处理设置。它们分别作用于同键过快重复和 Slide 撞尾。 |
| 学习负荷曲线引导强度／强度曲线指导强度 | 分别调节模型预测的负荷与手绘曲线。适用范围由架构和检查点决定。 |

候选拍位和动作配置的温度可分别设置。两项都设为 `0` 才是全程贪心。CLI 在省略专用温度时可回退到 `--temperature`。各历史策略的含义见[温度说明](COMPOSER_GENERATION_TEMPERATURE_CONTROLS_20261023_ZH.md)和曲线与目标记录（本地开发资料，不随源码发布）。

## 条件 JSON

旧 v1 入口可在 **condition JSON** 中填写 UTF-8 JSON 文件路径。文件内明确填写的字段优先，省略的字段沿用当前界面或程序默认值。对照前先查看任务保存的生效条件。

```json
{
  "level": 13.3,
  "style": {
    "tap_bias": 0.70,
    "slide_bias": 0.16,
    "hold_bias": 0.09,
    "touch_bias": 0.06
  }
}
```

`difficulty` 可指定输出的难度槽位，`level` 是输入给模型的等级。旧 v1 使用 `maichart.composer.generate --condition-file`，当前 v2 使用 `maichart.composer_v2.generate --condition` 传入显式 Radar／Style 字段，两者格式不同。
