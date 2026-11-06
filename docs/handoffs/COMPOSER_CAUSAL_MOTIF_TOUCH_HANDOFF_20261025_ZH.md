# Composer：Touch、因果持续状态与动机记忆接续（2026-10-25）

本轮按用户批注实施，不以独立审查中的P0标签代替复现。主界面仍是唯一维护的生成/训练入口；不扩展`$`、反引号或分段独立时值Slide语法。

## 当前训练与基线

- 原任务`ea4214e361fb`已在第4轮、12,500次优化器更新保存后停止。停止操作不会保存尚未落盘的少量更新；本轮明确以12,500为恢复点。
- `checkpoints/Prism_composer_v10`保留；另存`checkpoints/Prism_composer_v10_preserved_20261025`，含完整优化器的last与第3轮11,124更新的inference。两文件不是同一个时刻。
- 完整last SHA-256：`8e28101625301a56af90c0c102a625f89b1878b6319d7ee64318b596035c371f`。
- inference SHA-256：`08857a58d6f3b6d240f7d9df97ebcdcf3bd6314ec0c26b41a97678622f14a57e`。
- 未把小子集实验权重接入80轮长程训练。旧任务记录在7860，更新主界面在7870；重新启动以终端实际端口为准。
- Git分支：`codex/composer-causal-motif-touch-20261025`。权重、训练集、生成音频与scratch实验不上传GitHub。

## 已实施的行为

### Touch独立预算

生成默认`touch_layout_policy=open`，按钮仍最多两头；Touch不消耗按钮预算，可同拍输出三颗以上Touch。按钮`1`和屏幕Touch`A1`在此策略的占用/cooldown键中独立。TouchHold也不再挤占按钮预算。

保留语法、实际头重复占用、最短间隔及现有MiaCode事务检查。没有宣称Touch连通组就等于一只手，也没有新加未经验证的Touch几何硬规则。运行时对Touch的双手可玩性覆盖仍有限。

`legacy_two_heads`保留旧计数、头别名、候选容量门槛和cooldown，可复现原结果。训练恢复采样的策略单独保存：fresh自动open，旧检查点缺字段时继承legacy，resume不得偷偷改换。

回归中发现：在纯按钮词表中，直接放行双Hold占满的时刻会反复拒绝、推进搜索游标，改变后续落点。已补`_candidate_has_capacity`：两按钮被占时，只有词表中确有未被占用的Touch配置，才保留该候选。训练恢复与生成共用它。

### 几何与Slide因素

- `parsed_geometry_v3`新增为grounded新训练的自动角度锚。中心C没有外圈角度；D/Ek在相邻两键各分配0.5权重；多头逐音符累加后归一化。
- `legacy_digit_v1`与`parsed_heads_v2`含义不变。旧检查点缺字段时明确legacy，不自动升级。
- `branch_joint_v2`在原有边缘因素之外新增联合分支因素，绑定路径、相对终点、时值及段修饰符。分支排序不改变语义，交换不同分支的时值可以区分。
- 新联合因素从零初始化。显式迁移复制已有可学习参数，派生映射及fingerprint由新策略重建，不覆盖为旧映射。
- 原始动作ID仍保留，没有按因素碰撞合并词表，也没有扩大模型到词表以外生成新语法。

### 持续状态及因果性

已复现完整谱索引与增量索引的浮点起拍边界不一致；190 BPM合成序列36/47个非初始边界漏掉新Hold。这是合成复现率，不能外推为全语料比例。

| 策略 | 行为 |
| --- | --- |
| `committed_legacy_v1` | 保留旧查询与均值，为旧checkpoint精确继续提供兼容解释。 |
| `committed_causal_v2` | 新训练自动选项。仍在最后已提交事件时刻查询，修正浮点边界，严格屏蔽未提交事件及其following-head影响。 |
| `candidate_prefix_v2` | 实验选项。在各候选时刻推进同一已提交前缀，返回`[候选数,50]`，不包含目标或中间未来真值。 |

所有50维原特征保持含义，未挤掉EX维。Slide时间计算复用现有等待期、路径、时值、变BPM与释放语义，未添加“尾音必须命中瞬态”的硬限制。新增精确`source_beat`，可见性截止点与查询时刻分离；训练、恢复、生成共用`context_features`。

prep缓存把`source_beat`写为有理数字符串、读取为Fraction。旧缓存没有该字段时，新策略会重建索引；不能假设修改代码后所有旧缓存自动失效。

两张真实谱、各96候选的CPU查询中位数：已提交状态0.036/0.123 ms；逐候选2.14/3.30 ms。它是状态查询成本，不能直接外推整轮训练倍数；因此逐候选策略保留为非默认。

### 可选跨段落动机记忆

`motif_memory_policy=audio_recurrence_archive_v1`基于已有音频段落相似关系，检索普通历史窗口以外、已经提交的事件。事件可见性按提交数及精确拍位限制；无匹配或空前缀时残差严格为零。

- 默认`off`，只能通过新训练或新目录显式finetune增加分支。
- 每次最多16个档案事件，可设1–32。检索使用有界近邻，不逐候选展开全曲历史。
- **第一版只在已选择拍位后辅助配置解码**，不直接改变pointer、EOS、future或纯音频动作参考。训练与恢复、生成使用同一检索入口；训练用已选目标拍位查询已提交档案，不把未来谱面作为档案。
- 适配器输出层零初始化、输出有界。只开启新分支并保持其他策略不变时，迁移初始输出与父模型一致，且新分支能收到梯度。
- 生成`motif_strength`范围0–2、默认1，0关闭。旧模型off时调强度不会产生记忆能力。报告记录查询数和匹配数，不能把“开启开关”当成“实际调用到记忆”。

此版本不是整个旧Stream系统的移植，不包含显式手分配、强制配置复现、镜像增强或时长终点音频评分。

### 主界面与运行稳定性

- Composer隐藏不生效的通用Temperature，只使用两项独立温度；两个都设0才是完全贪心。
- 新策略在主工作台展示、提交、检查点继承、finetune保留、resume锁定、默认重置及任务回显中保持一致。
- `EventGuard`解析缓存改为共享纯函数，不再强引用旧Guard实例；返回独立副本，避免污染共享缓存。
- queued/进程启动空窗/线程任务都记录取消请求。实际退出前显示`cancelling`，不提前称已停止；取消中仍算忙碌，完成状态不覆盖请求。
- 非grounded架构不支持的motif、逐候选状态、联合Slide因素会在创建Dataset/任务前拒绝；支持的非grounded状态迁移正确写入model_config。

## 检查点与训练方法

全局训练语义仍为v11，新增解释由明确的model_config字段约束。旧v11仍可按原策略resume；源文件没有热更新正在运行模型的能力。改变策略应使用finetune和新目录，保持父权重不动。

主界面操作：选`finetune`，检查父checkpoint，再明确选择需要试验的策略。要试动机分支，开启动机记忆；其他策略可以保持继承，以便单独归因。训练前提交检查不会把这些显式选择重置。CLI对应：

```powershell
python -m maichart.composer.train --manifest YOUR_MANIFEST.json --init-checkpoint PARENT.pt --checkpoint-dir checkpoints/motif_trial --motif-memory-policy audio_recurrence_archive_v1 --motif-max-events 16
```

以上示例只改变动机分支；没有要求每次把几何、持续状态、Slide因素一起升级。`--action-state-policy committed_causal_v2`、`--history-anchor-policy parsed_geometry_v3`、`--slide-factor-policy branch_joint_v2`和`--touch-layout-policy open`是各自独立的显式选择。设置学习率、batch等时同时核对有效值。

## 已运行的验证与边界

### 同权重双种子Touch对照

固定step12500权重、同一音频、条件、BPM200、first0、候选/动作温度0.55/0.9、同头间隔70ms。两seed的当前legacy输出都与冻结修改前源码**逐字节相同**。

| seed | 模式 | notes | Touch含TouchHold | 最大同拍Touch | 全部hard / actionable | 最长纯Tap半拍链 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 709835971 | 原/legacy | 875 | 7 / 0.80% | 2 | 19 / 0 | 5 |
| 709835971 | open | 857 | 55 / 6.42% | 4 | 6 / 0 | 9 |
| 1696576229 | 原/legacy | 695 | 49 / 7.05% | 2 | 5 / 0 | 5 |
| 1696576229 | open | 737 | 52 / 7.06% | 3 | 8 / 0 | 8 |

六组均全曲结束、无后处理删除；低于50ms同头间隔计数为0。剩余hard为report类TapOnSlide/SlideHeadTap/SlideTooFast。Touch容量已恢复，但冲突和半拍长链未一致改善，**不是音乐质量验收通过**。

### 真实数据短微调

从父模型完整18,453原子词表选32个真实重复段样本，来自4张谱，事件索引201–479，档案每行13–16个事件。不是四首已验证独立歌曲，也不是未见歌曲验证集。

30步小batch微调后，对同一16批/32行关闭dropout评估：总loss4.384627→4.229252，动作2.937940→2.841322，指针0.753837→0.712847；指针准确率保持0.6875。新联合因素与动机输出层30/30步梯度有限且非零。峰值GPU分配约853MiB，训练加前后评估5.35s。

这证明接线与小子集优化正常，不能把总loss下降独自归因于动机记忆，也不能证明自由生成音乐质量改善。诊断checkpoint仅推理、无optimizer，显式标记diagnostic_only；不可resume到正式训练。

官方CLI另用真实一张谱做2次更新，开启逐候选状态、动机、几何v3、联合因素及每步恢复采样，完成保存/导出；一次恢复样本接受，另一次因原参考续接非法而拒绝，没有伪造动作恢复标签。

随后从该完整实验检查点经官方CLI成功resume到第2轮/4次更新，验证新配置、数据指纹和优化器恢复路径。该小模型只用于入口检查，与生产v10和30步诊断权重分开保存。

### 动机完整生成验证

同一30步诊断权重，沿用固定音频与双seed，分别用动机强度0/1生成四组完整谱面。所有模型输出有限，但该曲的唯一音频关联源仍在普通128事件历史内，档案匹配为0/590及0/571；同seed开关输出逐字节相同。这证明无匹配时保持中性，不是动机效用对照。

额外从上述4张训练谱内选择BUDDiES/11658：BPM170、first0、原条件level12.8，音频关联从拍[57,74)到[385,402)，间隔足够远。保持相似度阈值与128历史窗口，单seed开关0/1均实际匹配52/512次查询。所有输出有限，该seed最终谱面仍逐字节相同（491事件、679音符、hard20/actionable0）。这是已见音频上的功能验证，证明档案在自由生成中确实可用，尚未证明配置或音乐质量改善。

为排除“分支仍为零”的误判，另对同一真实2行批次切换强度0/1：动作logits最大差0.10982、RMS差0.03043；拍位、EOS、future及纯音频配置输出逐值不变。最终抽样相同不等于分支未接通，但这仍不是音乐收益证据。

### 最终回归

相关Python回归**459 passed**，主界面Node回归**28 passed**；真实Edge浏览器验证通过。两项独立温度、Touch/动机生成请求、真实旧模型检查、resume锁定、finetune提交前重检保留及fresh重置均已验证。更新后端重启后再次检查通过。Git diff空白检查通过；父目录与备份两份检查点的哈希均保持一致。

## 证据目录与剩余工作

本机证据：`.scratch/composer_revision_20261025/`，包括：

- `preserved.json`：原检查点配置、参数与哈希。
- `baseline_src/`、`before/`：本轮修改前基线源码。
- `paired_v10/touch_comparison.json`及各臂完整输出：六组Touch对照。
- `motif_finetune_256/preparation.json`、`training_trace.json`、`prefix_query_benchmark.json`：真实选样、短微调、状态查询成本。
- `official_cli.log`：正式训练入口、恢复采样及保存验证。
- `official_cli_resume.log`：新配置官方CLI恢复路径。
- `motif_generation/`、`seen_motif_generation/`：动机无匹配与实际匹配两类完整生成诊断，含权重/音频/条件指纹。
- `browser_policy_result.json`：真实Edge主界面验证；训练/生成POST被测试桩拦截，未产生额外正式任务。
- `regression_final.log`：当前相关回归；最终数量以末次运行记录为准。

尚需：未见歌曲验证、较长但受控的动机微调、对实际有音频复现匹配的曲目做自由生成和人工试听/游玩检查；Touch开放后的混合配置与残留report冲突也应继续测。默认宏采样、report/enforce策略、损失权重和未验证的重复惩罚没有被一起改动。
