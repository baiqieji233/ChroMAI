# ChartComposer 配置能力学习效率：项目、训练实现与研究交接

**用途：**交给下一位智能体，研究如何让模型用更少的更新、样本和GPU时间学会更好的配置，包括跨拍走位、Hold/Slide选择、Touch布局、段落呼应，以及生成自身历史上的稳定性。

> **根因调查最新接续：** 已完成6组GPU真实数据实验。主要结构性失稳入口是未独立约束的历史memory进入跨注意力K/V及tail融合，实测存在查询梯度放大和同向塌缩；保留0.0002、只约束记忆接口也能阻止本段退化。正式训练继续停止，原结构/新策略迁移的边界与15,500新增保留点见[根因调查](COMPOSER_INSTABILITY_ROOT_CAUSE_20261026_ZH.md)。不要把实验RMS上限8当成已经上线的控件。

> **后续审查接续：** 已修复诊断入口的motif/selected-state失配，并建立正确加权、固定权重的评估。额外确认第16,000→16,500步在同一probe退化，已保留16,000及17,000完整点并取消v12任务，尚未重启。复核与下一步稳定性实验见[配置能力审查复核](COMPOSER_CAPABILITY_REVIEW_20261026_ZH.md)。下文15:30运行快照保留为此前状态，不能再按“仍在训练”操作。

**源码基准：**`dfa5620`，工作区`E:\simai_transformer`。本交接只增加文档和索引，不改变训练实现或启动新实验。下文运行快照核实于 **2026-10-26 15:30:02，Asia/Shanghai**；训练会继续前进，接手时必须重新查询。

这里的“提高配置的学习率”指**配置能力的学习效率**。WebUI“学习率 lr”只是优化器步幅；下一位需要同时研究每次更新能学到什么、每小时能执行多少次更新，以及这些学习能否转化成实际谱面改善。

## 1. 项目是什么，目前处于什么阶段

项目输入音乐、准确时间设置、目标等级及Radar/Style条件，输出可继续编辑和试玩的Simai谱面。当前主线是事件级ChartComposer：先选择下一组音符的时间，再逐槽组合这个时间的音符；不是逐字符续写Simai文本。

| 名称 | 当前含义 |
| --- | --- |
| 模型族 | `chart_composer_v1` |
| 主架构 | `audio_grounded_v3` / `AudioGroundedComposer` |
| checkpoint schema | `maichart-chart-composer-checkpoint-v2` |
| 输入合同 | `chart_composer_inputs_v2_audio_only_history_state` |
| 当前grounded训练合同 | `chart_composer_training_v11_history_relation_mask` |
| 动作采样策略 | `calibrated_kind_touch_rings_v2` |
| 动机训练合同 | `symbolic_masked_archive_v1` |
| 当前实验目录 | `checkpoints/prism_composer_v12` |

**目录v12、架构v3、模型族v1、训练合同v11是不同版本轴，不能互相替代。**旧event、song-form、stream代码仍在仓库，也有共用工具，但不代表它们是当前正在训练的网络。`AudioGroundedComposer`暂以music-context配置初始化共享主干以保留参数名字，这是实现继承方式，不是误加载旧架构。

生成已能表现出一定的歌曲强弱跟随、舍音和类型选择；用户仍观察到八分连点堆砌、配置衔接不成熟、Touch反馈和区域分布问题。自动冲突少、物量多、loss低都不足以宣布音乐质量通过。当前没有完成专业试听/试玩意义上的整体质量验收。

## 2. 当前任务、环境与不可覆盖的基线

主工作台为`http://127.0.0.1:7860/`，训练和生成都在根路径主界面。**独立`/composer`页面已弃用，不要只在那里接参数。**

| 快照项目 | 15:30:02核实结果 |
| --- | --- |
| 正在运行的训练任务 | `78f828eaac2a`，状态running |
| 日志进度 | 第4轮，batch5950，优化器更新12,611 |
| 当轮累计总loss | 约4.2329；不是固定验证集loss |
| 当时磁盘完整检查点 | 第4轮 / 12,500更新，轮内未完成，`can_resume=true` |
| 当前输出目录 | `E:\simai_transformer\checkpoints\prism_composer_v12` |
| 数据清单 | `E:\全量官谱\training_manifest.json` |
| 训练缓存 | `E:\simai_transformer\cache\chart_composer_v1` |
| 目标结束轮 | 第80轮 |
| 最近恢复起点 | 第3轮 / 10,500更新，样本游标789,504 |
| 恢复请求epochs | 78，包含未完成的第3轮；不是另外增加78个完整轮 |
| 当前日志 | `webui_workspace/logs/20261026_144837_composer_train_78f828eaac2a.log` |

上述最新检查点路径会被训练正常更新，不能把路径相同当作权重相同。固定对照前复制一份并记录哈希，不在正在变化的`composer_last.pt`上反复比较。

已经独立保留的完整基线：

| 基线 | 路径 | 文件SHA-256 |
| --- | --- | --- |
| v12，第3轮/10,500更新，motif开启 | `checkpoints/Prism_composer_v12_preserved_20261026_step10500/composer_last.pt` | `140ebbe2cdb87818de3ab4169cee60e23b074f724fcc38c1e46789bb748cdfba` |
| v11，第5轮/17,500更新，motif关闭 | `checkpoints/Prism_composer_v11_preserved_20261026_step17500/composer_last.pt` | `6fa66cf808e977d6f0cee6154aec2c105aaa33b844a3d87f15b2a157ff06157f` |

v12上次停止前日志到10,747更新，恢复用的是10,500完整保存点，未保存尾段已重跑；不能宣称暂停保存了退出前所有内存状态。恢复后已实际验证第3轮完成/11,124更新的新完整检查点，配置、数据指纹和优化器LR均一致；本交接快照又核实了12,500点。

本机GPU为RTX 5070 Ti Laptop、约12GiB显存，主存约32GiB；PyTorch为`2.7.0+cu128`。实际训练由本机Python311启动，不能未经检查就假定`.venv`是相同环境。运行诊断前核实`sys.executable`、`torch.__version__`、实际导入路径和任务占用。

## 3. 先读哪些源码

| 路径 | 负责内容 |
| --- | --- |
| [composer/data.py](../src/maichart/composer/data.py)、[event_views.py](../src/maichart/composer/event_views.py) | 数据来源、可信监督、候选覆盖、标签、历史、采样权重、collate |
| [composer/events.py](../src/maichart/composer/events.py)、[candidates.py](../src/maichart/composer/candidates.py) | 原子/实际头位定义、历史特征、头位时间、候选网格与时间特征 |
| [composer/model.py](../src/maichart/composer/model.py)、[grounded_model.py](../src/maichart/composer/grounded_model.py) | 配置/策略版本、共享主干、音频锚定、指针与动作解码 |
| [composer/action_factors.py](../src/maichart/composer/action_factors.py) | 独立词表头之外的结构因子、几何锚、同拍相对关系 |
| [composer/audio_lens.py](../src/maichart/composer/audio_lens.py)、[acoustics.py](../src/maichart/composer/acoustics.py) | 局部听音、占用标签、长短窗口音频、积分与插值 |
| [composer/planning.py](../src/maichart/composer/planning.py)、[workload.py](../src/maichart/composer/workload.py) | 四拍跨度粗规划、负荷监督与已提交负荷台账 |
| [composer/prefix_state.py](../src/maichart/composer/prefix_state.py)、[context_features.py](../src/maichart/composer/context_features.py) | 持续状态的严格已提交前缀边界，训练/生成共用查询 |
| [composer/motif_memory.py](../src/maichart/composer/motif_memory.py)、[form/recurrence.py](../src/maichart/form/recurrence.py) | 音频段关系、旧配置档案检索、动机残差 |
| [composer/train.py](../src/maichart/composer/train.py)、[recovery.py](../src/maichart/composer/recovery.py) | sampler、forward、loss、优化器、短程恢复采样 |
| [composer/generate.py](../src/maichart/composer/generate.py)、[action_guidance.py](../src/maichart/composer/action_guidance.py)、[guard.py](../src/maichart/composer/guard.py) | 逐事件生成、分组采样、Touch保护、提案接受及审计 |
| [composer/checkpoint.py](../src/maichart/composer/checkpoint.py)、[training_lock.py](../src/maichart/composer/training_lock.py) | 合同兼容、原子保存、优化器/游标/RNG、同目录进程锁 |
| [form/layout_prefetch.py](../src/maichart/form/layout_prefetch.py) | 共用CPU预取、同音频源锁、准备线程 |
| [webui/composer_api.py](../src/maichart/webui/composer_api.py)、[server.py](../src/maichart/webui/server.py) | 主界面API、启动训练子进程、任务与日志状态 |
| [webui/static/index.html](../src/maichart/webui/static/index.html)、[app.js](../src/maichart/webui/static/app.js) | 当前实际控件、参数保存/重置/继承、运行摘要 |

`stream/grounded_inference.py`的`IncrementalState`、共用Simai编译器和`generation/muri`也被Composer使用；不要因文件位于旧模型目录就删掉这些依赖。

## 4. 数据和一条监督样本

训练来源是原始`maidata.txt`、对应音频、难度/等级及质量标注，经manifest和准备缓存建立事件视图。时间用`Fraction`和tempo map保留精确拍位，音频偏移进入统一时钟。

常规样本对应一个可监督的原谱事件：输入目标之前已经发生的事件，提议下一拍位候选；标签是原谱事件在候选中的索引，以及该拍的动作原子序列。music/grounded路径对同拍原子按字符串排序，建立唯一teacher-forcing顺序，末尾用STOP结束。

- 历史切片为`[start:ev]`，目标为`[ev]`。历史encoder虽可双向注意这个窗口，但窗口本身不含当前/未来目标。
- 质量未知或排除区域通过EventViews及各项mask控制，不能把未知事件、未知尾段当成应该结束谱面的证据。
- 候选提议并非所有有理拍位。当前参考路径候选总计952,710、miss3,695，有效样本949,015；未覆盖事件计数后不作为该目标监督，覆盖不足会触发质量检查。
- 参考路径约99.61%的覆盖率，不证明生成自身历史下也能到达正确拍相位。不要向候选插入原谱答案来伪造可达性。
- 条件是16维等级/Radar/Style目标。训练条件来自原谱统计，生成时来自用户等级预设/输入；条件分布差异也应作为学习诊断的一部分。`melody_following`等字段是统计代理，不是已有独立人声旋律标注。

当前主要张量：

| 输入/标签 | 典型形状与含义 |
| --- | --- |
| candidate_features | `[B,96,141]`，120维音频与21维时间信息 |
| condition | `[B,16]` |
| history / history_features | `[B,H,S_h]`原子ID及`[B,H,8]`事件特征，H≤128 |
| head_timing | `[B,96,99]`，候选相对历史头区的seen/秒龄/拍龄 |
| action_state | 当前v12为`[B,50]`，最后已提交拍位的持续状态摘要 |
| audio_lens | 通常`[B,112,31,26]`，96候选加16个全曲均匀监督探针 |
| workload_audio / interval_audio / trailing_audio | 短长音频上下文104维、间隔16维、过去窗口52维 |
| planner_audio / planner_mask | 每首歌四拍计算跨度的整曲音频与padding；跨度不等于已知音乐小节 |
| motif_ids / motif_mask | `[B,16,S_m]`及`[B,16]`，严格来自可见的旧档案 |
| action_target | `[B,S_t]`，本batch实际原子数+首个STOP，容量上限当前33 |

`S_h/S_m/S_t`动态裁掉无效尾列，不能把配置容量33误解成每拍必须输出33个音符。完整词表共有18,453个动作原子，另加ID0的STOP。原子可以包含紧凑多头或多分支Slide，不等于一颗按钮头。

一个应明确记录的输入约定：`events.py`的head timing使用33个pad组，按钮数字1..8映射到A1..A8，因此是99维，而不是41种独立按钮/Touch头位的123维。这是当前辅助时序表示的约定；原token和kind/ring因子仍区分按钮与Touch，不能据此认定二者在整个模型中完全混淆。若研究其影响，需独立版本化表征实验，不能直接改维度后硬resume。

## 5. 当前模型怎样作决策

```text
完整音频 + 时间设置 + 等级/Radar/Style
  ├─ 候选音频Transformer、31帧局部TCN、间隔上下文
  ├─ 负荷预测、占用预测、整曲四拍跨度粗规划
  └─ 音频选位参考与音频动作参考

严格已提交前缀
  ├─ 历史原子/几何锚/历史Transformer与注意力
  ├─ 持续状态、head timing、已提交负荷ledger
  └─ 可选的跨段motif档案

音频 + 受遮挡的符号上下文
  → 分时间区间的层次候选指针
  → 训练用真实目标索引；生成用模型选定索引
  → 同拍逐槽动作头：原子、原子、…、STOP
  → 生成时提案约束/提交，再编译Simai并做Muri审计
```

**主干与动作头。** 当前宽度384、音频与历史encoder各6层、8个注意力头、FF1536，约40,976,103个参数。历史表示融合解析几何锚、独立事件embedding和结构因子embedding。动作slot融合此前同拍原子的embedding和，训练时来自teacher forcing，生成时来自实际已选原子。

输出仍有独立全词表`Linear(384,18454)`，约710万参数，额外叠加结构因子logits。**因子共享不等于已经改成“先类型、再头位、再路径/时值”的层次解码，也未消除稀有独立词表行的样本稀疏问题。**

当前`branch_joint_v2`把每条Slide分支的路径/端点/时值绑定为联合属性，避免只看边际属性。`relative_logits`主要作用于同拍此前已选头位；跨事件仍通过history、`action_relation(last_heads)`、head timing与状态学习。起点锚不是完整双手轨迹；Slide的结束位置也不能在尚未完成时无条件当成当前手位。

**音频辅助与计划。** 音频动作参考不含跨事件符号历史，但后续slot仍有同拍teacher-forced前缀。它与主动作共享slot融合、词表头及因子。workload预测、候选plan读出与placement概率在进入主决策前detach，它们各有直接监督；不能说动作loss通过这些输出端到端训练了planner，也不能说整个音频encoder因此断梯度。

**状态策略。** 当前v12为“已提交事件的因果状态”。`selected_prefix_v3`另在选定拍位对同一已提交前缀查询一次状态，只替换动作输入，保持pointer/EOS/future/音频参考路径；当前任务没有启用它。`candidate_prefix_v2`则对所有候选查询，会改变选时输入并增加CPU成本。更换策略需要新阶段，不能仅在生成页临时切换。

**动机。** v12已开启`audio_recurrence_archive_v1`，v11保留点为off；fresh源码默认仍off。它按音频相似关系检索普通128事件窗口之前、已经提交的最多16个事件，以软残差辅助动作，不直接改pointer、EOS或future。当前固定生成曲目曾出现matched_queries=0；8-chart性能fixture也全部空，但已找到真实非空样本验证。step10500的motif输出权重norm约7.42，表明它已发生训练更新；这不证明检索覆盖率或质量收益已经充分。

## 6. 当前训练循环、优化器与运行设置

主界面`composer_api.start_train`构造训练子进程命令；`train.py`准备数据、加载合同和模型，再建立采样器、CPU预取与优化器。状态由结构化日志返回主界面。

| WebUI项目 | 当前运行值 |
| --- | --- |
| 学习率 lr / 优化器实现 | 0.0002 / CUDA fused AdamW |
| batch size / 梯度累积 | 64 / 4，有效样本量约256 |
| 数值精度 / 普通 dropout | BF16 / 0.1；模型参数与Adam状态FP32 |
| 历史分支遮挡概率 | 0.3，按历史事件遮挡，包含最后一个事件 |
| 历史模态遮挡概率 | 0.15，整条符号上下文与motif一致遮挡 |
| 台账遮挡概率 / 历史残差幅度上限 | 0.5 / 1 |
| CPU 算子线程 / 样本准备线程 / 后台预取 batch 数 | 2 / 1 / 2 |
| 样本局部性 | 4首活动谱面，保持缓存局部性 |
| checkpoint 保存间隔（更新步） | 500，并在轮末保存 |

AdamW采用betas=(0.9,0.999)、eps=1e-8、weight decay0.01。**当前没有LR scheduler、自动验证集或基于独立验证的早停。**BF16不使用FP16 GradScaler属于正常实现。

每个microbatch的loss除以实际累积窗口批数，轮尾短窗口也正确处理；每次更新前clip全局梯度norm到1，再step，检查参数finite后才允许保存。GPU输出、loss、梯度、更新后参数的非有限保护保留。

`SongLocalBatchSampler`随机打乱歌曲和歌内样本，但每批来自约4首活动谱面，相邻多个更新可能继续使用同一组歌曲。该策略改善缓存，是否造成梯度相关性和配置学习局限尚未证实。

`LayoutBatchPrefetch`是共用内存的CPU线程预取，CUDA始终由训练主线程执行；不是Windows多进程复制整个dataset。一次更多准备线程曾减少data等待却拖慢GPU提交，当前实际选1。已测试的跨源round-robin调度在真实负载变慢，生产实现已撤回；不要把其sleep微测收益当作GPU训练收益。

## 7. loss精确怎样计算

常规报表总loss为：

```text
主选时 + 主动作
+ 0.05 × EOS + 0.2 × future
+ 0.4 × workload + 0.05 × curve_slope
+ audio_pointer_weight × (音频选时 + 音频动作)
+ 0.2 × trailing_workload
+ placement_weight × placement
+ planner_weight × plan
```

当前共用音频权重0.15、placement权重0.15、planner权重0.2。WebUI已准确标为“音频辅助监督权重”；字段仍叫`audio_pointer_weight`以兼容旧配置，它同时控制两项音频损失。单独把动作辅助降到0.02的历史实验用的是scratch目标，不等同于把共用控件设成0.02。

动作CE按行处理：每个真实原子槽权重1，第一个STOP权重0.35，之后padding为0；先除以本行有效槽权重和，再按sample_weight、transition_weight和action_weight归一化。选时、EOS、负荷等各用自己的mask/权重。**slot0准确率、非STOP误差、STOP误差和总动作平均不是同一个指标。**

训练界面显示的是当轮/本次运行的累计及近期训练均值，受到随机遮挡和歌曲分布影响；eval关闭这些随机性。不能把训练累计均值、固定eval均值、辅助裸CE、加权贡献混用。旧版大量post-STOP padding计分也使旧loss不能直接横比。

各microbatch分别归一化，EOS的pos_weight还按本microbatch正负数计算。因此改变microbatch/累积，即使乘积和样本顺序相同，也不保证严格同梯度。真正想比较策略时保持64×4，或明确承认这是新的优化条件。

当前v12第1轮末→第2轮末→第3轮部分，总loss约4.9467→4.5291→4.3545，主动作约3.0074→2.7363→2.6222。第3轮4.3545约由主选时1.0241、主动作2.6222、两项音频辅助0.6260和其他0.0822组成。模型确实在学习，主配置误差仍大；没有证据把约一半解释为不可约底噪。

## 8. 恢复采样与生成差异

每100次优化器更新，恢复采样默认抽2个样本、真实滚动2步，允许2–4步。生成提案需经过约束并实际提交，再寻找仍合法、可达的原谱续接。原谱续接不是最优纠错oracle，有些样本会被拒绝。

**现行recovery只直接监督选时指针**，按sample_weight×pointer_weight归一化后乘0.1额外反传；动作、EOS、future等并没有在此分支获得同样的恢复目标。它不能直接等同于已解决动作配置的exposure bias。恢复loss单独记在`composer_recovery`，不直接计入常规总loss平均，分析全部梯度时要包括它。

常规训练依赖原谱前缀和同拍teacher forcing；自由生成依赖模型自己的已提交前缀，类型偏差、键位偏差及搜索失败可能逐步累积。只有被接受的事件进入history、持续状态和motif档案，失败提案或搜索游标不等于提交历史。

生成正温度下，默认保留模型大类和Touch环区概率，再在组内调温。新增“动作类别温度/min-p”独立作用于Tap/Hold/Slide/Touch四类，Touch环区内部相对比例保留；降低类别温度也可能强化Touch或Tap偏科。动作配置温度为0时逐槽贪心，仍可能生成多押。

“抑制Touch历史自强化”用音频分支限制历史对Touch赔率的额外放大，不是Touch配额；上限0仍允许音频证据生成Touch。“开放Touch布局”保留用户要求的多Touch能力，不把全部Touch占用按钮双头预算，但不保证任意远距离组合都可双手完成。Muri的remaining_actionable_hard=0也不等于所有hard=0或人工试玩通过。

生成页参数与训练目标要分别记录。旧condition JSON可能覆盖界面等级/Style；`TouchHold`同时进入部分Hold/Touch统计，也不能把这些统计强行加成互斥概率。生成只有`motif_strength`不能让off模型获得动机能力。

## 9. 已完成的验证，不要重复当成待修bug

| 证据 | 已验证结论与边界 |
| --- | --- |
| 缓存修复，`cdb7d0a` | 本机fused AdamW更新不增加参数_version，旧eval因子缓存可能跨更新过期，影响交替评估/恢复提案；已在模式切换和有grad查询清缓存。正常train原本不读eval缓存，不能拿它解释全部平台 |
| 性能修复，`dfa5620` | 一次完整训练forward共享有梯度factor table；声学查询复用gather与边界数组。pointer-only、STOP、输入合同保留 |
| 性能数值验证 | FP32/BF16真实空与非空motif输入，输出/loss相同，梯度max约2e-7，一步参数差max7.45e-9；无持久训练graph缓存 |
| 性能结果 | 8-chart预热受控207→282样本/s；同epoch同采样区间现场189→234，约快24%。仍有CPU供数瓶颈，不承诺所有窗口/80轮同倍提升 |
| 最新相关回归 | Python585项通过、1项默认跳过；前端相关19项烟测通过。不能把测试数字当音乐质量证明 |
| 128歌曲诊断 | fit96首768行；same-song384个不重叠事件；另32首384行heldout。父模型已经训练过这些歌曲，heldout只针对该短拟合阶段，不是父模型未见测试集 |
| 早晚v11检查点 | 同一阶段heldout probe主动作2.7706→2.2881，首槽精确率约23.5%→29.2%，确实有学习；主要误差仍在头区及细配置属性。这里不是父模型未见歌曲验证 |
| 分项诊断 | 一个384行probe总4.0313，其中主选时1.0429、主动作2.2881、加权音频两项0.6278、其他0.0725；辅助合计约17.4%，不是两项音频单独占17.4% |
| 历史信息消融 | 同probe主动作完整2.2881，全symbolic关闭3.0307，token历史关闭2.7650；不能说模型完全读不到上一拍 |
| 梯度与短拟合 | 24批共享动作头梯度均同向，音频encoder少数批次反向；不证明0.15永远最优，也不支持“必然整体拖拽” |
| 同目标拟合实验 | 原配置128更新可把小fit集总loss4.18→2.25，但其他歌probe4.03→4.39，说明能拟合且会过拟合；LR2e-5的阶段probe小幅改善，不足以替代fresh默认或宣布质量胜出 |

历史五组消融中的进程内32/64等中间eval受旧缓存影响，已经标记失效；只使用`ablation_rescored.json`的重新加载后终点评分。前轮selected短微调已更正为5.451832→4.905069，不继续引用旧after4.935296。

关键本机证据目录（未上传权重/音频）：

- `.scratch/composer_loss_deep_20261026/`：`diverse_batches_provenance.json`、`diverse_diagnostics.json`、`context_probe.json`、`ablation_rescored.json`、`paired_probe.json`、`selected_trial_rescored.json`。
- `.scratch/composer_v12_speed_20261026/`：`performance_summary.json`、`matched_production_window.json`、`factor_update_equivalence.json`、`matched_motif_batch.pt`及其provenance、`resumed_checkpoint_validation.json`、`astra_logic_review.md`。
- `.scratch/composer_review_20261026/generation_ab/`：固定音频双种子、源码快照、十组生成对照和哈希。部分组合Touch翻倍，没有被包装成整体质量改善。
- `.scratch/composer_learning_handoff_20261026/verified_state.json`：本文件运行快照的原始核实数据。

**scratch脚本不是都可安全import的库。**部分脚本顶层会解析参数或创建GPU张量；先读入口，独立进程执行，不能在生产占用GPU时批量运行旧诊断脚本。

## 10. 下一位应优先回答的问题

先回答“哪部分配置最难、是否能从现有输入识别、训练和生成是否都可表达”，再选择改进。下表都是研究方向，不是已经确认需要实施的bug修复。

| 方向 | 最小验证与可能实现 | 风险和准入条件 |
| --- | --- | --- |
| 固定评估与真实学习进展 | 固定父权重、跨歌曲/时间/类型样本，记录slot0/后续槽/STOP/类型/头区/路径/时值指标；用同输入比较多个保存步，不只看累计总loss | 当前无独立验证；当前所有父点见过全量数据。真正未见歌曲验证需新增歌曲或在下一次fresh前留出，不把阶段heldout伪装成全局泛化 |
| 训练标签与生成合法集一致性 | 在真实teacher前缀下统计目标原子是否被canonical顺序、同头/按钮预算或其他生成mask剔除；统计不可能类别获得的概率质量 | 先确认没有合法原谱目标被错误mask。若研究训练时合法性mask，需保留共同的未mask指标，不能靠缩小类别集合制造loss改善 |
| 词表长尾与语义等价 | 统计原子频次、频次桶CE与实际同义类占比；先评估语义等价类概率和，无需立刻改ID | 不凭举例假设大量别名存在。beat时值与绝对秒时值不能跨tempo语义随意合并。重编码或新loss须新合同/新目录 |
| 跨拍几何和结束位置 | 对相邻事件头位/相对位移、Slide等待/进行/结束后的转移分桶；比较当前history/state与新增显式几何辅助目标 | 当前relative_logits主要同拍使用。不要简单把所有Slide终点立刻当当前手位；手位不是起点列表的同义词。全旋转增广也未证明符合街机上下区域差异 |
| 层次配置监督 | 在保留原词表头下，研究显式类型/头区/时值辅助目标或相对几何目标，降低只靠稀有独立行学习的难度 | 当前结构因子不是已具备这些直接监督。新增任务会改变梯度与权重平衡；粗类别准确率提高不等于细配置或自由生成变好 |
| 动作端exposure bias | 先做无更新的短真实rollout：比较teacher前缀与模型提交前缀下的配置误差、目标合法性及覆盖；必要时扩展严格筛选的动作恢复监督 | 现行recovery只训pointer。原谱动作未必是生成前缀下的正确/最优续接，不能把随机扰动或非法续接当正确训练标签 |
| 动机覆盖和有效利用 | 分歌曲、前缀长度统计检索命中率；只在非空子集比较正常/关闭/打乱档案的动作CE，并量化残差/tanh激活 | 非空数值梯度存在仅证明可训练。普通history内的事件本来就不进旧档案，early-prefix零命中不自动是bug；不能靠提高强度修复无匹配 |
| sampler与优化节奏 | 测相邻更新梯度相关性、同歌集中/跨歌混合、LR阶段或调度，保持共同probe；先测单变量 | 改locality改变学习分布且可能恶化I/O。LR2e-5只有有限阶段证据。scheduler若加入必须保存/恢复状态并暴露主WebUI参数，不与表征/权重同时变更后归因 |

可以特别检查99维head timing的共享A组约定，但当前原token和因子仍有类型信息，尚未证明是恶性实现缺陷。若拆分到41种独立头位会改变输入维度和projection，必须定义新策略并迁移，不能作为“无语义变化优化”硬塞入当前续训。

用户已明确不优先扩展某些额外Simai写法；不要把`$`星形Tap、分段独立时值等解析器扩展当作本主题的默认入口。也不要为降低Touch把开放组合重新强制回双头，或恢复此前移除的错误物量配额logit修正。

## 11. 建议的实验顺序和判定标准

1. **刷新现状、固定基线。**先读本文件和最新运行API/检查点，确认训练是否仍在进行。仅阅读交接不应停止它；必要GPU实验先保留完整点、确认停止和目录锁释放，记录可能重跑的未保存尾段。
2. **建立统一配置能力账本。**从全量按歌曲、类型和时间分位抽样，冻结索引/条件/词表哈希；对当前稳定检查点做共同eval与配置分项分解。优先补标签可达性、长尾及语义等价统计。
3. **先用不更新权重的干预筛假设。**例如关闭token历史、关闭状态、motif正常/关闭/打乱；严格区分信息流负对照、分布外输入及真实缺陷，不凭一次小差异删模块。
4. **只选一个有证据的策略短训。**同父点、明确optimizer是否继承、相同样本/更新预算和seed；独立新目录finetune。若改结构/目标/输入合同，保持旧模型可生成，不静默resume。
5. **共同指标验收。**同时报告每1000更新、每百万样本、每GPU小时的配置收益，以及samples/s/显存/主存；不同loss权重或合法集的变体仍要按同一参考目标评分。
6. **自由生成A/B。**冻结检查点文件、音频、BPM/偏移、等级/条件及双seed；统计物量、真实头位/Touch环区、类型、同键/八分链、残留hard、回退和EOS，再试听/试玩。不要以更少Slide带来的冲突下降、或更多Touch带来的物量上升单独判胜。

现有固定音频为`webui_workspace/uploads/b07631395b89_track.mp3`，旧条件为`webui_workspace/generated/0fc5be834826/condition.json`（190BPM、first0、level14.5）；历史双seed为1922635584和1696576229。真正有检索机会的另一首音频为`E:\全量官谱\BUDDiES\11608\track.mp3`，时间设置应从其原谱/fixture provenance重新读取，不套用190BPM。

已有生成基线可先保持候选拍位温度0.5、选时历史0.65、动作配置温度0.65、动作历史1、类别1/min-p0、Touch保护开启且上限1、open布局、同键70ms、curve1、motif1。需要测试生成控件时另立单因素实验，不与训练改动同时归因；新架构本身不保证此组合在所有模型/音乐上最好。

**停止扩展或回退的条件：**共同probe无改善、收益只来自改loss显示口径、稀有Slide/Hold/Touch合法覆盖下降、自由生成明显退化、CPU/显存成本抵消单位时间学习收益、出现非有限或未来谱面泄漏。没有实际质量证据时标为“待音乐验收”，不要写“已根治配置学习”。

## 12. 交付物与接手任务书

后续交付至少包含：问题的最小复现/反证；训练目标和输入改动说明；版本/迁移方案；固定probe与自由生成结果；资源/速度；未解决点；原检查点保留位置。新增可调训练或生成参数要接入**主界面**的控件、保存/重置、检查点继承、API/CLI和实际参数回显。

基础操作：

```powershell
git status --short
git log -5 --oneline
python -m maichart.composer.train --help
python -m maichart.composer.generate --help
python -m pytest -q tests/test_composer_factor_reuse.py tests/test_composer_context_integration.py tests/test_composer_recovery.py tests/test_composer_factor_cache_updates.py
```

使用实际训练环境执行。不要提交本机音频、权重、cache、日志或密钥；不要改动其他人留下的`.workbuddy/`和未跟踪审查文档来凑干净工作区。当前用户倾向用Luna节省子代理用量；此前Astra只获得本会话临时逻辑审查授权，不应据此视为后续会话的长期模型使用授权。

可直接交给下一位智能体的任务描述：

> 在`E:\simai_transformer`工作，先阅读本交接并实时核实v12训练/检查点，不依据旧文档直接fresh或重启任务。目标是提高配置能力的样本、更新与GPU时间效率，尤其跨拍走位、Slide/Hold选择、Touch布局和段落呼应。
>
> 先理解当前数据、teacher forcing、独立词表头+结构因子、planner梯度边界、仅pointer的恢复监督和训练/自由生成差异。现有性能修复已验收，不要通过退回旧padding计分、降低辅助权重或缩小合法集制造loss变好。
>
> 优先建立共同配置probe和标签可达性/长尾统计，筛出最有证据的一个策略；用新目录、可复现短程对照和固定音频双seed验证，再决定是否值得长程训练。明确事实、假设、实现限制和未经过试听/试玩的判断。保留原检查点与已有效果，并更新交接。

进一步证据入口：[loss深审](COMPOSER_LOSS_DEEP_AUDIT_20261026_ZH.md)、[v12性能与恢复](COMPOSER_V12_GPU_PROFILE_20261026_ZH.md)、[生成参数与因果策略](COMPOSER_REVIEW_AND_PARAMETERS_20261026_ZH.md)、[项目总交接](COMPOSER_PROJECT_HANDOFF_20261024_ZH.md)。历史记录与当前源码冲突时，以实际源码、配置、任务和可重复证据为准。
