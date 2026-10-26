# 配置能力审查复核：证据边界、诊断修复与实测

> **后续根因已进一步定位：** 本文末尾“根因尚未确认”是停止训练时的阶段结论。后续经过梯度通路干预、模块替换与6组GPU短训，定位到历史memory的尺度失控及跨注意力反馈，见[根因调查与因果验证](COMPOSER_INSTABILITY_ROOT_CAUSE_20261026_ZH.md)。正式训练仍停止，结构修复目前只在隔离实验中验证。

源码起点：`7b9fd7c`。本轮审查用户提交的“配置能力量化账本”报告及其`.scratch/composer_config_ledger_20261026/`脚本，并用当前源码独立核实。**代码修复范围为配置评估与诊断链路；没有改训练目标、模型结构或采样默认值。** 诊断期间发现v12第5轮的真实权重退化，已额外保留完整点并保护性停止长程任务，尚未重启。

## 1. 报告哪些部分有价值

按动作类型、首槽/后续槽拆解误差，比只看累计总loss更有诊断价值。类型边际、给定类型后的键区/路径误差、真实motif命中样本、placement判别性都值得继续研究，但以下结论需要收窄。

| 报告结论 | 复核结果及处理 |
| --- | --- |
| 64.3%的生成Slide实际落在Tap位置 | 原脚本只在原谱候选、真实历史和首槽softmax上按类型概率估算，没有自由生成。它是该探针分布上的期望分配，不是实际生成落点，也不是已证明的音乐错误率 |
| 上一拍直接投影只有1.5%，被LayerNorm彻底淹没 | 分量RMS不能解释为可加的贡献百分比，也不是因果干预。上一拍信息还存在于history encoder、head timing和持续状态中，不是只有这一条通路 |
| placement无正类权重导致类型常数 | 无`pos_weight`是事实；标签偏移、mask反转或排除区泄漏未复现。候选分支和16个全曲探针已有分别归一的监督权重，原始“空点多”不能直接表示有效梯度占比 |
| motif已普遍改善配置；tanh必须放开 | 非空小批次上的正向敏感性值得保留，不能外推全曲覆盖或音乐质量。p95接近1只说明部分坐标接近边界；平均导数约0.447也不是全分支梯度消失证据。本轮保留边界 |
| 同义写法和合法集失配已完全排除 | 原检查可作局部负证据；1,536行不是全量穷举，也未覆盖自由滚动前缀、所有guard和回退路径，不升级为全局排除 |
| 两项新结构是“单因素”改进 | 显式层次监督和新增跨拍几何头是两个独立改变，必须拆开对照，不能同时改变后归因 |

原conditional脚本直接按行求均值，而fast ledger部分指标使用`sample_weight × transition_weight × action_weight`。384行的权重和404不是404个独立事件。原ledger把每批已归一loss乘batch行数再平均，也不等于跨批按真实分母归一。

原脚本多次读取正在更新的`composer_last.pt`，却未为所有输出保存一致的权重快照/哈希。报告中的14,000步不是本轮16,000步，以下数值不能直接作前后模型质量结论。

## 2. 已复现并修复的实现问题

### 正式诊断入口漏掉当前配置输入

`src/maichart/composer/diagnose.py`原先存在：

- `_inputs`不传motif三元组或selected state，诊断会漏测已训练的动机分支。
- `selected_prefix_v3`的decoder缺少选定拍位状态和该次forward的`selected_state_base/state_keep`，真实小模型复现报错：`selected_prefix_v3 requires a causal state at the selected beat`。
- 诊断dataset未继承检查点的状态策略、动机策略和事件上限。
- 删除最后一个前缀事件时，逐候选状态未按新网格重建，原目标拍位的动作上下文也不能沿用。

现在先固定基线指针选择，再让各对照在**同一个候选拍位、同一个已提交前缀**重建动作状态和motif，保持未来谱面不可见。motif原子ID映射到检查点词表；遇到未知非零ID显式拒绝，避免静默变为STOP。

符号历史关闭的变体同步向motif传`motif_keep=0`。非零动机权重测试验证：正常解码对档案变化敏感，而关闭符号历史后，改变档案的动作logits逐位相同。省略前缀路径仍不把原谱续接当成唯一纠错标签。

诊断报表标记为`composer_interventions_v2_causal_configuration`，附实际模型配置。旧v1报告涉及动机或selected state时不能当作修复后的等价对照。

### 固定配置评估的加权和追溯问题

新增[配置指标累加器](../src/maichart/composer/capability_metrics.py)与[只读CPU审查脚本](../scripts/audit_composer_capabilities.py)：

- 按真实分母合并批次，分别记录行数、权重和、首槽、后续非STOP及首个STOP；末尾padding不计分。
- 动作loss保持每行真实槽权重1、首STOP权重0.35的归一口径；类别、精确原子CE及类型概率矩阵分别报告，不改变训练loss。
- 类型来自parser支持的因子表，TouchHold并入Touch，复合类别单列；不用字符串猜类型。空组指标为`null`，非法ID、非有限值和STOP后动作显式拒绝。
- 每次先复制固定检查点，记录文件/词表/fixture哈希及输入策略，严格加载权重，拒绝错序词表或不兼容状态输入。
- 分支干预使用临时forward hook，结束或异常时移除；不修改参数、不创建优化器，默认单CPU线程，全程不用CUDA。

该脚本衡量teacher-forced配置能力，不包含生成温度、min-p、Touch保护、guard和自由滚动；不会把softmax混淆质量写成已生成事件的错误率。

## 3. 当前检查点的独立实测

固定父点：v12第5轮、**16,000更新**，SHA-256：

`8869edf8d4af0378b44c2127b538d93a871f814e6f0e13585a82a02e63fb8791`

完整本机快照：`.scratch/composer_capability_review_20261026/snapshots/composer_lta5r75q.pt`。所有以下变体都使用该点，FP32、CPU线程1、无参数更新。当前正式任务的检查点继续前进，不能用同一路径代替这个固定点。

### 384行阶段heldout，动作权重和404

| 变体 | 全槽动作loss | 首槽精确CE | 首槽精确率 | 首槽类型CE |
| --- | ---: | ---: | ---: | ---: |
| 正常 | 2.365868 | 3.149176 | 26.73% | 0.596860 |
| 关闭`action_relation`直接投影 | 2.364235 | 3.147438 | 26.98% | 0.595833 |
| 关闭placement投影 | 2.366056 | 3.149318 | 26.73% | 0.596825 |

这个probe来自motif关闭时准备的fixture，本轮以显式`--allow-missing-motif`执行**无motif输入**诊断，不能代表v12完整动机路径。父模型此前见过这些歌曲；它也不是全局未见歌曲测试集。直接投影与placement在此probe上的即时敏感性较小，不能由此证明删除它们更好，也不能证明新增通路一定没用。将一个分量置零也属于分布外干预。

### 真实非空motif小批次，8行、权重和9

| 变体 | 全槽动作loss | 首槽精确CE | 首槽加权精确率 |
| --- | ---: | ---: | ---: |
| 正常档案 | 2.362543 | 3.133046 | 22.22% |
| 动机强度0 | 2.843115 | 3.816404 | 11.11% |
| 档案按行错位，目标不变 | 3.032907 | 4.216265 | 0% |

这支持“该点在这8行上已能使用档案，并且对应档案优于关闭/错位”；只有一首歌和一次确定性置换，不能写成全量泛化收益。它也是保留当前动机边界、避免贸然删除或放大该分支的理由。

本机原始证据位于`.scratch/composer_capability_review_20261026/`：`heldout_interventions.json`、`motif_interventions.json`、`before_reproduction.json`和`diagnose_real_song.json`。两个账本文件包含fixture哈希、权重哈希及分组加权指标。旧diverse fixture的provenance未绑定文件哈希，报告标为`fixture_hash_verified=false`，仅记录本次观察到的输入哈希；motif fixture的原有哈希已核对通过。不能倒填一个哈希后声称证明了历史fixture来源。

### 新发现：第5轮中段出现真实退化，已停止长程训练

实时日志显示第5轮batch5000→8000累计loss约4.2855→4.6876，指针准确率65.12%→61.20%。进一步对同一384行、同一输入哈希、同一加权口径做晚点复测：

| 更新点 | 全槽动作loss | 首槽精确CE | 首槽加权精确率 | 首槽类型CE |
| --- | ---: | ---: | ---: | ---: |
| 16,000 | 2.365868 | 3.149176 | 26.73% | 0.596860 |
| 16,500 | 3.153143 | 4.086246 | 11.39% | 0.818876 |

这不能只用歌曲顺序或累计显示解释：固定输入的配置预测确实退化。仍不能据此证明报告中的“BCE常数”或“缺少跨拍头”是原因。没有在本轮给正式模型应用新辅助loss或新网络后再看到退化。

任务`78f828eaac2a`已通过主工作台取消，API确认`cancelled`、`cancel_error=null`，原训练进程已退出。最后日志17,032更新，最后磁盘完整点17,000；未保存尾段没有声称被保留。另存：

- `checkpoints/Prism_composer_v12_preserved_20261026_step16000/composer_last.pt`：较早对照点，哈希为上文`8869edf8…8791`。
- `checkpoints/Prism_composer_v12_preserved_20261026_instability/composer_last.pt`：第5轮17,000完整点及优化器，SHA-256为`d5497843df4b8de06b75b72b43f6c930bb1e20ee29abe750d6967f0a58a00293`。
- 16,500对照点仍在`.scratch/composer_capability_review_20261026/late_snapshot/composer_n60pd9fj.pt`，完整结果为`late_heldout.json`。

16,000与17,000两份完整点都检查了模型权重、Adam状态非有限值及负二阶矩，均未发现；LR均为0.0002，`can_resume/can_generate`均为true。恢复提案的模式切换/清缓存/反传及累积窗口路径暂未查到确定bug，训练日志也没有非有限警告。具体优化退化机制仍待受控实验，不能把“文件可恢复”写成“适合继续训练”。验证记录为`preserved_validation.json`。

## 4. 为什么没有直接套用两项训练改造

原子的负对数似然本身可分解为：

```text
原子CE = 类型CE + 给定类型的原子CE
```

额外添加类型边际监督，相当于增加某部分误差的权重；它不是修复一个原本缺失的梯度，也不保证“无损”改进。当前结构因子已通过全词表CE学习共享属性。新辅助目标仍可研究，但应独立版本化、用共同原子CE及自由生成验收，不能只看新加权loss。

新增跨拍圆周头也是表征实验。零初始化能在正确迁移时保留初始函数，不代表训练后必然改善；还需处理历史/符号遮挡、训练与生成一致性、开放Touch、长条占手、Slide结束位置及检查点迁移。本轮没有足够证据将它作为已确认bug合入。

对placement使用正类加权会改变概率最优点；它进入主决策的是概率特征，不能仅为让预测数值变高就改权重。应先补逐通道加权标签率、判别指标和更广泛干预，再单因素比较。

## 5. 验证、使用与下一步

本轮已由主代理核实：

- 修改前源码最小复现selected-state报错，并确认motif输入键全部缺失。
- CPU相关Composer回归：**523通过、1跳过**；最终输入保护等改动另经22项定向回归通过。跳过的是CUDA专项，本轮没有争用正式训练GPU。
- 新指标测试覆盖真实不等权重、改变batch划分、零权重、STOP/padding、18,454类数值稳定及异常恢复；临时hook不改参数，snapshot不覆盖旧基线。
- 真实v12检查点完成384行三组干预与8行motif三组对照。
- 真实`BUDDiES/11608`原谱/音频、一个难度、三个前缀位置的完整诊断CLI通过，包含省略末事件路径；这不是音乐质量验收。
- 原v12第10,500步和v11第17,500步保留点的完整SHA-256与此前记录一致；没有覆盖原检查点。

复查固定fixture的例子（先在本机确认文件存在）：

```powershell
python scripts/audit_composer_capabilities.py --checkpoint checkpoints/prism_composer_v12/composer_last.pt --snapshot-dir .scratch/my_capability_audit/snapshots --batch-file .scratch/composer_loss_deep_20261026/diverse_batches.pt --provenance .scratch/composer_loss_deep_20261026/diverse_batches_provenance.json --split heldout --output .scratch/my_capability_audit/heldout.json --threads 1 --allow-missing-motif
```

脚本会产生新的固定权重副本；跨实验比较应把同一个副本作为`--checkpoint`，不要反复读正在更新的last路径。旧fixture只在词表与输入策略相容时可用；上例显式允许缺失motif，结果必须按此限定。本机音频、fixture和权重未随Git上传。本轮CPU诊断并行期间不作正式任务吞吐对照，避免把资源竞争误归因于训练实现。

**诊断修复本身不要求调整训练或生成参数；但正式训练已因新确认的退化停止。**停止前为学习率lr0.0002、batch size64、梯度累积4、BF16、CPU2/样本准备1/预取2。不要从17,000点按原设置直接恢复80轮长程任务，也不要把累计loss回升简单当作显示口径。

下一步优先从16,000完整点保留相同optimizer与样本/更新预算，分别比较原学习率与较小学习率的短程稳定性，再决定恢复配置；较小LR尚非本轮已经验收的方案。同时扩大按歌曲/类型分层的固定probe和真实生成前缀对照。若要研究类型辅助监督或跨拍头，分别在新目录显式finetune；不要与优化稳定性实验同时变更后归因。尚未解决的是退化机制、配置泛化与实际音乐质量，而不是已经证明存在某个统一loss计算错误。

项目与训练全貌见[配置学习交接](COMPOSER_CONFIGURATION_LEARNING_HANDOFF_20261026_ZH.md)；此前证据见[loss深审](COMPOSER_LOSS_DEEP_AUDIT_20261026_ZH.md)。
