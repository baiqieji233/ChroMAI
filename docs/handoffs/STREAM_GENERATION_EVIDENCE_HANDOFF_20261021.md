# 连续流候选证据、结束决策与重复退化修复交接

日期：2026-10-21。基线：`797e2f6`。用户授权修改并检查训练效率。

备份：`E:\simai_transformer_backups\v16_generation_before_20261021_1132`，包含 src/scripts/tests/docs/configs、pyproject、HEAD 和状态；用户权重及原产出不覆盖。

## 工作边界与阶段

- [x] 重新确认源码和 v16 实际产出、训练停止状态并备份。
- [x] 分离 EOS/拍位温度，记录实际决策概率。
- [x] 接入原谱监督的下一事件音频指针，保持任意 Fraction 与旧模型兼容。
- [x] 增加结合真实秒数、拍数及学得负荷的有界重复软引导。
- [x] 按真实时间积分段落曲线，去除非均匀节点的采样数量偏差。
- [x] 同步 CLI / WebUI / 报告；阶段完成后集中验证。
- [x] 测量训练输入/梯度一致性和当前设备真实训练吞吐；结果已补齐，提交号见Git。

不提高循环退出损失、不修改双手特征、不切换旧权重的 motor_profile、不缩短短语窗口，不恢复并不存在的 Slide 冷却规则。不上响度阈值，不把强瞬态当作强制谱面落点。训练目标和训练缓存合同保持不变。

## 已确认的问题与本轮取舍

v16 的 `stream_last.pt` 已完成80轮/35360次更新，实际训练语义是 v13，`motor_profile` 缺省为 legacy，镜像关闭。当前源码里的 v14 双手能力不会自动进入旧权重。参考产出 `49cd37076695` 有72个连续单键4，约145.263–156.474秒；最后头157.816秒、最后动作158.132秒，158秒之后并没有新增头。仅把文件末尾约3秒底噪裁掉不能解释或解决这些早于它的落点。

第三方报告把运动学16维说成分母计数，把双手13维说成same_button，并引用不存在的kinematic_context.py及Slide冷却。实际16维是age，新双手13维是已知状态，连续长度是7/15维。没有按这些错误下标做遮罩或限幅。旧循环退出梯度25%是明确的有界设计，未当作归一化bug改掉。

## 具体实现

### 独立结束决策

CLI `--eos-temperature`、WebUI **结束决策温度** 默认1；仅用于事件边界EOS/AT。**拍位候选温度**仍是0.15，配置温度也独立。报告 `event_decisions_first_64/last_32` 增加实际温度和 `eos_probability`，这是采样概率，不是经校准的音乐结束置信度。设0使用贪心，设成拍位温度可对照旧策略；没有强制EOS阈值。

### 原谱监督的拍位证据

- `GroundedContext.alignment_window` 抽出训练原有的下一事件指针。训练仍用原来的BOS/END隐藏状态、128节点相对时间窗口和同一loss；参数形状、执行公式、标签和训练缓存身份不改。
- 生成在选择AT后立即使用**之前BOS/END输出**建立这一个事件的指针证据，不用当前AT或候选NOTE的隐藏状态冒充训练边界。不推进因果时钟，不提交候选，不修改K/V。
- 静态音频alignment keys和selection结果全曲只计算一次；每个事件仅计算128节点指针，合并一次读回CPU。完整Fraction候选在其真实秒数上插值证据，不吸附到音频节点。
- 生效要求checkpoint明确记录alignment监督且存在参数更新；未训练/旧式无证据模型自动关闭并在报告解释。v16有这项监督，能够使用。
- 采用指针分布集中度及selection的相对变化衰减强度；不把class-weighted BCE的sigmoid当已校准概率。扁平、无对比证据趋近0，窗口外候选不加该项，不强迫每个瞬态出现note。
- **谱师采音证据强度**默认0.25，最大单候选代价为其2倍。它参与原有完整拍位评分，不额外执行一遍解码器。音频指针没有null/rest类别，窗口外不打分仍是限制；因此保持小权重，不能代替休止学习或声称能消灭全部长空白。

### 有界长重复负荷引导

- `RepetitionGuard`只在已提交前缀寻找1–8事件周期，分别比较完整配置和有类型/头数差别的结构周期；不会把恒定Tap类型视为所有均匀节奏都违规。
- 保护起点必须同时超过 **长重复最短秒数**（默认4）和 **长重复最短拍数**（默认8），并持续同一拍距周期。没有“8颗以后禁止”的规则，短交互例如190BPM的12颗1,8保持不动。
- 把实际**音符头/秒**与学得曲线解码的头/秒比较，多押按多个头计数。仅明显超过预测（比值超过1.35）才渐进介入；预测支持的长重复不触发。同一事件的最终惩罚上限为 **长重复负荷引导强度** 的2倍，默认强度0.6，可设0关闭。系数1.35、最大周期8和1024条记录是当前有界实现常数，不是可玩性定律。
- 同时在提案前缀和完整配置终端加入同一偏好；原有重要性校正补偿提案偏置，不把模型概率或惩罚重复计算。仅重复压力实际存在时才为默认Style=0路径打开多候选，避免所有普通事件都凭空增加4倍提案。
- 拒绝/回退只影响当前候选；重复历史在事件最终提交时追加。不会用惩罚强迫新键、Slide或强迫每段都创新。依然可能漏掉低置信度的重复，也可能对刻意写得高于预测负荷的定式产生软偏好；这不是音乐质量奖励或无理证明。

### 段落曲线的真实时间积分

`SectionTypeProfile`原来对非均匀节点按数量平均；现在对分段线性曲线在秒轴积分。全曲类型参考采用密度和类型份额的乘积积分；局部窗口按真实覆盖秒数平均，重复时钟合并，非有限值拒绝。负荷引导将密度先按原合同解码为头/秒，再对该速率曲线积分；不会将头密度误当事件密度。

这是生成侧侧重比例的修正，解码器训练/生成共享的 `PHRASE_WINDOWS` 保持原含义，旧权重没有悄悄从12秒窗口换到4秒窗口。

## 训练效率和等价性

完成全部实现后集中执行242项Python回归、12项Node界面检查，全部通过；新增冻结版alignment参考，逐参数梯度和全部输出零容差比较通过。覆盖新设置默认/0值、边界指针时钟不变及静态keys复用、非均匀曲线积分、短交互保护、重复前缀/终端一致性和旧生成/训练合同。

真实流水线结果：`E:\simai_transformer_backups\v16_generation_before_20261021_1132\training_pipeline_comparison.json`。脚本新增 `--baseline-scope alignment`，两侧都使用本轮修改前已有的native双手后端，只切换本轮alignment抽取前后实现。RTX5070Ti Laptop 12GiB，384维/6层，batch16/累积2，BF16、cuDNN优先、预取2、数据线程4、原生后端；抽样32张真实谱面，每侧各16批次冷/热测，不保存权重。

| 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 热态样本/秒 | 28.95 | 29.50 |
| 热态批次/秒 | 1.810 | 1.844 |
| 固定输入计算中位数/批次 | 0.2695s | 0.2690s |
| 峰值已分配显存 | 5.984GiB | 5.943GiB |

约2%差异属于短测波动范围，结论是**未见训练退化**，不是宣称这轮又加速2%。真实批次输入完全一致、无缓存重建。冷态存在操作系统页缓存/JIT等偏差，不拿它作为本轮训练加速结论。抽样时GPU利用率曾达97%，不承诺全轮都满载；未启动正式训练。

复现命令：

```powershell
python -B scripts/benchmark_stream_v14_speed.py --manifest 'E:\全量官谱\training_manifest.json' --cache-dir 'E:\全量官谱\cache' --snapshot E:/simai_transformer_backups/v16_generation_before_20261021_1132 --output E:/simai_transformer_backups/v16_generation_before_20261021_1132/training_pipeline_comparison.json --batch-size 16 --accumulation 2 --limit-charts 32 --batches 16 --baseline-scope alignment
```

## 生成实证与交付状态

完整v16生成验证完成，复用原任务的音频/BPM/first/seed/配置温度/Style及物理保护，使用新的结束温度、采音证据和重复引导默认值。结果保存在备份目录下 `validation/maidata.txt`、`maidata.stream.json`、`generation.log`，未覆盖原产出；运行约182.24秒。合并多项修改的一次生成只能证明链路和个案，不能把所有差异都归功于单项机制，也不据此声称平均成品质量已经提高。

| 个案指标 | 原49cd37076695 | 本轮验证 |
| --- | ---: | ---: |
| 事件数／音符头数 | 1048／1280 | 884／1103 |
| 完全相同配置最长连发 | 72个Tap4 | 4个相同Hold |
| 102–110s四分之一拍间隔数量 | 92 | 37 |
| 同区间半拍间隔数量 | 1 | 21 |
| 最后note头 | 157.816s | 156.632s |
| 最后持续动作终点 | 158.132s | 158.211s |
| 最长内部空白 | 1.263s | 2.526s |
| 最大导出分音 | 16 | 32 |
| MiaCode运行时硬冲突 | 0 | 0 |
| MiaCode静态参考提示 | 5 | 9 |

指针有效启用890次（含重选），记录的候选代价非零；重复引导在89次事件评估介入，实际峰值惩罚约0.412，远低于1.2上限。最终EOS采样概率0.887，不靠阈值截断；原任务的72颗同键未复现，密集段有更多半拍间隔。这些是实际输出差异，不是对新机制的单项因果证明。

仍需保留的质量边界：最后头虽提前约1.18秒，最后Hold仍延续到158.21秒，**音乐结束后的持续动作选择未被本轮彻底解决**。新产出含15个`[1:0]`时长块；核对`notes.py`确认项目有意将无时长Hold标准化为它，解析为0时长，不是分母错误或语法漏检，因此没有擅自禁用。短Hold的音乐选择、静态参考提示从5增到9，都说明不能只拿“重复减少／无运行时硬冲突”当作质量验收。后续应听音及试玩，比较普通曲/密集曲/长尾音曲多个固定种子；本轮没有继续多轮刷生成。

WebUI合同升级为 `chart_stream_quality_controls_v6`，前端新控件、保存/恢复/重置、API/CLI传递和输出报告已同步。完成后核对7860/7861均无活动任务，已重启这两个WebUI服务，并从各自HTTP端点确认：新生成合同v6、训练加速合同native_hand_pipeline_v3、五个新控件和新版前端均实际加载，活动任务为0。用户只需刷新页面，没有自动重启训练。

服务重启只操作经命令行验证的127.0.0.1 Python监听进程；7860还有Tailscale监听，未停止该系统服务。服务stdout/stderr在备份目录中。

集中回归命令：

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_stream_generation_evidence.py tests/test_stream_geometry_and_sections.py tests/test_stream_grounding.py tests/test_stream_causal_motor_v14.py tests/test_stream_configuration_search.py tests/test_stream_quality_constraints.py tests/test_stream_pipeline.py tests/test_stream_residency.py tests/test_stream_training_contract.py
node --test tests/test_stream_generation_ui.js
```

此外通过轻量API参数构造检查，验证新选项的显式0和最短秒数6／拍数12都实际转成对应CLI参数；该检查没有发消息或创建真实任务。

## 文件索引

- `stream/decoding.py`、`generate.py`：独立EOS决策、新选项及有效性报告、候选接线。
- `stream/grounded_model.py`、`grounded_inference.py`、`position_evidence.py`：共享监督指针及静态缓存。
- `stream/repetition.py`：时间/拍位/负荷联合的有界重复反馈。
- `stream/section_profile.py`：真实时间积分。
- `webui/server.py`、`stream_api.py`、`static/app.js`、`static/index.html`：运行能力合同及控件。
- `tests/test_stream_generation_evidence.py`、`test_stream_generation_ui.js`：新增回归和界面合同。
- `scripts/benchmark_stream_v14_speed.py`：训练等价性短流水线比较。

未使用子智能体，`.workbuddy/`不上传。原始v16权重未改写，v14训练语义及数据指纹维持不变；本轮不要求因这些生成设置重新fresh。
