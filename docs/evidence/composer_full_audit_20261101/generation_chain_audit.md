# 生成、搜索、物理门禁与导出链路只读审计

范围：`0ff1f5a`；检查 `stream_generate.py`、`stream_search.py`、`stream_sampling.py`、`stream_proposal.py`、`stream_quality.py`、`guard.py`、`stream_data.py`、`event_compile.py`、`curve.py` 与生成评测。未修改源码、未用 GPU、未启动生成或重建缓存。以下“影响”若没有真实生成 A/B，均限于机制风险。

## 需要修复的确定边界缺陷

### G1：允许的短写入段可在末尾 EOS 后反复搜索同一段

- 类型/优先级：**bug / P2**；仅在非默认 `phrase_write_beats < 2` 且进入离音频末尾不超过两拍、当前目标段尚未到音频末尾时触发。默认写入 8 拍不触发这个构造。
- 直接证据：`stream_search.py:195-202` 只要求 `0 < commit <= write`，没有两拍下界。`stream_sampling.py:53-60` 在剩余不超过两拍时允许 EOS；`stream_generate.py:287-290` 在短段搜索中把 EOS 的 `cursor` 直接设成全曲 `end`，却不追加 StreamStep。`stream_search.py:246-258` 因 `cursor >= target` 将这个候选视为完整；`stream_search.py:325-328` 再按 `commit_end` 截取已有步骤；`stream_search.py:344-348` 只有 `target == end` 才退出。如果截取结果没有跨过 `start`，`while` 条件 `stream_search.py:210-213` 仍为真，反复重试同一位置，直到外部中断。单线程 CPU 探针确认剩余一拍时 EOS 为有限合法 logit、WAIT 被屏蔽（探针输出 `eos_eligible True wait_eligible False`）。
- 影响：自定义短段配置可造成无界生成耗时、重复候选开销；尚未观测到真实运行触发。
- 建议/代价：搜索候选必须证明真实采样的步骤跨过当前目标或明确只允许全曲末段 EOS 完成；提交后检查 `committed[-1].end > start`。补一条隔离搜索的回归用例。低改动成本。

### G2：候选段终点的 NOTE 可被提交，但不参与该段主要评分

- 类型/优先级：**bug / P2**；在 NOTE 恰落 `target` 且 `commit_end == target`（允许的配置），或 `_cut` 为保证推进而包含超出提交预算的一个真实步骤时触发。
- 直接证据：`stream_quality.py:290-304` 只解析 `start <= beat < end` 的 `current_points`，故终点 NOTE 不计入该段的覆盖、密度、事件数等；`stream_search.py:231-233` 的候选统计亦用半开区间。`stream_search.py:56-64` 的 `_cut` 却按 `step.end <= commit_end` 提交终点 NOTE，且若无任何新步骤进入预算会提交第一步（可超预算）。CPU 探针以 `commit_end=8.5`、新 NOTE 位于 beat 9，确认 `_cut` 返回 1 个新步骤。
- 影响：在多候选排序时，可能选出具有未计入主要评分的端点事件；默认 `write=8, commit=4` 时端点一般还在私有尾部，影响较窄。轨迹 softcomfort 会看 `actual_points`，因此不能说评分完全没看到该 NOTE。
- 建议/代价：统一候选评价区间与实际提交区间，并把超预算首步明确纳入评分及记录；或者禁止 `commit_end == target` 时提交未经同一区间评分的终点 NOTE。中等成本，需照顾交界处重复计数。

## 已确认的设计限制与质量边界

### G3：默认单候选模式的质量分数不产生选择压力

- 类型：**设计限制，不能当作评分提升证据**。
- 直接证据：`stream_generate.py:39-47` 默认 `phrase_candidates=1`；`stream_search.py:259-265,325,341-342` 只有一个有效候选时直接采用且标注 `diagnostic_only_single_candidate`。既有 2026-10-31 四份 `deep_audit_20261031/generations/ab_v{3,4}_.../maidata.composer.json` 均为候选数 1，101–104 个 phrase 中多有效候选数均为 0；这是历史产物，不证明 `0ff1f5a` 新生成的质量。
- 影响：分数不是对默认产物的优化信号。不能仅凭 `score.total` 宣称整句衔接、复读或可玩性改善。
- 建议/代价：若要评估排序能力，固定音频/检查点/种子做候选 1 对 2/4 的完整 A/B，记录合法率、硬冲突、段边界、复读指标并听音/试玩；2/4 候选近似倍增 rollout/门禁成本。

### G4：强制合法过滤会改变训练分布，且不可把“硬冲突为零”解读成可玩

- 类型：**明确的训练推理差异/设计权衡**。
- 直接证据：`stream_data.py:87-135` 训练步来自原谱 NOTE/WAIT/EOS；生成时 `stream_sampling.py:63-73` 屏蔽越界、零间隔等时间，`stream_proposal.py:33-69` 最多尝试三个不同时间，每个时间 `stream_sampling.py:102-195` 最多八个完整事件；`guard.py:176-235` 拦截重复头、越音频结尾、解析错误和新引入的受执行硬冲突；`stream_generate.py:345-361` 接受失败会改时点/进入搜索失败，而不是按原模型分布采样。`stream_generate.py:70-91` 已保留首提案、拒绝原因和换时间/配置的观察，不应重复报“没有诊断”。
- 影响：在高拒绝率区间，输出可能系统性偏向可解码、易合法的配置或留下空拍；是否形成听感复读/错拍需要真实 trace 与固定 A/B 验证。
- 建议/代价：用现有 `sampling_observation` 按曲段和配置拆出首提案到接受提案的转移、失败率及强制空拍，再与音频和人工标签对照。只读分析成本低。

### G5：物理硬门禁覆盖有限，报告中的软/仅报告冲突有实际意义

- 类型：**设计限制，不是当前硬门禁漏跑**。
- 直接证据：`guard.py:20-32,125-141` 仅将受执行的 MiaCode runtime 硬冲突、精确占用的静态 Overlap（及相应配置）纳入拦截；`SlideHeadTap` 始终 report，`runtime_hard_v2` 对静态 `TapOnSlide` 仍 report。`stream_generate.py:408-412` 默认 phrase 模式导出前全谱复判并拒绝可执行硬冲突；`stream_generate.py:94-114` 报告区分 `engine_action` 与最终 `action`、`remaining_actionable_hard` 与 `remaining_report_only_hard`。既有四份 2026-10-31 产物可执行硬冲突皆 0，仅报告硬冲突分别 0/7/2/1，不能称作全无物理问题。
- 影响：触屏几何、手部动作与实际玩家可玩性没有被硬门禁证明。把所有静态相交都强制拒绝也可能误杀合法触屏，因此需要逐类验证。
- 建议/代价：报告每一类冲突的例子与玩家反馈，先分离真实恶性问题和保守静态告警；人工试玩成本高于代码检查。

### G6：启发式评分不能单独证明“整句衔接/复读/节拍”

- 类型：**已显式标注的设计限制**。
- 直接证据：`stream_quality.py:1-5,285-390` 分数由声学覆盖、精度、段内形状、密度、弱参考和软轨迹组成，并标注 `heuristic_uncalibrated=True`；参考只在前一段声学相似度大于 .65 时启用，类型分布也只占总分很小权重。`stream_quality.py:136-210` 两手轨迹使用 50ms 取样与 beam32 软近似，可在 Slide 内换手，未知原生路径向 0.5 回退；不是真人动作判定。`stream_generate.py:175-211` 的连续性指标是描述性计数，不是坏谱判定；`stream_generate.py:455` 亦明确要求听音与试玩。
- 影响：错拍、复读、句间衔接可能未被候选分数惩罚，特别是单候选模式。仅凭更高分或更低硬冲突数推断听感改善属于证据越界。
- 建议/代价：固定音频/种子 A/B，单独看节奏相位、重复块、句边界和轨迹，再盲听与试玩；属于实证成本，不应改成任意硬阈值。

### G7：尾部时钟和导出契约有明确边界，未见当前确定错拍缺陷

- 类型：**当前实现核查结论**。
- 直接证据：`stream_data.py:55-67` 以有理数反转速度图；`stream_generate.py:238,406-413` 依据音频时长求终点，严格式 Simai 编译后才写文件；`event_compile.py:67-82,103-169` 以精确 Fraction 拍位编译并回读验证事件、速度与 metadata，拒绝同拍拆开的 EventPoint。`stream_sampling.py:63-73` 屏蔽落在音频时长之外的 NOTE；`guard.py:183-186` 拒绝拖尾超过音频的动作。导出 E 终点按上取整至 1/4 拍，只延长空谱尾锚，不会把既有 NOTE 改拍。
- 限制：模型的节奏选择仍可能与音乐错拍；以上只能证明时钟/序列化一致性，不能证明音乐对齐。

### G8：历史编辑的状态回放机制已存在，不应报告为“只改文字未重建状态”

- 类型：**已修复/已具备的边界**。
- 直接证据：`stream_search.py:100-188` 仅对训练词表可编码的 sustain 缩短或撤回做全反事实门禁，限预算；`stream_search.py:294-309` 再重新 rollout；`stream_generate.py:163-173,371-390` 从权威 StreamStep 日志重建上下文、guard、曲线计数和 RNG；`stream_search.py:329-334` 标记被后缀再生成失效的历史评分。故历史撤回后没有保留旧 tensor 状态的问题在当前源码没有证据。
- 限制：修复只考虑相关 sustain、预算有限，耗尽时 `stream_search.py:310-324` 拒绝导出。这是显式的有限搜索策略。

## 复核记录

- `git rev-parse --short HEAD`：`0ff1f5a`。
- 低负荷探针：`.scratch/python312_runtime/Scripts/python.exe`，`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，只 import 并调用 `kind_logits_for_audio`、`_cut`；输出为 `eos_eligible True wait_eligible False`、`cut_at_9 1 cut_at_8_5 1`。
- 既有四份 2026-10-31 A/B 产物仅作为报告字段实证，不当作当前代码生成的音乐质量结论。

## 2026-11-01 深入追查补充

### G1/G2 最小复现升级

探针源码：`probe_generation_boundaries.py`（同目录）。设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`torch.set_num_threads(1)`，用合成 song、假 rollout/门禁及纯 CPU 238 维音频 bank；从未读取检查点、音频或调用生成主流程。

- G1：假 rollout 在前面按真实 WAIT 契约推进，每次半拍；到 `start=17/2,target=9,end=10` 后模拟合法的 `learned_eos`。第 27 次调用主动抛出 `RuntimeError('bounded_probe_stopped')`；最后 9 次记录全部为相同的 `(17/2,9)`。说明 `search_stream` 既不推进也不在单段预算后结束；并非真实全曲运行验证。
- G2：在 beat 4 是评价端点的合成两秒声学峰上，`softcomfort_strength=0` 消除轨迹项，空候选 `total=0.40945674`，端点 NOTE 同为 `0.40945674`、`actual_nps=0`；把同一 NOTE 移到 beat 3.99，`total=0.69253468`、`actual_nps=0.5`。因此“主要段内评分漏计端点”得到实际函数复现。

### G9：learned plan 并非评分唯一“音乐真值”；默认的有效选择信号仍有限

- 类型：**事实澄清/设计限制**。
- 直接证据：`stream_quality.py:253-269,285-390` 的评分直接从 audio bank 的 onset/flux/energy、声学描述、参考段和两手轨迹计算，未从 `model.plan_head` 读取目标曲线。`stream_generate.py:125-160` 的 learned plan 用作曲线控制的 native 基线；`stream_model.py:1007-1017` 的 plan prior 若启用则进入 kind logits，并非候选评分的唯一真值。`stream_search.py:251-258` 用固定同一音频/条件的分数排序物理合法候选。
- 影响：评分没有“拿 learned plan 预测当唯一真值”的直接循环自证问题；但所用声学/密度启发式未用人类偏好校准，默认单候选时不能选择更好谱。
- 建议/代价：单候选报告保留“诊断”；做受控多候选 A/B 才评估选择力，人工听测/试玩仍必需。

### G10：长占用前缀在轨迹项保留，不在密度/覆盖项重复计数

- 类型：**事实澄清；未发现当前默认的长占用前缀重复/漏计确定 bug**。
- 直接证据：`stream_quality.py:288-293` 段内 `active` 只包含候选起点后的 NOTE，所以已提交前缀的 Hold 不会作为新 onset 重算密度；`stream_quality.py:337-344` 则合并前缀及本次完整候选，`stream_quality.py:136-150` 从当前段前 .6 秒启动轨迹并保留 `end>=warm` 的长占用 job。`stream_search.py:87-92` 的 native 完整门禁也用 guard 的 `interaction_start` 回溯仍活跃的 sustain，避免截断原生路径上下文。
- 限制：软轨迹的 beam32 与 50ms 取样仍是近似；若原生路径未知，`stream_quality.py:199-210` 降低 coverage 并向 0.5 回退，不能证明手感。

### G11：多候选分支是顺序独立重放，共享音频编码；不是并行搜索

- 类型：**性能/状态契约**。
- 直接证据：`stream_search.py:259-265` 的 `for candidate_index in range(count)` 顺序调用 `run`，不同 seed 用 `phrase_index` 与 `attempt` 派生（`216`）；`stream_generate.py:239-248` 在搜索前只建立一次 bank 和 `encode_audio`/plan cache，`370-390` 的各分支 `StreamInferenceContext(... audio_bank=audio_bank)` 共享只读 bank/模型 audio cache，同时 `replay_stream_prefix` 重建私有 context、guard、curve 计数与 RNG。`stream_model.py:925-926` 有 `audio_cache` 时不重新 `encode_audio`；`stream_data.py:833-945` 的前缀构造与 batch 仍重放/物化历史窗口，`stream_model.py:983-989` decoder 对完整窗口做注意力。
- 影响：相同 song 的单次 `generate` 没有每候选重复编码整曲，但每候选都重建前缀并逐步前向，CPU/门禁/decoder 花费随候选数及已提交长度增长。`evaluate_panel` 在不同 seed 间顺序调用 `generate`（`stream_generate.py:470-484`），故跨 seed 仍重复建立 bank/编码音频；这是评测时间成本，不是质量 bug。
- 建议/代价：先以每阶段 wall time、native_calls、audio encode 次数和单段长度做只读 profiling；优化前缀重放需维护状态隔离，实施成本较高。不要把“共享编码”误写成“多种子并行”。

### G12：多候选替换普通私有尾部时，较早段的评分记录可过期

- 类型：**记录准确性 bug / P3**；需要 `phrase_candidates` 为 2/4、后一段获胜分支不是保留前一胜者的私有尾部，并且较早段分数覆盖了那段尾部。默认候选 1 不触发此构造。
- 直接证据：`stream_search.py:251-253` 对 `[start,target)` 整个写入窗评分；`325-328` 只提交较短的 `[start,commit_end]` 而把后半留作 `retained`；下一段 `259-265` 只有候选 0 继承 `retained`，其他候选重新采样。如果候选 1/2/3 获胜，前一段评分所见的尾部被普通替换；但 `329-334` 仅在 `changes`（历史修复/撤回等）非空时标记先前记录失效，正常另一分支获胜不标记。
- 影响：`phrase_search.phrases` 的旧分数可能描述已不在最终谱中的事件；最终选择机制本身仍以当前候选评分，不据此认定谱质量受损。
- 建议/代价：记录每段 score 的实际候选覆盖范围与最终提交状态，普通尾部替换也标记对应先前记录失效或重算最终谱的离线评分。低至中等成本。

### G13：MiaCode static/runtime 来源及终止策略核查

- 类型：**事实/边界**。
- 直接证据：`miacode_conflicts.py:32-50` 将精确事件编译成 Simai 并调用本地 `analyze_inote`；`53-106` 收集 runtime diagnostics 与 static references，较强 Muri 静态记录先排序以免被同键较弱 runtime 遮蔽；`121-139` 去重后保留 `miacode_runtime:` 或 `miacode_static:` 证据；`172-185` 未映射原生音符会变为 parse error，标记 `playable_verified=False`。`guard.py:20-32,137-141` 再决定哪些结果用于实际拦截，不能只看引擎 `action`。默认 phrase 模式 `stream_search.py:246-250,310-324` 对 `max_events`、`step_budget`、`time_budget`、`physical_dead_end` 判为不完整并失败关闭；`stream_generate.py:287-305` 的 learned EOS 或尾部无合法 onset 则只在靠近曲末时将 cursor 设为 end，后者没有伪造 WAIT；`stream_data.py:87-135` 训练 EOS 同样表示尾部无更多事件，WAIT 固定 2 拍。`stream_generate.py:408-413` 最终默认做全谱物理复判再导出。显式 `search_policy=legacy` 则不走该最终硬门禁，在 `stream_generate.py:357-365` 可退化为 forced WAIT；这属于用户显式选择的旧模式风险。
- 建议/代价：报告和 WebUI 应持续保留实际 `search_policy`、physical policy、终止原因；默认路径不应被描述为“冲突只报告仍导出”。

### 默认配置的结论边界

默认 `phrase_write_beats=8`、`phrase_commit_beats=4`、`phrase_candidates=1`、`search_policy=phrase_v1`、`physical_policy=runtime_hard_v2`。本次深入核查没有在默认参数下复现 G1、G2、G12 的恶性生成故障；它们是允许配置的确定缺陷。默认真实存在的限制是：单候选不产生评分择优、合法过滤改变采样分布、软手感/听感未由硬门禁证明。现有历史报告中的成功/冲突数不能当作本提交的新生成质量验证。

## 对 `antigravity_review.md` 第 6 节与第三大节的专项复核

### A1：短 rest 导致全时点屏蔽再强制 WAIT——存在窄构造，但原表述过宽

- 类型：**有条件的机制风险，未证明真实默认运行已发生**。无曲线文件时 `_Curve.active=False`（`stream_generate.py:138-153`），故默认无显式 rest；用户曲线才可能触发。实际还要先抽到 NOTE，才能进入时点分支（`stream_generate.py:284-301`）。
- 条件链：`stream_generate.py:295-307` 把每个候选时点的 rest mask 置为 `-inf`，仅当 `note_delta_logits_for_audio` 的合法时点全部落 rest 内才强制 WAIT 2 拍。候选不是只看 rest 起点，而是看最大两拍内整个目录（`stream_timing.py:52-60`；`stream_sampling.py:63-73`）。相对 delta 模式的标准词表含 `delta=2`（`stream_data.py:147-152`）；只要还剩两拍且 rest 在两拍前结束，这个远端时点仍合法，**不会全 mask**。若不足两拍，`stream_generate.py:303-306` 走 `tail_no_legal_onset`，不会强制 WAIT。
- 绝对拍位网格的例外：它列出各相位接下来两次的绝对拍位（`stream_timing.py:52-60,131-137`），游标不在相位目录时最远候选可以略早于 `cursor+2`。合成 48 相位目录、`cursor=1/10`、剩余音频大于两拍时，`probe_curve_rest_window.py` 构造 rest 覆盖全部 96 个合法候选，rest 长 `319/160≈1.99375` 拍；此时强制 WAIT 至 `cursor+2`，超过 rest 尾 `21/4000≈0.00525` 拍。证实逻辑可能性，但这是**接近两拍的 rest 与极短尾差**，不能泛化为任意短休止被跳过；没有检查真实用户曲线/实际生成 trace 的发生率。
- 建议/代价：先在既有 `forced_waits` 与曲线 rest 窗口上做逐例只读定位，记录游标、合法候选、max candidate、rest 端点和 WAIT 后首音；只有找到实际可听损失再考虑细化 WAIT/绝对时点支持。修改 WAIT 契约牵涉训练数据与恢复模型，代价高于一段生成局部补丁。

### A2：候选 0 保留尾部，其他候选重采——分布不对称，但评分窗相同

- 类型：**设计权衡，不是已证实的分数比较 bug**。
- 直接证据：`stream_search.py:259-265` 明确让候选 0 带上上一轮未提交的真实采样尾部，其他候选从权威已提交前缀重采；`stream_search.py:231-253` 对所有候选的 `result['points']` 在同一 `[start,target)` 与同一 scorer 上评分；`stream_generate.py:371-390` 每条分支都重放自己的完整前缀、独立 guard/curve/RNG。故不能把“候选 0 只新生成半窗”推断成“只评价半窗”。
- 影响：候选来源不是同分布；候选 0 更接近上一轮被选中过的尾部，可提高衔接稳定性，也可能减小新探索幅度。哪种占优需通过候选日志统计，不是源码可证明的质量缺陷。若其他候选胜出，旧评分记录可过期，见 G12。
- 建议/代价：对 `candidate_logs` 分层统计 origin、有效率、得分与最终胜出率，再在相同音频/种子做有/无保留尾部的受控比较；不要直接抹除该机制，因为它保护前一轮采样的连续性。

### A3：固定搜索网格与动态音乐 phrase 边界不同——事实，影响待测

- 类型：**架构对齐限制，非已发生的状态错位 bug**。
- 直接证据：搜索默认每次写 8 拍、提交 4 拍（`stream_generate.py:41-42`；`stream_search.py:195-212`）；模型的 `music_phrase_layout` 按声学变化从 `2/3/4/6/8` 拍候选中选边界（`stream_phrase.py:174-202,217-248`）。然而模型每步当前 batch 都从 `PhraseIndex.rows_at` 重新取该游标的动态边界（`stream_data.py:884-945`），没有因搜索 4/8 网格而固定在错误 phrase。搜索评分窗可跨音乐 phrase，且提交点可落 phrase 内，是否伤害整句结构未被当前报告证明。
- 建议/代价：先用现成 2026-10-31 产物的音频 bank 与实际 `phrase_search.phrases`，离线计算提交点距动态边界的分布、跨界窗口比例，并与边界处复读/节奏/人工听评对应。若存在相关性，再试只改搜索 span 的固定种子 A/B；不能预设“必须对齐”一定更好，因为 2 拍短句可能导致更多门禁与重放，且 G1 对短 span 仍需先修。

### A4：远端声学参考与五类直方图确有评分缺口，但建议先建立可验证诊断

- 类型：**设计限制；改善效果是假设**。
- 直接证据：`stream_phrase.py:237-246` 的模型 phrase layout 会给近两段与最多两段远端声学参考；`stream_quality.py:347-378` 的搜索评分只取紧邻搜索窗之前的一段，并在声学相似度大于 .65 时比较 Tap/Hold/Slide/Touch/TouchHold 五类占比。该项总权重 .025（`stream_quality.py:379-389`），不比较事件相对拍位、键位几何、旋转/镜像或远端重现；所以它无法检验“远端副歌节奏/手势呼应”。但模型的远端参考仍可能影响候选的生成（`stream_data.py:833-945`），不能说整条生成链从未看远端。
- 下一步/代价：先给固定音频/种子候选做只读离线诊断：将模型 layout 中 `far_top1` 对应的已提交段映射到当前段归一化进度 8 槽；比较 onset 槽分布和相对按键运动（允许旋转/镜像匹配），同时记录声学相似度与类型直方图，并由盲听/试玩确认相关性。只有 `phrase_candidates>1` 时新增评分项才能改变选择；默认单候选加评分仍只会多一项诊断。若验证有效，再用小权重、显式 coverage 和 A/B 门槛加入可选排序，不引入硬复制约束或直接改变训练检查点。
