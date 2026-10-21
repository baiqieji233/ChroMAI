# Stream v18 音频贴合与恶性复读修复交接

## 目的

本轮修复针对连续谱面生成中最明显的质量缺陷：模型在歌曲弱段仍连续铺节奏、候选拍位没有真正使用音频选择证据，以及配置结构在生成时进入长周期自强化循环。目标是让音频证据参与候选拍位和配置决策，同时保留谱师有意的短句重复。

## 已确认的根因

1. `PositionEvidence` 原先只把 selection 头用于整体置信度，候选时间查询只看 alignment 分数，舍音位置没有得到直接排序影响。
2. `rhythm_cycle_cost` 原先只按复合间隔周期计费，缺少曲线/音频共同支持时的“正常舍音”区分，容易把重复节奏和正常节奏混在一起。
3. `RepeatBias` 对结构重复候选在前缀阶段可能给出相同常数；当所有提案都属于同一结构时，softmax 中的常数惩罚会完全抵消，导致重复结构继续被采样。
4. 历史位置与间隔通道保留得过于完整，训练可以通过复制前文 `{8}`、`{16}` 和间隔捷径完成位置预测，削弱音频依赖。
5. 音频投影后立即 LayerNorm，动态包络虽存在于输入列中，但没有独立的低风险旁路；局部 condition 也没有直接调制 GroundedContext。

## 本轮实现边界

- 生成侧：候选位置加入逐点 selection 相对代价、曲线密度弱先验和节奏循环的保守联合判定；结构重复进入候选级 escape bias，并在所有候选重复时进行一次高探索逃逸重采样。
- 训练侧：增加历史位置有限遮罩，保留最近位置锚点；不遮挡当前 teacher-forcing 事件的 `P:` 目标上下文；同时清除被遮挡历史事件的时钟几何回声。增加 0.1 默认的条件随机丢弃，避免模型把 Radar/Style 当成歌曲内容替代品。增加动态 RMS/attack/slope/novelty 旁路；局部 grounding 增加弱的零初始化 condition 投影。
- 旧检查点仍可加载，新增旁路为零初始化；改变训练语义与数据指纹后，旧 checkpoint 应使用 `finetune` 或 fresh，不能 resume。

## 验收重点

后续生成应重点观察：重复结构最长连续长度、弱段头密度、候选 selection 排名变化、复合节奏在有明确音频证据时是否保留、以及硬性运行时冲突。人工听音与试玩仍由使用者判断，量化诊断只用于定位和比较，不作为自动拒绝谱面的唯一标准。

结构逃逸是有界策略：当候选目录中存在不同结构时会进行一次高温逃逸提案并优先采用新结构；如果训练语料的原子目录本身没有可行替代，系统会保留模型候选并在报告中标记 `escape_retry_unavailable`，不会伪造不存在的音符类型。

## 本轮验证

- `tests/test_stream_generation_evidence.py`、`test_stream_grounding.py`、`test_stream_pipeline.py`：39 passed。
- `test_stream_structural_repeat.py`、`test_stream_timing_search.py`、`test_stream_training_contract.py`、`test_stream_configuration_search.py`：78 passed。
- `test_stream_generation_integrity.py`、`test_stream_audio_bounds.py`、`test_stream_runtime_optimization.py`、`test_stream_training_speed.py`：40 passed。
- `python -m compileall -q src/maichart/stream`：通过。
- 收尾回归（grounding、causal motor、pipeline、training contract、evidence、repeat、configuration、timing、runtime）：144 passed。
- WebUI Node 定向回归：16 passed。

- WebUI 静态/接口参数已同步：训练位置遮罩与 condition dropout、生成密度引导与重复逃逸尝试均有控件和 CLI 合同。

## 训练迁移

本轮训练语义升级为 `chart_stream_training_v15_audio_fidelity_position_mask`，目标与缓存指纹随之变化。已有 v17/v14 检查点仍可用于生成和 `--init-checkpoint` 微调；不能使用旧优化器直接 `resume`，因为位置遮罩和音频贴合旁路改变了训练合同。新增动态旁路与局部 condition 投影为零初始化，旧检查点加载时不会改变原函数，只有微调或 fresh 后才会获得新能力。
