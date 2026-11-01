# Composer 学习链路审计（只读，HEAD 0ff1f5a）

范围：`stream_model`、`stream_actions`、`stream_phrase_model`、`stream_duration`、`stream_train`、`stream_recovery`、`stream_transition`，以及旧 `event_ctx`、`chart_stream_v1`、`song_form` 和 WebUI 路由。未加载 GPU、未停止训练、未重建缓存、未修改实际代码。

## 优先结论

1. **确认的记录口径不一致，非概率实现错误。** `stream_model.py:1853-1857,1876` 把退出样本联合 NLL 额外乘 `transition_exit_weight` 加到总损失，`stream_train.py:549` 却在 recipe 写 `same_budget_as_main_configuration`。默认 0.5 表示退出目标在原有联合 NLL 之外再次得到权重。建议先修正说明与指标命名；若改变权重，则只能作为新训练阶段，不能无声明续训。
2. **确认的指标口径。** `stream_train.py:242-243,725-735` 的 loss 字段和 `exit_count` 是每成功批次平均，`exit_coverage` 是累计总数，`composer_recovery` 是每次调用结果。不能把三者混作同一种分母。
3. **v38 实际不是完全没有恢复监督。** 日志 `webui_workspace/logs/20261101_180957_composer_train_626129bc3951.log` 截至 18:45:42 的 37 次恢复调用有 74 槽（短 56、长 18；长槽中持久 9、合成 9），66 条有效监督、52 条实际 exit；NOTE/WAIT/EOS 为 60/4/2。合成长槽 9 次中 7 行成功、2 次无效；`overshot=9`。这支持恢复路径正在运行，但不足以证明长段自由生成已改善。
4. **v38 当前计算占主导。** 同一日志 epoch 1 batch 1825 / optimizer step 912：`compute_sec=1884.54`、`prefetch.consumer_wait_sec=0.559`、`data_sec=2.930`、`transfer_sec=21.93`、约 30.54 samples/s；无非有限警告和跳过窗口。优化训练吞吐应先测计算内的前反传及恢复耗时，不能据旧 CPU/I/O 病例调高 worker。

## 目标函数、梯度与音乐

`stream_model.py:1675-1681` 固定以 batch×rollout_steps 为主损失分母，保留时间窗采样权重；不是每有效事件均值。`stream_model.py:1849-1876` 的总损失为 kind、delta、shape、phase、detail、0.1×plan、0.5×退出联合 NLL、0.05×phrase，并可加 `output.recovery_loss`。训练器另在 `stream_train.py:687-698` 每 25 optimizer steps 取恢复批，以 0.1×完整 `model.loss` 反传。恢复行只有末位样本权重非零（`stream_data.py:948-1002`），不能将生成前缀的所有事件都算作有效学习目标。

日志 batch1825 的累计均值：`loss=6.7690`、`base_loss=5.8920`、`time_loss=0.6836`、`configuration_loss=3.3245`、`plan_loss=18.8385`（实际乘 0.1）、`transition_exit_loss=0.8006`、`phrase_weighted_loss=0.01805`、`recovery_loss=0.05843`。这说明短语辅助项在损失数值上较小，但不能单凭数值比较梯度大小。

音乐已接线：`stream_model.py:975-981` 局部音频与短语影响隐藏状态，`:1039-1054` 音频参与时间候选，`:1065-1075` 音乐参与退出决策，`:1175-1218` 音频决定事件族先验并与前一事件族混合。`stream_actions.py:69-86` 的概率混合保留归一；`:89-141` 用候选时刻的状态、短语意图和间隔修正事件族。音频先验在 `audio_state_phrase_v4` 的全分支被 detach（`stream_model.py:1210-1217`），但有独立音频家族 CE（`:1717-1726`），并非断梯度。是否真正随音乐改变需固定音频、历史、seed 对比原音频/错配音频/常量音频的自由生成。

整段手顺学习的具体限制：`stream_phrase.py:396-408` 每段最多抽取 32 个顺序事件；`stream_phrase_model.py:167-225` 只读至多四个已完结参考段，且短语预测辅助目标是段统计与压缩序列向量；主动作仍由逐事件 teacher-forced NLL 学习（`stream_model.py:1684-1801`）。批次按歌曲集中（`stream_train.py:176-194`），目标段在批内去重（`stream_phrase.py:451-465`）；若只有一个唯一可见标签，双向对比项精确为零（`stream_phrase_model.py:271-295`，现有测试覆盖）。余弦目标 detach，但目标编码器在多标签匹配项可更新。零初始化的参考与动作残差输出（`stream_phrase_model.py:38-39`）使首步主任务不能反传到前级短语编码器；输出权重更新后路径打开，因此是初期优化迟滞，不是永久错误。建议统计每批唯一标签数、非零 match 比例、各短语模块梯度范数，再决定改变采样/目标。

因果核查：参考段必须 `memory_end <= start`（`stream_phrase_model.py:173`）；目标有序序列只在辅助 loss 编码（`:262`），不送给 forward；`stream_model.py:995-1093` 在时间 logits 已产生后才用真实 `target_second` 条件化动作头，这是训练时的条件概率分解。全曲未来音频可见是生成时可得输入，不属于未来谱面泄漏。没有发现确定的未来标签泄漏。

## 时长音频条件分支

`stream_duration.py:93-113` 只为 regular shape 中可静态解析的 Hold、Slide 细节建表；TouchHold 和 literal 路线不在此版本覆盖内，复合 Slide 也明确跳过（`:23-25,101,146-149`）。零初始化 `score` 使迁移起点不改变原概率（`:84-89`）。训练 `stream_model.py:1749-1779` 与生成 `:1616-1625` 都调用相同的 `duration_detail_logits`，生成路径 `stream_generate.py:317-321` 和恢复路径 `stream_recovery.py:255-259,452-456` 都传入时长上下文，未发现确定的训练生成断线。

候选区间 `stream_duration.py:119-144` 用事件秒数、当地秒/拍和解析的拍数/秒数系数构造，`_read_intervals` 把查询裁剪至音频 bank 末端（`:63-81`），并只对 `end <= end_second`、tempo 有效的候选加入 residual（`:141-144`；`stream_model.py:870-906`）。这保证尾部越界候选无新增音频分数，但仍可用基础 logits 生成；它是覆盖边界，不是确认错误。`audio_tracking_strength=0` 时 `attention=1`，时长音频分数仍存在；v38 日志的生成默认值正是 0。训练 `stream_train.py:671-677` 以 25% 行随机启用非零 tracking，75% 行覆盖默认零设置。建议报告 supported/catalog 数量及在实际谱面中 Hold/Slide/TouchHold 的覆盖率；当前未运行模型加载以避免干扰 GPU。

## checkpoint、RNG 与验证

**确认的续训合同漏洞。** `_fingerprint`（`stream_train.py:292-319`）只纳入 manifest 文件字节、codec/delta 词表、若干窗口与策略参数，没有纳入 manifest 引用的音频或 maidata 内容哈希，也没有准备后 stream/sample/音频 bank 的内容指纹。续训只比较此值（`:581-591`）。若同一路径的音频或谱面被替换，而 manifest 文本、词表集合与策略保持一致，指纹仍相同，保存的样本游标及 RNG 状态不再代表原训练序列。生产缓存按内容重建不等于续训指纹拦截。CPU 最小复现：以同一临时 manifest、同一 codec/delta/策略构造两个 `SimpleNamespace` 数据集，唯独 `sample_content`/`audio_content` 模拟变化，`_fingerprint(args,a)==_fingerprint(args,b)` 为 `True`，哈希 `10e529254ba3df601d862dae3aa77b3fd7f29195b1d8b789727c270acae392ae`。该复现证明函数对内容不可见；没有实际替换生产音频。建议在下一代码阶段把训练素材内容清单或准备后数据指纹写入完整 checkpoint，resume 严格比较；旧 checkpoint 需显式兼容/拒绝策略，不应静默改变语义。

checkpoint 经 `.tmp` 和 `os.replace` 原子替换（`stream_checkpoint.py:15-27`），保存模型、优化器、epoch/step、训练 cursor、Python/NumPy/Torch/CUDA RNG、恢复 episode/cursors、学习策略（`stream_train.py:636-647`）；续训验证 fingerprint、batch/accumulation、若干目标参数并恢复 RNG（`:581-605`）。sampler 用局部 Torch generator（`:176-194,219-232`），恢复轨迹用 step 种子的局部 Python/Torch generator（`stream_recovery.py:303-305,655-656`）。这些代码支持优化器边界续训，但固定效果仍要做中断与连续运行的 CPU 小样本对比。

验证 `stream_train.py:427-440` 采用 eval/no_grad 与专用 DataLoader generator，无显式全局随机抽样；它在 epoch-end `persist` 之前执行（`:738-742`），所以若验证读取了任何全局 RNG，其后状态也会被保存。generation panel 在 persist **之后**执行，但由 `_preserve_rng` 包裹（`:283-290,746-752`），Python/NumPy/Torch/CUDA RNG 均恢复，故从当前接线看不会影响下一轮训练随机序列。v38 设置 `generation_eval_songs=0`，截至日志末尾尚未完成 epoch 1、未执行 validation；不能用这份日志声称验证后续训的实测一致性。已保存一次 step500 checkpoint，无 warning。

## 其他仍可进入的架构与风险边界

WebUI `server.py:1154-1177` 把 `chart_stream_v1`、`song_form_v1/v2`、`chart_composer_v1` 分别路由到独立训练入口；event_v1/event_ctx_v1 走旧事件训练支路。event-set Composer 的 WebUI 入口 `event_set_api.py:80-89` 启动 `maichart.composer.stream_train`，旧 Composer `composer_api.py:378-390` 走另一训练合同；不能仅凭同一 `composer_train` task kind 推断 checkpoint 兼容。`stream_checkpoint.py:7-9,29-38` 对 event-set profile 和 schema 做隔离。`song_form` 是另一模型族，不可作为本次 v38 梯度结论。

`event_ctx_v1` 在 `train_event_ctx.py:1-35,49-75` 与 `generation/event_ctx.py:1-29,70-73` 采用窗口前缀、邻域音频、窗口监督掩码、可选 overlap 和 context dropout。可借鉴的具体机制是**随机删减历史上下文，检验复制依赖**，并以窗口拼接 A/B 测段落接缝；Composer 当前有历史和短语参考，但本轮未发现同等的历史 dropout 训练开关。旧架构是对照思路，不能把其 token 或 checkpoint 合同直接搬到 event-set。

## Antigravity 第 4/5 项及 STATE50 复核

**模板保持改变语义类型族边际：确认的概率合同事实，影响程度待测。** `stream_model.py:1318-1323` 先按 `family_logits` 构造各路由概率，随后对单个旧 `reference_shape` 与其余全部路由做二组重分配；`sample_event_intent` 在 `:1337-1353` 生成时先抽 keep 组，再抽该组内的 family。训练 `:1119` 把重分配后的 route 边际当作 full family logits。CPU toy 调用真实 `residual_keep_logit` 和 `two_group_log_probs`：原路由 `[A模板=.1,A其他=.1,B=.8]`，正 residual `ln(9)/8` 后，A 边际 `.2→.5556`、B `.8→.4444`，总概率仍为 `1.0`。这证明模板偏好可压过音乐族先验；但 v3 的 residual 为零时保留原分布（`stream_transition.py:42-69`），模型可学习平衡二者，不能断言“必然架空”。单独音频族 CE 仍占既定 25% 预算，`audio_logits.detach()` 是避免 full 分支重复训练该先验的设计，不能称零梯度 bug。若要保证音频族质量严格不变，可先抽 family，再仅在该 family 内决定模板保持；这改变概率分解，需新模型阶段与重新校准。

**跨乐句候选仍用游标 intent：确认的条件性时钟缺口。** `stream_generate.py:275-277` 在游标构造一次 `result`，`:309-324` 对每个候选虽重查实际时刻音频，却把旧 `result` 交给 family context；`stream_model.py:653-671` 用旧 `result.hidden` 构造候选动作；`stream_phrase_model.py:115-122` 的 actor remaining 按 `step_beat` 而非候选 beat 计算。若候选跨过音乐乐句边界，family state phrase 修正仍见上一乐句的 `phrase_intent`（`stream_model.py:1212`）。训练 forward 同样按游标短语和目标时刻动作构造，故不是明显的训练/推理接线不一致，而是“事件归属游标乐句还是起音乐句”的策略缺口。可先统计跨界候选比例与同音频 A/B；若起音应归新乐句，则候选时刻重算短语 intent/actor，并新阶段微调。

**候选 STATE50 几何信息未完整推进：确认的表征不一致，具体恶性影响待测。** `stream_model.py:965-969` 把游标时刻 `action_state` 池化进入 `hidden`；`:632-651` 的 `action_context_at` 在候选时刻重算音频、按钮负荷和触控负荷，但未重算 STATE50。`stream_actions.py:107-130` 只在事件族残差中将 `remaining/until` 减去候选间隔，压成 9 个全局量，空间方向仍丢失。因此不能说几何解码器完全没有候选状态更新；真实缺口是旧 STATE50 摘要仍留在 hidden，当持续动作在候选间隔内结束时可能与新负荷冲突。建议在受控新阶段加入共享候选时刻 STATE50 残差并零初始化、训练校准；先做状态变化点分桶 A/B，勿直接把未知物理影响说成已证实冲突。

## 覆盖与待验证

已核对上述模块的概率分解、teacher-forced 与生成细节路径、恢复合法性与统计、RNG/checkpoint、WebUI 实际入口；没有运行 GPU、训练或缓存重建。未证明 v38 的音乐质量、长段手顺、可玩性改善；这些必须用冻结 checkpoint 的同音频同历史同 seed A/B、逐段指标以及听感和游玩验证。CPU 最小复现可优先做：单唯一有序目标的 match=0（已有测试）；连续训练与 step500 续训的相同 batch/order/RNG 校验；时长候选 support/尾部裁剪检查。
