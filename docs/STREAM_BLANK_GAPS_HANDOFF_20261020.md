# chart_stream 大段空白修复交接

日期：2026-10-20。工作分支：`codex/v11-quality-fixes-20261020`。

## 结论和证据边界

本轮参考用户提供的独立报告，重新核对实际工作区和最新输出 `webui_workspace/generated/f578eed8fc84/maidata.stream.json`，没有照搬报告中的全部归因。最新输出的主要空白来自绝对拍位大跳，不能简单归结为 EOS 提前结束。该输出有 128 个事件，最长内部间隔 50.1316 秒；同一报告记录过 `free_gib=3.954, beam=1, candidates=1`。旧记录没有显存降级发生的事件索引，因此不能断言它恰好发生在第 126 个事件。

已确认的缺陷是稀疏循环退出损失的梯度放大，以及低显存/宽度 1 时跳过完整拍位搜索。另有完整分数搜索对大跳与短间隔预测冲突复查不足的问题。修复已通过定向测试和保存候选的重评分；没有重新训练或对真实歌曲重新生成，不能据此承诺全部空白或音乐质量问题消失。

## 独立报告逐项核对

| 报告观点 | 当前代码结论 | 处理 |
| --- | --- | --- |
| cycle_exit 独立归一化放大稀疏 token | 成立。附加梯度相对主 token CE 为 `weight * valid_count / exit_count`，放大程度随 batch 内容变化 | 改为与主 token CE 共用有效 token 分母 |
| plan / boundary 错位一 token | 不成立。当前 codec 的 BOS=1、AT=3、NOTE=4、END=5。`target==AT` 的行，decoder 输入是 BOS/END，model 使用的边界一致 | 不修改正确边界；新增数据集、模型 hook 和时钟回归 |
| 64 拍后时钟饱和 | 不成立。当前 `exact_fraction_v2` 为 `beat/64`，未做 `min(beat,64)`；检索乘回 64 也没有此截断 | 保持当前时钟，新增超过 64、512、1024 拍及遮罩后的回归 |
| 假分数分子前缀可变成整数大跳 | 是真实的表征风险，但语法单调性本身正确，不能允许时间倒退 | 保留精确 Fraction 和单调性；发生计划冲突时额外比较附近候选 |
| 显存降级会绕过搜索 | 成立。旧代码 `effective_beam>1` 才执行完整搜索 | 宽度 1 同样执行完整搜索，候选按宽度分批计算 |

不能从退出 accuracy 高于整体 accuracy，或从权重范数变大，单独证明某 checkpoint 已被某一条损失破坏。循环退出检测按精确拍差判断，并不是看到整数与分数书写切换就自动标记退出。相关行为已补测试。

## 实际修改与目的

### 1. 有界训练目标

`stream/train.py::_loss` 中 `cycle_exit` 改为：

```python
(ce * exit_mask * local_quality * chart_quality).sum() / main_valid_tokens
```

默认退出权重 0.25 表示对标记 token 额外增加主 token CE 的 25% 梯度，而不是根据退出标记稀疏程度放大几十倍。保留原谱真实退出标签、质量标注、padding 和排除规则。没有增加伪造错误前缀或随意删除正常重复。

训练语义升级为 `chart_stream_training_v13_bounded_cycle_exit`；旧合同不能直接 resume，应使用 finetune 与新输出目录。v12 到 v13 的 plan 含义一致，迁移保留其全部 plan 权重。早于 v12 或缺失合同才重置旧绝对相位通道 1:3，保留间隔通道 0 和其余类型通道。

### 2. 小显存仍保留完整拍位评分

`stream/decoding.py` 统一使用 `CanonicalPositionGrammar` 和 `search_position`。无论手动宽度 1，还是显存保护降为 1，都保留完整序列原始概率、复杂度、间隔一致性和节奏循环成本。额外候选以 `max_batch_size=effective_beam` 分批评估，避免保护降为 1 后仍一次复制 16 份 KV cache。

`form/rational_position_search.py` 新增可选 `proposal_trigger`。stream 在计划未饱和且候选拍差超过 `max(2*predicted_gap, predicted_gap+4)` 时要求复查。额外候选与原候选同台打分，不是可输出拍位白名单；原来的罕见 Fraction 和长休止仍可获胜。未传新参数的 song_form 调用保持原规则。

### 3. 大跳软抑制与 WebUI

复用工作区已有的 `unplanned_gap_penalty_strength`，默认从 0 调整为 0.5；CLI、WebUI payload、API 默认值和输出报告同步。WebUI 用词为“非预期长空白抑制强度”。

计划头预测短间隔时，对超出上述阈值的拍差追加线性软成本；预测达到 8 拍上限表示“8 拍或更长”，不追加该成本。间隔引导强度设为 0 时，额外长空白抑制仍可独立工作。0 可关闭额外惩罚。错误的短间隔预测仍可能压低真实长休止，不能把该启发式称为音乐正确性保证。浏览器已保存的 0 不会被自动覆盖，请手动设为 0.5。

生成报告补充实际长空白抑制强度、触发复查标记，以及显存降级时的事件数和前一拍位。被拦住的大跳复查也保留在 trace 中，方便下次定位。

后端生成合同升级为 `chart_stream_quality_controls_v4`；状态接口增加 `stream_cycle_exit_normalization=main_valid_tokens_v1`。前端检查新标识，避免新页面静默向旧服务提交修复后的设置。

## 真实保存候选的重评分

以最新报告保存的候选分数为起点，只追加 0.5 强度的线性成本，没有重新运行模型：

| 前一拍位 | 保存时选中 | 新评分最优 |
| --- | --- | --- |
| 75 | 148 | 151/2 |
| 148 | 397/2 | 297/2 |
| 423/2 | 1885/8 | 427/2 |

该结果证明这些具体错误候选可以被新评分压低；前一步变化会改变后续模型上下文，因此不能将这张表当作整曲重生成结果。最新报告实际 `position_interval_strength=0.35`，与独立报告中部分表述不同。

## 定向验证

- Python 定向回归：124 passed。涵盖退出梯度上限、v12/v13 plan 迁移、实际循环中的低显存降级、宽度 1 完整搜索、数字进位候选、正常长休止、规范 Fraction、训练/推理边界和长时钟、HTTP 参数传递。
- WebUI Node 回归：27 passed。
- 补充真实 GroundedInference 候选分批与并行一致性、因果时钟不变，以及共享 rational 搜索/解码回归：13 passed。
- `git diff --check` 通过。没有启动训练、没有使用真实 checkpoint 重新生成，也没有进行全量训练质量验收。

## 操作与交接

1. 现有 v12 权重仍可生成；建议先同歌曲、同 seed 对比新生成侧效果，再决定微调或 fresh。训练目标修复不会追溯修改已有权重。
2. 启用新训练目标请选择 finetune 和新目录。v12/v13 plan 几何合同一致，v12 导入时不再被清空。
3. 生成“非预期长空白抑制强度”先用 0.5；异常跳跃多时可谨慎增至 1，若真实休止被压低则调小或关闭。保持原分母设置，不通过固定网格或补音强填空白。
4. 7860、7861 重载前已检查没有活动训练、生成或解压任务；服务和最终接口检查结果在下方运行记录补充。
5. 工作区已有大量历史未提交修改；本轮没有提交、回滚、删缓存或修改训练权重。备份快照在 `E:\simai_transformer_backups\blank_gap_before_20261020`，更早框架备份在 `E:\simai_transformer_backups\v11_quality_before_20261020`。

## 运行记录

7860、7861 均已在确认无活动任务和解压后重载：

- 7860 新进程 PID 57140；7861 新进程 PID 50196。
- 两端 `/api/status` 返回 `stream_generation_contract=chart_stream_quality_controls_v4`。
- 两端训练能力返回 `stream_cycle_exit_normalization=main_valid_tokens_v1`。
- 实际 HTTP 页面中“非预期长空白抑制强度”默认值为 0.5，返回的 app.js 检查新合同。
- 两端接口均无活动训练或生成任务；本轮没有自动开始训练。
- 合同标识及诊断字段更新后，生成/API 定向复查 20 passed，WebUI Node 复查 27 passed；这些属于前述范围的重复确认，不与之前数量相加。
