# song_form_v2 训练侧修复记录（2026-10-15）

## 已确认的问题

`song_form_v2` 的 Blueprint 只预测四通道负荷曲线；事件拍位和分音由后续布局/精确事件解码阶段生成。因此不能把 Blueprint 的输出直接解释为起音列表。

审查训练与生成的输入合同后，确认了一个会改变推理分布的问题：训练侧 `src/maichart/form/train.py` 原先调用 `audio_queries` 时不限制 `max_nodes`，会把整首歌曲的所有原生攻击证据送入 Planner；生成侧 `src/maichart/form/rational_generate.py` 固定使用 `max_nodes=1024`，仅保留预算内的较强证据。训练和生成因此看到的查询数量与证据选择不同。它不会必然造成每首歌退化，但会使 Planner 负荷曲线在生成时面对未训练过的输入分布，属于确定的合同不一致。

## 修复

- 在训练侧引入 `RATIONAL_PLANNER_MAX_AUDIO_NODES = 1024`。
- `_blueprint_input` 使用同一预算调用 `audio_queries`。
- Blueprint 输入缓存身份记录同一 `max_nodes=1024`，避免继续复用旧的无限节点缓存。
- 增加回归测试，确认训练调用实际传递 1024，并保持 rational Planner 的其他接口不变。
- Blueprint 日志新增 `planner_density_prediction` 与 `planner_density_target`，只用于监控预测负荷均值和目标均值，不改变梯度或 loss。
- `PLANNER_MAX_AUDIO_NODES` 与 `planner_audio_queries` 已移至 `rational_audio.py`，训练和曲线预测共用；曲线预测的 rational 分支复用同一次查询，不再重复构造整曲查询。

这样做的目的，是让训练和生成都使用相同的原生音频证据选择规则；它不把位置量化到固定帧，也不改变 Fraction 事件表达能力。

## 验证

通过：

```text
tests/test_form_rational_events.py::test_training_planner_query_budget_matches_generation
tests/test_form_blueprint_batch.py
tests/test_form_blueprint_cache.py
7 passed
```

未启动或重启正式训练。已有运行中的 checkpoint 不会自动改变；修改对新建 Blueprint 缓存和后续训练生效。

## 当前 checkpoint 的只读前向审计

以下结果来自真实命令：

```text
python -  # 读取 checkpoints/Prism_Re_v1/form_last.pt 与一个 Blueprint 输入缓存，CPU 前向 RationalPlanner
```

输入文件：

- `E:/simai_transformer/checkpoints/Prism_Re_v1/form_last.pt`
- `E:/simai_transformer/webui_workspace/datasets/full训练集/cache/blueprint_inputs/v1/00976b38869d616dd3c1e477a328b812efe4e56222fc04a29cc6eda5608e5991.pt`

输出：Planner 预测形状为 `(6114, 4)`；四个通道预测均值为 `[0.6821, 0.6479, 0.2570, 0.2095]`，目标均值为 `[0.5878, 0.5327, 0.2491, 0.3084]`；event-density 通道相关系数为 `0.7109`。这是单个缓存样本的只读诊断，不代表整套数据的质量验收，也不证明生成端已经可用。

## 尚未修改的质量风险

- revision 训练样本中的 draft 保留了被遮罩事件的精确拍位，可能使修订任务的拍位指标高于 fresh draft 的真实能力；这属于训练分布风险，本次未擅自改变任务合同。
- loss/token accuracy 主要衡量 token 级预测，不能单独证明 Fraction 拍位、节奏网格或整曲密度已经稳定。
- 生成端 `invalid_position_prefix` 导致的空段、Fraction 数字解码和局部回退由生成侧单独处理，本次没有改动。
