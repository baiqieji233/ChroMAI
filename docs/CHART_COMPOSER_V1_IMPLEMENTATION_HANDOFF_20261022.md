# chart_composer_v1 首版实现交接

日期：2026-10-22

## 已实现

- 新模型族：`chart_composer_v1`。
- 独立 checkpoint schema：`maichart-chart-composer-checkpoint-v1`。
- 事件级候选选位：候选包含常见节奏、秒轴间隔和长休止，并支持训练时覆盖真实下一事件。
- 结构化动作输出：模型预测固定数量动作槽位，每个槽位选择训练原子；生成器将去重后的原子拼成同拍动作集合。
- 未来意图状态：训练从后续四事件派生间隔、头密度、长休止、动作类型比例和结构变化；下一事件读取前一事件的意图状态，生成时读取模型自己的上一状态预测。
- 复用现有 `FormDataset`、rational audio query、曲线、精确 Fraction 时间和 `compile_event_points`。
- 训练 CLI：`python -m maichart.composer.train`。
- 生成 CLI：`python -m maichart.composer.generate`。
- WebUI 独立入口：`/composer`，包含训练、生成和 checkpoint inspection。

## 当前边界

首版是事件级候选指针、动作槽位和单分支未来意图原型，尚未实现未来草稿分支、双向 Critic 修订、多解谱面边缘化、预训练音乐编码器或真实偏好学习。训练候选会显式保留真实下一事件，以保证原谱监督覆盖；训练和自由生成的候选覆盖率必须在后续报告中单独统计，不能把当前原型当成最终自主谱师。

动作槽位目前以训练原子为词表，适合验证事件级训练链路；Slide 路径、时值和多事件动作短句的专门结构化编码仍是下一阶段工作。生成端只保留导出/语法级合法性边界，不宣称自动保证可玩性。

## 初步验证

- `tests/test_chart_composer_v1.py`：3 passed。
- `python -m compileall -q src/maichart/composer src/maichart/webui`：通过。
- 既有 Stream WebUI Node 定向测试：12 passed。

下一阶段应使用固定歌曲和固定 seed 比较 Composer 与 Stream：候选召回率、真实休止跨度、密度曲线相关性、结构重复长度、运行时冲突和人工试玩结果。只有候选覆盖率稳定后，才适合继续把训练目标扩展为真正的负样本舍音学习。
