# chart_composer_v1 首版实现交接

日期：2026-10-22

## 已实现

- 新模型族：`chart_composer_v1`。
- 独立 checkpoint schema：`maichart-chart-composer-checkpoint-v2`（旧 v1 合同含真实谱面负荷输入，禁止继续加载）。
- 事件级候选选位：候选包含常见节奏和长休止；训练与生成使用同一候选提案，真实下一事件不再被临时注入。
- 结构化动作输出：模型对每个候选拍点预测固定数量动作槽位，每个槽位选择训练原子；生成器按温度、STOP 和语法/按键合法性拼成同拍动作集合。
- 未来意图状态：训练从后续四事件派生间隔、头密度、长休止、动作类型比例和结构变化；下一事件读取前一事件的意图状态，生成时读取模型自己的上一状态预测。
- 复用现有 `FormDataset`、rational audio query、曲线、精确 Fraction 时间和 `compile_event_points`。
- 训练 CLI：`python -m maichart.composer.train`。
- 生成 CLI：`python -m maichart.composer.generate`。
- WebUI 独立入口：`/composer`，包含训练、生成和 checkpoint inspection。

## 当前边界

首版是事件级候选指针、动作槽位和单分支未来意图原型，尚未实现未来草稿分支、双向 Critic 修订、多解谱面边缘化、预训练音乐编码器或真实偏好学习。训练候选不再显式注入真实下一事件；未召回事件按候选覆盖率统计，超过阈值会阻断训练。候选覆盖率、生成质量和可玩性仍需分开验收。

动作槽位目前以训练原子为词表，适合验证事件级训练链路；Slide 路径、时值和多事件动作短句的专门结构化编码仍是下一阶段工作。生成端只保留导出/语法级合法性边界，不宣称自动保证可玩性。

## 2026-10-22 修复合同

- 候选输入只包含音频查询和有理拍位特征，不再输入由真实谱面统计的 `load_curve`；旧 v1 checkpoint 禁止加载。
- 候选拍位来自有界的音乐网格和长休止锚点，训练不会把真实下一拍偷偷注入候选集；未召回事件会计入 `candidate_misses`。
- 动作头按候选拍点条件化；历史保留同拍多个 atom，并额外提供相隔拍数、相隔秒数和动作结构特征。
- 生成维护已写入的 Hold/Slide 状态，动作按温度、STOP 和语法/重复按键检查采样；`write_logit` 改为候选条件化 EOS 监督。
- 新合同为 `chart_composer_training_v2_audio_only_candidate_actions`，checkpoint schema 为 `maichart-chart-composer-checkpoint-v2`。
- 首事件使用显式空前缀锚点，候选不再用负拍偏移；候选网格过滤到常见有限分母，并补充长休止的四分/半拍落点。
- Composer 音频查询使用 8192 个 detail nodes；按 `audio_cache_gb` 的四分之一、上限 1 GiB 管理整曲特征 LRU，避免 1024 节点稀释瞬态或 16 首固定缓存造成反复读盘。
- 历史位置编码改为相对当前事件的 recency；动作采样读取活动持续动作占用，默认限制新增事件最多两个独立按键；EOS 使用 batch 动态正例权重。
- 持续动作硬过滤只使用 Hold/Slide/TouchHold 的具体起始键位，不把 MiaCode 的手部空间覆盖 pads 当作已占用键；候选召回率超过 5% miss 才阻断训练，低于该阈值会记录并继续。
- Slide/Wifi 滑行不再被当作整段持续占用的起始键；精确 Hold 结束时刻允许释放后重新使用。候选优先级始终保留 2 拍内常用细分，再分配长休止锚点。

## 初步验证

- `tests/test_chart_composer_v1.py`：当前 6 项通过。
- `python -m compileall -q src/maichart/composer src/maichart/webui`：通过。
- 既有 Stream WebUI Node 定向测试：12 passed。

下一阶段应使用固定歌曲和固定 seed 比较 Composer 与 Stream：候选召回率、真实休止跨度、密度曲线相关性、结构重复长度、运行时冲突和人工试玩结果。只有候选覆盖率稳定后，才适合继续把训练目标扩展为真正的负样本舍音学习。
