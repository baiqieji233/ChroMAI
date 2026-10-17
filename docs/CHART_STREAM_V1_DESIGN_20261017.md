# chart_stream_v1 连续整曲架构（设计与首版实现）

日期：2026-10-17  
分支：`codex/chart-stream-v1`  
模型族：`chart_stream_v1`  
checkpoint：`maichart-chart-stream-checkpoint-v1`

## 目标

新模型把整首谱面作为一个连续事件流。音频查询、BPM、拍号、`first` 和 Radar/Style 条件持续存在；计算块只用于显存控制，不代表音乐边界。只有真实整曲结束才具有 EOS 语义。

谱师选择是监督主线。音频特征用于提供证据，不能要求每个瞬态都生成 note，也不使用响度阈值替代谱师选择。连续流的计划头只从官谱事件推导节奏间隔、类型和持续动作倾向，不能把“越密越好”当作目标。

## 已实现的首版骨架

- `src/maichart/stream/model.py`：`ChartStreamTransformer`。音频编码、条件、近期历史、持久 stream memory、因果事件解码、计划辅助头。
- `src/maichart/stream/data.py`：基于现有 `FormDataset` 和 `rational_audio_queries_v1` 的整曲事件 token。按事件边界建立 continuation chunk，chunk 末尾不添加 EOS。
- `src/maichart/stream/checkpoint.py`：独立 schema、独立模型族、manifest 身份和原子保存。
- `src/maichart/stream/train.py`：fresh/resume/finetune CLI 骨架，配置 token CE 加轻量计划辅助损失。
- `src/maichart/stream/generate.py`：连续 Fraction 事件生成和 Simai 编译桥接。它仍是首版原型，尚未宣称音乐质量。
- `src/maichart/webui/stream_api.py`：独立训练、生成和 checkpoint inspection dispatcher，不复用 `form_api.py` 的 song-form checkpoint 假设。
- `model_compatibility.py`：识别 `chart_stream_v1`，拒绝与 event/form checkpoint 互载。
- 主 WebUI：新增模型族选项、连续流参数面板、专用训练/生成请求路由。

首阶段集中验证已完成：`34 passed`，覆盖新模型前向/梯度、Fraction 事件往返、独立 checkpoint schema、旧 RationalCodec/FormTransformer 相关回归、WebUI Python 语法和前端 JavaScript 语法。没有启动训练、真实生成或 GPU 任务。

## 关键合同

音频采用已有的有理查询合同 `rational_audio_queries_v1`，输入宽度 120；事件采用 `RationalCodec` 的 `AT/P/N/END` 精确 Fraction 表示。`event_v1` 的 EventVocabulary、`song_form_v1/v2` 的 checkpoint 都不与它共享权重或词表。

流记忆的三个层次由首版接口预留：当前 decoder prefix、近期 history token、`stream_memory` 摘要。下一阶段需要把每个已提交事件的 Sustain/Slide 状态和音频位置索引加入 memory 更新，而不是只保存 token。

## 尚未完成且必须在下一阶段处理

1. 连续生成器的终止条件：首版生成原型需要增加“到达真实音频尾部后的终止候选”，并避免在 `audio_end` 之前因资源边界提前 EOS。
2. 持续动作状态：接入 `SustainStateIndex` 的增量等价状态更新，提交事件后重建候选状态。
3. 动态注意力：在远期检索、相似素材、近期详细写法和当前音频之间学习门控；呼应不是硬复制，相关性不足时应允许创新。
4. Planner/曲线：把 Planner 改成全曲趋势与局部计划条件，不能复用旧分段起音骨架语义。曲线应指导强度分布，不形成响度硬阈值。
5. 配置生成器：当前首版仍以字符级事件输出为主；下一阶段应增加事件内结构摘要，确保多押、Hold、Slide 组合作为完整配置监督，并支持单事件候选前瞻。
6. 训练数据：当前 chunk continuation 目标已经避免 chunk EOS，但计划标签仍是首版弱标签；需要按真实 event index、谱师选音和持续状态生成完整监督。
7. WebUI：新增控件已经接入 payload，但 checkpoint inspect 的 chart-stream 专属展示、曲线来源和生成报告摘要仍需完善。

## 研究依据与边界

Transformer-XL 的 segment-level recurrence、Music Transformer 的相对时间建模、Museformer 的细粒度/粗粒度注意力支持长程音乐上下文的可行性。它们不证明舞萌谱面质量会自动提升，首版必须以固定歌曲的边界连续性、静音区、节奏稳定性、前后呼应和 MiaCode 检查进行后续对照。

## 兼容策略

旧 event、song_form_v1/v2 checkpoint 继续由原路径加载。`chart_stream_v1` 只能加载自己的 checkpoint schema 和输入合同；WebUI 通过独立 `/api/chart-stream/*` 路由，避免把 stream 参数静默传给旧 form CLI。
