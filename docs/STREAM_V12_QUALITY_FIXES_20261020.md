# chart_stream v12 质量修复交接

> 后续空白修复及 v13 有界损失合同见 `STREAM_BLANK_GAPS_HANDOFF_20261020.md`。本文记录 v12 当时的实施结果；迁移重置规则以新交接文档和当前代码为准。

日期：2026-10-20

分支：`codex/v11-quality-fixes-20261020`

原始工作区备份：`E:\simai_transformer_backups\v11_quality_before_20261020`

## 修复范围

本轮只处理已确认会影响训练合同或谱面质量的项目：

1. 旧 `plan_dim=8` 检查点迁移到 10 维规划头。
2. 同起点多 Slide 的 `*` 分支上限。
3. 长 Slide 在途运动学的历史覆盖和数组边界。
4. 事件边界的直接音频残差路径。
5. Planner 曲线的相邻变化监督。
6. phrase 特征按真实音频时间加权。
7. Planner 规划标签从绝对拍位相位改为对称不变的相对手部移动相位。

`curve.detach()` 保留不变。它仍表示 Planner 曲线由独立曲线目标监督，避免配置 token loss 把曲线头退化成任意隐藏状态投影。

## 关键实现

### 训练合同与迁移

训练语义升级为 `chart_stream_training_v12_relative_plan_geometry`。加载旧 8 维检查点时，loader 将 `plan_head` 和 `intent_projection` 扩展到 10 维，前 8 维保留，Break/EX 对应的新增维度置零。迁移结果会写入 `plan_dim_migration` 元数据；旧优化器状态不能直接 resume，必须使用 finetune 和新输出目录。

`_loss` 现在要求 plan 标签和输出形状完全一致，防止新增标签被静默截断。旧 v11 语义的
finetune 会把 plan 前三行重置，让绝对拍位相位权重不会污染新的相对运动标签。

### 音频依从性

Grounding router 仍可选择历史或创新路线，但事件边界新增 `audio_boundary_projection`。它以零权重初始化，仅在 `<bos>` / `<end>` 边界直接注入局部音频证据；之后的位置和配置 token 通过因果自注意力继承该证据。这样不会把每个音频瞬态强行变成 note，也不会改变旧检查点的初始函数。

### 曲线与 phrase

曲线损失新增低权重相邻差分监督，监督预测曲线的变化方向和幅度跟随原谱负荷，同时保留逐点 SmoothL1，避免单纯 TV 平滑抹掉真实峰值。权重由 WebUI 的“Planner 曲线变化监督权重”控制，默认 `0.05`。

phrase 窗口改用非均匀音频节点的实际时间积分，不再按节点数量平均；通过前缀积分实现，避免为每个 query 构造完整的 `B×Q×T` 权重张量。训练和生成共用同一实现。

### 规划标签

plan 的第 2、3 维现在表示当前事件相对前一事件的最短手部角位移的 `sin/cos`，取绝对角位移以保持左右镜像下标签一致。没有前一事件或无法解析实体按键时使用 `(0, 1)`。

### Slide 与语法

chart_stream 的 `_NoteGrammar` 使用 `max_star_slides=2`。它只限制同一 note cell 的 `*`
Slide 分支，普通 `/` 多押不受影响。旧 event/song_form 路径默认保持原 atom catalog 行为；
CLI 只有显式传入正数 `--max-star-slides` 才启用同样限制。

运动学特征不再固定只扫描最近 64 个事件；它会根据轨迹最长持续时间选择需要扫描的历史范围，并限制到已提交且数组实际存在的事件，避免长 Slide 消失、未来轨迹泄漏或数组越界。

## 验证边界

本轮不启动训练和整曲生成。修改完成后只运行定向 CPU/语法测试，重点覆盖：

- 8 维检查点迁移与 resume 拒绝；
- 10 维 plan 形状合同；
- phrase 非均匀时间加权；
- 曲线差分损失梯度和零权重行为；
- 事件边界音频残差的旧权重零初始化；
- 同起点 Slide 上限、普通 slash 多押、长 Slide 运动学和未来轨迹隔离。

## 当前验证结果

- chart_stream 核心定向测试：`257 passed`。
- WebUI Node 测试：`27 passed`。
- 额外 form/物理约束回归：`69 passed`。
- `py_compile`、`git diff --check` 通过；未启动训练、生成或 WebUI 重启。
