# 拆分 Touch / Slide 采样后继续训练

此改动可重用现有 event_ctx_v1 权重，不需要从随机权重训练。更改采样分布时应选 WebUI 的“已有模型加强训练（finetune）”，命令行对应 `--init-checkpoint`。新阶段不恢复旧 optimizer、scheduler、AMP 和 RNG 状态；正常 `resume` 仅用于继续新阶段的同一份采样合同。

## 采样规则

| 窗口内容 | 采用倍率 |
| --- | --- |
| 没有 Slide / Touch | 1 |
| Slide，且没有 Touch / Touch Hold | `slide_oversample` |
| Touch 或 Touch Hold，且没有 Slide | `touch_oversample` |
| 同时有 Slide 与 Touch / Touch Hold | `touch_oversample` |

倍率不相乘、不相加。Touch 优先是为了让 `touch_oversample=1` 真正取消所有含 Touch 窗口的放大，即使窗口也含 Slide。1 表示保留自然频率，不能设成 0。

窗口分类根据解析后的 note_type，而不是原始文本中的字母。数据审计和启动日志分别记录 `sampling_windows`（复制前）和 `sampling_segments`（复制后），可以检查真实分布。

旧 checkpoint 中 `slide_oversample=3` 表示 Slide 或 Touch 窗口统一三倍，和新策略不是同一合同。程序只允许通过 finetune 迁移；即使新倍率填成 3/3，也不会把旧策略伪装成普通 resume。词表、输入特征、时序窗口、条件掩码合同继续严格校验。

## 推荐的第一阶段校正设置

| WebUI 项目 | 设置 |
| --- | --- |
| 模型族 | `event_ctx_v1` |
| 训练方式 | 已有模型加强训练（finetune） |
| 已有模型 checkpoint | 暂停原训练后，使用已完整保存的 `checkpoints/event_v10/event_last.pt`；可先另存一份固定源文件 |
| checkpoint 输出目录 | 新目录，例如 `checkpoints/event_v10_touch_rebalance` |
| 训练 manifest | `webui_workspace/datasets/full训练集/pretrain_manifest.json` |
| 验证 manifest | 保持原 `gold_val_manifest.json`，仅作为偏向性参考；不改训练数据为 gold 微调集 |
| 学习率 | `0.00003`（3e-5） |
| 额外 epochs | 先 3 轮；看每轮固定音频生成，再决定后续时长 |
| 保存间隔 | 1 |
| 早停耐心 | 0（该短阶段不由偏向性 gold loss 提前停止） |
| Slide 窗口过采样 | 2 |
| Touch 窗口过采样 | 1 |
| batch size / 梯度累积 | 8 / 4 |
| segment beats / max tokens | 16 / 1024 |
| 音频最大 token | 1024 |
| context beats / audio margin beats | 16 / 8 |
| context tokens / overlap beats | 320 / 0 |
| context dropout | 0.15 |
| condition mask prob / condition default | 0.5 / 0.5 |
| condition dropout | 0.15（该源 checkpoint 的实际值，并非新训练默认 0） |
| Planner loss weight | 0.1 |
| 模型结构 | 继承源权重：d_model 320、8 头、8 层、FFN 1280、dropout 0.12 |
| weight decay | 0.01 |
| 难度槽 / 定数范围 | 4–6 / 12–15 |
| 训练 batch 上限 / 验证 batch 上限 | 0 / 0（不截断） |
| 音频缓存 / CUDA 显存预算 | 6 GiB / 0（沿用自动预算） |

这些是校正实验的起点，不能保证三轮后 Touch 一定落在正确位置。若降低 Touch 后 Slide 也明显减少，要检查 mixed 窗口比例和实际生成，而不是继续放大两种倍率。

## WebUI 操作顺序

1. 由用户暂停原训练，确认最后一个完整 epoch 和 `event_last.pt` 已保存。暂停可能丢弃当前未完成轮的更新；没有写出的 epoch 不能恢复。
2. 重启你实际使用的 WebUI 服务（7860 或 7861）。仅 Ctrl+F5 不会加载新的 Python 服务端。新前端会通过服务端能力字段检查是否支持独立 Touch，发现旧服务时拦住训练提交并提醒重启，避免参数被静默忽略。
3. 选 `event_ctx_v1` 和 finetune，选择固定源 checkpoint，点击“检查 checkpoint”。模型结构、窗口布局、条件合同会回填；finetune 保留你填写的新采样倍率。
4. 填新输出目录、full 的 pretrain manifest，以及上表参数。重新核对条件随机丢弃为 0.15，Slide=2，Touch=1。
5. 启动后检查日志包含 `mode=finetune source_epoch=... optimizer_reset=true` 和 `slide_oversample=2 touch_oversample=1`。训练和验证的数据审计分别显示真实倍率；验证侧使用 1/1。
6. 新阶段日志 epoch 从 1 开始，这只是阶段编号，权重继承源模型。checkpoint 的 `training_origin` 记录源路径和源 epoch。后续续跑此阶段可使用其新 checkpoint 及同一合同选 resume。

最新检查时（2026-10-08 18:00:59），`event_last.pt` 已是第 13 轮。原任务每两轮保存一次编号文件，因此目录没有 `event_epoch_011.pt`。如果必须使用第 11 轮，应选择此前另存的第 11 轮文件；否则选择暂停后最后完整保存的权重，不要凭文件名猜 epoch。

full 数据只读核对：原合并三倍采样为 129,031 个训练窗口；Slide=2、Touch=1 后为 77,588 个窗口。原始有效窗口包括 slide_only=29,739、touch_only=2,897、mixed=7,955、plain=7,258。新策略只将 slide_only 复制到 59,478，其余三组保持原数量。窗口份额变化不是最终生成 note 比例保证。

## 生成评估

固定几首歌曲、seed、BPM、偏移和目标条件；保持 source checkpoint 和每轮新 checkpoint 独立输出。先用 sample 开启、temperature=0.65、top_p=0.9、repetition_penalty=1.05、repetition_window=56、guidance_scale=1 比较。不要用 1.8/1.9 这类高 token 重复惩罚来判断采样修复效果，它还会影响合法重复和音符类型分布。

看 Touch 在不合适位置的滥用是否减少，同时确认 Slide 没有消失、密度和空白没有退化。总体 Touch 比例下降不等于位置正确；仍需逐段人工审查。原 gold 集偏向性仍然存在，而且旧验证也复制了 Slide/Touch 窗口，新验证不再复制，因此新旧 validation loss 不应直接横向比较。
