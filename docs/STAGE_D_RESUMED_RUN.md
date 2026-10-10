# 阶段 D：监控修复后的 full 续训与留出评估

日期：2026-10-10。训练和评估仍在运行，本文件只记录已核实的迁移与当前证据，不代表谱面质量验收完成。

## 续训迁移

原始 full 训练任务 `d32a25d95f70` 在第 4 轮完整保存 `event_epoch_004.pt` 后被停止。停止前的第 4 轮指标为训练 token loss `0.900950`、验证 token loss `0.953507`、token accuracy `0.6899`、fine-rhythm MAE `0.1689`。没有使用未完成的第 5 轮状态。

为了让训练监控修复实际生效，续训提交到了新的 WebUI `http://127.0.0.1:7863`：

- 新任务：`b17f579bfd17`
- checkpoint：`checkpoints/music_structure_v3_full_stage1/event_epoch_004.pt`
- 训练参数中的 `epochs=11` 表示从第 4 轮追加 11 轮，训练循环目标为第 15 轮
- 使用 `--resume`，保留 checkpoint 的优化器、scheduler、best 和 global step 状态
- 日志：`webui_workspace/logs/20261010_234258_train_b17f579bfd17.log`

续训启动后已观察到新的每 60 秒进度行，例如 `optimizer_attempted=36 optimizer_successful=36`。这确认运行的是修复后的代码，并且目前没有 AMP 跳步迹象。训练启动时数据集仍报告 1,681 个可用难度、41,607 个训练窗口；验证集仍为 177 个难度、4,441 个窗口，7 个超出词表的窗口被单独跳过。

## 评估队列

新的串行控制器使用：

- 配置：`.scratch/full_v3_stage1/pipeline_config_v2.json`
- 状态：`.scratch/full_v3_stage1/pipeline_state_v2.json`
- 报告：`.scratch/full_v3_stage1/PIPELINE_REPORT_V2.md`
- base URL：`http://127.0.0.1:7863`

队列会先等待第 15 轮 full 续训完成，再提交独立目录的 5 轮 finetune，随后对 10 首按音频 SHA-256 与训练集完全分离的验证歌曲运行多组生成。每组保留固定 seed、first、BPM/tempo map 和条件，比较旧 v2、full v3、finetune、细碎节奏偏好、冲突修复重试和关闭检测等配置。旧 v2 可能见过这些歌曲，因此这不是同条件的数据泛化排名。

## 当前边界

验证报告会分别记录边界截断、空白段、密度、细分节奏、配置重复、类型比例、音频对齐和静态冲突。静态判定区检查不能替代实机 Perfect 判定或玩家双手试玩；在生成和人工审阅完成前，不把 loss、token accuracy 或测试通过数当作可玩性结论。

