# 阶段 D：监控修复后的 full 续训与留出评估

更新：2026-10-11。训练和评估仍在运行，本文件只记录已核实的迁移与当前证据，不代表谱面质量验收完成。

## 续训迁移

原始 full 训练任务 `d32a25d95f70` 在第 4 轮完整保存 `event_epoch_004.pt` 后被停止。停止前的第 4 轮指标为训练 token loss `0.900950`、验证 token loss `0.953507`、token accuracy `0.6899`、fine-rhythm MAE `0.1689`。没有使用未完成的第 5 轮状态。

为了让训练监控修复实际生效，续训提交到了新的 WebUI `http://127.0.0.1:7863`：

- 新任务：`b17f579bfd17`
- checkpoint：`checkpoints/music_structure_v3_full_stage1/event_epoch_004.pt`
- 训练参数中的 `epochs=11` 表示从第 4 轮追加 11 轮，训练循环目标为第 15 轮
- 使用 `--resume`，保留 checkpoint 的优化器、scheduler、best 和 global step 状态
- 日志：`webui_workspace/logs/20261010_234258_train_b17f579bfd17.log`

训练启动时数据集仍报告 1,681 个可用难度、41,607 个训练窗口；验证集最终为 177 个难度、4,431 个上下文窗口。基础分段阶段还单独记录了 7 个超出词表的窗口，不能把基础分段的 4,441 误作最终上下文窗口数。

2026-10-11 00:02:35，完整 `event_epoch_005.pt` 已保存。新监控记录：

| 指标 | epoch 5 |
|---|---:|
| 训练 token loss | 0.841458 |
| 训练完整 objective loss | 0.873683 |
| 验证 token loss | 0.842492 |
| 训练 token accuracy | 0.7112 |
| 验证 fine-rhythm MAE | 0.1621 |
| 成功优化器更新 / 尝试更新 | 1301 / 1301 |
| AMP 跳过更新 | 0 |
| 峰值显存 | 6.99 GiB |

这确认修复后的训练和验证均已实际执行。验证掩码开始使用固定种子 0；旧轮次使用随机掩码，因此不能把第 4→5 轮验证指标变化完全归因于模型进步。best/scheduler 的选择仍保留 token loss 语义。

## 评估队列

新的串行控制器使用：

- 配置：`.scratch/full_v3_stage1/pipeline_config_v2.json`
- 状态：`.scratch/full_v3_stage1/pipeline_state_v2.json`
- 报告：`.scratch/full_v3_stage1/PIPELINE_REPORT_V2.md`
- base URL：`http://127.0.0.1:7863`

队列会先等待第 15 轮 full 续训完成，再提交独立目录的 5 轮 finetune，随后对 10 首按音频 SHA-256 与训练集完全分离的验证歌曲运行 7 组、共 70 次完整生成。每组保留固定 seed、first、BPM/tempo map 和条件，比较旧 v2、full v3、finetune、偏好 0.25/0.75、冲突修复加 2 次重试和关闭检测等配置。旧 v2 可能见过这些歌曲，因此这不是同条件的数据泛化排名。

原控制器先于原训练停止，迁移时后续任务均尚未提交。原配置/状态/报告保存在 `.scratch/full_v3_stage1/legacy_queue_before_monitoring_resume`；新请求、响应和迁移信息分别为 `resume_training_request.json`、`resume_training_response.json`、`resume_transition.json`，原模型权重仍保留。

## 同期修复与界面验证

- 命令行与 WebUI 统一核对模型曲线的 checkpoint、条件、first 和 tempo 设置，避免同一首歌套用不同模型/定数或不同拍位体系的曲线。
- 新启动的 WebUI 后端会按“checkpoint 已有轮数 + 请求追加轮数”计算续训目标，并区分 fresh/finetune 的阶段轮数。当前 7863 的 Python 服务早于这项显示修复；保留运行避免再次打断训练。静态页面刷新即可加载旧服务的兼容提示，不会据缺失元数据虚构目标轮数。
- 真实浏览器在 7863 使用歌曲 `12052` 的完整 170.30 秒音频与 14 个 tempo 变更，完成模型曲线分析、点击“增强强弱对比”、保存并提交 CPU 生成。87 个曲线点、tempo map 指纹及编辑状态从界面完整传到请求，无 JavaScript 页面异常。证据在 `.scratch/full_v3_stage1/tempo_roundtrip`。

## 完整变 BPM 联测结果

2026-10-11 00:07:13，生成任务 `2cfccb224743` 完成。使用不可变的 `event_epoch_004.pt`，生成参数由真实浏览器界面收集；此处为链路联测，未计入训练结束后的 70 份正式对照。

- `maidata.txt` 导出后重新解析，与 `.chart.json` 的全部 423 个可发音事件在拍位、音符文本上精确一致，14 个 tempo 变更一致。
- 一共 651 个 note；首音 1.3333 秒、末音 167.5658 秒；音频开始前和文件结束后 note 均为 0。末尾静音窗被跳过。
- 将已保存曲线对应的 `first` 改为 0.1 后，WebUI 拒绝请求并要求重建曲线，未启动错误时钟下的新生成。
- 选择的是“仅检测”，报告保留 1 处 A5 的 Hold/Tap 按下区间重叠；没有自动删除。生成状态为 `success_with_warnings`，可玩性仍为未验证。

不足也很明显：参考谱 641 个 note，与生成总数相近，但参考有 54 个 Slide 和 61 个 Hold，生成只有 4 个 Slide、20 个 Hold。Tap 比例为 96.3%，参考为 81.7%；参考 120 个 `1/2` 拍起音间隔，生成只有 8 个。两者 `<=33/128` 拍的 fine 指标均为 0，这也说明只看单一 fine 指标会遗漏本例的八分节奏简化。不能根据总 note 数接近或语法通过宣布音乐质量改善。

对应核验和对照见 `tempo_roundtrip/verification.json` 与 `evaluation.json`，参考音频 SHA-256 完全匹配。针对实际使用相对目录执行评估时暴露的路径错误，已统一规范化输入和参考目录，相关 11 项评估测试通过。

## 当前边界

验证报告会分别记录边界截断、空白段、密度、细分节奏、配置重复、类型比例、音频对齐和静态冲突。静态判定区检查不能替代实机 Perfect 判定或玩家双手试玩；在生成和人工审阅完成前，不把 loss、token accuracy 或测试通过数当作可玩性结论。

