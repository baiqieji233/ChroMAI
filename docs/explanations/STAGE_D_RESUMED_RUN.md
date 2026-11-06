# 阶段 D：监控修复后的 full 续训与留出评估

更新：2026-10-11。训练和评估仍在运行，本文件只记录已核实的迁移与当前证据，不代表谱面质量验收完成。

## 当前有效运行：服务退出后的恢复

2026-10-11 00:50 核对时，原 7863 服务、训练进程和控制器均已退出，日志停在 00:40:27 的第 8 轮中途；退出原因未从这些日志中确定。最后完整保存的是第 7 轮，不能把未保存的第 8 轮当成已完成。

已重新启动 7863，并在 00:54:50 通过 WebUI `/api/train` 恢复：

- 当前训练任务：`94c977ca7b46`
- 起点：`checkpoints/music_structure_v3_full_stage1/event_epoch_007.pt`
- 追加 8 轮，目标第 15 轮；优化器、scheduler、AMP scaler 保留，`reset_optimizer=false`
- 已保存 global step 为 9106；validation mask schedule 保持 `fixed_seed_0_v1`
- 当前训练日志：`webui_workspace/logs/20261011_005450_train_94c977ca7b46.log`
- 当前队列配置：`.scratch/full_v3_stage1/pipeline_config_recovered.json`
- 当前持久状态：`.scratch/full_v3_stage1/pipeline_state_recovered.json`
- 当前队列报告：`.scratch/full_v3_stage1/PIPELINE_REPORT_RECOVERED.md`

原 v2 队列尚未提交任何后续任务，已连同配置/报告备份到 `.scratch/full_v3_stage1/recovery_20261011`。新的队列只观察当前训练，完成后仍执行既定 5 轮 finetune 和 70 次生成，不重复启动已运行的训练。请求、响应和预检也在该恢复目录中。00:58:29 实际日志记录 `optimizer_attempted=48 optimizer_successful=48`，确认已执行第 8 轮参数更新。

第 7 轮历史指标：训练 token loss `0.746608`、验证 token loss `0.783475`、fine-rhythm MAE `0.1881`；1301 次更新尝试中成功 1300 次，AMP 跳过 1 次。一次跳步不等于整轮失败；新计数如实保留该区别。

## 曲线模型身份误报修复

用户最新曲线 `2e5af81afe6b_automatic.json` 绑定的模型路径、文件大小和实际修改时间仍一致，因此此前仅解释为 `event_last.pt` 被覆盖并不完整。真实原因之一是 Python 纳秒时间戳 `1791649885336194900` 经 JavaScript Number/JSON 往返变成 `1791649885336194800`，原字典严格比较误判为换模型。

提交 `4e9ad5f` 已用 checkpoint SHA-256 与大小校验新曲线，并把纳秒值保存为字符串；旧曲线仅兼容同一个时间戳的浮点往返误差，实际改换/覆盖 checkpoint 仍拒绝。曲线预测期间文件变化也会中止。49 项定向测试通过。

恢复的 7863 服务已加载修复；真实旧曲线经 JavaScript 序列化发送后通过模型身份检查，同时故意改变 `first` 仍被后续条件/节拍校验拒绝，未提交生成任务。证据：`recovery_20261011/curve_identity_verified.json`。其它已运行的 WebUI 进程未被重启，未必已加载这项 Python 修复；可直接使用当前 7863。

训练时 `event_last.pt` / `event_best.pt` 会继续被后续轮次更新。若需要边训练边编辑模型曲线，应选定 `event_epoch_NNN.pt` 并用它重新生成曲线，避免真实的权重变化令编辑中的曲线失效。这与是否使用最新模型架构是两个不同问题。

## 续训迁移

以下保留首次从第 4 轮续训的历史记录；当前运行以本文件最上方的恢复信息为准。

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

00:17:16，第 6 轮训练结束并保存，训练 token loss `0.789431`、验证 token loss `0.792574`；1301 次更新全部成功。验证 fine-rhythm MAE 为 `0.1725`，比第 5 轮回升，即 token loss 降低不保证细碎目标每轮同步改善。随后进入第 7 轮。

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

## 规划器诊断：输入条件与回归指标

`tools/evaluate_planner.py` 原版使用参考谱的真实 radar/style 条件，属于“已经知道该谱整体偏好”的诊断，不能拿它直接证明模型面对新歌时能自主选对配置。新增默认 `generation_defaults` 模式：radar/style 来自固定生成默认值；仅保留已知目标定数、难度编号，以及参考谱提供的 BPM/first 时间轴。`--condition-source authored` 保留原诊断口径并显式注明限制。用户也可用 `--level`/`--difficulty` 测试固定要求；这并不代表所有 WebUI 预设。

由于 continuous planner 的训练目标使用 MSE，工具同时报告 MSE、MAE、常数预测基线及正目标子集。样本均值/中位数基线使用验收样本标签，只是诊断参照。不能仅用“未战胜全零预测的 MAE”断言规划器毫无学习。

相同 128 个固定验证窗口、匹配各谱目标定数、CPU 推理的结果：

| fine-rhythm 指标 | epoch 1 | epoch 4 |
|---|---:|---:|
| MSE | 0.07404 | 0.05643 |
| 相关系数 | 0.218 | 0.428 |
| 预测标准差 | 0.0151 | 0.0575 |
| 51 个正目标窗口上的预测均值 | 0.0656 | 0.1637 |
| 51 个正目标窗口上的目标均值 | 0.4153 | 0.4153 |

本样本全零预测 MSE 为 `0.09203`，样本均值常数 MSE 为 `0.06465`。第 4 轮已学到一部分可区分的信息，但对需要细碎节奏的窗口仍明显低估；它也不能独立代表实际 decoder 的采音选择。此处 fine 仍定义为相邻起音间隔 `<=33/128` 拍，并不覆盖所有复杂节奏。

完整证据：`.scratch/full_v3_stage1/planner_epoch1_vs4_generation_defaults.json`。原 authored 诊断和固定定数 13.3 的实验分别保留，没有混成同口径结果。下一步按已运行队列完成全量训练、细碎窗口增采样 finetune 和固定音频生成，检查这种低估是否减轻。

## 当前边界

验证报告会分别记录边界截断、空白段、密度、细分节奏、配置重复、类型比例、音频对齐和静态冲突。静态判定区检查不能替代实机 Perfect 判定或玩家双手试玩；在生成和人工审阅完成前，不把 loss、token accuracy 或测试通过数当作可玩性结论。

