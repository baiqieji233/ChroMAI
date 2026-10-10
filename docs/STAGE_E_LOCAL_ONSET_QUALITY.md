# 阶段 E：从“生成更多音符”转向“选择合适的落音位置”

更新：2026-10-11 06:50。阶段 D 的 full 基线已完成 15 轮，5 轮增采样微调和 70 份完整生成均已完成。局部采音分支已开始 3 轮训练（任务 `4d16b78f880f`），正式质量对照仍在继续，不能把小样本 checkpoint 作为可用成品。

## 目标与判断依据

目标是能生成合理、贴合音乐且配置有变化的舞萌 DX 谱面。需要同时检查采音位置、难度走势、留白、音符类型、重复配置及判定区冲突。参考谱并非唯一正确答案，静态检查也不能代替实机 Perfect 与双手调度验证。

当前 `music_structure_v3` 的 planner 预测窗口统计，并未逐个决定哪里落音。其中细碎目标是相邻起音间隔不超过 `33/128` 拍的比例；不能用它概括所有半拍、三分之一拍、附点与多声部节奏。增加细碎窗口采样或提高生成偏好，可能首先增加密度。

## 已取得的实际证据

**历史条件口径说明（04:00 补充）：** 以下早期自由生成实验使用的精简 API 请求，实际从离线 profile 读到了 `melody_following=0.0`。该字段当时为未测量占位，而 WebUI 完整参数和 token 诊断默认是 0.9。组内比较仍保持同条件，但不能混作与修复后默认生成相同的实验；修复后同条件重跑结果见本页最后部分。

### 正确历史条件下的局部音频诊断

不可变 epoch 7、相同 128 个验证窗口，给 decoder 提供正确谱面历史，使用生成默认条件加已知定数/难度。打乱局部声学和音频节奏特征、保留时间列与全局音频后，token accuracy 从 `0.743866` 变为 `0.743061`。

这说明该诊断口径下局部时序对整体 token 指标的贡献较弱，不能推导为模型完全忽略音频。词表本身能够表示三分音符；`1/3` 间隔正确 112/134，`1/6` 正确 64/87，而更短间隔样本很少。字符标记准确率也不能当作完整 Slide/Hold 类型 F1。

进一步用相同 128 窗、16,140 个受监督 token 比较 epoch 14：整体准确率 `0.73996`；半拍 gap 准确率 `0.8573 → 0.8757`，四分之一拍 gap `0.7653 → 0.6735`，六分之一拍 gap `0.7356 → 0.6552`。音频打乱的整体准确率差从 `0.0008` 增至 `0.0084`，但这没有抵消细分上的退化。不能通过追加轮数保证所需节奏能力；正式微调已核实从 epoch 7 的最佳权重开始。新诊断保存为 `event_tokens_epoch14.json`。

证据：`.scratch/full_v3_stage1/event_tokens_epoch7.json`；工具说明：[EVENT_TOKEN_DIAGNOSTICS.md](EVENT_TOKEN_DIAGNOSTICS.md)。

### 完整歌曲参数对照

歌曲 Ether Strike、epoch 7、同一种子 `20261919`，完整音频和同一时钟：

| 生成设置 | note 数 | 独立起音数 | 细碎比例 | 参考起音 F1 | 最长顺转 / 逆转 / 两键交替 |
|---|---:|---:|---:|---:|---:|
| 中性偏好 0.5，温度 0.65 | 942 | 760 | 0.4071 | 0.5486 | 11 / 19 / 17 |
| 细碎偏好 0.85，温度 0.65 | 1099 | 921 | 0.7533 | 0.5627 | 25 / 32 / 12 |
| 中性偏好 0.5，温度 0.9 | 897 | 746 | 0.4483 | 0.5543 | 8 / 5 / 8 |
| 参考谱 | 747 | 618 | — | — | 16 / 8 / 4 |

细碎偏好提高后，参考起音召回上升，但精确率降至 `0.470`，同时出现更长的环绕段。这支持改进局部采音监督，而非继续统一加密。单歌单种子不能用来给参数作最终排名；参考谱也有合理的重复段。

空间重复统计只识别单 Tap/Break 的相邻环绕和两键交替，和弦、其它类型及超过一拍的间隔会中断，不推断左右手。三个生成谱文件外音符均为 0，但文件内休止与歌曲语义仍需评估。

证据：`.scratch/full_v3_stage1/quality_probe_epoch7/evaluation_spatial.json`。

## 本阶段实现

1. **新增可选局部起音头 `chart_onsets_v1`。** 从音频 encoder 的逐帧表示预测谱师选择的音符起点，同拍和弦合并为一个起点。监督采用真实 tempo map、`audio_first` 和共用音频采样网格，20 ms 高斯软标签，排除 margin、padding、文件外及没有正向质量权重的区域。它没有把“三连音”作为必须追求的类别。
2. **保持旧模型生成。** 原 planner 的维度和目标不变。旧模型仅通过显式 finetune 初始化新头；resume 严格核对合同。有起音头的模型使用逐样本池化，防止批内长样本改变短样本的时间对齐；旧模型保留原策略。
3. **生成增加独立软指导。** `local_onset_bias_strength` 为 0–1，默认 0，只在合法 `AT:` 位置间调整分配；top-p 之前保留位置组总概率，已有休止和密度屏蔽优先。由于后续历史与采样会变化，不能声称最终 note 数完全不变。
4. **WebUI 识别能力。** 训练页可选起音监督及损失权重；生成页按 checkpoint 是否确实训练过新头启用滑块。切换回旧模型时有效值归零；旧服务未加载能力时页面解释原因。新功能服务为 `http://127.0.0.1:7864`，原训练服务 7863 保留运行。
5. **补齐评估。** 起音头可在 CPU 独立评估 BCE、Brier、平均精度及固定阈值 P/R，并对照常数基线与局部音频打乱。生成评估增加空间重复与真实时间位置，队列核查产物是否确实由本次任务生成，避免旧文件或缺失输出被计作成功。

完整接口与限制见 [LOCAL_ONSET_GUIDANCE.md](LOCAL_ONSET_GUIDANCE.md)、[ONSET_ALIGNMENT_EVALUATION.md](ONSET_ALIGNMENT_EVALUATION.md)。

新增 [可读审阅报告工具](RHYTHM_REVIEW_REPORT.md)，可以把评估 JSON 整理成逐曲摘要，包含上述空间重复发生的具体秒数。三组 epoch 7 结果及 epoch 7/14 对照各已生成 `QUALITY_REVIEW.md`，位于各自的 `.scratch/full_v3_stage1/quality_probe_epoch7/` 与 `quality_probe_epoch14/` 目录。

## 已完成的工程验证

从原 full epoch 7 初始化完整大小模型，在独立三谱训练/验证清单上执行 CPU 更新，真实 WebUI 生成和续训已跑通。新头增加 51,521 个参数，联测模型总参数 25,762,745。浏览器检查覆盖新头启用、旧模型禁用、请求值及 JavaScript 错误。

联测 checkpoint 只有 3 次起音头更新，16 窗评估 AP `0.249`，正例比例 `0.247`；打乱局部音频后 AP `0.243`。尚无明确局部对齐优势。该结果只验证链路和指标，不支持任何质量改善结论。

同一 checkpoint、相同 16 个验证窗补做 `time_only` 诊断：保留 10 列纯时钟特征，同时将 local/global 的 98 列声学特征和 4 列 audio-derived rhythm 置零。真实输入 AP `0.24919`、Brier `0.18787`；shuffle-acoustic AP `0.24268`；time-only AP `0.24321`、Brier `0.18767`。time-only 的 Brier 略好于真实输入，且两个模型 arm 都远差于同评估窗 oracle 常数 Brier `0.12038`。这组 16 窗、3 optimizer-step、小 checkpoint 的结果没有显示可靠的局部音频内容优势；zeroing 是分布外干预，只用于暴露时钟先验可能性，不是因果估计或部署对照。

报告：`.scratch/full_v3_stage1/onset_integration/frame_evaluation_time_only.json`。原 `frame_evaluation.json` 保留不变。

工程产物：`.scratch/full_v3_stage1/onset_integration/`。相关数据、池化、训练损失、旧模型兼容、生成软指导及 WebUI 定向测试已通过；各提交记录具体范围，没有将测试数量当成音乐质量指标。

## 正式实验与继续条件

02:56:11 基线任务 `94c977ca7b46` 正常退出（exit code 0），第 15 轮训练 token loss `0.533731`、验证 token loss `1.005618`，本轮成功优化器更新 1301/1301。验证表现已弱于第 7 轮，因此后续任务使用验证最佳 `event_best.pt`，不把末轮当最佳。微调任务 `8ee5bcb50f6d` 已由原控制器自动提交，没有重复启动。

按顺序执行，避免占用同一 GPU 的训练与生成相互干扰：

1. full 基线训练至 15 轮，再从验证最佳权重进行 5 轮细碎窗口增采样 finetune；10 首音频哈希与训练集分离的歌曲，7 组共 70 次完整生成。
2. 在前一队列所有任务及产物检查完成后，从不可变的 **full 基线 epoch 7** 独立启动 3 轮完整训练集起音头微调：LR `2e-5`、起音损失权重 `0.2`、细碎/Slide/Touch 采样倍率均为 **1**。这是提交任何新头训练前完成的调整，原因见下方实际对照。
3. 同一不可变 epoch 3 checkpoint，以指导强度 0 / 0.3 / 0.6 对上述 10 首歌做 30 次完整生成。比较音符密度、起音 P/R、各间隔比例、类型、空白段、空间重复及静态冲突。起音头另做同窗音频打乱诊断。

主队列：`.scratch/full_v3_stage1/pipeline_config_recovered.json`，已完成 71 个任务（5轮微调 + 70份生成）及评估；后续当前有效配置：`.scratch/full_v3_stage1/onset_pipeline_config_v2.json`。原 `onset_pipeline_state.json` 已明确标记 `superseded`，其配置/状态/报告备份在 `onset_revision_before_submission/`，当时提交任务数为 0。后续队列绑定主队列配置摘要，网络或状态观察失败不等于训练已经停止，不能据此重复提交任务。

同一模型的 0 / 0.3 / 0.6 对照用于判断指导力度；它与前阶段模型的比较还混入额外训练因素，不能把全部变化归功于指导。留出集合是开发验证集，已经用于调参，不是新的盲测集。起音头也不补全复杂 Slide/Wifi 的运行时判定或人体动作模型。

此外，新头微调同时改用 `per_sample_v2` 池化。旧策略在批内任一窗口超过长度上限时，会把短窗口也扩展到上限；新策略仅压缩实际超长窗口。该修复使训练和单窗生成的时间网格一致，但也是跨阶段比较中的额外变化因素。起音头还可能借助显式拍相位学习常见拍位先验，因此正式诊断须同时考察音频扰动与仅保留时钟输入的结果。

后续起音队列控制器已通过主队列完整状态门控，当前训练任务为 `4d16b78f880f`，状态 running。当前持久状态为 `.scratch/full_v3_stage1/onset_pipeline_state_v2.json`；阶段记录为 `ONSET_PIPELINE_REPORT_V2.md`。起音头完成后，30份强度 0/0.3/0.6 对照才会提交。

另做了相同 Ether Strike、相同 seed `20261919`、相同生成参数的 epoch 7/14 对照：epoch 14 为 827 notes、639 起音、细碎比例 `0.1897`、参考起音 F1 `0.5457`；epoch 7 为 942 notes、760 起音、细碎比例 `0.4071`、F1 `0.5486`。这只是单曲单 seed，但显示追加训练让该条件下的输出更保守，F1 没有实质改善。epoch 14 产物和状态在 `.scratch/full_v3_stage1/quality_probe_epoch14/`。

## 增采样微调第 1 轮：结果与分支调整

03:22:26，第 1 轮完成，成功更新 1985/1985，训练 loss `0.694190`，验证 loss `0.862331`（起点 epoch 7 为 `0.783475`），验证 fine-rhythm MAE `0.175585`（起点 `0.1881`）。同样的 128 个验证窗口、16,140 个受监督 token、batch 4 和 seed 1729，token accuracy 从 `0.743866` 降到 `0.719021`；半拍、四分之一拍、六分之一拍准确率分别从 `0.8573 / 0.7653 / 0.7356` 降为 `0.7983 / 0.6207 / 0.5977`。Hold 字符准确率从 `0.2784` 升为 `0.4431`；它不是完整音符类型 F1。

固定歌曲的自由生成也发生变化：第 1 轮微调为 **577 notes、479 起音、细碎比例 0.1632、参考起音 F1 0.5287**，起点为 **942 / 760 / 0.4071 / 0.5486**。微调谱有 58 Hold、53 Slide、14 Touch，长转圈减少，文件外音符为 0；这些局部变化没有证明整体设计更好。证据为 `event_tokens_finetune_epoch1.json` 和 `quality_probe_finetune_epoch1/evaluation.json`（同目录有 `QUALITY_REVIEW.md`）。

这不足以否定之后全部微调轮次，但已说明不能直接假设增采样得到更好的起点。因此保留原 5 轮实验及 70 份生成，**新头实验从 epoch 7 单独分支，并恢复均衡采样**，避免叠加尚未证实有效的分布改变。新头实验不是“在已改善的模型上进一步升级”，仍需用同模型指导强度 0/0.3/0.6 与基线结果独立验收。

## 验收链路补齐

- 队列评估 JSON 写盘后自动生成同 stem 的 Markdown 审阅报告；报告渲染失败会保留 JSON、记录错误并标为部分完成，不会虚报整轮成功。
- 审阅报告读取真实 expected/produced/missing/unverifiable 产物计数；关闭静态检测或缺少音频长度的检查显示未测量。音频哈希配对与仅文件名配对的指标分开统计。
- 新增 [CPU 诊断观察器](TRAINING_DIAGNOSTICS.md)。已启动 `finetune_diagnostics_config.json` 与 `onset_diagnostics_config.json`，分别等待原 5 轮任务和新的 3 轮起音任务完成，再自动对 128 窗做 token/shuffle 或 onset/shuffle/time-only 诊断。观察器只读训练状态，不提交训练或生成；正式结果路径为 `finetune_formal_diagnostics.json`、`onset_formal_diagnostics.json`。不可变权重在读取前后核对 SHA-256。
- 观察器已用真实完成的工程任务快照与 epoch 2 权重完成 8 窗端到端回放，证据保存在 `diagnostics_integration/`。此回放没有新增训练，不能当作正式质量结果。
- 合法但未覆盖的复杂 Slide 不会仅因 `partial` 被判为 hard 或强行拒绝。新增回归确认它与来源明确的独立 hard 冲突可以区别处理；导出成功与可玩性验证是两回事。旧“Goal 达成报告”已标为历史结论。

本轮相关组合回归 84 项通过，诊断观察器 29 项通过；这些只确认代码行为。当前尚未达到“最终说明/质量验收完成”的条件。

## 条件值传递与时间轴修复

发现并修复了两类条件错误：离线 profile 的“节奏贴合”没有音频测量，其零占位却被当作默认要求；无 profile 的定数-only 条件文件又会把其余值重置为数字 null 0.5。现在缺失或未测量的控制使用生成默认值；显式填写的零、嵌套 target/value 等仍优先，最后执行 checkpoint 条件合同保护。旧 profile 原始 JSON 保持不变，以兼容规则解释未测量字段。详见 [CONDITION_PROFILE.md](CONDITION_PROFILE.md)。

同歌曲、同 seed、同模型，用修正后的实际请求条件（节奏贴合 0.9，其余 profile 中位数）重新完成 CPU 全曲生成：

| 模型 | note 数 | 起音数 | 细碎比例 | 参考起音 P / R / F1 |
|---|---:|---:|---:|---:|
| full epoch 7 | 848 | 655 | 0.2615 | 0.5237 / 0.5550 / 0.5389 |
| 增采样微调 epoch 1 | 468 | 377 | 0.0239 | 0.6631 / 0.4045 / 0.5025 |

修复让请求符合默认语义，不代表指标必然提高。首轮微调在该例上仍明显偏保守，保留原 5 轮比较和独立起音头分支的决定。新证据：`quality_probe_condition_fix/evaluation.json`、`QUALITY_REVIEW.md`，任务 `166abb9bec9b` / `581a9da001f7`，均完整成功。原 probe 产物没有覆盖。

另对同歌 8 个验证窗口，用修正后 profile、明确 `condition_null=false` 做 planner 诊断：预测/注入真实目标时的位置准确率分别为 epoch 7 `0.572/0.641`、微调 `0.569/0.555`。真实目标没有稳定提升，因此尚不足以把稀疏退化单独归因于 `plan_teacher_mix`。该设置实际是每个有效样本的连续插值，不是随机一半样本，也不会随 epoch 自动减小；WebUI 说明已更正。可复现脚本与 JSON 为 `planner_condition_probe_epoch7_vs_finetune1.*`。

同时确认训练侧节奏贴合标签漏用 `chart.audio_first`：当前清单 1695 个高难度可用条目中，39 个偏移非零，13 个超过 80 ms 容差，最大绝对偏移 1.229 秒。这只影响音频起音匹配这一条件维度，未改变原 token、planner/onset 标签或输入音频。修复以独立 `condition_target_version` 记录：旧 checkpoint resume 沿用 `legacy_unshifted_v1`；fresh/finetune 使用 `chart_audio_clock_v2`，记录迁移，新版 resume 严格继承。三个诊断工具也继承版本。新起音分支将自动使用此修复，它是跨阶段对照中另一项明确记录的变化。

版本化修复 113 项定向测试通过；真实 WebUI CPU 迁移任务 `6bb13078e118` 与续训任务 `a97fcf1958ea` 均完成，checkpoint 记录从旧目标迁移到新目标，并在 resume 后保留。证据在 `condition_clock_integration/`，只完成各 1 次更新，是工程联测而非质量模型。正在运行的 5 轮微调进程继续原语义，没有被中途替换。

模型曲线分析、生成请求预检与实际 CLI 也统一使用同一个条件解析器。48 项相关回归通过；空闲 7864 服务刷新后，真实精简 HTTP 请求完成曲线任务 `bba96b401976` 与生成任务 `6d7f36eaccb0`，两边的节奏贴合均为 0.9，错误定数的旧曲线请求被 HTTP 400 拒绝。证据为 `sparse_curve_http/verification.json` 及同目录曲线/谱面。7863 的训练进程未重启；常驻旧服务若使用这项新的曲线预检，应在训练结束后重启或改用 7864。

## MiaCode 运行时参考联测

固定 MiaCode commit `6573efb611056bccd046dd17bc2f4cceda78bf88` 的 headless `miacode_muri_dump` 已在 `.scratch/muri_build` 独立构建成功（Qt 6.8.3 Core/Gui/Widgets、VS 18；未改用户安装或运行中的 GUI）。fixture 运行验证了 parse、runtime 和 static 三类输出。

随后对同一首歌的三份 CPU 生成谱运行参考检测（其中两份使用修复后的条件，一份为旧条件历史实验），报告为 `.scratch/full_v3_stage1/quality_probe_condition_fix/miacode_reference.json`：

| 谱面 | MiaCode runtime diagnostics | static references | 说明 |
|---|---:|---:|---|
| full epoch 7 | 11（MultiTouch 8、SlideTooFast 2、TapOnSlide 1） | 6 | 参考模型发现了当前静态检测未覆盖的运行时规则 |
| finetune epoch 1（修复条件） | 3（MultiTouch 1、Overlap 2） | 2 | 诊断数量较少，但音符量也明显减少 |
| finetune epoch 1（旧条件 probe） | 3（MultiTouch 2、Overlap 1） | 2 | 仅用于记录旧条件实验，不能和修复条件混作同口径 |

这些诊断来自谱面派生的 autoplay-like 动作模型，不接受真实玩家轨迹，也不是 SEGA 官方判定规范或实机 Perfect 证明。它们被保留为 `engine_reference_only=true`、`playable_verified=false` 的 sidecar，不接入自动 hard/reject/local repair。当前静态检测报告和 MiaCode 参考结果会并列供后续人工审阅；单纯减少 diagnostic 数不能作为质量提升结论。

又用相同参考程序检测了该歌曲原谱和 epoch 7 的三组历史参数实验。原谱 runtime/static 均为 **0**；中性生成 runtime **15**、细碎偏好 runtime **21**、升温生成 runtime **50**（其中 MultiTouch 42）。因此之前“升温后长转圈减少”确实不能推断为可玩性改善。四份输入均无解析错误或警告，报告为 `quality_probe_epoch7/miacode_reference.json`。不同工具对 Slide 头/路径和辅助头的计数语义不同，不将 MiaCode `note_count` 与本项目总 note 数直接混算。

文件边界过滤另修正了 20 ms 容差泄漏：音符头现在严格位于 `[0, audio_duration)`；可听区间裁剪仍独立。定向边界/曲线回归 45 项通过。起音配对新增 20/35/50 ms 三档评估容差，避免快速细分的相邻拍位被宽容差掩盖；它们不是实机判定窗。修复条件后的基线/首轮微调，20 ms F1 为 `0.5295/0.5025`，与宽容差比较方向一致。证据 `quality_probe_condition_fix/evaluation_tight_timing.json`。参考驱动、节奏评估、审阅和边界组合回归 77 项通过。

04:28:49 第 4 轮微调保存：训练 loss `0.623158`，验证 loss `0.944571`，成功更新 1985/1985。第 2 轮验证 loss `0.814548` 仍为当前微调阶段最佳；第 3 轮为 `0.830302`。这些均未优于起点 epoch 7，因此不将微调阶段的 `event_best.pt` 当作整个项目的最佳质量模型。阶段 best 与跨阶段最佳是不同概念。第 5 轮仍在训练，尚未开始 70 份正式生成。

运行时参考检查已经接入 CLI、后端和 WebUI，默认仍为 `off`。重启空闲 7864 后，用 report 模式完成了真实 API 任务 `2277105c0d3c`；生成质量报告的 `runtime_reference.status=completed`、解析无错误、MiaCode 诊断 0、静态参考 1。旧服务实例没有被假定已经加载新代码，7864 的实际重启和任务日志验证了传参链路。

第 5 轮微调完成后，正式 CPU 诊断 `finetune_formal_diagnostics.json` 已保存 epoch 5：token accuracy `0.7403`，局部声学打乱后 `0.7382`；1/4 gap `0.7313`、1/6 gap `0.7701`。同歌同 seed 的 epoch 2/5 完整生成对照也完成：epoch 2 为 757 notes、20 ms 起音 F1 `0.5485`、MiaCode runtime diagnostics 8；epoch 5 为 886 notes、F1 `0.5678`、runtime diagnostics 28。两者都不能单独判优，epoch 5 的起音匹配更高但参考运行时问题也更多；详见 `quality_probe_finetune_epoch2_vs5/`。

继续依据实际证据调整：若新头只学固定拍栅格，应先改善音频时序学习；若准确落音但配置单调，应针对完整配置及历史依赖；若密度或留白恶化，应先收紧软指导和预算之间的相互影响。不能用更高细碎比例或更低报警数掩盖这些问题。
