# 新架构 R2/R3 阶段报告

日期：2026-10-12。关联目标：[GOAL_SONG_FORM_REBUILD.md](GOAL_SONG_FORM_REBUILD.md)。

## 已经可以运行的链路

`maichart.form` 是独立的新模型族 `song_form_v1`，不加载旧事件权重，也不修改旧训练合同。

1. 全曲音频与结构描述进入 `SongBlueprint`，一次预测整首歌的谱面起音、宽泛类型和持续时长参考。音频使用时间卷积及全曲汇聚，避免对数千帧音频直接建立整曲平方级自注意力。
2. 起音概率峰映射到精确有理数拍位，生成一份全曲节奏草稿；候选网格兼容三等分、五等分及常见更细节奏。它是谱师起音选择模型，不是歌曲所有声音的检测器。
3. 配置模型读取局部声学特征、全曲区域关系、过去已完成谱面、相似段落参考、持续动作及未来计划起音。未来计划中没有提前提供按键/Slide 等真实配置。
4. 音频秒数和 token 预算只决定计算分块；布局沿同一份全曲草稿继续，预算耗尽时保留已完成事件，不把半个 note 写入结果，不重抽整段节奏。解码使用前缀缓存。
5. 最后可用有预算的整曲修复寻找少量可验证编辑；遇到早期长 Hold/Slide 来源时不只改当前窗口。每次修改都复检完整谱面，默认保留起音与语义音符数量，不靠删音获得低无理数。

音乐区域和计算块有独立身份映射；改变音频计算上限不应让对应副歌变成另一个参考段。所有自动关联仍属于候选证据，边界错误和检索失败都需要观察。

## “让模型学会改稿”的具体训练方式

配置模型共享条件编码，但区分两种任务：

- **草稿配置**：已知全曲起音骨架，读取已完成历史，学习当前位置的配置。当前/未来真实配置不能作为草稿输入。
- **遮蔽重建**：用可信谱面构造少量被遮蔽的 note 文本，保留起音与两侧上下文，学习恢复原本内容；混入干净样本，让模型也能选择不改。

训练以原谱起音作为教师草稿，实际生成以全曲模型预测草稿作为输入，仍存在暴露偏差。不能将当前重建损失解释为“平庸谱面已经能被自动改好”。审美改稿还需要真实采纳的前后版本/偏好标注或经校准的质量评价器；本轮没有伪造这种数据。

checkpoint 分开记录全曲规划更新次数以及草稿、修订任务有效样本数。只训练过其中一种任务的短跑模型不能被误认为两种都已学过。续训恢复优化器与随机状态；新阶段加强训练使用新输出目录保留源权重。

神经修订会保存未采纳的候选供人工查看，只有硬无理确实改善、解析与覆盖不退步、起音和音符数量得以保留时才自动替换。相同冲突数下没有可信质量改善证据，不宣称候选更优秀。

## 减少重写成本的实现

- 全曲草稿只生成一次，局部修复不反复决定整段采音。
- 原因候选优先考虑覆盖问题时刻的前置持续动作，再考虑直接参与冲突的当前 note。
- 改动是单 note 的位置/时值/简单路径候选，不进行全窗盲目重抽。
- 检测结果按整曲内容和检测策略指纹缓存；失败修改不重复跑。
- 限制检测次数、编辑数和时间，预算耗尽保留此前最佳版本和剩余问题。
- 单次检测无法被时间预算从内部强行打断，时间预算在调用前后检查；不会把“预计 10 秒”误称为严格实时上限。
- 原始/最终覆盖状态保留。解析不完整或未知覆盖时，零报警也不能称为已修好。

## 实际工程验证与不足

已经在隔离 CPU 上跑通新模型从头训练、加强训练、续训、全曲规划、精确解码、局部改稿和回滚。旧权重未被覆盖。

另外保存了一个 **111,842 参数、2 个 epoch、合成 6 秒音频**的工程夹具模型：

- `.scratch/form_cpu_smoke/model/form_last.pt`
- `.scratch/form_cpu_smoke/generated/maidata.txt`
- `.scratch/form_cpu_smoke/generated/maidata.form.json`
- `.scratch/form_cpu_smoke/revised/maidata.form.json`

其低阈值生成选择了 109 个起音，修复到 6 次检测预算后仍有 51 条参与处理的硬问题。这是有意保留的工程结果，说明“链路可运行”与“谱面质量达标”差距仍然很大；它不能用于正常出谱。

同一夹具上的神经改稿提议改变了语义音符数量，系统拒绝采纳并保留原谱。实际验证了保护规则，没有把“模型提出了不同写法”当成改进。

需要继续评估：全曲起音阈值校准、教师草稿与预测草稿的差异、从单纯重建转向真实编辑、相似区域对齐精度、参考强度与刻意变奏的可控性。当前类型/时值规划输出只作报告参考，布局解码主要受起音骨架约束，尚未充分消费这些类型参考。

当前新训练器保留简单的谱面权重与排除；局部正权重细分、旧版 DPO 全配置和部分旧生成曲线功能尚未迁入新路径，仍可在兼容工作台使用。超 token 预算的训练样本明确记录为跳过；后续应评估事件预算训练分块，不让高密度资料系统性流失。


## R4/R5 首次真实验证与 WebUI 最终核对（2026-10-12）

本轮把真实官方素材接入新路径，使用 `E:\后训练（官谱）\training_manifest.json` 的 16 张限量短训，得到：

- checkpoint：`checkpoints/song_form_v1_official16_epoch1/form_last.pt`
- epoch：1；blueprint steps：16；optimizer steps：567
- draft / revision 有效训练样本：577 / 556
- 模型族和输入合同：`song_form_v1` / `song_form_inputs_v1`

随后使用同一 checkpoint 对 `E:\后训练（官谱）\BUDDiES\10185` 的真实音频和 maidata 做了生成检查。命令成功退出，输出通过严格 simai 解析（0 errors、0 unknown tokens、coverage 1.0），生成统计约 1054 个 note（932 Tap、31 Hold、62 Slide、29 Touch、83 chords）。开启 6 次检查、2 次编辑和 10 秒修订预算后，硬冲突从 109 降到 107；报告仍为 `playable_verified=false`。这证明训练到生成的工程链路可用，不证明音乐性或实机可玩性。

WebUI 核对结果：

- `/form-lab` 明确显示模型族 `song_form_v1`，训练、生成、修订页面不再展示旧 event 的窗口、condition、Slide/Touch oversample 或 intensity curve 参数。
- fresh 训练页新增只读模型族标识；选择“从头训练”即创建新 `song_form_v1` checkpoint，旧 event checkpoint 只能回主工作台。请求中显式携带 `model_family=song_form_v1`。
- checkpoint 检查显示模型族、schema、epoch、blueprint/draft/revision readiness、token 与参数量；不兼容旧 checkpoint 会给出回主工作台的提示。
- 结构化训练日志分为数据准备、全曲 blueprint、布局/修订和 epoch，未知准备阶段不伪造百分比。
- 在 1440 px 桌面和 420 px 移动视口检查了四个 tab、checkpoint inspect、训练素材刷新、任务日志；没有横向溢出或脚本异常。

目前 R4 工程交付已完成；R5 仍是质量阶段。真实短训模型必须经过更多官方/高质量自制谱训练、固定歌曲 A/B 和专业玩家检查后，才可判断是否改善复杂节奏、重复段落和无理问题。


## 统一主工作台与真实条件/曲线验证（2026-10-13）

主 WebUI 已将 `song_form_v1` 设为默认模型族，并保留旧 event 模型族下拉切换。共享训练素材、质量权重、resume/finetune 手动路径、输出目录、生成页、Radar/Style、曲线编辑入口和任务日志仍在同一工作台；模型族切换只替换不兼容的参数合同。浏览器验证覆盖 1440 px 和 420 px：模型结构与上下文、条件/正则化/规划学习不再为空，Radar/Style 可见，曲线入口可见，移动端无横向溢出。

使用真实 `E:\后训练（官谱）\training_manifest.json` 限量 16 张、1 epoch、CUDA、`d_model=128/layers=2/nhead=4/max_tokens=1024/max_memory_tokens=256` 完成了新合同训练：

- `optimizer_steps=567`，`blueprint_steps=16`；draft/revision 计数为 577/556；
- checkpoint 能力检查：`condition_controls_trained=true`、`load_curve_trained=true`；
- 条件监督来自谱面事件统计（`event_derived_radar_style_v1`），训练带显式 `condition_null`，不是把 Radar/Style 静默替换成默认值；
- 强度曲线使用 2 秒相对音频时间 bin、event density/tap/sustain/slide_touch 四通道，参与 blueprint 负荷损失与布局输入。

同一 checkpoint 的生成链路验证：

- 主 WebUI `/api/train` 受控 CPU smoke：1 epoch、2 张谱面、1 step，任务状态和 `form_epoch` 日志正确；
- 主 WebUI `/api/generate` 真实路由成功，输出严格 simai 解析通过；
- CLI 自定义 Radar/Style 条件生成成功，报告保存了 `generation_condition`；同 seed 默认条件与自定义条件产生不同 blueprint/onset 数量，证明条件进入了规划输入，但不把一次短训差异当成质量提升；
- 模型曲线分析返回 `method=learned_song_form_v1`、64 个编辑点；带曲线的生成报告记录 `intensity_curve.applied=true`、`guidance=onset soft budget plus span-local layout condition`。

早期 `song_form_v1` checkpoint 仍可加载和生成，但 capability 检查会显示条件和曲线尚未训练；旧 `event_ctx_v1` Prism checkpoint 仍走原路径。一次受控生成中，undertrained checkpoint 出现少量 decoder fallback；系统不再直接崩溃，而是在报告记录 `decoder_fallbacks`，这只证明工程可用，不代表谱面质量达标。
