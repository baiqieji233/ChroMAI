# 全项目只读代码审核记录

日期：2026-10-14。Git 基线：`ce976b2`，分支 `codex/song-form-rebuild`。

用户要求：审核和记录，不修复，不干扰现有任务。本轮仅新增本文档；没有修改源码、测试、参数、数据集、缓存或检查点，没有提交 Git，没有启动/取消训练或生成，没有重启 WebUI。

工作区原有未提交文件：`src/maichart/cli.py`、`src/maichart/preprocess.py`、`src/maichart/webui/server.py`、`tests/test_webui.py`。审核包含它们在磁盘上的实际内容，不假设它们与 HEAD 一致。

## 审核方法与范围

- 对当前主模型 `song_form_v1` 的训练、数据准备、缓存、条件、Planner、解码、曲线、时间轴、段落关联、持续动作、修订、修复与导出做跨模块追踪。
- 对主 WebUI、独立 form-lab、素材/权重管理、任务/日志/下载及部署配置做静态审核。
- 对共享解析器、IR、标签、音频、序列化，以及旧 event/frame 模型的关键路径做审核；旧路径问题单独列出，不能归因于当前模型。
- 288 个受 Git 管理的 Python 文件、约 70,176 行完成 AST 语法扫描；4 个前端 JavaScript 文件和启动 PowerShell 完成语法检查，均无语法错误。这不代表每个函数都经过同等深度的人工验证。
- 关键行为使用 Python `-B` 与隔离 AST 函数/模拟对象复现；不写字节码，不跑 pytest，不读取大型 checkpoint，不使用 GPU，不触发服务端业务请求，不访问全量音频。
- 已知此前修复的曲线行错位、类型占比倍率、曲线 float32 终点越界、保存曲线 null 元数据、回退 KV 上下文等，没有重新列为未修复问题。

证据标记：**内存复现**表示存在独立小样例；**静态确认**表示代码分支能确定行为，但未触发真实业务；**条件风险**表示触发需要特定输入/并发/部署方式；**能力风险**不等同于 bug。

等级：P1 为可导致任务失败、输出覆盖或样本/时钟明显错误的高影响问题；P2 为条件性错误、工作流误导或数据一致性问题；P3 为资源/部署/可观测性缺口。没有发现需要本轮立即干预运行任务的已证实 P0。

## 当前模型与共享路径

### A01 · P1 条件性：note 内负 delay 被提升为全曲偏移，消费者解释不一致

位置：`src/maichart/event_timing.py:406`、`:432`；`src/maichart/generation/post_process.py:134`；`src/maichart/generation/muri/parser.py:79`；`src/maichart/generation/event_compile.py:159`；`src/maichart/form/data.py:418`；`src/maichart/generation/acceptance.py:507`。

**内存复现**：`&first=0`、开头 `1-4[-0.4##0.5]` 会得到 `time_offset=-0.4`，使后续 beat 1 的全曲时间从 0.5 变为 0.1 秒。共享静态几何又保留 note 内 `delay=-0.4`，在使用 `audio_first=-0.4` 时将 trace 起点计算成 -0.8 秒；MiaCode 本地解析则将负 wait 限为 0，计算成 -0.4 秒。

同一原子能通过 `_NoteGrammar` 和 `assess_event_chart`，但 `compile_event_points` 期望 `time_offset=0`，重解析却产生 -0.4，最终报 exact-event roundtrip 错误。训练使用 `chart.audio_first`，部分验收/条件统计使用 `chart.first`，进一步产生跨路径时间差。

影响：特定谱面可能被过滤、持续动作构造或末尾导出失败，相关评估与训练时钟不一致。未读取正式模型原子目录确认触发频率；不能认定当前模型已因它整体退化。现有负偏移测试主要确认某一局部解释，没有覆盖整条消费者链路。后续需要统一全曲偏移与 note-local delay 的含义，而不是仅让编译比较放行。

### A02 · P1 条件性：`||` 注释在共享谱面解析中被当作拍号

位置：`src/maichart/timing.py:283`；`src/maichart/generation/muri/parser.py:548`；`src/maichart/form/data.py:405`。

**内存复现**：`(120){4}1,\n|| chorus\n2,3,E`，MiaCode 成功得到 0、0.5、1.0 秒的三颗 Tap、无解析错误；`parse_event_chart_text` 报 `Malformed meter directive`。共享解析器要求 `||数字/数字`，MiaCode 将 `||` 后的本行视为注释。

影响：带段落注释的谱面不能进入当前训练或修订。注释里的逗号也会先进入共享逗号分割逻辑。这里确认的是两条生产路径语义冲突，不依靠外部资料宣称整个 Simai 方言已被验证。

### A03 · P2：共享时值语法覆盖小于 MiaCode，导致可检测原子先被拒绝

位置：`src/maichart/duration.py:12`、`:35`；`src/maichart/notes.py:173`；`src/maichart/generation/muri/parser.py:79`；`src/maichart/generation/miacode_conflicts.py:24`。

**内存复现**：MiaCode 对 `4h[150#2:1]` 得到 0.8 秒；对 `1-4[3##8:3]` 得到 wait 3 秒、duration 0.75 秒；对 `1-4[3##160#8:3]` 得到 wait 3 秒、duration 0.5625 秒。共享 note parser 对三者均失败。`parse_event_notes` 会将 `1<2[0##120#8:1]` 标为 fatal unsupported，尽管 MiaCode 可以解释。

影响：训练数据、修订与检测覆盖不一致；可能在进入 MiaCode 前就被排除。不能把这种语法覆盖缺口称为模型生成能力不足。

### A04 · P2：Planner 标签漏掉最后一个音频帧中心之后的有效起音

位置：`src/maichart/form/blueprint.py:223`；`src/maichart/generation/event_audio.py:1122`。

**内存复现**：2 秒音频的 50 Hz 帧为 0～1.98 秒。位于 1.99 秒的合法起音被 `second <= times[-1]` 排除，得到 onset 最大值 0、type 有效帧数 0。它仍在音频的 `[0, 2)` 范围内，且应能给 1.98 秒帧提供高斯监督。

影响仅限全曲尾部不到一个采样间隔的范围；不足以解释大段配置质量差。后续需区分“帧中心范围”和“帧覆盖的音频范围”。

### A05 · P2：部分 WebUI 认可的模型维度无法创建 Planner

位置：`src/maichart/form/model.py:35`；`src/maichart/form/train.py:209`；`src/maichart/form/blueprint.py:44`；`src/maichart/webui/form_api.py:229`。

**内存复现**：`d_model=180, nhead=6` 可通过配置模型和 API 的整除检查，但 Planner 使用 `width=90, form_heads=4`，抛出 `width must be divisible by form_heads`。

影响 fresh 训练的部分自定义设置；当前 512/8 不触发。创建模型发生在数据准备之后，可能先消耗准备时间再失败。

### A06 · P2：跨训练语义版本 resume 保留中途游标，跳过当前轮的新版 Planner

位置：`src/maichart/form/train.py:189`、`:220`、`:231`、`:285`、`:296`。

**真实日志与静态确认**：旧记录为 `song_form_training_v2`。当前任务 `082b749fc639` 的日志记录新版 v3、`optimizer_restored=true`，并从第 5 轮 `layout_revision completed=16000` 继续。代码保留游标，将该轮 `plan_start` 设为 `plan_total`。

结果：第 5 轮剩余配置更新用新版代码，而该轮 Planner 已在旧代码下跑完，不会在恢复后重放新版 Planner 目标。后续第 6 轮开始仍会正常训练 Planner，影响有界。能力计数重置后还可能在首次新版 Planner 更新前暂时不允许曲线预测。这不是本轮停止/重启训练的理由；这里只记录合同缺口。

### A07 · P1 条件性：并发生成可以覆盖同一个输出与报告

位置：`src/maichart/webui/static/app.js:3195`；`src/maichart/webui/form_api.py:265`；`src/maichart/webui/server.py:1664`；`src/maichart/form/cli.py:207`。

**静态确认**：生成按钮的 submitting 锁在 POST 返回 task_id 后释放；默认输出为共用 `maidata.txt`。后台任务独立执行，没有对输出路径加占用约束，两份输出还共用 `.form.json`。不同 WebUI 实例也使用相同目录。

影响：重叠任务可能覆盖谱面、将一份谱面与另一份参数报告配对，或让旧下载链接指向后来生成的内容。为了不干扰业务，没有真实发送并发生成请求。

### A08 · P2：启动期间的取消请求可能无效

位置：`src/maichart/webui/server.py:1538`、`:1666`、`:1697`；`src/maichart/webui/static/app.js:1441`。

**隔离 AST 内存复现**：task 已为 running、但 process 仍为 None 时，`_cancel_task` 返回 running，既不设置取消意图，也不阻止稍后的 Popen。UI 对 queued/running 开放取消，因此存在用户点击后工作仍启动的窗口。没有对真实任务调用取消接口。

### A09 · P2：任务先公布 completed，再收集结果，轮询可能漏掉文件

位置：`src/maichart/webui/server.py:1716`、`:1720`；`src/maichart/webui/static/app.js:1396`。

**隔离 AST 内存复现**：模拟 worker 在关闭日志时状态已为 completed，而 files 尚未收集；随后才补入结果。客户端在 terminal 状态停止轮询，若正好在这段时间读取，则不会获取后来补齐的文件或质量报告。后端读取 checkpoint 名单/JSON 耗时会扩大窗口。

重要核对：`_collect_task_results` 对 form_generate 也会填入 files/paths/download_urls，因此“song_form 始终没有下载链接”不成立；真正问题是发布完成的顺序，以及下述质量摘要未接入。

### A10 · P2：resume 界面允许调整实际被 checkpoint 覆盖的参数

位置：`src/maichart/webui/static/app.js:1044`、`:2147`；`src/maichart/webui/form_api.py:163`。

**静态确认**：界面恢复/锁定结构、学习率和区间上限，但 batch、累积、max_steps、limit_charts、条件省略、曲线/关联损失、质量开关等仍可能显示编辑值。API 在 resume 时读取 checkpoint 并覆盖这些值，界面摘要却反映提交值。

影响：用户以为降低了短跑 batch 上限、改变了遮罩或关闭权重，实际任务仍按 checkpoint 训练。继承本身可合理，隐瞒有效值是工作流问题。

### A11 · P2：共用工作流摘要展示 song_form 不使用的旧架构功能

位置：`src/maichart/webui/static/workspace.js:129`、`:154`；`src/maichart/webui/static/app.js:1044`；`src/maichart/webui/form_api.py:106`。

**静态确认**：训练摘要仍显示验证集/best、偏好训练、Slide/Touch/细碎采样倍率，生成摘要读取旧 conflictDetectionEnabled/conflictRepairMode。当前 song_form payload 不发送这些训练字段，实际修复开关是 formRepair。

影响：当前模型的有效训练与修复行为被错误描述。此项不要求增加验证集，仅应在后续让摘要按模型族表达实际行为。

### A12 · P2：控制方案“自动继承”与 finetune 实际行为不同

位置：`src/maichart/webui/static/index.html:372`；`src/maichart/webui/static/app.js:1063`；`src/maichart/webui/form_api.py:163`、`:239`。

**静态确认**：auto 与 song_form_controls_v1 两项没有不同的后端训练合同；只有 none 被特殊处理。说明称已有模型继承，但损失/条件省略等只在 resume 下继承，finetune 使用界面值。

影响：已有模型加强训练可能使用用户未察觉的界面默认损失权重。none 的真实说明已在上轮修正，这里没有把它列成仍未修复的旧问题。

### A13 · P2：独立 form-lab 与主工作台隐式默认值不同

位置：`src/maichart/webui/static/form_lab.js:14`、`:16`、`:18`；`src/maichart/webui/form_api.py:214`、`:218`。

**静态确认**：form-lab 省略 quality_enabled、condition_drop_prob 和 load_curve_loss_weight，API 默认分别为 true、0.1、1.0；主工作台初始发送质量关闭、0.15、0.2。form-lab 对缺失 BPM 默认 120，空字符串经 Number 转换为 0 后报错，与主页自动 BPM 不同；种子留空也变为固定 0。

影响：换入口可能得到不同有效训练/生成参数，不应被称为等价界面。用户已手动修改参数时不能仅按页面初始值推断本次实际设置。

### A14 · P2 条件性：并发保存质量标注会丢失更新

位置：`src/maichart/webui/quality_api.py:221`、`:297`；`src/maichart/training/quality_annotations.py:204`。

**静态确认**：读取全部记录、修改、写回不在同一个锁内；锁只保护最终原子替换。两个请求都从旧快照修改不同谱面，后写者可以覆盖先写者的修改。不同 WebUI 进程的进程内锁也不能彼此协调。没有实际写标注验证。

### A15 · P2 条件性：素材界面可能显示训练不会采纳的过期音频标注

位置：`src/maichart/webui/quality_api.py:170`；`src/maichart/training/quality_annotations.py:find_annotation`；`src/maichart/form/data.py:347`。

**静态确认**：界面查标注未启用 verify_audio，训练解析器检查存在的 audio_sha256。音频在原路径更换、谱面不变时，界面可能仍显示优质/减权标签，训练却拒绝套用该标签。当前实际语料是否触发未扫描确认。

### A16 · P2：song_form 的详细质量摘要没有接入主页质量面板

位置：`src/maichart/webui/static/app.js:1449`；`src/maichart/webui/server.py:1918`；`src/maichart/webui/form_api.py:371`。

**静态确认**：详细 `renderGenerationResult` 和后端 quality_reports 收集只对 kind=generate 执行，song_form 使用 form_generate/.form.json。文件链接在正常收集后可以存在，但重复配置、类型偏差等 `.form.json` 内容不会像旧模型一样进入主质量面板；目前只额外展示修复摘要。

### A17 · P2 条件性：部分检测覆盖的退化可能被修订接受判据漏掉

位置：`src/maichart/form/repair.py:98`、`:116`；`src/maichart/form/revision.py:181`。

**内存复现**：unsupported/unmapped 从 1 增至 3 时，两次 partial 报告都被压为 coverage=0.5。若可处理硬冲突减少、其余判据满足，`apply_patch_candidate` 返回 accepted，没有识别覆盖变差。

这个复现证明判据的信息丢失，不证明目前受 grammar 限制的实际神经/确定性候选一定能生成这种覆盖变化。最终结果仍标为 partial，不是错误宣称可玩。后续应保留可比较的覆盖证据。

### A18 · P3：修复同时覆盖不完整与预算耗尽时，仅返回一个原因

位置：`src/maichart/form/repair.py:307`。

**静态确认**：partial_coverage 分支先于 expired，最终 status 不表达 budget_exhausted。checks、耗时和剩余问题还在，因此并非完全隐瞒，但难以从状态直接区分“引擎覆盖不足”与“预算也不足”。

### A19 · P2，有限影响：EventChart 序列化丢失 time_offset

位置：`src/maichart/event_timing.py:87`、`:106`。

**内存复现**：to_dict 存 time_offset，from_dict 忽略。first=0.1/time_offset=-0.4 的 audio_first 从 -0.3 恢复成 0.1。未发现当前生产流程直接调用 EventChart.from_dict，当前主路径影响未证实；公共序列化往返合同确定破坏。

### A20 · P3：大文件下载和持久上传有资源开销风险

位置：`src/maichart/webui/server.py:1584`、`:1612`。

**静态确认**：下载先读取整个文件到内存，下载大 checkpoint 会额外占用接近文件大小的内存；并发下载叠加。上传文件长期保留，没有发现自动清理入口。未进行大文件请求，避免影响当前训练。ZIP 提取已有 staging 和路径穿越检查，未将其列为已确认漏洞。

## 旧架构与旧缓存入口专属问题

以下项目不作用于当前 song_form 训练损失，不能拿它们解释当前配置质量。

### L01 · P2：事件模型分桶采样器声明批数小于实际批数，训练漏样本

位置：`src/maichart/training/train_events.py:913`、`:925`、`:428`、`:444`。

**内存复现**：1000 样本、batch=16、shuffle=true，声明 63 批但迭代产生 64 批；训练按 len(loader) 截断，固定种子样例只消费 987 样本，漏 13。触发条件是 chunk 与 batch 不整除；shuffle=false 验证路径不触发，batch=1 或对齐大数据集也可能不触发。

### L02 · P1 条件性：事件起音辅助使用错误的质量归一化分母

位置：`src/maichart/training/train_events.py:301`、`:314`、`:533`。

**静态重新确认**：逐帧 onset loss 仍传入 quality_plan_normalizer。局部排除使 Planner 窗口权重全零，而起音帧仍有有效正权重时，分母被夹为 1e-8，会严重放大辅助损失。这是此前已发现但按当时范围未保留修复的旧路径问题；本轮不重复运行训练，不认定当前 song_form 受影响。

### L03 · P2：固定帧训练的末尾不足一组累积时梯度偏小

位置：`src/maichart/training/train_generator.py:163`、`:175`。

**内存复现**：每批损失除以完整 accumulation=4，最后只剩 1 批仍直接 step，标量样例梯度为 -0.5，而实际这一组的均值梯度应为 -2.0。该入口返回批数累计到 global_step，因此它也不等同 optimizer updates。旧事件模型另有短 batch 按等权 microbatch 累积的权重偏差，需要单独考虑样本数量。

### L04 · P2 条件性：固定帧 auto 正类权重把 Padding 计为负样本

位置：`src/maichart/training/losses_v25.py:87`。

**内存复现**：保持 1 个有效全按键帧，其余补齐帧 mask=false，长度从 1 增至 10 时，auto button 权重使同一个有效帧的 loss 从 0.693 变为 6.238。需要实际启用 auto；普通默认数值 1 不触发。Padding 会改变有效样本梯度，与只遮罩最终 loss 不同。

### L05 · P2：旧 dataset cache 的歌曲 ID 会碰撞

位置：`src/maichart/dataset.py:200`、`:221`、`:294`。

**内存复现**：两个纯日文目录均变为 unknown；Song Name 和 Song@Name 均变为 Song_Name。旧 builder 按 ID/难度写缓存，不能区分这些来源，可能串用缓存或覆盖。当前 form 的准备缓存使用源身份和内容哈希，不能将它也归入该缺陷。

### L06 · P2：旧预处理缓存仅按文件存在复用，没有内容/参数失效

位置：`src/maichart/preprocess.py:532`、`:787`、`:799`、`:810`、`:821`；`src/maichart/dataset.py:223`。

**静态确认**：maidata、音频、division、sample_rate 改变，但 force=false 且缓存存在时仍复用旧结果。新 manifest 的统计可能来自新源码，cache 标签仍来自旧源码。此项是旧 ChartIR/frame/QC 路径，不是当前 FormDataset 的内容校验准备缓存。

### L07 · P2：无 raw_inote 的 ChartIR 导出会压缩所有时间间隔

位置：`src/maichart/exporter.py:41`。

**内存复现**：tick=0 和 7680（beat=4）两个 Tap 导出为 `1,2,E`，重解析得到 beat=0 和 1；空拍、变 BPM、持续时间等也未完整恢复。保留 raw_inote 的正常回写不会进入此 fallback；当前 song_form 使用精确事件编译器，不使用它。

### L08 · P2/表示限制：固定帧 presence 同时代表起音与持续占用

位置：`src/maichart/labels.py:98`、`:279`；`src/maichart/training/frame_label_codec.py:45`、`:92`；`src/maichart/generation/inference.py:175`。

**静态确认**：hold/slide_active 令无新起音的帧 has_note=true，codec 又把它变成 presence=1 以及 hold/slide 类型，旧生成器把每个这种帧解码为新的音符头。即便完美学习了这个标签，也未必能恢复“一个长音”与“多个新头”的区别。只影响旧 frame 家族；当前 Blueprint 使用起音点监督。

## 测试、打包和部署

### T01 · P2：四个同名测试定义被后面的定义覆盖

位置：`tests/test_webui.py:299`、`:349`、`:372`；`:316`、`:366`、`:389`。

**AST 确认**：两个 test 名称各出现三次，Python 仅保留最后一次，四个早先定义不会分别被 pytest 收集。工作区已有未提交内容位于此文件，但重复定义并非本轮引入。没有改名或运行测试。

### D01 · P2 条件性：非 editable 安装可能缺失 WebUI 静态资产

位置：`pyproject.toml:40`；`src/maimai_ai_chart.egg-info/SOURCES.txt`；`src/maichart/webui/server.py:STATIC_ROOT`。

**静态证据**：package-data 只列 muri/reference，当前 SOURCES 也不包含 webui/static HTML/JS/CSS。editable 源码启动能够掩盖问题。没有构建或安装 wheel，因此以打包风险记录，不能称本轮实测 wheel 已损坏。

### D02 · P3：声明的 PyTorch 最低版本可能小于实际 API 需求

位置：`pyproject.toml:31`；`src/maichart/training/train_generator.py:74`；`src/maichart/form/training_runtime.py`。

要求写为 torch>=2.0，而实现使用 torch.amp.GradScaler 等较新接口。未安装旧环境或在线核验每个 API 的引入版本，因此不声称某个具体最低版本已经验证。后续应核对依赖下限与 CPU/AMP 两种路径。启动脚本只检查 import，并不能证明训练所需全部接口都存在。

### D03 · P3：CI 只跑 Python，不覆盖主前端行为测试

位置：`.github/workflows/python-package-conda.yml:36`；`tests/test_task_polling.js` 等。

**静态确认**：配置中没有 Node 测试步骤。现有前端 payload/轮询/曲线恢复测试即便本地存在，也不会随该 workflow 自动执行。没有触发远程 CI。

## 能力风险与未证实假设

这些不是本轮确认的实现 bug，不应据此直接破坏性修改：

1. 官谱历史、准确采音骨架、真实负荷曲线与自由生成输入存在分布差异；当前条件随机省略不覆盖所有这些输入。前轮的小样本诊断支持其有影响，但未证明它是全部配置退化的主因。
2. Planner 50 Hz 分辨率、单一优先类型监督和未强制使用的持续时值限制表达；布局层不能重新选择已定起音。副歌采音取舍不足可能包含训练不足和能力限制。
3. 相似音频区域不等于已确认的副歌/主歌角色；段落关联是候选证据。
4. token loss、训练准确率、语法通过和检测问题减少不能独立证明高质量或体感合理。用户暂缓验证集的决定保持不变，没有把验证集列为必做开发任务。
5. 质量标注中直接手写重复记录、相同 dataset_id 的多输入可能引发歧义。当前已读合并记录有不同 dataset_id，没有证明实际数据已发生该冲突。
6. 临时原子目录限制、软类型指导、记忆强度和温度可以影响表达，但没有发现足以把当前谱面退化完全归于缓存解码/目标位移的证据。

## 未发现新具体问题的重点路径

当前 form 目标位移、因果解码与回退、null 条件、样本/准备缓存身份、正常质量标签匹配、同语义版本 optimizer/RNG 保存恢复、完整累积更新、曲线生成时钟与有效空区间、休止最终复查、相似段落因果记忆、持续动作边界，未发现新的可复现 P1/P2。没有因“看起来复杂”或“小样本结果不理想”将这些模块判为错误。

相关 tools/scripts 已扫描其入口和审计路径，尤其是评价工具的时间参数、报告写入与 checkpoint 读取。没有执行会落盘缓存/评估报告、调用外部驱动或重载模型的工具。

## 当前任务观察与不干扰核对

审核开始时 PID 20044 是正在运行的 song_form 训练，其父 WebUI 为 PID 23456；任务 ID `082b749fc639`。只读取进程和已有日志。

最后读取的 00:52:01 日志：第 5 轮配置阶段完成 36502/65392 批，仍在运行；草稿配置字符 loss 约 0.727、准确率 74.7%，修订配置字符 loss 约 0.202、准确率 93.0%。这是日志快照，不能代替实时状态或音乐质量验收。

源码/测试/脚本/配置的 298 个已跟踪文件在审核开始记录哈希，完成时重新比较，全部一致。完成时再次确认 PID 20044 仍在运行。本文档是本轮唯一主动写入的文件；正式任务自行产出的日志或 checkpoint 不属于审核修改。

## 后续处理顺序，仅供后续决策

先统一共享时间与语法合同、补齐条件性 Planner 边界/模型设置检查，再处理并发输出和任务状态发布、resume 有效参数显示与质量标注事务。旧框架损失与缓存问题可以按用户是否仍使用对应入口安排。能力风险与训练策略应单独做受控评估，不能混入“修 bug”而静默改变模型行为。

**以上问题均未修复。** 本文是问题台账和证据边界，不是兼容承诺、音乐质量验收或所有潜在 bug 已被发现的保证。
