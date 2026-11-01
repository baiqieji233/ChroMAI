# 数据与标签链路只读审计（2026-11-01，0ff1f5a）

范围：原谱解析/编译、Composer 切分、Form 准备与缓存、Composer stream 数据、phrase 与前缀增强。仅阅读源码和 v38 现有 split_report；未触及训练进程、GPU 或生产缓存。这里的“影响”是代码路径的条件风险，不是已观察到的音乐质量下降。

## 结论

当前检查**未确认会普遍使 v38 正常出谱失败的解析、缓存错标或未来谱面输入泄漏 bug**。下面有两个值得后续评估的设计边界、一个条件性评估偏差。不要在正在训练的 v38 上直接改数据契约或继续 resume。

### D1：全曲控制量训练来自目标谱面，推理来自请求值（设计边界；优先评估）

- 证据：`form/data.py:602-609` 从本曲筛后完整事件计算 `condition_from_events`；该函数将全曲密度、峰值、Slide/Touch 等比例、动机重用、方向性写进控制向量（`generation/event_quality.py:543-599`）。`composer/stream_data.py:305-324,685-716` 将这个同曲全局控制量喂给每个流式训练窗口。推理由用户/默认值供给（`composer/stream_generate.py:116-123`）。训练代码也如实标注 `reference-derived requested controls`（`composer/stream_train.py:439`）。
- 最小触发：同一首曲在训练窗口仅给前缀时，控制向量已包含整张目标谱的 `pattern_reuse`、音符密度等统计；生成新曲只能传入手动/默认向量。
- 范围：所有 Composer StreamDataset 样本，包括 v38。它是允许的全局条件任务定义，但产生真实 train/inference 输入分布差距，并使“完全从音频/前缀学到整段结构”的归因不成立。**不等于**逐事件未来谱面张量泄漏，也不能从源码断言实际出谱退化。
- 建议：冻结 v38 后做同音频/种子/检查点的控制量扰动与默认值对照，记录整曲结构、首段/转段、重复和人听玩结果；若要改变训练控制契约，另起 finetune/checkpoint，并保留父模型。

### D2：元数据合格不保证进入流式训练或验证（设计边界；统计口径）

- 证据：`composer/evaluation.py:52-66,96-113` 的自动 holdout 门槛只检查难度槽和等级，注释明确不保证解析/准备成功；`form/data.py:560-669` 后续还有解析、sanitize、inspect、质量剔除、音频和准备异常，异常被记入 skipped。`composer/stream_data.py:297-323` 再从 FormDataset.songs 建流。
- v38 现有报告：来源 2578 首，元数据合格 2358 首；分组 1790；train/dev/final 各 2243/56/59 首、1700/45/45 组。split_report 三个分区的音频 SHA256 集合交集为 0。该报告未给各分区**实际准备成功 chart/window 数量**。
- 最小触发：任一元数据符合 4–6 槽、12–15 级但 `parse_event_chart` 或 `inspect_chart` 失败的谱面，会留在 split manifest，准备时才被跳过。
- 影响：若 dev/final 有集中跳过，报告中的保留比例与真正参与验证/生成的曲数不同；不构成 train/holdout 泄漏。建议从现有各分区准备日志抽取 accepted/skipped/OOV/窗口数并按来源、难度分层，不为本审计重跑全部准备。

### C1：`unknown_policy=mask` 将未知前一事件留作 NOTE 类但清空其内容（条件性评估偏差）

- 证据：`composer/stream_data.py:337-373` 在 OOV 时将 `(delta,event)` 重置为 `(0,None)`；`stream_data.py:667-680` 的下一位置 `history_kind` 仍由原始流的 NOTE 给出，形状/原子清零。目标位的 `sample_weight` 则归零（`stream_data.py:691-699`）。
- 最小触发：用现有 codec/delta vocab 在新 manifest 构建 `unknown_policy='mask'`，前一 NOTE 含不在词表内的 atom；当前受监督位置看到“NOTE + 零动作/零 gap”的非真实历史组合。
- 范围：仅 OOV 掩码验证/跨数据集场景；正常训练默认 `error` 不走这条路。可能污染紧随 OOV 的 teacher-forced 指标，不能据此断言生成质量变差。建议在评估报告单列 OOV 邻近窗口或从首个 OOV 之后截断可评样本；改变策略须检查旧 checkpoint/验证口径。

## 已核实的防护与非问题

- 原谱 `first_<slot>` 优先级已在 `maidata.py:167-180` 与 `event_timing.py:380-382` 接入；严格解析可拒绝未知 token 与重复头（`event_timing.py:384-409`）。编译器保留精确 Fraction 节拍并回读核对，过滤模型吐出的终止符 `E`（`generation/event_compile.py:34-55,67-82,118-150`）。本审计未进行全数据 roundtrip。
- 旧 `_preparation_annotation` 跨路径同内容错标问题**当前已修复**：匹配 source、可选 audio 路径、chart/audio 哈希（`form/data.py:514-543`）。准备缓存 key 包含文件内容、难度、等级、结构文件、关键设置与质量记录，读缓存时校验 identity 和数组哈希（`form/preparation_cache.py:247-310,320-365`）。
- `split_manifest` 将相同音频 SHA256、song_id、music_id、maidata 路径联通为同组（`composer/evaluation.py:120-153`）；`assert_unseen` 有独立身份交集拒绝（`evaluation.py:200-223`）。v38 现有 split_report 的音频集合两两不相交；这不是我们重新哈希全部源音频所得。
- Stream 的 `build_steps` 将未知区域权重清零并以 WAIT 前进，EOS 只表示已知音频尾部（`composer/stream_data.py:87-135`，`composer/event_views.py:32-135`）；所有 target 恰覆盖一次有断言（`stream_data.py:383-423`）。`block_aligned_v2` 的逐步窗口权重在 `__getitem__` 应用（`stream_data.py:402-407,691-699`）。
- Phrase 音乐分段来自音频，参考段索引只取历史段（`composer/stream_phrase.py:174-253`）。当前事件特征聚合的 query 使用严格 `index < query`（`stream_phrase.py:504-542`），ordered 参考序列还需完整段落已可见（`stream_phrase.py:434-446`）。`target_phrase_*` 是监督目标，未见它被当输入特征读取。增强会重建历史、动作状态、transition 与 phrase 行（`composer/stream_augmentation.py:267-325,443-536`），并保留总学习质量预算（`:227-255`）。

## 覆盖与未验证

已读：`maidata.py`、`event_timing.py`、`notes.py` 入口、`generation/event_compile.py`、`composer/evaluation.py`、`form/data.py`、`form/preparation_cache.py`、`form/segmentation.py` 入口、`composer/event_views.py`、`composer/stream_data.py`、`composer/stream_phrase.py`、`composer/stream_augmentation.py`、`form/cache_support.py`，并读取 v38 `data_split/split_report.json` 和三分区 manifest 的 songs 数。未重新解析 2883 张元数据合格谱、未运行整库准备/音频特征、未验证 checkpoints 自由生成，因此不能宣称语法通过率、音乐匹配或可玩性。

## 补查：2026-11-01 v38 真实准备、旧导出与打包

### F1：旧逐帧导出会漏写非帧格点 BPM（已确认 bug，**不在当前 Composer 出谱路径**）

- `generation/simai_compile.py:45-74` 只在 `frame_index * 4 / division` 等于变速拍时写 BPM 指令；`simai_compile.py:78-90,154-167` 却在 IR 中保留全部变速并按全部变速算 note 秒数。CLI 仍调用该旧函数（`generation/cli.py:60-75`）。当前 Composer 调用 `compile_event_points`（`composer/stream_generate.py:406`）。
- 低负荷 CPU 复现：division=16、beat 0 BPM120、beat 1/8 BPM240、beat 1/2 一个 Tap。`compile_predictions` 返回 Simai `(120){16},,1,,E`，没有 1/8 变速；IR 却有 `(0,120),(0.125,240)`，且 note time=0.15625 秒。文本被普通 Simai 播放器读取后按 120 BPM 将 Tap 放在 0.25 秒，若 first=0 则偏 93.75 ms。复现只在内存构建对象；未写输出。
- 修法：旧逐帧 CLI 对非帧格点 BPM 明确拒绝并提示使用 exact-event 编译，或改用可插入额外时间点的 `compile_event_points`；同时做文本回读和 IR 时钟一致性断言。别用这个旧函数的结果推断 v38 质量。

### V1：验证 OOV 的零权重会压低固定分母 loss（指标偏差，非训练 fatal）

- `composer/stream_train.py:160-172` 验证用 `unknown_policy='mask'`；`stream_data.py:337-373,691-699` 把 OOV 目标权重置零；`stream_model.py:1673-1685` 各目标损失以固定 `B * rollout_steps` 为分母，而非本批有效权重。`stream_train.py:427-440` 只按批次数平均指标；`:571-578` 只要求全体 trusted targets 非零，没有对已选验证批的 OOV 覆盖设阈值。`:223-230` 在 2056 个样本中均匀抽 128 个窗口（32 批×4）。
- v38 现有日志 `20261101_180957_composer_train_626129bc3951.log:2849`：dev 32536 steps 中 action OOV 420、delta OOV 0，known 32116，known_fraction 98.709%；这个比例是**全 dev**，不是实际抽中的 128 窗口覆盖率。同等预测难度下，OOV 比例增加可能让验证 loss 下降。应同时报告有效监督权重/候选数，评估时按有效质量做可比聚合，并保留当前指标供 v38 历史比较。

### v38 真实接受/跳过与质量标注接入

- 现有 train 日志 `:2783` 准备 **2675 charts / 138787 form examples**，dev `:2848` 为 **59 charts / 3209 examples**；stream train 为 **96065 samples**（`:2850`），dev 为 **2056 samples**（`:2849`）。按 split manifest 和 `numeric_level` 同口径计数：train 元数据合格 2747，dev 62，final 74。因此 train 后续准备少 72 chart，dev 少 3；final 此日志未准备，未知。`manifest_counts.py` 仅读 JSON。
- 从已有 chart 日志按源目录计数：train 全量官谱 1705/1706，后训练官谱 662/662，后训练自制 308/379；dev 全量官谱 38/38，后训练官谱 12/12，后训练自制 9/12。即 train 72 个缺口中 **71 个自制谱**，dev 3 个全为自制谱。对缺口中的 train 前 5、dev 全 3 做只读 `parse_event_chart→sanitize_chart→inspect_chart`：一个官谱因重复 BPM 拍被拒，所查自制谱均为不支持的 note/Slide 语法（示例 dev `Nightfall` 在 beat 200 的 `2bq8q6<2[4:9]/6$b` 被拒）。这是过滤实际发生的证据；尚未给 75 个缺口逐一归因。脚本 `missing_charts_probe.py`。
- 合并 manifest 来源为全量官谱 1524 首、后训练官谱 657 首、后训练自制 397 首（`webui_workspace/composed_manifests/composed_4bd08985b0706867/training_manifest.json` 的 `datasets`）。`dataset_catalog.py:189-205,248-292,347-353` 合并并校验标注；FormDataset 对 split 通过 `split_metadata.source_manifest` 回到合并侧车 `quality_annotations.json`（`form/data.py:786-806`），再按 source、slot、chart/audio 哈希匹配（`:514-543`）。因此 stream 会读取这些标签（`stream_data.py:297-324`），不是单纯 manifest 元数据。
- 侧车有 2851 条记录，全部 `excellent`，权重 1.4（2387）或 2.15（464），没有零权重或负区间；按 path+slot，当前 train/dev/final 各对应 1040/24/30 条，**未逐条重新哈希确认**。全量官谱无这些增强权重。当前 v38 不存在通过该侧车剔除负谱的实际证据，作用主要是给后训练数据加权。合并 summary 记有 1 条 stale annotation；其余未标注 chart 用默认权重 1（`form/data.py:589`）。
- `event_usable_for_training` 是旧预处理标志，当前 FormDataset 准备主要依难度、等级和实时 `sanitize_chart/inspect_chart`（`form/data.py:552-588`）；现有日志中 train 有 **11 张标志 false 但被现行解析器接受**的谱面（`accepted_flags.py`），这显示元数据可过时，不能把它直接当 v38 实际接受集合。

### 解析/编译支持边界与打包

- 现行 exact-event 解析按 `first_<slot>` 覆盖全局 first（`maidata.py:167-179`、`event_timing.py:380-382`）；`TempoClock` 逐段积分 BPM，`at` 与 `bpm` 的边界语义在 `tempo_clock.py:9-32`。现行 `compile_event_points` 对同 beat 的多个 EventPoint 明确拒绝，要求同一逗号组里以 `/` 合并多头（`generation/event_compile.py:49-53,75-82`）；它回读校验整个 EventChart（`:155-170`）。CPU 小样例：beat 1/8 BPM240、first=.25、beat 1/2 Tap，回读结果仍为同拍、同 BPM map、同 first。相反，旧 `compile_predictions` 只支持逐帧格点。`notes.py:22-60,165-215,774-815` 支持常见 Tap/Hold/Slide/Touch、分支和复合路径；自制谱仍有未支持 token，当前严格 `inspect_chart` 直接拒整谱（`training/event_dataset.py:86-106`），不会默默删坏目标；唯一显式清洗是识别的 headless Slide 原子（`training/chart_sanitization.py:60-96`）。
- [pyproject.toml](E:/simai_transformer/pyproject.toml) 的 `tool.setuptools.package-data` 仅列 `maichart.generation.muri` reference；WebUI 的 HTML/JS/CSS 位于 `src/maichart/webui/static`，`server.py:60,1783-1787` 在安装包相对路径找它们。现有 `build/lib/maichart/webui` **没有 static 目录**。这是轮子漏静态资源的强配置和旧构建证据；未在本轮 build 新 wheel，故不把旧 build/lib 当新 wheel 的直接内容证明。建议显式声明 `"maichart.webui" = ["static/*"]` 并在独立环境做 wheel 内容和 `/`、`/composer` 静态路由验收。

## 对 `antigravity_review.md` 第 1–3 点的核实与修正

### 1. 时长/BPM/密度统计捷径：有机制可能，**未证实塌缩或主导 loss**

`stream_phrase_model.py:41-76` 的有序目标编码确实读相对拍、相对秒、拍/秒 gap、atom 个数；`stream_phrase_model.py:249-298` 的双向匹配允许目标塔在 match loss 中得梯度，余弦项对目标塔 detach。由此可以提出“模型优先利用易辨别的时钟和密度”的可检验假设。但审查稿说同一个 batch 的唯一短句“来自不同 BPM 曲目”并不符合采样器的主要组织方式：`stream_train.py:176-194` 将**同一歌曲的窗口连续排列**，只在歌曲边界自然跨曲；具体跨曲比例和 BPM 分布未测量，不能写成全批多曲或不同 BPM 居多。`stream_phrase.py:87-107` 还按完整可见事件内容建立 positive identity；目标并非只靠时长定义。

若要证明捷径，需在冻结 v38 后用 held-out、固定窗口做匹配检索消融：仅时钟/密度，去时钟但保留 atom/相位，原模型；按单曲/跨曲、相同/不同 BPM、相同/不同短句长度分层，测同长度负例 top-k、正例余弦、code variance 和生成结构。当前的 loss 下降或旧的 0% 配置呼应都不能单独证明梯度塌缩。保留 gap seconds 可能对音乐对齐有用，不宜在未验证前一概删除。

### 2. 单向量短句压缩：真实信息瓶颈，**不构成 0% 呼应的已证因果解释**

`stream_phrase_model.py:71-77` 把最多 32 个有序事件经 GRU 取最后隐状态为一条段向量；`stream_phrase_model.py:191-210` 把几个历史参考混合成 `intent`。同一乐句内参考集和音频 anchor 通常固定，现有路径没有按当前相对拍位去读取参考乐句对应事件，这是可定位的结构限制。但 `actor_residual` 同时读随时间变化的 `phrase_progress`、`remaining` 和主干 `hidden`（`:115-122`），所以“解码器只有一个静态 128d 向量”也不准确。旧报告 `COMPOSER_DEEP_CAUSE_AUDIT_20261031_ZH.md:68-77` 的四次生成配置呼应 0/0/0/1.41% 是小样本描述，不是该瓶颈的单变量 A/B 或 v38 定论。

正当的后续方向：保留当前 causal reference 选择，在完整已提交参考句上保存**相对拍位分桶的有序事件键值**；当前步以目标句的相对拍位、音频片段和已写事件为 query，只读取参考句相邻拍位的历史键值。比较旋转/镜像等价变换下的相对运动（相邻键位位移、方向、手间张角、gap 序列），并让绝对键位作为可选几何特征以便落地。先做同曲固定窗口的离线检索/诊断 A/B，再考虑更改模型与 checkpoint 契约；不能把新的检索损失直接当音乐质量证明。

### 3. Literal 方位“完全看不见”：**明确错误**；相位 mask 对 literal 无区分是真事实

`event_set_codec.py:174-189,241-243` 中 fallback/literal 的 `phase=0`，但 `phase_equivalents` 通过 `decode(shape, candidate, details)==target` 判断；literal 解码忽略 phase（`:191-209`），因此其 `phase_mask` 是**8 个 True**，不是审查稿所说 `[True,False,...]`。`stream_phrase.py:344-352,409-416` 为有序句保存完整 atom ID；`stream_phrase_model.py:46-54,71-76` 的 `F.embedding_bag` 汇总这些 atom embedding 后送入 GRU，真实按键方位通过不同 atom ID 可见。CPU 小样例：`1-3[4:1]/A1` 与 `2-4[4:1]/A2` 都编码为 `shape=0, phase=0, phase_mask=[True]*8`，但 atoms 分别为 `[1,9]` 与 `[2,10]`。

准确的限制是：literal 的 `phase_values` 都是 8 个 phase embedding 的同一个均值，**该分支**不提供相位区分；atom embedding 对同拍原子做均值，显式相对运动、镜像/旋转等价关系没有独立建模，可能增加学习难度。不能说完整有序编码器完全不知道 literal 真实键位。v38 运行日志的 codec 统计为 168028/1439096 fallback events（约 11.68%）；它说明覆盖面，不证明这部分具体导致何种出谱错误。
