# maimai AI Chart Lab 项目交接摘要

> 日期：2026-10-05
> 项目目录：E:/simai_transformer
> 用途：供新的智能体继续开发“输入音频，生成 Simai/maidata 谱面”。

## 1. 项目介绍

这是一个面向舞萌 DX / maimai 风格谱面的研究型 Python 工程，目标流程为：

~~~text
音频 + 生成条件 -> BPM/音频特征 -> 精确 beat 事件 -> Simai note -> maidata、ChartIR、quality.json
~~~

工程目前能解析和保留 Maidata，建立 ChartIR，处理主要 note、BPM、分音和有理 beat，准备训练数据，训练 event_v1 Transformer，从音频生成事件谱面，并通过 localhost WebUI 完成预处理、训练、断点续训和生成。

这不是官方谱面编辑器，也不是已经达到 Master/Remaster 可玩质量的生产模型。

| 路径 | 状态 | 说明 |
| --- | --- | --- |
| event_v1 | 默认主路径 | Fraction beat、音频条件自回归 Transformer、组合式 Simai note |
| legacy_frame_v2 | 对照基线 | 固定帧和整数 1920 tick，不能与 event_v1 checkpoint 互换 |

新智能体默认继续 event_v1。

## 2. 已完成的关键能力

- 原始 metadata、multiline inote_N、ChartIR 和 JSON round-trip。
- Tap、Break、Hold、Slide、Touch、Touch Hold 基础解析。
- BPM、grid、first、E 和有理 beat。
- 单个坏文档跳过并写入 audit，不阻塞整个任务。
- lv_N 和 inote_N 是训练难度核心；wholebpm、des_N、title、artist 可缺省。
- 4-6*-7 是一个 Slide head 的分支；4-6/4-7 是两个独立 Slide 同时占用 head 4，应拒绝。
- event_v1 不强制把 note beat 量化为 1920 tick。
- checkpoint 保存 vocabulary、atom_catalog、模型配置、data contract、optimizer、scheduler、AMP scaler 和 RNG。
- style 不写入训练 ZIP；训练时在内存中从谱面推导 condition。
- 生成失败返回非零状态并写 quality/error 报告，不写伪造谱面。
- WebUI 显式区分 fresh、resume、finetune 和两种模型族。

## 3. 乱码问题当前结论

旧字符级解码器允许任意组合字符，两个 batch 的 smoke checkpoint 曾输出 A^A9、8427Dh08wACVq、随机 Slide 路径和未闭合 bracket。

event_cli.py 当前使用 checkpoint 的 atom_catalog：

- 字符仍逐个生成，但只能进入训练 atom 的合法前缀；
- event_end 只有在 Simai parser 能识别完整 note 时才能输出；
- 拦截非法 Slide、未闭合 bracket、重复 head 和非法 slash 组合；
- 保留 star Slide 分支和 slash chord 组合。

这解决的是非法输出隔离，不是模型质量。关闭 grammar 会恢复乱码；只依赖 atom_catalog 又会限制训练集没有出现的新合法 note，后续应增加独立有限状态 grammar。

## 4. 时间和旧 CLI

旧 src/maichart/timing.py 要求 beat * 1920 为整数，因此拒绝 8/7、179712/539 等 beat。当前 maichart CLI 对 parse --timing、parse --notes、parse --ir、validate、stats 增加了有条件 fallback。

只有旧路径明确报整数 tick 错误时才切换 exact-event parser。普通语法错误不会被吞掉；非整数 beat 的派生 tick 输出 null，不改变 note 时间。

当前 terminal E 可能输出大量空 timing point，例如 {539},,{77},,,,,{7},,E。这可 round-trip，但有兼容性和 token 浪费问题。

## 5. 当前真实实验

原始 D:/海鲜杯2.zip 约 182 张谱面。182 是压缩包规模，不等于 event train split 的有效样本数。

当前 event train audit：

~~~text
accepted samples = 68
accepted segments = 3345
event vocabulary = 499
skipped documents = 51
segment_beats = 8
style_in_dataset = false
frame_division_used = false
~~~

跳过原因主要是未支持的玩家语法、重复 button/Slide head、特殊 modifier、复杂或非法 duration、源文件缺失。应先按原因分桶，再决定 parser 扩展、无损规范化或拒绝。

smoke checkpoint 位于 E:/simai_transformer/webui_workspace/seafoodcup2_event_smoke/，参数约 229619，epoch 1、max_steps 2、loss 6.310582、token_accuracy 0、position_accuracy 0。它只证明流程连通，不能证明模型学会谱面。

真实 MP3 三候选产物位于 E:/simai_transformer/webui_workspace/seafoodcup2_event_smoke/generated/test_cases/：

~~~text
candidate 1: success, syntax_valid=true, events=5, notes=7, density=0.05 NPS
candidate 2: success, syntax_valid=true, events=5, notes=12, density=0.08 NPS
candidate 3: success, syntax_valid=true, events=4, notes=8, density=0.05 NPS
~~~

三份都通过 maichart validate --strict，但 quality_status 为 needs_review，playable_verified 为 false。

## 6. 核心问题和待解决事项

### P0：正式训练和稀疏输出

- 只训练两个 batch，token/position accuracy 都为 0。
- 生成容易过早 EOS，note 数极少，远低于 12+/13 级谱面。
- 需要先做单样本过拟合，再做完整 train/validation 训练。
- 需要检查 EOS loss、空 segment、segment_beats、max_tokens、长度辅助 loss、采样权重和覆盖范围。
- 禁止把 max_steps=2 的 smoke 结果当成质量结论。

### P0：自动变速还未真正进入生成

event 音频代码和训练数据有 tempo map 支持，但当前 event_cli 生成路径仍使用一个初始 BPM。需要将音频 tempo 检测、音频切片、beat/second 积分和 Simai BPM 输出统一，并测试固定 BPM、音频检测 BPM、真实 chart BPM 三者。

### P0：训练数据覆盖不足

需要统计 51 个跳过文档中的反引号组合、h/f/x/b modifier、复杂 duration、Touch/Slide 特殊写法、chord、meter、BPM directive 和空点。语义明确的语法无损支持并加测试；不确定语义不能静默改写；致命重复 head 继续拒绝。

### P1：grammar、terminal E 和条件控制

- atom_catalog 防乱码但过窄；应增加 parser-driven finite-state grammar，不能直接关闭约束。
- terminal E 造成大量空点和非常规分音，需要决定精确 ChartIR 与兼容文本 export 的策略，不能把所有 note 量化到 {16}。
- radar/style/level 是软条件，不保证生成结果达到目标；需要条件敏感性实验和真实 melody、pattern reuse、directionality 特征。
- 当前 quality report 不提供自动候选排序，也不自动修复目标难度。

### P1：音频表达不足

当前 profile 有固定维度音频、局部节拍、tempo 和相位，但缺少可靠主旋律、音高、乐器分轨、特殊音色、重复段落和 motif 对齐。Tap 与 Slide 的表达歧义仍主要交给模型学习。

### P2：没有 gameplay 验证

当前没有 gameplay harness、手部负担模拟、完整不可玩 Slide 检测、模拟器试玩和候选排序。playable_verified 必须保持 false。除非用户重新要求，不要擅自加入 gameplay harness。

### P2：WebUI 旧进程和旧产物

修改代码后重启 127.0.0.1:7860。旧 error JSON 可能只是历史日志；新候选在 generated/test_cases。上传 ZIP 后必须先做 event 预处理，不能混用旧 fixed-frame manifest、cache 或 checkpoint。

## 7. 推荐工作顺序

1. 读取本文件、docs/compatibility/EVENT_GUIDE.md、docs/explanations/EVENT_TEST_CASES.md，运行 python -m pytest -q。
2. 做单样本过拟合，检查事件位置、note 字符和 EOS。
3. 用完整 train/validation split 正式训练，禁止用 max_steps=2 判断质量。
4. 对 51 个跳过文档分桶并补高频、语义明确的 parser 语法。
5. 将 audio tempo map 接入生成并做 BPM 对照。
6. 处理 EOS/稀疏输出，再改善旋律、音色和 motif 特征。
7. 重新验证 radar/style 控制，再处理 terminal E 和人工试玩。

## 8. 最小复现命令

~~~powershell
cd E:/simai_transformer
python -m pytest -q
python -m maichart.training.prepare_events RAW_ROOT --cache-dir CACHE_DIR -o EVENT_MANIFEST.json --sample-rate 22050
python -m maichart.training.train_events --manifest EVENT_MANIFEST.json --validation-manifest VALIDATION_MANIFEST.json --cache-dir CACHE_DIR --checkpoint-dir CHECKPOINT_DIR --epochs 20 --batch-size 1 --segment-beats 8 --max-tokens 512 --device auto
python -m maichart.generation.event_cli --checkpoint CHECKPOINT_DIR/event_best.pt --audio AUDIO.mp3 --output generated/chart.txt --output-ir generated/chart.chart.json --quality-report generated/chart.quality.json --level 13 --device auto
python -m maichart.cli validate generated/chart.txt --difficulty 5 --strict
python -m maichart.cli parse generated/chart.txt --difficulty 5 --ir -o generated/chart.ir.json
python -m maichart.cli stats generated/chart.txt --difficulty 5 -o generated/chart.stats.json
~~~

resume 要求同一 event data contract；同 schema 新数据使用 init-checkpoint；旧 fixed-frame checkpoint 不能转成 event_v1。

## 9. 关键内容索引

### 项目、流程和 WebUI

- README.md：总览、安装、event_v1 和旧路径。
- docs/compatibility/BEGINNER_GUIDE.md：安装、ZIP、预处理、训练、生成和检查。
- docs/compatibility/CONTINUE_TRAINING.md：resume、已有模型加强训练和 checkpoint。
- docs/compatibility/webui.md：WebUI 字段、路径、训练模式和错误处理。
- src/maichart/webui/server.py：后端 API 和任务。
- src/maichart/webui/static/index.html、app.js、style.css：前端。

### Event 主路径

- src/maichart/event_timing.py：Fraction beat、BPM 积分和 exact parser。
- src/maichart/generation/event_audio.py：音频特征、tempo 和切片。
- src/maichart/generation/event_model.py：词表和 Transformer。
- src/maichart/generation/event_cli.py：grammar 解码、生成和失败报告。
- src/maichart/generation/event_compile.py：事件到 Simai、ChartIR 和 roundtrip。
- src/maichart/generation/event_quality.py：语法、密度和条件诊断。
- src/maichart/training/event_dataset.py：manifest、过滤、segment、audio 和 condition。
- src/maichart/training/train_events.py：训练、validation、resume 和 checkpoint。
- src/maichart/training/event_checkpoints.py：event checkpoint contract。
- docs/compatibility/EVENT_GUIDE.md、EVENT_AUDIO.md、EVENT_TEST_CASES.md：event 指南、音频合同和实测记录。

### Simai、数据和测试

- src/maichart/maidata.py、timing.py、notes.py、slides.py：原始读取、旧 timing、note 和 Slide。
- src/maichart/validation.py、ir.py：验证和 ChartIR。
- src/maichart/preprocess.py、training/dataset_v25.py、training/train_generator.py：旧数据和旧基线。
- tests/test_event_pipeline.py：event 数据、grammar、失败隔离和生成。
- tests/test_event_timing.py、test_event_compile.py、test_event_quality.py：时间、编译和质量。
- tests/test_stage3_roundtrip.py：CLI、exact fallback、validate 和 stats。
- tests/test_webui.py：WebUI API、模型族和生成路径。

## 10. 交接边界

1. syntax_valid=true 不等于音乐对齐。
2. validate 通过不等于目标等级达到。
3. quality_status=success 不等于可玩，必须同时看 playable_verified。
4. 没有独立 validation 时，event_best.pt 只代表训练 loss 较优。
5. 多候选、随机采样和目标不可行处理必须用户手动开启。
6. 不要为消除旧 tick 报错而把所有 beat 量化到固定 {16}。
7. 不要静默丢弃未知 note；要么无损支持，要么跳过并记录原因。
8. 新智能体开始前先核对测试、manifest、cache、checkpoint 和 WebUI 进程是否属于同一份 event_v1 实验。
