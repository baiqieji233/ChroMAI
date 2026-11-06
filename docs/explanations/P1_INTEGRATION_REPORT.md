> 2026-10-10 校正：本文件含历史规划/历史测量。Slide、Touch 机制及完成度请以 [STAGE_BC_REPORT.md](STAGE_BC_REPORT.md) 和 [MAIMAI_MECHANICS_RESEARCH.md](MAIMAI_MECHANICS_RESEARCH.md) 为准；旧勾选不表示已完成训练/实机验收。

# P1 改动说明：模型类型识别、可玩性后处理与细碎节奏过采样

> 日期：2026-10-10
> 对应提交：`5a4b2ad` → `5bed711` → `260d2b7`
> 目标文档：[GOAL_COMPLEX_RHYTHM_PLAYABILITY_FINAL.md](GOAL_COMPLEX_RHYTHM_PLAYABILITY_FINAL.md)

本轮把之前只存在于独立模块里的东西接进了实际生成与训练链路，并新增了一条针对「细碎节奏被压平成四分八分」的训练维度。下面按「新增了什么 / 为什么 / 怎么用」说明。

## 一、模型类型识别与兼容

问题：项目里同时存在旧固定帧模型、`event_v1`、`event_ctx_v1` 以及未来的扩展模型，生成入口不同（`cli` / `event_cli` / `event_ctx_cli`）。以前只有 WebUI 内部按 schema 判断，CLI 与后处理没有统一的识别，也没有「这个 checkpoint 到底支不支持某个新功能」的概念。

新增 [model_compatibility.py](../../src/maichart/generation/model_compatibility.py)：

- `detect_model_family()` 直接读 `data_contract.target_schema`。训练器实际写的是 `event_v1` / `event_ctx_v1`（少数旧信封写 `maichart-event-*-v1`），两种拼写都接受。若 `family_params.fine_rhythm_oversample > 1`，进一步识别为 `event_ctx_v1_extended`。
- `FeatureSupport` 把能力分成两组，因为它们的生效阶段完全不同：
  - **后处理组**（冲突检测、冲突修复）：在解码之后对已编译谱面做检查，只依赖音符几何，因此**任何**事件 checkpoint 都能用，不需要重训。
  - **采音倾向组**（`rhythm_bias`、`fine_note_strength`）：影响采样，只有合同里记录了细碎节奏训练目标的 checkpoint 才认。旧模型上传这些参数会被**忽略并告警**，而不是悄悄把一个训练分布外的提示塞给模型。
- `validate_args_for_model()` 给出逐条警告，`get_generation_config_for_model()` 返回该模型真正会采用的那一组值。

这样「导入什么模型就用什么生成方式」是数据驱动的：不需要用户记住哪个 checkpoint 支持什么。

## 二、生成侧：冲突检测真正接进 CLI

之前 `--enable-conflict-detection` 只是把配置写进质量报告，并没有执行检测。现在：

- 新增 `detect_chart_conflicts()`（[post_process.py](../../src/maichart/generation/post_process.py)），直接在 `compile_event_points` 之前拿到的 event 点上工作，用真实按键编号、Hold 时长、Slide 轨迹做判定，而不是在 token 流上猜。
- `event_cli.main` 在编译成功后调用它，把结果写进 `quality.playability`（总数、硬冲突数、前 50 条明细），并打印一行 `playability_conflicts=... hard=... mode=...`。
- `--conflict-repair-mode reject` 时，出现硬冲突会让该候选失败并继续下一个候选；`none` 只报告；`local` 目前与 `none` 行为一致（修复需要重写音符几何，属于后续工作，已在这份文档标注）。

### 检测规则的修正

第一版检测器在这份真实谱面上报了 172 条，其中绝大多数是误报。逐一核对后修了四处几何/阈值问题：

| 问题 | 现象 | 修正 |
|---|---|---|
| Slide 半转直线 | `2-6`（正好 180°）两个方向等长，代码随机走了「经过 1/8/7」的那条，于是把中段按键误判成被星星占用 | 180° 无方向直线返回「不在任何按键上」，不猜路径；`<`/`>` 仍按显式方向 |
| 星星判定容差 | 15° 容差接近按键间距（45°）的三分之一，任何 Slide 的中段都会落在某个按键的容差内 | 收紧到 6° |
| Slide 头同刻 Tap | `1` 与 `1-5` 同时出现是合法连招，星星在按下瞬间就离开头部 | 起始时刻（±1ms）的 Tap 不再判冲突 |
| 和弦被当成跳跃 | 同一时刻的两个按键被算成「0ms 内跨 1.85」，每个和弦都报一次 | 时间差 <1ms 视为和弦，交给握法规则处理 |

软冲突阈值同样按「实际演奏难度」重定：跳跃距离改为 2.0（全直径，1–5 / 3–7 对角线）且窗口 0.10s；单手两键跨度放宽到 135°（`4+7` 这类常用和弦不再报）；快速交替由「3 个音 167ms」改为「连续 ≥5 个左右手交替且平均 <100ms」，因为 180 BPM 的八分颤音本来就是 167ms，属于正常写法。

修正后同一谱面：**13 条（3 硬 + 10 软）**，3 条硬冲突经人工核对全部为真（Tap 落在 Slide 落点按键上，两者抢同一次按键判定）。

## 三、训练侧：细碎节奏窗口过采样

这是本轮针对「复杂节奏被简化」的核心手段。新增两个参数：

- `--fine-rhythm-oversample`（默认 1，关闭）：把「细碎节奏窗口」额外复制 N 次。
- `--fine-rhythm-ratio-threshold`（默认 0.25）：判定阈值。

判定标准是**窗口内相邻起音间隔小于四分之一拍的占比**达到阈值。用拍而不是秒做单位，所以慢速下的 12 分与快速下的 12 分同等对待——这条维度关心的是网格细分，不是 NPS。

它与已有的 Slide/Touch 过采样**相乘**而不是相加，并进入采样合同（`family_params` 与 `SAMPLING_CONTRACT_KEYS`），因此：

- 改变它必须用 **finetune**（`--init-checkpoint`），普通 resume 会因合同不一致而拒绝；
- 验证集始终按原始分布评估，不受倍率影响（沿用已有做法）。

设计上刻意保守：过采样只让模型**更多看到**这些写法，不会凭空补出音符；真正决定采音的是训练目标本身。

## 四、WebUI

生成页新增「可玩性约束（实验性）」区块：启用冲突检测、修复模式（仅检测 / 局部修复 / 拒绝生成）、软冲突开关。选中 checkpoint 后，检查结果会显示能力行，例如：

> 该 checkpoint 支持冲突检测（后处理，无需重训）；但未在细碎节奏目标上训练，「节奏倾向 / 细碎音符强度」将被忽略。

训练页新增「细碎节奏窗口过采样」与「细碎节奏判定阈值」两个控件，附中文说明与默认值来源。`/api/status` 增加 `fine_rhythm_oversample` 能力位，checkpoint 检查结果新增 `generation_capabilities`。

顺带清理了三个「有控件但后端不读」的字段（硬冲突阈值、最大手部跳跃、Slide 精度/Touch 重叠开关），避免界面显示与后端行为不一致。

## 五、验证

- 新增 [tests/test_playability_post_process.py](../../tests/test_playability_post_process.py)：11 项，覆盖按键布局、Hold-Tap 硬冲突、Slide 头和弦、半转轨迹、弧线方向、和弦不误报、短颤音不误报、reject/none 语义、模型族识别与能力门控。
- 同步更新 `tests/test_event_ctx.py` 的合同期望并补细碎节奏参数断言。
- 全量：`784 passed, 11 skipped`。

## 六、还没做

- `local` 修复模式尚未真正改写音符，当前等价于只检测。
- 采音倾向（`rhythm_bias` / `fine_note_strength`）目前只做到「门控与告警」，尚未在生成时把倾向转成条件向量——它要等一个在细碎节奏目标上训练过的 checkpoint 才有意义。
- 细碎节奏过采样尚未在此 checkpoint 之外做 A/B；需要一次 finetune + 同曲对照才能评估采音变化。
