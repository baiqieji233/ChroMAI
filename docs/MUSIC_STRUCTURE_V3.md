# music_structure_v3：细碎节奏进入规划器

> 日期：2026-10-10
> 提交：`5e44148`
> 上游文档：[PROGRESS_AND_NEXT_STEPS.md](PROGRESS_AND_NEXT_STEPS.md) 的「阶段 A」

阶段 A 的目标是把细碎节奏从「训练时多看一眼」变成「模型能表示、能被要求」。
本文记录它的实现与实测结论。

## 为什么需要这一层

在此之前，细碎节奏只存在于采样侧：`fine_rhythm_oversample` 让模型更多看到
这些段落，但模型的规划器只有 8 个连续通道（count/density/chord/slide/last/
touch/hold/intensity），没有任何一维表示「这一段该用多细的采音」。结果是
`--rhythm-bias` 与 `--fine-note-strength` 无处可写，只能被忽略并告警。

强度曲线之所以可控，正是因为 `intensity` 是规划器的训练目标；细碎节奏缺的
是同一个位置。

## v3 做了什么

`music_structure_v3` = v2 的 8 个连续通道 + `fine_rhythm`，共 9 个连续通道
与 4 个配置 logits，因此规划器头宽 13、原始目标宽 10。

一个实现细节值得单独说：原先代码在 18 处直接比较
`planner_profile == "music_structure_v2"`。这类字符串判断在新增 profile 时
会静默走 legacy 分支——不报错，只是行为悄悄退化。现在改成
`is_music_structure()` 与 `continuous_keys()`，宽度由 `PLAN_DIM` /
`PLAN_TARGET_DIM` / `CONTINUOUS_COUNT` 三张表推导。

## 细碎节奏怎么定义（以及为什么加容差）

定义：**相邻起音间隔不超过四分之一拍**的占比，含 1/128 拍容差。用拍而非秒，
所以同一段音型在 90 BPM 与 200 BPM 下判定一致——这一维关心网格细分，不是 NPS。

容差不是凑数，它决定了这一维有没有信号。实测 150 首真实 Master/Re:Master、
3905 个 16 拍窗口：

| 定义 | 达到 0.25 的窗口 | 均值 | p75 | p90 |
|---|---|---|---|---|
| 严格 `< 1/4` | 4.2% | 0.059 | — | — |
| 含 1/128 容差 | 27.3% | 0.168 | 0.282 | 0.593 |

严格定义之所以失效，是因为它把**普通 16 分音符排除在外**——而 16 分正是快速
钢琴与鼓点最常见的写法。一个在 94% 的窗口上都等于 0.06 的目标，规划器只会
学会预测它的均值。容差让 16 分计入、8 分不计入：

```
8 分音符（1/2 拍）   -> fine_rhythm = 0.0
16 分音符（1/4 拍）  -> fine_rhythm = 1.0
24 分音符（1/6 拍）  -> fine_rhythm = 1.0
```

采样侧与训练目标共用 `music_plan_targets.fine_note_ratio`（`event_ctx_dataset`
委托调用），避免「哪些窗口被过采样」与「模型被要求预测什么」各写一份而漂移。

## 怎么用

训练（必须 fresh，换 profile 不能续训）：

```bash
python -m maichart.training.train_event_ctx \
  --planner-profile auto ...        # auto 在 fresh 时即 v3
```

生成（需要 v3 checkpoint，否则该参数被忽略并告警）：

```bash
python -m maichart.generation.event_ctx_cli \
  --checkpoint <v3.pt> --fine-note-strength 0.9 ...
```

WebUI：训练页新增 `music_structure_v3` 选项；生成页选中 v3 checkpoint 后，
能力提示会显示「支持冲突检测与采音倾向」。

## 验证

- 真实数据 fresh 冒烟（2 步）跑通，planner 报告含 `fine_rhythm_mae`；
  checkpoint 记录 `planner_profile=music_structure_v3 / plan_dim=13`，
  并被识别为支持采音倾向。
- 用该 checkpoint 生成完整谱面，编译成功；`fine_note_strength` 改变时
  注入 decoder 的 planner token 确实变化。
- 新增 `tests/test_music_plan_v3.py` 13 项；全量 799 passed, 11 skipped。

## 还没做（原阶段 B–D，未动）

1. `local` 修复模式仍等价于只检测；冲突修复未接入生成链路。
2. 跨窗口边界的 Hold/Slide 不在检测范围内。
3. 标注脚本与实际 manifest 结构不符（按 §四 结论它本可删除，但保留为
   统计工具的改造尚未做）。
4. 未做真实训练与对照生成，因此「细碎采样率提升 30%」等验收指标仍未验证；
   `fine_note_strength` 目前只在冒烟级别确认「能传进去」，尚未证明
   「0.2 与 0.9 出来的谱面真的不同」。
