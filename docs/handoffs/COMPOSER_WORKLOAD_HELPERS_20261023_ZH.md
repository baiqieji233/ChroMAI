# Composer 负荷辅助模块交接文档（中文版）

原文：[COMPOSER_WORKLOAD_HELPERS_20261023.md](COMPOSER_WORKLOAD_HELPERS_20261023.md)\
日期：2026-10-23

## 范围

新增的 `composer/workload.py` 提供原谱负荷（authored workload）监督标签和已提交前缀（committed prefix）统计量。本模块不修改模型、训练、生成或 WebUI 代码。

## 变更内容

- `event_statistics` 使用现有的音符解析器解析音符语法。它统计去重后的起始位置（distinct starting positions），而非简单累加类型计数器或 Slide 分支数。紧凑双押（compact chords）和 Touch 区块均保留为独立的触发头（heads）。三个标志位分别表示事件是否包含 Hold/Touch Hold、Slide 以及 Touch/Touch Hold；它们是可重叠的事件类别，而非互斥概率或所需玩家手数。遇到未知语法会直接抛出错误，而不是静默编造监督信号。
- `build_workload_targets` 从精确的变速表（tempo map）和 `first` 计算绝对秒数，使用半径为 1 秒的三角核（triangular kernel）计算事件率/头率及类型占比。三角核面积会被裁剪到实际音频有效区间内，从而避免前奏/尾奏部分窗口中的头密度被静默拉低。速率编码为 `log1p(rate)`，不设密度上限。类型标签为加权的原谱事件占比，独立于 RMS 或峰值证据。通过计数与时间乘积的前缀积分，重复查询的时间复杂度对事件数呈对数级 $O(\log N)$。
- `causal_ledger` 返回过去 1 秒和 4 秒内的事件率/头率，随后是距最近一次前缀事件的对数时间差，以及可见前缀时间跨度除以 4（截断至 1）。它仅包含时间不晚于各查询时刻的已提交历史事件；未来的候选点不会把自身计入其中。
- `tests/test_composer_workload.py` 覆盖了同头多分支 Slide、紧凑双押、Touch Hold 标志位、音频时钟偏移、变速 BPM、部分窗口归一化、空历史以及候选时刻因果性。测试已编写完成，但在本编码阶段按约定暂不执行。

## 集成合同（Integration Contract）

`WORKLOAD_DIM=5` 包含 `(event_rate, head_rate, hold_fraction, slide_fraction, touch_fraction)`，其中前两项为对数速率（log rates）。`LEDGER_DIM=6` 包含 `(event_rate_1s, head_rate_1s, event_rate_4s, head_rate_4s, log_seconds_since_last, history_span_4s)`，其中前四项速率为线性速率。请使用导出的通道名称元组，避免依赖隐式推断的列顺序。

负荷（workload）数值仅作为监督标签使用。将这些原谱标签直接喂入决策输入会泄漏参考谱面；生成阶段使用的模型必须改为接收其自身基于音频预测出的负荷值。而台账（ledger）可以作为模型输入，因为其内容完全来自已提交的前缀事件。

## 证据与边界

现有的 `form/curve.py` 提供了版本化的原谱头密度目标，`stream/grounded_model.py` 提供了基于时间积分的乐句查询。本辅助模块保留了它们基于真实秒数的做法，同时补充了区分开的事件率/头率以及独立的 Hold/Slide/Touch 事件占比。它不把响度等同于难度，不强制执行硬性密度预算，也不直接证明音乐质量或可玩性。
