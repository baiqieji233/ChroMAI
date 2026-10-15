# 质量控件默认值、训练同步点与隐患复查（2026-10-15）

本记录供后续整理。本轮没有启动训练，也没有读取训练集做测量；耗时依据来自已有日志 `webui_workspace/logs/20261014_234252_form_train_d69dff3da685.log`（song_form_v2，d_model 320，6 层，batch 4 × 累积 8，bf16，loader_workers 0）。

## 1. 撤回的内容

上一轮加入的“速度优化”没有实际作用，已全部删除：

- `src/maichart/form/fast_attention.py`：只替换 `nn.MultiheadAttention` 的直接子模块，而本项目的注意力都在 `TransformerEncoderLayer` / `TransformerDecoderLayer` 内部，所以一个也替换不到。它的交叉注意力会直接抛错，还忽略 padding mask。本机也没有安装 `flash_attn`。PyTorch 2.7 的 `nn.MultiheadAttention` 在 `need_weights=False` 时本来就走 SDPA 融合核，额外的 flash-attn 收益很小。
- `--compile`：`torch.compile` 需要 triton，本机（Windows）没有安装；`reduce-overhead` 模式还要求固定形状，而本项目每个 batch 的长度都不同。另外它在 `model.validate_inputs` 赋值之前包装模型，checkpoint 的 state_dict 键名也会变成 `_orig_mod.*`。
- `--precompute-cache` / `enable_precompute`：只保存了一个参数和空字典，没有实现。
- 根目录 `TRAINING_SPEED_OPTIMIZATION_COMPLETE.md`：内容与事实不符，已删除。

## 2. 训练速度：移除逐 batch 的 GPU→CPU 同步

日志中每轮约 5040 秒，32,709 个 layout batch，其中 data_wait 约 1016 秒、compute 约 3953 秒。loader_workers=0 时，下一批数据在主线程准备。如果每个 batch 内都有阻塞等待 GPU 的操作，CPU 准备数据和 GPU 计算就无法重叠。

在本机 CUDA 上用 `torch.cuda.set_sync_debug_mode('error')` 确认，旧代码每个 batch 有以下强制同步，已逐一移除：

| 位置 | 原因 | 修改 |
|---|---|---|
| `model.py` `decode()` | `TransformerDecoder` 收到 `tgt_mask` 后，会逐次比较它是否为因果 mask（`.all()` 转 bool） | 传入 `tgt_is_causal=True`，mask 本来就是 `triu` 构造的因果 mask |
| `layout_metrics.py` | 每个 batch 用 Python 列表 `torch.tensor(..., device=cuda)` 建两个索引张量 | 改为按设备缓存的布尔查表，一次构建 |
| `rational_objectives.py` | `logits[..., (at_id, eos_id)]` 元组高级索引会把索引复制到 GPU | 改为两次整数索引后 `stack`，数值完全一致 |
| `train.py` 训练循环 | 每个 batch `if not torch.isfinite(loss_sum)` | 把有限性标记留在 GPU 上累积，在优化器更新前统一检查。仍在任何参数更新之前报 `Nonfinite layout loss` |

现在 layout 阶段的强制同步只剩：每次优化器更新一次（梯度范数判断），以及进度日志每约 10 秒一次。GPU 最多可以领先 CPU 一个累积组（默认 8 个 batch）。

**收益未测量。** 理论上限是把每轮约 1000 秒的 data_wait 隐藏到 GPU 计算之后。注意：进度日志里的 `compute_sec` / `data_wait_sec` 是主机墙钟时间。移除同步后 GPU 时间会更多地计入后续的同步点，这两个字段的比例会变化，不能直接与旧日志对比，应比较每轮总耗时。

本轮未处理、可以后续单独试验的选项：`--loader-workers > 0`（现有参数）。Windows 使用 spawn，每个 worker 复制一份数据集元数据和缓存，内存会随 worker 数增长，需要实际观察。

## 3. 其他隐患修复

- **重复惩罚损失没有梯度**：`compute_repetition_penalty_loss` 用 `torch.tensor(数值)` 返回常数，与 logits 之间没有计算图，开启后只会多出逐样本的 Python 计算。上一轮把它的界面控件解除禁用并写了“已集成”，这是错误的，现已改回禁用。后端 `_validate`、`layout_loss`、`/api/form/train` 都会拒绝非 0 值；resume 时强制为 0，因为它从未产生梯度，数值上等价。
- **v1 起音骨架偏移和添加会产生生成时不存在的拍位**：原实现用 `beat_at_second`，它明确不做分母吸附，会得到类似 `2469135/100000` 的分数。生成时的骨架分母最多为 48。现改为与生成相同的 `exact_candidate` 取整，并限制在负 first 时不产生负拍位。逐点取整约 50 µs，只对进入骨架记忆窗口的起音执行。
- **index.html 引号损坏**：上一轮写入的 7 个扰动控件使用了中文弯引号作为属性定界符（`class=”field”`、`id=”…”`）。浏览器因此不识别这些 id 和 class，样式与周围不一致，`valueOrDefault` 也找不到这些输入框而回退为代码里的 0 / 1，这就是界面上出现裸 “0” 的原因。已重写，并加测试禁止这种写法。

## 4. WebUI 默认值（song_form 训练页「训练鲁棒性与配置多样性」）

| 控件 | 新默认 | 理由 |
|---|---|---|
| 快捷方案 | 推荐：轻度曲线 + 历史扰动 | 与下列默认值一致；“全部关闭（基线）”保留用于严格对照 |
| 扰动应用概率 | 0.5 | 一半样本保留原始输入，避免模型只见到被扰动的条件 |
| 曲线噪声标准差 | 0.03 | 均值为 0，不改变曲线整体高低，只模拟小幅偏差 |
| 曲线整点置零概率 | 0（不推荐） | 曲线置零处的监督答案仍有音符，会教模型“零负荷处也可以写音符”，削弱低负荷段 |
| 历史配置扰动概率 | 0.02 | 只改单独 Tap 的相邻按键，保留拍位、多押、Hold / Slide |
| 起音骨架删除概率（仅 v1） | 0.05 | 模拟 Planner 漏采，开销低 |
| 起音骨架添加 / 偏移（仅 v1） | 0 | 每个点都要逐点取整，会拖慢数据准备；取整容差约 10 ms，小偏移大多回到原位 |
| 配置表示多样性损失权重 | 0（不推荐） | 作用于隐藏表示而非实际配置，可能削弱合理的重复与副歌呼应 |
| 重复惩罚损失权重 | 0，禁用 | 见第 3 节 |

一致性：HTML `value`、`app.js` 提交时的回退值和“推荐”快捷方案三处相同，由新测试 `tests/test_quality_controls_ui.js` 保证。

已保存设置的迁移：`restoreSettings` 发现浏览器保存的仍是旧的“全部关闭”默认（且快捷方案为 off）时，一次性丢弃这些值，改用新默认；用户手动改过的值和其他方案保持不变。新保存的设置带 `__songFormQualityDefaults: 2` 标记，迁移不会重复执行。

后端 API 不带这些字段时仍按旧的关闭值处理，保持 CLI / 脚本调用的兼容性；新默认只由主界面显式提交。resume 继续继承 checkpoint 中的原值。

扰动启用时，draft 样本会绕过样本缓存。旧日志显示该缓存每轮命中极少（约 266k 次未命中、几十次命中），影响可以忽略。

## 5. 验证

- 完整 Python 测试：2039 通过，11 跳过。
- 全部 JS 测试：83 通过。
- 新增 `tests/test_form_training_sync_safety.py`：查表与原成员判断一致；AT/EOS 损失与旧索引方式逐位相同；CUDA 上在 sync debug 模式下完成前向、损失和反向传播，不触发同步；注入 NaN 时仍在任何参数更新之前停止；重复惩罚被拒绝；偏移、添加后的骨架分母都在生成网格内。
- `tests/test_quality_controls_integration.py` 增加 API 拒绝重复惩罚的用例。
- 用本机 Edge 无头模式打开正在运行的 7860 WebUI：控件与相邻的“段落关联辅助损失权重”使用相同的 `field` 网格样式；v2 下隐藏 v1 专属项；页面无脚本错误。截图保存在 `.scratch/ui_check/`。

## 6. 训练音频内存峰值修复（本次补充）

日志 `20261015_163744_form_train_faca1c0e1785.log` 的失败发生在
`DataLoader worker 3 -> rational_audio.acoustic_at_times -> librosa.load`。
该调用把约 7,355,077 帧的双声道文件一次性读入，申请约 56.1 MiB；同时本次训练使用
`loader_workers=8`、音频缓存上限 2 GiB，多个 worker 的数据集与缓存会放大峰值。

当前修复新增 `src/maichart/form/waveform_cache.py`：

- 原生可读格式先按单声道 `float32` 分块写入只读 `.npy` memmap。
- 不同 DataLoader worker 映射同一个文件，操作系统共享页缓存，不复制整首双声道数组。
- 查询特征只从 memmap 取局部窗口，仍保留原采样率与原查询中心。
- cache 发布使用目录锁与临时文件，避免并行 worker 看到半成品。
- 原有 `waveform()` 仍保留给完整攻击检测和兼容调用；v2 局部训练查询走共享 memmap。
- preparation cache 的代码依赖合同包含 `waveform_cache.py`，修改读取策略会自动使旧准备索引失效重建。

真实音频逐样本对照结果：`full_read_equal=true`、`query_equal=true`、误差为 0。窗口方案不会改变训练输入数值，只改变读取方式。

建议当前设备把 `数据加载 worker` 从 8 降至 2–4，`准备进程数` 保持 4 或更低；这属于内存/吞吐折中。共享 memmap 修复降低了每 worker 的波形峰值，但不意味着可以无限增加 worker。对于当前这次任务，建议先用 `数据加载 worker=2`、`预取批次数=1`、`音频读取方式=按预算缓存`、`音频缓存上限=1–2 GB` 续训；确认稳定后再逐步增加 worker。

## 7. 仍过时、需要整理的文档

以下文档包含已撤回或错误的结论（Flash Attention / torch.compile 加速、重复惩罚“推荐 0.1–0.15”、分阶段扰动表等），应以本记录为准：

- `WORK_COMPLETION_SUMMARY.md`
- `QUALITY_IMPROVEMENTS_2026-10-15.md`
- `FURTHER_OPTIMIZATION_ROADMAP.md`
- `docs/TRAINING_PARAMETERS_GUIDE.md`
- `docs/QUALITY_CONTROLS_WEBUI_HANDOFF_20261015.md` 中“默认全部关闭”的描述

`src/maichart/form/planner_guidance.py` 与 `repetition_penalty.py` 中除 `compute_pattern_diversity_loss` 以外的函数目前没有调用方。
