# 质量改进 WebUI 接入交接记录

日期：2026-10-15

本次工作在[质量改进记录](archive/20261015/QUALITY_IMPROVEMENTS_2026-10-15.md)和[工作完成报告](archive/20261015/WORK_COMPLETION_SUMMARY.md)所描述的代码基础上，把已经存在的训练参数接入主 WebUI。控件位于训练页「条件、正则化与规划学习」下的「训练鲁棒性与配置多样性」折叠组，没有新增独立页面。

## 已接入并实际生效的控件

### 输入扰动

- `扰动应用概率` → `perturb_apply_prob`
- `曲线噪声标准差` → `perturb_curve_noise_std`
- `曲线整点置零概率` → `perturb_curve_dropout_prob`
- `历史配置扰动概率` → `perturb_history_corruption_prob`
- song_form_v1 专属：
  - `起音骨架删除概率` → `perturb_onset_drop_prob`
  - `起音骨架添加强度` → `perturb_onset_add_prob`
  - `起音骨架偏移标准差（毫秒）` → `perturb_onset_shift_ms`

曲线和历史扰动会进入 `input_example()`。样本缓存会在启用扰动时旁路，且每个样本使用稳定的独立随机种子，避免一个样本的第一次随机结果被整个训练过程永久复用。曲线噪声使用同一随机源，保证 `seed` 可复现。历史扰动只改独立 Tap，避免破坏 Hold/Slide 时值和路径。

v2 联合生成拍位，不读取 v1 的起音骨架，所以 v2 页面会禁用三项起音骨架控件并在 payload 中强制为 0。

### 多样性辅助目标

- `配置表示多样性损失权重` → `diversity_loss_weight`

它对模型返回的 decoder hidden states 分段计算相邻表示相似度，属于实验性训练目标。默认 0，可能削弱副歌合理重复，建议只在单独的 finetune 实验中启用。

本次接入同时修正了该辅助项的 padding 遮罩、逐样本归一化与质量权重，并改为只计算附近 1–3 组的相似度，避免构造完整平方矩阵。启用后训练日志会列出 `diversity_loss`。没有改变基础 token CE，也没有声称这项辅助目标已经通过音乐质量验收。

## 显示但禁用的功能

`重复惩罚损失权重（待接入）` 保留说明但始终禁用。当前 `compute_repetition_penalty_loss()` 需要真实的历史生成前缀；现有 teacher-forced layout 训练调用不能提供这个前缀，开启会产生没有有效训练意义的数值。Planner Guidance 模块同样尚未接入 `FormTransformer.forward()`，因此没有伪造一个“已启用”开关。

更具体地说，当前重复损失在没有前缀时返回 0；即使传入前缀，该函数构造的结果也未依赖 logits 的可导概率。不能只补传一个参数就认定重复训练已生效，后续需设计真正可导的事件级目标。原有两个总结和另行提供的路线文档属于其他智能体的交付记录，遇到生效范围冲突时应以本交接记录及实际代码为准。

## 快捷方案

- `全部关闭（基线）`
- `轻量曲线扰动`：应用概率 0.3、曲线噪声 0.02
- `曲线 + 少量历史扰动`：应用概率 0.5、曲线噪声 0.03、曲线 dropout 0.02、历史扰动 0.03
- `自定义`

快捷方案只设置本组输入扰动和多样性值，不覆盖学习率、模型结构、数据集或 checkpoint。

## resume / finetune 行为

- `resume`：从 checkpoint 的 `args` 读取上述参数并锁定；旧 checkpoint 未记录时使用关闭值。
- `finetune`：控件保持可编辑，输出目录必须使用新目录。
- `fresh`：使用页面当前值。

后端 API 与 CLI 参数保持原名；WebUI 不会把“尚未接入”的 Planner Guidance 或重复损失伪装成真实功能。

## 验证

- JS WebUI 回归：本轮相关 23 项通过，新增质量控件 payload 和 resume 锁定测试覆盖 v1/v2 行为。
- Python API/训练回归：本轮相关 93 项通过。
- `python -m compileall -q src tests` 通过。
- 预览 WebUI `127.0.0.1:7862` 已验证桌面/移动布局、快捷方案、v1/v2 模型切换和 resume/finetune 锁定；主页面 `127.0.0.1:7860` 已确认包含新控件。没有启动训练。
- 临时预览服务已停止；现有 7860 服务未被本次操作停止或替换。
- 修改前备份：`backups/quality_controls_webui_20261015/before_webui_integration.zip`。

## 运行服务与交接状态

代码与主页面静态控件已落盘。现有 7860 进程早于本次后端修改启动，因此应在下次训练前重启 WebUI，确保 resume 参数继承等后端更新全部生效。自动审批拒绝了本次重启 7860 的操作，返回 `blocked by policy`，未给出更具体原因；本次没有绕过该拒绝。独立 7862 预览服务用于核对新代码，已关闭。

没有启动正式训练或全量测试；仅运行上述定向检查。新控件默认保持增广幅度及实验损失为 0，快捷方案也不自动启用多样性或未接入目标。

## 后续工作

若要启用重复惩罚，应先把训练目标改为 prefix-aware 的事件/配置级重复指标，并在生成侧把按 token 的采样惩罚改为按事件配置窗口计算；若要启用 Planner Guidance，应修改模型 forward、checkpoint 合同、训练输入和生成推理链路后再添加控件。
