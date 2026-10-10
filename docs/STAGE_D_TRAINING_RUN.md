# 阶段 D：full 训练与串行对照（运行中）

日期：2026-10-10。本报告不表示验收完成。

> 此文件保留最初任务的历史记录。2026-10-10 23:42 已从完整 epoch 4 转到 7863 续训，并以真正留出的 10 首音频和完整 tempo map 替换旧 panel。当前任务、队列和证据以 [STAGE_D_RESUMED_RUN.md](STAGE_D_RESUMED_RUN.md) 为准；不要重新启动下述旧队列。

## 已启动的真实训练

- WebUI：`http://127.0.0.1:7862`，与此前 7861 实例独立。
- 初始任务：`d32a25d95f70`，通过 `/api/train` 提交。
- 日志：`webui_workspace/logs/20261010_221712_train_d32a25d95f70.log`。
- 模型目录：`checkpoints/music_structure_v3_full_stage1`。
- 源数据：`webui_workspace/datasets/full训练集/training_manifest.json`，不混同此前 composed 数据短跑。
- 数据划分：按完整音频文件内容 SHA-256 分组，训练/验证哈希交集为 0；派生清单含绝对源路径，原 manifest 不改写。审计在 `.scratch/full_v3_stage1/split_report.json`。
- 入选清单训练 1,695 个难度，实际可解析训练 1,681 个/41,607 窗；验证 177 个/4,431 窗。差额及原始原因见 checkpoint 目录 `dataset_audit.json`。

配置：v3、320 维、8 层、8 头、FFN 1280、每窗最多 1024 tokens、上下文 16 拍/320 tokens、音频 margin 8 拍、全曲 256 帧。microbatch 8、梯度累积 4、AMP 开启、学习率 0.0002、显存预算 7.2 GiB、音频内存预算 4 GiB。目标 15 epochs，每轮保存，Slide/Touch/细碎增采样先全部为 1。

采用 batch 8 是基于此前同规模、FP32 的 composed 一轮已实测峰值约 7.14 GiB，再结合本轮 AMP；没有把微型模型代替正式规模验收。音频预热完成后有 1,330 首驻留内存约 3.41 GiB，无 spills。

## 第一轮证据

2026-10-10 22:44:27，第一轮训练结束：

| 指标 | 值 |
|---|---:|
| 训练 token loss | 1.59046 |
| 验证 token loss | 1.48322 |
| 训练 token accuracy | 0.4987 |
| 验证 fine-rhythm MAE | 0.1706 |
| 训练秒数 | 980.0 |
| 每 batch 毫秒 | 188.4 |
| 峰值显存 GiB | 7.15 |

`event_epoch_001.pt` 已核对：epoch=1，profile=`music_structure_v3`，plan_dim=13，来源是本次 full 派生清单。已用它经 WebUI 对真实歌曲摘录生成并编译成功，旧 v2 同样可用；产物在 `.scratch/webui_v3_generation_check`。第一轮模型仍偏稀疏，不能凭这些数值宣布谱面质量达标。

## 自动继续的队列

控制器：`tools/run_staged_rhythm_validation.py`，配置 `.scratch/full_v3_stage1/pipeline_config.json`；持久状态 `pipeline_state.json`，运行报告 `PIPELINE_REPORT.md`。同一状态文件采用系统文件锁防止重复启动。

依赖顺序：

1. 观察已有初始任务完成，不重复提交。
2. 使用阶段 1 `event_best.pt` finetune 5 轮，独立目录 `checkpoints/music_structure_v3_full_finetune`，学习率 0.00005，细碎窗口倍率 3，Slide/Touch 倍率仍为 1。
3. 固定 10 首音频分别生成旧 v2、v3 阶段 1、v3 强化后，以及强化模型偏好 0.25/0.75 的 5 组结果，共 50 次完整生成；固定种子、BPM、first 与其余采样参数。
4. 输出分项评估 JSON 和阶段报告，不汇总成未经校准的“质量分”。

训练失败或取消时停止后续依赖；生成失败可以继续记录其它曲目，最终返回 `completed_with_errors`。网络观测失败保留原 task ID，提交超时保留 `submit_uncertain`，不会盲目重复 POST。控制器退出不会取消远端训练。

## 验收边界

- 固定 panel 中 9 首属于本次训练集、1 首验证集，属于功能/条件响应对照，不是盲测泛化分数。来源及内容哈希保存于配置。
- panel 中有一首参考谱为变 BPM，而当前事件生成 API 使用稳定的全局 BPM；已明确记录时序限制，不把该曲参考时序误差全归因于模型能力。
- 静态可玩性检测并非实机 Perfect 与人体工学认证，复杂轨迹未覆盖内容在报告中列出。
- 尚未获得 15+5 轮训练结束、50 份生成和人工试玩结果。“细碎采音提高”“无理率降低”“可玩性提高”均保持待验证。
