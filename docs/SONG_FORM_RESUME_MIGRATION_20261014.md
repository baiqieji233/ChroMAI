# song_form_v1 续训数据变化修复记录

日期：2026-10-14。

## 本次失败的根因

故障日志：`webui_workspace/logs/20261014_122741_form_train_be526cb85152.log`。

来源模型：`checkpoints/Prism_Plus_v1/form_last.pt`。使用的训练方式为 resume，目标输出目录为 `checkpoints/Prism_Plus_v2`。

模型记录和本次准备结果如下：

| 项目 | checkpoint 中的记录 | 本次准备结果 |
| --- | --- | --- |
| 训练清单 SHA256 | 相同 | 相同 |
| 拍号配置 | 无显式全局配置 | 相同 |
| 区域时长上限 | 14 秒 | 14 秒 |
| 可用谱面 | 1859 张 | 1866 张 |
| 配置训练样本 | 130783 个 | 131328 个 |
| 已有谱面的源文件/音频/质量标注/权重 | 无变化 | 无变化 |
| 已有谱面的派生关联记录 | 原记录 | 5 张的 links 数量变化 |

先前提交 `d075033` 修正了 `||` 行注释的解析，使 7 张之前被误排除的谱面重新可用。段落匹配修复也改变了部分派生关联记录。因此“清单文件没有变化”不代表“解析后的训练样本和顺序没有变化”。

checkpoint 已完成 11 轮；当前保存的是第 12 轮配置阶段的中途状态，已完成 16000 个配置 batch。其样本游标建立在旧的 1859 张谱面之上。直接把该游标套到新的 1866 张谱面上，会跳错或漏掉数据。原有 strict 拒绝是必要保护；问题在于错误没有说明差异，并且没有适合本次解析修复的迁移入口。

已用实际 checkpoint 和当前缓存索引核对：新增 7 张、移除 0 张、派生记录变化 5 张、旧素材身份变化 0 张。1859 个旧难度记录对应的 1476 份 maidata 原文件哈希全部通过检查。

## 新增的两种续训策略

- `strict`：默认策略。完整数据合同和游标均兼容时，继续原进度。
- `restart_epoch`：显式迁移策略。同一清单、相同源素材与权重标注，仅可用谱面增加、派生结构记录或顺序变化时，保留模型和优化器状态，清空旧的半轮样本游标，从 `checkpoint.epoch + 1` 的规划阶段重新开始。

迁移保留配置模型、Planner、优化器动量、优化器计数、适用时的 AMP scaler、任务能力记录、已完成轮次和训练历史。迁移事件会写入后续 checkpoint 的 `data_migrations`。

“重跑当前轮”不会把权重回滚到第 11 轮末尾：第 12 轮中途已经学到的更新仍然保留，只是不再沿用旧的第 16000 批位置。部分样本会重新训练，这是这项显式迁移的取舍。

以下情况不能使用这项迁移：清单变化、原有谱面删除、已训练源文件/音频变化、质量或训练权重变化、有效 first 变化、布局合同变化、身份哈希不足或重复身份。此时仍使用 finetune。finetune 会加载已学权重并开始新的优化阶段，不等于从随机权重开始。

迁移必须使用新的输出目录，避免覆盖来源模型。数据合同不兼容时，输出 `resume_contract_diff.json`，并在日志中输出 `form_resume_contract`；迁移成功时输出 `form_resume_migration`。清单和原始 maidata 变化会在昂贵的数据准备之前被发现；音频和质量等变化仍由准备后的完整合同检查。

## WebUI 正确操作

1. 在主页面选择模型族 `song_form_v1`，训练方式选择“从断点继续（resume）”。训练来源保持 full 原清单。
2. 在“已有模型 checkpoint”路径框填写 `E:\simai_transformer\checkpoints\Prism_Plus_v1\form_last.pt`，点击“检查 checkpoint”。
3. “续训数据变化处理”选择“解析结果变化时，保留状态并重跑当前轮”。这个控件只在 song_form 的 resume 模式显示。
4. 输出目录填写 `E:\simai_transformer\checkpoints\Prism_Plus_v2`，保留原模型目录。
5. 选择本次继续训练的轮数，再开始训练。`epochs=1` 会从第 12 轮开始，完成第 12 轮；不是只补跑原来的剩余批次。

配置、学习率和采样等严格续训参数仍继承来源 checkpoint。想主动调整这些训练条件时，使用“已有模型加强训练（finetune）”。

产生 v2 checkpoint 后，后续可加载它并恢复 strict 策略，继续新合同的批次进度。

## CLI

```powershell
python -m maichart.form.train `
  --manifest "webui_workspace/datasets/full训练集/pretrain_manifest.json" `
  --cache-dir "webui_workspace/datasets/full训练集/cache" `
  --checkpoint-dir "checkpoints/Prism_Plus_v2" `
  --resume "checkpoints/Prism_Plus_v1/form_last.pt" `
  --resume-data-policy restart_epoch `
  --device cuda --epochs 1
```

其他性能参数可按原运行设置添加。模型结构、合同相关参数、优化器学习率和精度继承来源模型。

## 为什么缓存全部重建

结构准备缓存以源文件、音频、配置和相关解析代码的内容哈希为身份。此前解析器及段落匹配代码变化后，旧缓存必须失效，才能避免继续使用错误解析结果。

本次故障显示重建 1866、损坏 0，不能据此判断缓存损坏。本次新增的训练/API/前端代码不属于结构准备缓存依赖；没有后续素材、配置或准备代码变化时，再次准备应复用已重建的缓存，但仍需要执行索引与完整性检查。

## 验证与运行状态

- 定向 Python 回归：98 项通过，覆盖合同、游标、CPU 真实迁移训练、API、原有 fresh/finetune/resume 和模型预检。
- Node 前端回归：26 项通过，覆盖策略提交、模型族/训练模式显隐和帮助文字。
- CPU 真实迁移验证了模型和 Planner 初始权重、AdamW 的 step/exp_avg/exp_avg_sq/param_groups 恢复，并验证从第 12 轮规划第 0 项重新开始。
- 实际 checkpoint 只读核验通过；没有为这项修复启动正式训练或全量生成。
- 在独立服务 `127.0.0.1:7862` 的真实 Chromium 页面验证了选择框、路径输入、checkpoint 检查和后端参数校验。
- 本次对原 7860 服务的重启命令被自动审批以 `blocked by policy` 拒绝。此后再次检查发现 7860 已换为新进程，后端能够校验新策略，并在 7860 重新完成了真实浏览器验证。当前 7860/7862 已加载新后端，7861 仍是旧后端；若继续使用 7861，需手动重启服务再刷新页面，仅刷新网页不足以重载 Python 后端。

本次源码修改前备份：`backups/resume_contract_20261014/source_before_fix.zip`。诊断、实际合同预检和浏览器截图保存在同目录下，按项目 gitignore 保持本地，不上传训练材料与 checkpoint。

分阶段源码提交：`80c08f8`（合同检查、安全迁移和 CPU 测试）；`164355c`（主 WebUI 策略、API、说明与前端测试）。本说明另行提交。工作区中原先未提交的预处理相关修改保持原状，不纳入上述提交。

原有事件模型训练/生成、song_form 模型维度和 checkpoint 权重格式未因本次修改而改变。本次验证确认续训机制与界面可用，不代表谱面质量已经验收。
