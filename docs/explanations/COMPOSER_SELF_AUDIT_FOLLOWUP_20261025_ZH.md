# Composer 自主复查与修复（2026-10-25）

接续`625c6dc`，按用户要求自行追踪实际链路并修复可复现问题。未改变采样温度策略、默认冲突策略或损失权重，未扩展已排除的Simai语法，也未修改弃用的独立Composer页面。

## 1. 动机分支绕过符号历史遮挡

`symbolic_keep=0`原本关闭历史、持续状态、head timing及台账，但新档案分支在`decode_candidate`中额外加入，没有共用此遮挡。使用真实30步实验模型及真实批次，历史被关闭时，改变档案仍使动作logits最大变化0.112566。

现在把相同的逐样本遮挡传入动机残差；关闭的样本不会从档案获取配置信息，也不给该分支贡献梯度，保留的样本仍可正常学习。修后同探针差异为0；正常推理的16个浮点输出与修改前最大差均为0。

该变化只影响启用动机的训练语义。新动机checkpoint保存`motif_training_policy=symbolic_masked_archive_v1`；旧动机权重仍可生成/显式finetune，但不能不告知地resume到新训练行为。没有动机的旧v10保持原续训资格，全局v11并未对所有模型升级。

## 2. 持续状态边界和缓存

- 整数秒附近，causal_v2允许`onset<=boundary+EPS`，但候选桶按`floor(onset)`索引，可能先把该音符丢掉。自然复现：92.55 BPM、1/3拍、first=-0.21609940572663427，边界约-2.78e-17秒、解析onset=0。桶起点改为`floor(onset-EPS)`，最终精确过滤及legacy比较保持原样。1040组密集/随机查询与全扫描逐值一致。
- `IncrementalState.append`原本未清候选桶缓存；缓存只保存列表地址和长度，连续提交中Python回收并复用列表地址后，会误用旧桶。实测缺失活动Hold，清缓存后恢复。现在append明确清缓存，索引同时保留原列表引用并用实际对象身份和长度验证；替换后释放旧缓存。测试不依赖地址复用碰运气。

这些修复没有增加未来谱面可见性，也没有改变Slide等待期、时长与路径公式。

## 3. Touch统计与实际守卫一致

open策略将按钮`1`与屏幕Touch`A1`视为不同实际头，但密度统计仍用33维合并pad，可能误报过快同头。复现：二者相隔31.25ms时，open守卫都接受，旧报告却计1次低于50ms的同头重复。

生成前后统计现在明确使用本次Touch策略的头身份：open分别计数，legacy仍保留旧合并统计。真正的`A1→A1`短间隔仍被守卫和报告识别。本项修正的是诊断口径，不是音乐质量提升。

## 4. 任务取消、重复训练与异常收尾

- 终止进程失败后，任务原本留在cancelling且无法再次点击取消。现在保留真实未退出状态、显示具体错误并提供“重试取消”；API与前端都阻止在途重复提交。确认进程退出前不会报告已中断。
- 刷新页面后恢复所有存活Composer训练的忙碌状态；一个任务结束不会解除其他任务的忙碌标记。
- 同一WebUI进程在锁内检查并预留规范化输出目录，queued/running/cancelling均占用；只有终态才能重新提交同目录。不同目录的API请求仍允许。
- CLI对整个准备、训练、保存阶段持有跨进程目录锁，避免不同WebUI端口或CLI同时覆盖同一模型。Windows真实子进程验证同目录拒绝、别名/大小写归一、不同目录独立，以及正常退出/强杀后的自动释放。POSIX使用标准flock；本轮实际测试环境是Windows。
- `.composer-training.lock`文件会保留，存在不等于仍被占用。不要删除它绕过正在持有的锁；正常退出或进程终止会由操作系统释放锁。
- 日志初始化纳入worker异常处理；日志关闭时的stat、元数据保存、清理失败写入`log_close_error`，不会覆盖真实任务成败或让输出目录永久保持忙碌。普通日志创建IO失败仍沿用原降级行为。

## 5. 验证结果

- 相关Python回归：**491 passed**；主界面Node回归：**28 passed**。
- 真实Edge主界面6场景通过：刷新发现两个存活任务、分批结束、取消失败重试、请求在途防连点、确认退出后显示终态。任务接口全部拦截模拟，未启动或取消用户真实任务。
- 官方CLI真实小数据：fresh两次更新、resume再两次更新、旧动机resume在数据准备前明确拒绝、显式finetune成功。新动机检查点可resume；原v10仍可resume。
- Windows临时文件实测保留mmap映射时atomic checkpoint replace仍成功，未复现“生成读取会阻塞训练保存”的假设，因此未修改该生命周期。
- 原`Prism_composer_v10`与`Prism_composer_v10_preserved_20261025`内last/inference四文件哈希均与保留记录一致。长程训练仍停在第4轮、12,500次更新，没有以实验权重替换原基线。

本轮没有新的音乐质量结论，也没有为代码修复重复宣称loss或成谱收益。未复现的候选专属4D档案接口边界、任意坏logits等没有扩展成生产修复。

## 6. 本机证据

目录`.scratch/composer_self_audit_20261025/`：

- `real_motif_mask_evidence.json`：真实模型遮挡差异与推理兼容。
- `regression.log`：相关Python回归。
- `cli_validation.json`及fresh/resume/explicit_finetune日志：实际训练入口与迁移门禁。
- `browser_tasks_result.json`：实际浏览器6场景。
- `checkpoint_integrity.json`：四个原检查点哈希核对。

更新主界面在7870；7860只保留原任务记录。源码推送到同一`codex/composer-causal-motif-touch-20261025`分支；数据、权重和scratch不上传。
