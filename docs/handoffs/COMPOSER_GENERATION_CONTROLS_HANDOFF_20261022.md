# Composer 生成相似、控件接线与保存停滞交接 · 2026-10-22

## 结论与实际证据

不是一个单纯的 seed 问题，同时存在生成参数接线缺失、历史输出覆盖，
以及当前训练没有发布新权重的独立问题。

四次旧任务的实际命令：相同音频、同一个 composer_last.pt 路径、
temperature=0.65、seed=0；前两次 level=14.5，后两次 level=14.2。
对应事件数 829、2196、2181、2181。没有 --condition-file。
历史 chart/report 都指向同一 maidata.txt / maidata.composer.json，
因此无法再逐字恢复旧谱面进行可靠比较；不能把事件数相同直接说成
旧文件已证实逐字相同。

### 已确认的生成缺陷

1. Composer 前端没有发送 collectCondition()，后端也没有接通 inline
   condition。页面的等级偏好、Radar/Style 即使变化，生成器仍使用固定
   GENERATION_DEFAULTS。只有单独 level 和外部条件文件能影响输入。
2. temperature 标签的 data-generation-family 没有 Composer，因此控件
   被隐藏，但请求仍读取其旧值 0.65。
3. Composer API 对 seed 使用普通整数读取，未实现其他生成器的留空
   随机语义。显式 null 会报整数错误，省略则默认 0；CLI 也默认 0。
   已有四次记录只能证明实际使用了 0，不能声称用户的空值必定被转成 0。
4. 不同任务覆盖同一输出，旧任务下载链接可能读到新文件。并行任务
   还可能使 chart/report 不对应。
5. CLI 已支持 tempo-map，但主界面隐藏该控件，Composer payload/API
   也没有转发。

### 独立的训练保存异常

当前旧进程 PID14504、任务 e59589bfaf51，界面显示 running，但日志停在
epoch2 batch375、optimizer_steps29991。第30000步保存已创建临时文件，
正式 composer_last.pt 的修改时间仍为 16:49:57；读取其合同确认是
epoch1 完整结束、29616次更新。

使用仅安装在忽略目录 `.scratch/profiling` 的 py-spy 做只读栈检查：

```
MainThread: torch/storage.py:262 cpu
  torch/serialization.py:1264 _save
  composer/checkpoint.py _write_payload
  composer/train.py main (periodic save)
```

进程约占一个 CPU 核，磁盘读写速率为零。可确认卡住的位置是保存中的
GPU storage→CPU 复制；不能仅凭 Python 堆栈断言底层 CUDA/驱动根因。
这也解释了16:58、17:01两次任务没有读到后续训练更新。

## 实现

- 共用 generation_settings：留空/省略 seed 用 secrets 产生实际种子；
  显式0保留固定种子含义；CLI与API统一。API会将随机来源标记传给
  CLI，任务和报告均记录实际 seed。
- 条件优先级为默认值→界面→外部文件明确字段。文件未指定字段继承
  界面，不重置成0.5；文件若显式包含level，则覆盖界面level并在页面
  与任务中提示。后端将最终条件快照为任务独立JSON再传给生成器。
- 主界面发送完整condition与tempo_map，开放temperature与tempo-map；
  温度同时控制位置和动作，0仍为greedy，seed在greedy中不改变结果。
  不擅自提高温度，不叠加强制反重复惩罚。
- 后端按task_id隔离每次输出，保存requested_output与实际output。
- 报告新增有效条件、温度、实际epoch/更新步数、加载权重SHA256、
  完整文件hash与不含等级头的音符序列hash、采样最大概率均值、fallback
  数和配置文本重复概况。主界面展示实际参数和身份，避免只比较文件名。
- 不可能的动作从有限-10000改为-inf：即使所有合法logit极小，非法
  STOP/占用组合也不能因有限mask值更高而被选中。
- 生成器在模型权重复制完并计算身份后释放文件mmap引用，降低Windows
  上长生成阻止训练原子替换checkpoint的风险。此为另一个潜在发布问题，
  与本次观测到的storage.cpu阻塞不同，不能混淆。
- 保存器先用不超过8MiB的显式Tensor复制完成CPU快照，再调用torch.save
  写CPU数据，避开已观察到的CUDA storage.cpu序列化路径。权重/Adam
  数值与checkpoint合同不变。CPU调用不复制已有CPU张量。
- CPU快照会暂存完整权重和Adam状态，当前85.5M参数模型约需额外1GiB
  主机内存；这比旧逐storage串行写入更占峰值RAM，需关注host_memory。
  独立验证成功不证明旧CUDA上下文可恢复，亦不保证所有驱动卡死消失。
- `/api/status` 发布 `composer_generation_controls_v3`，前端拒绝旧后端
  静默忽略新参数。已运行的服务需正常重启加载；不能靠刷新Python后端。

## 备份与当前运行状态

源文件备份：`backups/composer_generation_controls_before_20261022_171651.zip`；
保存器原文件另有 `backups/composer_checkpoint_before_*.py`。

完整第一轮末检查点已备份到：
`backups/composer_before_stalled_resume_20261022_174528/composer_last.pt`。
未覆盖、删除原权重或其未完成临时文件。

已准备 `.scratch/composer_resume_request.json`：保持原batch32、累积1、
384宽、双6层、FFN1536、BF16、lr2e-4、缓存4GiB、seed0等设置，从
composer_last.pt resume剩余79轮，最终仍到第80轮，而非fresh。
恢复会回到29616步，第二轮尚未保存的约384次更新不在正式文件内。

尝试停止卡住任务并重载7860的命令被自动审批审查整体拒绝，返回
`blocked by policy`，没有更具体原因；命令未执行。因此本轮没有停止
该训练，也没有完成服务重启或训练恢复，不能将恢复计划写成已恢复。
不要改用其他进程终止手段绕过该拒绝。用户需手动结束卡住任务，
重启希望使用的WebUI端口，再按原配置resume；新前端会提示旧服务版本。

## 修改结束后的集中验证

- 21项Python生成/模型回归通过；后续保存器阶段5项针对性检查通过，
  包含新增CPU快照容器元数据和既有训练/检查点/精确resume/导出覆盖。
- 24项Node检查通过：condition、tempo、温度、seed，控件可见性、
  固定0与空值区别、结果说明，原song-form/Composer UI合同。
- 用真实epoch1/29616步权重固定一个CPU模型快照，每组仅生成24事件：
  seed101两次音符序列hash相同；seed202改变序列；seed101改变Style也
  改变序列。没有fallback。原条件pointer平均最大概率约0.768、
  第一动作约0.509，不能断言模型被完全锁成argmax。
  改Style的短样本仍以Tap为主，不能把控件生效等同于已学会理想配置。
- 独立GPU复制/保存检查只占约0.211GiB：56,690,688个float32元素
  （约227MB）及嵌套optimizer标量，分块快照、CPU写入和读回一致，
  耗时约0.85秒。不涉及训练或覆盖用户权重。
- 一次已有7861服务的真实8事件CPU任务确认温度/条件/随机seed/独立
  输出生效，但该服务在编辑期间已缓存旧adapter，随机来源标志未转发；
  没把它当作完整部署验证。最终由新模块的临时HTTP服务执行同样实际
  CPU生成，任务3b8c79fd5136通过所有断言：实际seed1607361257、
  temperature0.91、hold_bias0.77、8事件、独立目录，报告随机来源正确。
  临时HTTP服务已关闭。此验证没有声称7860/7861已完成重启。

本地证据：`.scratch/composer_generation_ab_results.json`、
`.scratch/composer_generation_ab.log`、`.scratch/composer_train_stack.txt`、
`.scratch/composer_live_generation_controls.log`。
