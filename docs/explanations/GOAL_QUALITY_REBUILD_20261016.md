# 谱面质量修复与音乐证据增强

## 授权与边界

用户 2026-10-16 授权先上传当前版本，再全面修复、审查和必要测试；允许停止相关训练与 7860/7861 服务。保留旧 checkpoint 生成兼容；无需另建 gold/验证训练集。自动修复的音乐审美目标延期，不将重建、loss 或语法通过称为谱面质量验收。

## 可恢复基线

- GitHub 分支 `codex/song-form-rebuild`，基线提交 `8a71177`，标签 `pre-quality-rebuild-20261016`，已推送并核对远端 SHA。
- 本地主体 `backups/quality_rebuild_20261016/project_before_changes.zip`（518 文件）、`source_hashes.json`、`repository.bundle`。
- 正式训练 `Prism_Re_v6`；修改训练合同前备份最后安全断点，不自动在旧目录混入新监督。

## 分阶段实施清单

- [x] S0：主体备份、基线提交和先行推送。
- [x] S1：物理约束与时值语义。Slide 等待/轨迹按 MiaCode 参考，秒制时值不重新按 BPM 缩放；分别记录双手时间，避免过度判定。训练统计新语义显式版本化。
- [x] S2：曲线编辑与界面。插点时同步重采样组件；v2 显示曲线、设备、变速表；训练/生成/曲线共用拍号输入与身份检查。
- [x] S3：规整与记忆。只规整本段新增事件，保留半开边界和旧前缀；时间变更后重放指导状态；motif 先按当前范围查询再限量；修正规整计数。
- [x] S4：版本化音乐证据增强。`quality_profile=musical_evidence_v1`；音频证据按时间覆盖和多种声音变化平衡保留，不再用整段最强峰吞掉弱段；精确 Fraction 输出不绑定候选位置。
- [x] S5：新监督与轻量架构。官谱采音位置的查询级辅助目标（输入不得读取目标位置）、Planner 实际时间邻域汇聚、一秒负荷与非饱和密度编码；模型/训练/生成/缓存使用同一合同，旧权重不静默升级。
- [x] S6：定向回归和小规模训练→保存→加载→生成；核对旧 checkpoint、主 WebUI 实际参数和性能预算；仅遇失败/新修改时追加测试。
- [x] S7：交接与阶段报告、各阶段 Git 提交、最终推送。

## 合同与设计决策

- 模型族仍为 `song_form_v2`，以显式质量合同区分增强架构。fresh `auto` 使用新合同，resume 继承且锁定，finetune `auto` 继承；只有显式选择才升级旧模型。
- `quality_profile` 取 `legacy` / `musical_evidence_v1`；传输层另支持 `auto`。
- 新训练位置辅助项 `position_alignment_loss_weight` 默认 0.15；旧合同默认 0。目标来自拍师事件而不是所有音频 attack，不提供正确未来 skeleton，解码仍能自由输出任意有理数。
- 新 Planner 邻域参数 `planner_context_seconds` 默认 1 秒；按真实查询时间汇聚，不把不均匀节点索引当固定帧率。
- 增强曲线 `curve_bin_seconds` 默认 1 秒，旧合同 2 秒；密度编码需可逆，不在 8 heads/s 直接饱和。旧曲线合同保留原单位。
- 未来数据/标签语义变化必须提升准备缓存与 Blueprint 语义身份；生成侧修复不无故清空音频波形缓存。
- 真实拍号仍需人工提供或来源结构文件；未知时标记 unknown，不能宣称已学会 3/4。不得盲目降低 recurrence 阈值强造副歌模板。

## 持续记录

每阶段将修改目的、实际命令/测试结果、提交 SHA、生效与恢复方式写入 `QUALITY_REBUILD_HANDOFF_20261016.md`。本文件与交接记录是后续上下文恢复入口。尚未完成的项目不得提前勾选；用户实际试听/专业试玩留作音乐性验收。

2026-10-17：S1–S6 实现与必要验证完成，见 `QUALITY_REBUILD_RESULTS_20261017.md` 与 `docs/evidence/QUALITY_REBUILD_20261017.json`。全项目 Python 2223 passed / 12 skipped，Node 94 passed，收尾修改定向 68 passed。真实官谱 4 步 GPU 训练→模型曲线→基础音符目录生成→精确编译可用；完整目录四步模型仍耗尽配置文本预算，不冒充谱面质量验收。主 WebUI 7860 已恢复，正式训练未重开。

S7：修复提交 `bd80919`、`3e08f9d`、`8463d93` 与报告提交 `7f93b6c` 已推送 GitHub；远端 refs/heads/codex/song-form-rebuild 已核对为 `7f93b6cbfd3e80c0c51c740fd355d2359784462a`。本条状态记录随后单独提交。工作区仅余外部 `.workbuddy/` 私有笔记，未纳入上传。
