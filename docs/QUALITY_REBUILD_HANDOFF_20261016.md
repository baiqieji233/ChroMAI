# 音乐证据与谱面质量重构交接

本轮从 2026-10-16 开始，最新工作日期 2026-10-17。状态以本文件和 `GOAL_QUALITY_REBUILD_20261016.md` 为准，不以旧完成报告推断当前工程已完成。

## 恢复入口与安全基线

- 分支 `codex/song-form-rebuild`；先行快照 `8a71177aa42cee983233ea7b00ed45a2e09849b0` 已上传 GitHub，标签 `pre-quality-rebuild-20261016`。
- 主体、Git bundle 和训练断点在 `backups/quality_rebuild_20261016/`。
- 正式训练 `Prism_Re_v6` 已经由 WebUI 取消；最后安全 checkpoint 已复制，未覆盖来源输出目录。7860/7861 项目服务已停止；同端口列表里的 tailscaled.exe 不属于项目，未停止。
- 用户授权全面修复、必要 GPU 探针和新合同训练；不自动重开长训练。自动修复的音乐审美目标延期。

## 已实现、等待最终集成验收

1. 物理约束按 MiaCode 时值和动作解析，独立双手时间，Slide 等待期与实际轨迹分离。统计新语义 `miacode_head_bpm_v1` 显式 opt-in，旧默认保留。定向 CPU 首轮 87 项、补充 43 项通过。
2. 曲线插入节点同步重采样所有类型组件；v2 主页面显示设备、曲线与 tempo 字段；训练和生成增加拍号及质量合同控件。66 项曲线/API、5 项 HTTP 与 19 项 Node 定向验证通过；训练预检待主集成复查。
3. 规整只处理新增事件，严格半开段边界；改拍后重放 TypeGuidance 状态。Motif 先按参考范围裁剪再限量，避免长参考后半截丢失。54 项 CPU 定向通过。
4. `balanced_evidence_v1` 按时间覆盖与 attack/chroma/detail/dynamics 多路保留证据；不再让全曲最强 attack 占满预算。原始/查询缓存与旧合同分离；原匹配描述不改，额外 8 维保留全曲相对动态。37 项 CPU 定向通过，10 万候选筛到 8192 约 0.031 秒。
5. `musical_evidence_v1` 数据合同接入精确查询时间、段拍长/秒长、512 维布局描述及位置辅助标签。查询节点构造完全不读取谱面目标；不匹配的官谱事件不强制对齐错误的最近节点。新数据与缓存 11 项、旧相关 24 项、first 对齐 13 项通过。

## 正在集成

- 主代理：train.py、blueprint_batch.py、质量训练默认值/日志/能力计数，端到端探针、文档、提交。
- fix_ui_curve_contract：checkpoint.py、model_compatibility.py，新旧合同验证及显式 finetune 权重迁移。
- fix_timing_motif：CLI / rational_generate / revision 的质量合同、Planner 时间输入及曲线密度单位。
- fix_physical_semantics：alignment.py / model.py / rational_planner.py 的 masked-safe 边界与辅助目标定向测试。

## 新合同的边界

- fresh v2 的 `auto` 使用 `musical_evidence_v1`；resume 继承并锁定旧合同；finetune 的 `auto` 继承，显式选新版才迁移。v1 保留 legacy。
- 位置仍是任意 Fraction；音频查询只是证据，辅助目标既不插入真实位置，也不限定可输出位置。位置 CE 与实际采音对应的监督分别记录。
- 新负荷密度 `nps/(nps+8)` 可逆，1 秒训练标签；旧线性饱和密度和 2 秒时钟保持原状。编辑器显示分辨率独立于训练时钟。
- Planner 按真实秒数聚合左右音频邻域；避免将不均匀节点下标当作固定帧率。
- 额外辅助头规模小；尚无充分训练的实际音乐质量提升证据。测试只能证明合同、数值与流程可用。
- 未知拍号仍标记 unknown，重复段关联不能通过盲目降阈值强造。

## 后续验收与发布

完成接口后运行相关 CPU/Node 测试，必要时一次全项目回归；只对失败或后续修改追加测试。独立 scratch 目录做小型官谱 GPU 训练→保存→加载→生成；不把短跑模型认定为可玩或高质量。检查实际主 WebUI 的 fresh/resume/finetune 与新旧曲线流程，然后记录最终提交、测试数和启动状态并推送。

临时浏览器 profile 清理被工具自动审批阻止：`C:\Users\mywor\AppData\Local\Temp\maichart-ui-audit-af9f5111dcf04e3f915c7f237af0ef22`。浏览器已关闭；未绕过策略，不影响功能。
