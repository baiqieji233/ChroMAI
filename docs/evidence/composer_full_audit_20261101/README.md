# 全链路审计证据索引

对应报告：[Composer全链路诊断](../../COMPOSER_FULL_DIAGNOSTIC_20261101_ZH.md)。基线commit `0ff1f5a`。

- `audit_evidence.json`：审计末尾的源码哈希对照、v38/v37运行快照、恢复计数、证据文件哈希。计数随任务继续推进，不能把快照当当前实时状态。
- `source_inventory.json`：src下292个Python/JS/HTML/CSS文件路径、行数、SHA256；Python含AST函数计数。它证明库存范围和审计期间源码未变，不证明所有旧分支经过逐行动态验证。
- `busy_directory_probe.json`：实际API函数在隔离fixture中的写切分/拒绝顺序。split与reserve经过替换，未访问生产目录。
- `data_chain_findings.md`：数据代理的初稿、随后追加的真实计数、解析/编译和Antigravity复核。后续追加说明优先于初稿中“尚未检查”的描述。
- `learning_audit.md`：学习/恢复/续训/时长读取审计及模板保持toy结果。
- `generation_chain_audit.md`：搜索/评分/物理/导出审计，含默认条件与非默认边界的区分。
- `antigravity_review.md`：通过现有会话Copy取得的本轮完整第三方原始回复。**里面的“已证实”“必然塌缩”等措辞不是主报告裁定**，应与主报告第9节的逐条复核一起阅读。原文保留是为了可追溯。
- `*.py.txt`：本轮隔离CPU探针的文本归档，没有安装或接入产品代码。原脚本位于项目 `.scratch/audit_full_20261101/`。其中root探针按该scratch位置定位仓库，搬到别处运行前必须修正ROOT；不能直接把归档路径当原运行路径。

本轮未执行新的GPU训练、完整生成、全套回归或全语料缓存重建。已有4410项回归与双种子生成来自上一轮报告，不算本次重新执行。所有质量与性能因果结论须遵守主报告的证据等级。
