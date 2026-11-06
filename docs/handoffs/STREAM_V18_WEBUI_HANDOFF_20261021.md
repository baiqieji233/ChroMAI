# Stream v18 WebUI 接线交接

2026-10-21

- 训练页新增 `history_position_dropout`（默认 0.1，范围 0–0.5）、`history_position_span_events`（默认 4，范围 1–64）、`history_position_keep_events`（默认 2，范围 1–16）和 `condition_dropout`（默认 0.1，范围 0–1）。前端 payload、checkpoint 回填/resume 锁定以及 `stream_api.py` 的 CLI 传递已同步。
- 生成页新增 `position_density_strength`（默认 0.12，范围 0–3）和 `repetition_escape_attempts`（默认 1，范围 0–2），保留现有 Style 设置。
- 节奏循环帮助文案改为：只有长时间持续、局部谱师选择证据偏弱且预测负荷不支持时才软降权；正常切分、舍音、均匀节奏和短呼应不会因重复本身受罚。文案不承诺消除复读或保证可玩性。
- 本次只改 WebUI 接口、静态前端和本交接文档；未修改 `src/maichart/stream` 核心，未运行测试。
