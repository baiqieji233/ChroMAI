# Composer 曲线分析路由修复 · 2026-10-22

## 确认问题

任务 `b8ad397f2aa8` 使用 Composer 检查点进行强度曲线分析，失败信息为
“需要 event_v1 checkpoint；旧固定帧模型不能作为事件模型续训/初始化”。
检查点不是因此损坏，也不是要求重新训练。

实际原因是强度编辑页保留了旧模型的 `curve_source=model`。后端曲线
预测分派只处理 stream/form，剩余模型全部进入 event learned_curve，
所以 Composer 最终被错误交给 `load_event_checkpoint`。前端分析请求
也未传模型族，未区分 Composer 的能力。

当前 Composer 只有候选指针、动作、future 和 EOS 预测；其 future 是
条件于谱面前缀的局部辅助头，不是可替代整曲 Planner 的曲线预测头。
不能简单把它当 Planner 返回一份“模型已预测”的曲线。

## 修复内容

- 后端按检查点实际模型族分派曲线预测，不再被请求中的旧模型族字段
  覆盖。auto 仅在已支持的模型族启用 learned curve，Composer auto 走
  音频参考；显式 Composer model 返回准确的不支持说明，避免进入
  event loader。未知/旧帧模型 model 同样提前给出准确能力说明。
- 单次任务只加载一次 checkpoint 元数据。显式 audio 完全不加载
  checkpoint，避免分析时把大型训练文件及 Adam 状态再次读入 RAM。
- 前端 Composer 分析强制发送 audio、checkpoint=null、实际 model_family。
  禁用不支持的模型 Planner 选项；从 song_form 切换时，先完成旧选项
  恢复，再移除不支持的 model，避免恢复逻辑重新引入问题。
- 编辑页明确说明 Composer 当前曲线仅供音频参考，不参与生成。保存
  按钮改为“保留参考并返回”，不会把 intensityCurveEnabled 勾上，也
  不会声称休止区或编辑曲线已约束 Composer 生成。
- 修正一个既有 song_form UI 测试夹具缺少 trainingSourcePaths mock
  的问题，未改动该模型族的生产训练逻辑。

备份：`backups/composer_curve_route_before_20261022_165240.zip`。
未改动模型权重、训练输入/损失/采样、生成算法及检查点合同。
未结束当前训练、生成或 WebUI 进程，未使用 GPU 测试。

## 验证

- 56 项 Python API/UI 检查通过。新增覆盖 Composer auto/audio/model、
  错误请求模型族、audio 不加载权重，以及旧 event/form 正常预测路由。
- 首次 Node 检查 29/30 通过；失败为既有 song_form 测试夹具缺少
  trainingSourcePaths。补齐夹具后该文件 7/7 通过，合计 30 项不同
  Node 检查均已通过。
- 新增前端测试覆盖旧 model 选项、Composer 不传 checkpoint、切换
  song_form→Composer、Composer 参考保存不启用生成曲线。
- 在已运行的 7860 后端，使用原失败任务的真实音频按新前端请求执行
  CPU 分析：`f85063eb16ed` completed，source=audio，method=multifeature_v1，
  时长 160.73687 秒、82 个点，并通过 validate_intensity_profile。
- HTTP 读取 7860 的 `/app.js` 确认已包含新 Composer audio 分支。

## 使用与运行边界

当前已运行后端即使尚未重载 Python，也能接受新前端显式 audio 请求，
因此 Ctrl+F5 刷新界面后可直接重新分析，无需中断训练或重启后端。
后端完整模型族防错分派需下次正常重启 WebUI 后生效；外部客户端若
仍向旧服务显式发送 Composer model，旧进程尚会执行旧路由。不要为了
热加载本次防错逻辑结束用户正在运行的任务。
