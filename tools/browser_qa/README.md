# 可选浏览器验证脚本

保留了连续等级与偏好、曲线密度与休止操作、框选休止及可视化、缩放与多点编辑四组实际浏览器检查。无需 checkpoint 或训练集，也不会执行真实训练或生成。

需要 Node.js、npm 和 Microsoft Edge。Python 全量测试也会使用 `node --check` 检查 WebUI JavaScript；一键启动 WebUI 本身不需要 Node.js。

在此目录执行一次 `npm install`。启动本项目 WebUI，然后在 PowerShell 中运行：

```powershell
$env:CURVE_QA_URL = "http://127.0.0.1:7860"
npm run rests
npm run density
npm run levels
npm run zoom
```

将地址改为启动窗口实际显示的端口。每个脚本使用独立无头浏览器，不修改你正常浏览器中的设置。结果 JSON 与截图运行时写入项目 `.scratch` 目录，该目录不是交付内容。
