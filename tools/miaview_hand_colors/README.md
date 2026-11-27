# 编译手顺预览的音符着色版

ChroMAI 计算每个音符及滑轨所属的手。`miaview-web.patch` 把这份分配送入播放器，`miacode.patch` 让原渲染器给对应贴图着色。两份补丁配合使用。手顺训练和批量标注可以直接使用 Python。

## 取出对应版本

在 ChroMAI 根目录运行以下 PowerShell 命令，用新的相邻目录存放播放器源码。指定版本的 MiaView 已经固定了对应的 MiaCode 子模块。[MiaView_Web 源码](https://github.com/Team-MiaCode/MiaView_Web/tree/ed790008ec0be5758194f17af002e9740b2954c4) · [MiaCode 源码](https://github.com/Team-MiaCode/MiaCode/tree/ff9f65856b8591589aba0582ac867220202a44c6)

```powershell
$chromai = (Get-Location).Path
$patches = Join-Path $chromai 'tools/miaview_hand_colors'
git clone https://github.com/Team-MiaCode/MiaView_Web.git ../MiaView_Web_Hand
Set-Location ../MiaView_Web_Hand
git checkout --detach ed790008ec0be5758194f17af002e9740b2954c4
git submodule update --init --recursive
git -C vendor/MiaCode apply (Join-Path $patches 'miacode.patch')
git apply (Join-Path $patches 'miaview-web.patch')
```

## 构建并接入本地预览

按该版本的 MiaView 说明准备 Qt 和 Emscripten，然后在刚才的 MiaView 目录运行 `node native/build.mjs --development`。生成的 `renderer` 包含配套的 JS、WASM、资源和上游许可文件。[上游构建说明](https://github.com/Team-MiaCode/MiaView_Web/blob/ed790008ec0be5758194f17af002e9740b2954c4/README.md) · [ChroMAI 第三方声明](../../THIRD_PARTY_NOTICES.md)

在 ChroMAI 手顺预览页点击一次「安装 MiaView 播放器」以准备 SDK。构建成功后，在同一个 PowerShell 窗口执行以下命令，把整份手色渲染器放到单独的版本目录。重新打开「启动手顺预览.bat」，选择谱面并点击「计算并预览」，切换「手顺标注」或「方案着色」即可查看。

```powershell
$release = (Get-FileHash -Algorithm SHA256 -LiteralPath 'renderer/build-identity.json').Hash.Substring(0,20).ToLowerInvariant()
$folder = Join-Path $chromai "webui_workspace/preview_runtime/miaview/releases/$release"
if (-not (Test-Path -LiteralPath $folder)) {
    New-Item -ItemType Directory -Path $folder | Out-Null
    Copy-Item -LiteralPath 'renderer' -Destination (Join-Path $folder 'renderer') -Recurse
}
```
