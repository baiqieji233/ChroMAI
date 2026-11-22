# Windows PowerShell 5.1. ZIP installations only require built-in Windows tools.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$repository = 'https://github.com/baiqieji233/ChroMAI'
$archive = $null
$lock = $null

function Git-Result([string[]]$Arguments) {
    $result = & git.exe -C $root @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Git 操作失败。请查看上一行提示，确认网络与本地文件状态。' }
    return $result
}

function Update-Checkout {
    if (-not (Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue)) {
        throw '这是 Git 工作副本，请安装 Git for Windows 后重试。也可以在独立目录解压 GitHub 的 ZIP 版使用。'
    }
    if ((Git-Result @('branch','--show-current')) -ne 'main') { throw '当前不是 main 分支。请切换到 main 后再更新。' }
    if (Git-Result @('status','--porcelain','--untracked-files=no')) { throw '有尚未提交的项目修改。请先保存这些修改，再更新。' }
    # Fetch only from the official release branch, including on installations with a custom origin.
    Git-Result @('fetch','--no-tags',($repository + '.git'),'refs/heads/main')
    $target = Git-Result @('rev-parse','FETCH_HEAD')
    & git.exe -C $root merge-base --is-ancestor HEAD $target
    if ($LASTEXITCODE -ne 0) { throw '本地包含独有提交，无法直接快进。请手动合并。' }
    Git-Result @('merge','--ff-only',$target)
    Write-Host "当前版本 $target"
}

function Project-Path([string]$Relative) {
    # ZIP entries and the previous update manifest must both name shipped project files.
    $parts = $Relative.Split('/')
    if ($Relative -match '[\\:<>|"*?\x00-\x1f]' -or
        @($parts | Where-Object { $_ -in @('','.','..','.git') -or $_ -match '[ .]$' }).Count) {
        throw "更新清单含无效路径：$Relative"
    }
    $folders = @('src','tools','docs','configs','examples','.github')
    $files = @('README.md','START_WEBUI.bat','从GitHub更新.bat','pyproject.toml','environment.yml','MANIFEST.in',
               'LICENSE','LICENSE_SCOPE.md','THIRD_PARTY_NOTICES.md','.gitignore','.gitattributes')
    if (($parts.Count -eq 1 -and $Relative -notin $files) -or ($parts.Count -gt 1 -and $parts[0] -notin $folders)) {
        throw "当前更新器不认识项目路径 $Relative，请从 GitHub 下载完整新版。"
    }
    $path = $root
    foreach ($part in $parts) {
        $path = Join-Path $path $part
        if ([IO.File]::Exists($path) -or [IO.Directory]::Exists($path)) {
            if ([IO.File]::GetAttributes($path) -band [IO.FileAttributes]::ReparsePoint) { throw "项目文件经过链接目录：$path" }
        }
    }
    return $path
}

function Ensure-Parent([string]$Path) {
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
}

function Get-ArchiveVersion([string]$ZipPath) {
    # GitHub's git archive stores the 40-character commit in the ZIP end comment.
    $stream = [IO.File]::OpenRead($ZipPath)
    try {
        if ($stream.Length -ge 62) {
            [void]$stream.Seek(-62,[IO.SeekOrigin]::End)
            $tail = New-Object byte[] 62
            if ($stream.Read($tail,0,62) -eq 62 -and [BitConverter]::ToUInt32($tail,0) -eq 0x06054b50 -and
                [BitConverter]::ToUInt16($tail,20) -eq 40) {
                $commit = [Text.Encoding]::ASCII.GetString($tail,22,40)
                if ($commit -match '^[0-9a-f]{40}$') { return $commit }
            }
        }
    } finally { $stream.Dispose() }
    return "ZIP SHA256 $((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash)"
}

function Install-Archive([string]$ZipPath, [string]$Store) {
    $version = Get-ArchiveVersion $ZipPath
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    $entries = @{}
    $applied = New-Object 'Collections.Generic.List[string]'
    $manifest = Join-Path $Store 'files.txt'
    $backup = Join-Path $Store ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
    try {
        foreach ($entry in $zip.Entries) {
            if (-not $entry.Name) { continue }
            $prefix = 'ChroMAI-main/'
            if (-not $entry.FullName.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw '下载的 ZIP 与请求的版本不符。' }
            $name = $entry.FullName.Substring($prefix.Length)
            $null = Project-Path $name
            if ($entries.ContainsKey($name)) { throw "重复文件：$name" }
            $entries[$name] = $entry
        }
        foreach ($name in 'pyproject.toml','src/maichart/webui/server.py','START_WEBUI.bat') {
            if (-not $entries.ContainsKey($name)) { throw "下载的项目缺少 $name。" }
        }
        $old = if ([IO.File]::Exists($manifest)) { [IO.File]::ReadAllLines($manifest) }
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            foreach ($name in (@($entries.Keys) + @($old) | Where-Object { $_ } | Sort-Object -Unique)) {
                $dest = Project-Path $name
                $entry = $entries[$name]
                if ([IO.Directory]::Exists($dest)) { throw "文件位置已有同名目录：$dest" }
                if ([IO.File]::Exists($dest)) {
                    if ($entry -and (Get-Item -LiteralPath $dest).Length -eq $entry.Length) {
                        $inputStream = $entry.Open()
                        try { $hash = [BitConverter]::ToString($sha.ComputeHash($inputStream)).Replace('-','') }
                        finally { $inputStream.Dispose() }
                        if ((Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash -eq $hash) { continue }
                    }
                    $saved = Join-Path $backup $name
                    Ensure-Parent $saved
                    [IO.File]::Copy($dest,$saved)
                } elseif (-not $entry) { continue }
                $applied.Add($name)
                if ($entry) {
                    Ensure-Parent $dest
                    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$dest,$true)
                } else { [IO.File]::Delete($dest) }
            }
        } finally { $sha.Dispose() }
        # Replace the owned-file list only after all archive entries were installed.
        $next = Join-Path $Store 'files.next'
        [IO.File]::WriteAllLines($next,[string[]]$entries.Keys)
        if ([IO.File]::Exists($manifest)) { [IO.File]::Replace($next,$manifest,(Join-Path $Store 'files.previous')) }
        else { [IO.File]::Move($next,$manifest) }
    } catch {
        $failure = $_
        $rollbackErrors = @()
        for ($i=$applied.Count-1; $i -ge 0; $i--) {
            $name = $applied[$i]
            try {
                $dest = Project-Path $name
                $saved = Join-Path $backup $name
                if ([IO.File]::Exists($saved)) { [IO.File]::Copy($saved,$dest,$true) }
                elseif ([IO.File]::Exists($dest)) { [IO.File]::Delete($dest) }
            } catch { $rollbackErrors += $name }
        }
        if ($rollbackErrors) { Write-Warning "需要核对文件：$($rollbackErrors -join '、')。备份位置：$backup" }
        throw $failure
    } finally { $zip.Dispose() }
    Write-Host "当前版本 $Version，本次更新 $($applied.Count) 个文件。"
    if ([IO.Directory]::Exists($backup)) { Write-Host "已替换的原文件备份在：$backup" }
}

try {
    Write-Host '正在从 GitHub 检查 ChroMAI 主线更新……'
    if (Test-Path -LiteralPath (Join-Path $root '.git')) { Update-Checkout }
    else {
        $store = Join-Path $root 'backups\github-update'
        [void][IO.Directory]::CreateDirectory($store)
        try { $lock = [IO.File]::Open((Join-Path $store 'update.lock'),'OpenOrCreate','ReadWrite','None') }
        catch { throw '无法取得更新锁。请确认项目目录可写，或等待另一个更新窗口完成。' }
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Write-Host '正在下载并核对项目文件……'
        $archive = [IO.Path]::GetTempFileName()
        Invoke-WebRequest -UseBasicParsing -Uri "https://codeload.github.com/baiqieji233/ChroMAI/zip/refs/heads/main" -OutFile $archive -TimeoutSec 300
        Install-Archive $archive $store
    }
    Write-Host '更新完成。请重新双击 START_WEBUI.bat 启动。'
} catch {
    [Console]::Error.WriteLine("更新未完成：$($_.Exception.Message)")
    exit 1
} finally {
    if ($lock) { $lock.Dispose() }
    if ($archive -and [IO.File]::Exists($archive)) { [IO.File]::Delete($archive) }
}
