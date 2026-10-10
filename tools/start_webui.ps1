[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ScriptArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = $null
$CheckOnly = $false
$Cpu = $false
$Cuda = $false
$NoBrowser = $false
$NoInstall = $false
$Reinstall = $false
$Port = 7860

function Fail([string] $Message) {
    Write-Error $Message
    exit 1
}

for ($index = 0; $index -lt $ScriptArgs.Count; $index++) {
    $argument = [string]$ScriptArgs[$index]
    switch -Regex ($argument.ToLowerInvariant()) {
        "^--check$" { $CheckOnly = $true; continue }
        "^--cpu$" { $Cpu = $true; continue }
        "^--cuda$" { $Cuda = $true; continue }
        "^--no-browser$" { $NoBrowser = $true; continue }
        "^--no-install$" { $NoInstall = $true; continue }
        "^--reinstall$" { $Reinstall = $true; continue }
        "^--port$" {
            if ($index + 1 -ge $ScriptArgs.Count) { Fail "--port requires a number." }
            $index++
            if (-not [int]::TryParse($ScriptArgs[$index], [ref]$Port) -or $Port -lt 1 -or $Port -gt 65535) {
                Fail "Invalid port. Use a number from 1 to 65535."
            }
            continue
        }
        "^--port=(\d+)$" {
            $Port = [int]$Matches[1]
            continue
        }
        default { Fail "Unknown option '$argument'. Supported options: --check --cpu --cuda --port N --no-browser --no-install --reinstall" }
    }
}

if ($Cpu -and $Cuda) { Fail "Choose only one of --cpu and --cuda." }
if ($Reinstall) { $NoInstall = $false }

$overridePython = $env:MAICHART_PYTHON
if ($overridePython) {
    $Python = (Resolve-Path -LiteralPath $overridePython -ErrorAction SilentlyContinue).Path
    if (-not $Python) { Fail "MAICHART_PYTHON does not point to an existing Python executable: $overridePython" }
} else {
    $localPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $Python = $localPython
    } else {
        $candidate = Get-Command py -ErrorAction SilentlyContinue
        if (-not $candidate) { $candidate = Get-Command python -ErrorAction SilentlyContinue }
        if (-not $candidate) { Fail "Python 3.10-3.13 was not found. Install Python or set MAICHART_PYTHON." }
        if ($CheckOnly) {
            $Python = $candidate.Source
        } else {
            Write-Host "Creating project virtual environment: $localPython"
            & $candidate.Source -m venv (Join-Path $ProjectRoot ".venv")
            if ($LASTEXITCODE -ne 0) { Fail "Failed to create the project virtual environment." }
            $Python = $localPython
        }
    }
}

if (-not (Test-Path -LiteralPath $Python)) { Fail "Python executable not found: $Python" }
$versionText = (& $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0) { Fail "Unable to run Python: $Python" }
$version = [version]("$versionText.0")
if ($version -lt [version]"3.10.0" -or $version -ge [version]"3.14.0") {
    Fail "Python 3.10-3.13 is required; found $versionText."
}
Write-Host "Python: $Python ($versionText)"

function Test-PythonImports {
    & $Python -c "import torch, librosa, numpy, scipy, soundfile, pytest"
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-PythonImports)) {
    if ($CheckOnly -or $NoInstall) {
        Write-Error "Required Python modules are missing. Run without --check/--no-install to install them."
        exit 1
    }
    $pipArgs = @("-m", "pip", "install", "-e", ".[$([string]::Join(',', @('dev','audio','training')))]")
    if ($Reinstall) { $pipArgs += "--upgrade" }
    if ($env:MAICHART_TORCH_INDEX_URL) {
        $pipArgs += @("--extra-index-url", $env:MAICHART_TORCH_INDEX_URL)
    }
    Write-Host "Installing project dependencies..."
    Push-Location $ProjectRoot
    try { & $Python @pipArgs } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed." }
    if (-not (Test-PythonImports)) { Fail "Dependencies are still incomplete after installation." }
}
Write-Host "Python dependencies: OK"

if ($Cpu) {
    $env:CUDA_VISIBLE_DEVICES = ""
    Write-Host "Device mode: CPU"
} elseif ($Cuda) {
    & $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) { Fail "CUDA was requested, but this Python environment cannot access a CUDA device." }
    Write-Host "Device mode: CUDA"
}

$env:PYTHONPATH = if ($env:PYTHONPATH) { "$ProjectRoot\src;$env:PYTHONPATH" } else { "$ProjectRoot\src" }
if ($CheckOnly) {
    Write-Host "Environment check passed. WebUI was not started."
    exit 0
}

if (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue) {
    $listeners = @(Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) { Fail "Port $Port is already in use. Choose another port with --port." }
}

$url = "http://127.0.0.1:$Port/"
Write-Host "Starting WebUI at $url"
if (-not $NoBrowser) { Start-Process $url }
Push-Location $ProjectRoot
try {
    & $Python -m maichart.webui.server --host 127.0.0.1 --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
