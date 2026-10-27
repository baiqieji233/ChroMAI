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
$SupportedPythonMin = [version]"3.10"
$SupportedPythonMax = [version]"3.16"
$SupportedPythonRange = "Python >= $($SupportedPythonMin.ToString(2)) and < $($SupportedPythonMax.ToString(2))"

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

function Get-PythonInfo([string] $Executable, [string[]] $PythonArgs = @()) {
    try {
        $output = @(& $Executable @PythonArgs -c "import sys; print('%d.%d' % sys.version_info[:2]); print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -ne 0 -or $output.Count -ne 2) { return $null }
        return [pscustomobject]@{
            Path = ([string]$output[1]).Trim()
            Version = [version](([string]$output[0]).Trim())
        }
    } catch {
        return $null
    }
}

function Test-SupportedPython($Info) {
    return ($null -ne $Info -and $Info.Version -ge $SupportedPythonMin -and $Info.Version -lt $SupportedPythonMax)
}

function Get-AvailableBackupPath([string] $Path) {
    $candidate = "$Path.invalid"
    $suffix = 1
    while (Test-Path -LiteralPath $candidate) {
        $candidate = "$Path.invalid-$suffix"
        $suffix++
    }
    return $candidate
}

function Find-PythonFromLauncher($Launcher, [string] $VersionTag) {
    # Both syntaxes are supported by installed Windows Python launchers:
    # the classic launcher accepts -V:3.11 and the newer manager also accepts -3.11.
    foreach ($launcherArgs in @(@("-V:$VersionTag"), @("-$VersionTag"))) {
        $candidateInfo = Get-PythonInfo $Launcher.Source $launcherArgs
        if (Test-SupportedPython $candidateInfo) {
            return $candidateInfo
        }
    }
    return $null
}

$overridePython = $env:MAICHART_PYTHON
$overrideSource = "MAICHART_PYTHON"
$pythonConfig = Join-Path $ProjectRoot ".webui-python"
if (-not $overridePython -and (Test-Path -LiteralPath $pythonConfig)) {
    $overridePython = ([string](Get-Content -LiteralPath $pythonConfig -Raw -Encoding UTF8)).Trim()
    $overrideSource = ".webui-python"
    if (-not $overridePython) { Fail ".webui-python is empty. Set it to the path of an existing $SupportedPythonRange executable." }
}
if ($overridePython) {
    if (-not [System.IO.Path]::IsPathRooted($overridePython)) {
        $overridePython = Join-Path $ProjectRoot $overridePython
    }
    $Python = (Resolve-Path -LiteralPath $overridePython -ErrorAction SilentlyContinue).Path
    if (-not $Python) { Fail "$overrideSource does not point to an existing Python executable: $overridePython" }
    $pythonInfo = Get-PythonInfo $Python
    if (-not $pythonInfo) { Fail "Unable to run Python from ${overrideSource}: $Python" }
    if (-not (Test-SupportedPython $pythonInfo)) {
        Fail "$overrideSource must point to $SupportedPythonRange; found $($pythonInfo.Version)."
    }
} else {
    $configuredVenv = [bool]$env:MAICHART_VENV
    $venvRoot = if ($configuredVenv) { $env:MAICHART_VENV } else { Join-Path $ProjectRoot ".venv" }
    if (-not [System.IO.Path]::IsPathRooted($venvRoot)) {
        $venvRoot = Join-Path $ProjectRoot $venvRoot
    }
    $venvPython = Join-Path $venvRoot "Scripts\python.exe"
    $existingVenvInfo = $null
    if (Test-Path -LiteralPath $venvPython) {
        $existingVenvInfo = Get-PythonInfo $venvPython
        if (Test-SupportedPython $existingVenvInfo) {
            $Python = $venvPython
        } else {
            Write-Warning "Ignoring incompatible Python environment '$venvRoot'. $SupportedPythonRange is required."
        }
    }

    $bootstrapPython = $null
    if (-not $Python) {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            foreach ($versionTag in "3.11", "3.12", "3.13", "3.10") {
                $candidateInfo = Find-PythonFromLauncher $pyLauncher $versionTag
                if ($candidateInfo) {
                    $bootstrapPython = $candidateInfo.Path
                    break
                }
            }
        }
        if (-not $bootstrapPython) {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCommand) {
                $candidateInfo = Get-PythonInfo $pythonCommand.Source
                if (Test-SupportedPython $candidateInfo) {
                    $bootstrapPython = $candidateInfo.Path
                }
            }
        }
        if (-not $bootstrapPython) { Fail "$SupportedPythonRange was not found. Install Python 3.11 or set MAICHART_PYTHON." }

        if ($CheckOnly -or $NoInstall) {
            $Python = $bootstrapPython
        } else {
            if (Test-Path -LiteralPath $venvRoot) {
                if (-not $configuredVenv) {
                    $versionLabel = if ($existingVenvInfo) { $existingVenvInfo.Version } else { "incomplete" }
                    $backupVenv = Get-AvailableBackupPath $venvRoot
                    $backupVenv = "$backupVenv-$versionLabel"
                    if (Test-Path -LiteralPath $backupVenv) {
                        $backupVenv = Get-AvailableBackupPath "$venvRoot.invalid-$versionLabel"
                    }
                    Write-Warning "Moving the incompatible project environment to '$backupVenv'."
                    Move-Item -LiteralPath $venvRoot -Destination $backupVenv
                } else {
                    Fail "Existing environment '$venvRoot' is incompatible. Set MAICHART_VENV to a new directory or MAICHART_PYTHON to an existing $SupportedPythonRange environment."
                }
            }
            Write-Host "Creating virtual environment: $venvRoot"
            & $bootstrapPython -m venv $venvRoot
            if ($LASTEXITCODE -ne 0) { Fail "Failed to create the virtual environment: $venvRoot" }
            $Python = $venvPython
        }
    }
}

if (-not (Test-Path -LiteralPath $Python)) { Fail "Python executable not found: $Python" }
$pythonInfo = Get-PythonInfo $Python
if (-not $pythonInfo) { Fail "Unable to run Python: $Python" }
if (-not (Test-SupportedPython $pythonInfo)) {
    Fail "$SupportedPythonRange is required; found $($pythonInfo.Version)."
}
Write-Host "Python: $Python ($($pythonInfo.Version))"

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
& $Python -c "from maichart.generation.miacode_native import ensure_available; ensure_available(); print('MiaCode Python judge: ready (no DLL or Qt required)')"
if ($LASTEXITCODE -ne 0) { Fail "MiaCode Python judge/reference data failed verification. Restore the project package; no native build is required." }
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
