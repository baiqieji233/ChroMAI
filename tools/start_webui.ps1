[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ScriptArgs
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LaunchStage = "startup options"
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
    [Console]::Error.WriteLine("ChroMAI failed during $LaunchStage.")
    [Console]::Error.WriteLine($Message)
    exit 1
}

trap { Fail $_.Exception.Message }

foreach ($file in "pyproject.toml", "src\maichart\webui\server.py", "src\maichart\webui\static\index.html") {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $file) -PathType Leaf)) {
        Fail "Missing $file. Extract the full ChroMAI archive to a writable folder before launching."
    }
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
        "^--port=(.*)$" {
            if (-not [int]::TryParse($Matches[1], [ref]$Port) -or $Port -lt 1 -or $Port -gt 65535) {
                Fail "Invalid port. Use a number from 1 to 65535."
            }
            continue
        }
        default { Fail "Unknown option '$argument'. Supported options: --check --cpu --cuda --port N --no-browser --no-install --reinstall" }
    }
}

if ($Cpu -and $Cuda) { Fail "Choose only one of --cpu and --cuda." }
if ($Reinstall) { $NoInstall = $false }

function Read-Probe([string] $Executable, [string] $Arguments) {
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo.FileName = $Executable
    $process.StartInfo.Arguments = $Arguments
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $process.StartInfo.EnvironmentVariables["PYTHON_MANAGER_AUTOMATIC_INSTALL"] = "false"
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(10000)) {
            $process.Kill()
            return $null
        }
        if ($process.ExitCode -eq 0 -and $stdout.Wait(1000)) { return $stdout.Result }
    } catch {
        return $null
    } finally {
        $process.Dispose()
    }
}

function Get-PythonInfo([string] $Executable) {
    if ($Executable -match '[\\/]WindowsApps[\\/]python[^\\/]*\.exe$') { return $null }
    $command = '-I -S -c "import sys,struct,sysconfig; print(''%d.%d|%d|%s|%s|%s|%s'' % (*sys.version_info[:2],struct.calcsize(''P'')*8,sys.implementation.name,sys.version_info.releaselevel,getattr(sys,''abiflags'',''''),sysconfig.get_platform()))"'
    $output = Read-Probe $Executable $command
    if ($output -and $output.Trim() -match '^(\d+\.\d+)\|(\d+)\|([^|]+)\|([^|]+)\|([^|]*)\|([^|]+)$') {
        return [pscustomobject]@{
            Path = $Executable
            Version = [version]$Matches[1]
            Bits = [int]$Matches[2]
            Implementation = $Matches[3]
            Release = $Matches[4]
            FreeThreaded = $Matches[5] -match 't'
            Platform = $Matches[6]
        }
    }
}

function Test-SupportedPython($Info) {
    return ($null -ne $Info -and $Info.Version -ge $SupportedPythonMin -and $Info.Version -lt $SupportedPythonMax -and
        $Info.Bits -eq 64 -and $Info.Implementation -eq "cpython" -and $Info.Release -eq "final" -and -not $Info.FreeThreaded)
}

function Test-AutomaticPython($Info) {
    # The Composer v2 Transformers pin resolves tokenizers 0.15.x.
    # Its Windows x64 wheels stop at CPython 3.12.
    return ((Test-SupportedPython $Info) -and $Info.Platform -eq "win-amd64" -and
        $Info.Version -le [version]"3.12")
}

function Get-InstalledPythonPaths {
    foreach ($launcher in @(Get-Command py.exe -CommandType Application -All -ErrorAction SilentlyContinue)) {
        foreach ($line in ((Read-Probe $launcher.Source "-0p") -split "\r?\n")) {
            if ($line -match '^\s*-\S+\s+\*?\s*(.+?\.exe)\s*$') { $Matches[1] }
        }
    }
    foreach ($name in "python.exe", "python3.exe") {
        Get-Command $name -CommandType Application -All -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Source }
    }
    if ($env:LOCALAPPDATA) {
        Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
        Join-Path $env:LOCALAPPDATA "Python\pythoncore-3.12-64\python.exe"
    }
    foreach ($key in "HKCU:\Software\Python\PythonCore", "HKLM:\Software\Python\PythonCore") {
        foreach ($versionKey in @(Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
            $install = Get-Item -LiteralPath (Join-Path $versionKey.PSPath "InstallPath") -ErrorAction SilentlyContinue
            if ($install) {
                $exe = $install.GetValue("ExecutablePath")
                if ($exe) { $exe }
                elseif ($install.GetValue("")) { Join-Path ($install.GetValue("")) "python.exe" }
            }
        }
    }
}

function Find-BootstrapPython {
    Get-InstalledPythonPaths | Select-Object -Unique |
        ForEach-Object { Get-PythonInfo $_ } | Where-Object { Test-AutomaticPython $_ } |
        Sort-Object Version -Descending | Where-Object {
            Read-Probe $_.Path '-I -c "import ensurepip,venv; print(ensurepip.version())"'
        } | Select-Object -First 1
}

function Install-BootstrapPython {
    $manager = Get-Command pymanager.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($manager) {
        Write-Host "Installing Python 3.12 x64 with Python Install Manager..."
        try {
            & $manager.Source install "PythonCore/3.12-64" | Out-Host
        } catch {
            Write-Warning $_.Exception.Message
        }
        $installed = Find-BootstrapPython
        if ($installed) { return $installed }
        Write-Warning "Python Install Manager could not supply Python 3.12 x64. Trying the standalone installer."
    }

    $url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
    $sha256 = "67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB"
    $installer = Join-Path ([IO.Path]::GetTempPath()) ("chromai-python-" + [guid]::NewGuid() + ".exe")
    try {
        Write-Host "Downloading Python 3.12.10 x64 from python.org..."
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer -TimeoutSec 180
        if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne $sha256) {
            Fail "Python installer checksum verification failed. No installer was run."
        }
        Write-Host "Installing Python 3.12 for the current Windows user..."
        $process = Start-Process -FilePath $installer -ArgumentList @(
            "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
            "Include_test=0", "Include_pip=1", "Shortcuts=0", "AssociateFiles=0"
        ) -WindowStyle Hidden -Wait -PassThru
        $installed = Find-BootstrapPython
        if ($installed) { return $installed }
        Fail "Python 3.12 was unavailable after setup (installer exit code $($process.ExitCode)). Check Windows installation restrictions and https://www.python.org/downloads/windows/."
    } catch {
        Fail "Could not set up Python 3.12. Check internet/proxy access and Windows installation permissions. $($_.Exception.Message)"
    } finally {
        Remove-Item -LiteralPath $installer -ErrorAction SilentlyContinue
    }
}

$LaunchStage = "startup lock"
# Serialize environment creation and pip repair for simultaneous double-clicks.
$hasher = [Security.Cryptography.SHA256]::Create()
try {
    $rootHash = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($ProjectRoot.ToUpperInvariant())))
} finally { $hasher.Dispose() }
$startupLock = New-Object System.Threading.Mutex($false, "Local\ChroMAI-$rootHash")
try {
    $lockAcquired = $startupLock.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $lockAcquired = $true
}
if (-not $lockAcquired) {
    Fail "ChroMAI is already preparing this project. Wait for that window to finish."
}

$LaunchStage = "Python discovery"
$overridePython = $env:MAICHART_PYTHON
$overrideSource = "MAICHART_PYTHON"
$pythonConfig = Join-Path $ProjectRoot ".webui-python"
if (-not $overridePython -and (Test-Path -LiteralPath $pythonConfig)) {
    $overridePython = ([string](Get-Content -LiteralPath $pythonConfig -Raw -Encoding UTF8)).Trim()
    $overrideSource = ".webui-python"
    if (-not $overridePython) { Fail ".webui-python is empty. Set it to the path of an existing $SupportedPythonRange executable." }
}
if ($overridePython) {
    $overridePython = $overridePython.Trim().Trim('"')
    if (-not $overridePython) { Fail "$overrideSource is empty. Enter the full path of python.exe." }
    if (-not [System.IO.Path]::IsPathRooted($overridePython)) {
        $overridePython = Join-Path $ProjectRoot $overridePython
    }
    $Python = if (Test-Path -LiteralPath $overridePython -PathType Leaf) { (Resolve-Path -LiteralPath $overridePython).Path }
    if (-not $Python) { Fail "$overrideSource does not point to an existing Python executable: $overridePython" }
    $pythonInfo = Get-PythonInfo $Python
    if (-not $pythonInfo) { Fail "Unable to run Python from ${overrideSource}: $Python" }
    if (-not (Test-SupportedPython $pythonInfo)) {
        Fail "$overrideSource needs stable 64-bit CPython ($SupportedPythonRange); found $($pythonInfo.Implementation) $($pythonInfo.Version), $($pythonInfo.Bits)-bit, $($pythonInfo.Release)."
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
        if ((Test-AutomaticPython $existingVenvInfo) -or
            ($configuredVenv -and (Test-SupportedPython $existingVenvInfo))) {
            $Python = $venvPython
        }
    }

    if (-not $Python -and -not $configuredVenv) {
        $alternateRoot = Join-Path $ProjectRoot ".venv312"
        $alternate = $alternateRoot
        $suffix = 1
        while (Test-Path -LiteralPath $alternate) {
            $venvPython = Join-Path $alternate "Scripts\python.exe"
            if (Test-AutomaticPython (Get-PythonInfo $venvPython)) {
                $Python = $venvPython
                break
            }
            $alternate = "$alternateRoot-$suffix"
            $suffix++
        }
        if (Test-Path -LiteralPath $venvRoot) {
            Write-Host "Preserving existing environment: $venvRoot"
            $venvRoot = $alternate
        }
    }

    if (-not $Python) {
        if ($configuredVenv -and (Test-Path -LiteralPath $venvRoot)) {
            Fail "Configured environment '$venvRoot' is incomplete or incompatible. Set MAICHART_VENV to a new directory or select another interpreter with MAICHART_PYTHON."
        }
        $selected = Find-BootstrapPython
        if (-not $selected) {
            if ($CheckOnly -or $NoInstall) {
                Fail "No compatible Python 3.10-3.12 x64 found. Launch normally to allow installation, or select an existing environment with MAICHART_PYTHON."
            }
            $LaunchStage = "Python installation"
            $selected = Install-BootstrapPython
        }
        $bootstrapPython = $selected.Path

        if ($CheckOnly -or $NoInstall) {
            $Python = $bootstrapPython
        } else {
            $LaunchStage = "virtual environment setup"
            Write-Host "Creating virtual environment: $venvRoot"
            & $bootstrapPython -I -m venv $venvRoot
            if ($LASTEXITCODE -ne 0) { Fail "Could not create '$venvRoot'. Check disk space and folder permissions. A full Python installation must include venv and ensurepip." }
            $Python = Join-Path $venvRoot "Scripts\python.exe"
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

# Run the shared check in the interpreter that will launch all task children.
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue

$LaunchStage = "dependency check"
$DependencyArgs = @("--profile", "composer-v2")
$autoCuda = $false
if ($Cpu) { $env:CUDA_VISIBLE_DEVICES = "-1" }
elseif ($Cuda) { $DependencyArgs += "--require-cuda" }
elseif ($env:CUDA_VISIBLE_DEVICES -notin @("", "-1") -and
    (Get-Command nvidia-smi.exe -CommandType Application -ErrorAction SilentlyContinue)) {
    $autoCuda = $true
    $DependencyArgs += "--auto-device"
}
Push-Location $ProjectRoot
try {
    & $Python -m maichart.webui.startup_check @DependencyArgs
    if ($LASTEXITCODE -ne 0 -and $autoCuda) {
        # Reuse a working CPU environment after an unavailable CUDA setup.
        & $Python -m maichart.webui.startup_check --profile composer-v2
        if ($LASTEXITCODE -eq 0) {
            Write-Warning "CUDA is unavailable in this environment. Continuing on CPU. Use --cuda to request CUDA setup."
            $Cpu = $true
            $env:CUDA_VISIBLE_DEVICES = "-1"
        }
    }
    if ($LASTEXITCODE -ne 0) {
        if ($CheckOnly -or $NoInstall) {
            Fail "Dependencies or device checks failed. See the details above. Rerun without --check/--no-install to allow repair."
        }
        $LaunchStage = "dependency installation"
        & $Python -m pip --version
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Restoring pip in the selected Python environment..."
            & $Python -m ensurepip --upgrade
            if ($LASTEXITCODE -ne 0) { Fail "Could not restore pip in '$Python'. Check the error above or select a full Python installation." }
        }
        & $Python -m maichart.webui.startup_check @DependencyArgs --install
        if ($LASTEXITCODE -ne 0 -and $autoCuda) {
            Write-Warning "Automatic device setup failed. Checking whether startup can continue on CPU."
            $Cpu = $true
            $env:CUDA_VISIBLE_DEVICES = "-1"
            & $Python -m maichart.webui.startup_check --profile composer-v2
            if ($LASTEXITCODE -ne 0) {
                & $Python -m maichart.webui.startup_check --profile composer-v2 --install
            }
        }
        if ($LASTEXITCODE -ne 0) { Fail "See the first error above. For download errors, check the network or proxy. For unavailable wheels, use 64-bit Python 3.12." }
    }
} finally { Pop-Location }

if ($Cpu) {
    Write-Host "Device mode: CPU"
} elseif ($Cuda) {
    & $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) { Fail "CUDA was requested, but this Python environment cannot access a CUDA device." }
    Write-Host "Device mode: CUDA"
}

$LaunchStage = "MiaCode reference check"
& $Python -c "from maichart.generation.miacode_native import ensure_available; ensure_available(); print('MiaCode Python judge: ready (no DLL or Qt required)')"
if ($LASTEXITCODE -ne 0) { Fail "MiaCode Python judge/reference data failed verification. Restore the project package; no native build is required." }
$startupLock.ReleaseMutex()
$startupLock.Dispose()
if ($CheckOnly) {
    Write-Host "Environment check passed. WebUI was not started."
    exit 0
}

$LaunchStage = "WebUI startup"
$serverArgs = @("--host", "127.0.0.1", "--port", "$Port")
if (-not $NoBrowser) { $serverArgs += "--open-browser" }
Push-Location $ProjectRoot
try {
    & $Python -m maichart.webui.server @serverArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
