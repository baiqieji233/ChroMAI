# Windows entry point. Keep the source tree first even if another checkout is installed.
$ErrorActionPreference = 'Stop'
$projectDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectDir
$env:PYTHONPATH = Join-Path $projectDir 'src'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$mode = 'auto'
$checkOnly = $false
$noInstall = $false
$noBrowser = $false
$reinstall = $false
$requestedPort = 7860

try {
    for ($i = 0; $i -lt $args.Count; $i++) {
        switch ($args[$i].ToLowerInvariant()) {
            '--check' { $checkOnly = $true }
            '--no-install' { $noInstall = $true }
            '--no-browser' { $noBrowser = $true }
            '--reinstall' { $reinstall = $true }
            '--cpu' { $mode = 'cpu' }
            '--cuda' { $mode = 'cuda' }
            '--mode' { $i++; if ($i -ge $args.Count) { throw '--mode requires auto, cpu or cuda.' }; $mode = $args[$i].ToLowerInvariant() }
            '--port' { $i++; if ($i -ge $args.Count) { throw '--port requires a number.' }; $requestedPort = [int]$args[$i] }
            '--help' {
                Write-Host 'START_WEBUI.bat [--cpu|--cuda|--mode auto] [--port 7860] [--check] [--no-install] [--no-browser] [--reinstall]'
                Write-Host 'MAICHART_PYTHON: optional base interpreter path; MAICHART_TORCH_INDEX_URL: optional wheel index.'
                Write-Host 'Double-click: create .venv, install dependencies once, choose a free local port and open WebUI.'
                exit 0
            }
            default { throw "Unknown option: $($args[$i]). Use --help." }
        }
    }
    if ($mode -notin @('auto','cpu','cuda')) { throw 'Mode must be auto, cpu or cuda.' }
    if ($requestedPort -lt 1024 -or $requestedPort -gt 65535) { throw 'Port must be 1024..65535.' }

    function Invoke-PythonCommand($command, $prefix, $arguments) {
        & $command @prefix @arguments
        if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE): $($arguments -join ' ')" }
    }
    $candidates = @()
    if ($env:MAICHART_PYTHON) {
        $candidates += @{Command=$env:MAICHART_PYTHON; Prefix=@()}
    } else {
        foreach ($version in @('-3.11','-3.12','-3.10','-3.13')) { $candidates += @{Command='py'; Prefix=@($version)} }
        $candidates += @{Command='python'; Prefix=@()}
    }
    $pythonBase = $null
    $versionCode = 'import sys; print(sys.executable); raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)'
    foreach ($candidate in $candidates) {
        try {
            $found = & $candidate.Command @($candidate.Prefix) -c $versionCode 2>$null
            if ($LASTEXITCODE -eq 0 -and $found) { $pythonBase = [string]($found | Select-Object -Last 1); break }
        } catch { }
    }
    if (-not $pythonBase) { throw 'Python 3.10-3.13 was not found. Install Python 3.11 (64-bit) and run this BAT again.' }
    $venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
    $pythonRun = $pythonBase
    if (Test-Path -LiteralPath $venvPython) { $pythonRun = $venvPython }
    Write-Host "Project: $projectDir"
    Write-Host "Python: $pythonRun"

    if ($mode -eq 'auto') {
        $hasNvidia = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'NVIDIA' }).Count -gt 0
        $resolvedMode = if ($hasNvidia) { 'cuda' } else { 'cpu' }
    } else { $resolvedMode = $mode }
    Write-Host "Dependency mode: $resolvedMode"
    if ($resolvedMode -eq 'cpu') { $env:CUDA_VISIBLE_DEVICES = '-1' }

    $healthCode = @'
import importlib, pathlib, sys
for name in ('torch','numpy','scipy','soundfile','librosa','pytest'):
    importlib.import_module(name)
import torch, maichart
from maichart.generation.event_cli import build_parser
from maichart.training.train_events import build_parser as train_parser
root = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(maichart.__file__).resolve().is_relative_to(root / 'src'), 'Imported another checkout'
assert (root / 'src/maichart/webui/static/generation_presets.json').is_file(), 'WebUI presets missing'
assert (root / 'src/maichart/webui/static/app.js').is_file(), 'WebUI scripts missing'
build_parser().format_help(); train_parser().format_help()
print('Source:', maichart.__file__)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available())
if sys.argv[2] == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA mode requested but CUDA is unavailable. Check the NVIDIA driver or explicitly use --cpu.')
print('Dependency and source checks passed.')
'@
    if ($checkOnly) {
        Invoke-PythonCommand $pythonRun @() @('-c', $healthCode, $projectDir, $resolvedMode)
        exit 0
    }

    $marker = Join-Path $projectDir '.venv\.maichart-install-mode'
    $installedMode = if (Test-Path -LiteralPath $marker) { (Get-Content -LiteralPath $marker -Raw).Trim() } else { '' }
    if (-not $noInstall -and ($reinstall -or $installedMode -ne $resolvedMode -or -not (Test-Path -LiteralPath $venvPython))) {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            Write-Host 'Creating project virtual environment...'
            Invoke-PythonCommand $pythonBase @() @('-m','venv',(Join-Path $projectDir '.venv'))
        }
        $pythonRun = $venvPython
        Write-Host 'Installing dependencies. The first GPU installation may take some time...'
        Invoke-PythonCommand $pythonRun @() @('-m','pip','install','--upgrade','pip','setuptools','wheel')
        $torchIndex = if ($env:MAICHART_TORCH_INDEX_URL) { $env:MAICHART_TORCH_INDEX_URL } elseif ($resolvedMode -eq 'cuda') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
        $torchVersion = if ($resolvedMode -eq 'cuda') { 'torch==2.7.0+cu128' } else { 'torch==2.7.0+cpu' }
        Invoke-PythonCommand $pythonRun @() @('-m','pip','install',$torchVersion,'--index-url',$torchIndex)
        Invoke-PythonCommand $pythonRun @() @('-m','pip','install','-e',"${projectDir}[dev,audio,training]")
        Invoke-PythonCommand $pythonRun @() @('-c',$healthCode,$projectDir,$resolvedMode)
        [IO.File]::WriteAllText($marker, $resolvedMode)
    } else {
        Invoke-PythonCommand $pythonRun @() @('-c',$healthCode,$projectDir,$resolvedMode)
    }

    $chosenPort = $null
    for ($candidatePort = $requestedPort; $candidatePort -le [Math]::Min($requestedPort + 30,65535); $candidatePort++) {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$candidatePort)
        try { $listener.Start(); $chosenPort=$candidatePort; break } catch { } finally { $listener.Stop() }
    }
    if ($null -eq $chosenPort) { throw 'No free local port found; specify another --port.' }
    $url = "http://127.0.0.1:$chosenPort/"
    Write-Host "WebUI: $url"
    Write-Host 'Keep this window open while using WebUI. Ctrl+C stops the service.'
    if ($noBrowser) {
        Invoke-PythonCommand $pythonRun @() @('-B','-m','maichart.webui.server','--host','127.0.0.1','--port',[string]$chosenPort)
    } else {
        # Open only after the server responds; a cold import can take seconds.
        $serveCode = @'
import sys, threading, time, urllib.request, webbrowser
from maichart.webui.server import main
url = 'http://127.0.0.1:' + sys.argv[1] + '/'
def open_when_ready():
    for _ in range(100):
        try:
            with urllib.request.urlopen(url + 'api/status', timeout=.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(.1)
threading.Thread(target=open_when_ready, daemon=True).start()
raise SystemExit(main(['--host','127.0.0.1','--port',sys.argv[1]]))
'@
        Invoke-PythonCommand $pythonRun @() @('-B','-c',$serveCode,[string]$chosenPort)
    }
    exit 0
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'For a check without installation: START_WEBUI.bat --check'
    exit 1
}
