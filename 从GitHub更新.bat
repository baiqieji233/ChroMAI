@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if not exist "%~dp0tools\update_project.ps1" (
    echo 找不到更新组件。请解压完整的 ChroMAI 项目后再运行。
    if not "%MAICHART_NO_PAUSE%"=="1" pause
    exit /b 1
)
(
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\update_project.ps1"
    if errorlevel 1 (
        if not "%MAICHART_NO_PAUSE%"=="1" pause
        exit /b 1
    )
    if not "%MAICHART_NO_PAUSE%"=="1" pause
    exit /b 0
)
