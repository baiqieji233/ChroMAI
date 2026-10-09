@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_webui.ps1" %*
set "MAICHART_EXIT_CODE=%ERRORLEVEL%"
if not "%MAICHART_EXIT_CODE%"=="0" echo Deployment or startup failed. Read the message above.
if not "%MAICHART_NO_PAUSE%"=="1" pause
exit /b %MAICHART_EXIT_CODE%
