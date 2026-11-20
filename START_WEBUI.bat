@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0" || (
    echo ChroMAI could not open the project folder. Extract the full project to a local folder you can access.
    if not "%MAICHART_NO_PAUSE%"=="1" pause
    exit /b 1
)
set "MAICHART_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "tools\start_webui.ps1" (
    echo ChroMAI startup files are missing. Extract the full project archive.
    popd
    if not "%MAICHART_NO_PAUSE%"=="1" pause
    exit /b 1
)
if not exist "%MAICHART_POWERSHELL%" (
    echo Windows PowerShell is missing. Restore Windows PowerShell before starting ChroMAI.
    popd
    if not "%MAICHART_NO_PAUSE%"=="1" pause
    exit /b 1
)
"%MAICHART_POWERSHELL%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%CD%\tools\start_webui.ps1" %*
set "MAICHART_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%MAICHART_EXIT_CODE%"=="0" echo ChroMAI startup failed ^(exit code %MAICHART_EXIT_CODE%^). See the failed step above.
if not "%MAICHART_NO_PAUSE%"=="1" pause
exit /b %MAICHART_EXIT_CODE%
