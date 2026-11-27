@echo off
call "%~dp0START_WEBUI.bat" --hand-preview %*
exit /b %ERRORLEVEL%
