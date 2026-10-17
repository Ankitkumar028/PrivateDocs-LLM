@echo off
if exist "%~dp0make.ps1" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0make.ps1" %*
) else (
    echo [ERROR] make.ps1 not found in %~dp0
    exit /b 1
)
