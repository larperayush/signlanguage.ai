@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: 1. Search for pythonw.exe (Windows GUI subsystem - zero terminal attached)
set "PYTHONW_EXE="

if exist ".venv\Scripts\pythonw.exe" (
    set "PYTHONW_EXE=.venv\Scripts\pythonw.exe"
)

if not defined PYTHONW_EXE (
    where pythonw >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHONW_EXE=pythonw"
    )
)

if not defined PYTHONW_EXE (
    where pyw >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHONW_EXE=pyw"
    )
)

:: 2. If pythonw found, launch detached with zero persistent terminal and exit
if defined PYTHONW_EXE (
    start "" "!PYTHONW_EXE!" run_app.py %*
    exit /b 0
)

:: 3. Fallback to standard python if pythonw is not present
if exist ".venv\Scripts\python.exe" (
    start "" ".venv\Scripts\python.exe" run_app.py %*
    exit /b 0
)

where python >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    start "" python run_app.py %*
    exit /b 0
)

where py >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    start "" py run_app.py %*
    exit /b 0
)

echo [ERROR] No working Python installation found in PATH or .venv.
pause
exit /b 1
