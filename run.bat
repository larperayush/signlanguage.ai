@echo off
setlocal enabledelayedexpansion
title SignFlow Studio - ASL Recognition

:: Change directory to where this batch file is located
cd /d "%~dp0"

echo ========================================================
echo Launching SignFlow Studio...
echo ========================================================

:: 1. Search for available Python executable dynamically
set "PYTHON_EXE="

:: Check if local virtual environment exists and is functional
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "exit(0)" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
    )
)

:: Check system python
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        python -c "exit(0)" >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            set "PYTHON_EXE=python"
        )
    )
)

:: Check py launcher
if not defined PYTHON_EXE (
    where py >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        py -c "exit(0)" >nul 2>&1
        if !ERRORLEVEL! EQU 0 (
            set "PYTHON_EXE=py"
        )
    )
)

:: If still not found, alert user
if not defined PYTHON_EXE (
    echo [ERROR] No working Python installation found in PATH or .venv.
    echo Please make sure Python is installed and added to PATH on this PC.
    echo.
    pause
    exit /b 1
)

:: 2. Launch the application
echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% run_app.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error code: %ERRORLEVEL%
    pause
)

