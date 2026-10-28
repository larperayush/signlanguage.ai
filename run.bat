@echo off
title SignFlow Studio - ASL Recognition
echo Launching SignFlow Studio...
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe run_app.py %*
) else (
    python run_app.py %*
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error code: %ERRORLEVEL%
    pause
)
