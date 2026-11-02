@echo off
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Run setup.cmd first.
    pause
    exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0src\app.py" %*
