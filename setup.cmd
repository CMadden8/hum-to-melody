@echo off
py -3.11 "%~dp0scripts\setup_windows.py"
if errorlevel 1 (echo Setup failed. See the message above. & pause & exit /b 1)
