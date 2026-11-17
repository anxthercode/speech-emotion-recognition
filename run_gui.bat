@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the environment and install requirements-gui.txt first. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m qt_gui
if errorlevel 1 pause
