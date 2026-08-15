@echo off
REM Launch Kestrel. Any arguments are passed through (--cli, --url, ...).
cd /d "%~dp0"
if not exist .venv (
  echo Not installed yet. Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m kestrel %*
if errorlevel 1 pause
