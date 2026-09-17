@echo off
REM Launch Kestrel. Any arguments are passed through (--cli, --url, ...).
REM
REM The window closes as soon as Kestrel is up: pythonw has no console, and
REM this batch file exits rather than waiting. A console left sitting behind
REM the interface looks like leftover rubbish, and closing it would take
REM Kestrel with it — it owns whatever it started.
cd /d "%~dp0"

if not exist .venv (
  echo Not installed yet. Run install.bat first.
  pause
  exit /b 1
)

REM The built launcher first, when there is one: its own icon and name in the
REM taskbar, rather than Python's.
if exist "Kestrel.exe" (
  start "" "Kestrel.exe" %*
  exit /b 0
)
if exist "dist\Kestrel\Kestrel.exe" (
  start "" "dist\Kestrel\Kestrel.exe" %*
  exit /b 0
)

REM Anything after --cli or --headless wants a console to print into, so those
REM keep one and wait.
echo %* | findstr /i /c:"--cli" /c:"--headless" >nul
if not errorlevel 1 (
  call .venv\Scripts\activate.bat
  python -m kestrel %*
  if errorlevel 1 pause
  exit /b 0
)

if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m kestrel %*
  exit /b 0
)

REM No pythonw — a console is unavoidable, so at least explain it.
echo Starting Kestrel. This window has to stay open; closing it closes Kestrel.
call .venv\Scripts\activate.bat
python -m kestrel %*
if errorlevel 1 pause
