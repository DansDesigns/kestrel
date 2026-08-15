@echo off
REM Turn this machine into a cluster worker: runs llama.cpp's rpc-server and
REM announces itself so the Kestrel head node can find it.
REM
REM   node.bat --mem 8192
REM   node.bat --bin C:\llama.cpp\build\bin\rpc-server.exe --mem 24576
cd /d "%~dp0"
if not exist .venv (
  echo Not installed yet. Run install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m kestrel.node %*
if errorlevel 1 pause
