@echo off
REM Kestrel installer (Windows)
REM
REM Run with no arguments and it will ask what you want; every choice has a
REM sensible default, so pressing Enter throughout is a valid way to install.
REM
REM   install.bat                              interactive
REM   install.bat --no-llama                   dependencies only, ask nothing
REM   install.bat --llama                      install llama.cpp, auto backend
REM   install.bat --llama --backend vulkan     choose the accelerator
REM   install.bat --llama --source             build from source
REM   install.bat --llama --no-rpc             omit the RPC backend (no clustering)
REM   install.bat --reinstall                  remove Kestrel's copy and install again
REM   install.bat --uninstall-llama            remove Kestrel's copy of llama.cpp
REM   install.bat --speech                     also install offline speech engines
REM   install.bat --no-speech                  skip them without asking
REM   install.bat --yes                        accept every default, no prompts
REM
REM Backends: auto, cpu, cuda, vulkan, hip, sycl.
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "INSTALL_LLAMA="
set "BACKEND="
set "SOURCE="
set "WITH_RPC="
set "REMOVE_FIRST=0"
set "INSTALL_SPEECH="
set "ASSUME_YES=0"

:parse
if "%~1"=="" goto endparse
if /i "%~1"=="--llama"      set "INSTALL_LLAMA=1"
if /i "%~1"=="--no-llama"   set "INSTALL_LLAMA=0"
if /i "%~1"=="--skip-llama" set "INSTALL_LLAMA=0"
if /i "%~1"=="--backend"    (set "BACKEND=%~2" & set "INSTALL_LLAMA=1" & shift)
if /i "%~1"=="--source"     (set "SOURCE=1" & set "INSTALL_LLAMA=1")
if /i "%~1"=="--build"      (set "SOURCE=1" & set "INSTALL_LLAMA=1")
if /i "%~1"=="--prebuilt"   set "SOURCE=0"
if /i "%~1"=="--no-rpc"     set "WITH_RPC=0"
if /i "%~1"=="--rpc"        set "WITH_RPC=1"
if /i "%~1"=="--reinstall"  (set "INSTALL_LLAMA=1" & set "REMOVE_FIRST=1")
if /i "%~1"=="--uninstall-llama" set "INSTALL_LLAMA=2"
if /i "%~1"=="--speech"     set "INSTALL_SPEECH=1"
if /i "%~1"=="--no-speech"  set "INSTALL_SPEECH=0"
if /i "%~1"=="-y"           set "ASSUME_YES=1"
if /i "%~1"=="--yes"        set "ASSUME_YES=1"
if /i "%~1"=="-h"           goto showhelp
if /i "%~1"=="--help"       goto showhelp
shift
goto parse
:showhelp
findstr /b "REM" "%~f0"
exit /b 0
:endparse

REM ---------------------------------------------------------------- python ---
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo No Python found. Install Python 3.10 or newer from python.org,
  echo tick "Add python.exe to PATH", then run this again.
  pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo Kestrel needs Python 3.10 or newer.
  %PY% --version
  pause
  exit /b 1
)

if not exist .venv (
  echo Creating virtual environment in .venv
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
echo Installing dependencies ^(PySide6 is a large download the first time^)
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

REM ------------------------------------------------------------- llama.cpp ---
echo.
echo ----------------------------------------------------------------
echo llama.cpp - the inference backend Kestrel drives
echo.

REM Detection is written out as a batch fragment and executed, rather than
REM parsed with for/f. `for /f` collapses consecutive delimiters exactly as the
REM shell does, so an empty field (no version, no rpc-server) silently shifts
REM every value after it into the wrong variable.
set "SCANFILE=%TEMP%\kestrel-scan-%RANDOM%.cmd"
python -m kestrel.setup_backend --emit-batch > "%SCANFILE%" 2>nul
if exist "%SCANFILE%" call "%SCANFILE%"
if exist "%SCANFILE%" del "%SCANFILE%" >nul 2>&1
if not defined DETECTED set "DETECTED=cpu"

if defined HAVE_SERVER (
  echo   Found: !HAVE_SERVER! !HAVE_VER!
  if defined HAVE_RPC (
    echo   rpc-server:        !HAVE_RPC!  - this machine can join a cluster
  ) else (
    echo   rpc-server:        not present - clustering unavailable
  )
  if "!HAVE_OK!"=="1" (
    echo   It runs correctly.
    set "DEFAULT_ACTION=2"
  ) else (
    echo.
    echo   PROBLEM: this installation does not work.
    if defined HAVE_PROBLEM echo            !HAVE_PROBLEM!
    set "DEFAULT_ACTION=1"
  )
  if "!HAVE_MANAGED!"=="1" (
    set "OPT1=Remove it and install a fresh copy"
  ) else (
    set "OPT1=Install a fresh copy for Kestrel to use (leaves yours untouched)"
  )
  if not defined INSTALL_LLAMA (
    if "%ASSUME_YES%"=="1" (set "INSTALL_LLAMA=0") else (
      echo.
      echo   What would you like to do?
      echo     1^) !OPT1!
      echo     2^) Keep this installation and carry on
      echo     3^) Nothing for now - decide later in the Backend tab
      set "R="
      set /p "R=  Choice [!DEFAULT_ACTION!]: "
      if not defined R set "R=!DEFAULT_ACTION!"
      if "!R!"=="1" (
        set "INSTALL_LLAMA=1"
        if "!HAVE_MANAGED!"=="1" set "REMOVE_FIRST=1"
      ) else (
        set "INSTALL_LLAMA=0"
      )
    )
  )
  if "!INSTALL_LLAMA!"=="1" if not "!HAVE_MANAGED!"=="1" (
    echo.
    echo   Note: Kestrel did not install that copy, so it will not be removed.
    echo   The new copy goes in Kestrel's own directory and will be preferred.
  )
) else (
  echo   No llama.cpp installation was found on this machine.
  if not defined INSTALL_LLAMA (
    if "%ASSUME_YES%"=="1" (set "INSTALL_LLAMA=1") else (
      set "R="
      set /p "R=  Install it now? [Y/n] "
      if /i "!R!"=="n" (set "INSTALL_LLAMA=0") else (set "INSTALL_LLAMA=1")
    )
  )
)

if "%INSTALL_LLAMA%"=="2" (
  echo.
  python -m kestrel.setup_backend --uninstall
  goto done
)
if "%INSTALL_LLAMA%"=="1" goto configure
echo.
echo   Skipping. Kestrel can find or install llama.cpp later from the Backend tab.
goto done

:configure
if defined SOURCE goto pickbackend
if "%ASSUME_YES%"=="1" (set "SOURCE=0" & goto pickbackend)
echo.
echo   How would you like it installed?
echo     1^) Download an official prebuilt build   ^(fast, recommended^)
echo     2^) Compile from source                   ^(slower; needs cmake and a compiler^)
set "R="
set /p "R=  Choice [1]: "
if "!R!"=="2" (set "SOURCE=1") else (set "SOURCE=0")

:pickbackend
if defined BACKEND goto pickrpc
if "%ASSUME_YES%"=="1" (set "BACKEND=auto" & goto pickrpc)
echo.
echo   Which accelerator? This machine looks like: %DETECTED%
echo     1^) auto   - detect and choose ^(%DETECTED%^)   ^(recommended^)
echo     2^) cpu    - no GPU acceleration
echo     3^) cuda   - NVIDIA
echo     4^) vulkan - any modern GPU, including Intel and older AMD
echo     5^) hip    - AMD ROCm
echo     6^) sycl   - Intel oneAPI
set "R="
set /p "R=  Choice [1]: "
set "BACKEND=auto"
if "!R!"=="2" set "BACKEND=cpu"
if "!R!"=="3" set "BACKEND=cuda"
if "!R!"=="4" set "BACKEND=vulkan"
if "!R!"=="5" set "BACKEND=hip"
if "!R!"=="6" set "BACKEND=sycl"

:pickrpc
if defined WITH_RPC goto install
if "%ASSUME_YES%"=="1" (set "WITH_RPC=1" & goto install)
echo.
echo   The RPC backend lets this machine join or host a cluster, so a model
echo   too large for one machine can be spread across several.
set "R="
set /p "R=  Include RPC support? [Y/n] "
if /i "!R!"=="n" (set "WITH_RPC=0") else (set "WITH_RPC=1")

:install
set "EXTRA="
if "%SOURCE%"=="1"   set "EXTRA=%EXTRA% --source"
if "%WITH_RPC%"=="0" set "EXTRA=%EXTRA% --no-rpc"
if "%REMOVE_FIRST%"=="1" set "EXTRA=%EXTRA% --reinstall"
set "HOW=prebuilt release"
if "%SOURCE%"=="1" set "HOW=source build"
set "RPCSTATE=on"
if "%WITH_RPC%"=="0" set "RPCSTATE=off"
echo.
echo   Installing: %HOW%, backend %BACKEND%, RPC %RPCSTATE%
echo.
python -m kestrel.setup_backend --backend %BACKEND%%EXTRA%
if errorlevel 1 (
  echo.
  echo   llama.cpp installation did not complete.
  echo   Kestrel is installed and usable: open it and use the Backend tab to
  echo   retry, or point it at an existing build.
) else (
  echo   llama.cpp ready.
)

:done
echo.
echo ----------------------------------------------------------------
echo Offline speech ^(optional^)
echo.
echo   Piper reads replies aloud and faster-whisper transcribes dictation.
echo   Both run locally; nothing is sent anywhere. About 150 MB.
if not defined INSTALL_SPEECH (
  if "%ASSUME_YES%"=="1" (set "INSTALL_SPEECH=0") else (
    set "R="
    set /p "R=  Install them? [y/N] "
    if /i "!R!"=="y" (set "INSTALL_SPEECH=1") else (set "INSTALL_SPEECH=0")
  )
)
if "%INSTALL_SPEECH%"=="1" (
  python -m pip install piper-tts faster-whisper sounddevice soundfile
  if errorlevel 1 (
    echo   Speech engines did not install. Kestrel still works; the Speech tab
    echo   has an Install button to retry.
  )
)

python -m kestrel.shortcut --quiet

echo.
echo ----------------------------------------------------------------
echo.
echo Installed.
echo.
echo   run.bat                  open the interface
echo   run.bat --cli            headless mode
echo   node.bat --mem 8192      turn this machine into a worker for the cluster
echo.
echo Kestrel drives any OpenAI-compatible endpoint, so an existing llama-server,
echo LM Studio, llamafile or vLLM will work too. Point it at one under Settings,
echo or load a GGUF directly from the Models tab.
echo.
pause
