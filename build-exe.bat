@echo off
REM ============================================================
REM  Build Kestrel.exe
REM
REM  Double-click this file. It finds Python, installs PyInstaller
REM  if it is missing, and builds a launcher with Kestrel's own
REM  icon so the taskbar stops showing Python's.
REM
REM  Put this in the same folder as install.bat and the kestrel
REM  folder. It takes a few minutes the first time.
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo   Building Kestrel.exe
echo   --------------------
echo.

REM --- find a Python we can use -------------------------------
set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    echo   Using the virtual environment in .venv
) else (
    where py >nul 2>&1 && set "PY=py -3"
    if "!PY!"=="" (
        where python >nul 2>&1 && set "PY=python"
    )
)

if "!PY!"=="" (
    echo   Python was not found.
    echo.
    echo   Install it from python.org, tick "Add Python to PATH"
    echo   during setup, then run this again.
    echo.
    pause
    exit /b 1
)

REM --- the entry point the exe is built from -------------------
if not exist "kestrel-run.py" (
    echo   kestrel-run.py is missing from this folder.
    echo   It should sit beside install.bat. Without it there is
    echo   nothing to build.
    echo.
    pause
    exit /b 1
)

REM --- PyInstaller ---------------------------------------------
!PY! -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo   Installing PyInstaller...
    !PY! -m pip install --quiet pyinstaller
    if errorlevel 1 (
        echo.
        echo   Could not install PyInstaller. Check your internet
        echo   connection and try again.
        echo.
        pause
        exit /b 1
    )
)

REM --- icon and bundled folders --------------------------------
REM  Absolute paths throughout: PyInstaller resolves relative ones
REM  against wherever the spec file lands, which is not always here.
set ICON=
if exist "%~dp0assets\kestrel.ico" set ICON=--icon "%~dp0assets\kestrel.ico"

set DATA=
if exist "%~dp0assets"   set DATA=!DATA! --add-data "%~dp0assets;assets"
if exist "%~dp0skills"   set DATA=!DATA! --add-data "%~dp0skills;skills"
if exist "%~dp0personas" set DATA=!DATA! --add-data "%~dp0personas;personas"

REM --- build ----------------------------------------------------
REM  --onedir rather than --onefile: a single-file build unpacks
REM  itself to a temporary folder on every launch, which is slower
REM  to start and hides the source that kestrel-run.py prefers.
echo   Building. This takes a few minutes the first time.
echo.
!PY! -m PyInstaller --noconfirm --clean --windowed --onedir ^
    --name Kestrel !ICON! !DATA! ^
    --hidden-import PySide6.QtSvg ^
    --hidden-import PySide6.QtNetwork ^
    --hidden-import PySide6.QtMultimedia ^
    --collect-submodules kestrel ^
    "%~dp0kestrel-run.py"

if errorlevel 1 (
    echo.
    echo   The build failed. The messages above say why; the usual
    echo   cause is a missing package, which pip will name.
    echo.
    pause
    exit /b 1
)

REM --- tidy away the scratch folder ------------------------------
REM  PyInstaller works in "build" and puts the result in "dist".
REM  The exe that appears under build is an intermediate and does
REM  not run on its own — leaving it there only invites someone to
REM  double-click it and meet a missing-DLL error.
if exist "build\Kestrel" rmdir /s /q "build" 2>nul

REM --- put it where the shortcut looks for it -------------------
if exist "dist\Kestrel\Kestrel.exe" (
    echo.
    echo   Built: dist\Kestrel\Kestrel.exe
    echo   ^(the only one that runs — there is no other^)
    echo.
    echo   Making the desktop and menu shortcut point at it...
    !PY! -m kestrel.shortcut >nul 2>&1
    echo.
    echo   Done. Launch Kestrel from dist\Kestrel\Kestrel.exe, or
    echo   from the shortcut, and the taskbar will show Kestrel's
    echo   own icon.
    echo.
    echo   Updating Kestrel stays a git pull — the exe loads the
    echo   code from this folder rather than carrying its own copy,
    echo   so you only run this again if PySide6 changes.
    echo.
    echo   To make one installer file to give to other people, run
    echo   build-installer.bat next.
) else (
    echo.
    echo   The build reported success but dist\Kestrel\Kestrel.exe
    echo   is not there. Look in the dist folder to see what was
    echo   produced.
)

echo.
pause
