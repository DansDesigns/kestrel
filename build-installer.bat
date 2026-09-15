@echo off
REM ============================================================
REM  Build Kestrel_Installer.exe
REM
REM  Double-click this. It makes one small installer file you can
REM  give to anybody: they run it, choose a folder, and it fetches
REM  Kestrel from GitHub and sets it up.
REM
REM  The program is NOT inside the installer. That keeps it small
REM  and means it never goes stale — it always fetches the current
REM  version, so this only needs rebuilding when installer.py
REM  itself changes.
REM
REM  Produces:  Kestrel_Installer.exe   (in this folder)
REM ============================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo   Building Kestrel_Installer.exe
echo   ------------------------------
echo.

if not exist "installer.py" (
    echo   installer.py is not in this folder. It should sit beside
    echo   install.bat. Without it there is nothing to build.
    echo.
    pause
    exit /b 1
)

REM --- find Python ----------------------------------------------
set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    where py >nul 2>&1 && set "PY=py -3"
    if "!PY!"=="" where python >nul 2>&1 && set "PY=python"
)
if "!PY!"=="" (
    echo   Python was not found. Install it from python.org and
    echo   tick "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

!PY! -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo   Installing PyInstaller...
    !PY! -m pip install --quiet pyinstaller
    if errorlevel 1 (
        echo   Could not install PyInstaller.
        echo.
        pause
        exit /b 1
    )
)

REM  Absolute, not relative. --specpath below moves where PyInstaller
REM  resolves relative paths from, so "assets\kestrel.ico" would be
REM  looked for inside the scratch folder and not found.
set ICON=
if exist "%~dp0assets\kestrel.ico" set ICON=--icon "%~dp0assets\kestrel.ico"

REM --- build -----------------------------------------------------
REM  --onefile, because an installer is one file you hand over. It
REM  runs once, so unpacking to a temporary folder costs nothing
REM  that matters. Only the standard library is needed, so the
REM  result is a few megabytes rather than a hundred.
echo   Building. A minute or two.
echo.
!PY! -m PyInstaller --noconfirm --clean --windowed --onefile ^
    --name Kestrel_Installer !ICON! ^
    --distpath "installer_out" ^
    --workpath "installer_build" ^
    --specpath "installer_build" ^
    "%~dp0installer.py"

if errorlevel 1 (
    echo.
    echo   The build failed. The messages above say why.
    echo.
    pause
    exit /b 1
)

REM --- bring it here and clear the scratch away -------------------
if exist "installer_out\Kestrel_Installer.exe" (
    move /y "installer_out\Kestrel_Installer.exe" "Kestrel_Installer.exe" >nul
    rmdir /s /q "installer_out" 2>nul
    rmdir /s /q "installer_build" 2>nul
    echo.
    echo   Built: Kestrel_Installer.exe
    echo.
    echo   Give that one file to anybody. Running it finds their
    echo   Python, downloads Kestrel, installs the libraries into
    echo   its own virtual environment, and adds a shortcut.
    echo.
    echo   Windows will warn that it is from an unknown publisher,
    echo   because it is not code-signed. Only a signing certificate
    echo   removes that.
) else (
    echo.
    echo   The build reported success but the exe is not in
    echo   installer_out. Look there to see what was produced.
)

echo.
pause
