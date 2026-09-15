@echo off
REM ============================================================
REM  Uninstall Kestrel
REM
REM  Run it from the folder Kestrel was installed to, or give it
REM  the folder:
REM
REM      uninstaller.bat "C:\Program Files\Kestrel"
REM
REM  Removes Kestrel's own files and shortcuts. Models, llama.cpp
REM  and your saved conversations are left alone.
REM
REM  Folders like Program Files need administrator rights. This
REM  asks for them rather than failing quietly — an uninstaller
REM  that says "removed" while the folder is still there is worse
REM  than one that refuses.
REM ============================================================
setlocal EnableDelayedExpansion

set "TARGET=%~dp0"
if "%~1" neq "" set "TARGET=%~1"
if "!TARGET:~-1!"=="\" set "TARGET=!TARGET:~0,-1!"

echo.
echo   Uninstall Kestrel
echo   -----------------
echo   Folder: !TARGET!
echo.

if not exist "!TARGET!\kestrel\__init__.py" (
    echo   That is not a Kestrel installation — there is no
    echo   kestrel\__init__.py in it. Nothing was changed.
    echo.
    pause
    exit /b 1
)

REM --- can we write there? ---------------------------------------
REM  Tested by trying, not guessed from the path: a folder can be
REM  writable anywhere and read-only anywhere.
set "WRITABLE=1"
2>nul ( >"!TARGET!\.kestrel-write-test" echo. ) || set "WRITABLE=0"
if exist "!TARGET!\.kestrel-write-test" del /f /q "!TARGET!\.kestrel-write-test" >nul 2>&1

if "!WRITABLE!"=="0" (
    if "%~2"=="elevated" (
        echo   Even as administrator this folder cannot be written to.
        echo   Something may have the files open — close Kestrel and
        echo   any Explorer window showing that folder, then retry.
        echo.
        pause
        exit /b 1
    )
    echo   This folder needs administrator rights. Asking for them...
    echo.
    powershell -NoProfile -Command ^
        "Start-Process -Verb RunAs -FilePath '%~f0' -ArgumentList '\"!TARGET!\"','elevated'"
    exit /b 0
)

echo   This removes:
echo     the program, its Python libraries and its shortcuts
echo.
echo   This leaves alone:
echo     your models, llama.cpp, your workspaces and conversations,
echo     and your settings in %%APPDATA%%\kestrel
echo.

set /p "SURE=  Type YES to remove Kestrel: "
if /i not "!SURE!"=="YES" (
    echo.
    echo   Nothing was changed.
    echo.
    pause
    exit /b 0
)

REM --- close it first, or locked files leave a half-deletion -------
taskkill /im Kestrel.exe /f >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1

echo.
echo   Removing shortcuts...
if exist "%USERPROFILE%\Desktop\Kestrel.lnk" del /f /q "%USERPROFILE%\Desktop\Kestrel.lnk"
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Kestrel.lnk" del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Kestrel.lnk"

echo   Removing the program...
set "FAILED="

for %%D in (kestrel assets personas skills launcher launcher-build dist build .venv) do (
    if exist "!TARGET!\%%D" (
        rmdir /s /q "!TARGET!\%%D" >nul 2>&1
        REM  Checked afterwards rather than trusted: rmdir reports success
        REM  even when it removed only part of a tree.
        if exist "!TARGET!\%%D" set "FAILED=!FAILED! %%D"
    )
)
for %%F in (kestrel-run.py installer.py build-exe.bat build-installer.bat ^
            install.bat install.sh run.bat run.sh node.bat node.sh ^
            requirements.txt version.txt README.md LICENSE models.json ^
            Kestrel.spec Screenshot.png) do (
    if exist "!TARGET!\%%F" (
        del /f /q "!TARGET!\%%F" >nul 2>&1
        if exist "!TARGET!\%%F" set "FAILED=!FAILED! %%F"
    )
)

REM --- the folder itself, only when nothing of yours is left -------
rmdir "!TARGET!" >nul 2>&1

echo.
if defined FAILED (
    echo   Some things could not be removed:
    echo    !FAILED!
    echo.
    echo   The usual cause is a file still in use. Close Kestrel and
    echo   anything looking at that folder — Explorer windows and
    echo   editors included — then run this again.
    echo.
    pause
    exit /b 1
)

if exist "!TARGET!" (
    echo   Kestrel is removed. The folder was kept because something
    echo   else is in it:
    echo.
    dir /b "!TARGET!"
    echo.
    echo   Those are yours. Delete the folder by hand if you want them
    echo   gone too.
) else (
    echo   Kestrel is removed, and the folder with it.
)

echo.
echo   Settings and conversation history are still in:
echo     %APPDATA%\kestrel
echo   Delete that folder for a completely clean slate.
echo.
pause
