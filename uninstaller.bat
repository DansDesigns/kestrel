@echo off
REM ============================================================
REM  Uninstall Kestrel
REM
REM  Put this in the folder Kestrel was installed to and run it,
REM  or run it from anywhere and give it the folder.
REM
REM  It removes Kestrel's own files and shortcuts. It does not
REM  touch models, llama.cpp, or anything else on the machine,
REM  and it asks before it deletes a folder.
REM ============================================================
setlocal EnableDelayedExpansion

REM --- which folder --------------------------------------------
set "TARGET=%~dp0"
if "%~1" neq "" set "TARGET=%~1"
REM  Trailing slash off, or the comparisons below read oddly.
if "!TARGET:~-1!"=="\" set "TARGET=!TARGET:~0,-1!"

echo.
echo   Uninstall Kestrel
echo   -----------------
echo.
echo   Folder: !TARGET!
echo.

REM --- is this actually a Kestrel folder? ------------------------
REM  Checked before anything is deleted. A mistyped path should end
REM  in a refusal, not in somebody's Documents folder being emptied.
if not exist "!TARGET!\kestrel\__init__.py" (
    echo   That does not look like a Kestrel installation — there is
    echo   no kestrel\__init__.py in it.
    echo.
    echo   Run this from inside the folder Kestrel was installed to,
    echo   or pass the folder:
    echo.
    echo       uninstaller.bat "C:\path\to\Kestrel"
    echo.
    pause
    exit /b 1
)

REM --- what will go ----------------------------------------------
echo   This will delete:
echo.
echo     !TARGET!\kestrel          the program
echo     !TARGET!\.venv            its Python libraries
echo     !TARGET!\assets           icons and sounds
if exist "!TARGET!\launcher"  echo     !TARGET!\launcher         the built launcher
if exist "!TARGET!\dist"      echo     !TARGET!\dist             a previous build
echo     the desktop and Start menu shortcuts
echo.
echo   This will NOT touch:
echo.
echo     your models, wherever they are
echo     llama.cpp
echo     your workspaces and saved conversations
echo     settings in %%APPDATA%%\kestrel — delete that folder by hand
echo     if you want them gone too
echo.

set /p "SURE=  Type YES to remove Kestrel: "
if /i not "!SURE!"=="YES" (
    echo.
    echo   Nothing was changed.
    echo.
    pause
    exit /b 0
)

REM --- shortcuts, before the folder they point into disappears ----
echo.
echo   Removing shortcuts...
REM  Only ones named Kestrel, and only in Kestrel's own places. No
REM  other program's entries are read or altered.
if exist "%USERPROFILE%\Desktop\Kestrel.lnk" (
    del /f /q "%USERPROFILE%\Desktop\Kestrel.lnk" 2>nul
)
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Kestrel.lnk" (
    del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Kestrel.lnk" 2>nul
)
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Kestrel" (
    rmdir /s /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Kestrel" 2>nul
)

REM --- the program ------------------------------------------------
echo   Removing the program...

REM  Kestrel may still be running, and a locked file would leave the
REM  folder half-deleted. Asking it to close first is politer than
REM  discovering the problem afterwards.
taskkill /im Kestrel.exe /f >nul 2>&1

for %%D in (kestrel assets personas skills launcher launcher-build dist build .venv) do (
    if exist "!TARGET!\%%D" rmdir /s /q "!TARGET!\%%D" 2>nul
)
for %%F in (kestrel-run.py installer.py build-exe.bat build-installer.bat ^
            install.bat install.sh run.bat run.sh node.bat node.sh ^
            requirements.txt version.txt README.md LICENSE models.json) do (
    if exist "!TARGET!\%%F" del /f /q "!TARGET!\%%F" 2>nul
)

REM --- the folder itself, only if nothing of yours is left ---------
REM  rmdir without /s refuses a folder that still has anything in it,
REM  which is exactly the behaviour wanted: whatever you put there
REM  stays, and so does the folder holding it.
rmdir "!TARGET!" 2>nul

echo.
if exist "!TARGET!" (
    echo   Kestrel is removed. The folder was kept because there is
    echo   still something in it:
    echo.
    dir /b "!TARGET!" 2>nul
    echo.
    echo   Those are yours — delete the folder by hand if you want
    echo   them gone.
) else (
    echo   Kestrel is removed, and the folder with it.
)

echo.
echo   Settings and conversation history are still in:
echo     %APPDATA%\kestrel
echo   Delete that folder if you want a completely clean slate.
echo.
pause
