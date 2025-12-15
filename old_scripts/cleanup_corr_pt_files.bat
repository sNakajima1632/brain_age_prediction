@echo off
REM Cleanup script to delete all .pt files from CoRR_Preprocessed directory

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
set "TARGET_DIR=%SCRIPT_DIR%CoRR_Preprocessed"

REM Check if target directory exists
if not exist "%TARGET_DIR%" (
    echo Error: CoRR_Preprocessed directory not found at %TARGET_DIR%
    exit /b 1
)

echo Cleaning up .pt files from: %TARGET_DIR%
echo.

REM Count files to be deleted
set "count=0"
for /r "%TARGET_DIR%" %%F in (*.pt) do (
    set /a count+=1
)

if %count% equ 0 (
    echo No .pt files found to delete.
    exit /b 0
)

echo Found %count% .pt file(s) to delete.
echo.

REM Delete all .pt files
set "deleted=0"
for /r "%TARGET_DIR%" %%F in (*.pt) do (
    echo Deleting: %%F
    del /f /q "%%F"
    if !errorlevel! equ 0 (
        set /a deleted+=1
    ) else (
        echo   WARNING: Failed to delete %%F
    )
)

echo.
echo Cleanup complete!
echo Total files deleted: %deleted%/%count%

endlocal
