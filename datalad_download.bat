@echo off
REM Batch script to download T1w and T2w files from OpenNeuro dataset using datalad
REM Usage: datalad_download.bat
REM
REM This script assumes you are in the dataset directory and have datalad installed.
REM It will recursively search for sub-*/anat/ directories and download T1w and T2w files.

setlocal enabledelayedexpansion

REM Check if datalad is available
where datalad >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: datalad is not installed or not in PATH
    echo Install datalad with: pip install datalad
    exit /b 1
)

REM Check if we're in a datalad dataset
if not exist ".datalad" (
    echo Error: .datalad directory not found
    echo Make sure you're in the root directory of a datalad dataset
    exit /b 1
)

echo.
echo Starting datalad downloads for T1w and T2w files...
echo.

setlocal enabledelayedexpansion
set /a total_files=0
set /a successful=0
set /a failed=0

REM Loop through all sub-*/anat/ directories (non-session datasets)
for /d %%S in (sub-*) do (
    if exist "%%S\anat" (
        echo.
        echo Processing %%S\anat...
        
        REM Download all T1w files
        for %%F in ("%%S\anat\*T1w.nii.gz") do (
            if exist "%%F" (
                set /a total_files+=1
                echo   Downloading %%F...
                datalad get "%%F"
                if !errorlevel! equ 0 (
                    set /a successful+=1
                    echo   ✓ Success
                ) else (
                    set /a failed+=1
                    echo   ✗ Failed
                )
            )
        )
        
        REM Download all T2w files
        for %%F in ("%%S\anat\*T2w.nii.gz") do (
            if exist "%%F" (
                set /a total_files+=1
                echo   Downloading %%F...
                datalad get "%%F"
                if !errorlevel! equ 0 (
                    set /a successful+=1
                    echo   ✓ Success
                ) else (
                    set /a failed+=1
                    echo   ✗ Failed
                )
            )
        )
    )
)

REM Loop through all sub-*/ses-*/anat/ directories (session datasets)
for /d %%S in (sub-*) do (
    for /d %%E in ("%%S\ses-*") do (
        if exist "%%E\anat" (
            echo.
            echo Processing %%E\anat...
            
            REM Download all T1w files
            for %%F in ("%%E\anat\*T1w.nii.gz") do (
                if exist "%%F" (
                    set /a total_files+=1
                    echo   Downloading %%F...
                    datalad get "%%F"
                    if !errorlevel! equ 0 (
                        set /a successful+=1
                        echo   ✓ Success
                    ) else (
                        set /a failed+=1
                        echo   ✗ Failed
                    )
                )
            )
            
            REM Download all T2w files
            for %%F in ("%%E\anat\*T2w.nii.gz") do (
                if exist "%%F" (
                    set /a total_files+=1
                    echo   Downloading %%F...
                    datalad get "%%F"
                    if !errorlevel! equ 0 (
                        set /a successful+=1
                        echo   ✓ Success
                    ) else (
                        set /a failed+=1
                        echo   ✗ Failed
                    )
                )
            )
        )
    )
)

echo.
echo ============================================================
echo Download Summary
echo ============================================================
echo Total files:    !total_files!
echo Successful:     !successful!
echo Failed:         !failed!
echo ============================================================
echo.

if %failed% gtr 0 (
    exit /b 1
) else (
    echo All downloads completed successfully!
    exit /b 0
)
