@echo off
REM Batch script to clean up OpenNeuro dataset directory
REM Keeps only: *.tsv, *T1w.nii.gz, *T2w.nii.gz, and their .json sidecars
REM Deletes all other files and empty folders
REM
REM Usage: cleanup_dataset.bat
REM        cleanup_dataset.bat /dry-run  (preview without deleting)

setlocal enabledelayedexpansion

REM Parse arguments
set dry_run=0
if "%1"=="/dry-run" set dry_run=1
if "%1"=="--dry-run" set dry_run=1

if %dry_run% equ 1 (
    echo.
    echo *** DRY RUN MODE - No files will be deleted ***
    echo.
) else (
    echo.
    echo *** REAL MODE - Files will be deleted ***
    echo.
)

set /a files_kept=0
set /a files_deleted=0
set /a folders_deleted=0

REM Delete unwanted files in all directories
echo Scanning for unwanted files...
echo.

for /r . %%F in (*) do (
    set "filename=%%~nxF"
    set "filepath=%%F"
    
    REM Check if file matches keep patterns
    set keep=0
    
    REM Keep .tsv files
    if "!filename:~-4!"==".tsv" set keep=1
    
    REM Keep *T1w.nii.gz files
    if "!filename:*T1w.nii.gz=!"=="" set keep=1
    
    REM Keep *T2w.nii.gz files
    if "!filename:*T2w.nii.gz=!"=="" set keep=1
    
    REM Keep .json sidecars for T1w/T2w files
    if "!filename:*T1w.json=!"=="" set keep=1
    if "!filename:*T2w.json=!"=="" set keep=1
    
    REM Keep dataset metadata files
    if "!filename!"=="dataset_description.json" set keep=1
    if "!filename!"=="CHANGES" set keep=1
    if "!filename!"=="README" set keep=1
    if "!filename!"=="participants.json" set keep=1
    
    REM Delete if not in keep list
    if !keep! equ 0 (
        set /a files_deleted+=1
        if %dry_run% equ 1 (
            echo [DRY-RUN] DELETE: !filepath!
        ) else (
            echo DELETE: !filepath!
            del "!filepath!"
        )
    ) else (
        set /a files_kept+=1
    )
)

echo.
echo Scanning for empty folders...
echo.

echo Removing non-anat directories (top-level and non-anat subfolders)...
echo.

REM Iterate top-level directories
for /f "delims=" %%D in ('dir /ad /b') do (
    REM If directory name does not start with "sub-", remove it entirely
    echo %%D | findstr /r /c:"^sub-" >nul
    if errorlevel 1 (
        set /a folders_deleted+=1
        if %dry_run% equ 1 (
            echo [DRY-RUN] RMDIR /s "%%D"
        ) else (
            echo RMDIR /s "%%D"
            rmdir /s /q "%%D"
        )
    ) else (
        REM It's a subject folder; check for sessions or direct anat folder
        REM Look for ses-* subdirectories
        set has_sessions=0
        for /d %%S in ("%%D\ses-*") do (
            set has_sessions=1
            REM Within each session, keep only 'anat' and remove other modalities
            for /d %%C in ("%%S\*") do (
                if /i not "%%~nC"=="anat" (
                    set /a folders_deleted+=1
                    if !dry_run! equ 1 (
                        echo [DRY-RUN] RMDIR /s "%%C"
                    ) else (
                        echo RMDIR /s "%%C"
                        rmdir /s /q "%%C"
                    )
                )
            )
        )
        
        REM If no sessions found, check for direct anat folder (non-session dataset)
        if !has_sessions! equ 0 (
            for /d %%C in ("%%D\*") do (
                if /i not "%%~nC"=="anat" (
                    set /a folders_deleted+=1
                    if !dry_run! equ 1 (
                        echo [DRY-RUN] RMDIR /s "%%C"
                    ) else (
                        echo RMDIR /s "%%C"
                        rmdir /s /q "%%C"
                    )
                )
            )
        )
    )
)

REM Delete empty folders (keep scanning until no more empty folders exist)
:delete_empty_folders
set empty_found=0

for /d /r . %%D in (*) do (
    REM Check if folder is empty
    dir "%%D" /b >nul 2>&1
    if errorlevel 1 (
        set empty_found=1
        set /a folders_deleted+=1
        if %dry_run% equ 1 (
            echo [DRY-RUN] RMDIR: %%D
        ) else (
            echo RMDIR: %%D
            rmdir "%%D"
        )
    )
)

REM If we found and deleted empty folders, scan again
if %empty_found% equ 1 goto delete_empty_folders

REM Improved empty-folder removal: iterate directories bottom-up and remove those with no contents
echo Scanning for empty folders (bottom-up)...
for /f "delims=" %%D in ('dir /ad /s /b . ^| sort /r') do (
    REM skip the current directory itself
    if /i not "%%~fD"=="%cd%" (
        REM If directory contains any files or directories, the wildcard will exist
        if exist "%%D\*" (
            REM not empty, skip
        ) else (
            set /a folders_deleted+=1
            if !dry_run! equ 1 (
                echo [DRY-RUN] RMDIR: %%D
            ) else (
                echo RMDIR: %%D
                rmdir "%%D"
            )
        )
    )
)

echo.
echo ============================================================
echo Cleanup Summary
echo ============================================================
echo Files kept:        !files_kept!
echo Files deleted:     !files_deleted!
echo Folders deleted:   !folders_deleted!
echo ============================================================
echo.

if %dry_run% equ 1 (
    echo DRY RUN COMPLETE - No files were actually deleted
    echo.
    echo To perform the actual cleanup, run:
    echo   cleanup_dataset.bat
) else (
    echo CLEANUP COMPLETE
)

echo.
