@echo off
REM fix_annotations.bat - Windows batch script to fix annotation files

python fix_annotations.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Annotations fixed successfully!
) else (
    echo.
    echo Error: Failed to fix annotations.
    exit /b 1
)
