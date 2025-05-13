@echo off
REM combine_dataset.bat - Windows batch script to run combine_dataset.py

python combine_dataset.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Dataset combination completed successfully!
) else (
    echo.
    echo Error: Dataset combination failed.
    exit /b 1
)
