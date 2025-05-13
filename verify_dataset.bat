@echo off
REM verify_dataset.bat - Windows batch script to verify and fix dataset

python verify_dataset.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Dataset verification completed successfully!
) else (
    echo.
    echo Error: Dataset verification failed.
    exit /b 1
)
