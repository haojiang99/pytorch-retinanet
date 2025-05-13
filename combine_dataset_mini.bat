@echo off
REM combine_dataset_mini.bat - Windows batch script to run combine_dataset_mini.py

python combine_dataset_mini.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Mini dataset creation completed successfully!
) else (
    echo.
    echo Error: Mini dataset creation failed.
    exit /b 1
)
