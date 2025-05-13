@echo off
REM fix_image_paths.bat - Windows batch script to fix image paths in annotations

python fix_image_paths.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Image paths fixed successfully!
) else (
    echo.
    echo Error: Failed to fix image paths.
    exit /b 1
)
