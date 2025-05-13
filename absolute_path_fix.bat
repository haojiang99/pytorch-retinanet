@echo off
REM absolute_path_fix.bat - Windows batch script to fix paths in annotations

python absolute_path_fix.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Paths fixed with absolute references!
) else (
    echo.
    echo Error: Failed to fix paths.
    exit /b 1
)
