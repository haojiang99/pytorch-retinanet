@echo off
REM train_ddsm.bat - Windows batch script to train RetinaNet on DDSM data

python train_ddsm.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo Training completed successfully!
) else (
    echo.
    echo Error: Training failed.
    exit /b 1
)
