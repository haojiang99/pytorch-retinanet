@echo off
REM visualize_mass_data.bat - Windows batch version to visualize prepared mass data

REM Default paths
set ANNOTATIONS=ddsm_retinanet_data_mass_test\annotations.csv
set CLASS_MAP=ddsm_retinanet_data_mass_test\class_map.csv
set OUTPUT_DIR=ddsm_mass_visualization
set LIMIT=

REM Parse command line arguments
:parse_args
if "%~1"=="" goto check_files
if "%~1"=="--annotations" (
    set ANNOTATIONS=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--class_map" (
    set CLASS_MAP=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--output" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--limit" (
    set LIMIT=--limit %~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [--annotations FILE] [--class_map FILE] [--output DIR] [--limit N]
    echo.
    echo Options:
    echo   --annotations FILE  Path to annotations CSV file
    echo   --class_map FILE    Path to class_map CSV file
    echo   --output DIR        Output directory for visualization images
    echo   --limit N           Limit visualization to N samples
    echo   --help              Show this help message
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:check_files
REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.x
    exit /b 1
)
set PYTHON_CMD=python

REM Check if the annotations file exists
if not exist "%ANNOTATIONS%" (
    echo Error: Annotations file not found: %ANNOTATIONS%
    echo Run prepare_mass_data script first to generate the annotations.
    exit /b 1
)

REM Check if the class map file exists
if not exist "%CLASS_MAP%" (
    echo Error: Class map file not found: %CLASS_MAP%
    echo Run prepare_mass_data script first to generate the class map.
    exit /b 1
)

REM Run the visualization script
echo Starting visualization...
echo Annotations file: %ANNOTATIONS%
echo Class map file: %CLASS_MAP%
echo Output directory: %OUTPUT_DIR%
if defined LIMIT (
    echo Processing limit: %LIMIT% samples
)
echo.

%PYTHON_CMD% visualize_calc_data.py --annotations "%ANNOTATIONS%" --class_map "%CLASS_MAP%" --output_dir "%OUTPUT_DIR%" %LIMIT%

REM Check if the script ran successfully
if %ERRORLEVEL% equ 0 (
    echo.
    echo Visualization completed successfully!
    echo Check the output directory: %OUTPUT_DIR%
) else (
    echo.
    echo Error: Visualization failed.
    exit /b 1
)
