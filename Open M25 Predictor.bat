@echo off
setlocal
cd /d "%~dp0"
title MMA3001 - Monash Motorsport M25 Downforce Predictor

if not exist "scripts\model_gui.py" (
    echo ERROR: Could not find scripts\model_gui.py
    echo Place this launcher in your project root, next to the scripts folder.
    pause
    exit /b 1
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3 "scripts\model_gui.py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python 3 is not available.
        echo Install Python 3 and the existing project dependencies: numpy, pandas, matplotlib.
        pause
        exit /b 1
    )
    python "scripts\model_gui.py"
)

if errorlevel 1 (
    echo.
    echo The predictor did not start. Check the error message above.
    pause
)
endlocal
