@echo off
REM ============================================================
REM  HR Analytics - Employee Attrition Prediction
REM  Windows Run Script
REM ============================================================

echo.
echo ============================================================
echo   HR Analytics - Employee Attrition Prediction
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r requirements.txt -q
echo       Done.

echo [2/4] Generating HR Dataset...
python data\generate_dataset.py
echo       Done.

echo [3/4] Running Full Analysis Pipeline...
python hr_attrition_analysis.py

echo.
echo [4/4] Pipeline complete!
echo       Charts saved to:  outputs\
echo       Models saved to:  models\
echo       Report saved to:  outputs\HR_Analytics_Report.txt
echo.

set /p launch="Launch Jupyter Notebook? (y/n): "
if /i "%launch%"=="y" (
    echo Starting Jupyter...
    jupyter notebook notebooks\HR_Analytics_Notebook.ipynb
)

pause
