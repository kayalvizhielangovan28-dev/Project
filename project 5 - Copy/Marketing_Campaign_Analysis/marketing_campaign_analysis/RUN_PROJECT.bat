@echo off
REM ============================================================
REM  Marketing Campaign Analysis — Windows Launcher
REM  Double-click this file to run the full project
REM ============================================================

title Marketing Campaign Analysis

echo.
echo  =====================================================
echo    Marketing Campaign Performance Analysis
echo  =====================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    pause
    exit /B 1
)

REM Install dependencies
echo [1/4] Installing required packages...
pip install pandas numpy matplotlib seaborn scikit-learn jupyter --quiet
IF ERRORLEVEL 1 (
    echo ERROR: Package installation failed.
    pause
    exit /B 1
)

REM Generate dataset
echo [2/4] Generating marketing dataset...
python src\generate_data.py
IF ERRORLEVEL 1 (
    echo ERROR: Dataset generation failed.
    pause
    exit /B 1
)

REM Run analysis
echo [3/4] Running full analysis (charts + insights)...
python src\analysis.py
IF ERRORLEVEL 1 (
    echo ERROR: Analysis failed.
    pause
    exit /B 1
)

REM Open outputs folder
echo [4/4] Opening outputs folder...
start "" outputs

echo.
echo =====================================================
echo   SUCCESS! All outputs saved in the 'outputs' folder
echo =====================================================
echo.
echo   To explore interactively, run:
echo     jupyter notebook notebooks\marketing_analysis.ipynb
echo.
pause
