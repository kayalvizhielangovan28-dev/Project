@echo off
title Sales Forecasting & Revenue Prediction System
color 0A
echo ============================================================
echo   SALES FORECASTING ^& REVENUE PREDICTION SYSTEM
echo   Project 2 - Data Science
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b
)

:: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
echo       Done.

:: Run main script
echo [2/3] Running Sales Forecasting Pipeline...
echo.
python sales_forecasting.py

:: Open outputs folder
echo.
echo [3/3] Opening outputs folder...
start "" "outputs"

echo.
echo ============================================================
echo   Project complete! Check the outputs\ folder for charts
echo   and the business_insights_report.txt
echo ============================================================
pause
