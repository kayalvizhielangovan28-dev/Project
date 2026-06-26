@echo off
echo =====================================================
echo   FRAUD DETECTION PROJECT — Setup and Run
echo =====================================================
echo.

echo [Step 1] Installing required Python libraries...
pip install -r requirements.txt
echo.

echo [Step 2] Running Fraud Detection Analysis...
python fraud_detection_project.py
echo.

echo =====================================================
echo   Done! Check the 'outputs' folder for results.
echo =====================================================
pause
