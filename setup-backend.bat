@echo off
echo ========================================
echo AI Waste Segregation - Backend Setup
echo ========================================
echo.

cd backend

echo Creating virtual environment...
python -m venv venv

echo.
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the backend server:
echo   1. cd backend
echo   2. venv\Scripts\activate
echo   3. uvicorn app.main:app --reload
echo.
pause
