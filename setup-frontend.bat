@echo off
echo ========================================
echo AI Waste Segregation - Frontend Setup
echo ========================================
echo.

cd frontend

echo Installing dependencies...
call npm install

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the frontend server:
echo   1. cd frontend
echo   2. npm run dev
echo.
echo The app will be available at http://localhost:3000
echo.
pause
