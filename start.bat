@echo off
REM DataSentinel Quick Start Script (Windows)

REM Check if models are trained
if not exist "src\models\model_a.pkl" (
    echo  Model A not found. Training...
    cd src
    python model_a.py
    cd ..
)

if not exist "src\models\model_b.pkl" (
    echo  Model B not found. Training...
    cd src
    python3.12 model_b.py
    cd ..
)

if not exist "src\models\model_c.pkl" (
    echo  Model C not found. Training...
    cd src
    python model_c.py
    cd ..
)

echo.
echo All models found or trained.
echo.
echo Starting DataSentinel services:
echo   - FastAPI Backend: http://localhost:8000
echo   - Streamlit Dashboard: http://localhost:8501
echo   - API Documentation: http://localhost:8000/docs
echo.

REM Start FastAPI in a new window
echo Launching FastAPI Backend...
start "DataSentinel FastAPI" cmd /k "python3.12 -m uvicorn src.api:app --reload --port 8000"

REM Wait a moment for FastAPI to start
timeout /t 3

REM Start Streamlit in a new window
echo Launching Streamlit Dashboard...
start "DataSentinel Dashboard" cmd /k "python -m streamlit run src/app.py"

echo.
echo Both services launched! Check the new terminal windows.
echo.
pause
