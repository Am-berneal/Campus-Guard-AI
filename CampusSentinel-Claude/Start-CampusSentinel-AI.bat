@echo off
setlocal

set "AI_DIR=C:\Users\hp\Projects\CampusSentinel-Claude\ai-service"

if not exist "%AI_DIR%\api\main.py" (
    echo ERROR: Could not find the CampusSentinel AI service:
    echo %AI_DIR%
    pause
    exit /b 1
)

if not exist "%AI_DIR%\.venv\Scripts\activate.bat" (
    echo ERROR: Python virtual environment not found:
    echo %AI_DIR%\.venv\Scripts\activate.bat
    pause
    exit /b 1
)

cd /d "%AI_DIR%"
call .venv\Scripts\activate.bat

echo.
echo Starting CampusSentinel AI Service...
echo API: http://127.0.0.1:8000
echo API documentation: http://127.0.0.1:8000/docs
echo Keep this window open while using CampusSentinel.
echo.

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

echo.
echo AI service has stopped.
pause
