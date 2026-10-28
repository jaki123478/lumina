@echo off
cd /d "%~dp0.."
start "Lumina API" /min cmd /c "if not exist .venv\Scripts\python.exe exit /b 1 & .venv\Scripts\python.exe serve.py"
timeout /t 2 /nobreak >nul
start "Lumina Desktop" .venv\Scripts\python.exe lumina_desktop\lumina_app.py
