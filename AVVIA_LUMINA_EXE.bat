@echo off
set "ROOT=%~dp0"
cd /d "%ROOT%"
start "Lumina API locale" /min cmd /c "if exist .venv\Scripts\python.exe .venv\Scripts\python.exe serve.py"
timeout /t 3 /nobreak >nul
start "Lumina Desktop" "%ROOT%dist_v12\LuminaDesktop\LuminaDesktop.exe"
