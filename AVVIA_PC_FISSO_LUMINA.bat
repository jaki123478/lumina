@echo off
setlocal
cd /d "%~dp0"
title Lumina - PC fisso

where python >nul 2>nul
if errorlevel 1 (
  echo Python non trovato. Installare Python 3.11 o superiore e riprovare.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creo l'ambiente locale di Lumina...
  python -m venv .venv
  if errorlevel 1 goto :error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

echo Avvio il server locale di Lumina...
start "Lumina Server" /min cmd /c "cd /d "%~dp0" && .venv\Scripts\python.exe serve.py --port 8000"
timeout /t 5 /nobreak >nul

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo Server avviato su http://127.0.0.1:8000
  echo cloudflared non trovato: installarlo per ottenere un link pubblico.
  start http://127.0.0.1:8000
  pause
  exit /b 0
)

echo Avvio il link HTTPS pubblico...
cloudflared tunnel --url http://127.0.0.1:8000
exit /b 0

:error
echo Installazione o avvio non riuscito.
pause
exit /b 1
