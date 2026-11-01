@echo off
setlocal
set "PROJECT=%~dp0"
if exist "%~dp0requirements.txt" goto project_ready
if exist "%~dp0ai lumina\requirements.txt" set "PROJECT=%~dp0ai lumina"
:project_ready
cd /d "%PROJECT%"
title Lumina - PC fisso
echo ================================================
echo          AVVIO LUMINA - PC FISSO
echo ================================================
echo Cartella progetto: %PROJECT%

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

echo Server locale avviato su http://127.0.0.1:8000
echo Nessun browser o tunnel viene avviato.
echo Per il test locale lascia questa finestra aperta.
pause
exit /b 0

:error
echo Installazione o avvio non riuscito.
pause
exit /b 1
