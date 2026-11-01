@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
title Lumina locale
echo Avvio Lumina locale: chat, documenti e voce.
if exist ".venv\Scripts\python.exe" (
  start "Lumina API locale" /min cmd /c ".venv\Scripts\python.exe serve.py"
  timeout /t 2 /nobreak >nul
  if exist "%ROOT%dist_v12\LuminaDesktop\LuminaDesktop.exe" (
    start "Lumina Desktop" "%ROOT%dist_v12\LuminaDesktop\LuminaDesktop.exe"
  ) else (
    start "Lumina Desktop" /D "%ROOT%" cmd /c ".venv\Scripts\python.exe lumina_desktop\lumina_app.py"
  )
) else (
  echo Ambiente locale non trovato: avvio Python di sistema.
  start "Lumina Desktop" /D "%ROOT%" cmd /c "python lumina_desktop\lumina_app.py"
)
echo Lumina e' stata avviata sul PC. Questa finestra puo' restare aperta.
endlocal
