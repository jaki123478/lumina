@echo off
chcp 65001 > nul
title LUMINA VOICE INTERACTIVE - PC LENOVO RTX 5060
color 0A
echo =====================================================================
echo       LUMINA VOICE INTERACTIVE - CONSOLE VOCALE PER LENOVO
echo       Pacchetto consegnato dal PC AMD al PC Principale Lenovo
echo =====================================================================
echo.
"%~dp0..\.venv\Scripts\python.exe" "%~dp0lumina_voice_engine_lenovo.py"
if %errorlevel% neq 0 (
    echo.
    echo [!] Se python non e nel PATH globale, avvia con il python dell'ambiente virtuale:
    echo     & "C:\lumina\venv\Scripts\python.exe" "%~dp0lumina_voice_engine_lenovo.py"
)
pause
