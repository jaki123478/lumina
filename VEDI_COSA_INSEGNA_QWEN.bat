@echo off
chcp 65001 >nul
title Lumina & Qwen - Live Teacher Stream
color 0F
cd /d "%~dp0"

echo =====================================================================
echo    🧠 FINESTRA IN TEMPO REALE: COSA QWEN INSEGNA A LUMINA
echo =====================================================================
echo.
echo Avvio del visualizzatore streaming...
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% stream_qwen_live.py
pause
