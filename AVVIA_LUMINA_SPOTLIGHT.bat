@echo off
chcp 65001 >nul
title Lumina AI - Spotlight Desktop Popup
color 0B

echo =====================================================================
echo         ⚡ AVVIO LUMINA SPOTLIGHT (BARRA DI RICERCA CYBERPUNK)
echo =====================================================================
echo.
echo Avvio del widget fluttuante stile Raycast...
echo Premi [ESC] in qualsiasi momento per chiuderlo.
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

start "" %PYTHON_EXEC% lumina_spotlight.py
