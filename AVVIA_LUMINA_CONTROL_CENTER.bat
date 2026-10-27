@echo off
chcp 65001 >nul
title Lumina AI - Sovereign Control Center
color 09

echo =====================================================================
echo       ✨ AVVIO LUMINA SOVEREIGN CONTROL CENTER (GUI DESKTOP)
echo =====================================================================
echo.
echo Avvio dell'interfaccia grafica di controllo in dark mode...
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

start "" %PYTHON_EXEC% lumina_control_center.py
