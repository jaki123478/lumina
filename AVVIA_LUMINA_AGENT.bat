@echo off
chcp 65001 >nul
title Lumina AI - Action Agent (Automazioni Windows)
color 0B

echo =====================================================================
echo         🛠️  AVVIO LUMINA ACTION AGENT (CONTROLLO WINDOWS)
echo =====================================================================
echo.
echo Modello: Lumina Edge-AI con Function Calling
echo Protezione: Lumina Sentinel
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% lumina_agent.py

echo.
pause
