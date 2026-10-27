@echo off
chcp 65001 >nul
title Lumina AI - Configurazione VS Code Copilot (Continue)
color 0B

echo =====================================================================
echo       🚀 CONFIGURAZIONE RAPIDA: LUMINA IN VS CODE COPILOT
echo =====================================================================
echo.
echo Collega l'estensione 'Continue' all'API locale protetta di Lumina.
echo Endpoint: http://localhost:8000/v1
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% setup_vscode_copilot.py

echo.
pause
