@echo off
chcp 65001 >nul
title Lumina AI - Code Sandbox & Autocorrezione
color 0E

echo =====================================================================
echo    🧪 LUMINA CODE SANDBOX & AUTOCORREZIONE AUTOMATICA (GPU)
echo =====================================================================
echo.
echo Scrittura, esecuzione e verifica live del codice Python.
echo Se viene rilevato un errore, Lumina si autocorregge da sola!
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% lumina_sandbox.py

echo.
pause
