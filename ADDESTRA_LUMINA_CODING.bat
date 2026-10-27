@echo off
chcp 65001 >nul
title Lumina AI - Addestramento Intensivo di Coding (GPU)
color 0A

echo =====================================================================
echo         💻 ADDESTRAMENTO DI CODING ACCELERATO SU GPU (RTX 5060)
echo =====================================================================
echo.
echo Addestra Lumina su Python, funzioni, liste, dict, OOP, algoritmi ed errori.
echo Visualizzazione in tempo reale del codice generato da Lumina!
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% train_coding.py

echo.
pause
