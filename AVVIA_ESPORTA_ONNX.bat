@echo off
chcp 65001 >nul
title Lumina AI - Esportatore ONNX
color 0E

echo =====================================================================
echo          ⚡ ESPORTAZIONE LUMINA IN FORMATO ONNX (CPU / WASM)
echo =====================================================================
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% export_onnx.py

echo.
pause
