@echo off
chcp 65001 >nul
title Lumina AI - Sincronizzazione Hugging Face
color 0D

echo =====================================================================
echo    🔄 SINCRONIZZAZIONE AUTOMATICA ASSET SU HUGGING FACE
echo =====================================================================
echo.
echo Aggiorna Modelli (safetensors + ONNX + GGUF), Demo Space e Dataset Parquet...
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% sync_all_to_huggingface.py

echo.
pause
