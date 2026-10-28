@echo off
cd /d "%~dp0"
chcp 65001 >nul
title Lumina AI - Assistente Vocale Italiano
color 0A

echo =====================================================================
echo          🎙️  AVVIO ASSISTENTE VOCALE LUMINA AI (OFFLINE)
echo =====================================================================
echo.
echo Microfono: Audio Input (SoundDevice)
echo Voce: Piper Paola italiana, locale
echo Trascrizione: richiede Vosk e un modello italiano gia presenti sul PC
echo Protezione: Lumina Sentinel Guardrails
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% lumina_voice.py

echo.
pause
