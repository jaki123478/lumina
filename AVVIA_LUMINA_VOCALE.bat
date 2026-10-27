@echo off
chcp 65001 >nul
title Lumina AI - Assistente Vocale Italiano
color 0A

echo =====================================================================
echo          🎙️  AVVIO ASSISTENTE VOCALE LUMINA AI (OFFLINE)
echo =====================================================================
echo.
echo Microfono: Audio Input (SoundDevice)
echo Voce: Sintesi Italiana Naturale (Microsoft Elsa it-IT)
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
