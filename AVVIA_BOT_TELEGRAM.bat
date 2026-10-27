@echo off
chcp 65001 >nul
title Lumina AI - Telegram Bot Sovrano
color 0B

echo =====================================================================
echo           🤖 AVVIO LUMINA TELEGRAM BOT (LOCALE + PROTETTO)
echo =====================================================================
echo.
echo Connessione al bot Telegram...
echo Supporto: Chat testo, sintesi vocale e protezione Lumina Sentinel.
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% lumina_telegram.py

echo.
pause
