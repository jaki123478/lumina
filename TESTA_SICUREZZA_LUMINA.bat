@echo off
title Test Suite Sicurezza Lumina Sentinel
chcp 65001 >nul
cd /d "%~dp0"
echo =====================================================================
echo  🛡️ LUMINA - COLLAUDO SISTEMA DI SICUREZZA LUMINA SENTINEL
echo =====================================================================
echo Avvio in corso...
.\.venv\Scripts\python tests/test_guardrails.py
echo.
pause
