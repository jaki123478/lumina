@echo off
title Lumina Chat Protetta (Lumina Sentinel)
chcp 65001 >nul
cd /d "%~dp0"
echo =====================================================================
echo  🛡️ AVVIO LUMINA CHAT PROTETTA (LUMINA SENTINEL + ANTI-JAILBREAK)
echo =====================================================================
echo.
.\.venv\Scripts\python chat_sentinel.py
pause
