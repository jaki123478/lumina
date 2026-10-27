@echo off
title Qwen Maestro - Chat Locale GPU
color 0A
cd /d "%~dp0"
echo ============================================================
echo  Avvio di Qwen Maestro Locale (NVIDIA RTX 5060)
echo ============================================================
echo.
.\.venv\Scripts\python.exe chat_teacher.py
pause
