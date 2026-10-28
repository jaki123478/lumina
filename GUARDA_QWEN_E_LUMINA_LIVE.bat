@echo off
title ADDESTRAMENTO LIVE QWEN E LUMINA (GPU RTX 5060)
chcp 65001 >nul
color 0B
cd /d "C:\Users\Jaki1\Desktop\ai lumina"
echo =====================================================================
echo   AVVIO VISUALIZZATORE LIVE LUMINA (UTF-8 NATIVO SENZA BUG)
echo =====================================================================
".venv\Scripts\python.exe" view_live_training.py
pause