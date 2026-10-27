@echo off
title Auto-Distillazione & Addestramento Lumina in Tempo Reale
color 0B
cd /d "%~dp0"
echo ============================================================
echo  Monitor in Tempo Reale: Qwen genera e allena Lumina
echo ============================================================
echo.
.\.venv\Scripts\python.exe auto_distill_train.py
pause
