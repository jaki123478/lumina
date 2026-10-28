@echo off
title PARLA CON LUMINA
color 0A
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" lumina_voice_pack\lumina_voice_engine_lenovo.py
pause
