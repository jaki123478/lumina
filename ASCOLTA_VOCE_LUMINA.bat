@echo off
cd /d "%~dp0"
title VOCE LOCALE DI LUMINA
".venv\Scripts\python.exe" lumina_voice_neural.py "Ciao, sono Lumina. La mia voce italiana viene generata sul tuo computer. Come posso aiutarti?"
pause
