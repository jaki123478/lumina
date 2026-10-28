@echo off
chcp 65001 >nul
cd /d "C:\Users\Jaki1\Desktop\ai lumina"
title Lumina - ricerche online con voce locale
".venv\Scripts\python.exe" lumina_ricerca_vocale.py
pause
