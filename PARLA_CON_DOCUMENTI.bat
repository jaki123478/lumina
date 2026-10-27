@echo off
title Lumina RAG - Parla con i tuoi Documenti
chcp 65001 >nul
cd /d "%~dp0"
echo =====================================================================
echo  📄 LUMINA RAG - CHAT CON FILE PDF, TXT E DOCUMENTI PERSONALI
echo =====================================================================
echo.
.\.venv\Scripts\python lumina_rag.py
pause
