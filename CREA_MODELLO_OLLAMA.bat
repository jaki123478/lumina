@echo off
title Creazione Modello Lumina in Ollama
chcp 65001 >nul
cd /d "%~dp0"
echo =====================================================================
echo  🦙 REGISTRAZIONE LUMINA IN OLLAMA
echo =====================================================================
echo Verifica installazione Ollama...
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [!] Ollama non sembra installato nel PATH del sistema.
    echo Puoi scaricarlo gratuitamente da: https://ollama.ai
    echo.
    pause
    exit /b
)

echo Creazione modello 'lumina' (2.75M) da Modelfile...
ollama create lumina -f Modelfile

if exist Modelfile-14m (
    echo.
    echo Creazione modello 'lumina-14m' da Modelfile-14m...
    ollama create lumina-14m -f Modelfile-14m
)

echo.
echo =====================================================================
echo  ✅ Modelli Lumina registrati con successo in Ollama!
echo  Per avviare la chat in qualsiasi momento digita:
echo     ollama run lumina
echo     ollama run lumina-14m
echo =====================================================================
echo.
pause
