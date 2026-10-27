@echo off
chcp 65001 > nul
title ASCOLTA LA VOCE DI LUMINA SU PC LENOVO
color 0E
powershell -ExecutionPolicy Bypass -File "%~dp0play_audio_lenovo.ps1"
pause
