@echo off
chcp 65001 >nul
title Lumina AI - WebAssembly Browser Runner
echo =====================================================================
echo    ⚡ APERTURA LUMINA AI IN-BROWSER (WEBASEMBLY OFFLINE)
echo =====================================================================
echo.
echo Apertura di lumina_in_browser.html nel tuo browser predefinito...
echo.

start "" "%~dp0lumina_in_browser.html"
