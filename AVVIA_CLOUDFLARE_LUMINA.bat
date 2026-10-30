@echo off
setlocal
cd /d "%~dp0"
echo Avvio del tunnel Cloudflare verso Lumina (porta 8000)...
where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared non e' installato. Installalo dal sito ufficiale Cloudflare.
  pause
  exit /b 1
)
cloudflared tunnel --url http://127.0.0.1:8000
