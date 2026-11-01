@echo off
setlocal
fltmc >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
echo Creo la regola firewall per Lumina solo sulle reti private...
powershell -NoProfile -Command "New-NetFirewallRule -DisplayName 'Lumina locale (reti private)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000,8765 -Profile Private -Description 'Consente il server locale Lumina solo sulle reti private'"
echo.
echo Operazione completata. Premi un tasto per chiudere.
pause >nul
