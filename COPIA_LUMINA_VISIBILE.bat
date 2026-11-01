@echo off
title COPIA LUMINA - percentuale visibile
echo Copia completa Lumina verso D:\ai lumina
echo I file gia presenti vengono riutilizzati.
robocopy "C:\Users\Jaki1\Desktop\ai lumina" "D:\ai lumina" /E /COPY:DAT /DCOPY:DAT /Z /J /MT:32 /R:1 /W:1 /ETA
echo.
echo Copia terminata. Codice Robocopy: %ERRORLEVEL%
pause
