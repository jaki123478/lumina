@echo off
setlocal
set "ROOT=C:\Users\Jaki1\Desktop\ai lumina"
cd /d "%ROOT%"
start "Lumina API" /min "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\serve.py" --port 8000
timeout /t 3 /nobreak >nul
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\gemma_asks_lumina.py"
echo.
echo Monitor terminato: nessuna modifica automatica ai pesi.
cmd /k
