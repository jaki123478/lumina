@echo off
setlocal
set "ROOT=C:\Users\Jaki1\Desktop\ai lumina"
cd /d "%ROOT%"
if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\chat_gemma.py" %*
) else (
  python "%ROOT%\chat_gemma.py" %*
)
if errorlevel 1 pause
