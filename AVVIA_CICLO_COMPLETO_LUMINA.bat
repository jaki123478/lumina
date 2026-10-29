@echo off
setlocal
set "ROOT=C:\Users\Jaki1\Desktop\ai lumina"
cd /d "%ROOT%"
set "LUMINA_SERVER_URL=http://127.0.0.1:8000"
start "Lumina API" /min "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\serve.py" --port 8000
timeout /t 5 /nobreak >nul
set LUMINA_INIT_CHECKPOINT=checkpoints_lumina_next
set LUMINA_CHECKPOINT_DIR=checkpoints_lumina_autorun
set LUMINA_EXPORT_DIR=exports_lumina_autorun
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\lumina_gemma_teacher_loop.py" --cycles 1 --train-steps 1
echo Ciclo sicuro completato. Il checkpoint stabile non e' stato modificato.
echo Il launcher termina senza richiedere input. I risultati restano nei checkpoint separati.
exit /b 0
