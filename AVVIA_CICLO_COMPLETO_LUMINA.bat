@echo off
setlocal
set "ROOT=C:\Users\Jaki1\Desktop\ai lumina"
cd /d "%ROOT%"
start "Lumina API" /min "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\serve.py" --port 8000
timeout /t 3 /nobreak >nul
start "Monitor Gemma-Lumina" "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\gemma_asks_lumina.py"
set LUMINA_EXTRA_DATASET=data\lumina_gemma_approved.jsonl
set LUMINA_INIT_CHECKPOINT=checkpoints_lumina_next
set LUMINA_CHECKPOINT_DIR=checkpoints_lumina_autorun
set LUMINA_EXPORT_DIR=exports_lumina_autorun
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\train_lumina_next.py" --steps 10 --seq-len 128 --batch-size 1 --grad-accum 1 --lr 0.000005
echo Ciclo sicuro completato. Il checkpoint stabile non e' stato modificato.
pause
