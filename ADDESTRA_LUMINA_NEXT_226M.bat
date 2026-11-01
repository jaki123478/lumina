@echo off
setlocal
cd /d "%~dp0"
title Addestramento Lumina Next 226M - candidato separato
echo ================================================
echo LUMINA NEXT 226M - CICLO CANDIDATO
echo ================================================
if not exist ".venv\Scripts\python.exe" (
  echo ERRORE: ambiente Python non trovato.
  pause
  exit /b 1
)
if not exist "config\lumina_next_256m.json" (
  echo ERRORE: configurazione 226M non trovata.
  pause
  exit /b 1
)
set LUMINA_CHECKPOINT_DIR=checkpoints_lumina_226m_candidate
echo Checkpoint candidato: %LUMINA_CHECKPOINT_DIR%
echo Modello stabile: non viene modificato.
echo Avvio con output immediato...
".venv\Scripts\python.exe" -u train_lumina_next.py --config config\lumina_next_256m.json --steps 25 --seq-len 512 --batch-size 1 --grad-accum 1 --lr 1e-4
echo.
echo Processo terminato. Verifica il candidato prima della promozione.
pause
