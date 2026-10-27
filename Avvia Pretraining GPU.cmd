@echo off
title Lumina AI - Addestramento GPU (14.4 Milioni di parametri)
cd /d "%~dp0"
echo =======================================================
echo Avvio Pretraining di Lumina (14.4M parametri) su GPU...
echo Corpus: 50.7 Milioni di token in italiano
echo Scheda Video: NVIDIA RTX 5060
echo =======================================================
.\.venv\Scripts\python.exe pretrain.py --preset xlarge --steps 30000 --batch-size 32 --eval-every 500
pause
