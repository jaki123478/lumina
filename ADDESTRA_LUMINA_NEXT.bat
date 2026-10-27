@echo off
chcp 65001 >nul
title Lumina-Next 135M - Addestramento Long Context (8k-32k)
color 0B

echo =====================================================================
echo    🚀 AVVIO ADDESTRAMENTO LUMINA-NEXT 135M (LONG CONTEXT & CODING)
echo =====================================================================
echo.
echo Architettura: RoPE + YaRN, GQA 3:1, SwiGLU, RMSNorm
echo Tokenizer: BPE 32.000 Vocab (Italiano + Coding)
echo GPU: Accelerazione bfloat16 + Gradient Checkpointing
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% train_lumina_next.py --seq-len 2048 --steps 100 --batch-size 2

echo.
pause
