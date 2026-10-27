@echo off
chcp 65001 >nul
title Lumina AI - Suite Benchmark e Certificazione
color 0F

echo =====================================================================
echo         🏆 AVVIO SUITE DI BENCHMARK E CERTIFICAZIONE LUMINA
echo =====================================================================
echo.
echo Misurazione Throughput (GPU vs CPU), Accuratezza Domini e Sicurezza Sentinel...
echo.

if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

%PYTHON_EXEC% valuta_benchmark.py

echo.
pause
