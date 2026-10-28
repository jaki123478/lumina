@echo off
cd /d "%~dp0"
start "Lumina UI" http://127.0.0.1:7020
python server.py
