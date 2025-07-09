@echo off
cd /d %~dp0
call venv6\Scripts\activate.bat
python main.py
pause
