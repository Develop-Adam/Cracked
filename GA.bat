@echo off
REM 1) Go to this batch file's folder
cd /d "%~dp0"

REM 2) Activate venv in this folder
call Scripts\activate.bat

REM 3) Run the script
python measure_image.py

REM 4) Keep window open
pause