@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv 2>nul || py -m venv .venv
)

call ".venv\Scripts\activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
python serve_dashboard.py

