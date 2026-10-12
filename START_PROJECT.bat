@echo off
cd /d "%~dp0"
echo =======================================================
echo   Starting AI HydroGuard - CNN-LSTM Flood Warning App  
echo =======================================================
python -m pip install -r requirements.txt
start http://localhost:5000
python server.py
pause
