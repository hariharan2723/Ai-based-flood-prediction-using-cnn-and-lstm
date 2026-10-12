@echo off
cd /d "%~dp0"
echo =======================================================
echo   Starting AI HydroGuard - CNN-LSTM Flood Warning App  
echo =======================================================
echo Installing requirements...
python -m pip install -r requirements.txt
echo.
echo Checking trained model...
if not exist "models\flood_cnn_lstm.pt" (
    python train.py
)
echo.
echo Starting Web Server...
start http://localhost:5000
python server.py
pause
