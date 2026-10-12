# AI Flood Prediction using CNN + LSTM — Second Review Demo

## Fastest way to show output
1. Install Python 3.10+.
2. Open this folder in VS Code.
3. Open Terminal.
4. Run:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`
6. Browser opens the project UI. Go to **Prediction** and click **Predict Flood Risk**.

The ZIP already contains a trained demo model, so you do **not** need to train before the review demo.

## If you want to retrain
Run:
`python train.py`

## Output screens
- Home Page
- Environmental Parameters
- Safe / Unsafe prediction
- Risk probability
- Flood Risk Heatmap

## Important academic note
The bundled model is trained on synthetic/demo data so the review pipeline can run immediately. It demonstrates the CNN + LSTM architecture described in the second-review PPT. For the final project, replace the synthetic generator with validated satellite imagery and rainfall time-series data.
