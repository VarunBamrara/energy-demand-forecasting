# Energy Demand Forecasting Model

PySpark time-series forecasting model on national energy datasets using Random Forest
Regression, reaching R² = 0.989 and RMSE = 1.01.

- `code/` — training notebook + dataset
- `app/streamlit_app.py` — Streamlit forecasting dashboard with pre-trained model
- `REPORT.pdf` — full project report

## Run locally
```
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```
