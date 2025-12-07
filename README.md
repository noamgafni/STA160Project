# Cryptocurrency Price Range Forecasting Dashboard

Team 17 | STA 160 | UC Davis | Fall 2025

## Local Setup

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this folder to GitHub
2. Go to share.streamlit.io
3. Connect your repo, select branch, set main file to `app.py`
4. Deploy

## File Structure

```
frontend/
├── app.py              # Main entry point
├── requirements.txt    # Dependencies
├── data/
│   ├── arima_metrics.csv   # Model parameters
│   └── arima_summary.png   # Results visualization
└── pages/
    ├── overview.py     # Project results page
    ├── analysis.py     # Live market data page
    └── forecast.py     # Prediction tool page
```

## Data Source

Live market data fetched from Yahoo Finance API.
Model parameters loaded from pre-trained ARIMA results.
