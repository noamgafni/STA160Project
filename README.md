# Cryptocurrency Volatility Forecasting

**[Live demo →](https://cryptoforecasting-g17.streamlit.app/)** *(hosted on Streamlit Community Cloud — the app may be asleep if inactive; click the link and give it ~30 seconds to spin back up)*

Team project (STA 160, UC Davis) building a time-series forecasting pipeline across 18 cryptocurrencies, covering 348,200+ observations from 2014–2025.

## Project Evolution

**Initial approach:** Predicting extreme downside price moves (tail-risk events, returns below the 5th percentile) using Logistic Regression, Random Forest, and XGBoost classifiers with 35 engineered technical features.

**Why it failed:** Tail events are driven by external shocks (regulatory news, exchange hacks, influential tweets) that leave no advance trace in historical price data. Random Forest achieved an AUC of 0.76 but only 6% recall, missing 94% of actual crash events, making it impractical for real risk management.

**Revised approach:** Pivoted to forecasting the daily price range (High − Low) using log-transformed ARIMA models, since volatility exhibits clustering (high-volatility days tend to follow high-volatility days).

## Key Results

- **14% median MAPE** on 1-day price range forecasts
- **81.4% average 80% coverage** and **95.0% average 95% coverage** — well-calibrated prediction intervals
- ARIMA performed statistically equivalently to ARIMAX with 18 exogenous features (paired t-test, p = 0.50), showing that historical price-range patterns alone contain most of the predictable signal

## My Contribution
Built the ARIMA-based time-series modeling pipeline and developed the interactive Streamlit dashboard used to explore forecasts and compare model performance across the 18 cryptocurrencies.

## Tools
Python, ARIMA, Streamlit, walk-forward validation

---
**Contact:** [noam.gafni@gmail.com](mailto:noam.gafni@gmail.com) | [LinkedIn](https://www.linkedin.com/in/noam-gafni-14341b255/)

---

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
