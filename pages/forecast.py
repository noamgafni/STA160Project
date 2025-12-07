import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

COINS = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "XRP": "XRP-USD",
    "BNB": "BNB-USD", "Solana": "SOL-USD", "TRON": "TRX-USD",
    "Dogecoin": "DOGE-USD", "Cardano": "ADA-USD", "Hyperliquid": "HYPE-USD",
    "Bitcoin Cash": "BCH-USD", "Chainlink": "LINK-USD", "UNUS SED LEO": "LEO-USD",
    "Zcash": "ZEC-USD", "Stellar": "XLM-USD", "Monero": "XMR-USD",
    "Litecoin": "LTC-USD", "Hedera": "HBAR-USD", "Avalanche": "AVAX-USD"
}

def load_model_params():
    paths = ["data/arima_metrics.csv", "arima_metrics.csv", "../data/arima_metrics.csv"]
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            params = {}
            for _, row in df.iterrows():
                ticker = row['ticker']
                horizon = row['horizon']
                order_str = row['order'].strip("()").replace(" ", "")
                order = tuple(map(int, order_str.split(",")))
                
                if ticker not in params:
                    params[ticker] = {}
                
                params[ticker][horizon] = {
                    'order': order,
                    'mape': row['mape'],
                    'rmse': row['rmse'],
                    'coverage_80': row['coverage_80'],
                    'coverage_95': row['coverage_95']
                }
            return params, df
    return None, None

@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if data.empty:
            return None
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        return data
    except Exception:
        return None

def prepare_target(df):
    df = df.copy()
    df['Price_Range'] = df['High'] - df['Low']
    df['Log_Range'] = np.log(df['Price_Range'].replace(0, np.nan))
    df = df.dropna(subset=['Log_Range'])
    return df

def fit_and_forecast(series, order, horizon):
    try:
        train_data = series.tail(730)
        model = ARIMA(train_data, order=order)
        fitted = model.fit()
        
        forecast_result = fitted.get_forecast(steps=horizon)
        ci_80 = forecast_result.conf_int(alpha=0.20)
        ci_95 = forecast_result.conf_int(alpha=0.05)
        
        return {
            'point': forecast_result.predicted_mean,
            'lower_80': ci_80.iloc[:, 0],
            'upper_80': ci_80.iloc[:, 1],
            'lower_95': ci_95.iloc[:, 0],
            'upper_95': ci_95.iloc[:, 1],
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def render():
    st.title("Price Range Forecast Tool")
    
    st.markdown("""
    Generate predictions for the **daily price range (High - Low)** using trained ARIMA models.
    
    **Important:** This model predicts volatility (how much the price moves within a day), 
    NOT the price itself. For example, if Bitcoin is at $100,000 and the model predicts a 
    range of $1,500, it means the price is expected to fluctuate by about $1,500 during the day.
    """)
    
    st.markdown("---")
    
    model_params, metrics_df = load_model_params()
    
    if model_params is None:
        st.error("Could not load model parameters. Ensure arima_metrics.csv is in data/ folder.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_coin = st.selectbox("SELECT CRYPTOCURRENCY", list(COINS.keys()))
    with col2:
        horizon_choice = st.selectbox("FORECAST HORIZON", ["1-Day", "7-Day"])
    
    ticker = COINS[selected_coin]
    horizon = 1 if horizon_choice == "1-Day" else 7
    
    if ticker not in model_params:
        st.warning(f"No pre-trained parameters found for {ticker}.")
        return
    
    params = model_params[ticker].get(horizon, model_params[ticker].get(1))
    
    st.markdown("---")
    
    st.header("Model Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    order_str = f"({params['order'][0]}, {params['order'][1]}, {params['order'][2]})"
    
    col1.metric("ARIMA Order", order_str)
    mape_fmt = f"{params['mape']:.1f}%" if params['mape'] < 100 else f"{params['mape']:.0f}%"
    col2.metric("Historical MAPE", mape_fmt)
    col3.metric("80% Coverage", f"{params['coverage_80']:.1f}%")
    col4.metric("95% Coverage", f"{params['coverage_95']:.1f}%")
    
    if params['mape'] > 100:
        st.warning("High MAPE is due to log-scale division by small values. Coverage metrics confirm good calibration.")
    
    st.markdown("---")
    
    st.header("Live Forecast")
    
    if not ARIMA_AVAILABLE:
        st.error("statsmodels not installed. Run: pip install statsmodels")
        return
    
    with st.spinner("Fetching latest market data..."):
        df = fetch_data(ticker)
    
    if df is None or df.empty:
        st.error("Unable to fetch market data.")
        return
    
    df_prepared = prepare_target(df)
    series = df_prepared['Log_Range']
    
    with st.spinner(f"Fitting ARIMA{order_str} model..."):
        forecast = fit_and_forecast(series, params['order'], horizon)
    
    if not forecast['success']:
        st.error(f"Forecast failed: {forecast.get('error', 'Unknown error')}")
        return
    
    latest_date = df_prepared['Date'].iloc[-1] if 'Date' in df_prepared.columns else df_prepared.index[-1]
    latest_range = df_prepared['Price_Range'].iloc[-1]
    latest_price = df.iloc[-1]['Close']
    
    point_log = forecast['point'].iloc[-1]
    point_range = np.exp(point_log)
    lower_80 = np.exp(forecast['lower_80'].iloc[-1])
    upper_80 = np.exp(forecast['upper_80'].iloc[-1])
    lower_95 = np.exp(forecast['lower_95'].iloc[-1])
    upper_95 = np.exp(forecast['upper_95'].iloc[-1])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {horizon_choice} Price Range Forecast")
        
        fmt = "${:,.2f}" if point_range > 1 else "${:.6f}"
        
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Predicted Range (High-Low)** | **{fmt.format(point_range)}** |
        | 80% Confidence Interval | {fmt.format(lower_80)} - {fmt.format(upper_80)} |
        | 95% Confidence Interval | {fmt.format(lower_95)} - {fmt.format(upper_95)} |
        | Yesterday's Actual Range | {fmt.format(latest_range)} |
        | Current Price | ${latest_price:,.2f} |
        | Forecast Date | {(pd.to_datetime(latest_date) + timedelta(days=horizon)).strftime('%Y-%m-%d')} |
        """)
    
    with col2:
        st.markdown("#### Interpretation")
        
        change_pct = ((point_range - latest_range) / latest_range) * 100
        direction = "higher" if change_pct > 0 else "lower"
        range_pct = (point_range / latest_price) * 100
        
        st.markdown(f"""
        The model predicts {selected_coin}'s price will move within a range of 
        **{fmt.format(point_range)}** over the next {horizon} day(s).
        
        This is **{abs(change_pct):.1f}% {direction}** than yesterday's range, 
        representing about **{range_pct:.2f}%** of the current price.
        
        **What this means:** If the current price is ${latest_price:,.0f}, expect 
        the price to fluctuate roughly ±${point_range/2:,.0f} from the daily average.
        """)
    
    st.markdown("---")
    
    st.header("Forecast Visualization")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Log Price Range with Forecast", "Price Range in USD"))
    
    x_data = df_prepared['Date'] if 'Date' in df_prepared.columns else df_prepared.index
    
    fig.add_trace(go.Scatter(x=x_data[-90:], y=series.iloc[-90:], mode='lines',
                             name='Actual Log Range', line=dict(color='#4da6ff', width=1.5)), row=1, col=1)
    
    last_date = pd.to_datetime(x_data.iloc[-1])
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
    
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(forecast['upper_95']) + list(forecast['lower_95'][::-1]),
        fill='toself', fillcolor='rgba(34, 197, 94, 0.15)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(forecast['upper_80']) + list(forecast['lower_80'][::-1]),
        fill='toself', fillcolor='rgba(34, 197, 94, 0.3)',
        line=dict(color='rgba(0,0,0,0)'), name='80% CI'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast['point'], mode='lines+markers',
                             name='Forecast', line=dict(color='#22c55e', width=2),
                             marker=dict(size=8)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=x_data[-90:], y=df_prepared['Price_Range'].iloc[-90:], mode='lines',
                             name='Price Range (USD)', line=dict(color='#f59e0b', width=1.5)), row=2, col=1)
    
    fig.update_layout(
        plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f', font=dict(color='#c0c0d0'),
        height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor='#1a1a24', title="Log(Range)"),
        yaxis2=dict(showgrid=True, gridcolor='#1a1a24', title="USD")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.header("All Coins Comparison")
    
    if metrics_df is not None:
        metrics_1d = metrics_df[metrics_df['horizon'] == 1].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            mape_viz = metrics_1d[metrics_1d['mape'] < 100].sort_values('mape')
            
            fig_mape = go.Figure(go.Bar(
                x=mape_viz['ticker'], y=mape_viz['mape'],
                marker_color=['#22c55e' if m < 15 else '#4da6ff' if m < 30 else '#f59e0b' for m in mape_viz['mape']],
                text=[f"{m:.1f}%" for m in mape_viz['mape']], textposition='outside'
            ))
            fig_mape.update_layout(
                title="1-Day MAPE (excluding outliers)",
                plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f', font=dict(color='#c0c0d0'),
                height=350, xaxis=dict(tickangle=45), yaxis=dict(showgrid=True, gridcolor='#1a1a24')
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with col2:
            fig_cov = go.Figure()
            fig_cov.add_trace(go.Bar(name='80% Cov', x=metrics_1d['ticker'], y=metrics_1d['coverage_80'], marker_color='#4da6ff'))
            fig_cov.add_trace(go.Bar(name='95% Cov', x=metrics_1d['ticker'], y=metrics_1d['coverage_95'], marker_color='#22c55e'))
            fig_cov.add_hline(y=80, line_dash="dash", line_color="#f59e0b")
            fig_cov.add_hline(y=95, line_dash="dash", line_color="#f59e0b")
            fig_cov.update_layout(
                title="Interval Coverage",
                plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f', font=dict(color='#c0c0d0'),
                height=350, barmode='group', xaxis=dict(tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='#1a1a24', range=[70, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_cov, use_container_width=True)
        
        with st.expander("View All Model Parameters"):
            display_df = metrics_1d[['ticker', 'order', 'mape', 'rmse', 'coverage_80', 'coverage_95']].copy()
            display_df.columns = ['Coin', 'Order', 'MAPE (%)', 'RMSE', '80% Cov', '95% Cov']
            st.dataframe(display_df.sort_values('MAPE (%)'), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    with st.expander("Methodology"):
        st.markdown("""
        **Training:** 730-day rolling window, refit every 60 days  
        **Order Selection:** BIC criterion via auto_arima  
        **Target:** Log-transformed daily price range  
        **Validation:** Walk-forward with 365-day test periods
        """)
    
    with st.expander("Disclaimer"):
        st.markdown("""
        This tool is for educational purposes only. Cryptocurrency markets are highly volatile. 
        Do not use these forecasts as the sole basis for investment decisions.
        """)
