import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

COINS = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "XRP": "XRP-USD",
    "BNB": "BNB-USD", "Solana": "SOL-USD", "TRON": "TRX-USD",
    "Dogecoin": "DOGE-USD", "Cardano": "ADA-USD", "Hyperliquid": "HYPE-USD",
    "Bitcoin Cash": "BCH-USD", "Chainlink": "LINK-USD", "UNUS SED LEO": "LEO-USD",
    "Zcash": "ZEC-USD", "Stellar": "XLM-USD", "Monero": "XMR-USD",
    "Litecoin": "LTC-USD", "Hedera": "HBAR-USD", "Avalanche": "AVAX-USD"
}

@st.cache_data(ttl=300)
def fetch_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        return data
    except Exception:
        return None

def engineer_features(df):
    data = df.copy()
    data.columns = [str(col) for col in data.columns]
    
    # Momentum
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    for lag in [1, 2, 3, 5, 7]:
        data[f'Return_Lag_{lag}d'] = data['Log_Return'].shift(lag)
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data['RSI_14'] = 100 - (100 / (1 + gain / loss))
    
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = ema_12 - ema_26
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    
    data['ROC_10'] = data['Close'].pct_change(10) * 100
    data['ROC_20'] = data['Close'].pct_change(20) * 100
    
    # Volatility
    for window in [5, 10, 20, 30]:
        data[f'Volatility_{window}d'] = data['Log_Return'].rolling(window).std()
    
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()
    data['ATR_21'] = tr.rolling(21).mean()
    
    data['BB_Middle'] = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * bb_std
    data['BB_Lower'] = data['BB_Middle'] - 2 * bb_std
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    
    # Trend
    for window in [5, 20, 50, 200]:
        data[f'SMA_{window}'] = data['Close'].rolling(window).mean()
    
    data['Price_Dev_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
    data['Price_Dev_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Volume
    for window in [5, 10, 20]:
        data[f'Vol_Ratio_{window}d'] = data['Volume'] / data['Volume'].rolling(window).mean()
    data['Vol_Pct_Change'] = data['Volume'].pct_change()
    
    # Target variable
    data['Price_Range'] = data['High'] - data['Low']
    data['Log_Range'] = np.log(data['Price_Range'].replace(0, np.nan))
    data['Price_Range_Pct'] = (data['Price_Range'] / data['Close']) * 100
    
    return data

def render():
    st.title("Market Analysis")
    
    st.markdown("""
    Real-time market data and technical feature analysis. Select a cryptocurrency to view 
    price action, volume, and the 35 engineered features used in our models.
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_coin = st.selectbox("SELECT CRYPTOCURRENCY", list(COINS.keys()))
    with col2:
        period = st.selectbox("TIME PERIOD", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    with col3:
        interval = st.selectbox("INTERVAL", ["1d", "1wk"])
    
    ticker = COINS[selected_coin]
    
    with st.spinner("Fetching live market data..."):
        df = fetch_data(ticker, period, interval)
    
    if df is None or df.empty:
        st.error("Unable to fetch data. Please try again later.")
        return
    
    df_features = engineer_features(df)
    
    latest = df_features.iloc[-1]
    prev = df_features.iloc[-2] if len(df_features) > 1 else latest
    
    price_change_pct = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    price_fmt = f"${latest['Close']:,.2f}" if latest['Close'] > 1 else f"${latest['Close']:.6f}"
    range_fmt = f"${latest['Price_Range']:,.2f}" if latest['Price_Range'] > 1 else f"${latest['Price_Range']:.6f}"
    
    col1.metric("Current Price", price_fmt, f"{price_change_pct:+.2f}%")
    col2.metric("Daily Range (High-Low)", range_fmt, f"{latest['Price_Range_Pct']:.2f}% of price")
    col3.metric("Volume", f"{latest['Volume']:,.0f}")
    
    rsi = latest['RSI_14']
    if not pd.isna(rsi):
        status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        col4.metric("RSI (14)", f"{rsi:.1f}", status)
    
    st.markdown("---")
    
    # Candlestick chart
    st.header("Price Action")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.7, 0.3], subplot_titles=("", "Volume"))
    
    x_data = df_features['Date'] if 'Date' in df_features.columns else df_features.index
    
    fig.add_trace(go.Candlestick(
        x=x_data, open=df_features['Open'], high=df_features['High'],
        low=df_features['Low'], close=df_features['Close'], name="OHLC",
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444'
    ), row=1, col=1)
    
    if 'BB_Upper' in df_features.columns:
        fig.add_trace(go.Scatter(x=x_data, y=df_features['BB_Upper'], mode='lines',
                                name='BB Upper', line=dict(color='#6b7280', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_data, y=df_features['BB_Lower'], mode='lines',
                                name='BB Lower', line=dict(color='#6b7280', width=1, dash='dot'),
                                fill='tonexty', fillcolor='rgba(77, 166, 255, 0.1)'), row=1, col=1)
    
    if 'SMA_20' in df_features.columns:
        fig.add_trace(go.Scatter(x=x_data, y=df_features['SMA_20'], mode='lines',
                                name='SMA 20', line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
    
    colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(df_features['Close'], df_features['Open'])]
    fig.add_trace(go.Bar(x=x_data, y=df_features['Volume'], marker_color=colors, showlegend=False), row=2, col=1)
    
    fig.update_layout(
        plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f', font=dict(color='#c0c0d0'),
        height=550, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor='#1a1a24', title="Price (USD)"),
        yaxis2=dict(showgrid=True, gridcolor='#1a1a24', title="Volume")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Price Range Analysis
    st.header("Price Range Analysis (Target Variable)")
    
    st.markdown("""
    The daily price range (High - Low) is our target variable. We apply a log transformation 
    to stabilize variance. This is what the ARIMA model predicts.
    """)
    
    fig_range = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                              row_heights=[0.6, 0.4],
                              subplot_titles=("Daily Price Range (USD)", "Log Price Range (Model Target)"))
    
    df_features['Range_MA20'] = df_features['Price_Range'].rolling(20).mean()
    
    fig_range.add_trace(go.Scatter(x=x_data, y=df_features['Price_Range'], mode='lines',
                                   name='Price Range', line=dict(color='#4da6ff', width=1),
                                   fill='tozeroy', fillcolor='rgba(77, 166, 255, 0.2)'), row=1, col=1)
    fig_range.add_trace(go.Scatter(x=x_data, y=df_features['Range_MA20'], mode='lines',
                                   name='20-day MA', line=dict(color='#f59e0b', width=2)), row=1, col=1)
    fig_range.add_trace(go.Scatter(x=x_data, y=df_features['Log_Range'], mode='lines',
                                   name='Log Range', line=dict(color='#22c55e', width=1.5)), row=2, col=1)
    
    fig_range.update_layout(
        plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f', font=dict(color='#c0c0d0'),
        height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor='#1a1a24'),
        yaxis2=dict(showgrid=True, gridcolor='#1a1a24', title="Log(Range)")
    )
    st.plotly_chart(fig_range, use_container_width=True)
    
    st.markdown("---")
    
    # Technical Features
    st.header("Technical Features (35 Total)")
    
    feature_categories = {
        "Momentum": ["Log_Return", "RSI_14", "MACD_Line", "MACD_Signal", "MACD_Histogram", 
                    "ROC_10", "ROC_20", "Return_Lag_1d", "Return_Lag_2d", "Return_Lag_3d"],
        "Volatility": ["Volatility_5d", "Volatility_10d", "Volatility_20d", "Volatility_30d",
                      "ATR_14", "ATR_21", "BB_Width"],
        "Trend": ["SMA_5", "SMA_20", "SMA_50", "SMA_200", "Price_Dev_SMA20", 
                 "Price_Dev_SMA50", "BB_Position", "BB_Upper", "BB_Lower"],
        "Volume": ["Vol_Ratio_5d", "Vol_Ratio_10d", "Vol_Ratio_20d", "Vol_Pct_Change"]
    }
    
    tab1, tab2 = st.tabs(["Feature Charts", "Data Table"])
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            category = st.selectbox("CATEGORY", list(feature_categories.keys()))
            available = [f for f in feature_categories[category] if f in df_features.columns]
            selected = st.multiselect("FEATURES", available, default=available[:2] if len(available) >= 2 else available)
        
        with col2:
            if selected:
                fig_feat = go.Figure()
                colors = ['#4da6ff', '#22c55e', '#f59e0b', '#a855f7', '#ef4444']
                for i, feat in enumerate(selected):
                    fig_feat.add_trace(go.Scatter(x=x_data, y=df_features[feat], mode='lines',
                                                  name=feat, line=dict(color=colors[i % len(colors)], width=1.5)))
                fig_feat.update_layout(
                    plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f', font=dict(color='#c0c0d0'),
                    height=400, yaxis=dict(showgrid=True, gridcolor='#1a1a24'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info("Select one or more features to visualize.")
    
    with tab2:
        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price_Range', 'Log_Range']
        display_cols = [c for c in display_cols if c in df_features.columns]
        st.dataframe(df_features[display_cols].tail(50).round(4), use_container_width=True, height=400)
        
        csv = df_features.to_csv(index=False)
        st.download_button("Download Full Dataset", csv, f"{ticker}_features.csv", "text/csv")
