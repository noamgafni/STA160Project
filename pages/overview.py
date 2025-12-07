import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

def load_metrics():
    paths = ["data/arima_metrics.csv", "arima_metrics.csv", "../data/arima_metrics.csv"]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

def render():
    st.title("Cryptocurrency Price Range Forecasting")
    
    st.markdown("""
    This project develops a time-series forecasting pipeline for cryptocurrency markets using OHLCV data. 
    Originally aimed at predicting tail-risk events, early analysis revealed that extreme market moves 
    are fundamentally difficult to predict with price and volume data alone. The project pivoted to 
    forecasting log-transformed daily price ranges with calibrated uncertainty intervals.
    """)
    
    st.markdown("---")
    
    # Project Evolution
    st.header("Project Evolution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Initial Approach: Tail-Risk Prediction")
        st.markdown("""
        The original goal was to predict extreme downside moves, defined as returns 
        below the 5th percentile (approximately -9.28% daily). We trained Logistic Regression, 
        Random Forest, and XGBoost classifiers using 35 engineered technical features.
        
        **Why it failed:** Tail events are driven by external shocks (regulatory news, 
        exchange hacks, influential tweets) that leave no advance trace in 
        historical price data. Models achieved high precision but critically low recall, 
        missing 94% of actual crash events.
        """)
    
    with col2:
        st.markdown("#### Revised Approach: Price Range Forecasting")
        st.markdown("""
        We pivoted to forecasting the daily price range (High - Low) at 1-day and 7-day 
        horizons using log transformation. This target is more predictable because volatility 
        exhibits clustering: high-volatility days tend to follow high-volatility days.
        
        **Key insight:** ARIMA using only historical price range data performed statistically 
        equivalently to ARIMAX with 18 exogenous features (p = 0.50). The historical pattern 
        of volatility itself contains most of the predictable signal.
        """)
    
    st.markdown("---")
    
    # Dataset Overview
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observations", "348,200")
    col2.metric("Cryptocurrencies", "18")
    col3.metric("Time Period", "2014-2025")
    col4.metric("Features Engineered", "35")
    
    st.markdown("""
    Data sourced from Kaggle's "Cryptocurrency Prices (Top 200+)" dataset covering the top 250 
    cryptocurrencies by market capitalization. After cleaning and filtering, analysis focused on 
    18 major non-stablecoin assets with sufficient historical data for walk-forward validation.
    """)
    
    coins = ["BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "HYPE",
             "BCH", "LINK", "LEO", "ZEC", "XLM", "XMR", "LTC", "HBAR", "AVAX"]
    st.markdown("**Coins Analyzed:** " + " | ".join(coins))
    
    st.markdown("---")
    
    # Tail-Risk Classification Results
    st.header("Phase 1: Tail-Risk Classification Results")
    
    st.markdown("""
    Three classification models were trained using walk-forward validation with a 70/15/15 
    train-validation-test split. Target: binary indicator of next-day returns below the 5th percentile.
    """)
    
    tail_risk_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC-AUC": [0.565, 0.760, 0.742],
        "Precision": [0.12, 0.84, 0.78],
        "Recall": [0.08, 0.06, 0.09],
        "F1-Score": [0.10, 0.12, 0.16]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(tail_risk_results, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("""
        **Critical Finding:**  
        Random Forest achieved AUC of 0.76 but recall of only 6%. 
        This means the model missed **94% of actual crash events**, 
        rendering it impractical for risk management.
        """)
    
    with st.expander("Why Tail-Risk Prediction Failed"):
        st.markdown("""
        1. **External Shocks:** Most crypto crashes are triggered by unpredictable events 
           (regulatory announcements, exchange hacks, stablecoin de-pegs, influential tweets).
        
        2. **Information Asymmetry:** Sophisticated market participants act on private information 
           before patterns become visible in price data.
        
        3. **Class Imbalance:** Only ~6% of days are tail-risk events.
        
        4. **Temporal Instability:** Drivers of tail events evolve over time.
        """)
    
    st.markdown("---")
    
    # ARIMA Results
    st.header("Phase 2: Price Range Forecasting Results")
    
    st.markdown("""
    ARIMA models were fit for each cryptocurrency using walk-forward validation 
    with 730-day training windows and 365-day test periods. Target: log-transformed daily 
    price range (High - Low). Models were refit every 60 days.
    """)
    
    metrics_df = load_metrics()
    
    if metrics_df is not None:
        metrics_1d = metrics_df[metrics_df['horizon'] == 1].copy()
        metrics_7d = metrics_df[metrics_df['horizon'] == 7].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median 1-Day MAPE", f"{metrics_1d['mape'].median():.1f}%")
        col2.metric("Avg 80% Coverage", f"{metrics_1d['coverage_80'].mean():.1f}%", "Target: 80%")
        col3.metric("Avg 95% Coverage", f"{metrics_1d['coverage_95'].mean():.1f}%", "Target: 95%")
        col4.metric("ARIMAX Improvement", "None", "p = 0.50")
        
        st.markdown("---")
        st.markdown("#### Per-Coin ARIMA Performance (1-Day Horizon)")
        
        display_df = metrics_1d[['ticker', 'order', 'mape', 'rmse', 'coverage_80', 'coverage_95', 'bias']].copy()
        display_df.columns = ['Coin', 'ARIMA Order', 'MAPE (%)', 'RMSE', '80% Cov', '95% Cov', 'Bias']
        display_df = display_df.sort_values('MAPE (%)')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.info("""
        **Note on MAPE values:** LINK-USD (217%) and AVAX-USD (339%) show inflated MAPE 
        due to division by small actual values near zero. This is a mathematical artifact. 
        Their coverage metrics (81-82% and 94-95%) confirm well-calibrated predictions.
        """)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mape_display = metrics_1d[metrics_1d['mape'] < 100].sort_values('mape')
            
            fig_mape = go.Figure(go.Bar(
                x=mape_display['ticker'],
                y=mape_display['mape'],
                marker_color=['#22c55e' if m < 15 else '#4da6ff' if m < 30 else '#f59e0b' 
                              for m in mape_display['mape']],
                text=[f"{m:.1f}%" for m in mape_display['mape']],
                textposition='outside'
            ))
            
            fig_mape.update_layout(
                title="1-Day MAPE by Coin (excluding outliers)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0c0d0'),
                height=400,
                xaxis=dict(showgrid=False, tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='#2a2a3a', title="MAPE (%)")
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with col2:
            fig_cov = go.Figure()
            
            fig_cov.add_trace(go.Scatter(
                x=metrics_1d['ticker'], y=metrics_1d['coverage_80'],
                mode='markers', name='80% Coverage',
                marker=dict(size=10, color='#4da6ff')
            ))
            
            fig_cov.add_trace(go.Scatter(
                x=metrics_1d['ticker'], y=metrics_1d['coverage_95'],
                mode='markers', name='95% Coverage',
                marker=dict(size=10, color='#22c55e')
            ))
            
            fig_cov.add_hline(y=80, line_dash="dash", line_color="#f59e0b", 
                            annotation_text="Target 80%", annotation_position="right")
            fig_cov.add_hline(y=95, line_dash="dash", line_color="#f59e0b", 
                            annotation_text="Target 95%", annotation_position="right")
            
            fig_cov.update_layout(
                title="1-Day Interval Coverage by Coin",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c0c0d0'),
                height=400,
                xaxis=dict(showgrid=False, tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='#2a2a3a', title="Coverage (%)", range=[70, 100]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_cov, use_container_width=True)
        
        # Show summary image if available
        img_paths = ["data/arima_summary.png", "arima_summary.png"]
        for img_path in img_paths:
            if os.path.exists(img_path):
                st.markdown("#### ARIMA Model Summary")
                st.image(img_path, use_container_width=True)
                break
    
    else:
        st.warning("Metrics file not found. Please ensure arima_metrics.csv is in the data/ folder.")
    
    st.markdown("---")
    
    # Statistical Comparison
    st.header("ARIMA vs ARIMAX: Statistical Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Paired t-Test Results
        
        | Statistic | Value |
        |-----------|-------|
        | Mean MAPE Difference | +0.14% (ARIMAX worse) |
        | t-statistic | 0.69 |
        | p-value | 0.50 |
        | Conclusion | Not statistically significant |
        """)
    
    with col2:
        st.markdown("""
        #### Interpretation
        
        This aligns with the **Efficient Market Hypothesis**: publicly available 
        technical indicators are already priced into the market.
        
        For volatility forecasting, the historical pattern of price ranges contains 
        most of the predictable signal. **Simplicity wins.**
        """)
    
    st.markdown("---")
    
    # Key Takeaways
    st.header("Key Takeaways")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Limitations Matter**
        
        OHLCV data lacks information needed to predict tail events. 
        Extreme moves are driven by external shocks not encoded in historical prices.
        """)
    
    with col2:
        st.markdown("""
        **Simplicity Wins**
        
        ARIMA matched ARIMAX performance statistically. 
        More features do not guarantee better predictions.
        """)
    
    with col3:
        st.markdown("""
        **Calibration is Key**
        
        Well-calibrated prediction intervals (81% actual vs 80% target) 
        are more valuable than marginally lower point errors.
        """)
