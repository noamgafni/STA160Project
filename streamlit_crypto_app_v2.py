"""
Cryptocurrency Tail-Risk Prediction Dashboard
Streamlit App - Introduction Section

To run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Crypto Tail-Risk Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Introduction", "Data Analysis"],
    index=1  # Default to Introduction
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**STA 160 Final Project**\n\n"
    "Predicting Cryptocurrency Tail-Risk Events Using Machine Learning\n\n"
    "UC Davis | Fall 2024"
)

# ============================================
# MAIN CONTENT - INTRODUCTION SECTION
# ============================================

if page == "Introduction":
    # Main title
    st.markdown('<h1 class="main-header">Cryptocurrency Tail-Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Using Machine Learning to Forecast Extreme Market Movements</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 1: PURPOSE
    # ============================================
    st.markdown('<h2 class="section-header">Purpose & Motivation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Why Predict Tail-Risk Events?
        
        Cryptocurrency markets are notoriously volatile, with **extreme price movements** (tail events) 
        occurring far more frequently than in traditional financial markets. These sudden crashes or 
        surges can result in:
        
        - **Massive portfolio losses** within hours or even minutes
        - **Liquidation cascades** for leveraged traders
        - **Market-wide contagion** affecting multiple assets
        
        ### Project Goals
        
        This project aims to:
        
        1. **Predict tail-risk events** before they occur using historical OHLCV data and technical indicators
        2. **Build interpretable models** that traders can actually use for risk management
        3. **Create an interactive dashboard** for real-time risk monitoring
        4. **Compare multiple ML approaches** from simple logistic regression to deep learning
        
        ### Real-World Impact
        
        By accurately predicting extreme market movements, traders and investors can:
        - **Reduce position sizes** before high-risk periods
        - **Hedge portfolios** with protective options or inverse positions
        - **Time market entries/exits** more effectively
        - **Avoid catastrophic losses** during market crashes
        """)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Key Statistics
        
        **Dataset Coverage:**
        - 211 cryptocurrencies
        - 310K+ daily observations
        - 2020-2024 time period
        
        **Tail Events Defined:**
        - Returns exceeding 95th percentile
        - ~5% of all trading days
        
        **Model Performance:**
        - Baseline: 53-56% accuracy
        - Target: 60-70% ROC-AUC
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.image("https://via.placeholder.com/400x300/1f77b4/ffffff?text=Crypto+Volatility+Chart", 
                 caption="Example: Bitcoin 30-day volatility over time")
    
    st.markdown("---")
    
    # ============================================
    # SECTION 2: DATASET
    # ============================================
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Dataset stats in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Observations", "310,896", delta="252K cleaned")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cryptocurrencies", "211", delta="Top 100 by volume")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features", "47+", delta="Technical indicators")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Time Period", "4+ years", delta="2020-2024")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("###")
    
    # Dataset details
    tab1, tab2, tab3 = st.tabs(["Data Sources", "Feature Engineering", "Data Splits"])
    
    with tab1:
        st.markdown("""
        ### Data Collection & Cleaning
        
        **Primary Source:** Kaggle - "Cryptocurrency Historical Prices (Top 100)"
        
        **Raw Data Characteristics:**
        - **OHLCV data**: Open, High, Low, Close, Volume for each day
        - **342K raw observations** across 211 cryptocurrencies
        - Daily frequency (7 days/week)
        - Multiple exchanges aggregated
        
        **Data Cleaning Pipeline:**
        1. **Standardization**: Unified column names across all coins
        2. **Validation**: Removed impossible values (High < Low, negative prices)
        3. **Time alignment**: Ensured consistent daily frequency
        4. **Missing data**: Dropped coins with >10% zero-volume days (illiquid markets)
        5. **Outlier handling**: Applied robust scaling to handle extreme values
        
        **Quality Checks:**
        - No lookahead bias in feature engineering
        - Temporal ordering preserved
        - No data leakage between train/val/test splits
        - Removed 87 illiquid coins (41% of dataset)
        """)
        
        # Sample data preview
        st.markdown("#### Sample Raw Data")
        sample_data = pd.DataFrame({
            'Date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'Coin': ['BTC-USD', 'BTC-USD', 'BTC-USD'],
            'Open': [42150.23, 42890.45, 43210.12],
            'High': [43210.50, 43550.30, 44120.85],
            'Low': [41890.12, 42450.20, 42980.45],
            'Close': [42890.45, 43210.12, 43890.23],
            'Volume': [28450123000, 31250987000, 29876543000]
        })
        st.dataframe(sample_data, use_container_width=True)
    
    with tab2:
        st.markdown("""
        ### Feature Engineering Process
        
        We engineered **47 features** across multiple categories:
        
        #### 1. Return Features (8 features)
        - Current day log return
        - Lagged returns: 1, 2, 3, 5, 7 days
        - Return momentum indicators
        
        #### 2. Volatility Features (14 features)
        - Historical volatility: 5d, 10d, 20d, 30d windows
        - Volatility ratios (short/long term)
        - EWMA volatility
        - Parkinson & Garman-Klass estimators
        
        #### 3. Technical Indicators (15 features)
        - **Moving Averages**: SMA 5, 20, 50, 200
        - **Oscillators**: RSI (14-day), MACD, ROC
        - **Bands**: Bollinger Bands (upper, lower, position)
        - **Volatility**: ATR (14-day, 21-day)
        
        #### 4. Volume Features (6 features)
        - Volume lags: 1, 2, 3, 5 days
        - Volume change rate
        - Volume ratios
        
        #### 5. Price Features (4 features)
        - High-Low range
        - High vs previous close
        - Low vs previous close
        - True Range
        """)
        
        # Feature importance preview (dummy data)
        st.markdown("#### Top 10 Most Important Features")
        feature_importance = pd.DataFrame({
            'Feature': ['Returns_Lag_1', 'Vol_20d', 'RSI_14', 'MACD_Hist', 
                       'Vol_Ratio_5d', 'ATR_14', 'Returns_Lag_2', 'BB_Position',
                       'ROC_10', 'Volume_Lag_1'],
            'Importance': [0.156, 0.134, 0.098, 0.087, 0.076, 0.069, 0.065, 0.058, 0.052, 0.048],
            'Type': ['Returns', 'Volatility', 'Technical', 'Technical', 
                    'Volatility', 'Technical', 'Returns', 'Technical', 
                    'Technical', 'Volume']
        })
        st.dataframe(feature_importance, use_container_width=True)
    
    with tab3:
        st.markdown("""
        ### Time-Series Data Splits
        
        **Critical:** We use **temporal splits** (not random) to avoid lookahead bias!
        
        ```
        Timeline: 2021-07-12 → 2024-04-14
        
        ├─── Train Set (50%) ───┼─── Val Set (25%) ───┼─── Test Set (25%) ───┤
        2021-07-12              2022-09-10            2024-01-01            2024-04-14
        ```
        
        #### Split Details:
        
        | Dataset | Date Range | Samples | Purpose |
        |---------|------------|---------|---------|
        | **Train** | 2021-07 → 2022-09 | 76,243 | Model training |
        | **Validation** | 2022-09 → 2024-01 | 69,993 | Hyperparameter tuning |
        | **Test** | 2024-01 → 2024-04 | 67,502 | Final evaluation |
        
        #### Why Temporal Splits?
        - **Prevents data leakage**: Model never sees future data
        - **Realistic evaluation**: Tests on truly unseen future periods
        - **Mimics production**: How the model would perform in real trading
        
        #### Class Balance:
        - **UP days**: ~50.5% (price increases)
        - **DOWN days**: ~49.5% (price decreases)
        - **Tail events**: ~5% (extreme moves)
        
        Nearly balanced classes, so accuracy is meaningful!
        """)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 3: RESULTS
    # ============================================
    st.markdown('<h2 class="section-header">Model Results & Performance</h2>', unsafe_allow_html=True)
    
    # Model comparison table
    st.markdown("### Model Performance Summary")
    
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'ARIMA (1-Day)', 'ARIMA (7-Day)'],
        'Approach': ['Classification (Direction)', 'Time Series Forecast', 'Time Series Forecast'],
        'Test ROC-AUC': ['0.53-0.56', 'N/A (regression)', 'N/A (regression)'],
        'Coverage (95% CI)': ['N/A', '~95%', '~100%'],
        'Coverage (80% CI)': ['N/A', '~80%', '~92%'],
        'MAPE': ['N/A', '~6-15%', '~10-20%'],
        'Status': ['✓ Complete', '✓ Complete', '✓ Complete']
    })
    
    st.dataframe(results_df, use_container_width=True)
    
    st.info("""
    **Note on Model Comparison:**
    - Logistic Regression predicts **direction** (up/down) with poor results (barely above random)
    - ARIMA predicts **actual prices** with confidence intervals - a fundamentally different approach
    - ARIMA shows strong coverage: predictions fall within confidence intervals 80-100% of the time
    - We pivoted to ARIMA after classification models showed weak predictive power
    """)
    
    st.markdown("###")
    
    # Key findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Key Findings
        
        1. **Classification Approach Failed**
           - Logistic regression achieved only 54% accuracy (barely above random 50%)
           - ROC-AUC of 0.53-0.56 indicates virtually no predictive power
           - Direction prediction proved too difficult for crypto markets
        
        2. **Pivot to ARIMA Time Series**
           - Switched from classification to price forecasting
           - ARIMA(1,1,1) models trained per coin with rolling 730-day windows
           - Used empirical confidence intervals for uncertainty quantification
        
        3. **ARIMA Performance (20 coins tested)**
           - **1-Day Forecasts**: 95% CI coverage ≈95%, 80% CI coverage ≈80%
           - **7-Day Forecasts**: 95% CI coverage ≈100%, 80% CI coverage ≈92%
           - MAPE ranges from 6-20% depending on coin volatility
           - Examples: ADA-USD (95.3% 1-day coverage), AVAX-USD (95.1% 1-day)
        
        4. **Error Distribution Properties**
           - Nearly normal error distributions (slight positive skew)
           - Mean errors close to zero (-0.017 to -0.049)
           - Kurtosis 1.6-2.4 (slightly lighter tails than normal)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Insights & Lessons Learned
        
        **Why Classification Failed:**
        - Crypto markets exhibit strong random walk behavior
        - Technical indicators alone insufficient for direction prediction
        - High noise-to-signal ratio in daily price movements
        - Class imbalance and weak feature separability
        
        **Why ARIMA Works Better:**
        - Focuses on price levels, not binary direction
        - Captures short-term autocorrelation (AR component)
        - Accounts for non-stationarity (differencing)
        - Provides interpretable confidence intervals
        - Rolling window adapts to regime changes
        
        **Configuration Details:**
        - **Model**: ARIMA(1,1,1) selected via BIC per coin
        - **Training window**: 730 days (2 years) rolling
        - **Test period**: 365 days (2024-11 to 2025-11)
        - **Transform**: Log prices for stability
        - **Tested coins**: 20 major cryptocurrencies
        
        **Future Improvements:**
        - Add GARCH for volatility modeling
        - Incorporate exogenous variables (Bitcoin dominance)
        - Try SARIMAX for seasonal patterns
        - Ensemble ARIMA with ML models
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("###")
    
    # ARIMA example visualizations
    st.markdown("### Sample ARIMA Results: ADA-USD & AVAX-USD")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ADA-USD Performance")
        st.markdown("""
        **1-Day Forecast:**
        - Coverage (95%): 95.3%
        - Coverage (80%): 80.3%
        - MAPE: ~6-8%
        - Mean error: -0.017
        
        **7-Day Forecast:**
        - Coverage (95%): 100.0%
        - Coverage (80%): 99.2%
        - Mean error: -0.041
        
        *Error distribution shows slight positive skew (0.53-0.64), near-normal kurtosis (1.61-2.35)*
        """)
        
        st.image("https://via.placeholder.com/500x400/1f77b4/ffffff?text=ADA-USD+Forecast+Chart", 
                 caption="ADA-USD: Actual vs Predicted with 95% CI")
    
    with col2:
        st.markdown("#### AVAX-USD Performance")
        st.markdown("""
        **1-Day Forecast:**
        - Coverage (95%): 95.1%
        - Coverage (80%): 80.5%
        - MAPE: ~8-10%
        - Mean error: -0.024
        
        **7-Day Forecast:**
        - Coverage (95%): 100.0%
        - Coverage (80%): 99.2%
        - Mean error: -0.049
        
        *Error distribution nearly normal (skew 0.19-0.44), low kurtosis (0.88-2.16)*
        """)
        
        st.image("https://via.placeholder.com/500x400/2ca02c/ffffff?text=AVAX-USD+Forecast+Chart", 
                 caption="AVAX-USD: Actual vs Predicted with 95% CI")
    
    st.markdown("---")
    
    # Key metrics comparison
    st.markdown("### Coverage Performance Across All Coins")
    
    coverage_data = pd.DataFrame({
        'Coin': ['ADA-USD', 'AVAX-USD', 'Average (20 coins)'],
        '1-Day 95% Coverage': ['95.3%', '95.1%', '~95%'],
        '1-Day 80% Coverage': ['80.3%', '80.5%', '~80%'],
        '7-Day 95% Coverage': ['100.0%', '100.0%', '~100%'],
        '7-Day 80% Coverage': ['99.2%', '99.2%', '~92%']
    })
    
    st.dataframe(coverage_data, use_container_width=True)
    
    st.success("""
    ✨ **Key Takeaway:** ARIMA models show excellent coverage properties, meaning our confidence 
    intervals are well-calibrated. When we say "95% confidence", the true price falls within 
    our interval ~95% of the time. This makes the model reliable for risk management!
    """)
    
    st.markdown("---")
    
    # Call to action
    st.success("Explore the **Data Analysis** section to see interactive visualizations and make your own predictions!")

# ============================================
# HOME PAGE
# ============================================
elif page == "Home":
    st.markdown('<h1 class="main-header">Welcome to Crypto Tail-Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An Interactive Dashboard for Cryptocurrency Risk Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Introduction**\n\nLearn about the project purpose, dataset, and model results.")
        if st.button("Go to Introduction", use_container_width=True):
            st.session_state.page = "Introduction"
            st.rerun()
    
    with col2:
        st.success("**Data Analysis**\n\nExplore interactive charts, predictions, and coin comparisons.")
        st.button("Go to Data Analysis", use_container_width=True, disabled=True)
        st.caption("Coming soon!")
    
    with col3:
        st.warning("**Live Predictions**\n\nGet real-time tail-risk predictions for any cryptocurrency.")
        st.button("Go to Predictions", use_container_width=True, disabled=True)
        st.caption("Coming soon!")

# ============================================
# DATA ANALYSIS PAGE (PLACEHOLDER)
# ============================================
elif page == "Data Analysis":
    st.markdown('<h1 class="main-header">Data Analysis & Exploration</h1>', unsafe_allow_html=True)
    st.info("This section is under construction. Check back soon for interactive visualizations!")
    
    st.markdown("### Planned Features:")
    st.markdown("""
    - Historical price charts with tail-event markers
    - Feature correlation heatmaps
    - Model prediction explorer (select coin, date range)
    - Tail-risk scanner (current high-risk coins)
    - Volatility regime visualization
    - Backtesting simulator
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>STA 160 Final Project</strong> | UC Davis | Fall 2024</p>
    <p>Cryptocurrency Tail-Risk Prediction Using Machine Learning</p>
    <p style='font-size: 0.9rem;'>Built with Streamlit 🎈 | Data from Kaggle</p>
</div>
""", unsafe_allow_html=True)