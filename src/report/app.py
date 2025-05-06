# src/report/app.py

import streamlit as st  # Streamlit for UI :contentReference[oaicite:0]{index=0}
import pandas as pd
import plotly.express as px  # Plotly Express for simple charts 
import plotly.graph_objects as go
from pathlib import Path
from streamlit_flow import streamlit_flow  # Interactive flowchart component :contentReference[oaicite:2]{index=2}
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
import pickle
import streamlit as st


# ——————————————
# Page config & constants
# ——————————————
st.set_page_config(
    page_title="AI Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)  # must be first Streamlit call :contentReference[oaicite:3]{index=3}



BACKTEST_DIR = Path("data/backtest")

# ——————————————
# Sidebar navigation
# ——————————————
page = st.sidebar.radio(
    "Navigate",
    ("Home", "Stock Analysis", "Project Workflow","Glossary"),
    index=0
)

# ——————————————
# 1) HOME VIEW
# ——————————————
if page == "Home":
    

    st.title("📈 Predicting and Forecasting Stock Prices")
    st.markdown(
        """
        Welcome! This dashboard helps us to understand how different AI models perform on stock‑price forecasting.

        **What We Did:** We gathered daily stock data from Polygon.io, created helpful indicators (moving averages, RSI, Bollinger Bands),
        and trained three neural networks (LSTM, 1D‑CNN, Transformer) to forecast next‑day closing prices.

        **Explore:**
        - **Stock Analysis**: Pick a symbol, compare model predictions vs actuals.
        - **Error Analysis**: Inspect error distributions and rolling MAE.
        - **Returns Simulation**: See how a simple trading rule based on predictions would perform.
        - **Project Workflow**: Visualize the end‑to‑end development pipeline.
        """,
        unsafe_allow_html=True
    )

# ——————————————
# 2) STOCK ANALYSIS & ERROR ANALYSIS & RETURNS
# ——————————————
elif page == "Stock Analysis":
    st.header("🔍 Stock Analysis")

    

    # Select symbol & models
    preds = sorted(BACKTEST_DIR.glob("*_predictions.csv"))
    symbols = sorted({p.stem.split("_")[0] for p in preds})
    symbol = st.sidebar.selectbox("Select stock", symbols)

    available = [
        f.stem.replace(f"{symbol}_", "").replace("_predictions","")
        for f in BACKTEST_DIR.glob(f"{symbol}_*_predictions.csv")
    ]
    models = st.sidebar.multiselect(
        "Select models to compare", available, default=available
    )

    # Load data & metrics
    dfs, metrics = {}, {}
    for m in models:
        df = pd.read_csv(
            BACKTEST_DIR / f"{symbol}_{m}_predictions.csv",
            parse_dates=['datetime'], index_col='datetime'
        )
        dfs[m] = df
        lines = (BACKTEST_DIR / f"{symbol}_{m}_metrics.txt").read_text().splitlines()
        metrics[m] = {
            k: float(v) for line in lines if ":" in line
            for k,v in [line.split(":")]
        }

    # Volatility summary
    vol = dfs[models[0]]['true'].pct_change().std()
    vol_desc = "very choppy" if vol>0.02 else "somewhat variable" if vol>0.01 else "relatively calm"
    st.markdown(f"**During the test period, {symbol} was {vol_desc} (daily σ ~{vol:.2%}).**")

    # Performance table
    st.subheader("Model Performance Summary")
    dfm = pd.DataFrame(metrics).T.rename(columns={
        "MSE":"Avg. Sq. Error","MAE":"Avg. $ Error","DirAcc":"% Correct Direction"
    })
    st.dataframe(
        dfm.style.format({
            "Avg. Sq. Error":"{:.2f}",
            "Avg. $ Error":"{:.2f}",
            "% Correct Direction":"{:.1%}"
        }),
        use_container_width=True
    )
    with st.expander("Metric Explanations"):
        st.markdown(
            """
            - **Average Squared Error (MSE):** Mean of squared prediction errors; penalizes large misses.
            - **Average Dollar Error (MAE):** Mean of absolute errors in dollars; straightforward deviation.
            - **Directional Accuracy:** Percent of times the model predicted the correct up/down move.
            """
        )

    # Tabs for charts
    tabs = st.tabs(["💲 Price Chart","📊 Error Analysis","💹 Returns Simulation"])

    # Price Chart
    with tabs[0]:
        st.subheader("Actual vs Predictions")
        price_df = pd.DataFrame({'Actual': dfs[models[0]]['true']})
        for m,df in dfs.items(): price_df[m] = df['pred']
        fig = px.line(
            price_df,
            labels={'value':'Price (USD)','index':'Date','variable':'Series'},
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("*Comparision of Models against actual market performance*")

    # Error Analysis
    with tabs[1]:
        st.subheader("Prediction Error Distribution & Rolling MAE")
        st.markdown(
            "Below are two views of model errors: a histogram of error frequencies and a 20‑day rolling average of absolute errors (MAE)."
        )
        err_df = pd.DataFrame({m: dfs[m]['pred']-dfs[m]['true'] for m in models})
        fig_hist = px.histogram(
            err_df, nbins=50, marginal="box",
            labels={'value':'Error (USD)','variable':'Model'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown(
            "*Histogram shows how often models over‑ or under‑predict by various amounts.*"
        )

        st.subheader("20‑day Rolling MAE")
        fig_roll = go.Figure()
        for m in models:
            roll = err_df[m].abs().rolling(20).mean()
            fig_roll.add_trace(go.Scatter(x=roll.index, y=roll, name=m))
        fig_roll.update_layout(xaxis_title="Date", yaxis_title="MAE (USD)")
        st.plotly_chart(fig_roll, use_container_width=True)
        st.markdown(
            "*The 20‑day rolling MAE smooths daily errors over a month. The curve begins at day 20 when enough data points exist.*"
        )

    # Returns Simulation
    with tabs[2]:
        st.subheader("Cumulative Returns: Simple Long Strategy")
        ret_df = pd.DataFrame()
        for m,df in dfs.items():
            sig = df['pred'].diff().shift() > 0
            strat_ret = df['true'].pct_change().fillna(0) * sig
            ret_df[m] = (1+strat_ret).cumprod() - 1
        cum = px.line(
            ret_df,
            labels={'value':'Cumulative Return','index':'Date','variable':'Model'},
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(cum, use_container_width=True)
        st.markdown(
            "*Compounds $1 on days when the model predicts an up‑day. This correctly shows cumulative strategy growth.*"
        )

    
    # Download buttons
    st.markdown("---")
    st.download_button(
        "📥 Download comparison CSV",
        data=price_df.to_csv().encode(),
        file_name=f"{symbol}_comparison.csv",
        mime="text/csv"
    )
    st.download_button(
        "📥 Download metrics CSV",
        data=pd.DataFrame(metrics).T.to_csv().encode(),
        file_name=f"{symbol}_metrics.csv",
        mime="text/csv"
    )

# ——————————————
# 3) PROJECT WORKFLOW VIEW
# ——————————————
elif page == "Project Workflow":
    # inside your `elif page == "Project Workflow":` block

    st.header("🗂️ Project Workflow")

    # Ingestion folder
    with st.expander("📥 Ingestion (src/ingestion)", expanded=False):
        st.write("Fetches raw OHLC bars from Polygon.io and saves CSVs under `data/raw/bars`.")
        st.markdown(
            """
            - **historical_data.py**  
            Defines `fetch_and_save_bars()`: handles API calls, rate‑limit backoff, and writes raw CSVs.  
            - **__init__.py**  
            Marks this folder as a Python package.
            """
        )

    # Preprocessing folder
    with st.expander("⚙️ Preprocessing (src/preprocessing)", expanded=False):
        st.write("Cleans timestamps, drops NaNs, engineers features, and outputs processed CSVs.")
        st.markdown(
            """
            - **clean_and_feature.py**  
            Parses `t`→`datetime` index, renames, and saves cleaned data.  
            - **remove_non_numeric.py**  
            Drops any non‑numeric columns before modeling.  
            - **run_preprocessing.py**  
            Runner: loops raw CSVs → `clean_and_feature.py`.  
            - **run_feature_engineering.py**  
            Runner: loops cleaned CSVs → `remove_non_numeric.py`.
            """
        )

    # Dataset‑splitting folder
    with st.expander("✂️ Splitting (src/dataset)", expanded=False):
        st.write("Chronologically splits each numeric CSV into train/val/test sets.")
        st.markdown(
            """
            - **split_dataset.py**  
            `split_time_series()`: loads numeric CSV, slices by fractions, saves three splits.  
            - **run_split.py**  
            Runner: applies `split_time_series()` across all symbols.
            """
        )

    # Modeling folder
    with st.expander("🤖 Modeling (src/modeling)", expanded=False):
        st.write("Defines and trains LSTM, 1D‑CNN, and Transformer models.")
        st.markdown(
            """
            - **models_lstm.py**  
            `build_lstm()`: constructs and compiles LSTM network.  
            - **models_cnn.py**  
            `build_cnn1d()`: constructs and compiles 1D‑CNN network.  
            - **models_transformer.py**  
            `build_transformer()`: defines a simple Transformer encoder.  
            - **train_models.py**  
            Loads CSV splits, scales data, windows sequences, fits models with EarlyStopping.  
            - **run_training.py**  
            Runner: loops symbols & architectures, calls `train_and_save()` for each.
            """
        )

    # Backtesting folder
    with st.expander("📊 Backtesting (src/backtest)", expanded=False):
        st.write("Loads saved models & scalers, runs on test split, computes and saves metrics + predictions.")
        st.markdown(
            """
            - **backtest_models.py**  
            `backtest_model()`: windows test data, predicts, inverts scaling, computes MSE/MAE/DirAcc, writes CSV/TXT.  
            - **run_backtest.py**  
            Runner: iterates `.keras` model files, loads corresponding scaler & test CSV, calls backtest.
            """
        )

    # Reporting folder
    with st.expander("📈 Reporting (src/report)", expanded=False):
        st.write("Streamlit app (`app.py`) that presents charts, error analysis, and download links.")
        st.markdown(
            """
            - **app.py**  
            Main dashboard: Home, Stock Analysis (price/error/returns), and this Workflow view.  
            - **(optional)** any helper modules for custom charts or styling.
            """
        )

# in src/report/app.py, replace your “Glossary” section with this:

elif page == "Glossary":
    st.header("📚 Glossary of Technical Terms")

    # Bollinger Bands
    with st.expander("Bollinger Bands"):
        st.markdown(
            """
            <p><strong>Definition:</strong> Envelopes plotted ±2 standard deviations around a 20‑day simple moving average to capture volatility swings.</p>
            <p><strong>Use:</strong> Identify overbought or oversold extremes when price touches the upper or lower band.</p>
            """,
            unsafe_allow_html=True
        )
        # example chart (unchanged)
        import numpy as np, pandas as pd, plotly.express as px
        prices = pd.Series(np.sin(np.linspace(0,6,50))*5 + 100)
        ma = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        df = pd.DataFrame({
            'Price': prices,
            'MA (20)': ma,
            'Upper Band': ma + 2*std,
            'Lower Band': ma - 2*std
        }).dropna()
        fig = px.line(df, labels={'value':'Price','index':'Time'})
        st.plotly_chart(fig, use_container_width=True)

    # LSTM
    with st.expander("LSTM (Long Short‑Term Memory)"):
        st.markdown(
            """
            <p><strong>Definition:</strong> A type of recurrent neural network cell with input, forget, and output gates that capture long‑term dependencies in sequences.</p>
            <p><strong>Use:</strong> Excellent for forecasting time series where past context over many steps matters.</p>
            """,
            unsafe_allow_html=True
        )
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/3/37/LSTM_gate.svg",
            caption="LSTM cell diagram"
        )

    # 1D‑CNN
    with st.expander("1D‑CNN (One‑Dimensional Convolutional Neural Network)"):
        st.markdown(
            """
            <p><strong>Definition:</strong> Applies convolutional filters along the time axis to automatically extract local temporal patterns.</p>
            <p><strong>Use:</strong> Captures short‑term features (e.g. spikes) in time‑series data with fewer parameters than RNNs.</p>
            """,
            unsafe_allow_html=True
        )

    # Transformer
    with st.expander("Transformer"):
        st.markdown(
            """
            <p><strong>Definition:</strong> Attention‑based architecture that models long‑range dependencies by weighting all time steps in parallel, without recurrence.</p>
            <p><strong>Use:</strong> Handles very long sequences efficiently and captures global context via self‑attention.</p>
            """,
            unsafe_allow_html=True
        )

    # RSI
    with st.expander("RSI (Relative Strength Index)"):
        st.markdown(
            """
            <p><strong>Definition:</strong> A momentum oscillator (0–100) that measures the magnitude of recent price changes to detect overbought (>70) or oversold (<30) conditions.</p>
            <p><strong>Use:</strong> Signals potential reversals when divergences occur between RSI and price.</p>
            """,
            unsafe_allow_html=True
        )

    # MACD
    with st.expander("MACD (Moving Average Convergence/Divergence)"):
        st.markdown(
            """
            <p><strong>Definition:</strong> The difference between 12‑day and 26‑day exponential moving averages, plotted with a 9‑day signal line.</p>
            <p><strong>Use:</strong> Identifies shifts in trend momentum via crossovers of the MACD line and its signal line.</p>
            """,
            unsafe_allow_html=True
        )

    # Moving Average
    with st.expander("MA (Moving Average)"):
        st.markdown(
            """
            <p><strong>Definition:</strong> A trend‑following indicator that smooths price data by averaging closing prices over N periods.</p>
            <p><strong>Use:</strong> Determines trend direction and dynamic support/resistance; crossovers of short vs. long MAs generate trading signals.</p>
            """,
            unsafe_allow_html=True
        )

    # MAE
    with st.expander("MAE (Mean Absolute Error)"):
        st.markdown(
            """
            <p><strong>Definition:</strong> The average of the absolute differences between predicted and actual values:</p>
            <pre style="font-size:90%;">MAE = (1/n) ∑ |y<sub>i</sub> − x<sub>i</sub>|</pre>
            <p><strong>Use:</strong> Evaluates forecast accuracy in the same units as the data.</p>
            """,
            unsafe_allow_html=True
        )

