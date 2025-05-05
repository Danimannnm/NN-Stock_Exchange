# src/report/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ——————————————
# Page config & constants
# ——————————————
st.set_page_config(
    page_title="Stock‑NN Backtest Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
BACKTEST_DIR = Path("data/backtest")

# ——————————————
# Sidebar navigation
# ——————————————
page = st.sidebar.radio(
    "Navigate",
    ("Home", "Stock Analysis","Project Workflow"),
    index=0,
    key="page_nav"
)

# ——————————————
# 1) HOME VIEW
# ——————————————
if page == "Home":
    st.title("📈 Stock Forecast Comparison Portal")
    st.markdown(
        """
        **Project Overview**

        We collected historical daily OHLC data from Polygon.io, engineered technical indicators (MA, RSI, Bollinger Bands),
        and trained three neural network architectures (LSTM, 1D‑CNN, Transformer) to forecast next‑day closing prices.
        We then backtested each model on an unseen test set to compare accuracy, directional correctness, and simulated returns.

        **Key Sections:**
        - **Stock Analysis**: Pick a symbol, compare model predictions vs actuals.
        - **Error Analysis**: Inspect error distributions and rolling MAE.
        - **Returns Simulation**: See how a simple trading rule based on predictions would perform.
        """,
        unsafe_allow_html=True
    )
    

else:
    st.header("🔍 Stock Analysis")

    # choose symbol & models
    preds = sorted(BACKTEST_DIR.glob("*_predictions.csv"))
    symbols = sorted({p.stem.split("_")[0] for p in preds})
    symbol = st.sidebar.selectbox("Select stock", symbols)

    available = [f.stem.replace(f"{symbol}_", "").replace("_predictions", "")
                 for f in BACKTEST_DIR.glob(f"{symbol}_*_predictions.csv")]
    models = st.sidebar.multiselect("Select models to compare", available, default=available)

    # load data and metrics
    dfs, metrics = {}, {}
    for m in models:
        df = pd.read_csv(
            BACKTEST_DIR / f"{symbol}_{m}_predictions.csv",
            parse_dates=['datetime'],
            index_col='datetime'
        )
        dfs[m] = df
        lines = (BACKTEST_DIR / f"{symbol}_{m}_metrics.txt").read_text().splitlines()
        metrics[m] = {k: float(v) for line in lines if ":" in line for k,v in [line.split(":")]}

    # summary volatility description
    vol = dfs[models[0]]['true'].pct_change().std()
    vol_desc = "very choppy" if vol>0.02 else "somewhat variable" if vol>0.01 else "relatively calm"
    st.markdown(f"**During the test period, {symbol} was {vol_desc} (daily σ ~{vol:.2%}).**")

    # metrics table with explanations
    st.subheader("Model Performance Summary")
    dfm = pd.DataFrame(metrics).T.rename(columns={
        "MSE":"Avg. Sq. Error", 
        "MAE":"Avg. $ Error", 
        "DirAcc":"% Correct Direction"
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
            - **Average Dollar Error (MAE):** Mean of absolute errors in dollars; simple average deviation.
            - **Directional Accuracy:** Percent of times the model predicted the correct up/down move.
            """
        )

    # tabs
    tabs = st.tabs(["💲 Price Chart", "📊 Error Analysis", "💹 Returns Simulation", "📝 Addendum"])

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
        st.markdown("*This chart shows the actual closing prices (solid line) alongside each model's predicted prices (dashed lines).* ")

    # Error Analysis
    with tabs[1]:
        st.subheader("Prediction Error Distribution & Rolling MAE")
        err_df = pd.DataFrame({m: (df['pred']-df['true']) for m,df in dfs.items()})
        fig_hist = px.histogram(
            err_df, nbins=50, marginal="box",
            labels={'value':'Error','variable':'Model'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("*Histogram of the prediction errors, showing how often and by how much each model over- or under-predicts.*")

        st.subheader("20‑day Rolling MAE")
        fig_roll = go.Figure()
        for m in models:
            roll = err_df[m].abs().rolling(20).mean()
            fig_roll.add_trace(go.Scatter(x=roll.index, y=roll, name=m))
        fig_roll.update_layout(xaxis_title="Date", yaxis_title="MAE")
        st.plotly_chart(fig_roll, use_container_width=True)
        st.markdown("*The rolling MAE smooths daily errors over a 20-day window. The first 19 days have no value because there aren't enough data points to compute a full 20-day average.*")

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
        st.markdown("*Shows growth of $1 invested following a simple rule: go long when the model predicts an up-day.*")

    # Addendum tab
    with tabs[3]:
        st.subheader("Addendum: Rolling MAE Window Effect")
        st.markdown(
            "When computing a rolling metric over N days (here N=20), the first N-1 days lack enough prior points to form a full window."
            " Therefore the plot appears blank until day 20. Once 20 days of errors accumulate, the average becomes defined and the curve begins.*"
        )

    # downloads
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
