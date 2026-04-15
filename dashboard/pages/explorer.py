import sys
import os
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.models.statistical import StatisticalDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.ensemble import EnsembleDetector
from dashboard.components.charts import price_with_anomalies, anomaly_timeline

st.set_page_config(page_title="Historical Explorer", layout="wide")
st.title("🔍 Historical Anomaly Explorer")

# --- Sidebar ---
st.sidebar.header("Explorer Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
sensitivity = st.sidebar.slider(
    "Detection Sensitivity",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Lower = more sensitive (more anomalies flagged)",
)

show_models = st.sidebar.multiselect(
    "Show Scores For",
    ["Statistical", "Isolation Forest", "Ensemble"],
    default=["Ensemble"],
)


@st.cache_data(show_spinner="Analyzing historical data...")
def analyze(ticker, period):
    fetcher = DataFetcher()
    df = fetcher.fetch(ticker=ticker, period=period)
    fe = FeatureEngineer()
    featured = fe.engineer(df)
    feature_cols = [c for c in fe.get_feature_columns() if c in featured.columns]

    # Statistical
    stat = StatisticalDetector()
    stat_result = stat.detect(featured["returns"])

    # Isolation Forest
    iso = IsolationForestDetector()
    iso_result = iso.detect(featured, feature_cols=feature_cols)

    # Ensemble
    n = min(len(stat_result), len(iso_result))
    ens = EnsembleDetector()
    scores = {
        "statistical": stat_result["anomaly_score"].values[-n:],
        "isolation_forest": iso_result["anomaly_score"].values[-n:],
    }
    ens_result = ens.detect(scores, index=featured.index[-n:])

    # Build combined results
    result = pd.DataFrame(index=featured.index[-n:])
    result["close"] = featured["Close"].values[-n:]
    result["volume"] = featured["Volume"].values[-n:]
    result["returns"] = featured["returns"].values[-n:]
    result["statistical_score"] = stat_result["anomaly_score"].values[-n:]
    result["isolation_forest_score"] = iso_result["anomaly_score"].values[-n:]
    result["ensemble_score"] = ens_result["ensemble_score"].values

    return result


if st.sidebar.button("🔎 Analyze", type="primary"):
    result = analyze(ticker, period)

    # Apply sensitivity threshold
    result["anomaly"] = result["ensemble_score"] > (1 - sensitivity)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", len(result))
    col2.metric("Anomalies Detected", int(result["anomaly"].sum()))
    col3.metric("Anomaly Rate", f"{result['anomaly'].mean():.1%}")
    col4.metric("Max Score", f"{result['ensemble_score'].max():.3f}")

    # Main chart
    fig = price_with_anomalies(result, f"{ticker} — Historical Analysis")
    st.plotly_chart(fig, use_container_width=True)

    # Model scores timeline
    if show_models:
        score_series = {}
        if "Statistical" in show_models:
            score_series["statistical"] = result["statistical_score"]
        if "Isolation Forest" in show_models:
            score_series["isolation_forest"] = result["isolation_forest_score"]
        if "Ensemble" in show_models:
            score_series["ensemble"] = result["ensemble_score"]

        fig = anomaly_timeline(score_series, threshold=1 - sensitivity)
        st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    st.subheader("Detected Anomalies")
    anomalies = result[result["anomaly"]].copy()
    anomalies["date"] = anomalies.index
    if len(anomalies) > 0:
        display_cols = ["date", "close", "returns", "ensemble_score"]
        available = [c for c in display_cols if c in anomalies.columns]
        st.dataframe(
            anomalies[available]
            .sort_values("ensemble_score", ascending=False)
            .head(50)
            .style.format(
                {"close": "${:.2f}", "returns": "{:.4f}", "ensemble_score": "{:.3f}"}
            ),
            use_container_width=True,
        )
    else:
        st.info("No anomalies detected at this sensitivity level.")

else:
    st.info("Enter a ticker and click **Analyze** to explore historical anomalies.")
