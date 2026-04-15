import sys
import os
import time
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.models.statistical import StatisticalDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.ensemble import EnsembleDetector
from src.detection.stream_simulator import StreamSimulator
from dashboard.components.charts import price_with_anomalies

st.set_page_config(page_title="Real-Time Detection", layout="wide")
st.title("🔴 Real-Time Anomaly Detection")

# --- Sidebar Controls ---
st.sidebar.header("Stream Settings")
ticker = st.sidebar.selectbox("Ticker", ["SPY", "AAPL", "MSFT", "BTC-USD"], index=0)
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=0)
speed = st.sidebar.slider("Stream Speed (points/sec)", 1, 50, 10)
inject = st.sidebar.checkbox("Inject Synthetic Anomalies", value=True)

active_models = st.sidebar.multiselect(
    "Active Models",
    ["Statistical", "Isolation Forest", "Ensemble"],
    default=["Statistical", "Isolation Forest", "Ensemble"],
)

# --- Data Loading ---
@st.cache_data(show_spinner="Fetching data...")
def load_data(ticker, period):
    fetcher = DataFetcher()
    df = fetcher.fetch(ticker=ticker, period=period)
    fe = FeatureEngineer()
    return fe.engineer(df)


@st.cache_resource(show_spinner="Training models...")
def train_models(ticker, period):
    df = load_data(ticker, period)
    feature_cols = FeatureEngineer().get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]

    stat = StatisticalDetector()
    iso = IsolationForestDetector()
    iso.fit(df[available])

    return stat, iso, available


if st.sidebar.button("▶️ Start Stream", type="primary"):
    df = load_data(ticker, period)
    stat, iso, feature_cols = train_models(ticker, period)
    ensemble = EnsembleDetector()

    # Stream simulation
    simulator = StreamSimulator()
    simulator.config["stream"]["replay_speed"] = speed
    simulator.config["stream"]["inject_anomalies"] = inject
    simulator.load(df)

    # Placeholders for live updating
    chart_placeholder = st.empty()
    metrics_cols = st.columns(4)
    score_placeholder = st.empty()
    alert_placeholder = st.empty()

    results = []
    alerts = []

    progress = st.progress(0)

    for row, is_injected in simulator.stream(realtime=False):
        idx = simulator._index - 1

        # Statistical detection on returns
        if "returns" in df.columns and idx >= 20:
            window = df["returns"].iloc[max(0, idx - 100) : idx + 1]
            stat_result = stat.detect(window)
            stat_score = float(stat_result["anomaly_score"].iloc[-1])
        else:
            stat_score = 0.0

        # Isolation Forest detection
        if idx >= 1:
            row_features = df[feature_cols].iloc[idx : idx + 1]
            if len(row_features) > 0 and not row_features.isna().any(axis=1).iloc[0]:
                if_scores = iso.score_samples(row_features.values)
                raw = iso.model.score_samples(row_features.values)
                if_score = float(
                    (raw.max() - raw) / (raw.max() - raw.min() + 1e-10)
                ) if raw.max() != raw.min() else 0.0
            else:
                if_score = 0.0
        else:
            if_score = 0.0

        # Ensemble
        scores = {
            "statistical": np.array([stat_score]),
            "isolation_forest": np.array([if_score]),
        }
        ens_result = ensemble.combine_scores(scores)
        ens_score = float(ens_result[0])

        is_anomaly = ens_score > 0.5 or stat_score > 1.0

        results.append(
            {
                "close": df["Close"].iloc[idx] if "Close" in df.columns else row.get("Close", 0),
                "volume": df["Volume"].iloc[idx] if "Volume" in df.columns else 0,
                "statistical_score": stat_score,
                "isolation_forest_score": if_score,
                "ensemble_score": ens_score,
                "anomaly": is_anomaly,
                "injected": is_injected,
            }
        )

        if is_anomaly:
            alerts.append(
                f"⚠️ **{df.index[idx].strftime('%Y-%m-%d') if hasattr(df.index[idx], 'strftime') else idx}** — "
                f"Score: {ens_score:.2f} | Stat: {stat_score:.2f} | IF: {if_score:.2f}"
            )

        # Update UI every N points
        if idx % max(speed, 5) == 0 or simulator.is_exhausted:
            res_df = pd.DataFrame(results, index=df.index[: len(results)])

            with chart_placeholder.container():
                fig = price_with_anomalies(res_df, f"{ticker} — Live Detection")
                st.plotly_chart(fig, use_container_width=True)

            n_anomalies = sum(1 for r in results if r["anomaly"])
            metrics_cols[0].metric("Points Processed", len(results))
            metrics_cols[1].metric("Anomalies Found", n_anomalies)
            metrics_cols[2].metric("Latest Score", f"{ens_score:.3f}")
            metrics_cols[3].metric("Progress", f"{simulator.get_progress():.0%}")

            if alerts:
                with alert_placeholder.container():
                    st.subheader("Recent Alerts")
                    for alert in alerts[-5:]:
                        st.markdown(alert)

            progress.progress(simulator.get_progress())

        time.sleep(1.0 / speed)

    st.success(f"Stream complete. Processed {len(results)} data points.")

else:
    st.info("Configure settings in the sidebar and click **Start Stream** to begin.")
