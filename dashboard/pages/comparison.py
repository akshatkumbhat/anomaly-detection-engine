import sys
import os
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.models.statistical import StatisticalDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.naive_baseline import NaiveBaselineDetector
from src.models.ensemble import EnsembleDetector
from src.evaluation.metrics import AnomalyEvaluator
from src.utils.helpers import load_config
from dashboard.components.charts import (
    model_comparison_bars,
    roc_curves,
    confusion_matrix_heatmap,
)

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison & Benchmarks")

config = load_config()
eval_cfg = config.get("evaluation", {})

# --- Sidebar ---
st.sidebar.header("Benchmark Settings")
ticker = st.sidebar.selectbox("Ticker", ["SPY", "AAPL", "MSFT", "BTC-USD"], index=0)
period = st.sidebar.selectbox("Period", ["2y", "5y", "max"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Labeling Criteria")
min_return = st.sidebar.slider(
    "|Return| threshold",
    0.01, 0.10, eval_cfg.get("point_label", {}).get("min_abs_return", 0.03), 0.005,
    help="A day is 'anomalous' if |daily return| exceeds this",
)
min_vol_z = st.sidebar.slider(
    "Volume Z-score threshold",
    1.0, 5.0, eval_cfg.get("point_label", {}).get("min_volume_zscore", 3.0), 0.5,
    help="A day is 'anomalous' if |volume z-score| exceeds this",
)

# --- Caveats ---
with st.expander("Methodology & Caveats", expanded=False):
    st.markdown("""
    **Why these results look different from most tutorials:**

    1. **No objective ground truth.** Anomaly detection in finance is unsupervised.
       We define "anomalous" as days with extreme price moves (|return| > {:.1%}) or
       extreme volume (z-score > {:.1f}). This is a *proxy*, not truth.

    2. **Temporal train/test split.** Models are trained on data up to {} and
       tested on data after {}. This prevents look-ahead bias — the model has
       never seen the test data.

    3. **Naive baseline included.** A simple "flag days with big moves" detector
       is shown alongside ML models. If an ML model can't beat this, it's not
       adding value over a basic rule.

    4. **Contamination set to 'auto'.** We don't tell Isolation Forest what % of
       data is anomalous — that would leak information.

    5. **Point-level labels, not date ranges.** Not every day in a "crash" is
       equally anomalous. Some days during COVID had positive returns.

    **Expected results:** AUC of 0.55-0.75 is realistic. Perfect ROC curves
    should make you suspicious, not impressed.
    """.format(
        min_return,
        min_vol_z,
        eval_cfg.get("train_end", "2024-06-30"),
        eval_cfg.get("test_start", "2024-07-01"),
    ))


@st.cache_data(show_spinner="Loading data and running models...")
def run_benchmark(ticker, period, min_return, min_vol_z):
    train_end = eval_cfg.get("train_end", "2024-06-30")
    test_start = eval_cfg.get("test_start", "2024-07-01")

    # --- Fetch & engineer ---
    fetcher = DataFetcher()
    df = fetcher.fetch(ticker=ticker, period=period)
    fe = FeatureEngineer()
    featured = fe.engineer(df)
    feature_cols = [c for c in fe.get_feature_columns() if c in featured.columns]

    # --- Temporal split ---
    evaluator = AnomalyEvaluator()
    train_df, test_df = evaluator.temporal_split(featured, train_end, test_start)

    if len(test_df) < 10:
        return None, None, None, None, None, None, "Not enough test data after {}. Try a longer period.".format(test_start)

    # --- Point-level labels on TEST set ---
    y_true = evaluator.label_points(
        test_df,
        min_abs_return=min_return,
        min_volume_zscore=min_vol_z,
    )

    # --- Train models on TRAIN set only ---
    train_X = train_df[feature_cols]

    # Naive baseline (no training)
    naive = NaiveBaselineDetector(config)
    naive_result = naive.detect(test_df)
    naive_pred = naive_result["anomaly"].astype(int).values
    naive_scores = naive_result["anomaly_score"].values

    # Statistical (no training, runs on test data)
    stat = StatisticalDetector()
    stat_result = stat.detect(test_df["returns"])
    stat_pred = stat_result["anomaly"].astype(int).values
    stat_scores = stat_result["anomaly_score"].values

    # Isolation Forest — trained on train, predict on test
    iso = IsolationForestDetector(config)
    iso.fit(train_X)
    test_X = test_df[feature_cols]
    iso_result = iso.detect(test_X, feature_cols=feature_cols)
    iso_pred = iso_result["anomaly"].astype(int).values
    iso_scores = iso_result["anomaly_score"].values

    # Ensemble (stat + IF, no autoencoder for speed)
    n = min(len(stat_scores), len(iso_scores))
    ens = EnsembleDetector()
    ens_combined = ens.combine_scores({
        "statistical": stat_scores[-n:],
        "isolation_forest": iso_scores[-n:],
    })
    ens_pred = (ens_combined > 0.5).astype(int)
    y_true_aligned = y_true[-n:]

    predictions = {
        "Naive Baseline": naive_pred[-n:],
        "Statistical": stat_pred[-n:],
        "Isolation Forest": iso_pred[-n:],
        "Ensemble (Stat+IF)": ens_pred,
    }
    scores = {
        "Naive Baseline": naive_scores[-n:],
        "Statistical": stat_scores[-n:],
        "Isolation Forest": iso_scores[-n:],
        "Ensemble (Stat+IF)": ens_combined,
    }

    comparison = evaluator.compare_models(y_true_aligned, predictions, scores)

    roc_data = {}
    for name, s in scores.items():
        roc_data[name] = evaluator.compute_roc(y_true_aligned, s)

    cms = {}
    for name, pred in predictions.items():
        cms[name] = evaluator.compute_confusion_matrix(y_true_aligned, pred)

    split_info = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_end": train_end,
        "test_start": test_start,
        "n_true_anomalies": int(y_true_aligned.sum()),
        "anomaly_rate": float(y_true_aligned.mean()),
    }

    return comparison, roc_data, cms, y_true_aligned, test_df, split_info, None


if st.sidebar.button("Run Benchmark", type="primary"):
    comparison, roc_data, cms, y_true, test_df, split_info, error = run_benchmark(
        ticker, period, min_return, min_vol_z
    )

    if error:
        st.error(error)
    else:
        # --- Split info ---
        st.subheader("Dataset Split")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Size", f"{split_info['train_size']} days")
        c2.metric("Test Size", f"{split_info['test_size']} days")
        c3.metric("True Anomalies (test)", split_info["n_true_anomalies"])
        c4.metric("Anomaly Rate (test)", f"{split_info['anomaly_rate']:.1%}")

        st.caption(
            f"Train: up to {split_info['train_end']} | "
            f"Test: from {split_info['test_start']} | "
            f"Models trained on train set only, evaluated on unseen test set."
        )

        st.markdown("---")

        # --- Metrics table ---
        st.subheader("Performance on Test Set")

        display_cols = ["precision", "recall", "f1", "auc", "n_true_anomalies", "n_predicted_anomalies"]
        available = [c for c in display_cols if c in comparison.columns]

        # Highlight: does any ML model beat the baseline?
        baseline_f1 = comparison.loc["Naive Baseline", "f1"] if "Naive Baseline" in comparison.index else 0
        st.dataframe(
            comparison[available].style.format("{:.3f}", subset=[c for c in available if c not in ["n_true_anomalies", "n_predicted_anomalies"]]),
            use_container_width=True,
        )

        # Baseline comparison callout
        best_ml_f1 = comparison.drop("Naive Baseline", errors="ignore")["f1"].max()
        if best_ml_f1 > baseline_f1:
            delta = best_ml_f1 - baseline_f1
            st.success(f"Best ML model beats naive baseline by +{delta:.3f} F1")
        elif best_ml_f1 == baseline_f1:
            st.warning("ML models perform on par with the naive baseline — consider tuning hyperparameters")
        else:
            st.error(
                "Naive baseline outperforms ML models. This is common in anomaly detection — "
                "simple rules often beat complex models when the signal is in obvious features (large returns)."
            )

        st.markdown("---")

        # --- Charts ---
        col1, col2 = st.columns(2)

        with col1:
            fig = model_comparison_bars(comparison)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = roc_curves(roc_data)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- Confusion matrices ---
        st.subheader("Confusion Matrices (Test Set)")
        cm_cols = st.columns(len(cms))
        for i, (name, cm_data) in enumerate(cms.items()):
            with cm_cols[i]:
                fig = confusion_matrix_heatmap(cm_data["matrix"], name)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- Interpretation guide ---
        st.subheader("How to Read These Results")
        st.markdown("""
        **What "good" looks like in anomaly detection:**

        | Metric | Ideal tutorial | Realistic |
        |--------|---------------|-----------|
        | AUC | 0.95+ | 0.55 - 0.75 |
        | Precision | 0.90+ | 0.10 - 0.40 |
        | Recall | 0.95+ | 0.30 - 0.70 |
        | F1 | 0.92+ | 0.15 - 0.45 |

        **Why precision is low:** Anomalies are rare (~2-5% of days). Even a good model
        will flag some normal days as anomalous. A precision of 0.20 means 1 in 5
        flagged days is a true anomaly — that's actually useful if it catches crashes early.

        **Why the baseline might win:** The naive baseline flags big moves directly,
        which is exactly what our labels measure. ML models can only add value by
        catching anomalies the baseline misses (e.g., unusual volume without price move)
        or by having fewer false positives.

        **The honest takeaway:** If your ML model consistently beats the baseline across
        multiple tickers and time periods, it's learning real patterns. If not, the
        simple rule is the better tool.
        """)

else:
    st.info("Select a ticker and period, then click **Run Benchmark** to compare models.")
    st.markdown("""
    This page evaluates anomaly detection models with:
    - **Temporal train/test split** — no look-ahead bias
    - **Point-level ground truth** — individual days labeled by actual market behavior
    - **Naive baseline comparison** — proves whether ML adds value
    """)
