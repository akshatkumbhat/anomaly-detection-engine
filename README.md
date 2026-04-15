# Real-Time Anomaly Detection Engine for Financial Markets

A multi-model anomaly detection system that identifies unusual patterns in financial time-series data (flash crashes, volume spikes, unusual volatility) using statistical methods, machine learning, and deep learning — with an interactive Streamlit dashboard for real-time monitoring and model benchmarking.

Built as an advanced data science project for Computational Modeling and Data Analytics.

---

## Architecture

```
                          Yahoo Finance API
                                │
                        ┌───────▼───────┐
                        │  Data Fetcher  │
                        │  (yfinance)    │
                        └───────┬───────┘
                                │
                        ┌───────▼───────┐
                        │   Feature      │
                        │   Engineering  │
                        │  RSI, MACD,    │
                        │  Bollinger,    │
                        │  ATR, VWAP,    │
                        │  Vol Z-score   │
                        └───────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
     ┌────────▼──────┐  ┌──────▼───────┐  ┌───────▼───────┐
     │  Statistical   │  │  Isolation   │  │    LSTM       │
     │  Z-score +     │  │  Forest      │  │  Autoencoder  │
     │  EWMA Control  │  │  (sklearn)   │  │  (Keras)      │
     └────────┬──────┘  └──────┬───────┘  └───────┬───────┘
              │                │                   │
              └────────┬───────┘───────────────────┘
                       │
              ┌────────▼────────┐
              │ Weighted Ensemble│
              │ + Naive Baseline │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   Streamlit      │
              │   Dashboard      │
              │  - Real-time     │
              │  - Comparison    │
              │  - Explorer      │
              └─────────────────┘
```

## Detection Models

| Model | Method | What It Catches |
|-------|--------|-----------------|
| **Statistical** | Z-score + EWMA control charts | Sudden deviations from recent behavior |
| **Isolation Forest** | Random partitioning of feature space | Multi-dimensional outliers (price + volume + volatility) |
| **LSTM Autoencoder** | Reconstruction error on sequences | Temporal pattern breaks the model hasn't seen before |
| **Ensemble** | Weighted voting across all models | Higher confidence detections via consensus |
| **Naive Baseline** | Simple return/volume thresholds | Sanity check — ML models must beat this to prove value |

## Honest Evaluation

This project deliberately avoids the inflated metrics common in anomaly detection tutorials:

- **Temporal train/test split** — models train on historical data only, tested on unseen future data. No look-ahead bias.
- **Point-level labeling** — individual days are labeled anomalous based on actual behavior (|return| > 3% or volume z-score > 3), not hand-picked date ranges.
- **Naive baseline comparison** — a simple threshold rule is included. If ML models can't beat it, they aren't adding value.
- **Contamination set to `auto`** — no information leakage about anomaly rates.

Expected realistic metrics: AUC 0.55–0.75, F1 0.15–0.45. If you see 0.95+, something is wrong.

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/akshatkumbhat/anomaly-detection-engine.git
cd Personal-Project-1
pip install -r requirements.txt
```

### Download Data

```bash
python scripts/download_data.py
```

Downloads OHLCV data for SPY, AAPL, MSFT, and BTC-USD from Yahoo Finance (cached locally).

### Train Models (optional)

```bash
python scripts/train_models.py
```

Trains the LSTM Autoencoder. Requires TensorFlow. The dashboard works without this step using Statistical + Isolation Forest only.

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Dashboard Pages

### Real-Time Detection
Stream historical data point-by-point and watch anomalies get flagged live. Configure ticker, speed, and which models are active.

### Model Comparison
Benchmark all models against each other with honest evaluation — temporal split, point-level labels, baseline comparison, ROC curves, confusion matrices, and an interpretation guide.

### Historical Explorer
Select any ticker and time period, adjust detection sensitivity, and explore flagged anomalies in an interactive chart.

## Project Structure

```
├── config/config.yaml          # Hyperparameters, thresholds, evaluation settings
├── src/
│   ├── data/                   # Fetcher, preprocessor, feature engineer
│   ├── models/                 # Statistical, Isolation Forest, LSTM AE, Ensemble, Baseline
│   ├── detection/              # Unified detector, stream simulator
│   └── evaluation/             # Metrics with honest evaluation framework
├── dashboard/                  # Streamlit app with 3 pages
├── scripts/                    # Data download and model training CLIs
├── tests/                      # 25 unit tests (pytest)
└── notebooks/                  # EDA, experiments, evaluation (WIP)
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Data | yfinance, pandas, numpy |
| ML | scikit-learn, TensorFlow/Keras |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Testing | pytest (25 tests) |

## Research Papers

### Core Methods
- Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008) — [Isolation Forest](https://ieeexplore.ieee.org/document/4781136). Foundational tree-based anomaly detection algorithm.
- Malhotra, P. et al. (2016) — [LSTM-based Encoder-Decoder for Multi-Sensor Anomaly Detection](https://arxiv.org/abs/1607.00148). Architecture used for the autoencoder model.
- [Combination of Isolation Forest and LSTM Autoencoder for Anomaly Detection](https://ieeexplore.ieee.org/document/9590143/) — validates the ensemble approach.

### Surveys
- [Deep Learning for Time Series Anomaly Detection: A Survey (ACM 2024)](https://dl.acm.org/doi/10.1145/3691338)
- [Survey of Deep Anomaly Detection in Multivariate Time Series (2025)](https://www.mdpi.com/1424-8220/25/1/190)
- [Open Challenges in Time Series Anomaly Detection (2025)](https://arxiv.org/html/2502.05392v1)

### Benchmarking
- [Evaluating Reconstruction vs Proximity Methods: AE, LSTM-AE, OCSVM, IF (2025)](https://www.mdpi.com/1999-5903/18/2/96)
- [Hybrid Framework: IF + Autoencoder + ConvLSTM (2025)](https://link.springer.com/article/10.1007/s10115-025-02580-6)

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

See [LICENSE](LICENSE) for details.
