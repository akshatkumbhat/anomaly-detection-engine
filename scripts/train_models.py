"""Train all anomaly detection models and save artifacts."""

import sys
import os
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import LSTMAutoencoder
from src.utils.helpers import load_config, ensure_dir


def main():
    config = load_config()
    ticker = config["data"]["default_ticker"]
    model_dir = ensure_dir("models/saved")

    print(f"Training models on {ticker}...\n")

    # Load data
    fetcher = DataFetcher(config)
    df = fetcher.fetch(ticker=ticker)
    fe = FeatureEngineer(config)
    featured = fe.engineer(df)
    feature_cols = [c for c in fe.get_feature_columns() if c in featured.columns]
    X = featured[feature_cols]

    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features\n")

    # --- Statistical (no training needed) ---
    print("1. Statistical Detector — no training needed ✓")

    # --- Isolation Forest ---
    print("2. Training Isolation Forest...")
    iso = IsolationForestDetector(config)
    iso.fit(X)
    iso_path = os.path.join(model_dir, "isolation_forest.pkl")
    with open(iso_path, "wb") as f:
        pickle.dump(iso.model, f)
    print(f"   Saved to {iso_path} ✓")

    # --- LSTM Autoencoder ---
    print("3. Training LSTM Autoencoder...")
    preprocessor = Preprocessor(config)
    X_scaled = preprocessor.scale(X).values
    windows = preprocessor.create_windows(X_scaled)
    print(f"   Windows: {windows.shape}")

    ae = LSTMAutoencoder(config)
    ae.fit(windows, verbose=1)
    ae_path = os.path.join(model_dir, "lstm_autoencoder.keras")
    ae.save(ae_path)
    print(f"   Threshold: {ae.threshold:.6f}")
    print(f"   Saved to {ae_path} ✓")

    print("\nAll models trained successfully!")
    print(f"Artifacts saved to {model_dir}/")


if __name__ == "__main__":
    main()
