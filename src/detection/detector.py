from __future__ import annotations

import pandas as pd
from src.models.statistical import StatisticalDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import LSTMAutoencoder
from src.models.ensemble import EnsembleDetector
from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import Preprocessor
from src.utils.helpers import load_config


class AnomalyDetector:
    """Unified interface for running all anomaly detection models.

    Manages the full pipeline: feature engineering -> model inference -> ensemble.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.feature_engineer = FeatureEngineer(self.config)
        self.preprocessor = Preprocessor(self.config)

        # Initialize detectors
        self.statistical = StatisticalDetector(self.config)
        self.isolation_forest = IsolationForestDetector(self.config)
        self.autoencoder = LSTMAutoencoder(self.config)
        self.ensemble = EnsembleDetector(self.config)

        self._is_fitted = False
        self._feature_cols = self.feature_engineer.get_feature_columns()

    def fit(self, df: pd.DataFrame, verbose: int = 1) -> "AnomalyDetector":
        """Fit all models on training data.

        Args:
            df: Raw OHLCV DataFrame
            verbose: Verbosity level for training output
        """
        # Feature engineering
        featured = self.feature_engineer.engineer(df)

        # Get feature matrix
        available_cols = [c for c in self._feature_cols if c in featured.columns]
        X = featured[available_cols]

        # Fit statistical (no-op)
        self.statistical.fit(X)

        # Fit Isolation Forest
        self.isolation_forest.fit(X)

        # Fit LSTM Autoencoder
        X_scaled = self.preprocessor.scale(X).values
        windows = self.preprocessor.create_windows(X_scaled)
        self.autoencoder.fit(windows, verbose=verbose)

        self._is_fitted = True
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all detectors on data and return combined results.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame with per-model scores and ensemble results
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        # Feature engineering
        featured = self.feature_engineer.engineer(df)
        available_cols = [c for c in self._feature_cols if c in featured.columns]
        X = featured[available_cols]

        # Statistical detection (on returns)
        stat_result = self.statistical.detect(featured["returns"])

        # Isolation Forest detection
        if_result = self.isolation_forest.detect(X, feature_cols=available_cols)

        # LSTM Autoencoder detection
        X_scaled = self.preprocessor.scale(X, fit=False).values
        windows = self.preprocessor.create_windows(X_scaled)
        ae_result = self.autoencoder.detect(windows)

        # Align indices — autoencoder results are shorter due to windowing
        window_size = self.config["preprocessing"]["window_size"]
        aligned_index = featured.index[window_size - 1 :]
        n = len(aligned_index)

        # Build scores dict for ensemble
        scores = {
            "statistical": stat_result["anomaly_score"].iloc[-n:].values,
            "isolation_forest": if_result["anomaly_score"].iloc[-n:].values,
            "autoencoder": ae_result["anomaly_score"].values[:n],
        }

        # Ensemble
        ensemble_result = self.ensemble.detect(scores, index=aligned_index)

        # Add price context
        ensemble_result["close"] = featured["Close"].iloc[-n:].values
        ensemble_result["returns"] = featured["returns"].iloc[-n:].values
        ensemble_result["volume"] = featured["Volume"].iloc[-n:].values

        return ensemble_result

    def detect_single(self, series_buffer: pd.DataFrame) -> dict:
        """Detect anomaly for the latest point given a buffer of recent data.

        Useful for streaming detection where you maintain a rolling window.

        Args:
            series_buffer: Recent OHLCV data (at least window_size + warmup rows)

        Returns:
            Dict with anomaly scores from each model
        """
        result = self.detect(series_buffer)
        if len(result) == 0:
            return {"anomaly": False, "ensemble_score": 0.0}
        last = result.iloc[-1]
        return last.to_dict()
