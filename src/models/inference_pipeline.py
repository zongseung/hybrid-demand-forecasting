"""
Inference pipeline for the Hybrid Power Demand Forecaster

Components:
1. Trend model (log-linear regression via statsmodels)
2. Fourier seasonality model (sklearn LinearRegression)
3. Seq2Seq LSTM residual model (PyTorch)

This module loads the trained artifacts produced by `train_pipeline.py`,
reconstructs the full decomposition (trend + seasonality + residual),
and produces multi-step ahead forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from src.models.seq2seq_lstm import UnivariateSeq2SeqLSTM


EXOG_COLUMNS = [
    "hm",
    "ta",
    "weekday",
    "weekend",
    "spring",
    "summer",
    "autoum",
    "winter",
    "is_holiday_dummies",
]

CSV_COLUMN_MAP = {
    "일시": "timestamp",
    "power demand(MW)": "power demand(MW)",
    "hm": "hm",
    "ta": "ta",
    "holiday_name": "holiday_name",
    "weekday": "weekday",
    "weekend": "weekend",
    "spring": "spring",
    "summer": "summer",
    "autoum": "autoum",
    "winter": "winter",
    "is_holiday_dummies": "is_holiday_dummies",
}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _infer_freq(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer frequency (defaults to 1 hour)."""
    freq = pd.infer_freq(index)
    if freq is not None:
        return pd.tseries.frequencies.to_offset(freq)
    # fallback: use median difference
    diffs = index.to_series().diff().dropna()
    if not diffs.empty:
        return diffs.median()
    return pd.Timedelta(hours=1)


def create_exog_features(
    timestamps: pd.DatetimeIndex,
    historical_data: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build exogenous feature matrix for the provided timestamps.

    Priority:
    1. Load exact rows from CSV (treat as forecasted exogenous signals)
    2. Derive deterministic calendar features
    3. Backfill temperature from historical context or seasonal defaults
    """

    df = pd.DataFrame(index=timestamps)

    # Step 1: attempt to load from CSV
    if csv_path:
        try:
            csv_df = pd.read_csv(csv_path)
            if "일시" in csv_df.columns:
                csv_df.rename(columns=CSV_COLUMN_MAP, inplace=True)
            elif "timestamp" not in csv_df.columns:
                csv_df = csv_df.rename(columns={"timestamp": "timestamp"})
            csv_df["timestamp"] = pd.to_datetime(csv_df["timestamp"])
            csv_df.set_index("timestamp", inplace=True)

            matching = csv_df.reindex(timestamps.intersection(csv_df.index))
            if not matching.empty:
                df = df.combine_first(matching[EXOG_COLUMNS])
                print(
                    f"  ℹ️  Loaded {len(matching)}/{len(timestamps)} exogenous rows from CSV"
                )
        except Exception as exc:
            print(f"  ⚠️  Could not load exogenous features from CSV: {exc}")

    # Step 2: deterministic calendar features
    if "hm" not in df:
        df["hm"] = timestamps.hour + timestamps.minute / 60.0
    if "weekday" not in df:
        df["weekday"] = (timestamps.dayofweek < 5).astype(int)
    if "weekend" not in df:
        df["weekend"] = (timestamps.dayofweek >= 5).astype(int)

    month = timestamps.month
    if "spring" not in df:
        df["spring"] = ((month >= 3) & (month <= 5)).astype(int)
    if "summer" not in df:
        df["summer"] = ((month >= 6) & (month <= 8)).astype(int)
    if "autoum" not in df:
        df["autoum"] = ((month >= 9) & (month <= 11)).astype(int)
    if "winter" not in df:
        df["winter"] = ((month == 12) | (month <= 2)).astype(int)

    if "is_holiday_dummies" not in df:
        try:
            import holidays

            years = timestamps.year.unique()
            kr_holidays = holidays.SouthKorea(years=years)
            df["is_holiday_dummies"] = timestamps.map(
                lambda x: 1 if x.date() in kr_holidays else 0
            )
        except Exception:
            df["is_holiday_dummies"] = 0

    # Step 3: temperature fallback
    if "ta" not in df:
        if (
            historical_data is not None
            and "ta" in historical_data.columns
            and not historical_data["ta"].dropna().empty
        ):
            df["ta"] = historical_data["ta"].tail(24 * 7).mean()
        else:
            seasonal_defaults = {
                "spring": 14.0,
                "summer": 26.0,
                "autoum": 16.0,
                "winter": 2.0,
            }

            def _seasonal_temp(row):
                for season, temp in seasonal_defaults.items():
                    if row.get(season, 0) == 1:
                        return temp
                return 15.0

            df["ta"] = df.apply(_seasonal_temp, axis=1)

    # Ensure all columns exist and ordered
    for col in EXOG_COLUMNS:
        if col not in df:
            df[col] = 0
    return df[EXOG_COLUMNS]


@dataclass
class ForecastComponents:
    trend: np.ndarray
    seasonality: np.ndarray
    residual: np.ndarray


class PowerDemandForecaster:
    """Load trained artifacts and perform hybrid forecasting."""

    def __init__(self, model_dir: str = "models/production"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.trend_model = _load_pickle(self.model_dir / "trend_model.pkl")
        self.fourier_model = _load_pickle(self.model_dir / "fourier_model.pkl")

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("Missing config.json in model directory")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.window_size = self.config.get("window_size", 168)
        self.horizon = self.config.get("horizon", 24)
        self.Kd = self.config.get("Kd", 3)
        self.Kw = self.config.get("Kw", 13)
        self.Ky = self.config.get("Ky", 3)
        self.train_data_length = self.config.get("train_data_length", 0)

        residual_scaler_path = self.model_dir / "residual_scaler.pkl"
        self.residual_scaler = (
            _load_pickle(residual_scaler_path) if residual_scaler_path.exists() else None
        )

        exog_scaler_path = self.model_dir / "exog_scaler.pkl"
        self.exog_scaler = (
            _load_pickle(exog_scaler_path) if exog_scaler_path.exists() else None
        )

        lstm_state_path = self.model_dir / "lstm_model.pth"
        self.lstm_params = self.config.get("lstm_params", {})
        if lstm_state_path.exists() and self.lstm_params:
            self.lstm_model = UnivariateSeq2SeqLSTM(
                hidden_size=self.lstm_params.get("hidden_size", 128),
                num_layers=self.lstm_params.get("num_layers", 2),
                dropout=self.lstm_params.get("dropout", 0.2),
                output_size=self.horizon,
                bidirectional=self.lstm_params.get("bidirectional", False),
                use_attention=self.lstm_params.get("use_attention", False),
            )
            state_dict = torch.load(lstm_state_path, map_location="cpu")
            self.lstm_model.load_state_dict(state_dict)
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.lstm_model = self.lstm_model.to(self.device)
            self.lstm_model.eval()
        else:
            self.lstm_model = None
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def generate_fourier_terms(
        self, t_idx: np.ndarray, period: int, harmonics: int
    ) -> np.ndarray:
        """Generate sin/cos terms for given period and harmonics."""
        features = []
        for k in range(1, harmonics + 1):
            features.append(np.sin(2 * np.pi * k * t_idx / period))
            features.append(np.cos(2 * np.pi * k * t_idx / period))
        if not features:
            return np.zeros((len(t_idx), 0))
        return np.column_stack(features)

    def _build_fourier_matrix(self, t_idx: np.ndarray) -> np.ndarray:
        daily = self.generate_fourier_terms(t_idx, 24, self.Kd)
        weekly = self.generate_fourier_terms(t_idx, 24 * 7, self.Kw)
        yearly = self.generate_fourier_terms(t_idx, int(24 * 365.25), self.Ky)
        parts = [arr for arr in [daily, weekly, yearly] if arr.size > 0]
        if not parts:
            return np.zeros((len(t_idx), 0))
        return np.hstack(parts)

    def _prepare_residual_window(self, residuals: np.ndarray) -> torch.Tensor:
        if self.residual_scaler is None:
            raise ValueError("Residual scaler not found; cannot scale residuals.")
        if len(residuals) < self.window_size:
            raise ValueError(
                f"Not enough historical residuals ({len(residuals)}) for window_size={self.window_size}"
            )
        residual_scaled = self.residual_scaler.transform(
            residuals.reshape(-1, 1)
        ).flatten()
        window = residual_scaled[-self.window_size :]
        tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1)
        return tensor.to(self.device)

    # ------------------------------------------------------------------ #
    # Main API
    # ------------------------------------------------------------------ #
    def forecast(
        self,
        historical_data: pd.DataFrame,
        horizon: Optional[int] = None,
        csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Produce hybrid forecasts for the next `horizon` steps.

        Args:
            historical_data: DataFrame containing the latest context window
                (must include `power demand(MW)` column).
            horizon: Forecast horizon (defaults to model horizon).
            csv_path: Optional CSV for future exogenous variables.
        """

        if "power demand(MW)" not in historical_data.columns:
            raise ValueError("historical_data must contain 'power demand(MW)' column.")

        horizon = horizon or self.horizon
        df = historical_data.copy().sort_index()

        freq = _infer_freq(df.index)
        future_index = pd.date_range(
            start=df.index[-1] + freq, periods=horizon, freq=freq, tz=df.index.tz
        )

        # trend component
        history_start_idx = self.train_data_length
        t_hist = np.arange(history_start_idx, history_start_idx + len(df))
        t_future = np.arange(history_start_idx + len(df), history_start_idx + len(df) + horizon)

        X_hist = sm.add_constant(t_hist)
        X_future = sm.add_constant(t_future)

        trend_hist = np.exp(self.trend_model.predict(X_hist))
        trend_future = np.exp(self.trend_model.predict(X_future))

        df["trend"] = trend_hist
        df["detrend"] = df["power demand(MW)"].values - trend_hist

        # seasonality component
        fourier_hist = self._build_fourier_matrix(t_hist)
        fourier_future = self._build_fourier_matrix(t_future)

        exog_hist = create_exog_features(df.index, historical_data=df, csv_path=csv_path)
        exog_future = create_exog_features(
            future_index, historical_data=df, csv_path=csv_path
        )

        if self.exog_scaler is not None and not exog_hist.empty:
            X_hist_full = np.hstack([fourier_hist, self.exog_scaler.transform(exog_hist.values)])
            X_future_full = np.hstack(
                [fourier_future, self.exog_scaler.transform(exog_future.values)]
            )
        else:
            X_hist_full = fourier_hist
            X_future_full = fourier_future

        seasonality_hist = self.fourier_model.predict(X_hist_full)
        seasonality_future = self.fourier_model.predict(X_future_full)
        df["seasonality"] = seasonality_hist
        df["residual"] = df["detrend"] - df["seasonality"]

        # residual forecast via LSTM
        if self.lstm_model is None or self.residual_scaler is None:
            residual_forecast = np.zeros(horizon)
            print("  ⚠️  LSTM model missing; residual forecast will be zeros.")
        else:
            residual_tensor = self._prepare_residual_window(df["residual"].values)
            with torch.no_grad():
                pred_scaled = self.lstm_model(residual_tensor)
            pred_scaled = pred_scaled.squeeze(0).squeeze(-1).cpu().numpy()[:horizon]
            residual_forecast = self.residual_scaler.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).flatten()

        # compose final forecast
        forecast_values = trend_future + seasonality_future + residual_forecast
        result = pd.DataFrame(
            {
                "trend": trend_future,
                "seasonality": seasonality_future,
                "residual": residual_forecast,
                "forecast": forecast_values,
            },
            index=future_index,
        )

        return result