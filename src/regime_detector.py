"""
regime_detector.py
=================
Signal/regime logic layer. Manages state of regime detection and signals.

This layer:
- Maintains rolling windows of market data
- Runs regime detection on new bars
- Exposes current signals to strategy layer
- Keeps regime state persistence
- Serializes models for persistence

This is separate from strategy so strategy only sees signals,
not the complexity of regime detection.
"""

import logging
import joblib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from src.regimehmm import Regime, RegimeHMM
from src.config import RegimeDetectorConfig

logger = logging.getLogger(__name__)

@dataclass
class RegimeSignal:
    """Current market signal: regime and confidence."""
    regime: Regime
    confidence: float
    timestamp: datetime

    def __repr__(self) -> str:
        return (
            f"RegimeSignal(regime={self.regime.name}, "
            f"confidence={self.confidence:.2%}, "
            f"timestamp={self.timestamp})"
        )


@dataclass
class RegimeState:
    """Full signal state for strategy."""
    current_signal: RegimeSignal
    previous_signal: Optional[RegimeSignal]
    bars_in_current_regime: int


class RegimeDetector:
    """
    Manages regime detection and signal generation.

    Maintains:
    - Rolling window of market data
    - Regime detector model (with serialization)
    - Current/previous signal state
    - Tracks retraining times for model staleness monitoring
    """

    def __init__(
        self,
        initial_data: pl.DataFrame,
        config: RegimeDetectorConfig,
        model_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        initial_data : pl.DataFrame
            Historical data with columns: date, vix, vvix, vix3m, spy, gld
        regime_detector_config : RegimeDetectorConfig
            Configuration for regime detector (from config.yaml)
        model_path : Optional[str]
            Path to save/load serialized model (default: models/regime_hmm.pkl)
        """
        self.data = initial_data.clone()
        self.config = config
        self.model_path = Path(model_path) if model_path else None
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Retraining parameters
        self.last_retrain_date: Optional[datetime] = None
        self.retrain_count = 0

        # Initialize regime detector
        self.detector = RegimeHMM(
            n_init=config.n_init,
            n_iter=config.n_iter,
            random_state=config.random_state,
            min_holding_period=config.min_holding_period,
            transition_penalty=config.transition_penalty,
            low_confidence_threshold=config.low_confidence_threshold,
        )

        # Try to load cached model
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading cached model from {self.model_path}")
            try:
                self.detector = joblib.load(self.model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, will retrain")
                self._retrain_model(self.data[-1, "date"])
        else:
            # Initial fit
            logger.info("Initializing regime detector with historical data...")
            self._retrain_model(self.data[-1, "date"])

        # Generate annotations on all data
        self.annotated = self.detector.annotate(self.data, smoothed=False)

        # Extract latest signal
        latest_row = self.annotated.row(-1, named=True)
        self.current_signal = RegimeSignal(
            regime=Regime(latest_row["regime"]),
            confidence=float(latest_row["regime_confidence"]),
            timestamp=latest_row["date"],
        )
        self.previous_signal: Optional[RegimeSignal] = None
        self.bars_in_current_regime = 1

        logger.info(f"Initial signal: {self.current_signal}")

    def _retrain_model(self, retrain_date: datetime) -> None:
        """Retrain the HMM on data window."""
        logger.info(
            f"Retraining regime detector on last {self.config.retrain_window_days} days..."
        )
        # Use sliding window of retrain_window_days
        self.detector.fit(self.data, max_fit_days=self.config.retrain_window_days)
        self.last_retrain_date = retrain_date
        self.retrain_count += 1
        logger.info(
            f"Retraining #{self.retrain_count} completed. "
        )
        
        # Serialize model
        if self.model_path:
            try:
                joblib.dump(self.detector, self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    def on_new_bar(self, bar_data: pl.DataFrame) -> None:
        """
        Process new market bar and update signals.

        Parameters
        ----------
        bar_data : pl.DataFrame
            New bar(s) with columns: date, vix, vvix, vix3m, spy, gld
            (missing values will be forward-filled from previous bar)

        Returns
        -------
        SignalState
            Updated signal state for feeding to strategy
        """
        # Ensure bar_data has all required columns (forward-fill missing from previous bar)
        if bar_data.shape[0] > 0 and len(self.data) > 0:
            last_row_dict = self.data.row(-1, named=True)
            bar_row_dict = bar_data.row(0, named=True)
            
            # Fill in any missing columns or null values from last row
            for col in self.data.columns:
                if col not in bar_row_dict or bar_row_dict[col] is None:
                    bar_row_dict[col] = last_row_dict[col]
            
            # Reconstruct bar_data with all columns in correct order
            bar_data = pl.DataFrame([bar_row_dict]).select(self.data.columns)
        
        # Ensure date columns have matching precision (milliseconds)
        bar_data = bar_data.with_columns(pl.col("date").cast(pl.Datetime("ms")))
        self.data = self.data.with_columns(pl.col("date").cast(pl.Datetime("ms")))
        
        # Append new data
        self.data = pl.concat([self.data, bar_data])

        # Periodically retrain on parameterized interval
        if (
            self.last_retrain_date is None
            or (bar_data[-1, "date"] - self.last_retrain_date).days >= self.config.retrain_interval_days
        ):
            self._retrain_model(bar_data[-1, "date"])

        # Annotate with current model
        self.annotated = self.detector.annotate(self.data, smoothed=False)

        # Extract latest signal
        latest_row = self.annotated.row(-1, named=True)
        new_regime = Regime(latest_row["regime"])
        new_confidence = float(latest_row["regime_confidence"])

        # Check if regime changed
        if new_regime == self.current_signal.regime:
            self.bars_in_current_regime += 1
        else:
            logger.info(
                f"{latest_row['date']}"
                f" | {self.current_signal.regime.name} to {new_regime.name} "
                f"(conf={new_confidence:.1%}, bars_in_regime={self.bars_in_current_regime})"
            )
            self.previous_signal = self.current_signal
            self.bars_in_current_regime = 1

        self.current_signal = RegimeSignal(
            regime=new_regime,
            confidence=new_confidence,
            timestamp=latest_row["date"],
        )

    def get_state(self) -> RegimeState:
        """Get current signal state without processing new data."""
        return RegimeState(
            current_signal=self.current_signal,
            previous_signal=self.previous_signal,
            bars_in_current_regime=self.bars_in_current_regime,
        )

    def get_data_snapshot(self, *, tail: Optional[int] = None) -> pl.DataFrame:
        """Get full annotated data with regime predictions."""
        if tail is not None:
            return self.annotated.tail(tail).clone()
        return self.annotated.clone()
