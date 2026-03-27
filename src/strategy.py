"""
strategy.py
===========
Strategy abstraction layer. Defines the interface for trading strategies.

Key principle: Strategies are PURE DECISION MAKERS
- Input: Market signals (regime + confidence) and price data
- Output: Target positions for each asset
- No execution, no state mutation, no trading

This abstraction makes strategies easily swappable. New strategies only need
to implement the Strategy protocol/interface. No changes to orchestrator required.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from src.config import Asset, RegimeDetectorConfig, StrategyConfig
from src.regime_detector import RegimeDetector


@dataclass
class StrategyAction:
    """Strategy's decision: target allocations for next period."""
    allocations: dict[Asset, float]  # {symbol: target_allocation_value}
    reason: str = ""  # For logging/debugging


class Strategy(ABC):
    """
    Abstract base for all trading strategies.

    Subclass this and implement compute_allocations() to create a new strategy.
    
    Principles:
    - Strategies are PURE DECISION MAKERS (no execution, no state mutation)
    
    The orchestrator:
    1. Calls compute_allocations() to get target allocations
    Strategy subclasses can initialize whatever they need in __init__.
    """

    @abstractmethod
    def compute_allocations(self) -> StrategyAction:
        """
        Compute weight allocations given market conditions.
        
        Returns
        -------
        StrategyAction
            Target allocations {symbol: dollars_to_allocate}
        """
        pass
    
    @abstractmethod
    def on_new_bar(self, bar_data: pl.DataFrame):
        """
        Optional hook for strategy to process new market data as it arrives.
        Can be used to update internal state, indicators, etc.

        Parameters
        ----------
        bar_data : pl.DataFrame
            New market data (e.g. latest OHLCV bar)
        """
        pass

# ============================================================================
# Volatility-Adaptive Strategy
# ============================================================================

class VolAdaptiveStrategy(Strategy):
    """
    Volatility-adaptive regime-based strategy.

    Position sizing:
    1. Regime weights (base leverage per regime)
    2. Confidence scaling (reduce by confidence level)
    3. Volatility targeting (scale by vol ratio to maintain target vol)
    4. Position clamping (max leverage bounds)
    
    Instantiates its own regime_detector from initial data.
    """

    def __init__(
        self,
        assets: list[Asset],
        config: StrategyConfig,
        regime_detector_config: RegimeDetectorConfig,
        initial_data: pl.DataFrame,
        hmm_model_path: Optional[str] = None,
    ):
        self.assets = assets
        self.config = config
        self.regime_detector = RegimeDetector(
            initial_data=initial_data,
            config=regime_detector_config,
            model_path=hmm_model_path,
        )
        self.regime_state = self.regime_detector.get_state()
        
    def compute_allocations(self) -> StrategyAction:
        """
        Compute allocations using regime and volatility.
        """
        # Get current signal from own signal manager
        if not self.regime_detector:
            return StrategyAction(
                allocations={},
                reason="Signal manager not initialized (no regime data available)",
            )
        
        signal_state = self.regime_detector.get_state()
        regime = signal_state.current_signal.regime
        confidence = signal_state.current_signal.confidence
        data = self.regime_detector.get_data_snapshot(tail=252)

        # Normalize confidence to [0, 1] over [conf_min, 1].
        # This matches the notebook backtest behavior where low-confidence
        # signals are de-emphasized instead of hard-clipped to a floor.
        conf_span = max(1 - self.config.conf_min, 1e-9)
        conf_scale = float(np.clip((confidence - self.config.conf_min) / conf_span, 0.0, 1.0))

        # Calculate per-asset raw allocations first, then normalize at
        # portfolio level to enforce total gross exposure cap.
        raw_allocations: dict[Asset, float] = {}

        # vol_scale_cap is derived from the max base weight across ALL assets
        # and ALL regimes (not just the current regime), matching backtest behavior.
        max_base_weight = max(
            abs(w) for asset in self.assets for w in asset.regime_weights.values()
        )
        vol_scale_cap = (
            self.config.max_exposure / max_base_weight if max_base_weight > 0 else 1.0
        )
        
        for asset in self.assets:
            close_col = f"{asset.name}_close"
            if close_col not in data:
                raw_allocations[asset] = 0.0
                continue

            asset_data = data.select(close_col).drop_nulls()

            if len(asset_data) < max(self.config.vol_lookback + 1, 2):
                # Not enough data - neutral position
                raw_allocations[asset] = 0.0
                continue

            # Calculate annualized realized volatility using trailing window.
            returns = asset_data[close_col].log().diff().drop_nulls()
            if len(returns) < self.config.vol_lookback:
                raw_allocations[asset] = 0.0
                continue
            rv_ann = returns.tail(self.config.vol_lookback).std() * np.sqrt(252)
            rv_ann = max(float(rv_ann), self.config.vol_floor)

            # Vol targeting scale
            vol_scale = min(self.config.target_vol / rv_ann, vol_scale_cap)

            # Compute exposure for this asset
            leveraged_position = (
                self.config.base_leverage
                * asset.regime_weights.get(regime.name, 0)
                * conf_scale
                * vol_scale
            )
  
            raw_allocations[asset] = float(leveraged_position)

        # Normalize to portfolio-level gross exposure cap. Proportional scaling
        # preserves relative allocations — no additional per-asset clip.
        gross_exposure = float(sum(abs(v) for v in raw_allocations.values()))
        scale_factor = min(1.0, self.config.max_exposure / gross_exposure) if gross_exposure > 0 else 1.0

        allocations: dict[Asset, float] = {
            asset: raw * scale_factor for asset, raw in raw_allocations.items()
        }

        return StrategyAction(
            allocations=allocations,
            reason=(
                f"{regime.name=}, "
                f"{confidence=:.2%}, "
                f"{conf_scale=:.2%}, "
                f"{gross_exposure=:.2f}x, "
                f"{scale_factor=:.2f}"
            ),
        )     
    def on_new_bar(self, bar_data: pl.DataFrame):
        """Process new market data. Update regime detector with new data."""
        self.regime_detector.on_new_bar(bar_data)