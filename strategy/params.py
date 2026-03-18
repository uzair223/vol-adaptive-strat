"""
strategy/params.py
──────────────────
StrategyParams dataclass — single source of truth for all tunable knobs.

  • Loaded from config.yaml on startup.
  • Can be patched at runtime via the control socket (hot-reload).
  • All bar-count properties are derived lazily so they stay consistent
    when raw day-unit params are updated mid-run.
"""

import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import yaml

# ── Calendar constants (match backtest notebook) ──────────────────────────────
YEAR:          int   = 252
MONTH:         int   = 21
WEEK:          int   = 5
HOURS_PER_DAY: float = 6.5

_TF_HOURS: dict[str, float] = {
    "1Hour": 1.0,
    "30Min": 0.5,
    "15Min": 0.25,
    "5Min":  5 / 60,
    "1Min":  1 / 60,
}

StrategyType = Literal["combined", "mean_reversion", "trend"]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StrategyParams:
    # ── Market ────────────────────────────────────────────────────────────────
    symbol:        str          = "SPY"
    timeframe:     str          = "1Hour"
    strategy_type: StrategyType = "combined"
    long_only:     bool         = False

    # ── Vol regime ────────────────────────────────────────────────────────────
    band_std_dev:         float = 0.8
    ema_span:             int   = WEEK
    zscore_window:        int   = MONTH
    vol_regime_threshold: float = 0.0
    use_stress_flag:      bool  = True

    # ── VWAP anchor ───────────────────────────────────────────────────────────
    anchor_recompute: float = 1.0    # trading days; 0 = cumulative daily VWAP

    # ── Trend following ───────────────────────────────────────────────────────
    trend_ma_window: int = 2 * WEEK  # trading days

    # ── Risk ──────────────────────────────────────────────────────────────────
    atr_period:     float = float(WEEK)
    atr_multiplier: float = 3.0

    # ── Execution ─────────────────────────────────────────────────────────────
    # Fraction of total account equity to deploy per trade (0.0-1.0).
    mr_exposure:      float = 0.25
    tf_exposure:      float = 0.25
    # Allow fractional shares on long entries (Alpaca supports this for longs
    # only -- short orders must be whole shares regardless of this flag).
    allow_fractional: bool  = True

    # ── Derived (read-only) ───────────────────────────────────────────────────
    @property
    def bars_per_day(self) -> float:
        hours = _TF_HOURS.get(self.timeframe)
        if hours is None:
            raise ValueError(
                f"Unknown timeframe '{self.timeframe}'. Valid: {list(_TF_HOURS)}"
            )
        return HOURS_PER_DAY / hours

    @property
    def anchor_recompute_bars(self) -> int:
        return max(0, math.ceil(self.anchor_recompute * self.bars_per_day))

    @property
    def trend_ma_window_bars(self) -> int:
        return math.ceil(self.trend_ma_window * self.bars_per_day)

    @property
    def atr_period_bars(self) -> int:
        return math.ceil(self.atr_period * self.bars_per_day)

    # ── Warmup ────────────────────────────────────────────────────────────────
    @property
    def warmup_bars(self) -> int:
        """Minimum intraday bars needed before signals are reliable."""
        return max(self.trend_ma_window_bars, self.atr_period_bars) + 10

    @property
    def warmup_vix_days(self) -> int:
        """Minimum daily VIX observations needed for z-score warmup."""
        return self.zscore_window + self.ema_span + 5

    # ── Alpaca helpers ────────────────────────────────────────────────────────
    def to_alpaca_timeframe(self) -> "TimeFrame":
        mapping: dict[str, TimeFrame] = {
            "1Hour": TimeFrame.Hour, # type: ignore
            "30Min": TimeFrame(30, TimeFrameUnit.Minute), # type: ignore
            "15Min": TimeFrame(15, TimeFrameUnit.Minute), # type: ignore
            "5Min":  TimeFrame(5,  TimeFrameUnit.Minute), # type: ignore
            "1Min":  TimeFrame(1,  TimeFrameUnit.Minute), # type: ignore
        }
        tf = mapping.get(self.timeframe)
        if tf is None:
            raise ValueError(f"No Alpaca mapping for timeframe '{self.timeframe}'")
        return tf

    def to_alpaca_stream_tf(self) -> str:
        """Bar timeframe string for the Alpaca data stream subscription."""
        mapping: dict[str, str] = {
            "1Hour": "1Hour",
            "30Min": "30Min",
            "15Min": "15Min",
            "5Min":  "5Min",
            "1Min":  "1Min",
        }
        result = mapping.get(self.timeframe)
        if result is None:
            raise ValueError(f"No stream mapping for timeframe '{self.timeframe}'")
        return result

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def patch(self, key: str, raw_value: str) -> 'StrategyParams':
        """
        Return a NEW StrategyParams with one field changed.
        raw_value is a string (from CLI/socket); cast to the correct type.
        Raises ValueError for unknown keys or invalid casts.
        """
        known = {f.name: f for f in fields(self)}
        if key not in known:
            raise ValueError(
                f"Unknown parameter '{key}'. Valid keys: {sorted(known)}"
            )
        field_obj = known[key]
        origin    = field_obj.type
        coerced: Any
        if origin in (int, "int"):
            coerced = int(raw_value)
        elif origin in (float, "float"):
            coerced = float(raw_value)
        elif origin in (bool, "bool"):
            coerced = raw_value.lower() in ("1", "true", "yes", "on")
        else:
            coerced = raw_value

        kwargs: dict[str, Any] = asdict(self)
        kwargs[key] = coerced
        return StrategyParams(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> tuple[StrategyParams, dict[str, Any]]:
    """
    Parse config.yaml.  Returns (StrategyParams, full_config_dict).
    Environment variables APCA_API_KEY_ID / APCA_API_SECRET_KEY override
    the alpaca section if set.
    """
    with open(path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    s = cfg.get("strategy", {})
    valid_keys = {f.name for f in fields(StrategyParams)}
    params = StrategyParams(**{k: v for k, v in s.items() if k in valid_keys})

    alpaca_cfg: dict[str, str] = cfg.setdefault("alpaca", {})
    alpaca_cfg["api_key"]    = os.environ.get("APCA_API_KEY_ID",     alpaca_cfg.get("api_key",    ""))
    alpaca_cfg["secret_key"] = os.environ.get("APCA_API_SECRET_KEY", alpaca_cfg.get("secret_key", ""))

    return params, cfg
