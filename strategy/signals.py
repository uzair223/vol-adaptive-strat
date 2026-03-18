"""
strategy/signals.py
-------------------
Incremental, bar-by-bar signal engine that replicates the backtest logic
without ever looking ahead.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from strategy.params import StrategyParams

YEAR: int = 252


# ─────────────────────────────────────────────────────────────────────────────
# Primitive rolling accumulators
# ─────────────────────────────────────────────────────────────────────────────

class _EMA:
    """
    Exponential Moving Average using alpha = 2/(span+1),
    matching Polars ewm_mean(span=span, adjust=False).
    """
    __slots__ = ("alpha", "_val")

    def __init__(self, span: int) -> None:
        self.alpha: float = 2.0 / (span + 1)
        self._val: Optional[float] = None

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = self.alpha * x + (1.0 - self.alpha) * self._val
        return self._val  # type: ignore[return-value]

    @property
    def value(self) -> Optional[float]:
        return self._val

    @property
    def ready(self) -> bool:
        return self._val is not None


class _CausalZScore:
    """
    Causal rolling z-score: z[i] = (x[i] - mean(window ending at i-1)) / std(...)
    Excludes the current observation from its own mean/std calculation.
    """
    __slots__ = ("window", "_buf")

    def __init__(self, window: int) -> None:
        self.window: int = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, x: float) -> float:
        """Return z-score of x vs past window. NaN until window is full."""
        if len(self._buf) < self.window:
            self._buf.append(x)
            return math.nan

        arr: NDArray[np.float64] = np.asarray(self._buf, dtype=np.float64)
        mu:    float = float(arr.mean())
        sigma: float = float(arr.std(ddof=0))
        z: float = (x - mu) / sigma if sigma > 0.0 else 0.0
        self._buf.append(x)
        return z

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window


class _TrailingMean:
    """Simple causal trailing mean over the last `window` observations."""
    __slots__ = ("window", "_buf")

    def __init__(self, window: int) -> None:
        self.window: int = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, x: float) -> float:
        self._buf.append(x)
        return float(np.mean(self._buf)) if len(self._buf) >= self.window else math.nan

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window


class _RollingATR:
    """
    True Range = max(H-L, |H-prev_C|, |L-prev_C|).
    Rolling mean of TR over `window` bars.
    """
    __slots__ = ("window", "_buf", "_prev_close")

    def __init__(self, window: int) -> None:
        self.window:     int            = window
        self._buf:       deque[float]   = deque(maxlen=window)
        self._prev_close: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_close is not None:
            tr: float = max(
                high - low,
                abs(high - self._prev_close),
                abs(low  - self._prev_close),
            )
        else:
            tr = high - low
        self._buf.append(tr)
        self._prev_close = close
        return float(np.mean(self._buf)) if len(self._buf) >= self.window else math.nan

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.window


class _VWAPAnchor:
    """
    Intraday VWAP with optional block resets.

    anchor_recompute_bars == 0  ->  cumulative daily VWAP (reset at session open)
    anchor_recompute_bars  > 0  ->  rolling block VWAP, reset every N bars

    Returns the PREVIOUS bar's VWAP as the anchor for the current bar.
    """
    __slots__ = ("_arb", "_cum_pv", "_cum_v",
                 "_blk_pv", "_blk_v", "_bar", "_prev_vwap")

    def __init__(self, anchor_recompute_bars: int) -> None:
        self._arb:       int            = anchor_recompute_bars
        self._cum_pv:    float          = 0.0
        self._cum_v:     float          = 0.0
        self._blk_pv:    float          = 0.0
        self._blk_v:     float          = 0.0
        self._bar:       int            = 0
        self._prev_vwap: Optional[float] = None

    def new_session(self) -> None:
        """Reset cumulative accumulators at market open."""
        self._cum_pv = self._cum_v = 0.0
        self._blk_pv = self._blk_v = 0.0
        self._bar = 0
        # Keep _prev_vwap so the first bar of the session has a valid anchor

    def update(self, close: float, volume: float) -> float:
        """Feed one bar. Returns the ANCHOR price (prev-bar VWAP, or close if none yet)."""
        anchor: float = self._prev_vwap if self._prev_vwap is not None else close

        if self._arb > 0 and self._bar > 0 and self._bar % self._arb == 0:
            self._blk_pv = self._blk_v = 0.0

        vol: float = max(volume, 0.0)
        self._cum_pv += close * vol
        self._cum_v  += vol
        self._blk_pv += close * vol
        self._blk_v  += vol

        if self._arb > 0:
            self._prev_vwap = (
                self._blk_pv / self._blk_v if self._blk_v > 0 else close
            )
        else:
            self._prev_vwap = (
                self._cum_pv / self._cum_v if self._cum_v > 0 else close
            )

        self._bar += 1
        return anchor


# ─────────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """Daily VIX-derived regime snapshot passed into each bar update."""
    vix_ema:       float = math.nan
    vol_zscore:    float = math.nan
    term_slope:    float = math.nan
    term_slope_ma: float = math.nan
    vol_d:         float = math.nan   # per-bar vol fraction: vix_ema/100/sqrt(YEAR)
    stress_flag:   bool  = False
    is_high_vol:   bool  = False
    ready:         bool  = False


@dataclass
class BarSignals:
    """All signals produced after processing one intraday bar."""
    # Levels
    close:       float = math.nan
    anchor:      float = math.nan
    upper_band:  float = math.nan
    lower_band:  float = math.nan
    atr:         float = math.nan
    trend_ma:    float = math.nan
    trend_slope: float = math.nan
    # Boolean indicators
    trend_up:    bool = False
    trend_down:  bool = False
    cross_upper: bool = False
    cross_lower: bool = False
    # Regime pass-through
    is_high_vol:  bool  = False
    stress_flag:  bool  = False
    vol_zscore:   float = math.nan
    # Raw entry signals
    mr_long_entry:  bool = False
    mr_short_entry: bool = False
    tf_long_entry:  bool = False
    tf_short_entry: bool = False
    # Whether all warmup requirements are satisfied
    ready: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Vol-regime tracker (daily cadence)
# ─────────────────────────────────────────────────────────────────────────────

class VolRegimeEngine:
    """
    Maintains incremental VIX regime state.
    Call update() once per trading day (at market open or after VIX close).
    """

    def __init__(self, ema_span: int, zscore_window: int) -> None:
        self._ema:      _EMA          = _EMA(ema_span)
        self._zscore:   _CausalZScore = _CausalZScore(zscore_window)
        self._slope_ma: _TrailingMean = _TrailingMean(21)
        self.state:     RegimeState   = RegimeState()

    def update(
        self,
        vix:                   float,
        vix3m:                 float,
        vol_regime_threshold:  float,
    ) -> RegimeState:
        vix_ema:    float = self._ema.update(vix)
        vol_zscore: float = self._zscore.update(vix_ema)
        term_slope: float = vix3m - vix
        slope_ma:   float = self._slope_ma.update(term_slope)
        vol_d:      float = vix_ema / 100.0 * math.sqrt(1.0 / YEAR)
        ready:      bool  = not math.isnan(vol_zscore)

        self.state = RegimeState(
            vix_ema       = vix_ema,
            vol_zscore    = vol_zscore,
            term_slope    = term_slope,
            term_slope_ma = slope_ma,
            vol_d         = vol_d,
            stress_flag   = (term_slope < 0),
            is_high_vol   = (ready and vol_zscore > vol_regime_threshold),
            ready         = ready,
        )
        return self.state


# ─────────────────────────────────────────────────────────────────────────────
# Main signal engine
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Wraps ALL incremental signal state for one symbol.

    Typical usage:
        engine = SignalEngine(params)

        # warm-up phase (no orders placed)
        for date, vix, vix3m in historical_vix_rows:
            engine.update_vix(vix, vix3m)
        for bar in historical_ohlcv_bars:
            engine.update_bar(bar_date, is_new_session, open_, high, low, close, volume)

        # live bar
        signals = engine.update_bar(...)
        if signals.ready:
            # pass to StateMachine
    """

    def __init__(self, params: StrategyParams) -> None:
        self._p:          StrategyParams  = params
        self._vol_regime: VolRegimeEngine = VolRegimeEngine(params.ema_span, params.zscore_window)
        self._vwap:       _VWAPAnchor     = _VWAPAnchor(params.anchor_recompute_bars)
        self._atr:        _RollingATR     = _RollingATR(params.atr_period_bars)
        self._trend_ema:  _EMA            = _EMA(params.trend_ma_window_bars)

        # Cross-bar carry state
        self._prev_above_upper: bool           = False
        self._prev_below_lower: bool           = False
        self._prev_trend_up:    bool           = False
        self._prev_trend_down:  bool           = False
        self._prev_trend_ma:    Optional[float] = None
        self._prev_high:        Optional[float] = None
        self._prev_low:         Optional[float] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def update_vix(self, vix: float, vix3m: float) -> RegimeState:
        """Feed one daily VIX observation. Call once per session."""
        return self._vol_regime.update(vix, vix3m, self._p.vol_regime_threshold)

    def update_bar(
        self,
        date:           str,
        is_new_session: bool,
        open_:          float,
        high:           float,
        low:            float,
        close:          float,
        volume:         float,
    ) -> BarSignals:
        """Feed one OHLCV bar. Returns BarSignals; .ready == False during warmup."""
        sig = BarSignals()

        if is_new_session:
            self._vwap.new_session()

        anchor: float = self._vwap.update(close, volume)
        atr:    float = self._atr.update(high, low, close)
        t_ma:   float = self._trend_ema.update(close)

        regime: RegimeState = self._vol_regime.state

        # Not warm yet -> still update carry state and return empty signals
        if not regime.ready or math.isnan(atr) or math.isnan(t_ma):
            self._carry(high, low, close, t_ma, math.nan, math.nan)
            return sig

        # ── Vol-based bands ───────────────────────────────────────────────────
        vol_move_dist: float = anchor * regime.vol_d * self._p.band_std_dev
        upper_band:    float = anchor + vol_move_dist
        lower_band:    float = anchor - vol_move_dist

        # ── Band re-entry crosses ─────────────────────────────────────────────
        cross_upper: bool
        cross_lower: bool
        if self._prev_high is not None:
            cross_upper = self._prev_above_upper and (high < upper_band)
            cross_lower = self._prev_below_lower and (low  > lower_band)
        else:
            cross_upper = cross_lower = False

        # ── Trend ─────────────────────────────────────────────────────────────
        trend_slope:    float = (t_ma - self._prev_trend_ma) if self._prev_trend_ma is not None else 0.0
        trend_strength: float = close - t_ma
        trend_up:   bool = (trend_slope > 0) and (trend_strength >  atr * 0.5)
        trend_down: bool = (trend_slope < 0) and (trend_strength < -atr * 0.5)

        # ── Raw entry signals ─────────────────────────────────────────────────
        p = self._p
        is_mr: bool = p.strategy_type in {"combined", "mean_reversion"}
        is_tf: bool = p.strategy_type in {"combined", "trend"}

        mr_long_entry: bool = (
            is_mr
            and regime.is_high_vol
            and (not p.use_stress_flag or regime.stress_flag)
            and cross_lower
        )
        mr_short_entry: bool = (
            is_mr
            and not p.long_only
            and regime.is_high_vol
            and (not p.use_stress_flag or regime.stress_flag)
            and cross_upper
        )
        tf_long_entry: bool = (
            is_tf
            and not regime.is_high_vol
            and trend_up
            and not self._prev_trend_up
        )
        tf_short_entry: bool = (
            is_tf
            and not p.long_only
            and not regime.is_high_vol
            and trend_down
            and not self._prev_trend_down
        )

        # ── Populate output ───────────────────────────────────────────────────
        sig.close       = close
        sig.anchor      = anchor
        sig.upper_band  = upper_band
        sig.lower_band  = lower_band
        sig.atr         = atr
        sig.trend_ma    = t_ma
        sig.trend_slope = trend_slope
        sig.trend_up    = trend_up
        sig.trend_down  = trend_down
        sig.cross_upper = cross_upper
        sig.cross_lower = cross_lower
        sig.is_high_vol    = regime.is_high_vol
        sig.stress_flag    = regime.stress_flag
        sig.vol_zscore     = regime.vol_zscore
        sig.mr_long_entry  = mr_long_entry
        sig.mr_short_entry = mr_short_entry
        sig.tf_long_entry  = tf_long_entry
        sig.tf_short_entry = tf_short_entry
        sig.ready = True

        self._carry(high, low, close, t_ma, upper_band, lower_band,
                    trend_up, trend_down)
        return sig

    def rebuild(self, new_params: "StrategyParams") -> None:  # noqa: F821
        """
        Hot-reload structural params that require rebuilding rolling buffers.
        WARNING: resets all warmup state.
        """
        self.__init__(new_params)

    # ── Private ───────────────────────────────────────────────────────────────

    def _carry(
        self,
        high:       float,
        low:        float,
        close:      float,
        trend_ma:   float,
        upper_band: float,
        lower_band: float,
        trend_up:   bool = False,
        trend_down: bool = False,
    ) -> None:
        self._prev_high       = high
        self._prev_low        = low
        self._prev_trend_ma   = trend_ma
        self._prev_trend_up   = trend_up
        self._prev_trend_down = trend_down
        if not math.isnan(upper_band):
            self._prev_above_upper = (high > upper_band)
            self._prev_below_lower = (low  < lower_band)
