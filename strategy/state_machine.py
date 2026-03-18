"""
strategy/state_machine.py
-------------------------
Incremental position state machine.

States
------
 0   flat
 1   MR long   (mean-reversion long;  ATR trailing stop)
-1   MR short  (mean-reversion short; ATR trailing stop)
 2   TF long   (trend-following long; vol-regime / counter-trend exit)
-2   TF short  (trend-following short; same exit)
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional

from strategy.params import StrategyParams

from .signals import BarSignals

# Type alias for the four action strings
Action = Literal["enter_long", "enter_short", "exit", "none"]

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderDecision:
    action:     Action         = "none"
    prev_state: int            = 0
    new_state:  int            = 0
    reason:     str            = ""
    stop_price: Optional[float] = None   # for trailing-stop logging

    @property
    def is_trade(self) -> bool:
        return self.action != "none"


# ─────────────────────────────────────────────────────────────────────────────

class StateMachine:
    """
    Bar-by-bar state machine. Trades execute immediately on signal.

    Usage:
        sm = StateMachine(params)

        # warm-up (no orders)
        for bar in history:
            sm.process_bar(bar_open, bar_high, bar_low, signals, warmup=True)

        # live bar (may produce orders)
        decision = sm.process_bar(open_price, high, low, signals)
        if decision.is_trade:
            place_order(decision)
    """

    def __init__(self, params: StrategyParams) -> None:
        self._p: StrategyParams = params
        self.state: int   = 0
        self.hwm:   float = math.nan   # high-water-mark for ATR stop tracking

    # ── Public ────────────────────────────────────────────────────────────────

    def process_bar(
        self,
        open_price: float,
        high:       float,
        low:        float,
        signals:    BarSignals,
        warmup:     bool = False,
    ) -> OrderDecision:
        """
        Process one bar.  Returns the order decision for this bar.
        Pass warmup=True to advance internal state without producing orders.
        """
        if not signals.ready:
            return OrderDecision()

        decision: OrderDecision = self._tick(open_price, high, low, signals)

        if warmup:
            return OrderDecision()
        return decision

    @property
    def position_size(self) -> float:
        """Current net exposure (+1, -1, or 0 scaled by exposure fraction)."""
        lookup: dict[int, float] = {
            0:  0.0,
            1:  self._p.mr_exposure,
           -1: -self._p.mr_exposure,
            2:  self._p.tf_exposure,
           -2: -self._p.tf_exposure,
        }
        return lookup.get(self.state, 0.0)

    @property
    def is_flat(self) -> bool:
        return self.state == 0

    def reset(self) -> None:
        """Force-reset to flat (e.g. on manual intervention or params rebuild)."""
        self.state = 0
        self.hwm   = math.nan

    # ── Internal ──────────────────────────────────────────────────────────────

    def _tick(
        self,
        open_: float,
        high:  float,
        low:   float,
        sig:   BarSignals,
    ) -> OrderDecision:
        prev_state: int = self.state

        # ── 1. Exit check for open positions ──────────────────────────────────
        if self.state != 0:
            should_exit: bool
            exit_reason: str
            stop_px:     Optional[float]
            should_exit, exit_reason, stop_px = self._check_exit(high, low, sig)
            if should_exit:
                self.state = 0
                return OrderDecision(
                    action="exit",
                    prev_state=prev_state,
                    new_state=0,
                    reason=exit_reason,
                    stop_price=stop_px,
                )

        # ── 2. Entry signals ───────────────────────────────────────────────────
        if self.state == 0:
            new_state: int = self._evaluate_entries(sig)
            if new_state != 0:
                self.state = new_state
                self.hwm   = open_
                action: Action = "enter_long" if new_state > 0 else "enter_short"
                return OrderDecision(
                    action=action,
                    prev_state=0,
                    new_state=new_state,
                    reason=self._entry_reason(new_state),
                )

        return OrderDecision(prev_state=prev_state, new_state=self.state)

    def _check_exit(
        self,
        high: float,
        low:  float,
        sig:  BarSignals,
    ) -> tuple[bool, str, Optional[float]]:
        """Returns (should_exit, reason, stop_level)."""
        st: int = self.state

        # MR positions -> ATR trailing stop
        if abs(st) == 1:
            stop: float = self._atr_stop_level(sig.atr)
            if math.isnan(stop):
                return False, "", None

            if st == 1:    # MR long
                if low < stop:
                    return True, "atr_stop_long", stop
                self.hwm = max(self.hwm, high)
            else:           # MR short
                if high > stop:
                    return True, "atr_stop_short", stop
                self.hwm = min(self.hwm, low)

            return False, "", stop

        # TF positions -> regime or counter-trend flip
        if st == 2:    # TF long: exit when high vol or trend turns down
            if sig.is_high_vol or sig.trend_down:
                return True, "tf_long_vol_or_reversal", None
        elif st == -2:  # TF short: exit when high vol or trend turns up
            if sig.is_high_vol or sig.trend_up:
                return True, "tf_short_vol_or_reversal", None

        return False, "", None

    def _atr_stop_level(self, atr: float) -> float:
        if math.isnan(atr) or math.isnan(self.hwm):
            return math.nan
        mult: float = self._p.atr_multiplier
        return (self.hwm - atr * mult) if self.state == 1 else (self.hwm + atr * mult)

    def _evaluate_entries(self, sig: BarSignals) -> int:
        """Return new state int (0 if no entry)."""
        if sig.mr_long_entry:  return  1
        if sig.mr_short_entry: return -1
        if sig.tf_long_entry:  return  2
        if sig.tf_short_entry: return -2
        return 0

    @staticmethod
    def _entry_reason(state: int) -> str:
        reasons: dict[int, str] = {
            1: "mr_long", -1: "mr_short",
            2: "tf_long", -2: "tf_short",
        }
        return reasons[state]
