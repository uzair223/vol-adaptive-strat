"""
data.py
============
Abstraction:
- DataProvider: Abstract interface for data sources (Alpaca, websocket, file, etc.)
- AlpacaDataProvider: Concrete implementation using Alpaca API
- BarCloseListener: Fires on_bar_close callback at market close
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time, date
import logging
from typing import Callable, Optional
import pytz

import pandas as pd
import polars as pl

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import yfinance as yf
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    
logger = logging.getLogger(__name__)


# ============================================================================
# Types and Data Structures
# ============================================================================

@dataclass
class BarData:
    """Single bar of OHLCV data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ============================================================================
# Abstract Data Provider Interface
# ============================================================================

class DataProvider(ABC):
    """Abstract interface for market data sources."""

    @abstractmethod
    def get_historical_bars(
        self,
        symbols: list[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Fetch historical daily closing prices.

        Returns
        -------
        pl.DataFrame
            Columns: date, symbol, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> BarData:
        """Get latest bar for a single symbol."""
        pass

    @abstractmethod
    def get_latest_bars(self, symbols: list[str]) -> dict[str, BarData]:
        """Get latest bars for multiple symbols."""
        pass


# ============================================================================
# Alpaca Implementation
# ============================================================================

class YahooDataProvider(DataProvider):
    """Fetch market data from Yahoo Finance."""

    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance library is required for YahooDataProvider. Install with `pip install yfinance`.")
        pass
    
    def get_historical_bars(
        self,
        symbols: list[str],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ):
        end = end or datetime.now()
        start = start or (end - pd.Timedelta(days=365))

        result = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )

        if result is None or result.empty:
            raise ValueError("No data downloaded. Check date range and ticker symbols.")

        if isinstance(result.columns, pd.MultiIndex):
            df = (
                result
                .stack(level=1)
                .rename_axis(["date", "symbol"])
                .reset_index()
            )
        else:
            # single ticker case
            symbol = symbols[0] if symbols else None

            df = (
                result
                .reset_index()
                .assign(symbol=symbol)
            )

        return pl.from_dataframe(
            df.rename(columns=str.lower)
        )
        
    def get_latest_bar(self, symbol: str) -> BarData:
        bars = self.get_latest_bars([symbol])
        if symbol not in bars:
            raise ValueError(f"No data for {symbol}")
        return bars[symbol]

    def get_latest_bars(self, symbols: list[str]) -> dict[str, BarData]:
        result = yf.download(
            tickers=symbols,
            start=datetime.now() - pd.Timedelta(days=7),
            end=datetime.now(),
            progress=False,
            auto_adjust=True,
        )

        if result is None or result.empty:
            raise ValueError("No data downloaded. Check ticker symbols.")

        bars: dict[str, BarData] = {}

        last_row = result.iloc[-1]
        timestamp = result.index[-1].to_pydatetime()

        is_multi = isinstance(result.columns, pd.MultiIndex)

        for symbol in symbols:
            try:
                if is_multi:
                    # robust presence check
                    if ("Close", symbol) not in result.columns:
                        logger.warning(f"{symbol} not found in yfinance result")
                        continue

                    close_val = last_row[("Close", symbol)]
                    if pd.isna(close_val):
                        logger.warning(f"{symbol} has no valid latest data")
                        continue

                    bar = BarData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=last_row[("Open", symbol)],
                        high=last_row[("High", symbol)],
                        low=last_row[("Low", symbol)],
                        close=close_val,
                        volume=last_row[("Volume", symbol)],
                    )

                else:
                    # single ticker case
                    if not symbols:
                        continue
                    symbol0 = symbols[0]

                    if pd.isna(last_row["Close"]):
                        logger.warning(f"{symbol0} has no valid latest data")
                        continue

                    bar = BarData(
                        symbol=symbol0,
                        timestamp=timestamp,
                        open=last_row["Open"],
                        high=last_row["High"],
                        low=last_row["Low"],
                        close=last_row["Close"],
                        volume=last_row["Volume"],
                    )
                    bars[symbol0] = bar
                    break  # only one symbol possible

                bars[symbol] = bar
            except KeyError:
                logger.warning(f"{symbol} missing fields in yfinance result")
                continue

        return bars