"""
trader.py
==============
Main orchestrator for the live trading system.

Coordinates:
- Data fetching and bar open events
- Signal generation (from previous day's close)
- Strategy decision making
- Trade execution
- CLI control

Architecture:
    market_open (daily callback)
       ↓
    fetch_previous_day_bar
       ↓
    strategy (compute positions)
       ↓
    broker (execute trades)

CLI runs in background, can pause/resume and update config.
"""

import logging
import time
import os
import re
from typing import Optional, cast
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta, date
from apscheduler.schedulers.background import BackgroundScheduler

import polars as pl
import numpy as np

from src.data import DataProvider
from src.strategy import Strategy, VolAdaptiveStrategy
from src.broker import Broker
from cli import CliCommandHandler, CliSocketServer
from src.config import Asset, Config
from src.regime_detector import RegimeSignal
from src.market_timing import MarketTiming

logger = logging.getLogger(__name__)

class Trader:
    """
    Main trading orchestrator.

    Lifecycle:
    1. Initialize: Load config, fetch historical data, set up components
    2. Run: Enter event loop, process signals at market open using previous day's data
    3. Shut down: Close positions, save state
    """

    def __init__(self,
                 config: Config,
                 broker: Optional[Broker] = None,
                 data_provider: Optional[DataProvider] = None,
                 strategy: Optional[Strategy] = None,
                ):
        """
        Initialize live trader.

        Parameters
        ----------
        config : Config
            Configuration object
        broker : Broker, optional
            Broker for order execution
        data_provider : DataProvider, optional
            Data provider for market data
        strategy : Strategy, optional
            Trading strategy
        """
        self.config = config

        # Initialize components
        self.broker = broker
        self.data_provider = data_provider
        self.strategy: Optional[Strategy] = strategy
        self.cli_handler = CliCommandHandler()
        
        # Read CLI settings from environment variables (used in Docker/Cloud deployments)
        cli_host = os.getenv("CLI_HOST", "127.0.0.1")
        cli_port = int(os.getenv("CLI_PORT", "9000"))
        cli_timeout = int(os.getenv("CLI_TIMEOUT_SECONDS", "30"))
        
        self.cli_server = CliSocketServer(
            host=cli_host,
            port=cli_port,
            handler=self.cli_handler,
        )
        self.cli_timeout = cli_timeout

        # Fetch historical data
        self.historical_data: Optional[pl.DataFrame] = None

        # Initialize market timing
        self.market_timing = MarketTiming(self.config)

        # Initialize APScheduler for market open events
        self.scheduler = BackgroundScheduler()
        self.market_timing.add_market_open_job(self.scheduler, self._on_market_open_scheduled)

        # Register CLI callbacks
        self.cli_handler.register_callback("pause", self._on_cli_pause)
        self.cli_handler.register_callback("resume", self._on_cli_resume)
        self.cli_handler.register_callback("config_update", self._on_config_update)
        self.cli_handler.register_callback("get_positions", self._on_get_positions)
        self.cli_handler.register_callback("get_signal", self._on_get_signal)
        self.cli_handler.register_callback("get_config", self._on_get_config)

        # State
        self.is_paused = False
        self._last_rebalance_bucket: Optional[tuple[str, int]] = None

        logger.info("Live trader initialized")

    def run(self) -> None:
        """
        Main event loop.

        Continuously checks for market open and processes previous day's data.
        """
        logger.info("Fetching historical data...")
        self.historical_data = self._fetch_historical_data(self.config.trading.assets)
        logger.info(f"Historical data: {len(self.historical_data)} rows")
        
        if self.strategy is None:
            logger.info("Initializing strategy...")
            hmm_model_path = os.getenv("HMM_MODEL_PATH") or "models/regime_hmm.pkl"
            self.strategy = VolAdaptiveStrategy(
                assets=self.config.trading.assets,
                config=self.config.strategy,
                regime_detector_config=self.config.regime_detector,
                initial_data=self.historical_data,
                hmm_model_path=hmm_model_path,
            )
            logger.info("Strategy initialized")
        
        logger.info("Starting scheduler and CLI server...")
        self.scheduler.start()
        
        # Brief delay to allow socket cleanup on Windows after previous connection
        time.sleep(0.5)
        
        self.cli_server.start()
        
        # Check if market is already open and fire event immediately if so
        self._check_and_fire_if_market_open()
        
        logger.info("Trading event loop running (press Ctrl+C to stop)...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.scheduler.shutdown()
        self.cli_server.stop()

        # Close all positions
        try:
            if(self.broker):
                self.broker.close_all_positions()
                closed = self.broker.cancel_open_orders()
                logger.info(f"All positions and orders closed ({closed} orders canceled)")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")

        logger.info("Shutdown complete")

    # ========================================================================
    # Market Data Callbacks
    # ========================================================================

    def _on_market_open_scheduled(self) -> None:
        """Scheduled callback fired by APScheduler at market open time."""
        if self.is_paused:
            logger.debug("Trading paused - skipping scheduled market open event")
            return
        self._on_market_open()

    def _check_and_fire_if_market_open(self) -> None:
        """
        Check if market is already open at startup.
        
        If the app starts during market hours on a trading day, fire the event 
        immediately to process previous day's close data for today's trades.
        """
        if self.market_timing.should_fire_at_startup():
            logger.info(f"Market is open. Firing market open event immediately with previous day's data.")
            if not self.is_paused:
                self._on_market_open()
            else:
                logger.debug("Trading paused - skipping immediate market open event")

    def _on_market_open(self) -> None:
        """Called when market opens each day. Generates trades based on previous day's close."""
        if not self.strategy:
            logger.warning("Strategy not initialized - skipping signal processing")
            return
        
        logger.info("=" * 60)
        logger.info(f"Market open event at {datetime.now()}")
        logger.info("Generating signals from previous day's close data")

        try:
            # Fetch latest bars (which are previous day's close during market open)
            new_data = self._fetch_daily_update(self.config.trading.assets)

            if new_data.is_empty():
                logger.warning("No new data received")
                return

            logger.info(f"Processing {len(new_data)} bar(s)")
            self._on_new_bar(new_data)

        except Exception as e:
            logger.error(f"Error processing market open signal: {e}", exc_info=True)

    def _on_new_bar(self, bar_data: pl.DataFrame) -> None:
        """
        Process new market bar and update strategy state.
        """
        if self.strategy is None:
            logger.warning("Strategy not initialized - skipping bar processing")
            return
        
        if self.broker is None:
            logger.warning("Broker not initialized - skipping bar processing")
            return
        
        if self.historical_data is None:
            logger.warning("Historical data not available - skipping bar processing")
            return
        
        if bar_data.is_empty():
            logger.warning("Empty bar data received - skipping processing")
            return
        
        if np.isnan(bar_data.to_numpy().sum()):
            logger.warning("Bar contains missing values - skipping processing")
            return
        
        # Get portfolio state
        portfolio = self.broker.get_portfolio_state()
        logger.info(f"Portfolio value: ${portfolio.total_value:,.2f}")

        self.strategy.on_new_bar(bar_data)
        action = self.strategy.compute_allocations()

        logger.info(f"Strategy action: {action.reason}")
        target_allocations = {asset.name: f"{alloc:.2f}" for asset, alloc in action.allocations.items()}
        logger.info(f"Target allocations: {target_allocations}")

        # Execute rebalance only on configured frequency buckets.
        if not self._should_rebalance(bar_data):
            logger.info(
                f"Skipping order execution (rebalance_freq={self.config.trading.rebalance_freq})"
            )
            return

        # Execute rebalance
        self.broker.execute_rebalance(action.allocations, self.config.trading.min_rebalance_threshold)

    def _should_rebalance(self, bar_data: pl.DataFrame) -> bool:
        """Return True when current bar falls into a new rebalance bucket."""
        if "date" not in bar_data.columns or bar_data.is_empty():
            return True

        ts_value = bar_data[0, "date"]
        if ts_value is None:
            return True

        if isinstance(ts_value, datetime):
            ts = ts_value
        elif isinstance(ts_value, date):
            ts = datetime.combine(ts_value, datetime.min.time())
        else:
            ts = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))

        bucket = self._rebalance_bucket(ts, self.config.trading.rebalance_freq)
        if self._last_rebalance_bucket is None:
            self._last_rebalance_bucket = bucket
            return True

        should_rebalance = bucket != self._last_rebalance_bucket
        if should_rebalance:
            self._last_rebalance_bucket = bucket
        return should_rebalance

    @staticmethod
    def _rebalance_bucket(ts: datetime, rebalance_freq: str) -> tuple[str, int]:
        """
        Convert timestamp to a rebalance bucket key.

        Supported formats:
        - Nd  (days)
        - Nw  (weeks)
        - Nm / Nmo (months)
        - Nq  (quarters)
        """
        value = (rebalance_freq or "1d").strip().lower()
        match = re.fullmatch(r"(\d+)(d|w|m|mo|q)", value)
        if match is None:
            logger.warning(
                f"Invalid rebalance_freq='{rebalance_freq}', falling back to daily (1d)"
            )
            return ("d", ts.toordinal())

        n = max(int(match.group(1)), 1)
        unit = match.group(2)

        if unit == "d":
            return ("d", ts.toordinal() // n)

        if unit == "w":
            iso_year, iso_week, _ = ts.isocalendar()
            week_index = iso_year * 53 + iso_week
            return ("w", week_index // n)

        if unit in {"m", "mo"}:
            month_index = ts.year * 12 + (ts.month - 1)
            return ("m", month_index // n)

        # quarter
        quarter_index = ts.year * 4 + ((ts.month - 1) // 3)
        return ("q", quarter_index // n)
    # ========================================================================
    # CLI Callbacks
    # ========================================================================

    def _on_cli_pause(self) -> None:
        """Called when CLI requests pause."""
        self.is_paused = True
        logger.info("Trading paused via CLI")

    def _on_cli_resume(self) -> None:
        """Called when CLI requests resume."""
        self.is_paused = False
        logger.info("Trading resumed via CLI")

    def _on_config_update(self, updates: dict) -> None:
        """Called when CLI sends config update (supports nested keys)."""
        logger.info(f"Updating config: {updates}")
        self.config.update_from_dict(updates)
        if "trading" in updates and "rebalance_freq" in updates["trading"]:
            # Force immediate rebalance on next bar after frequency change.
            self._last_rebalance_bucket = None

    def _on_get_signal(self) -> dict:
        """Called when CLI requests regime signal info."""
        if hasattr(self.strategy, "regime_detector"):
            signal = cast(RegimeSignal, self.strategy.regime_detector.current_signal) # type: ignore
            return {
                "regime": signal.regime.name,
                "confidence": signal.confidence,
                "timestamp": str(signal.timestamp),
            }
        return {"error": "Strategy does not have regime detector"}

    def _on_get_config(self, keys: list[str] | None = None) -> dict:
        """Called when CLI requests config (entire or specific keys with dot notation)."""
        if not keys:
            # Return entire config
            return {"config": self.config.to_dict()}
        
        # Return specific keys
        result = {}
        for key_path in keys:
            value = Config.get_nested_value(self.config, key_path)
            result[key_path] = value
        return {"config": result}

    def _on_get_positions(self) -> dict:
        """Called when CLI requests position info."""
        if self.broker is None:
            logger.warning("Broker not initialized - cannot fetch positions")
            return {"error": "Broker not initialized"}
        portfolio = self.broker.get_portfolio_state()
        positions = {
            symbol: {
                "qty": pos.qty,
                "market_value": pos.market_value,
                "avg_entry_price": pos.avg_entry_price,
            }
            for symbol, pos in portfolio.positions.items()
        }
        return {
            "positions": positions,
            "total_value": portfolio.total_value,
            "cash": portfolio.cash,
        }

    # ========================================================================
    # Data Fetching
    # ========================================================================

    def _fetch_historical_data(self, extra_assets: list[Asset] = [], extra_days: int = 0) -> pl.DataFrame:
        """
        Fetch historical data for initialization.

        Returns
        -------
        pl.DataFrame
            Columns: date, sym_o, sym_h, sym_l, sym_c, sym_v for each symbol
            Example: date, vix_o, vix_h, vix_l, vix_c, vix_v, spy_o, spy_h, ...
        """
        if self.data_provider is None:
            logger.warning("Data provider not initialized - cannot fetch historical data")
            return pl.DataFrame()
        
        lookback_days = self.config.data.lookback_days
        end = datetime.now()
        start = end - timedelta(days=(lookback_days + extra_days) * 1.5)

        # Map assets to data fetching
        tickers = {
            "^VIX": "vix",
            "^VIX3M": "vix3m",
            "^VVIX": "vvix",
            "TIP": "tip",
            "IEF": "ief",
            **{asset.yf: asset.name for asset in extra_assets}
        }

        df = self.data_provider.get_historical_bars(
            symbols=list(tickers.keys()),
            start=start,
            end=end,
        )

        df = (df.with_columns(pl.col("symbol").replace(tickers))
                .unpivot(
                    index=["date", "symbol"],
                    variable_name="metric",
                    value_name="value"
                )
                .with_columns(
                    pl.concat_str(
                        [
                            pl.col("symbol"),
                            pl.col("metric").str.to_lowercase()
                        ],
                        separator="_"
                    ).alias("col_name")
                )
                .select(["date", "col_name", "value"])
                .pivot(values="value", index="date", on="col_name", sort_columns=True)
                .sort("date")
                .with_columns(pl.all().forward_fill()))
        
        # Ensure date column is in millisecond precision for consistent concatenation
        df = df.with_columns(pl.col("date").cast(pl.Datetime("ms")))
        
        return df.tail(lookback_days+extra_days)

    def _fetch_daily_update(self, extra_assets: list[Asset]) -> pl.DataFrame:
        """
        Fetch latest daily bars.

        Returns
        -------
        pl.DataFrame
            Latest bars with columns: date, sym_o, sym_h, sym_l, sym_c, sym_v for each symbol
            Example: date, vix_o, vix_h, vix_l, vix_c, vix_v, spy_o, spy_h, ...
        """
        if self.data_provider is None:
            logger.warning("Data provider not initialized - cannot fetch daily update")
            return pl.DataFrame()
        
        # Define required columns and defaults
        ticker_map = {
            "^VIX": "vix",
            "^VIX3M": "vix3m",
            "^VVIX": "vvix",
            "TIP": "tip",
            "IEF": "ief",
            "SPY": "spy",
            **{asset.yf: asset.name for asset in extra_assets}
        }

        bars = self.data_provider.get_latest_bars(list(ticker_map.keys()))

        row_data = {}
        
        for ticker, col_name in ticker_map.items():
            if ticker in bars:
                bar = bars[ticker]
                if row_data.get("date") is None:
                    row_data["date"] = bar.timestamp
                
                # Add OHLCV columns with metric suffix
                row_data[f"{col_name}_open"] = bar.open if hasattr(bar, 'open') else None
                row_data[f"{col_name}_high"] = bar.high if hasattr(bar, 'high') else None
                row_data[f"{col_name}_low"] = bar.low if hasattr(bar, 'low') else None
                row_data[f"{col_name}_close"] = bar.close if hasattr(bar, 'close') else None
                row_data[f"{col_name}_volume"] = bar.volume if hasattr(bar, 'volume') else None
            else:
                # Missing ticker - use None for all metrics
                row_data[f"{col_name}_open"] = None
                row_data[f"{col_name}_high"] = None
                row_data[f"{col_name}_low"] = None
                row_data[f"{col_name}_close"] = None
                row_data[f"{col_name}_volume"] = None
                logger.warning(f"Failed to fetch {ticker} - columns will be None")

        # Check if we have a valid date
        if row_data.get("date") is None:
            logger.error("No data fetched for any ticker")
            return pl.DataFrame()

        # Create DataFrame with all required columns
        df = pl.DataFrame([row_data])
        
        # Ensure date column is in millisecond precision for consistent concatenation
        df = df.with_columns(pl.col("date").cast(pl.Datetime("ms")))
        
        return df
    
    def set_broker(self, broker: Broker) -> None:
        self.broker = broker
        
    def set_strategy(self, strategy: Strategy) -> None:
        self.strategy = strategy
        
    def set_data_provider(self, data_provider: DataProvider) -> None:
        self.data_provider = data_provider
        