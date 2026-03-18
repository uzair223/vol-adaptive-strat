#!/usr/bin/env python3
"""
trader.py
---------
Live VIX-regime strategy runner built on alpaca-py.

Start:
    python trader.py [--config config.yaml]

Architecture
------------
  trader.py (asyncio event loop)
    |
    +-- Alpaca StockDataStream  --bar callback--> SignalEngine -> StateMachine
    |                                                                  |
    +-- Control socket (TCP JSON, port 9999)  <-- OrderDecision ------+
    |                                                                  |
    +-- TradingClient  <---------------------------------------------- (order submit)

Control protocol (newline-delimited JSON over TCP):
  {"cmd":"status"}
  {"cmd":"get_params"}
  {"cmd":"set_param","key":"band_std_dev","value":"0.9"}
  {"cmd":"reload_config"}
  {"cmd":"pause"}
  {"cmd":"resume"}
  {"cmd":"close_position"}
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime, timedelta, UTC
from typing import Any, Optional, cast
from pandas import DataFrame as pd_DataFrame
from zoneinfo import ZoneInfo

import polars as pl
import yfinance as yf
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from strategy import StrategyParams, load_config, SignalEngine, StateMachine, OrderDecision
from strategy.signals import BarSignals

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def resolve_data_feed(execution_cfg: dict[str, Any]) -> DataFeed:
    raw_exchange: str = str(execution_cfg.get("exchange", "iex")).strip().lower()
    if raw_exchange == "iex":
        return DataFeed.IEX
    if raw_exchange == "sip":
        return DataFeed.SIP
    raise ValueError("execution.exchange must be 'iex' or 'sip'")

def setup_logging(log_cfg: dict[str, Any]) -> logging.Logger:
    level: int = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file: str = str(log_cfg.get("file", "")).strip()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger: logging.Logger = logging.getLogger("trader")
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    if hasattr(ch.stream, "reconfigure"):
        ch.stream.reconfigure(encoding="utf-8", errors="replace") # type: ignore
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# VIX helpers
# ─────────────────────────────────────────────────────────────────────────────

VIX_TICKERS: dict[str, str] = {"^VIX": "vix", "^VIX3M": "vix3m", "^VIX9D": "vix9d"}


def fetch_vix(days_back: int = 200) -> pl.DataFrame:
    """Download recent VIX term-structure from Yahoo Finance (daily)."""
    end:   datetime = datetime.now(UTC)
    start: datetime = end - timedelta(days=days_back * 2)

    raw = yf.download( # type: ignore
        list(VIX_TICKERS.keys()),
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    
    if raw is None:
        raise ValueError("Failed to fetch VIX data from Yahoo Finance.")

    raw = cast(pd_DataFrame, raw["Close"]).rename(columns=VIX_TICKERS)

    df: pl.DataFrame = (
        pl.from_pandas(raw.reset_index())
        .rename({"Date": "date"})
        .with_columns(pl.col("date").cast(pl.Date))
        .sort("date")
        .with_columns(
            pl.col("vix").forward_fill(),
            pl.col("vix3m").forward_fill(),
            pl.col("vix9d").forward_fill(),
        )
        .drop_nulls(subset=["vix", "vix3m"])
    )
    return df.tail(days_back)


# ─────────────────────────────────────────────────────────────────────────────
# Market hours helpers (NYSE / US equities, ET)
# ─────────────────────────────────────────────────────────────────────────────

_ET: ZoneInfo = ZoneInfo("America/New_York")
_MARKET_OPEN:  tuple[int, int] = (9, 30)
_MARKET_CLOSE: tuple[int, int] = (16, 0)


def _now_et() -> datetime:
    return datetime.now(_ET)


def market_is_open() -> bool:
    """True if NYSE regular session is currently in progress."""
    now: datetime = _now_et()
    if now.weekday() >= 5:
        return False
    open_dt:  datetime = now.replace(hour=_MARKET_OPEN[0],  minute=_MARKET_OPEN[1],  second=0, microsecond=0)
    close_dt: datetime = now.replace(hour=_MARKET_CLOSE[0], minute=_MARKET_CLOSE[1], second=0, microsecond=0)
    return open_dt <= now < close_dt


def minutes_since_market_close() -> float:
    """Minutes elapsed since the most recent regular-session close."""
    now: datetime = _now_et()
    if market_is_open():
        return 0.0

    candidate: datetime = now.replace(
        hour=_MARKET_CLOSE[0], minute=_MARKET_CLOSE[1], second=0, microsecond=0
    )
    if candidate > now:
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)

    return (now - candidate).total_seconds() / 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Historical bar fetch for warmup
# ─────────────────────────────────────────────────────────────────────────────

def fetch_historical_bars(
    symbol:       str,
    timeframe:    TimeFrame,
    warmup_bars:  int,
    api_key:      str,
    secret_key:   str,
    feed:         DataFeed,
    extra_buffer: int = 50,
) -> pl.DataFrame:
    """Fetch enough historical intraday bars to warm up the signal engine."""
    client: StockHistoricalDataClient = StockHistoricalDataClient(
        api_key=api_key, secret_key=secret_key
    )
    end: datetime = datetime.now(UTC) - timedelta(minutes=15)

    total_bars_needed: int = warmup_bars + extra_buffer
    hours_per_bar: float
    if timeframe == TimeFrame.Hour:
        hours_per_bar = 1.0
    elif timeframe.unit == TimeFrameUnit.Minute:
        hours_per_bar = timeframe.amount / 60.0
    else:
        hours_per_bar = 1.0

    days_needed: int = int(total_bars_needed * hours_per_bar / 6.5 * 1.5) + 10
    start: datetime = end - timedelta(days=max(days_needed, 60))

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=feed,
        adjustment=Adjustment.ALL,
    )
    bars_response = client.get_stock_bars(request)
    # get_stock_bars returns BarSet | RawData; cast to access .df
    from alpaca.data.models import BarSet
    bars = cast(BarSet, bars_response)
    df: pl.DataFrame = pl.from_pandas(bars.df.loc[symbol].reset_index())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main trader
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    """
    Orchestrates data ingestion, signal computation, and order execution.
    All mutable state is protected by asyncio (single-threaded), except
    params which is guarded by a threading.Lock for socket hot-reload.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path:  str            = config_path
        self._params_lock: threading.Lock = threading.Lock()

        params: StrategyParams
        cfg:    dict[str, Any]
        params, cfg = load_config(config_path)
        self.params: StrategyParams  = params
        self.cfg:    dict[str, Any]  = cfg

        self.logger: logging.Logger = setup_logging(cfg.get("logging", {}))
        self.log:    logging.Logger = self.logger

        alpaca_cfg: dict[str, str] = cfg["alpaca"]
        api_key:    str            = alpaca_cfg["api_key"]
        secret_key: str            = alpaca_cfg["secret_key"]
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API key/secret not set. "
                "Use env vars APCA_API_KEY_ID / APCA_API_SECRET_KEY "
                "or set them in config.yaml."
            )

        execution_cfg: dict[str, Any] = cfg.get("execution", {})
        paper: bool = bool(execution_cfg.get("paper_trading", True))
        self.data_feed: DataFeed = resolve_data_feed(execution_cfg)
        self.trading_client: TradingClient  = TradingClient(
            api_key=api_key, secret_key=secret_key, paper=paper
        )
        self.data_stream: StockDataStream = StockDataStream(
            api_key=api_key, secret_key=secret_key, feed=self.data_feed
        )

        # Signal engine and state machine (rebuilt on warmup / param reload)
        self.engine: Optional[SignalEngine] = None
        self.sm:     Optional[StateMachine] = None

        # Runtime state
        self._paused:         bool                   = False
        self._last_bar_date:  Optional[str]          = None
        self._last_vix_date:  Optional[str]          = None
        self._vix_df:         Optional[pl.DataFrame] = None

        ctrl: dict[str, Any] = cfg.get("control", {})
        self.ctrl_host: str = str(ctrl.get("host", "127.0.0.1"))
        self.ctrl_port: int = int(ctrl.get("port", 9999))
        self._control_task: Optional[asyncio.Task[None]] = None
        self._shutting_down: bool = False

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        try:
            self.log.info(
                "\n===================================\n"
                " VIX Strategy Trader starting up\n"
                "===================================\n"
                f"Symbol: {self.params.symbol}\tTF: {self.params.timeframe}\n"
                f"Mode: {self.params.strategy_type}\n"
                f"Paper: {self.cfg['execution'].get('paper_trading', True)}\t"
                f"Exchange: {str(self.data_feed.value).upper()}"
            )

            await self._warm_up()

            self._control_task = asyncio.create_task(self._start_control_server())

            symbol: str = self.params.symbol
            self.data_stream.subscribe_bars(self._on_bar, symbol) # type: ignore
            self.log.info(
                f"Subscribed to live {self.params.timeframe} bars for {symbol} "
                f"({str(self.data_feed.value).upper()})"
            )
            self.log.info(f"Control socket listening on {self.ctrl_host}:{self.ctrl_port}")

            await self.data_stream._run_forever()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully stop background tasks and websocket before loop teardown."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if self._control_task and not self._control_task.done():
            self._control_task.cancel()
            try:
                await self._control_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.log.debug(f"Control server shutdown error: {exc}")

        try:
            await self.data_stream.stop_ws()
        except Exception as exc:
            self.log.debug(f"Data stream stop signal error: {exc}")

        try:
            await asyncio.wait_for(self.data_stream.close(), timeout=5)
        except Exception as exc:
            self.log.debug(f"Data stream close error: {exc}")

    async def _warm_up(self) -> None:
        """Fetch historical VIX + bars and replay them to prime all indicators."""
        p: StrategyParams = self.params
        delayed: bool = bool(self.cfg.get("execution", {}).get("delayed_warmup", False))

        if delayed and self.data_feed == DataFeed.IEX:
            await self._wait_for_fresh_history()

        self.log.info("Fetching VIX history...")
        self._vix_df = fetch_vix(days_back=p.warmup_vix_days + 30)
        self.log.info(f"  {len(self._vix_df)} VIX days loaded")

        self.log.info(f"Fetching {p.warmup_bars + 50} warmup bars for {p.symbol}...")
        alpaca_cfg: dict[str, str] = self.cfg["alpaca"]
        bars_df: pl.DataFrame = fetch_historical_bars(
            symbol      = p.symbol,
            timeframe   = p.to_alpaca_timeframe(),
            warmup_bars = p.warmup_bars,
            api_key     = alpaca_cfg["api_key"],
            secret_key  = alpaca_cfg["secret_key"],
            feed        = self.data_feed,
        )
        self.log.info(f"  {len(bars_df)} intraday bars loaded")

        self.engine = SignalEngine(p)
        self.sm     = StateMachine(p)

        # Replay historical VIX
        row: dict[str, Any]
        for row in self._vix_df.iter_rows(named=True):
            self.engine.update_vix(float(row["vix"]), float(row["vix3m"]))
            self._last_vix_date = str(row["date"])

        # Replay historical bars
        prev_date: Optional[str] = None
        for row in bars_df.iter_rows(named=True):
            ts   = row["timestamp"]
            date: str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            is_new: bool = (date != prev_date)
            if is_new:
                prev_date = date

            sig: BarSignals = self.engine.update_bar(
                date=date, is_new_session=is_new,
                open_=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]),    close=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
            )
            self.sm.process_bar(
                float(row["open"]), float(row["high"]), float(row["low"]),
                sig, warmup=True,
            )
            self._last_bar_date = date

        self.log.info(
            f"Warm-up complete. State={self.sm.state}  Ready={self.engine is not None}"
        )
        await self._sync_position()

    async def _wait_for_fresh_history(self) -> None:
        """
        With IEX's 15-minute historical delay, ensure we don't warm up on
        stale bars that haven't cleared the delay window yet.

        - Market closed >= 15 min ago  -> fetch immediately, data is fresh
        - Market closed  < 15 min ago  -> wait until 15 min after close
        - Market is open               -> collect 15 live bars via the stream
        """
        if self.data_feed != DataFeed.IEX:
            self.log.info("Skipping delayed warmup wait because exchange is SIP.")
            return

        IEX_DELAY_MINUTES: int = 15

        if not market_is_open():
            mins: float = minutes_since_market_close()
            if mins >= IEX_DELAY_MINUTES:
                self.log.info(
                    f"Market closed {mins:.0f} min ago -- historical data is fresh, "
                    "fetching immediately."
                )
                return

            wait_secs: float   = (IEX_DELAY_MINUTES - mins) * 60
            ready_at:  datetime = _now_et() + timedelta(seconds=wait_secs)
            self.log.info(
                f"Market closed {mins:.1f} min ago -- waiting "
                f"{wait_secs/60:.1f} min until {ready_at.strftime('%H:%M:%S ET')} "
                "for IEX delay to clear..."
            )
            await asyncio.sleep(wait_secs)
            self.log.info("IEX delay cleared -- proceeding with historical fetch.")
            return

        self.log.info(
            f"Market is open -- waiting {IEX_DELAY_MINUTES} min before fetching "
            "historical data to clear IEX delay..."
        )

        bars_received: list[Bar] = []
        ready_event: asyncio.Event = asyncio.Event()

        async def _counting_handler(bar: Bar) -> None:
            bars_received.append(bar)
            self.log.info(
                f"  Pre-warmup bar {len(bars_received)}/{IEX_DELAY_MINUTES}: "
                f"close={bar.close:.2f}"
            )
            if len(bars_received) >= IEX_DELAY_MINUTES:
                ready_event.set()

        self.data_stream.subscribe_bars(_counting_handler, self.params.symbol) # type: ignore
        await ready_event.wait()
        self.data_stream.unsubscribe_bars(self.params.symbol)
        self.log.info(
            f"Collected {len(bars_received)} live bars -- "
            "proceeding with historical fetch."
        )

    async def _sync_position(self) -> None:
        """Query Alpaca for existing position and align state machine."""
        try:
            pos: Position = cast(
                Position,
                self.trading_client.get_open_position(self.params.symbol),
            )
            qty: float = float(pos.qty_available or 0)
            self.log.info(f"Existing Alpaca position: {qty} shares")
            if abs(qty) > 0 and self.sm and self.sm.is_flat:
                self.log.warning(
                    "State machine is FLAT but Alpaca shows an open position. "
                    "You may want to flatten manually via the control socket."
                )
        except Exception:
            self.log.info("No existing position in Alpaca.")

    # ── Live bar callback ─────────────────────────────────────────────────────

    async def _on_bar(self, bar: Bar) -> None:
        """Called by Alpaca's StockDataStream for each completed bar."""
        if self._paused:
            return

        ts:   datetime = bar.timestamp
        date: str      = ts.strftime("%Y-%m-%d")

        # ── VIX refresh once per day ──────────────────────────────────────────
        if date != self._last_vix_date:
            try:
                self.log.info(f"New session {date}: refreshing VIX...")
                self._vix_df = fetch_vix(days_back=10)
                last_row: dict[str, Any] = self._vix_df.tail(1).to_dicts()[0]
                assert self.engine is not None
                self.engine.update_vix(float(last_row["vix"]), float(last_row["vix3m"]))
                self._last_vix_date = date
                self.log.info(
                    f"VIX updated: vix={last_row['vix']:.2f}  "
                    f"vix3m={last_row['vix3m']:.2f}"
                )
            except Exception as exc:
                self.log.error(f"VIX refresh failed: {exc}")
                return

        # ── Signal update ─────────────────────────────────────────────────────
        is_new_session: bool = (date != self._last_bar_date)
        with self._params_lock:
            params: StrategyParams = self.params

        assert self.engine is not None
        assert self.sm     is not None

        sig: BarSignals = self.engine.update_bar(
            date=date, is_new_session=is_new_session,
            open_=float(bar.open),   high=float(bar.high),
            low=float(bar.low),      close=float(bar.close),
            volume=float(bar.volume) if bar.volume is not None else 0.0,
        )
        self._last_bar_date = date

        if not sig.ready:
            self.log.debug("Signal engine still warming up...")
            return

        self.log.debug(
            f"Bar {ts} | close={bar.close:.2f}  anchor={sig.anchor:.2f}  "
            f"z={sig.vol_zscore:.2f}  high_vol={sig.is_high_vol}  "
            f"state={self.sm.state}"
        )

        decision: OrderDecision = self.sm.process_bar(
            float(bar.open), float(bar.high), float(bar.low), sig
        )
        if decision.is_trade:
            await self._execute(decision, bar, sig)

    # ── Order execution ───────────────────────────────────────────────────────

    def _account_equity(self) -> float:
        """Return total account equity from Alpaca. Raises on failure."""
        # get_account() returns TradeAccount | RawData
        account: TradeAccount = cast(
            TradeAccount, self.trading_client.get_account()
        )
        return float(account.equity or 0)

    def _compute_qty(
        self,
        exposure_fraction: float,
        price:             float,
        is_short:          bool,
    ) -> Optional[float]:
        """
        Compute order quantity from account equity and an exposure fraction.

        - Long  + allow_fractional=True  -> fractional qty (rounded to 3 dp)
        - Short or allow_fractional=False -> whole shares (floor)

        Returns None if the account cannot afford the minimum order size.
        """
        equity:   float = self._account_equity()
        notional: float = equity * exposure_fraction
        raw_qty:  float = notional / price

        if is_short or not self.params.allow_fractional:
            if raw_qty < 1.0:
                return None
            return float(int(raw_qty))
        else:
            qty: float = round(raw_qty, 3)
            if qty < 0.001:
                return None
            return qty

    async def _execute(
        self,
        decision: OrderDecision,
        bar:      Bar,
        sig:      BarSignals,
    ) -> None:
        """Translate an OrderDecision into an Alpaca market order."""
        params: StrategyParams = self.params
        price:  float          = float(bar.close)

        side: OrderSide
        qty:  float

        try:
            if decision.action == "exit":
                side = OrderSide.SELL if decision.prev_state > 0 else OrderSide.BUY
                self.log.info(
                    f"EXIT  side={side.value}  reason={decision.reason}  "
                    f"stop={decision.stop_price}"
                )
                self.trading_client.close_position(params.symbol)
                self.log.info("Position closed via close_position()")
                return

            elif decision.action == "enter_long":
                side     = OrderSide.BUY
                fraction: float = (
                    params.mr_exposure if abs(decision.new_state) == 1
                    else params.tf_exposure
                )
                qty_or_none: Optional[float] = self._compute_qty(fraction, price, is_short=False)
                if qty_or_none is None:
                    self.log.warning(
                        f"SKIPPED LONG entry -- insufficient equity for "
                        f"{fraction:.0%} of account at price {price:.2f} "
                        f"(allow_fractional={params.allow_fractional})"
                    )
                    assert self.sm is not None
                    self.sm.reset()
                    return
                qty = qty_or_none
                self.log.info(
                    f"ENTER LONG  qty={qty}  price~{price:.2f}  "
                    f"exposure={fraction:.0%}  reason={decision.reason}  "
                    f"anchor={sig.anchor:.2f}  z={sig.vol_zscore:.2f}"
                )

            elif decision.action == "enter_short":
                side      = OrderSide.SELL
                fraction2: float = (
                    params.mr_exposure if abs(decision.new_state) == 1
                    else params.tf_exposure
                )
                qty_or_none2: Optional[float] = self._compute_qty(fraction2, price, is_short=True)
                if qty_or_none2 is None:
                    self.log.warning(
                        f"SKIPPED SHORT entry -- insufficient equity for 1 whole "
                        f"share at price {price:.2f} "
                        "(fractional shorts not supported by Alpaca)"
                    )
                    assert self.sm is not None
                    self.sm.reset()
                    return
                qty = qty_or_none2
                self.log.info(
                    f"ENTER SHORT  qty={qty}  price~{price:.2f}  "
                    f"exposure={fraction2:.0%}  reason={decision.reason}  "
                    f"anchor={sig.anchor:.2f}  z={sig.vol_zscore:.2f}"
                )
            else:
                return

            order_req = MarketOrderRequest(
                symbol=params.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            # submit_order returns Order | RawData
            order: Order = cast(Order, self.trading_client.submit_order(order_req))
            self.log.info(f"Order submitted: id={order.id}  status={order.status}")

        except Exception as exc:
            self.log.error(f"Order failed: {exc}")
            assert self.sm is not None
            self.sm.reset()

    # ── Control socket server ─────────────────────────────────────────────────

    async def _start_control_server(self) -> None:
        server = await asyncio.start_server(
            self._handle_control_conn,
            host=self.ctrl_host,
            port=self.ctrl_port,
        )
        async with server:
            await server.serve_forever()

    async def _handle_control_conn(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer: Any = writer.get_extra_info("peername")
        self.log.info(f"Control connection from {peer}")
        try:
            while True:
                line: bytes = await reader.readline()
                if not line:
                    break
                reply: dict[str, Any]
                try:
                    msg: dict[str, Any] = json.loads(line.decode().strip())
                    reply = await self._dispatch_cmd(msg)
                except json.JSONDecodeError as exc:
                    reply = {"ok": False, "error": f"JSON parse error: {exc}"}
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            self.log.info(f"Control connection closed {peer}")

    async def _dispatch_cmd(self, msg: dict[str, Any]) -> dict[str, Any]:
        cmd: str = str(msg.get("cmd", ""))

        if cmd == "status":
            return self._cmd_status()
        elif cmd == "get_params":
            with self._params_lock:
                return {"ok": True, "params": self.params.to_dict()}
        elif cmd == "set_param":
            return await self._cmd_set_param(
                key=str(msg.get("key", "")),
                value=str(msg.get("value", "")),
            )
        elif cmd == "reload_config":
            return await self._cmd_reload_config()
        elif cmd == "pause":
            self._paused = True
            self.log.info("Trading PAUSED via control socket")
            return {"ok": True, "paused": True}
        elif cmd == "resume":
            self._paused = False
            self.log.info("Trading RESUMED via control socket")
            return {"ok": True, "paused": False}
        elif cmd == "close_position":
            return await self._cmd_close_position()
        else:
            return {"ok": False, "error": f"Unknown command '{cmd}'"}

    def _cmd_status(self) -> dict[str, Any]:
        sm: Optional[StateMachine] = self.sm
        return {
            "ok":            True,
            "state":         sm.state         if sm else None,
            "position_size": sm.position_size if sm else 0.0,
            "hwm":           sm.hwm           if sm else None,
            "paused":        self._paused,
            "last_bar_date": self._last_bar_date,
            "last_vix_date": self._last_vix_date,
            "symbol":        self.params.symbol,
            "timeframe":     self.params.timeframe,
            "exchange":      str(self.data_feed.value).lower(),
        }

    async def _cmd_set_param(self, key: str, value: str) -> dict[str, Any]:
        """Hot-patch a single strategy parameter."""
        REQUIRES_REBUILD: frozenset[str] = frozenset({
            "ema_span", "zscore_window", "atr_period", "trend_ma_window",
            "anchor_recompute", "timeframe",
        })
        try:
            with self._params_lock:
                new_params: StrategyParams = self.params.patch(key, value)
                self.params = new_params

            needs_rebuild: bool = key in REQUIRES_REBUILD
            if needs_rebuild:
                self.log.warning(
                    f"Param '{key}' changed -- signal engine rebuild needed. "
                    "Run 'reload_config' to trigger full re-warm-up."
                )
            self.log.info(f"Param updated: {key} = {getattr(new_params, key)}")
            return {
                "ok":            True,
                "key":           key,
                "value":         getattr(new_params, key),
                "rebuild_needed": needs_rebuild,
            }
        except (ValueError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}

    async def _cmd_reload_config(self) -> dict[str, Any]:
        """Reload config.yaml and re-warm-up the engine with new params."""
        try:
            new_params: StrategyParams
            new_cfg:    dict[str, Any]
            new_params, new_cfg = load_config(self.config_path)
            new_feed: DataFeed = resolve_data_feed(new_cfg.get("execution", {}))
            if new_feed != self.data_feed:
                return {
                    "ok": False,
                    "error": (
                        "execution.exchange cannot be hot-reloaded; "
                        "restart trader.py to apply feed change"
                    ),
                }
            with self._params_lock:
                self.params = new_params
                self.cfg    = new_cfg
            self.log.info(f"Config reloaded from {self.config_path} -- re-warming up...")
            await self._warm_up()
            return {"ok": True, "message": "Config reloaded and engine re-initialised"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _cmd_close_position(self) -> dict[str, Any]:
        """Force-close any open position immediately."""
        if not self.sm or self.sm.is_flat:
            return {"ok": True, "message": "Already flat"}
        try:
            self.trading_client.close_position(self.params.symbol)
            self.sm.reset()
            self.log.info("Position closed via control command")
            return {"ok": True, "message": "Position closed"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="VIX regime live trader")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    args = parser.parse_args()
    trader = Trader(config_path=args.config)
    await trader.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown requested -- bye!")
