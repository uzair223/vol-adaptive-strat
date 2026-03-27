"""
broker.py
=========
Trading/execution layer. Only executes buy/sell orders from strategy.

Key principle: This layer is PASSIVE EXECUTOR
- Receives target positions from strategy
- Manages actual positions
- Handles order execution and risk controls
- Never makes trading decisions

Separation: This layer doesn't know WHY positions were chosen,
only that they need to be executed.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal, cast

import numpy as np
import polars as pl
from src.config import Asset

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import alpaca.trading as apca_t
    import alpaca.data as apca_d
try:
    import alpaca.trading as apca_t
    import alpaca.data as apca_d
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float


@dataclass
class Order:
    """Executed order."""
    order_id: str
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    status: str
    created_at: datetime
    filled_at: Optional[datetime] = None


@dataclass
class PortfolioState:
    """Current portfolio holdings and cash."""
    total_value: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)


class Broker(ABC):
    """
    Abstract base class for trading brokers.
    
    Defines interface for:
    - Getting portfolio state
    - Executing rebalances
    - Managing orders and positions
    """
    def __init__(self, allow_fractional_shares: bool = True):
        self.allow_fractional_shares = allow_fractional_shares

    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state.

        Returns
        -------
        PortfolioState
            Total value, cash, and position details
        """
        pass

    @abstractmethod
    def execute_rebalance(
        self,
        target_allocations: dict[Asset, float],
        min_rebalance_threshold: float
    ) -> list[Order]:
        """
        Execute target positions.

        Parameters
        ----------
        target_allocations : dict[Asset, float]
            Target allocated weights per asset

        Returns
        -------
        list[Order]
            Orders that were submitted
        """
        pass

    @abstractmethod
    def cancel_open_orders(self) -> int:
        """
        Cancel all open orders.

        Returns
        -------
        int
            Number of orders cancelled
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> None:
        """Close all open positions."""
        pass


class SimulatedBroker(Broker):
    """
    Simulated paper trading broker.
    
    Executes trades using pre-known prices provided via on_new_bar() calls.
    Tracks prices, positions, and cash for backtesting/paper trading.
    Simulates realistic trading costs: commission and slippage.
    """
    
    def __init__(
        self,
        allow_fractional_shares: bool = True,
        initial_value: float = 100_000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        slippage_bps: float = 1.0,  # 1 basis point
    ):
        """
        Initialize simulated broker.
        
        Parameters
        ----------
        allow_fractional_shares : bool
            Whether to allow fractional shares (default: True)
        initial_value : float
            Starting cash and portfolio value
        commission_rate : float
            Commission as a fraction of trade value (e.g., 0.001 = 0.1%)
        slippage_bps : float
            Slippage in basis points (e.g., 1.0 = 1 bps = 0.01%)
        """
        super().__init__(allow_fractional_shares=allow_fractional_shares)
        self.initial_value = initial_value
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        
        # Current state
        self.portfolio_state = PortfolioState(
            total_value=initial_value,
            cash=initial_value,
            positions={}
        )
        
        # Price tracking - maps symbol to latest close price
        # format: date, sym_o, sym_h, sym_l, sym_c, sym_v for each symbol
        self.current_prices: Optional[pl.DataFrame] = None
        
        # Order tracking
        self.orders: list[Order] = []
        self.next_order_id = 0
        
        # Fee tracking for diagnostics
        self.total_fees_paid = 0.0
        
        # Pending allocations to execute on next bar (one-bar delay to avoid look-ahead bias)
        self.pending_allocations: Optional[dict[Asset, float]] = None
    
    def on_new_bar(self, bar_data: pl.DataFrame, min_rebalance_threshold: float) -> None:
        """
        Update current prices and execute pending allocations from previous decision.
        
        Called at the start of each bar. Executes any pending allocations from
        the previous bar's decision at these prices (one-bar delay to avoid
        look-ahead bias).
        
        Parameters
        ----------
        bar_data : pl.DataFrame
            Bar data with current prices
        """
        self.current_prices = bar_data
        
        # Execute pending allocations from previous bar at current prices
        if self.pending_allocations is not None:
            self._execute_allocations(self.pending_allocations, min_rebalance_threshold)
            self.pending_allocations = None
        
        self._update_portfolio_value()
    
    def _update_portfolio_value(self) -> None:
        """Recalculate total portfolio value based on current close prices."""
        if self.current_prices is None:
            return
        market_value = 0.0
        
        for symbol, position in self.portfolio_state.positions.items():
            # Look for close price column: symbol_close (e.g., "spy_close", "vix_close")
            close_col = f"{symbol}_close"
            if close_col in self.current_prices.columns:
                price_value = self.current_prices[-1, close_col]
                # Guard clause: skip if price is not finite (keeps previous market_value)
                if not np.isfinite(price_value):
                    logger.warning(f"Invalid price (None/NaN/Inf) for {symbol}, keeping previous valuation")
                    market_value += position.market_value
                    continue
                position.market_value = position.qty * price_value
                market_value += position.market_value
        
        self.portfolio_state.total_value = self.portfolio_state.cash + market_value
    
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state.

        Returns
        -------
        PortfolioState
            Total value, cash, and position details
        """
        self._update_portfolio_value()
        return self.portfolio_state

    def execute_rebalance(
        self,
        target_allocations: dict[Asset, float],
        min_rebalance_threshold: float
    ) -> list[Order]:
        """
        Queue target allocations to execute on next bar.
        
        Stores allocations decided at current bar to be executed at
        next bar's prices (one-bar delay to avoid look-ahead bias).

        Parameters
        ----------
        target_allocations : dict[Asset, float]
            Target fraction allocation per asset
        min_rebalance_threshold : float
            Minimum change in target weights to trigger rebalance (0-1)

        Returns
        -------
        list[Order]
            Empty list (orders will be executed and returned on next bar)
        """
        self.pending_allocations = target_allocations
        return []
    
    def _execute_allocations(self, target_allocations: dict[Asset, float], min_rebalance_threshold: float) -> list[Order]:
        """
        Execute target allocations at current prices.
        
        Calculates dollar amounts based on target allocations and current prices,
        then executes buy/sell orders.

        Parameters
        ----------
        target_allocations : dict[Asset, float]
            Target fraction allocation per asset

        Returns
        -------
        list[Order]
            Orders that were executed
        """
        if self.current_prices is None:
            logger.warning("No price data available, cannot execute trades")
            return []
        
        executed_orders = []
        
        # Get latest portfolio value
        current_state = self.get_portfolio_state()
        portfolio_value = current_state.total_value
        
        # Calculate target dollar amounts
        target_dollars: dict[Asset, float] = {}
        for asset, weight in target_allocations.items():
            target_dollars[asset] = portfolio_value * weight
        
        # Current dollar amounts in positions
        current_dollars: dict[str, float] = {}
        for symbol, position in current_state.positions.items():
            close_col = f"{symbol}_close"
            if close_col in self.current_prices.columns:
                price_value = self.current_prices[-1, close_col]
                # Guard clause: skip if price is not finite
                if not np.isfinite(price_value):
                    logger.warning(f"Invalid close price (None/NaN/Inf) for {symbol}, skipping valuation")
                    continue
                current_dollars[symbol] = position.qty * price_value
            else:
                current_dollars[symbol] = 0.0
        
        # Execute trades for each asset at open price (day t+1 open)
        for asset, target_value in target_dollars.items():
            current_qty = 0.0
            
            if asset.name in current_state.positions:
                current_qty = current_state.positions[asset.name].qty
            
            # Use open price for execution: symbol_open (e.g., "spy_open", "vix_open")
            open_col = f"{asset.name}_open"
            if open_col not in self.current_prices.columns:
                logger.warning(f"No open price column for {asset.name}, skipping trade")
                continue
            
            execution_price = self.current_prices[-1, open_col]
            
            # Guard clause: skip if price is not finite
            if not np.isfinite(execution_price):
                logger.warning(f"Invalid open price (None/NaN/Inf) for {asset.name}, skipping trade")
                continue
            
            if execution_price <= 0:
                logger.warning(f"Invalid open price (≤0) for {asset.name}, skipping trade")
                continue
            
            target_qty = target_value / execution_price
            
            # Calculate trade
            qty_diff = target_qty - current_qty
            if not self.allow_fractional_shares:
                qty_diff = int(qty_diff)
            
            if abs(qty_diff) < 0.001:  # Avoid tiny trades due to rounding
                continue
            
            side = "buy" if qty_diff > 0 else "sell"
            
            # Execute order
            order = self._execute_order(asset.name, abs(qty_diff), side, execution_price)
            executed_orders.append(order)
            
            logger.info(
                f"Order executed: {side.upper()} {abs(qty_diff):.2f} {asset.name} "
                f"@ ${execution_price:.2f} = ${abs(qty_diff * execution_price):.2f}"
            )
        
        return executed_orders
    
    def _execute_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        price: float
    ) -> Order:
        """
        Execute a single order with commission and slippage.
        
        Parameters
        ----------
        symbol : str
            Asset symbol
        qty : float
            Quantity
        side : str
            "buy" or "sell"
        price : float
            Base execution price (midpoint)
            
        Returns
        -------
        Order
            The executed order
        """
        order_id = f"SIM-{self.next_order_id}"
        self.next_order_id += 1
        
        # Apply slippage: worsen the price for the trader
        slippage_multiplier = 1.0 + (self.slippage_bps / 10_000)
        if side == "buy":
            execution_price = price * slippage_multiplier  # Pay more
        else:
            execution_price = price / slippage_multiplier  # Receive less
        
        # Calculate trade value and fees
        gross_trade_value = qty * execution_price
        commission = gross_trade_value * self.commission_rate
        total_cost = gross_trade_value + (commission if side == "buy" else -commission)
        
        now = datetime.now()
        
        # Update cash
        if side == "buy":
            self.portfolio_state.cash -= total_cost
        else:
            self.portfolio_state.cash += (gross_trade_value - commission)
        
        # Track fees
        self.total_fees_paid += commission
        
        # Update position
        if symbol in self.portfolio_state.positions:
            pos = self.portfolio_state.positions[symbol]
            if side == "buy":
                # Update average entry price (includes commission)
                total_cost_with_pos = pos.qty * pos.avg_entry_price + total_cost
                pos.qty += qty
                pos.avg_entry_price = total_cost_with_pos / pos.qty if pos.qty > 0 else 0
            else:
                pos.qty -= qty
                if pos.qty < 0.001:  # Close out position
                    del self.portfolio_state.positions[symbol]
        else:
            # New position
            if side == "buy":
                self.portfolio_state.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    market_value=gross_trade_value,
                    avg_entry_price=execution_price
                )
        
        # Create order record
        order = Order(
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            status="filled",
            created_at=now,
            filled_at=now
        )
        self.orders.append(order)
        
        logger.debug(
            f"Order executed: {side.upper()} {qty:.4f} {symbol} "
            f"@ ${execution_price:.2f} | Base: ${price:.2f}, Slippage: {self.slippage_bps:.1f} bps, "
            f"Commission: ${commission:.2f}"
        )
        
        return order

    def cancel_open_orders(self) -> int:
        """
        Cancel all open orders (no-op for simulated broker).

        Returns
        -------
        int
            Number of orders cancelled (always 0 for simulated)
        """
        return 0

    def close_all_positions(self) -> None:
        """Close all open positions by selling them at current open prices."""
        if self.current_prices is None:
            logger.warning("No price data available, cannot close positions")
            return
        
        positions_to_close = list(self.portfolio_state.positions.items())
        
        for symbol, position in positions_to_close:
            open_col = f"{symbol}_open"
            if open_col not in self.current_prices.columns or position.qty < 0.001:
                continue
            
            price_value = self.current_prices[-1, open_col]
            
            # Guard clause: skip if price is not finite
            if not np.isfinite(price_value):
                logger.warning(f"Invalid open price (None/NaN/Inf) for {symbol}, cannot close position")
                continue
            
            if price_value <= 0:
                logger.warning(f"Invalid open price (≤0) for {symbol}, cannot close position")
                continue
            
            order = self._execute_order(
                symbol=symbol,
                qty=position.qty,
                side="sell",
                price=price_value
            )
            logger.info(f"Closed position: SELL {order.qty:.2f} {symbol}")
    
    def get_trading_stats(self) -> dict:
        """
        Get trading statistics including fees.
        
        Returns
        -------
        dict
            Statistics including:
            - total_trades: Number of orders executed
            - total_fees_paid: Total commission and fees
            - avg_commission_per_trade: Average fee per trade
            - commission_rate: Configured commission rate
            - slippage_bps: Configured slippage in basis points
        """
        total_trades = len(self.orders)
        avg_fee = self.total_fees_paid / total_trades if total_trades > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "total_fees_paid": self.total_fees_paid,
            "avg_commission_per_trade": avg_fee,
            "commission_rate": self.commission_rate,
            "slippage_bps": self.slippage_bps,
        }


class AlpacaBroker(Broker):
    """
    Live trading broker using Alpaca API (alpaca-py).
    
    Executes trades against the real Alpaca broker for live trading.
    Handles order submission, position tracking, and portfolio management.
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = False, allow_fractional_shares: bool = True):
        """
        Initialize Alpaca broker connection.
        
        Parameters
        ----------
        api_key : str
            Alpaca API key.
        secret_key : str
            Alpaca secret key.
        paper : bool
            Whether to use paper trading (default: False for live trading).
        allow_fractional_shares : bool
            Whether to allow fractional shares (default: True).
            
        Raises
        ------
        ImportError
            If alpaca-py is not installed.
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for AlpacaBroker. "
                "Install it with: pip install alpaca-py"
            )
        
        super().__init__(allow_fractional_shares=allow_fractional_shares)
        
        try:
            self.client = apca_t.TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper
            )
            self.data_client = apca_d.StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            mode = "paper" if paper else "live"
            logger.info(f"Alpaca broker initialized in {mode} trading mode")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Alpaca client: {e}")
    
    def get_portfolio_state(self) -> PortfolioState:
        """
        Get current portfolio state from Alpaca.
        
        Returns
        -------
        PortfolioState
            Total portfolio value, cash, and position details
            
        Raises
        ------
        Exception
            If API call fails
        """
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py is not installed")
        
        try:
            # Get account information
            account = cast(apca_t.TradeAccount, self.client.get_account())
            
            # Get all positions
            positions_data = cast(list[apca_t.Position], self.client.get_all_positions())
            
            # Build positions dict
            positions = {}
            for pos in positions_data:
                symbol = pos.symbol
                positions[symbol] = Position(
                    symbol=symbol,
                    qty=float(pos.qty or 0.0),
                    market_value=float(pos.market_value or 0.0),
                    avg_entry_price=float(pos.avg_entry_price or 0.0)
                )
            
            # Create portfolio state
            portfolio_state = PortfolioState(
                total_value=float(account.portfolio_value or 0.0),
                cash=float(account.cash or 0.0),
                positions=positions
            )
            
            return portfolio_state
        
        except Exception as e:
            logger.error(f"Failed to get portfolio state: {e}")
            raise
    
    def execute_rebalance(
        self,
        target_allocations: dict[Asset, float],
        min_rebalance_threshold: float
    ) -> list[Order]:
        """
        Execute target allocations by submitting orders to Alpaca.
        
        Parameters
        ----------
        target_allocations : dict[Asset, float]
            Target fraction allocation per asset (Asset -> weight)
        min_rebalance_threshold : float
            Minimum change in target weights to trigger rebalance (0-1)
            
        Returns
        -------
        list[Order]
            Orders that were submitted to Alpaca
            
        Raises
        ------
        Exception
            If order submission fails
        """
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py is not installed")
        
        try:
            executed_orders = []
            
            # Get current portfolio state
            current_state = self.get_portfolio_state()
            portfolio_value = current_state.total_value
            
            if portfolio_value <= 0:
                logger.warning("Portfolio value is zero or negative, cannot execute trades")
                return []
            
            # Get current positions by symbol
            current_positions = {pos.symbol: pos for pos in current_state.positions.values()}
            
            latest_quotes = cast(dict[str, apca_d.Quote], self.data_client.get_stock_latest_quote(
                apca_d.StockLatestQuoteRequest(symbol_or_symbols=[asset.apca for asset in target_allocations.keys()])
            ))
            
            # Execute trades for each asset
            for asset, target_weight in target_allocations.items():
                target_value = portfolio_value * target_weight  # Dollar amount
                
                # Get current quantity
                current_qty = 0.0
                if asset.apca in current_positions:
                    current_qty = current_positions[asset.apca].qty
                
                # Get current price from latest quote
                try:
                    if latest_quotes[asset.apca].ask_price and latest_quotes[asset.apca].ask_price > 0:
                        current_price = latest_quotes[asset.apca].ask_price
                    elif latest_quotes[asset.apca].bid_price and latest_quotes[asset.apca].bid_price > 0:
                        current_price = latest_quotes[asset.apca].bid_price
                    else:
                        logger.warning(f"Invalid price quote for {asset.apca}, skipping trade")
                        continue
                except Exception as e:
                    logger.warning(f"Failed to get quote for {asset.apca}: {e}, skipping trade")
                    continue
                
                # Calculate target quantity from target value
                target_qty = target_value / current_price
                
                # Calculate quantity change
                qty_diff = round(target_qty - current_qty, 4)
                
                # Apply fractional shares setting
                if not self.allow_fractional_shares:
                    qty_diff = int(qty_diff)
                
                # Skip tiny trades
                if abs(qty_diff) < min_rebalance_threshold:
                    continue
                
                # Determine side and submit order
                side = apca_t.OrderSide.BUY if qty_diff > 0 else apca_t.OrderSide.SELL
                
                # Submit market order
                order_request = apca_t.MarketOrderRequest(
                    symbol=asset.apca,
                    qty=abs(qty_diff),
                    side=side,
                    time_in_force=apca_t.TimeInForce.DAY
                )
                
                try:
                    alpaca_order = cast(apca_t.Order, self.client.submit_order(order_request))
                    
                    # Convert Alpaca order to our Order dataclass
                    order = Order(
                        order_id=str(alpaca_order.id),
                        symbol=asset.apca,
                        qty=float(alpaca_order.qty or 0.0),
                        side="buy" if alpaca_order.side == apca_t.OrderSide.BUY else "sell",
                        status=str(alpaca_order.status),
                        created_at=alpaca_order.created_at,
                        filled_at=alpaca_order.filled_at
                    )
                    executed_orders.append(order)
                    
                    logger.info(
                        f"Order submitted: {side.name} {abs(qty_diff):.4f} {asset.apca} "
                        f"@ ${current_price:.2f} (ID: {alpaca_order.id})"
                    )
                
                except Exception as e:
                    logger.error(f"Failed to submit order for {asset.apca}: {e}")
                    continue
            
            return executed_orders
        
        except Exception as e:
            logger.error(f"Failed to execute rebalance: {e}")
            raise
    
    def cancel_open_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns
        -------
        int
            Number of orders cancelled
            
        Raises
        ------
        Exception
            If API call fails
        """
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py is not installed")
        
        try:
            cancelled_orders = self.client.cancel_orders()
            num_cancelled = len(cancelled_orders) if cancelled_orders else 0
            
            logger.info(f"Cancelled {num_cancelled} open orders")
            return num_cancelled
        
        except Exception as e:
            logger.error(f"Failed to cancel open orders: {e}")
            raise
    
    def close_all_positions(self) -> None:
        """
        Close all open positions by submitting sell orders.
        
        Raises
        ------
        Exception
            If API call fails
        """
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-py is not installed")
        
        try:
            # Get all open positions
            positions = cast(list[apca_t.Position], self.client.get_all_positions())
            
            if not positions:
                logger.info("No open positions to close")
                return
            
            # Submit sell orders for each position
            for position in positions:
                try:
                    close_order = cast(apca_t.Order, self.client.close_position(position.symbol))
                    logger.info(f"Closed position: SELL {position.qty} {position.symbol} (Order ID: {close_order.id})")
                except Exception as e:
                    logger.error(f"Failed to close position for {position.symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            raise


