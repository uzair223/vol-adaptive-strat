"""
dry_run.py
==========
Paper trading simulation/backtest mode.

Runs the strategy logic on historical or live market data without executing any trades.
Useful for testing strategy decisions and reviewing past trading periods.

Usage:
    python dry_run.py                    # Run once with latest market data
    python dry_run.py --days 30          # Backtest last 30 days of trading
    python dry_run.py --config config.yaml --days 60
"""

import argparse
import contextlib
import logging
import os
import sys
import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv

from src.config import Config
from src.trader import Trader
from src.strategy import VolAdaptiveStrategy
from src.broker import SimulatedBroker
from src.data import YahooDataProvider
from src.util import suppress, setup_logging


load_dotenv()

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - risk_free_rate / 252
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return == 0:
        return np.nan
    
    return (mean_return / std_return) * np.sqrt(252)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return np.nan
    
    excess_returns = returns - risk_free_rate / 252
    mean_return = np.mean(excess_returns)
    
    # Only consider negative returns for downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.nan
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.nan
    
    return (mean_return / downside_std) * np.sqrt(252)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = np.min(drawdown)
    
    return max_dd * 100  # Convert to percentage

def main():
    """Main entry point for dry run."""
    parser = argparse.ArgumentParser(
        description="Paper trading simulation (no trades executed)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100_000.0,
        help="Initial account balance for simulation (default: 100,000.0)"
    )
    parser.add_argument(
        "--no-fractional-shares",
        action="store_true",
        help="Disallow fractional shares (default: False)"
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0015,
        help="Commission as fraction of trade value (default: 0.0015 = 0.15%)"
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0 = 5 bps)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=252,
        help="Number of OOS days to backtest (default: 252 ~2y)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (default: False)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("=" * 70)
    logger.info("DRY RUN")
    
    logger.info(f"  Loading config from: {args.config}")
    config = Config.from_yaml(args.config)
    
    logger.info(f"  Assets: {[a.name for a in config.trading.assets]}")
    logger.info(f"  Rebalance frequency: {config.trading.rebalance_freq}")
    logger.info(f"  Minimum rebalance threshold: {config.trading.min_rebalance_threshold:.2%}")
    logger.info(f"  Backtest OOS period: {args.days} days")
    logger.info(f"  Initial balance: ${args.initial_balance:,.2f}")
    logger.info(f"  Fractional shares allowed: {not args.no_fractional_shares}")
    logger.info(f"  Commission rate: {args.commission_rate*100:.3f}%")
    logger.info(f"  Slippage: {args.slippage_bps:.1f} bps")
    logger.info("=" * 70)
    
    suppress_ctx = contextlib.nullcontext() if args.verbose else suppress(logging.INFO)
    try:
        with suppress_ctx:
            # Initialize data provider
            broker = SimulatedBroker(
                allow_fractional_shares=not args.no_fractional_shares,
                initial_value=args.initial_balance,
                commission_rate=args.commission_rate,
                slippage_bps=args.slippage_bps
            )
            data_provider = YahooDataProvider()
            trader = Trader(config, broker, data_provider)
            
            # Fetch historical data
            logger.info("Fetching historical data...")
            extra_days = args.days if args.days else 0
            
            historical_data = trader._fetch_historical_data(
                extra_assets=config.trading.assets,
                extra_days=extra_days
            )
        
            if historical_data.is_empty():
                logger.error("No historical data available")
                return
            
            logger.info(f"Historical data: {len(historical_data)} rows")
            
            training_data = historical_data.slice(0, config.data.lookback_days)
            trader.historical_data = training_data
            trader.set_strategy(VolAdaptiveStrategy(
                assets=config.trading.assets,
                config=config.strategy,
                regime_detector_config=config.regime_detector,
                initial_data=training_data,
                hmm_model_path=(os.getenv("HMM_MODEL_PATH") or "models/regime_hmm.pkl"),
            ))
            if args.verbose:
                logger.info(f"Strategy initialized with {len(training_data)} row(s) of training data")
            
            # Split data into training (first lookback_days) and test (remaining days)
            if len(historical_data) < config.data.lookback_days + args.days:
                available_oos = len(historical_data) - config.data.lookback_days
                logger.warning(
                    f"Only {len(historical_data)} days available "
                    f"({config.data.lookback_days} training + {available_oos} OOS), "
                    f"requesting {args.days} OOS"
                )
                if available_oos <= 0:
                    logger.error("Not enough data for training")
                    return
                sim_days = available_oos
            else:
                sim_days = args.days
            
            # Training data is already used to initialize regime_detector
            # During OOS simulation, full historical context (training + OOS) is used for features
            initial_value = broker.get_portfolio_state().total_value
            
            logger.info(f"Simulating {sim_days} OOS trading day(s)...")
            logger.info(f"Initial portfolio value: ${initial_value:,.2f}")
            logger.info("")
            
            # Track portfolio values for metrics calculation
            portfolio_values = [initial_value]
            
            # Simulate each day
            pbar = tqdm(total=sim_days, postfix=f"${initial_value:,.2f}", unit="day") if not args.verbose else None
            
            for day in range(sim_days):
                day_idx = config.data.lookback_days + day
                date_str = str(historical_data[day_idx, "date"])
                logger.info(f"Day {day+1}/{sim_days}: {date_str}")
                broker.on_new_bar(historical_data[day_idx], min_rebalance_threshold=config.trading.min_rebalance_threshold)
                trader._on_new_bar(historical_data[day_idx])
                
                # Track portfolio value
                current_state = broker.get_portfolio_state()
                portfolio_values.append(current_state.total_value)
                
                logger.info("")
                pbar.update(1) if pbar else None
                pbar.set_postfix_str(f"${current_state.total_value:,.2f}") if pbar else None
        pbar.close() if pbar else None
        
        # Final summary
        final_state = broker.get_portfolio_state()
        pnl = final_state.total_value - initial_value
        pnl_pct = (pnl / initial_value) * 100
        
        # Get trading stats
        trading_stats = broker.get_trading_stats()
        
        # Calculate metrics
        portfolio_values_arr = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values_arr) / portfolio_values_arr[:-1]
        sharpe = calculate_sharpe_ratio(daily_returns)
        sortino = calculate_sortino_ratio(daily_returns)
        max_dd = calculate_max_drawdown(portfolio_values_arr)
        
        logger.info("=" * 70)
        logger.info(f"Backtest complete!")
        logger.info(f"  Initial value: ${initial_value:,.2f}")
        logger.info(f"  Final value:   ${final_state.total_value:,.2f}")
        logger.info(f"  P&L:           ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        logger.info(f"  Sharpe Ratio:  {sharpe:.2f}")
        logger.info(f"  Sortino Ratio: {sortino:.2f}")
        logger.info(f"  Max Drawdown:  {max_dd:.2f}%")
        logger.info("-" * 70)
        logger.info(f"Trading Statistics:")
        logger.info(f"  Total trades:       {trading_stats['total_trades']}")
        logger.info(f"  Total fees paid:    ${trading_stats['total_fees_paid']:,.2f}")
        logger.info(f"  Avg fee per trade:  ${trading_stats['avg_commission_per_trade']:,.2f}")
        logger.info(f"  Commission rate:    {trading_stats['commission_rate']*100:.3f}%")
        logger.info(f"  Slippage:           {trading_stats['slippage_bps']:.1f} bps")
        logger.info("=" * 70)

        
        
    except KeyboardInterrupt:
        logger.info("\nDry run interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during dry run: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
