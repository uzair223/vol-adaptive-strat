# Vol-Adaptive Regime Strategy

This repository implements a regime-aware, volatility-targeted allocation strategy.

The core idea is simple:

- Risk is not constant. Market behavior changes by regime.
- Allocation should adapt to both regime and confidence.
- Realized volatility should be targeted so total risk remains controlled.

## Strategy Intuition

The strategy combines three layers of risk control:

1. Regime layer

- A Hidden Markov Model (HMM) classifies market state into four regimes:
  - CALM: low implied volatility, benign conditions
  - STRESS: elevated risk, risk building
  - CRISIS: acute dislocation
  - RECOVERY: post-shock normalization
- Each traded asset has regime-specific base weights.

2. Confidence layer

- The HMM produces posterior probabilities over regimes.
- Confidence is derived from posterior entropy and mapped to a scale factor.
- Low-confidence signals are de-emphasized rather than hard-switched.

3. Volatility-targeting layer

- For each asset, realized volatility is estimated from trailing log returns.
- Exposure is scaled by target_vol / realized_vol (with a volatility floor).
- Portfolio gross exposure is capped at max_exposure.

This creates a strategy that can be aggressive in favorable regimes, defensive in stressed regimes, and less reactive when the model is uncertain.

## Regime Detection Overview

Regime inference uses a feature set built from volatility and macro-proxy structure:

- Vol level and slope: VIX, VIX3M term structure
- Vol-of-vol: VVIX level and momentum
- IV vs realized vol spread
- Short-horizon momentum and long-horizon context (z-score)
- Inflation proxy dynamics using TIP and IEF

Additional stabilizers in the detector:

- Soft transition penalties to discourage semantically unlikely jumps
- Minimum holding period filter to suppress short-lived flips
- Periodic retraining on a rolling window
- Optional model serialization for warm starts

## Allocation Formula (Conceptual)

Per-asset raw exposure is computed as:

exposure_raw = base_leverage x regime_weight x confidence_scale x vol_scale

Where:

- regime_weight comes from config for the current regime
- confidence_scale maps model confidence into [0, 1]
- vol_scale = min(target_vol / realized_vol, cap)

Then all raw exposures are proportionally scaled to enforce:

- sum(abs(exposure_i)) <= max_exposure

## Data and Trading Flow

Daily flow in live mode:

1. Load historical data and initialize regime detector
2. At market-open event, fetch latest daily bars
3. Update regime state and compute target allocations
4. Rebalance only when rebalance_freq bucket changes
5. Skip tiny changes under min_rebalance_threshold

Assets and behavior are configured in config.yaml.

## Why This Design

This architecture attempts to balance:

- Adaptivity: regime-conditioned behavior
- Robustness: confidence-aware sizing and smoothing
- Risk discipline: volatility targeting plus gross exposure cap
- Practicality: periodic retraining, serialization, CLI control

## Current Scope and Assumptions

- Primary runtime target is daily bars (not intraday execution logic).
- Regime model is statistical, not causal explanation of markets.
- Performance depends on feature stability and data quality.
- Broker/API failures can still dominate realized behavior in production.

## Running

Local scripts:

- Live: python live.py
- Backtest: python backtest.py --config config.yaml --days 252
- CLI (from another shell while live is running): python cli.py status

Docker:

- Start live service: docker compose up -d --build live
- Run backtest profile: docker compose run --rm backtest python backtest.py --config config.yaml --days 30

## Configuration Pointers

Key controls in config.yaml:

- strategy.target_vol
- strategy.base_leverage
- strategy.max_exposure
- strategy.conf_min
- trading.rebalance_freq
- trading.min_rebalance_threshold
- regime_detector.retrain_interval_days
- regime_detector.retrain_window_days

Container env controls:

- APCA_API_KEY_ID
- APCA_API_SECRET_KEY
- HMM_MODEL_PATH
- CLI_HOST / CLI_PORT
