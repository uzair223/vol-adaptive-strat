# VIX Regime-Adaptive Trading Strategy

Lightweight live trading project built around a VIX regime signal engine and a state-machine executor.

## Contents

- trader.py: live runtime (data stream, signals, state transitions, order execution)
- control.py: CLI and REPL control socket client
- strategy/: signal and state-machine logic
- config.yaml: runtime parameters
- backtest/: research notebook and related artifacts

## Quick Start (Local)

1. Create/activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set Alpaca credentials in environment variables:

```bash
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
```

4. Review config.yaml (symbol, timeframe, exchange, risk params).
5. Start trader:

```bash
python trader.py
```

6. Open control REPL in another terminal:

```bash
python control.py repl
```

## Backtesting

See backtest/README.md for notebook usage notes.
