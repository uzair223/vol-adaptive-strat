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

## Common Control Commands

- status
- get
- get <key1> <key2>
- set <key> <value>
- set <key> <value> --persist
- reload-config
- pause / resume
- close-position

## Docker

Use docker-compose.yml for containerized runs.

### Coolify Note (Config Mount)

If you mount a host path to `/app/config.yaml`, ensure the source is a file.
Mounting a directory to that target can fail container startup (OCI mount type mismatch).

This image is resilient to missing/invalid `/app/config.yaml` and will fall back to
the bundled `/app/default_config.yaml` automatically.

If you want to override config from a directory mount in Coolify, place a file at:

- `/app/config.yaml/config.yaml`

## Backtesting

See backtest/README.md for notebook usage notes.
