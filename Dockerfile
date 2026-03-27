FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build tools for scientific Python packages when wheels are unavailable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy only runtime files required by live/backtest execution.
COPY src ./src
COPY live.py ./
COPY backtest.py ./
COPY cli.py ./
COPY config.yaml ./

# Run as non-root user.
RUN useradd -m appuser \
    && mkdir -p /app/models /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# Ensure orchestrators send SIGINT for graceful shutdown handling.
STOPSIGNAL SIGINT

CMD ["python", "live.py"]
