FROM python:3.12-slim

# System deps (tzdata for correct market-hours handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=America/New_York

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY strategy/ ./strategy/
COPY trader.py control.py config.yaml ./

# Logs dir (can also be mounted as a volume)
RUN mkdir -p logs

# Control socket is internal-only; 9999 is exposed for host/sidecar access
EXPOSE 9999

CMD ["python", "trader.py", "--config", "config.yaml"]
