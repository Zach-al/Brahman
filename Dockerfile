# ── Stage 1: Build ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time dependencies only (not shipped to runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install only runtime system deps (wget for model bootstrap, ca-certs for TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder (no build-essential in runtime)
COPY --from=builder /install /usr/local

# SECURITY: Create non-root runtime user
RUN groupadd -r brahman && useradd -r -g brahman -d /app -s /sbin/nologin brahman

# Copy application code with explicit ownership
COPY --chown=brahman:brahman app/ ./app/
COPY --chown=brahman:brahman kernel/ ./kernel/
COPY --chown=brahman:brahman ingestion_engine.py .
COPY --chown=brahman:brahman run.py .

# Create data directory with correct permissions
RUN mkdir -p /app/app/data && chown -R brahman:brahman /app

# Expose ports
EXPOSE 8420
EXPOSE 8080

# SECURITY: Run as non-root user
USER brahman

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8420/health')" || exit 1

# Default: start the Sovereign Node
CMD ["python", "kernel/sovereign_node.py"]
