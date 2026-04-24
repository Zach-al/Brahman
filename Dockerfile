FROM python:3.11-slim

# System deps for torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Generate the SQLite database natively during Docker build
COPY ingestion_engine.py .
RUN python -c "from ingestion_engine import BrahmanIngestion; BrahmanIngestion('app/data/brahman_v2.db').ingest_dhatupatha()"

# Copy test suite
COPY tests/ ./tests/

# Expose port
EXPOSE 8000

# Hardware optimization for 8 vCPUs
ENV TORCH_NUM_THREADS=8

# Production server: Uvicorn workers
# Railway sets PORT via env var, default 8000
# Copy the nuclear entrypoint
COPY run.py .

CMD ["python", "run.py"]
