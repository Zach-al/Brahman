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

# Copy test suite
COPY tests/ ./tests/

# Expose port
EXPOSE 8000

# Production server: Gunicorn with Uvicorn workers
# Railway sets PORT via env var, default 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 8
