FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full application
COPY app/ ./app/
COPY kernel/ ./kernel/
COPY ingestion_engine.py .
COPY run.py .

# Expose Sovereign Node port
EXPOSE 8420

# Default: start the Sovereign Node
CMD ["python", "kernel/sovereign_node.py"]
