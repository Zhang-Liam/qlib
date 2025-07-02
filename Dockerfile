FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY examples/production_requirements.txt .
RUN pip install --no-cache-dir -r production_requirements.txt

# Copy the qlib package
COPY qlib/ ./qlib/

# Copy configuration files
COPY config/ ./config/
COPY examples/ ./examples/

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV BROKER_TYPE=mock

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import qlib.production; print('Health check passed')" || exit 1

# Default command: run the production workflow
CMD ["python", "-m", "qlib.production.workflow", "--config", "config/production_config.yaml", "--continuous"]
