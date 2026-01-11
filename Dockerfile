FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY streamlit_app/ ./streamlit_app/

# Environment variables
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "src/bot.py", "--mode", "rebalance"]



