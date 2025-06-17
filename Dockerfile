FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models && chown -R appuser:appuser models

# Copy application files
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 7987

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7987
ENV WORKERS=1
ENV RERANKER_MODEL_NAME=maidalun1020/bce-reranker-base_v1
ENV RERANKER_MODEL_DIR=/app/models/bce-reranker-base_v1

# Run the application
CMD ["python", "run.py"]