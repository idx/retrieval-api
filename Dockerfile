FROM python:3.13-slim

# Set proxy environment variables if provided
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user with home directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser/.cache

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies with proxy support
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory with proper permissions
RUN mkdir -p models && chmod 755 models && chown -R appuser:appuser models

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
ENV RERANKER_MODEL_NAME=jinaai/jina-reranker-v2-base-multilingual
ENV RERANKER_MODELS_DIR=/app/models
ENV EMBEDDING_MODEL_NAME=BAAI/bge-m3
ENV HF_HOME=/home/appuser/.cache/huggingface

# Run the application
CMD ["python", "run.py"]