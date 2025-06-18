FROM python:3.10-slim

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

# Install Python dependencies
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
ENV RERANKER_MODEL_NAME=maidalun1020/bce-reranker-base_v1
ENV RERANKER_MODELS_DIR=/app/models
ENV EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
ENV HF_HOME=/home/appuser/.cache/huggingface

# Run the application
CMD ["python", "run.py"]