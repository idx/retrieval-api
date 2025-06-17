#!/bin/bash

# Script to automatically start the rerank API with GPU if available, otherwise CPU

echo "Checking for GPU availability..."

# Check if nvidia-smi is available and working
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected! Starting with GPU support..."
    docker-compose up -d
else
    echo "No GPU detected or NVIDIA driver not available. Starting with CPU support..."
    docker-compose -f docker-compose.cpu.yml up -d
fi

# Show container status
echo ""
echo "Container status:"
docker ps | grep rerank-api

echo ""
echo "To view logs: docker logs -f rerank-api"
echo "To stop: docker-compose down (or docker-compose -f docker-compose.cpu.yml down for CPU mode)"