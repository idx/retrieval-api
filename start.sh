#!/bin/bash

# Script to automatically start the rerank API with GPU if available, otherwise CPU

echo "Checking Docker permissions..."

# Check if user has Docker permissions
if ! docker info &> /dev/null; then
    echo "❌ Docker permission denied. Please run one of the following:"
    echo "   sudo docker compose up -d"
    echo "   OR add your user to docker group:"
    echo "   sudo usermod -aG docker \$USER && newgrp docker"
    echo "   OR use sudo with the script:"
    echo "   sudo ./start.sh"
    exit 1
fi

echo "✅ Docker permissions OK"
echo "Checking for GPU availability..."

# Function to detect GPU type
detect_gpu() {
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "nvidia"
        return
    fi
    
    # Check for AMD GPU
    if command -v rocm-smi &> /dev/null && rocm-smi &> /dev/null; then
        echo "amd"
        return
    fi
    
    # Check for AMD GPU via alternative methods
    if [ -d "/opt/rocm" ] || [ -n "$ROCR_VISIBLE_DEVICES" ] || [ -n "$HIP_VISIBLE_DEVICES" ]; then
        echo "amd"
        return
    fi
    
    # Check for AMD GPU devices
    if ls /dev/kfd &> /dev/null || ls /dev/dri/card* &> /dev/null; then
        echo "amd"
        return
    fi
    
    echo "none"
}

GPU_TYPE=$(detect_gpu)

case $GPU_TYPE in
    "nvidia")
        echo "🟢 NVIDIA GPU detected! Starting with NVIDIA GPU support..."
        COMPOSE_FILE="docker-compose.yml"
        CONTAINER_NAME="rerank-api"
        docker-compose up -d
        ;;
    "amd")
        echo "🔵 AMD GPU detected! Starting with AMD GPU support..."
        COMPOSE_FILE="docker-compose.amd.yml"
        CONTAINER_NAME="rerank-api-amd"
        docker-compose -f docker-compose.amd.yml up -d
        ;;
    "none")
        echo "⚪ No GPU detected. Starting with CPU support..."
        COMPOSE_FILE="docker-compose.cpu.yml"
        CONTAINER_NAME="rerank-api"
        docker-compose -f docker-compose.cpu.yml up -d
        ;;
esac

# Show container status
echo ""
echo "Container status:"
docker ps | grep rerank-api

echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker-compose -f $COMPOSE_FILE down"