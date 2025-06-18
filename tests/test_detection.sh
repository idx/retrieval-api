#!/bin/bash

# Test script to verify GPU/CPU detection logic

echo "=== GPU Detection Tests (NVIDIA/AMD/CPU) ==="
echo

# Import detection function from start.sh
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

# Test 1: Current GPU detection
echo "Test 1: Current system GPU detection"
GPU_TYPE=$(detect_gpu)
case $GPU_TYPE in
    "nvidia")
        echo "🟢 NVIDIA GPU detected - would use docker-compose.yml"
        ;;
    "amd")
        echo "🔵 AMD GPU detected - would use docker-compose.amd.yml"
        ;;
    "none")
        echo "⚪ No GPU detected - would use docker-compose.cpu.yml"
        ;;
esac
echo

# Test 2: Simulate AMD GPU environment (without NVIDIA)
echo "Test 2: Simulate AMD GPU environment"
detect_gpu_amd_test() {
    # Skip NVIDIA check for this test
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

export HIP_VISIBLE_DEVICES=0
GPU_TYPE_AMD=$(detect_gpu_amd_test)
unset HIP_VISIBLE_DEVICES
if [ "$GPU_TYPE_AMD" = "amd" ]; then
    echo "✓ AMD GPU environment correctly detected"
else
    echo "✗ AMD GPU environment not detected: $GPU_TYPE_AMD"
fi
echo

# Test 3: Test Docker Compose files exist
echo "Test 3: Check all Docker Compose files"
for file in "docker-compose.yml" "docker-compose.cpu.yml" "docker-compose.amd.yml"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done
echo

# Test 4: Test Dockerfiles exist
echo "Test 4: Check Dockerfiles"
for file in "Dockerfile" "Dockerfile.amd"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done
echo

echo "=== Detection Test Complete ==="