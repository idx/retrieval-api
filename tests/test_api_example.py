#!/usr/bin/env python3
"""Test script to demonstrate the new dynamic model selection API"""

import requests
import json

# Test different models
test_models = [
    "bce-reranker-base_v1",  # default model (short name)
    "maidalun1020/bce-reranker-base_v1",  # full name
    "bge-reranker-base",  # alternative model (short name)
    "BAAI/bge-reranker-base"  # alternative model (full name)
]

query = "What is artificial intelligence?"
documents = [
    "Artificial intelligence is the simulation of human intelligence in machines.",
    "Today is a beautiful sunny day.",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers."
]

base_url = "http://localhost:7987"

def test_models_endpoint():
    """Test the /models endpoint"""
    print("Testing /models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        if response.status_code == 200:
            data = response.json()
            print("Available models:")
            for model in data.get("data", []):
                print(f"  - {model['id']} (owned by: {model['owned_by']})")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")

def test_rerank_with_model(model_name):
    """Test reranking with a specific model"""
    print(f"\nTesting rerank with model: {model_name}")
    
    payload = {
        "model": model_name,
        "query": query,
        "documents": documents,
        "top_n": 3,
        "return_documents": True
    }
    
    try:
        response = requests.post(f"{base_url}/v1/rerank", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"Model used: {data['model']}")
            print(f"Processing time: {data['meta']['processing_time_ms']}ms")
            print("Top results:")
            for i, result in enumerate(data['results'][:3]):
                print(f"  {i+1}. Score: {result['relevance_score']:.3f} - {result['document'][:50]}...")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    print("Dynamic Model Selection API Test")
    print("=" * 40)
    
    # Test models endpoint
    test_models_endpoint()
    
    # Test reranking with different models
    for model in test_models:
        test_rerank_with_model(model)
    
    print("\nTest completed!")