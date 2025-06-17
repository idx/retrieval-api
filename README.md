# Rerank API Service

[🇯🇵 日本語](README.ja.md) | 🇺🇸 English

OpenAI-compatible Rerank API service using BGE Reranker models for high-precision document reranking.

## Features

- OpenAI-compatible Rerank API endpoints
- Dynamic model selection via API requests
- High-precision document reranking using BGE Reranker models
- Multi-GPU support (NVIDIA CUDA, AMD ROCm) with automatic detection
- CPU fallback support
- Easy deployment with Docker
- Async processing for high-speed responses
- Model caching and efficient memory management

## Supported Models

This API supports multiple reranking models that can be dynamically selected:

| Model Name | Short Name | Description |
|-----------|------------|-------------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | BGE Reranker Base Model v1 (Default) |
| BAAI/bge-reranker-base | bge-reranker-base | BGE Reranker Base Model |
| BAAI/bge-reranker-large | bge-reranker-large | BGE Reranker Large Model |

## Quick Start

### Docker Deployment

#### Automatic GPU/CPU Detection

Use the provided start script for automatic detection:

```bash
# Make script executable
chmod +x start.sh

# Start with automatic GPU/CPU detection
./start.sh
```

#### Manual Docker Commands

```bash
# Build for NVIDIA GPU
docker build -t rerank-api .

# Build for AMD GPU  
docker build -f Dockerfile.amd -t rerank-api:amd .

# Run with NVIDIA GPU support
docker run -d --name rerank-api \
  -p 7987:7987 \
  --gpus all \
  rerank-api

# Run with AMD GPU support
docker run -d --name rerank-api-amd \
  -p 7987:7987 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  rerank-api:amd

# Run with CPU only
docker run -d --name rerank-api \
  -p 7987:7987 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  rerank-api
```

#### Docker Compose

```bash
# NVIDIA GPU support
docker-compose up -d

# AMD GPU support
docker-compose -f docker-compose.amd.yml up -d

# CPU only mode
docker-compose -f docker-compose.cpu.yml up -d
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start service
python run.py
```

## API Usage

### Available Models

Check available models:

```bash
curl http://localhost:7987/models
```

### Rerank Endpoint

Rerank documents with dynamic model selection:

#### Using Default Model

```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bce-reranker-base_v1",
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a branch of artificial intelligence.",
      "Today is a beautiful sunny day.",
      "Deep learning is a method of machine learning."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### Using Different Models

```bash
# Using large model
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-large",
    "query": "What is artificial intelligence?",
    "documents": [
      "AI simulates human intelligence in machines.",
      "The weather forecast shows rain tomorrow.",
      "Machine learning is a subset of AI technology."
    ],
    "top_n": 2,
    "return_documents": true
  }'

# Using full model name
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-base",
    "query": "Natural language processing",
    "documents": [
      "NLP helps computers understand human language.",
      "Cooking pasta requires boiling water first.",
      "Text analysis is a core component of NLP."
    ]
  }'
```

#### Response Example

```json
{
  "model": "bce-reranker-base_v1",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "Machine learning is a branch of artificial intelligence."
    },
    {
      "index": 2,
      "relevance_score": 0.87,
      "document": "Deep learning is a method of machine learning."
    }
  ],
  "meta": {
    "api_version": "v1",
    "processing_time_ms": 145,
    "total_documents": 3,
    "returned_documents": 2
  }
}
```

### Other Endpoints

#### Health Check
```bash
curl http://localhost:7987/health
```

#### Model List
```bash
curl http://localhost:7987/models
```

## API Specification

### POST /v1/rerank

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | No | Model to use (short name or full name, default: "bce-reranker-base_v1") |
| query | string | Yes | Query string for ranking documents |
| documents | array[string] | Yes | List of documents to rerank (max 1000) |
| top_n | integer | No | Number of top results to return |
| return_documents | boolean | No | Whether to include document texts (default: false) |

#### Response

| Field | Type | Description |
|-------|------|-------------|
| model | string | Model name used |
| results | array | List of ranking results |
| results[].index | integer | Original document index |
| results[].relevance_score | float | Relevance score (0-1) |
| results[].document | string | Document text (if return_documents=true) |
| meta | object | Metadata |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 0.0.0.0 | Service host |
| PORT | 7987 | Service port |
| WORKERS | 1 | Number of workers |
| RERANKER_MODEL_NAME | maidalun1020/bce-reranker-base_v1 | Default model name |
| RERANKER_MODELS_DIR | /app/models | Base directory for model storage |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=.

# Run specific test
python -m pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy app.py
```

### Test API Manually

Use the included test script:

```bash
python test_api_example.py
```

## Docker Configuration

### Build Arguments

```bash
# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.10 -t rerank-api .

# Build for production
docker build --target production -t rerank-api:prod .
```

### GPU Support

For GPU support, ensure you have:

1. NVIDIA drivers installed
2. NVIDIA Container Toolkit installed
3. Docker configured for GPU access

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Model Management

### Model Caching

Models are automatically cached after first load. The cache directory structure:

```
/app/models/
├── maidalun1020_bce-reranker-base_v1/
├── BAAI_bge-reranker-base/
└── BAAI_bge-reranker-large/
```

### Custom Models

To add custom models, update the `supported_models` dictionary in `model_loader.py`:

```python
self.supported_models = {
    "your-custom/model-name": {
        "name": "custom-model",
        "description": "Your Custom Model",
        "max_length": 512
    }
}
```

## Performance Optimization

### GPU Configuration

#### NVIDIA GPU Support
- NVIDIA drivers (latest version recommended)
- CUDA 11.8+ support
- GPU memory 4GB+ recommended
- NVIDIA Container Toolkit for Docker

#### AMD GPU Support  
- ROCm 6.0+ support
- AMD GPU drivers (AMDGPU-PRO or open-source)
- GPU memory 4GB+ recommended
- Docker with AMD GPU device access (`/dev/kfd`, `/dev/dri`)

#### Automatic Detection
The service automatically detects available GPU hardware:
- 🟢 NVIDIA GPU → Uses CUDA acceleration
- 🔵 AMD GPU → Uses ROCm acceleration  
- ⚪ No GPU → Falls back to CPU

### Memory Management

- Efficient model caching
- Batch processing for improved throughput
- Configurable worker count
- Automatic memory cleanup

## Troubleshooting

### GPU Not Detected

If you encounter the error: `could not select device driver "nvidia" with capabilities: [[gpu]]`

1. Use CPU-only mode:
```bash
# Using docker-compose
docker-compose -f docker-compose.cpu.yml up -d

# Using docker run
docker run -d --name rerank-api \
  -p 7987:7987 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  rerank-api

# Or use the automatic start script
./start.sh
```

2. To fix GPU support, check:
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Slow Model Downloads

Use Hugging Face mirrors:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Memory Issues

For Docker deployments, increase memory limits:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

### Model Loading Errors

1. Check disk space
2. Verify network connectivity to Hugging Face Hub
3. Check model name spelling
4. Review logs for detailed error messages

## API Examples

### Python Client Example

```python
import requests

def rerank_documents(query, documents, model="bce-reranker-base_v1"):
    response = requests.post("http://localhost:7987/v1/rerank", json={
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": 5,
        "return_documents": True
    })
    return response.json()

# Example usage
query = "machine learning algorithms"
docs = [
    "Machine learning is a subset of artificial intelligence",
    "Today's weather is sunny and warm",
    "Neural networks are powerful ML algorithms",
    "Cooking requires fresh ingredients"
]

results = rerank_documents(query, docs)
for result in results["results"]:
    print(f"Score: {result['relevance_score']:.3f} - {result['document']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function rerankDocuments(query, documents, model = 'bce-reranker-base_v1') {
  try {
    const response = await axios.post('http://localhost:7987/v1/rerank', {
      model,
      query,
      documents,
      top_n: 5,
      return_documents: true
    });
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Example usage
const query = "machine learning algorithms";
const docs = [
  "Machine learning is a subset of artificial intelligence",
  "Today's weather is sunny and warm",
  "Neural networks are powerful ML algorithms",
  "Cooking requires fresh ingredients"
];

rerankDocuments(query, docs).then(results => {
  results.results.forEach(result => {
    console.log(`Score: ${result.relevance_score.toFixed(3)} - ${result.document}`);
  });
});
```

## License

This project is released under the MIT License.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed changes.

## Support

If you encounter any issues, please report them on the GitHub Issues page.

---

**Note**: This service provides document reranking capabilities and is designed for production use with proper monitoring and scaling considerations.