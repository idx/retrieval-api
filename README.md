# Rerank & Embedding API Service

[🇯🇵 日本語](README.ja.md) | 🇺🇸 English

OpenAI-compatible Rerank and Embedding API service using BGE Reranker models for document reranking and sentence-transformers for text embeddings.

## Features

- OpenAI-compatible Rerank and Embedding API endpoints
- Dynamic model selection via API requests
- High-precision document reranking using BGE Reranker models
- Text embeddings generation using sentence-transformers
- Multi-GPU support (NVIDIA CUDA, AMD ROCm) with automatic detection
- CPU fallback support
- Easy deployment with Docker
- Async processing for high-speed responses
- Model caching and efficient memory management

## Supported Models

### Reranking Models

#### Japanese Language Models

| Model Name | Short Name | Max Length | Size | Description |
|-----------|------------|------------|------|-------------|
| hotchpotch/japanese-reranker-cross-encoder-large-v1 | japanese-reranker-large | 512 | 334MB | Japanese Reranker Large v1 (日本語最高性能) |
| hotchpotch/japanese-reranker-cross-encoder-base-v1 | japanese-reranker-base | 512 | 111MB | Japanese Reranker Base v1 (日本語バランス型) |
| hotchpotch/japanese-reranker-cross-encoder-small-v1 | japanese-reranker-small | 512 | 67MB | Japanese Reranker Small v1 (高速推論) |
| hotchpotch/japanese-bge-reranker-v2-m3-v1 | japanese-bge-v2-m3 | 8192 | ~500MB | Japanese BGE Reranker v2-M3 v1 (日本語特化版) |

#### Multilingual Models

| Model Name | Short Name | Max Length | Size | Description |
|-----------|------------|------------|------|-------------|
| jinaai/jina-reranker-v2-base-multilingual | jina-reranker-v2-multilingual | 1024 | 278MB | Jina Reranker v2 Multilingual (100+ languages) **Default** |
| BAAI/bge-reranker-v2-m3 | bge-reranker-v2-m3 | 32000 | ~600MB | BGE Reranker v2 M3 (Multilingual, up to 32k tokens) |
| Alibaba-NLP/gte-multilingual-reranker-base | gte-multilingual-reranker | 8192 | 560MB | GTE Multilingual Reranker (70+ languages) |
| mixedbread-ai/mxbai-rerank-large-v1 | mxbai-rerank-large | 8192 | 1.5GB | MixedBread AI Rerank Large v1 (high performance) |
| Cohere/rerank-multilingual-v3.0 | cohere-rerank-multilingual | 4096 | ~400MB | Cohere Rerank Multilingual v3.0 |

#### English Language Models

| Model Name | Short Name | Max Length | Size | Description |
|-----------|------------|------------|------|-------------|
| jinaai/jina-reranker-v1-turbo-en | jina-reranker-turbo | 8192 | 37.8MB | Jina Reranker v1 Turbo (fast inference) |

#### Legacy Models

| Model Name | Short Name | Max Length | Size | Description |
|-----------|------------|------------|------|-------------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | 512 | ~400MB | BGE Reranker Base Model v1 (Legacy) |

### Embedding Models

#### Japanese Language Models

| Model Name | Short Name | Max Length | Dimensions | Description |
|-----------|------------|------------|------------|-------------|
| cl-nagoya/ruri-large | ruri-large | 512 | 1024 | Ruri Large Japanese Embedding (JMTEB最高性能) |
| cl-nagoya/ruri-base | ruri-base | 512 | 768 | Ruri Base Japanese Embedding (日本語バランス型) |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup | japanese-simcse-large | 512 | 1024 | Japanese SimCSE BERT Large (教師なし学習) |
| sonoisa/sentence-luke-japanese-base-lite | luke-japanese-base | 512 | 768 | LUKE Japanese Base Lite (知識強化型) |
| pkshatech/GLuCoSE-base-ja-v2 | glucose-ja-v2 | 512 | 768 | GLuCoSE Japanese v2 (企業開発) |

#### Multilingual Models

| Model Name | Short Name | Max Length | Dimensions | Description |
|-----------|------------|------------|------------|-------------|
| BAAI/bge-m3 | bge-m3 | 8192 | 1024 | BGE M3 Multilingual Embedding **Default** |
| nvidia/NV-Embed-v2 | nv-embed-v2 | 32768 | 4096 | NVIDIA NV-Embed v2 (SOTA performance) |
| intfloat/e5-mistral-7b-instruct | e5-mistral-7b | 32768 | 4096 | E5 Mistral 7B Instruct (high quality) |
| mixedbread-ai/mxbai-embed-large-v1 | mxbai-embed-large | 512 | 1024 | MixedBread AI Large v1 (production ready) |
| intfloat/multilingual-e5-large | multilingual-e5-large | 512 | 1024 | Multilingual E5 Large (100+ languages) |
| intfloat/multilingual-e5-base | multilingual-e5-base | 512 | 768 | Multilingual E5 Base (Legacy) |

#### English Language Models

| Model Name | Short Name | Max Length | Dimensions | Description |
|-----------|------------|------------|------------|-------------|
| sentence-transformers/all-mpnet-base-v2 | all-mpnet-base-v2 | 514 | 768 | All MPNet Base v2 (balanced performance) |
| sentence-transformers/all-MiniLM-L6-v2 | all-minilm-l6-v2 | 256 | 384 | All MiniLM L6 v2 (fast and efficient) |
| intfloat/e5-base | e5-base | 512 | 768 | E5 Base Model (English optimized) |
| intfloat/e5-large | e5-large | 512 | 1024 | E5 Large Model (English optimized) |

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

# Build with proxy support
docker build -t rerank-api \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 .

# Build for AMD GPU  
docker build -f docker/Dockerfile.amd -t rerank-api:amd .

# Build with flexible configuration
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t rerank-api:cpu .

# Run with NVIDIA GPU support
docker run -d --name rerank-api \
  -p 7987:7987 \
  --gpus all \
  rerank-api

# Run with proxy settings
docker run -d --name rerank-api \
  -p 7987:7987 \
  --gpus all \
  -e HTTP_PROXY=http://proxy.company.com:8080 \
  -e HTTPS_PROXY=http://proxy.company.com:8080 \
  -e NO_PROXY=localhost,127.0.0.1 \
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

# With proxy settings
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1
docker-compose up -d

# AMD GPU support
docker-compose -f docker/docker-compose.amd.yml up -d

# CPU only mode
docker-compose -f docker/docker-compose.cpu.yml up -d
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
    "model": "jina-reranker-v2-multilingual",
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
# Using Japanese high-performance model
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-large",
    "query": "人工知能とは何ですか？",
    "documents": [
      "AIは機械で人間の知能を模倣します。",
      "明日の天気予報は雨です。",
      "機械学習はAI技術の一部です。"
    ],
    "top_n": 2,
    "return_documents": true
  }'

# Using Japanese balanced model
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-base",
    "query": "自然言語処理",
    "documents": [
      "NLPはコンピュータが人間の言語を理解するのを助けます。",
      "パスタを茹でるにはまずお湯を沸かします。",
      "テキスト解析はNLPの中核的な要素です。"
    ]
  }'

# Using high-performance large model (8k context)
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mxbai-rerank-large",
    "query": "What is artificial intelligence?",
    "documents": [
      "AI simulates human intelligence in machines.",
      "The weather forecast shows rain tomorrow.",
      "Machine learning is a subset of AI technology."
    ],
    "top_n": 2,
    "return_documents": true
  }'

# Using ultra-long context model (32k tokens)
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "Natural language processing",
    "documents": [
      "NLP helps computers understand human language.",
      "Cooking pasta requires boiling water first.",
      "Text analysis is a core component of NLP."
    ]
  }'

# Using fast turbo model for quick inference
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-turbo",
    "query": "Machine learning algorithms",
    "documents": [
      "Deep learning uses neural networks with multiple layers.",
      "Today's lunch menu includes sandwiches and salads.",
      "Support vector machines are powerful ML algorithms."
    ]
  }'
```

#### Response Example

```json
{
  "model": "jina-reranker-v2-multilingual",
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

### Embeddings API

#### Create Embeddings

```bash
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": "Natural language processing is fascinating."
  }'
```

#### Batch Embeddings

```bash
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": [
      "First text to embed",
      "Second text to embed",
      "Third text to embed"
    ]
  }'

# Using Japanese high-performance model
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-large",
    "input": "自然言語処理は人工知能の重要な分野です。"
  }'

# Using Japanese balanced model
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-base",
    "input": "日本語のテキスト埋め込みを生成します。"
  }'

# Using high-performance SOTA model (32k context)
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nv-embed-v2",
    "input": "This is a very long document that requires high-quality embeddings with support for extended context lengths up to 32,768 tokens."
  }'

# Using efficient lightweight model
curl -X POST "http://localhost:7987/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-minilm-l6-v2",
    "input": "Quick embedding for fast inference."
  }'
```

#### Response Example

```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, 0.789, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
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
| model | string | No | Model to use (short name or full name, default: "jina-reranker-v2-multilingual") |
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

### POST /v1/embeddings

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | No | Model to use (short name or full name, default: "bge-m3") |
| input | string or array[string] | Yes | Text(s) to embed (max 2048 texts) |
| encoding_format | string | No | Format for embeddings ("float" or "base64", default: "float") |
| dimensions | integer | No | Number of dimensions to reduce embeddings to |
| user | string | No | User identifier |

#### Response

| Field | Type | Description |
|-------|------|-------------|
| object | string | Always "list" |
| model | string | Model name used |
| data | array | List of embedding objects |
| data[].object | string | Always "embedding" |
| data[].index | integer | Index of the input text |
| data[].embedding | array[float] or string | Embedding vector (float array or base64 string) |
| usage | object | Token usage information |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 0.0.0.0 | Service host |
| PORT | 7987 | Service port |
| WORKERS | 1 | Number of workers |
| RERANKER_MODEL_NAME | jinaai/jina-reranker-v2-base-multilingual | Default reranker model name |
| RERANKER_MODELS_DIR | /app/models | Base directory for model storage |
| EMBEDDING_MODEL_NAME | BAAI/bge-m3 | Default embedding model name |
| HTTP_PROXY | - | HTTP proxy server URL |
| HTTPS_PROXY | - | HTTPS proxy server URL |
| NO_PROXY | - | Comma-separated list of hosts to bypass proxy |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=.

# Run specific test
python -m pytest tests/test_api.py -v

# Run API example test
python tests/test_api_example.py

# Run hardware detection test
bash tests/test_detection.sh
```

### Docker Testing

```bash
# Test different Docker configurations
docker-compose up -d                                      # NVIDIA GPU (root)
docker-compose -f docker/docker-compose.yml up -d         # NVIDIA GPU (docker/)
docker-compose -f docker/docker-compose.amd.yml up -d     # AMD GPU
docker-compose -f docker/docker-compose.cpu.yml up -d     # CPU only

# Test with specific Docker files
docker build -f docker/Dockerfile.amd -t test:amd .
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t test:cpu .
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
python tests/test_api_example.py
```

## Docker Configuration

### Build Arguments

```bash
# Build with proxy support
docker build \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t rerank-api .

# Build AMD GPU version
docker build -f docker/Dockerfile.amd -t rerank-api:amd .

# Build CPU-only version
docker build -f docker/Dockerfile.flexible \
  --build-arg COMPUTE_MODE=cpu \
  -t rerank-api:cpu .
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

### Docker File Structure

The project includes multiple Docker configurations:

```
docker/
├── Dockerfile                  # Standard NVIDIA GPU build
├── Dockerfile.amd             # AMD ROCm GPU support
├── Dockerfile.flexible        # CPU/GPU flexible build
├── docker-compose.yml         # Standard compose file
├── docker-compose.amd.yml     # AMD GPU compose
├── docker-compose.cpu.yml     # CPU-only compose
├── requirements.txt           # Standard requirements
├── requirements.amd.txt       # AMD-specific requirements
└── requirements-cpu.txt       # CPU-only requirements
```

**Note**: For convenience, the main `docker-compose.yml` is also available in the root directory.

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
docker-compose -f docker/docker-compose.cpu.yml up -d

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