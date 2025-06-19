# Retrieval API Service

[🇯🇵 日本語](README.ja.md) | 🇺🇸 English

OpenAI-compatible Retrieval API service providing document reranking and text embeddings using state-of-the-art models with unified model management and Japanese language support.

## Features

- **Dual API Support**: OpenAI-compatible Rerank and Embedding API endpoints
- **Unified Model Management**: Pre-loading, caching, and dynamic switching for both rerankers and embeddings
- **Japanese Language Support**: Specialized models for Japanese text processing
- **Multilingual Models**: Support for 100+ languages with high-performance models
- **Dynamic Model Selection**: Switch between models via API requests with automatic fallback
- **Multi-GPU Support**: NVIDIA CUDA, AMD ROCm with automatic detection
- **CPU Fallback**: Seamless operation without GPU dependencies
- **Docker Deployment**: Easy deployment with multiple Docker configurations
- **Production Ready**: Async processing, memory management, and monitoring

## Supported Models

### Reranking Models

#### Japanese Language Models

| Model Name | Short Name | Max Length | Size | Description |
|-----------|------------|------------|------|-------------|
| hotchpotch/japanese-reranker-cross-encoder-large-v1 | japanese-reranker-large | 512 | 334MB | Japanese Reranker Large v1 (日本語最高性能) |
| hotchpotch/japanese-reranker-cross-encoder-base-v1 | japanese-reranker-base | 512 | 111MB | Japanese Reranker Base v1 (日本語バランス型) |
| pkshatech/GLuCoSE-base-ja | glucose-base-ja | 512 | ~400MB | GLuCoSE Base Japanese Model |

#### Multilingual Models

| Model Name | Short Name | Max Length | Size | Description |
|-----------|------------|------------|------|-------------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | 512 | ~400MB | BGE Reranker Base Model v1 **Default** |
| jinaai/jina-reranker-v2-base-multilingual | jina-reranker-v2 | 1024 | 278MB | Jina Reranker v2 Multilingual (100+ languages) |
| mixedbread-ai/mxbai-rerank-large-v1 | mxbai-rerank-large | 512 | 1.5GB | MixedBread AI Rerank Large v1 (high performance) |

### Embedding Models

#### Japanese Language Models

| Model Name | Short Name | Max Length | Dimensions | Description |
|-----------|------------|------------|------------|-------------|
| cl-nagoya/ruri-large | ruri-large | 512 | 768 | RURI Large Japanese Embedding (JMTEB最高性能) |
| cl-nagoya/ruri-base | ruri-base | 512 | 768 | RURI Base Japanese Embedding (日本語バランス型) |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup | japanese-simcse-large | 512 | 1024 | Japanese SimCSE BERT Large |
| sonoisa/sentence-luke-japanese-base-lite | sentence-luke-base | 512 | 768 | LUKE Japanese Base Lite |
| pkshatech/GLuCoSE-base-ja-v2 | glucose-base-ja-v2 | 512 | 768 | GLuCoSE Japanese v2 |

#### Multilingual Models

| Model Name | Short Name | Max Length | Dimensions | Description |
|-----------|------------|------------|------------|-------------|
| BAAI/bge-m3 | bge-m3 | 8192 | 1024 | BGE M3 Multilingual Embedding **Default** |
| intfloat/multilingual-e5-large | multilingual-e5-large | 512 | 1024 | Multilingual E5 Large (100+ languages) |
| mixedbread-ai/mxbai-embed-large-v1 | mxbai-embed-large | 512 | 1024 | MixedBread AI Large v1 |
| nvidia/NV-Embed-v2 | nv-embed-v2 | 32768 | 4096 | NVIDIA NV-Embed v2 (SOTA performance) |

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
docker build -t retrieval-api .

# Build with proxy support
docker build -t retrieval-api \
  --build-arg HTTP_PROXY=http://proxy.company.com:8080 \
  --build-arg HTTPS_PROXY=http://proxy.company.com:8080 \
  --build-arg NO_PROXY=localhost,127.0.0.1 .

# Build for AMD GPU  
docker build -f docker/Dockerfile.amd -t retrieval-api:amd .

# Build with flexible configuration
docker build -f docker/Dockerfile.flexible --build-arg COMPUTE_MODE=cpu -t retrieval-api:cpu .

# Run with NVIDIA GPU support
docker run -d --name retrieval-api \
  -p 8000:8000 \
  --gpus all \
  retrieval-api

# Run with proxy settings
docker run -d --name retrieval-api \
  -p 8000:8000 \
  --gpus all \
  -e HTTP_PROXY=http://proxy.company.com:8080 \
  -e HTTPS_PROXY=http://proxy.company.com:8080 \
  -e NO_PROXY=localhost,127.0.0.1 \
  retrieval-api

# Run with AMD GPU support
docker run -d --name retrieval-api-amd \
  -p 8000:8000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  retrieval-api:amd

# Run with CPU only
docker run -d --name retrieval-api \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  retrieval-api
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
curl http://localhost:8000/models
```

### Rerank Endpoint

Rerank documents with dynamic model selection:

#### Using Default Model

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
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

#### Using Japanese Models

```bash
# Using Japanese high-performance model
curl -X POST "http://localhost:8000/v1/rerank" \
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
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-base",
    "query": "自然言語処理の技術について",
    "documents": [
      "NLPはコンピュータが人間の言語を理解するのを助けます。",
      "パスタを茹でるにはまずお湯を沸かします。",
      "テキスト解析はNLPの中核的な要素です。"
    ]
  }'
```

#### Using Multilingual Models

```bash
# Using high-performance multilingual model
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jina-reranker-v2",
    "query": "artificial intelligence applications",
    "documents": [
      "AI is used in healthcare for diagnosis",
      "The weather is nice today",
      "Machine learning powers recommendation systems",
      "Natural language processing enables chatbots"
    ],
    "top_n": 3,
    "return_documents": true
  }'

# Using high-performance large model
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mxbai-rerank-large",
    "query": "sustainable energy solutions",
    "documents": [
      "Solar panels convert sunlight into electricity",
      "Today is a beautiful day",
      "Wind turbines generate clean energy",
      "Electric vehicles reduce carbon emissions"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### Response Example

```json
{
  "model": "jinaai/jina-reranker-v2-base-multilingual",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9823,
      "document": "AI is used in healthcare for diagnosis"
    },
    {
      "index": 2,
      "relevance_score": 0.9156,
      "document": "Machine learning powers recommendation systems"
    }
  ],
  "meta": {
    "api_version": "v1",
    "processing_time_ms": 245,
    "total_documents": 4,
    "returned_documents": 2
  }
}
```

### Embeddings API

#### Create Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": "Natural language processing is fascinating."
  }'
```

#### Batch Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": [
      "First text to embed",
      "Second text to embed",
      "Third text to embed"
    ]
  }'
```

#### Using Japanese Models

```bash
# Using Japanese high-performance model
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-large",
    "input": [
      "自然言語処理は人工知能の重要な分野です。",
      "機械学習アルゴリズムは大量のデータを必要とします。",
      "深層学習は多層ニューラルネットワークを使用します。"
    ]
  }'

# Using Japanese balanced model
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-base",
    "input": "日本語のテキスト埋め込みを生成します。"
  }'
```

#### Using Multilingual Models

```bash
# Using high-performance multilingual model
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "multilingual-e5-large",
    "input": [
      "Tokyo is the capital of Japan",
      "Machine learning is a subset of AI",
      "Natural language processing is important"
    ]
  }'

# Using SOTA model with high dimensions
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nv-embed-v2",
    "input": "This model provides state-of-the-art embedding quality with 4096 dimensions."
  }'
```

#### Response Example

```json
{
  "object": "list",
  "model": "BAAI/bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0234, -0.0156, 0.0789, ...]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0412, -0.0298, 0.0634, ...]
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "total_tokens": 16
  }
}
```

### Other Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model List
```bash
curl http://localhost:8000/models
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

## Model Management Features

### Pre-loading and Caching

- **Default Models**: Pre-loaded during service startup for immediate response
- **On-demand Loading**: Models loaded automatically when first requested
- **Memory Caching**: Models cached in memory for subsequent requests
- **Automatic Fallback**: Falls back to default models on loading errors

### Dynamic Model Switching

- **API-level Selection**: Switch models using short names or full model names
- **Unified Management**: Both reranker and embedding models use identical management patterns
- **Error Handling**: Graceful error handling with fallback mechanisms
- **Model Information**: Detailed model metadata available via API

### Language Support

#### Japanese Language Optimization
- **Specialized Tokenization**: Japanese-specific text processing
- **High Performance**: Models trained specifically on Japanese corpora
- **Cultural Context**: Better understanding of Japanese language nuances

#### Multilingual Capabilities
- **100+ Languages**: Support for diverse language processing
- **Cross-lingual**: Consistent performance across different languages
- **Unicode Support**: Full Unicode character set handling

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 0.0.0.0 | Service host |
| PORT | 8000 | Service port |
| WORKERS | 1 | Number of workers |
| RERANKER_MODEL_NAME | maidalun1020/bce-reranker-base_v1 | Default reranker model name |
| EMBEDDING_MODEL_NAME | BAAI/bge-m3 | Default embedding model name |
| RERANKER_MODELS_DIR | /app/models | Base directory for model storage |
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
  -t retrieval-api .

# Build AMD GPU version
docker build -f docker/Dockerfile.amd -t retrieval-api:amd .

# Build CPU-only version
docker build -f docker/Dockerfile.flexible \
  --build-arg COMPUTE_MODE=cpu \
  -t retrieval-api:cpu .
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
├── rerankers/
│   ├── maidalun1020_bce-reranker-base_v1/
│   ├── jinaai_jina-reranker-v2-base-multilingual/
│   └── hotchpotch_japanese-reranker-cross-encoder-large-v1/
└── embeddings/
    ├── BAAI_bge-m3/
    ├── cl-nagoya_ruri-large/
    └── intfloat_multilingual-e5-large/
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

To add custom models, update the `supported_models` dictionary in `reranker_loader.py` or `embedding_loader.py`:

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

- **Efficient Caching**: Models cached after first load for faster subsequent requests
- **Batch Processing**: Multiple documents/texts processed together for improved throughput
- **Memory Monitoring**: Automatic memory cleanup and monitoring
- **Resource Limits**: Configurable memory limits for Docker deployments

## Troubleshooting

### GPU Not Detected

If you encounter the error: `could not select device driver "nvidia" with capabilities: [[gpu]]`

1. Use CPU-only mode:
```bash
# Using docker-compose
docker-compose -f docker/docker-compose.cpu.yml up -d

# Using docker run
docker run -d --name retrieval-api \
  -p 8000:8000 \
  -e CUDA_VISIBLE_DEVICES=-1 \
  retrieval-api

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
import numpy as np

# Reranking example
def rerank_documents(query, documents, model="jina-reranker-v2"):
    response = requests.post("http://localhost:8000/v1/rerank", json={
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": 5,
        "return_documents": True
    })
    return response.json()

# Embedding example
def create_embeddings(texts, model="bge-m3"):
    response = requests.post("http://localhost:8000/v1/embeddings", json={
        "model": model,
        "input": texts
    })
    return response.json()

# Example usage with Japanese models
japanese_query = "人工知能の応用分野について"
japanese_docs = [
    "AIは医療診断で重要な役割を果たしています",
    "今日は良い天気です",
    "機械学習は推薦システムに使われています",
    "自然言語処理はチャットボットを可能にします"
]

# Rerank with Japanese model
rerank_results = rerank_documents(
    japanese_query, 
    japanese_docs, 
    model="japanese-reranker-large"
)

for result in rerank_results["results"]:
    print(f"Score: {result['relevance_score']:.3f} - {result['document']}")

# Generate embeddings with Japanese model
embed_results = create_embeddings(
    ["東京は日本の首都です", "機械学習は人工知能の分野です"],
    model="ruri-large"
)

embeddings = [item['embedding'] for item in embed_results['data']]
print(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function rerankDocuments(query, documents, model = 'jina-reranker-v2') {
  try {
    const response = await axios.post('http://localhost:8000/v1/rerank', {
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

async function createEmbeddings(input, model = 'bge-m3') {
  try {
    const response = await axios.post('http://localhost:8000/v1/embeddings', {
      model,
      input
    });
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Example usage
const query = "sustainable energy solutions";
const docs = [
  "Solar panels convert sunlight into electricity",
  "Today is a beautiful day",
  "Wind turbines generate clean energy",
  "Electric vehicles reduce carbon emissions"
];

// Rerank documents
rerankDocuments(query, docs, 'mxbai-rerank-large').then(results => {
  console.log('Reranking Results:');
  results.results.forEach((result, index) => {
    console.log(`${index + 1}. Score: ${result.relevance_score.toFixed(3)} - ${result.document}`);
  });
});

// Generate embeddings
createEmbeddings([
  "Artificial intelligence is transforming industries",
  "Deep learning models require large datasets"
], 'multilingual-e5-large').then(results => {
  console.log(`Generated ${results.data.length} embeddings`);
  console.log(`Embedding dimension: ${results.data[0].embedding.length}`);
});
```

## License

This project is released under the MIT License.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the proposed changes.

## Support

If you encounter any issues, please report them on the GitHub Issues page.

---

**Note**: This service provides document reranking and text embedding capabilities and is designed for production use with proper monitoring and scaling considerations.