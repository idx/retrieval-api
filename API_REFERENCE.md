# Retrieval API Reference

## Overview

The Retrieval API provides OpenAI-compatible endpoints for document reranking and text embedding using state-of-the-art models. This service supports dynamic model selection, efficient caching, and both Japanese and multilingual models.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, authentication is optional. You can include an `Authorization` header with a Bearer token if needed:

```
Authorization: Bearer your-token-here
```

## Models

### Supported Reranker Models

| Model Name | Short Name | Description | Max Length | Language |
|-----------|------------|-------------|------------|----------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | BGE Reranker Base Model v1 (Default) | 512 | Multilingual |
| jinaai/jina-reranker-v2-base-multilingual | jina-reranker-v2 | Jina Reranker v2 Multilingual (100+ languages) | 1024 | Multilingual |
| mixedbread-ai/mxbai-rerank-large-v1 | mxbai-rerank-large | MixedBread AI Rerank Large v1 | 512 | Multilingual |
| hotchpotch/japanese-reranker-cross-encoder-large-v1 | japanese-reranker-large | Japanese Reranker Large (Cross-encoder) | 512 | Japanese |
| hotchpotch/japanese-reranker-cross-encoder-base-v1 | japanese-reranker-base | Japanese Reranker Base (Cross-encoder) | 512 | Japanese |
| pkshatech/GLuCoSE-base-ja | glucose-base-ja | GLuCoSE Base Japanese Model | 512 | Japanese |

### Supported Embedding Models

| Model Name | Short Name | Description | Dimensions | Language |
|-----------|------------|-------------|------------|----------|
| BAAI/bge-m3 | bge-m3 | BGE M3 Multilingual Embedding (Default) | 1024 | Multilingual |
| intfloat/multilingual-e5-large | multilingual-e5-large | Multilingual E5 Large (100+ languages) | 1024 | Multilingual |
| mixedbread-ai/mxbai-embed-large-v1 | mxbai-embed-large | MixedBread AI Embed Large v1 | 1024 | Multilingual |
| cl-nagoya/ruri-large | ruri-large | RURI Large Japanese Embedding | 768 | Japanese |
| cl-nagoya/ruri-base | ruri-base | RURI Base Japanese Embedding | 768 | Japanese |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup | japanese-simcse-large | Japanese SimCSE BERT Large | 1024 | Japanese |
| sonoisa/sentence-luke-japanese-base-lite | sentence-luke-base | Sentence LUKE Japanese Base | 768 | Japanese |

### GET /models

List all available models.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "bce-reranker-base_v1",
      "object": "model",
      "created": 1699123456,
      "owned_by": "huggingface"
    },
    {
      "id": "bge-m3",
      "object": "model",
      "created": 1699123456,
      "owned_by": "huggingface"
    }
  ]
}
```

## Reranking

### POST /v1/rerank

Rerank a list of documents based on their relevance to a query.

#### Request Body

```json
{
  "model": "jina-reranker-v2",
  "query": "日本の伝統的な料理について教えてください",
  "documents": [
    "寿司は日本の代表的な料理で、新鮮な魚を使用します。",
    "今日は天気が良いです。",
    "天ぷらは江戸時代から親しまれている日本料理です。",
    "パリは美しい都市です。"
  ],
  "top_n": 2,
  "return_documents": true
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | No | Model to use for reranking (default: "bce-reranker-base_v1") |
| query | string | Yes | Query string to rank documents against |
| documents | array[string] | Yes | List of documents to rerank (max 1000) |
| top_n | integer | No | Number of top results to return |
| return_documents | boolean | No | Whether to return document texts (default: false) |
| max_chunks_per_doc | integer | No | Maximum chunks per document (not currently used) |

#### Response

```json
{
  "model": "jinaai/jina-reranker-v2-base-multilingual",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9823,
      "document": "寿司は日本の代表的な料理で、新鮮な魚を使用します。"
    },
    {
      "index": 2,
      "relevance_score": 0.9456,
      "document": "天ぷらは江戸時代から親しまれている日本料理です。"
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

## Embeddings

### POST /v1/embeddings

Generate embeddings for given text(s).

#### Request Body

```json
{
  "model": "ruri-large",
  "input": [
    "東京は日本の首都です",
    "機械学習は人工知能の一分野です",
    "自然言語処理は重要な技術です"
  ],
  "encoding_format": "float"
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | No | Model to use for embeddings (default: "bge-m3") |
| input | string or array[string] | Yes | Text(s) to embed |
| encoding_format | string | No | Format of embeddings: "float" or "base64" (default: "float") |
| dimensions | integer | No | Number of dimensions for embeddings (model-dependent) |
| user | string | No | User identifier |

#### Response

```json
{
  "object": "list",
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
    },
    {
      "object": "embedding",
      "index": 2, 
      "embedding": [0.0156, -0.0387, 0.0823, ...]
    }
  ],
  "model": "cl-nagoya/ruri-large",
  "usage": {
    "prompt_tokens": 15,
    "total_tokens": 15
  }
}
```

## Health Check

### GET /health

Check the health status of the API service.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45.123456",
  "model_manager_loaded": true,
  "embedding_manager_loaded": true,
  "default_reranker_model": "maidalun1020/bce-reranker-base_v1",
  "default_embedding_model": "BAAI/bge-m3",
  "loaded_reranker_models": [
    "maidalun1020/bce-reranker-base_v1",
    "jinaai/jina-reranker-v2-base-multilingual"
  ],
  "loaded_embedding_models": [
    "BAAI/bge-m3",
    "cl-nagoya/ruri-large"
  ]
}
```

## Root Endpoint

### GET /

Get basic information about the API service.

#### Response

```json
{
  "message": "Retrieval API Service",
  "version": "1.0.0",
  "endpoints": {
    "rerank": "/v1/rerank",
    "embeddings": "/v1/embeddings",
    "health": "/health",
    "models": "/models"
  },
  "features": [
    "Document reranking",
    "Text embeddings",
    "Japanese language support",
    "Multilingual models",
    "Model pre-loading",
    "Dynamic model switching"
  ]
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": 400
  }
}
```

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Endpoint not found |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Model not loaded |

## Request Examples

### cURL Examples

#### Japanese Reranking

```bash
curl -X POST "http://localhost:8000/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "japanese-reranker-large",
    "query": "日本の伝統文化について",
    "documents": [
      "茶道は日本の伝統的な文化です。",
      "今日は雨が降っています。",
      "歌舞伎は日本の古典芸能の一つです。",
      "コンピューターは便利な道具です。"
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

#### Japanese Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ruri-large",
    "input": [
      "東京は日本の首都です",
      "機械学習は人工知能の一分野です"
    ]
  }'
```

#### Multilingual Models

```bash
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
    "top_n": 3
  }'
```

### Python Examples

#### Using requests for Reranking

```python
import requests

# Japanese reranking
response = requests.post(
    "http://localhost:8000/v1/rerank",
    json={
        "model": "japanese-reranker-base",
        "query": "日本の観光地について教えてください",
        "documents": [
            "富士山は日本で最も有名な観光地の一つです。",
            "今日の天気は晴れです。",
            "京都には多くの歴史的な寺院があります。",
            "東京タワーは東京のランドマークです。"
        ],
        "top_n": 2,
        "return_documents": True
    }
)

result = response.json()
for i, res in enumerate(result['results']):
    print(f"{i+1}. Score: {res['relevance_score']:.4f}")
    print(f"   Document: {res['document']}")
```

#### Using requests for Embeddings

```python
import requests
import numpy as np

# Generate embeddings
response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "bge-m3",
        "input": [
            "Tokyo is the capital of Japan",
            "Machine learning is a subset of AI",
            "Natural language processing is important"
        ]
    }
)

result = response.json()
embeddings = [item['embedding'] for item in result['data']]

# Calculate similarity between first two embeddings
emb1 = np.array(embeddings[0])
emb2 = np.array(embeddings[1])
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Similarity: {similarity:.4f}")
```

### JavaScript Examples

#### Using fetch for Multilingual Reranking

```javascript
const response = await fetch('http://localhost:8000/v1/rerank', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'mxbai-rerank-large',
    query: 'sustainable energy solutions',
    documents: [
      'Solar panels convert sunlight into electricity',
      'Today is a beautiful day',
      'Wind turbines generate clean energy',
      'Electric vehicles reduce carbon emissions'
    ],
    top_n: 3,
    return_documents: true
  })
});

const result = await response.json();
result.results.forEach((item, index) => {
  console.log(`${index + 1}. Score: ${item.relevance_score.toFixed(4)}`);
  console.log(`   Document: ${item.document}`);
});
```

#### Using fetch for Embeddings

```javascript
const response = await fetch('http://localhost:8000/v1/embeddings', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'multilingual-e5-large',
    input: [
      'Artificial intelligence is transforming industries',
      'Deep learning models require large datasets'
    ]
  })
});

const result = await response.json();
console.log(`Generated ${result.data.length} embeddings`);
console.log(`Embedding dimension: ${result.data[0].embedding.length}`);
```

## Model Features

### Pre-loading

- Default models are pre-loaded during service startup
- Subsequent model requests are cached for faster response times
- Models are loaded on-demand when first requested

### Language Support

#### Japanese Models
- Optimized for Japanese text processing
- Support for Japanese-specific tokenization
- Trained on Japanese corpora for better accuracy

#### Multilingual Models
- Support for 100+ languages
- Cross-lingual capabilities
- Consistent performance across different languages

### Dynamic Model Switching

- Switch between models within the same request
- Automatic fallback to default models on errors
- Seamless model management and caching

## Performance Considerations

### Model Loading
- First request to a new model may take 10-30 seconds
- Models are cached in memory after loading
- Pre-loaded models respond immediately

### Batch Processing
- Process multiple documents/texts in single requests
- More efficient than individual requests
- Recommended for production workloads

### Memory Usage
- Each model consumes 500MB-2GB of memory
- Monitor memory usage with multiple models
- Consider model rotation for memory constraints

## Development and Testing

### Local Testing

Test reranking functionality:
```bash
python -c "
import requests
response = requests.post('http://localhost:8000/v1/rerank', 
    json={'query': 'test', 'documents': ['doc1', 'doc2']})
print(response.json())
"
```

Test embedding functionality:
```bash
python -c "
import requests
response = requests.post('http://localhost:8000/v1/embeddings',
    json={'input': ['test text', 'another text']})
print(f'Generated {len(response.json()[\"data\"])} embeddings')
"
```

### Integration Testing

Example comprehensive test:

```python
import requests
import numpy as np

def test_retrieval_api():
    base_url = "http://localhost:8000"
    
    # Test reranking
    rerank_response = requests.post(
        f"{base_url}/v1/rerank",
        json={
            "query": "machine learning",
            "documents": ["ML is AI", "Weather today", "Deep learning"],
            "model": "jina-reranker-v2"
        }
    )
    assert rerank_response.status_code == 200
    rerank_data = rerank_response.json()
    assert len(rerank_data["results"]) == 3
    
    # Test embeddings
    embed_response = requests.post(
        f"{base_url}/v1/embeddings",
        json={
            "input": ["test text 1", "test text 2"],
            "model": "bge-m3"
        }
    )
    assert embed_response.status_code == 200
    embed_data = embed_response.json()
    assert len(embed_data["data"]) == 2
    
    # Test health
    health_response = requests.get(f"{base_url}/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"
    
    print("All tests passed!")

test_retrieval_api()
```

## Deployment Notes

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  retrieval-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RERANKER_MODEL_NAME=jinaai/jina-reranker-v2-base-multilingual
      - EMBEDDING_MODEL_NAME=BAAI/bge-m3
      - RERANKER_MODELS_DIR=/app/models
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### Production Considerations

- **Memory**: Allocate 8GB+ for multiple models
- **Storage**: 20GB+ for model cache persistence  
- **Monitoring**: Track model load times and memory usage
- **Scaling**: Use multiple instances with load balancing
- **Security**: Implement proper authentication and rate limiting
- **Logging**: Configure structured logging for debugging

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| RERANKER_MODEL_NAME | Default reranker model | maidalun1020/bce-reranker-base_v1 |
| EMBEDDING_MODEL_NAME | Default embedding model | BAAI/bge-m3 |
| RERANKER_MODELS_DIR | Model cache directory | /app/models |
| CUDA_VISIBLE_DEVICES | GPU devices (empty for CPU) | "" |