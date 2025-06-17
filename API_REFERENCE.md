# Rerank API Reference

## Overview

The Rerank API provides OpenAI-compatible endpoints for document reranking using BGE Reranker models. This service supports dynamic model selection and efficient caching.

## Base URL

```
http://localhost:7987
```

## Authentication

Currently, authentication is optional. You can include an `Authorization` header with a Bearer token if needed:

```
Authorization: Bearer your-token-here
```

## Models

### Supported Models

| Model Name | Short Name | Description | Max Length |
|-----------|------------|-------------|------------|
| maidalun1020/bce-reranker-base_v1 | bce-reranker-base_v1 | BGE Reranker Base Model v1 (Default) | 512 |
| BAAI/bge-reranker-base | bge-reranker-base | BGE Reranker Base Model | 512 |
| BAAI/bge-reranker-large | bge-reranker-large | BGE Reranker Large Model | 512 |

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
  "model": "bce-reranker-base_v1",
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a branch of artificial intelligence.",
    "Today is a sunny day.",
    "Deep learning is a subset of machine learning."
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
  "model": "bce-reranker-base_v1",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95123,
      "document": "Machine learning is a branch of artificial intelligence."
    },
    {
      "index": 2,
      "relevance_score": 0.87456,
      "document": "Deep learning is a subset of machine learning."
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

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| model | string | Model name used for reranking |
| results | array | Array of reranking results |
| results[].index | integer | Original index of the document in the input array |
| results[].relevance_score | float | Relevance score between 0 and 1 |
| results[].document | string | Document text (only if return_documents=true) |
| meta | object | Metadata about the request |
| meta.api_version | string | API version used |
| meta.processing_time_ms | integer | Processing time in milliseconds |
| meta.total_documents | integer | Total number of input documents |
| meta.returned_documents | integer | Number of documents returned |

## Health Check

### GET /health

Check the health status of the API service.

#### Response

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45.123456",
  "model_manager_loaded": true,
  "default_model": "maidalun1020/bce-reranker-base_v1",
  "loaded_models": [
    "maidalun1020/bce-reranker-base_v1"
  ]
}
```

## Root Endpoint

### GET /

Get basic information about the API service.

#### Response

```json
{
  "message": "Rerank API Service",
  "version": "1.0.0",
  "endpoints": {
    "rerank": "/v1/rerank",
    "health": "/health",
    "models": "/models"
  }
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

### Common Errors

#### 400 Bad Request

```json
{
  "error": {
    "message": "Documents list cannot be empty",
    "type": "validation_error",
    "code": 400
  }
}
```

#### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### 503 Service Unavailable

```json
{
  "error": {
    "message": "Model manager not initialized",
    "type": "service_error",
    "code": 503
  }
}
```

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production use.

## Request Examples

### cURL Examples

#### Basic Reranking

```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bce-reranker-base_v1",
    "query": "machine learning",
    "documents": [
      "ML is a subset of AI",
      "Weather is nice today",
      "Neural networks are powerful"
    ]
  }'
```

#### With Authorization

```bash
curl -X POST "http://localhost:7987/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "model": "bge-reranker-large",
    "query": "artificial intelligence",
    "documents": ["AI doc 1", "AI doc 2"],
    "top_n": 1,
    "return_documents": true
  }'
```

### Python Examples

#### Using requests

```python
import requests

response = requests.post(
    "http://localhost:7987/v1/rerank",
    json={
        "model": "bce-reranker-base_v1",
        "query": "machine learning",
        "documents": [
            "Machine learning is a subset of AI",
            "Today is sunny",
            "Deep learning uses neural networks"
        ],
        "top_n": 2,
        "return_documents": True
    }
)

result = response.json()
print(f"Top result: {result['results'][0]}")
```

#### Using httpx (async)

```python
import httpx
import asyncio

async def rerank_async():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:7987/v1/rerank",
            json={
                "model": "bge-reranker-base",
                "query": "natural language processing",
                "documents": [
                    "NLP is a field of AI",
                    "Cooking is fun",
                    "Text processing is important"
                ]
            }
        )
        return response.json()

result = asyncio.run(rerank_async())
```

### JavaScript Examples

#### Using fetch

```javascript
const response = await fetch('http://localhost:7987/v1/rerank', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'bce-reranker-base_v1',
    query: 'machine learning',
    documents: [
      'ML is a subset of AI',
      'Weather is nice',
      'Neural networks are powerful'
    ],
    top_n: 2
  })
});

const result = await response.json();
console.log(result);
```

#### Using axios

```javascript
const axios = require('axios');

const result = await axios.post('http://localhost:7987/v1/rerank', {
  model: 'bge-reranker-large',
  query: 'artificial intelligence',
  documents: [
    'AI simulates human intelligence',
    'Today is Monday',
    'Machine learning is part of AI'
  ],
  return_documents: true
});

console.log(result.data);
```

## Performance Considerations

### Model Loading

- Models are loaded on first use and cached in memory
- Initial requests may take longer due to model loading
- Subsequent requests with the same model are faster

### Batch Processing

- Process multiple documents in a single request for better performance
- Maximum 1000 documents per request
- Consider breaking larger document sets into batches

### Memory Usage

- Each loaded model consumes GPU/CPU memory
- Models are cached until service restart
- Monitor memory usage in production environments

## Development and Testing

### Local Testing

Use the provided test script:

```bash
python test_api_example.py
```

### Integration Testing

Example test case:

```python
def test_rerank_api():
    response = requests.post(
        "http://localhost:7987/v1/rerank",
        json={
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "model": "bce-reranker-base_v1"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert all("relevance_score" in result for result in data["results"])
```

## Deployment Notes

### Docker Deployment

- Ensure adequate memory allocation (8GB+ recommended)
- Mount model cache directory for persistence
- Configure GPU access if available

### Production Considerations

- Implement proper logging and monitoring
- Set up health checks and alerting
- Consider load balancing for high availability
- Implement rate limiting and authentication
- Monitor model memory usage and performance