import os
import time
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
import torch

from model_loader import ModelManager
from embedding_loader import EmbeddingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager: Optional[ModelManager] = None
embedding_models: Dict[str, EmbeddingModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_manager
    try:
        default_model = os.getenv("RERANKER_MODEL_NAME", "maidalun1020/bce-reranker-base_v1")
        models_base_dir = os.getenv("RERANKER_MODELS_DIR", "/app/models")
        default_embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
        
        logger.info(f"Initializing model manager with default model: {default_model}")
        model_manager = ModelManager(
            default_model=default_model,
            models_base_dir=models_base_dir
        )
        
        # Pre-load default model
        logger.info("Pre-loading default reranker model...")
        model_manager.load_model(default_model)
        logger.info("Model manager initialized successfully")
        
        # Pre-load default embedding model
        logger.info(f"Pre-loading default embedding model: {default_embedding_model}")
        embedding_models[default_embedding_model] = EmbeddingModel(
            model_name=default_embedding_model,
            model_dir=os.path.join(models_base_dir, "embeddings", default_embedding_model.replace("/", "_"))
        )
        logger.info("Embedding model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    if model_manager:
        model_manager.clear_cache()
        logger.info("Model manager cleaned up")
    embedding_models.clear()
    logger.info("Embedding models cleaned up")


# Initialize FastAPI app
app = FastAPI(
    title="Rerank & Embedding API",
    description="OpenAI-compatible Rerank and Embedding API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RerankRequest(BaseModel):
    model: str = Field(default="bce-reranker-base_v1", description="Model to use for reranking")
    query: str = Field(..., description="Query string to rank documents against")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of top results to return")
    return_documents: bool = Field(default=False, description="Whether to return document texts")
    max_chunks_per_doc: Optional[int] = Field(default=None, description="Maximum chunks per document")
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        if not v:
            raise ValueError("Documents list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 documents allowed")
        return v
    
    @model_validator(mode='after')
    def validate_top_n(self):
        if self.top_n is not None:
            if self.top_n < 1:
                raise ValueError("top_n must be at least 1")
            if self.documents and self.top_n > len(self.documents):
                self.top_n = len(self.documents)
        return self


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[str] = None


class RerankResponse(BaseModel):
    model: str
    results: List[RerankResult]
    meta: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "bce"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class EmbeddingRequest(BaseModel):
    model: str = Field(default="multilingual-e5-base", description="Model to use for embeddings")
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    encoding_format: Optional[str] = Field(default="float", description="Format of embeddings (float or base64)")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions for embeddings")
    user: Optional[str] = Field(default=None, description="User identifier")
    
    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input text cannot be empty")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            if len(v) > 2048:
                raise ValueError("Maximum 2048 inputs allowed")
            for item in v:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("All inputs must be non-empty strings")
        return v


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: Union[List[float], str]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: List[EmbeddingData]
    usage: EmbeddingUsage


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Rerank API Service",
        "version": "1.0.0",
        "endpoints": {
            "rerank": "/v1/rerank",
            "embeddings": "/v1/embeddings",
            "health": "/health",
            "models": "/models"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_manager_loaded": model_manager is not None,
        "default_model": model_manager.default_model if model_manager else "unknown",
        "loaded_models": list(model_manager.loaded_models.keys()) if model_manager else []
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    models = []
    for model_name in model_manager.get_supported_models():
        model_info = model_manager.get_model_info(model_name)
        models.append(ModelInfo(
            id=model_info["name"],
            created=int(time.time()),
            owned_by="huggingface"
        ))
    
    return ModelsResponse(data=models)


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    authorization: Optional[str] = Header(None)
):
    """Rerank documents based on query relevance"""
    try:
        if not model_manager:
            raise HTTPException(status_code=503, detail="Model manager not initialized")
        
        # Convert model name from short form to full name if needed
        model_name = request.model
        if model_name == "bce-reranker-base_v1":
            model_name = "maidalun1020/bce-reranker-base_v1"
        elif model_name == "bge-reranker-base":
            model_name = "BAAI/bge-reranker-base"
        elif model_name == "bge-reranker-large":
            model_name = "BAAI/bge-reranker-large"
        elif model_name == "japanese-reranker-large":
            model_name = "hotchpotch/japanese-reranker-cross-encoder-large-v1"
        elif model_name == "japanese-reranker-base":
            model_name = "hotchpotch/japanese-reranker-cross-encoder-base-v1"
        elif model_name == "glucose-base-ja":
            model_name = "pkshatech/GLuCoSE-base-ja"
        elif model_name == "jina-reranker-v2":
            model_name = "jinaai/jina-reranker-v2-base-multilingual"
        elif model_name == "mxbai-rerank-large":
            model_name = "mixedbread-ai/mxbai-rerank-large-v1"
        
        # Log request info
        logger.info(f"Rerank request - model: {model_name}, query length: {len(request.query)}, documents: {len(request.documents)}")
        
        # Perform reranking
        start_time = time.time()
        scores = model_manager.compute_scores(model_name, request.query, request.documents)
        
        # Create results with indices and scores
        results = []
        for idx, score in enumerate(scores):
            result = RerankResult(
                index=idx,
                relevance_score=float(score),
                document=request.documents[idx] if request.return_documents else None
            )
            results.append(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply top_n if specified
        if request.top_n is not None:
            results = results[:request.top_n]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        response = RerankResponse(
            model=request.model,
            results=results,
            meta={
                "api_version": "v1",
                "processing_time_ms": int(processing_time * 1000),
                "total_documents": len(request.documents),
                "returned_documents": len(results)
            }
        )
        
        logger.info(f"Rerank completed in {processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Rerank error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    authorization: Optional[str] = Header(None)
):
    """Create embeddings for given text(s)"""
    try:
        # Convert model name from short form to full name if needed
        model_name = request.model
        if model_name == "multilingual-e5-base":
            model_name = "intfloat/multilingual-e5-base"
        elif model_name == "e5-base":
            model_name = "intfloat/e5-base"
        elif model_name == "e5-large":
            model_name = "intfloat/e5-large"
        elif model_name == "multilingual-e5-large":
            model_name = "intfloat/multilingual-e5-large"
        elif model_name == "ruri-large":
            model_name = "cl-nagoya/ruri-large"
        elif model_name == "ruri-base":
            model_name = "cl-nagoya/ruri-base"
        elif model_name == "japanese-simcse-large":
            model_name = "MU-Kindai/Japanese-SimCSE-BERT-large-unsup"
        elif model_name == "sentence-luke-base":
            model_name = "sonoisa/sentence-luke-japanese-base-lite"
        elif model_name == "glucose-base-ja-v2":
            model_name = "pkshatech/GLuCoSE-base-ja-v2"
        elif model_name == "mxbai-embed-large":
            model_name = "mixedbread-ai/mxbai-embed-large-v1"
        elif model_name == "nv-embed-v2":
            model_name = "nvidia/NV-Embed-v2"
        
        # Prepare input
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
            
        # Log request info
        logger.info(f"Embedding request - model: {model_name}, texts: {len(texts)}")
        
        # Generate embeddings using the manager
        start_time = time.time()
        embeddings = embedding_manager.encode(
            model_name,
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build response data
        data = []
        for idx, embedding in enumerate(embeddings):
            if request.encoding_format == "base64":
                # Convert to base64 if requested
                import base64
                import numpy as np
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                embedding_data = embedding_b64
            else:
                # Default to float list
                embedding_data = embedding.tolist()
                
            # Apply dimension reduction if requested
            if request.dimensions and isinstance(embedding_data, list) and len(embedding_data) > request.dimensions:
                embedding_data = embedding_data[:request.dimensions]
                
            data.append(EmbeddingData(
                index=idx,
                embedding=embedding_data
            ))
        
        # Calculate usage (approximate)
        # Assuming ~4 chars per token on average
        total_chars = sum(len(text) for text in texts)
        prompt_tokens = total_chars // 4
        
        processing_time = time.time() - start_time
        
        response = EmbeddingResponse(
            model=request.model,
            data=data,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens
            )
        )
        
        logger.info(f"Embeddings created in {processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": {
            "message": f"Endpoint {request.url.path} not found",
            "type": "not_found_error",
            "code": 404
        }
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return {
        "error": {
            "message": "Internal server error",
            "type": "internal_error",
            "code": 500
        }
    }