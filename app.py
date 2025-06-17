import os
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
import torch

from model_loader import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_manager
    try:
        default_model = os.getenv("RERANKER_MODEL_NAME", "maidalun1020/bce-reranker-base_v1")
        models_base_dir = os.getenv("RERANKER_MODELS_DIR", "/app/models")
        
        logger.info(f"Initializing model manager with default model: {default_model}")
        model_manager = ModelManager(
            default_model=default_model,
            models_base_dir=models_base_dir
        )
        
        # Pre-load default model
        logger.info("Pre-loading default model...")
        model_manager.load_model(default_model)
        logger.info("Model manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model manager: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    if model_manager:
        model_manager.clear_cache()
        logger.info("Model manager cleaned up")


# Initialize FastAPI app
app = FastAPI(
    title="Rerank API",
    description="OpenAI-compatible Rerank API using BGE Reranker",
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




@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Rerank API Service",
        "version": "1.0.0",
        "endpoints": {
            "rerank": "/v1/rerank",
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