import os
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch

from model_loader import BGERerankerLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Rerank API",
    description="OpenAI-compatible Rerank API using BGE Reranker",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader instance
model_loader: Optional[BGERerankerLoader] = None


class RerankRequest(BaseModel):
    model: str = Field(default="bce-reranker-base_v1", description="Model to use for reranking")
    query: str = Field(..., description="Query string to rank documents against")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of top results to return")
    return_documents: bool = Field(default=False, description="Whether to return document texts")
    max_chunks_per_doc: Optional[int] = Field(default=None, description="Maximum chunks per document")
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v:
            raise ValueError("Documents list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 documents allowed")
        return v
    
    @validator('top_n')
    def validate_top_n(cls, v, values):
        if v is not None:
            if v < 1:
                raise ValueError("top_n must be at least 1")
            if 'documents' in values and v > len(values['documents']):
                v = len(values['documents'])
        return v


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


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_loader
    try:
        model_name = os.getenv("RERANKER_MODEL_NAME", "maidalun1020/bce-reranker-base_v1")
        model_dir = os.getenv("RERANKER_MODEL_DIR", "/app/models/bce-reranker-base_v1")
        
        logger.info(f"Loading reranker model: {model_name}")
        model_loader = BGERerankerLoader(model_name=model_name, model_dir=model_dir)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


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
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model_loader is not None,
        "device": str(model_loader.device) if model_loader else "unknown"
    }


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models"""
    models = [
        ModelInfo(
            id="bce-reranker-base_v1",
            created=int(time.time()),
            owned_by="bce"
        )
    ]
    return ModelsResponse(data=models)


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    authorization: Optional[str] = Header(None)
):
    """Rerank documents based on query relevance"""
    try:
        if not model_loader:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Log request info
        logger.info(f"Rerank request - query length: {len(request.query)}, documents: {len(request.documents)}")
        
        # Perform reranking
        start_time = time.time()
        scores = model_loader.compute_scores(request.query, request.documents)
        
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