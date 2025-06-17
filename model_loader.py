import os
import logging
from typing import List, Union, Optional, Dict
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class RerankerModel:
    """Single reranker model loader and scorer"""
    
    def __init__(self, model_name: str = "maidalun1020/bce-reranker-base_v1", 
                 model_dir: Optional[str] = None, 
                 max_length: int = 512):
        """
        Initialize BGE Reranker model
        
        Args:
            model_name: Hugging Face model name
            model_dir: Local directory to cache model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing BGE Reranker on device: {self.device}")
        
        # Try to load from local directory first if specified
        if model_dir and os.path.exists(model_dir):
            logger.info(f"Loading model from local directory: {model_dir}")
            model_path = model_dir
        else:
            logger.info(f"Loading model from Hugging Face Hub: {model_name}")
            model_path = model_name
            
            # Create model directory if specified
            if model_dir:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Load model using sentence-transformers CrossEncoder
            self.model = CrossEncoder(
                model_path,
                max_length=self.max_length,
                device=self.device.type
            )
            
            # Save model locally if directory specified and not already saved
            if model_dir and not os.path.exists(os.path.join(model_dir, "config.json")):
                logger.info(f"Saving model to local directory: {model_dir}")
                self.model.save(model_dir)
                
        except Exception as e:
            logger.error(f"Failed to load with CrossEncoder, trying direct loading: {str(e)}")
            
            # Fallback to direct transformers loading
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Save locally if needed
                if model_dir and not os.path.exists(os.path.join(model_dir, "config.json")):
                    logger.info(f"Saving model to local directory: {model_dir}")
                    self.tokenizer.save_pretrained(model_dir)
                    self.model.save_pretrained(model_dir)
                    
                # Set flag for direct inference
                self._use_direct_inference = True
            except Exception as e2:
                logger.error(f"Failed to load model: {str(e2)}")
                raise
        else:
            self._use_direct_inference = False
            
        logger.info("Model loaded successfully")
    
    def compute_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute relevance scores for documents given a query
        
        Args:
            query: Query string
            documents: List of document strings
            
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        if self._use_direct_inference:
            # Direct inference using transformers
            scores = []
            with torch.no_grad():
                for pair in pairs:
                    # Tokenize
                    inputs = self.tokenizer(
                        pair[0], pair[1],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get logits
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply sigmoid to get score
                    score = torch.sigmoid(logits[0][0]).cpu().item()
                    scores.append(score)
            
            return scores
        else:
            # Use CrossEncoder predict method
            scores = self.model.predict(pairs)
            
            # Convert numpy array to list if needed
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            return scores
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[tuple]:
        """
        Rerank documents and return sorted results
        
        Args:
            query: Query string
            documents: List of document strings
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_index, score, document_text)
        """
        scores = self.compute_scores(query, documents)
        
        # Create tuples of (index, score, document)
        results = [(idx, score, doc) for idx, (score, doc) in enumerate(zip(scores, documents))]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None and top_k < len(results):
            results = results[:top_k]
        
        return results


class ModelManager:
    """Manages multiple reranker models"""
    
    def __init__(self, default_model: str = "maidalun1020/bce-reranker-base_v1", 
                 models_base_dir: str = "/app/models"):
        """
        Initialize model manager
        
        Args:
            default_model: Default model name to use
            models_base_dir: Base directory for model storage
        """
        self.default_model = default_model
        self.models_base_dir = models_base_dir
        self.loaded_models: Dict[str, RerankerModel] = {}
        self.supported_models = {
            "maidalun1020/bce-reranker-base_v1": {
                "name": "bce-reranker-base_v1",
                "description": "BGE Reranker Base Model v1",
                "max_length": 512
            },
            "BAAI/bge-reranker-base": {
                "name": "bge-reranker-base",
                "description": "BGE Reranker Base Model",
                "max_length": 512
            },
            "BAAI/bge-reranker-large": {
                "name": "bge-reranker-large", 
                "description": "BGE Reranker Large Model",
                "max_length": 512
            }
        }
        
        logger.info(f"Model manager initialized with default model: {default_model}")
    
    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Get model information"""
        if model_name in self.supported_models:
            return self.supported_models[model_name]
        else:
            # Return generic info for unknown models
            return {
                "name": model_name.split("/")[-1] if "/" in model_name else model_name,
                "description": f"Custom model: {model_name}",
                "max_length": 512
            }
    
    def is_model_supported(self, model_name: str) -> bool:
        """Check if model is in supported list"""
        return model_name in self.supported_models
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names"""
        return list(self.supported_models.keys())
    
    def load_model(self, model_name: str) -> RerankerModel:
        """
        Load a model, using cache if available
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            RerankerModel instance
        """
        # Use default model if None or empty
        if not model_name:
            model_name = self.default_model
        
        # Return cached model if available
        if model_name in self.loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        # Create model directory path
        model_dir = os.path.join(self.models_base_dir, 
                                model_name.replace("/", "_"))
        
        try:
            logger.info(f"Loading new model: {model_name}")
            model_info = self.get_model_info(model_name)
            
            # Create and cache the model
            reranker = RerankerModel(
                model_name=model_name,
                model_dir=model_dir,
                max_length=model_info.get("max_length", 512)
            )
            
            self.loaded_models[model_name] = reranker
            logger.info(f"Model {model_name} loaded and cached successfully")
            return reranker
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # If loading fails and it's not the default model, try default
            if model_name != self.default_model:
                logger.info(f"Falling back to default model: {self.default_model}")
                return self.load_model(self.default_model)
            else:
                raise
    
    def compute_scores(self, model_name: str, query: str, documents: List[str]) -> List[float]:
        """
        Compute scores using specified model
        
        Args:
            model_name: Name of the model to use
            query: Query string
            documents: List of documents
            
        Returns:
            List of relevance scores
        """
        model = self.load_model(model_name)
        return model.compute_scores(query, documents)
    
    def rerank(self, model_name: str, query: str, documents: List[str], 
              top_k: Optional[int] = None) -> List[tuple]:
        """
        Rerank documents using specified model
        
        Args:
            model_name: Name of the model to use
            query: Query string
            documents: List of documents
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_index, score, document_text)
        """
        model = self.load_model(model_name)
        return model.rerank(query, documents, top_k)
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                logger.info(f"Cleared cache for model: {model_name}")
        else:
            self.loaded_models.clear()
            logger.info("Cleared all model cache")


# Backward compatibility alias
BGERerankerLoader = RerankerModel