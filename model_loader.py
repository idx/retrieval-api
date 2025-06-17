import os
import logging
from typing import List, Union, Optional
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class BGERerankerLoader:
    """BGE Reranker model loader and scorer"""
    
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