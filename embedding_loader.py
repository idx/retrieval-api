import os
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Embedding model loader and inference class"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 8192,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name or path of the embedding model
            model_dir: Local directory to save/load model
            device: Device to use (cuda/cpu/auto)
            max_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.tokenizer = None
        self._use_sentence_transformers = True
        
        # Detect device
        if device == "auto" or device is None:
            self.device = self._detect_device()
        else:
            self.device = torch.device(device)
        
        # Supported models configuration
        self.supported_models = {
            "BAAI/bge-m3": {
                "name": "bge-m3",
                "description": "BGE M3 Multilingual Embedding (8192 tokens, 1024 dim) - Default",
                "max_length": 8192,
                "dimensions": 1024,
                "language": "multilingual"
            },
            "cl-nagoya/ruri-large": {
                "name": "ruri-large",
                "description": "Ruri Large Japanese Embedding (512 tokens, 1024 dim, JMTEB最高性能)",
                "max_length": 512,
                "dimensions": 1024,
                "language": "japanese"
            },
            "cl-nagoya/ruri-base": {
                "name": "ruri-base",
                "description": "Ruri Base Japanese Embedding (512 tokens, 768 dim, 日本語バランス型)",
                "max_length": 512,
                "dimensions": 768,
                "language": "japanese"
            },
            "MU-Kindai/Japanese-SimCSE-BERT-large-unsup": {
                "name": "japanese-simcse-large",
                "description": "Japanese SimCSE BERT Large (512 tokens, 1024 dim, 教師なし学習)",
                "max_length": 512,
                "dimensions": 1024,
                "language": "japanese"
            },
            "sonoisa/sentence-luke-japanese-base-lite": {
                "name": "luke-japanese-base",
                "description": "LUKE Japanese Base Lite (512 tokens, 768 dim, 知識強化型)",
                "max_length": 512,
                "dimensions": 768,
                "language": "japanese"
            },
            "pkshatech/GLuCoSE-base-ja-v2": {
                "name": "glucose-ja-v2",
                "description": "GLuCoSE Japanese v2 (512 tokens, 768 dim, 企業開発)",
                "max_length": 512,
                "dimensions": 768,
                "language": "japanese"
            },
            "nvidia/NV-Embed-v2": {
                "name": "nv-embed-v2",
                "description": "NVIDIA NV-Embed v2 (32768 tokens, 4096 dim, SOTA performance)",
                "max_length": 32768,
                "dimensions": 4096,
                "language": "multilingual"
            },
            "intfloat/e5-mistral-7b-instruct": {
                "name": "e5-mistral-7b",
                "description": "E5 Mistral 7B Instruct (32768 tokens, 4096 dim, high quality)",
                "max_length": 32768,
                "dimensions": 4096,
                "language": "multilingual"
            },
            "mixedbread-ai/mxbai-embed-large-v1": {
                "name": "mxbai-embed-large",
                "description": "MixedBread AI Large v1 (512 tokens, 1024 dim, production ready)",
                "max_length": 512,
                "dimensions": 1024,
                "language": "multilingual"
            },
            "sentence-transformers/all-mpnet-base-v2": {
                "name": "all-mpnet-base-v2",
                "description": "All MPNet Base v2 (514 tokens, 768 dim, balanced performance)",
                "max_length": 514,
                "dimensions": 768,
                "language": "english"
            },
            "sentence-transformers/all-MiniLM-L6-v2": {
                "name": "all-minilm-l6-v2",
                "description": "All MiniLM L6 v2 (256 tokens, 384 dim, fast and efficient)",
                "max_length": 256,
                "dimensions": 384,
                "language": "english"
            },
            "intfloat/multilingual-e5-large": {
                "name": "multilingual-e5-large", 
                "description": "Multilingual E5 Large (512 tokens, 1024 dim, 100+ languages)",
                "max_length": 512,
                "dimensions": 1024,
                "language": "multilingual"
            },
            "intfloat/multilingual-e5-base": {
                "name": "multilingual-e5-base",
                "description": "Multilingual E5 Base (512 tokens, 768 dim, legacy default)",
                "max_length": 512,
                "dimensions": 768,
                "language": "multilingual"
            },
            "intfloat/e5-base": {
                "name": "e5-base",
                "description": "E5 Base Model (512 tokens, 768 dim, English optimized)",
                "max_length": 512,
                "dimensions": 768,
                "language": "english"
            },
            "intfloat/e5-large": {
                "name": "e5-large",
                "description": "E5 Large Model (512 tokens, 1024 dim, English optimized)",
                "max_length": 512,
                "dimensions": 1024,
                "language": "english"
            }
        }
        
        # Update max_length based on model if not explicitly set
        if model_name in self.supported_models:
            if max_length == 8192:  # Default value, update based on model
                self.max_length = self.supported_models[model_name]["max_length"]
        
        # Load the model
        self._load_model()
        
    def _detect_device(self):
        """Detect the best available device (NVIDIA CUDA, AMD ROCm, or CPU)"""
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            logger.info("NVIDIA CUDA detected for embeddings")
            return torch.device("cuda")
        
        # Check for AMD ROCm
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                logger.info("AMD ROCm detected for embeddings")
                return torch.device("cuda")  # ROCm uses cuda API
        except:
            pass
        
        # Check if AMD GPU is available via environment variables
        if os.getenv('HIP_VISIBLE_DEVICES') is not None or os.getenv('ROCR_VISIBLE_DEVICES') is not None:
            logger.info("AMD GPU environment detected for embeddings")
            return torch.device("cuda")
        
        # Default to CPU
        logger.info("No GPU detected for embeddings, using CPU")
        return torch.device("cpu")
        
    def _load_model(self):
        """Load the embedding model"""
        logger.info(f"Initializing embedding model on device: {self.device}")
        
        # Try to load from local directory first if specified
        if self.model_dir and os.path.exists(self.model_dir):
            logger.info(f"Loading embedding model from local directory: {self.model_dir}")
            model_path = self.model_dir
        else:
            logger.info(f"Loading embedding model from Hugging Face Hub: {self.model_name}")
            model_path = self.model_name
            
            # Create model directory if specified
            if self.model_dir:
                Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Try loading with sentence-transformers first
            self.model = SentenceTransformer(
                model_path,
                device=self.device.type,
                trust_remote_code=True
            )
            self.model.max_seq_length = self.max_length
            
            # Save model locally if directory specified and not already saved
            if self.model_dir and not os.path.exists(os.path.join(self.model_dir, "config.json")):
                logger.info(f"Saving embedding model to local directory: {self.model_dir}")
                self.model.save(self.model_dir)
                
        except Exception as e:
            logger.error(f"Failed to load with SentenceTransformer, trying direct loading: {str(e)}")
            
            # Fallback to direct transformers loading
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Save locally if needed
                if self.model_dir and not os.path.exists(os.path.join(self.model_dir, "config.json")):
                    logger.info(f"Saving embedding model to local directory: {self.model_dir}")
                    self.tokenizer.save_pretrained(self.model_dir)
                    self.model.save_pretrained(self.model_dir)
                    
                # Set flag for direct inference
                self._use_sentence_transformers = False
            except Exception as e2:
                logger.error(f"Failed to load embedding model: {str(e2)}")
                raise RuntimeError(f"Could not load embedding model {self.model_name}: {str(e2)}")
                
        logger.info(f"Embedding model loaded successfully, using {'sentence-transformers' if self._use_sentence_transformers else 'direct transformers'}")
        
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: Optional[bool] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences into embeddings
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to normalize embeddings (overrides instance setting)
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        if isinstance(sentences, str):
            sentences = [sentences]
            
        if normalize_embeddings is None:
            normalize_embeddings = self.normalize_embeddings
            
        if self._use_sentence_transformers:
            # Use sentence-transformers encode method
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
                device=self.device.type
            )
        else:
            # Direct transformers inference
            embeddings = self._encode_direct(
                sentences,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings
            )
            
            if convert_to_numpy and isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
                
        return embeddings
        
    def _encode_direct(
        self,
        sentences: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ) -> torch.Tensor:
        """Direct encoding using transformers models"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                all_embeddings.append(embeddings)
                
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        return all_embeddings
        
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self._use_sentence_transformers:
            return self.model.get_sentence_embedding_dimension()
        else:
            # Get a sample embedding to determine dimension
            sample = self.encode("sample", convert_to_numpy=False)
            return sample.shape[-1]
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        model_info = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "device": str(self.device),
            "use_sentence_transformers": self._use_sentence_transformers
        }
        
        # Add model-specific info if available
        if self.model_name in self.supported_models:
            model_config = self.supported_models[self.model_name]
            model_info.update({
                "short_name": model_config["name"],
                "description": model_config["description"],
                "dimensions": model_config["dimensions"],
                "language": model_config.get("language", "unknown")
            })
            
        return model_info
        
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of supported models"""
        return self.supported_models.copy()
        
    def get_models_by_language(self, language: str = None) -> Dict[str, Dict[str, Any]]:
        """Get models filtered by language"""
        if language is None:
            return self.get_supported_models()
            
        filtered_models = {}
        for model_name, config in self.supported_models.items():
            if config.get("language", "").lower() == language.lower():
                filtered_models[model_name] = config
                
        return filtered_models
    
    def to(self, device: Union[str, torch.device]):
        """Move model to specified device"""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self._use_sentence_transformers:
            self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)
        return self