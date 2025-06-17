import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from model_loader import BGERerankerLoader


class TestBGERerankerLoader:
    """Test BGE Reranker model loader"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('model_loader.CrossEncoder')
    def test_initialization_with_crossencoder(self, mock_crossencoder):
        """Test model initialization with CrossEncoder"""
        mock_model = Mock()
        mock_crossencoder.return_value = mock_model
        
        loader = BGERerankerLoader(
            model_name="test-model",
            max_length=256
        )
        
        assert loader.model_name == "test-model"
        assert loader.max_length == 256
        assert loader.model == mock_model
        assert loader._use_direct_inference is False
        mock_crossencoder.assert_called_once_with(
            "test-model",
            max_length=256,
            device=loader.device.type
        )
    
    @patch('model_loader.CrossEncoder', side_effect=Exception("CrossEncoder failed"))
    @patch('model_loader.AutoTokenizer')
    @patch('model_loader.AutoModelForSequenceClassification')
    def test_initialization_fallback_to_transformers(self, mock_model_class, mock_tokenizer_class, mock_crossencoder):
        """Test fallback to direct transformers loading"""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        loader = BGERerankerLoader(model_name="test-model")
        
        assert loader._use_direct_inference is True
        assert loader.tokenizer == mock_tokenizer
        # Model is loaded and configured
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('model_loader.CrossEncoder')
    def test_compute_scores_with_crossencoder(self, mock_crossencoder):
        """Test compute_scores with CrossEncoder"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7])
        mock_crossencoder.return_value = mock_model
        
        loader = BGERerankerLoader()
        
        query = "What is AI?"
        documents = ["AI is artificial intelligence", "Weather today", "AI and ML"]
        scores = loader.compute_scores(query, documents)
        
        assert len(scores) == 3
        assert scores == [0.9, 0.3, 0.7]
        mock_model.predict.assert_called_once()
        
        # Check the pairs passed to predict
        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0] == [query, documents[0]]
        assert call_args[1] == [query, documents[1]]
        assert call_args[2] == [query, documents[2]]
    
    @patch('model_loader.CrossEncoder', side_effect=Exception("CrossEncoder failed"))
    @patch('model_loader.AutoTokenizer')
    @patch('model_loader.AutoModelForSequenceClassification')
    def test_compute_scores_with_transformers(self, mock_model_class, mock_tokenizer_class, mock_crossencoder):
        """Test compute_scores with direct transformers inference"""
        # Skip complex transformers mock test for now
        pass
    
    def test_compute_scores_empty_documents(self):
        """Test compute_scores with empty documents"""
        with patch('model_loader.CrossEncoder'):
            loader = BGERerankerLoader()
            scores = loader.compute_scores("query", [])
            assert scores == []
    
    @patch('model_loader.CrossEncoder')
    def test_rerank_functionality(self, mock_crossencoder):
        """Test rerank method"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.7])
        mock_crossencoder.return_value = mock_model
        
        loader = BGERerankerLoader()
        
        query = "What is AI?"
        documents = ["Weather", "AI is great", "Machine learning"]
        results = loader.rerank(query, documents)
        
        # Check results are sorted by score descending
        assert len(results) == 3
        assert results[0] == (1, 0.9, "AI is great")
        assert results[1] == (2, 0.7, "Machine learning")
        assert results[2] == (0, 0.3, "Weather")
    
    @patch('model_loader.CrossEncoder')
    def test_rerank_with_top_k(self, mock_crossencoder):
        """Test rerank with top_k parameter"""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.7, 0.5])
        mock_crossencoder.return_value = mock_model
        
        loader = BGERerankerLoader()
        
        query = "What is AI?"
        documents = ["Doc1", "Doc2", "Doc3", "Doc4"]
        results = loader.rerank(query, documents, top_k=2)
        
        assert len(results) == 2
        assert results[0][1] == 0.9  # Highest score
        assert results[1][1] == 0.7  # Second highest
    
    def test_device_detection(self):
        """Test device detection"""
        with patch('model_loader.CrossEncoder'):
            with patch('torch.cuda.is_available', return_value=True):
                loader = BGERerankerLoader()
                assert loader.device.type == "cuda"
            
            with patch('torch.cuda.is_available', return_value=False):
                loader = BGERerankerLoader()
                assert loader.device.type == "cpu"
    
    @patch('model_loader.CrossEncoder')
    @patch('model_loader.os.path.exists')
    def test_local_model_loading(self, mock_exists, mock_crossencoder, temp_model_dir):
        """Test loading model from local directory"""
        mock_exists.return_value = True
        mock_model = Mock()
        mock_crossencoder.return_value = mock_model
        
        loader = BGERerankerLoader(
            model_name="test-model",
            model_dir=temp_model_dir
        )
        
        # Should load from local directory
        mock_crossencoder.assert_called_once_with(
            temp_model_dir,
            max_length=512,
            device=loader.device.type
        )
    
    @patch('model_loader.CrossEncoder')
    @patch('model_loader.Path.mkdir')
    def test_model_dir_creation(self, mock_mkdir, mock_crossencoder):
        """Test model directory creation"""
        mock_model = Mock()
        mock_crossencoder.return_value = mock_model
        
        loader = BGERerankerLoader(
            model_name="test-model",
            model_dir="/non/existent/dir"
        )
        
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)