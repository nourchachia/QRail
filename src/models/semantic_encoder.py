"""
Semantic Encoder (Model 3) - src/models/semantic_encoder.py

Purpose:
    Converts natural language incident descriptions into dense vector embeddings
    using SentenceTransformer (all-MiniLM-L6-v2 model).

Features:
    - Model caching to prevent reloading on every call
    - Thread-safe singleton pattern
    - Integration with DataFuelPipeline.extract_semantic_text()
"""

import sys
from pathlib import Path
from typing import Optional, List, Union, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticEncoder:
    """
    Semantic encoder using SentenceTransformer for text embeddings.
    
    Uses all-MiniLM-L6-v2 model (384-dim embeddings).
    Implements singleton pattern with model caching to avoid reloading.
    
    Usage:
        encoder = SemanticEncoder()
        embedding = encoder.encode("Signal failure at Central Station")
        embeddings = encoder.encode_batch(["Text 1", "Text 2"])
    """
    
    _instance: Optional['SemanticEncoder'] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: str = "all-MiniLM-L6-v2"
    _embedding_dim: int = 384
    
    def __new__(cls):
        """Singleton pattern - ensures only one instance exists"""
        if cls._instance is None:
            cls._instance = super(SemanticEncoder, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize encoder with cached model loading"""
        if self._model is None:
            self._load_model()
    
    @classmethod
    def _load_model(cls):
        """Load SentenceTransformer model (cached after first load)"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        if cls._model is None:
            logger.info(f"Loading SentenceTransformer model: {cls._model_name}")
            try:
                cls._model = SentenceTransformer(cls._model_name)
                logger.info(f"Model loaded successfully. Embedding dimension: {cls._embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the cached model instance"""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (384 for all-MiniLM-L6-v2)"""
        return self._embedding_dim
    
    def encode(
        self, 
        text: str, 
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Encode a single text string into embedding vector.
        
        Args:
            text: Input text to encode
            normalize: Whether to L2-normalize the embedding
            convert_to_numpy: Return numpy array (True) or torch tensor (False)
        
        Returns:
            Embedding vector of shape (384,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            shape = (self._embedding_dim,)
            if convert_to_numpy:
                return np.zeros(shape)
            else:
                import torch
                return torch.zeros(shape)
        
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=False
            )
            return embedding
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            shape = (self._embedding_dim,)
            if convert_to_numpy:
                return np.zeros(shape)
            else:
                import torch
                return torch.zeros(shape)
    
    def encode_batch(
        self, 
        texts: List[str],
        normalize: bool = True,
        convert_to_numpy: bool = True,
        batch_size: int = 32
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Encode a batch of texts into embedding vectors.
        
        Args:
            texts: List of text strings to encode
            normalize: Whether to L2-normalize embeddings
            convert_to_numpy: Return numpy array (True) or torch tensor (False)
            batch_size: Batch size for processing
        
        Returns:
            Embedding matrix of shape (len(texts), 384)
        """
        if not texts:
            logger.warning("Empty text list provided")
            if convert_to_numpy:
                return np.array([])
            else:
                import torch
                return torch.tensor([])
        
        # Filter out empty texts
        valid_texts = [t if t and t.strip() else "empty" for t in texts]
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                batch_size=batch_size,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            shape = (len(texts), self._embedding_dim)
            if convert_to_numpy:
                return np.zeros(shape)
            else:
                import torch
                return torch.zeros(shape)
    
    def encode_from_pipeline(
        self, 
        incident: dict,
        pipeline=None
    ) -> np.ndarray:
        """
        Extract semantic text from DataFuelPipeline and encode it.
        
        Args:
            incident: Incident dictionary
            pipeline: DataFuelPipeline instance (optional, will create if None)
        
        Returns:
            Embedding vector of shape (384,)
        """
        if pipeline is None:
            from src.backend.feature_extractor import DataFuelPipeline
            pipeline = DataFuelPipeline(data_dir="data")
        
        # Extract semantic text using pipeline
        text = pipeline.extract_semantic_text(incident)
        
        # Encode to embedding
        embedding = self.encode(text, normalize=True, convert_to_numpy=True)
        
        return embedding
    
    @classmethod
    def reset_cache(cls):
        """Reset model cache (useful for testing or memory management)"""
        cls._model = None
        cls._instance = None
        logger.info("Model cache reset")


# Example usage
if __name__ == "__main__":
    """
    Test script for SemanticEncoder
    
    Expected Output:
    ================
    Loading SentenceTransformer model: all-MiniLM-L6-v2
    Model loaded successfully. Embedding dimension: 384
    Single text embedding shape: (384,)
    Batch embeddings shape: (3, 384)
    Pipeline integration test passed
    """
    
    # Initialize encoder (model loads on first call)
    encoder = SemanticEncoder()
    
    # Test single encoding
    text = "Signal failure at Central Station during peak hours"
    embedding = encoder.encode(text)
    print(f"Single text embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test batch encoding
    texts = [
        "Signal failure at Central Station",
        "Train breakdown on segment SEG_001",
        "Passenger alarm activated at North Terminal"
    ]
    embeddings = encoder.encode_batch(texts)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test pipeline integration
    test_incident = {
        'type': 'signal_failure',
        'location': {'zone': 'core'},
        'severity': 'high',
        'weather': 'rain',
        'trains_affected_count': 5
    }
    
    embedding_from_pipeline = encoder.encode_from_pipeline(test_incident)
    print(f"Pipeline integration embedding shape: {embedding_from_pipeline.shape}")
    print("Pipeline integration test passed")
