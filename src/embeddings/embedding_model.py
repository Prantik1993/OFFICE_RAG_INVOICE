"""
HuggingFace embedding model wrapper.
Uses sentence-transformers for local embeddings.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class EmbeddingModel:
    """Wrapper for HuggingFace sentence-transformers embeddings."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model name (defaults to config)
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        
        try:
            logger.debug(f"Embedding {len(texts)} documents")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
        
        Returns:
            Embedding vector
        """
        try:
            logger.debug(f"Embedding query: {text[:50]}...")
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
# Singleton instance
_embedding_model = None

def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model