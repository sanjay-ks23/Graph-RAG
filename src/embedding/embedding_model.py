"""Embedding generation using EmbeddingGemma"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingModel:
    """Generate embeddings using EmbeddingGemma"""
    
    def __init__(self, model_id: str = "google/embeddinggemma-300M", 
                 device: str = "cuda", batch_size: int = 32):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        
        logger.info(f"Loading embedding model: {model_id} on {self.device}")
        self.model = SentenceTransformer(model_id).to(device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: Union[str, List[str]], 
               show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        return embeddings
    
    def encode_batch(self, texts: List[str], 
                    show_progress: bool = True) -> np.ndarray:
        """Encode batch of texts"""
        return self.encode(texts, show_progress=show_progress)
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity (already normalized)
        similarity = np.dot(emb1[0], emb2[0])
        return float(similarity)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
