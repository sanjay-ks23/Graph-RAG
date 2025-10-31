"""FAISS-based vector store for embeddings"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FAISSVectorStore:
    """FAISS vector store for efficient similarity search"""
    
    def __init__(self, dimension: int, index_type: str = "IVF", 
                 nlist: int = 100, persist_path: str = "data/vector_store.index",
                 metadata_path: str = "data/vector_metadata.pkl"):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.persist_path = Path(persist_path)
        self.metadata_path = Path(metadata_path)
        
        self.index = None
        self.metadata = []  # Store chunk metadata
        self.id_to_idx = {}  # Map chunk IDs to index positions
        
        # Load existing index if available
        if self.persist_path.exists() and self.metadata_path.exists():
            self.load()
        else:
            self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_type == "IVF":
            # IVF (Inverted File Index) for faster search
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.is_trained = False
        else:
            # Flat index (exact search)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.is_trained = True
        
        logger.info(f"Initialized {self.index_type} index with dimension {self.dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, 
                      metadata: List[Dict[str, Any]]):
        """Add embeddings with metadata"""
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
        
        # Convert to float32
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity using L2 distance
        faiss.normalize_L2(embeddings)
        
        # Train index if needed
        if hasattr(self, 'is_trained') and not self.is_trained:
            if len(embeddings) >= self.nlist:
                logger.info("Training IVF index...")
                self.index.train(embeddings)
                self.is_trained = True
            else:
                logger.warning(f"Not enough samples to train IVF index (need {self.nlist}, got {len(embeddings)})")
                # Switch to flat index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.is_trained = True
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata
        for i, meta in enumerate(metadata):
            idx = start_idx + i
            chunk_id = meta.get('chunk_id', f'chunk_{idx}')
            self.id_to_idx[chunk_id] = idx
            self.metadata.append(meta)
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty - no embeddings to search")
            return []
        
        # Convert to float32 and reshape
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Normalize query embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result found
                continue
            
            # For normalized vectors, L2 distance relates to cosine similarity:
            # cosine_sim = 1 - (L2_dist^2 / 2)
            # Since embeddings are normalized, we can use this conversion
            similarity = max(0.0, 1.0 - (dist ** 2) / 2.0)
            
            if threshold is None or similarity >= threshold:
                result = {
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'distance': float(dist),
                    'index': int(idx)
                }
                results.append(result)
        
        return results
    
    def get_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Get metadata by chunk ID"""
        if chunk_id in self.id_to_idx:
            idx = self.id_to_idx[chunk_id]
            return self.metadata[idx]
        return None
    
    def save(self):
        """Save index and metadata to disk"""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.persist_path))
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'is_trained': getattr(self, 'is_trained', True)
            }, f)
        
        logger.info(f"Vector store saved: {self.index.ntotal} vectors")
    
    def load(self):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(str(self.persist_path))
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.id_to_idx = data['id_to_idx']
            self.is_trained = data.get('is_trained', True)
        
        logger.info(f"Vector store loaded: {self.index.ntotal} vectors")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': getattr(self, 'is_trained', True)
        }
