"""Rebuild vector store with normalized embeddings"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import pickle
import faiss
import numpy as np
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def rebuild_vector_store():
    """Rebuild vector store with proper normalization"""
    
    vector_path = Path("data/vector_store.index")
    metadata_path = Path("data/vector_metadata.pkl")
    
    if not vector_path.exists() or not metadata_path.exists():
        logger.error("Vector store files not found!")
        return
    
    logger.info("Loading existing vector store...")
    
    # Load old index
    old_index = faiss.read_index(str(vector_path))
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
        metadata = data['metadata']
        id_to_idx = data['id_to_idx']
    
    logger.info(f"Loaded {old_index.ntotal} vectors")
    
    # Extract all vectors - handle IVF index
    logger.info("Extracting vectors from IVF index...")
    
    # For IVF index, we need to extract from inverted lists
    if isinstance(old_index, faiss.IndexIVFFlat):
        logger.info("Detected IVF index, extracting from inverted lists...")
        
        # Get the inverted lists
        invlists = old_index.invlists
        dimension = old_index.d
        n_total = old_index.ntotal
        
        # Extract all vectors (order doesn't matter, metadata is separate)
        vectors = []
        
        for list_no in range(old_index.nlist):
            list_size = invlists.list_size(list_no)
            if list_size > 0:
                # Get vectors from this list
                # For IndexIVFFlat, codes are stored as raw float32 vectors
                codes_size = list_size * dimension * 4  # 4 bytes per float32
                list_codes = faiss.rev_swig_ptr(invlists.get_codes(list_no), codes_size)
                list_vecs = np.frombuffer(list_codes, dtype='float32').reshape(list_size, dimension)
                vectors.append(list_vecs)
        
        if vectors:
            vectors = np.vstack(vectors)
            logger.info(f"Extracted {len(vectors)} vectors from {old_index.nlist} inverted lists")
        else:
            logger.error("No vectors found in IVF index!")
            return
    else:
        # For flat index, can reconstruct directly
        logger.info("Extracting from flat index...")
        vectors = np.zeros((old_index.ntotal, old_index.d), dtype='float32')
        for i in range(old_index.ntotal):
            vectors[i] = old_index.reconstruct(int(i))
    
    logger.info(f"Extracted {len(vectors)} vectors")
    
    # Normalize vectors
    logger.info("Normalizing vectors...")
    faiss.normalize_L2(vectors)
    
    # Create new index (use Flat for simplicity and speed with normalized vectors)
    logger.info("Creating new normalized Flat index...")
    new_index = faiss.IndexFlatL2(vectors.shape[1])
    new_index.add(vectors)
    
    # Save
    logger.info("Saving normalized vector store...")
    faiss.write_index(new_index, str(vector_path))
    
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'metadata': metadata,
            'id_to_idx': id_to_idx,
            'is_trained': True
        }, f)
    
    logger.info(f"✓ Vector store rebuilt with {new_index.ntotal} normalized vectors")
    logger.info("✓ Switched to Flat index for better accuracy with normalized vectors")
    logger.info("You can now run the app with proper similarity search!")

if __name__ == "__main__":
    rebuild_vector_store()
