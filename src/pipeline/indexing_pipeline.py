"""Pipeline for indexing documents into graph and vector store"""

from typing import List, Dict, Any
from tqdm import tqdm
from src.ingestion.document_loader import DocumentLoader, TextChunker
from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_builder import GraphBuilder
from src.embedding.embedding_model import EmbeddingModel
from src.vector_store.faiss_store import FAISSVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class IndexingPipeline:
    """Pipeline to process documents and build graph + vector store"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        logger.info("Initializing indexing pipeline...")
        
        self.document_loader = DocumentLoader(
            config['data']['books_directory']
        )
        
        self.text_chunker = TextChunker(
            chunk_size=config['data']['chunk_size'],
            chunk_overlap=config['data']['chunk_overlap']
        )
        
        self.entity_extractor = EntityExtractor(
            entity_types=config['graph']['entity_types'],
            relationship_types=config['graph']['relationship_types']
        )
        
        self.graph_builder = GraphBuilder(
            persist_path=config['graph_db']['persist_path']
        )
        
        self.embedding_model = EmbeddingModel(
            model_id=config['models']['embedding']['model_id'],
            device=config['models']['embedding']['device'],
            batch_size=config['models']['embedding']['batch_size']
        )
        
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_model.get_dimension(),
            index_type=config['vector_store']['index_type'],
            nlist=config['vector_store']['nlist'],
            persist_path=config['vector_store']['persist_path'],
            metadata_path=config['vector_store']['metadata_path']
        )
        
        logger.info("Indexing pipeline initialized")
    
    def run(self, force_reindex: bool = False):
        """Run the complete indexing pipeline"""
        
        logger.info("=" * 60)
        logger.info("Starting Graph RAG Indexing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load documents
        logger.info("\n[1/5] Loading documents...")
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            logger.warning("No documents found! Please add therapy books to the data directory.")
            return
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 2: Chunk documents
        logger.info("\n[2/5] Chunking documents...")
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            chunks = self.text_chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Step 3: Extract entities and build graph
        logger.info("\n[3/5] Building knowledge graph...")
        for chunk in tqdm(all_chunks, desc="Extracting entities"):
            # Extract entities
            entities = self.entity_extractor.extract_entities(
                chunk['text'],
                min_confidence=self.config['graph']['min_entity_confidence']
            )
            
            # Extract relationships
            relationships = self.entity_extractor.extract_relationships(
                chunk['text'], entities
            )
            
            # Add to graph
            chunk_id = f"{chunk['source']}_{chunk['chunk_id']}"
            self.graph_builder.add_entities(entities, chunk_id, chunk['source'])
            self.graph_builder.add_relationships(relationships, chunk_id, chunk['source'])
        
        # Save graph
        self.graph_builder.save()
        graph_stats = self.graph_builder.get_statistics()
        logger.info(f"Graph built: {graph_stats}")
        
        # Step 4: Generate embeddings
        logger.info("\n[4/5] Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode_batch(chunk_texts, show_progress=True)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 5: Index embeddings
        logger.info("\n[5/5] Indexing embeddings...")
        self.vector_store.add_embeddings(embeddings, all_chunks)
        self.vector_store.save()
        
        vector_stats = self.vector_store.get_statistics()
        logger.info(f"Vector store: {vector_stats}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Indexing Pipeline Completed Successfully!")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Chunks created: {len(all_chunks)}")
        logger.info(f"Graph nodes: {graph_stats['num_nodes']}")
        logger.info(f"Graph edges: {graph_stats['num_edges']}")
        logger.info(f"Vectors indexed: {vector_stats['total_vectors']}")
        logger.info("=" * 60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'graph': self.graph_builder.get_statistics(),
            'vector_store': self.vector_store.get_statistics()
        }
