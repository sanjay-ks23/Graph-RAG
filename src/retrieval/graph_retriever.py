"""Graph-based retrieval and reasoning module"""

import numpy as np
from typing import List, Dict, Any, Set
from src.graph.graph_builder import GraphBuilder
from src.vector_store.faiss_store import FAISSVectorStore
from src.embedding.embedding_model import EmbeddingModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GraphRetriever:
    """Retrieve and reason over graph and vector store"""
    
    def __init__(self, graph_builder: GraphBuilder, 
                 vector_store: FAISSVectorStore,
                 embedding_model: EmbeddingModel,
                 top_k_chunks: int = 5,
                 top_k_graph_nodes: int = 10,
                 similarity_threshold: float = 0.7,
                 graph_traversal_depth: int = 2):
        self.graph_builder = graph_builder
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k_chunks = top_k_chunks
        self.top_k_graph_nodes = top_k_graph_nodes
        self.similarity_threshold = similarity_threshold
        self.graph_traversal_depth = graph_traversal_depth
    
    def retrieve(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve relevant context using hybrid approach"""
        
        # 1. Vector-based retrieval
        vector_results = self._vector_retrieval(query)
        
        # 2. Graph-based retrieval
        graph_results = self._graph_retrieval(query, vector_results)
        
        # 3. Combine and rank results
        combined_context = self._combine_results(vector_results, graph_results)
        
        # 4. Add user context if available
        if user_context:
            combined_context['user_context'] = user_context
        
        return combined_context
    
    def _vector_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve similar chunks using vector search"""
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=self.top_k_chunks,
            threshold=self.similarity_threshold
        )
        
        logger.info(f"Vector retrieval: {len(results)} chunks found")
        return results
    
    def _graph_retrieval(self, query: str, 
                        vector_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrieve graph context based on query and vector results"""
        
        # Extract entities from query
        from src.graph.entity_extractor import EntityExtractor
        entity_extractor = EntityExtractor(
            entity_types=['therapeutic_technique', 'emotion', 'behavior', 
                         'cognitive_pattern', 'coping_strategy'],
            relationship_types=['treats', 'causes', 'relates_to']
        )
        
        query_entities = entity_extractor.extract_entities(query)
        
        # Find relevant graph nodes
        relevant_nodes = set()
        
        # Add nodes from query entities
        for entity in query_entities:
            node_id = f"{entity['type']}:{entity['text'].lower()}"
            if node_id in self.graph_builder.graph:
                relevant_nodes.add(node_id)
        
        # Add nodes from vector results (chunks)
        for result in vector_results:
            chunk_metadata = result['metadata']
            chunk_id = f"{chunk_metadata['source']}_{chunk_metadata['chunk_id']}"
            
            # Find nodes associated with this chunk
            for node, data in self.graph_builder.graph.nodes(data=True):
                if chunk_id in data.get('chunks', set()):
                    relevant_nodes.add(node)
        
        # Expand with neighbors
        expanded_nodes = set(relevant_nodes)
        for node in relevant_nodes:
            neighbors = self.graph_builder.get_node_neighbors(
                node, depth=self.graph_traversal_depth
            )
            expanded_nodes.update(neighbors[:self.top_k_graph_nodes])
        
        # Get node contexts
        node_contexts = []
        for node_id in list(expanded_nodes)[:self.top_k_graph_nodes]:
            context = self.graph_builder.get_node_context(node_id)
            if context:
                node_contexts.append(context)
        
        # Get subgraph
        subgraph = self.graph_builder.get_subgraph(
            list(expanded_nodes)[:self.top_k_graph_nodes],
            depth=1
        )
        
        logger.info(f"Graph retrieval: {len(node_contexts)} nodes, {len(subgraph.edges())} edges")
        
        return {
            'nodes': node_contexts,
            'subgraph': subgraph,
            'query_entities': query_entities
        }
    
    def _combine_results(self, vector_results: List[Dict[str, Any]], 
                        graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine vector and graph results into unified context"""
        
        # Extract text chunks
        chunks = []
        for result in vector_results:
            chunks.append({
                'text': result['metadata']['text'],
                'source': result['metadata']['source'],
                'similarity': result['similarity']
            })
        
        # Extract graph knowledge
        graph_knowledge = []
        for node_context in graph_results['nodes']:
            # Create knowledge statement
            knowledge = f"{node_context['text']} ({node_context['type']})"
            
            # Add relationships
            relations = []
            for rel in node_context['outgoing_relations'][:3]:
                relations.append(f"{rel['type']} {rel['target_text']}")
            
            if relations:
                knowledge += ": " + ", ".join(relations)
            
            graph_knowledge.append(knowledge)
        
        # Combine into context
        combined_context = {
            'chunks': chunks,
            'graph_knowledge': graph_knowledge,
            'entities': graph_results['query_entities'],
            'num_graph_nodes': len(graph_results['nodes']),
            'num_graph_edges': len(graph_results['subgraph'].edges())
        }
        
        return combined_context
    
    def get_therapeutic_context(self, age: int = None, 
                               emotion: str = None) -> Dict[str, Any]:
        """Get therapeutic context based on user attributes"""
        context = {
            'age_appropriate_techniques': [],
            'emotion_specific_strategies': [],
            'cultural_considerations': []
        }
        
        # Get age-appropriate techniques
        if age is not None:
            dev_stage = self._get_developmental_stage(age)
            age_nodes = self.graph_builder.get_nodes_by_type('developmental_stage')
            
            for node_id in age_nodes:
                node_data = self.graph_builder.graph.nodes[node_id]
                if dev_stage.lower() in node_data['text'].lower():
                    node_context = self.graph_builder.get_node_context(node_id)
                    context['age_appropriate_techniques'].append(node_context)
        
        # Get emotion-specific strategies
        if emotion:
            emotion_nodes = self.graph_builder.get_nodes_by_type('emotion')
            for node_id in emotion_nodes:
                node_data = self.graph_builder.graph.nodes[node_id]
                if emotion.lower() in node_data['text'].lower():
                    node_context = self.graph_builder.get_node_context(node_id)
                    context['emotion_specific_strategies'].append(node_context)
        
        # Get cultural considerations
        cultural_nodes = self.graph_builder.get_nodes_by_type('cultural_context')
        for node_id in cultural_nodes[:5]:
            node_context = self.graph_builder.get_node_context(node_id)
            context['cultural_considerations'].append(node_context)
        
        return context
    
    def _get_developmental_stage(self, age: int) -> str:
        """Map age to developmental stage"""
        if age < 3:
            return "toddler"
        elif age < 6:
            return "early childhood"
        elif age < 12:
            return "middle childhood"
        elif age < 18:
            return "adolescent"
        else:
            return "young adult"
