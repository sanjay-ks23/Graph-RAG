"""Graph construction and management"""

import networkx as nx
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GraphBuilder:
    """Build and manage knowledge graph"""
    
    def __init__(self, persist_path: str = "data/graph_db.pkl"):
        self.persist_path = Path(persist_path)
        self.graph = nx.MultiDiGraph()
        self.node_attributes = {}
        self.edge_attributes = {}
        
        # Load existing graph if available
        if self.persist_path.exists():
            self.load()
    
    def add_entities(self, entities: List[Dict[str, Any]], 
                    chunk_id: str, source: str):
        """Add entities as nodes to the graph"""
        for entity in entities:
            node_id = self._create_node_id(entity['text'], entity['type'])
            
            if node_id in self.graph:
                # Update existing node
                self.graph.nodes[node_id]['frequency'] += 1
                self.graph.nodes[node_id]['sources'].add(source)
                self.graph.nodes[node_id]['chunks'].add(chunk_id)
            else:
                # Add new node
                self.graph.add_node(
                    node_id,
                    text=entity['text'],
                    type=entity['type'],
                    confidence=entity['confidence'],
                    frequency=1,
                    sources={source},
                    chunks={chunk_id}
                )
    
    def add_relationships(self, relationships: List[Dict[str, Any]], 
                         chunk_id: str, source: str):
        """Add relationships as edges to the graph"""
        for rel in relationships:
            source_id = self._find_node_id(rel['source'])
            target_id = self._find_node_id(rel['target'])
            
            if source_id and target_id:
                # Add edge
                self.graph.add_edge(
                    source_id,
                    target_id,
                    type=rel['type'],
                    confidence=rel['confidence'],
                    source=source,
                    chunk_id=chunk_id
                )
    
    def get_node_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """Get neighbors of a node up to certain depth"""
        if node_id not in self.graph:
            return []
        
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Get successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.discard(node_id)  # Remove the original node
        return list(neighbors)
    
    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> nx.MultiDiGraph:
        """Get subgraph around specified nodes"""
        all_nodes = set(node_ids)
        
        for node_id in node_ids:
            neighbors = self.get_node_neighbors(node_id, depth)
            all_nodes.update(neighbors)
        
        return self.graph.subgraph(all_nodes).copy()
    
    def get_paths_between_nodes(self, source: str, target: str, 
                               max_length: int = 3) -> List[List[str]]:
        """Find paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            return paths[:10]  # Limit to top 10 paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def get_most_important_nodes(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Get most important nodes using PageRank"""
        if len(self.graph) == 0:
            return []
        
        try:
            pagerank = nx.pagerank(self.graph)
            sorted_nodes = sorted(
                pagerank.items(), key=lambda x: x[1], reverse=True
            )
            return sorted_nodes[:top_k]
        except:
            # Fallback to degree centrality
            degree_centrality = nx.degree_centrality(self.graph)
            sorted_nodes = sorted(
                degree_centrality.items(), key=lambda x: x[1], reverse=True
            )
            return sorted_nodes[:top_k]
    
    def get_nodes_by_type(self, entity_type: str) -> List[str]:
        """Get all nodes of a specific type"""
        return [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == entity_type
        ]
    
    def get_node_context(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive context for a node"""
        if node_id not in self.graph:
            return {}
        
        node_data = self.graph.nodes[node_id]
        
        # Get incoming and outgoing relationships
        incoming = []
        for pred in self.graph.predecessors(node_id):
            edges = self.graph.get_edge_data(pred, node_id)
            for edge_data in edges.values():
                incoming.append({
                    'source': pred,
                    'source_text': self.graph.nodes[pred]['text'],
                    'type': edge_data['type']
                })
        
        outgoing = []
        for succ in self.graph.successors(node_id):
            edges = self.graph.get_edge_data(node_id, succ)
            for edge_data in edges.values():
                outgoing.append({
                    'target': succ,
                    'target_text': self.graph.nodes[succ]['text'],
                    'type': edge_data['type']
                })
        
        return {
            'node_id': node_id,
            'text': node_data['text'],
            'type': node_data['type'],
            'frequency': node_data['frequency'],
            'incoming_relations': incoming,
            'outgoing_relations': outgoing,
            'sources': list(node_data['sources']),
            'chunks': list(node_data['chunks'])
        }
    
    def save(self):
        """Save graph to disk"""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert sets to lists for serialization
        graph_copy = self.graph.copy()
        for node in graph_copy.nodes():
            if 'sources' in graph_copy.nodes[node]:
                graph_copy.nodes[node]['sources'] = list(graph_copy.nodes[node]['sources'])
            if 'chunks' in graph_copy.nodes[node]:
                graph_copy.nodes[node]['chunks'] = list(graph_copy.nodes[node]['chunks'])
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(graph_copy, f)
        
        logger.info(f"Graph saved: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
    
    def load(self):
        """Load graph from disk"""
        with open(self.persist_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Convert lists back to sets
        for node in self.graph.nodes():
            if 'sources' in self.graph.nodes[node]:
                self.graph.nodes[node]['sources'] = set(self.graph.nodes[node]['sources'])
            if 'chunks' in self.graph.nodes[node]:
                self.graph.nodes[node]['chunks'] = set(self.graph.nodes[node]['chunks'])
        
        logger.info(f"Graph loaded: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if len(self.graph) == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'node_types': {},
                'edge_types': {}
            }
        
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        edge_types = {}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        return {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'node_types': node_types,
            'edge_types': edge_types,
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph)
        }
    
    def _create_node_id(self, text: str, entity_type: str) -> str:
        """Create unique node ID"""
        return f"{entity_type}:{text.lower().strip()}"
    
    def _find_node_id(self, text: str) -> str:
        """Find node ID by text"""
        text = text.lower().strip()
        for node, data in self.graph.nodes(data=True):
            if data['text'] == text:
                return node
        return None
