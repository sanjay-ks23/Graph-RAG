"""Flask web application for Graph RAG Psychotherapy Chatbot"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import uuid
import os
from pathlib import Path

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.llm.gemma_model import GemmaModel
from src.embedding.embedding_model import EmbeddingModel
from src.graph.graph_builder import GraphBuilder
from src.vector_store.faiss_store import FAISSVectorStore
from src.retrieval.graph_retriever import GraphRetriever
from src.conversation.memory_manager import ConversationMemory, SessionManager
from src.chat.chat_service import ChatService
from src.feedback.learning_system import FeedbackSystem, InteractionLogger

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)

# Setup logger
logger = setup_logger(__name__)

# Global variables for components
config_loader = None
chat_service = None
session_manager = None
feedback_system = None
interaction_logger = None

def initialize_system():
    """Initialize all system components"""
    global config_loader, chat_service, session_manager, feedback_system, interaction_logger
    
    logger.info("Initializing Graph RAG Psychotherapy System...")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.config
    
    # Initialize embedding model
    logger.info("Loading embedding model...")
    embedding_model = EmbeddingModel(
        model_id=config['models']['embedding']['model_id'],
        device=config['models']['embedding']['device'],
        batch_size=config['models']['embedding']['batch_size']
    )
    
    # Initialize graph builder
    logger.info("Loading knowledge graph...")
    graph_builder = GraphBuilder(
        persist_path=config['graph_db']['persist_path']
    )
    
    # Initialize vector store
    logger.info("Loading vector store...")
    vector_store = FAISSVectorStore(
        dimension=embedding_model.get_dimension(),
        index_type=config['vector_store']['index_type'],
        nlist=config['vector_store']['nlist'],
        persist_path=config['vector_store']['persist_path'],
        metadata_path=config['vector_store']['metadata_path']
    )
    
    # Initialize retriever
    logger.info("Initializing retriever...")
    graph_retriever = GraphRetriever(
        graph_builder=graph_builder,
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k_chunks=config['retrieval']['top_k_chunks'],
        top_k_graph_nodes=config['retrieval']['top_k_graph_nodes'],
        similarity_threshold=config['retrieval']['similarity_threshold'],
        graph_traversal_depth=config['retrieval']['graph_traversal_depth']
    )
    
    # Initialize LLM
    logger.info("Loading Gemma model...")
    gemma_model = GemmaModel(
        model_id=config['models']['llm']['model_id'],
        device=config['models']['llm']['device'],
        max_length=config['models']['llm']['max_length'],
        temperature=config['models']['llm']['temperature'],
        top_p=config['models']['llm']['top_p'],
        top_k=config['models']['llm']['top_k']
    )
    
    # Initialize session manager
    session_manager = SessionManager(
        session_timeout_minutes=config['conversation']['session_timeout_minutes']
    )
    
    # Initialize feedback system
    feedback_system = FeedbackSystem(
        feedback_storage=config['learning']['feedback_storage'],
        quality_threshold=config['learning']['quality_threshold'],
        update_frequency=config['learning']['update_frequency']
    )
    
    # Initialize interaction logger
    interaction_logger = InteractionLogger()
    
    # Create a template chat service (actual instances per session)
    logger.info("System initialization complete!")
    
    return {
        'gemma_model': gemma_model,
        'graph_retriever': graph_retriever,
        'config': config
    }

# Initialize on startup
system_components = initialize_system()

@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get or create session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Get conversation memory for this session
        conversation_memory = session_manager.get_or_create_session(
            session_id,
            max_history_turns=config_loader.config['conversation']['max_history_turns'],
            context_window_tokens=config_loader.config['conversation']['context_window_tokens']
        )
        
        # Create chat service for this session
        chat_svc = ChatService(
            gemma_model=system_components['gemma_model'],
            graph_retriever=system_components['graph_retriever'],
            conversation_memory=conversation_memory,
            config=system_components['config']
        )
        
        # Generate response
        result = chat_svc.generate_response(user_message, session_id)
        
        # Log interaction
        interaction_logger.log_interaction(
            session_id=session_id,
            user_message=user_message,
            assistant_response=result['response'],
            context_used=result['context_used'],
            metadata={'crisis_detected': result['crisis_detected']}
        )
        
        return jsonify({
            'response': result['response'],
            'session_id': session_id,
            'crisis_detected': result['crisis_detected'],
            'context_used': result['context_used']
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a response"""
    try:
        data = request.json
        session_id = session.get('session_id', 'unknown')
        
        feedback_system.record_feedback(
            session_id=session_id,
            message_id=data.get('message_id', str(uuid.uuid4())),
            user_message=data.get('user_message', ''),
            assistant_response=data.get('assistant_response', ''),
            rating=data.get('rating'),
            feedback_text=data.get('feedback_text'),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation for current session"""
    try:
        session_id = session.get('session_id')
        if session_id:
            session_manager.delete_session(session_id)
            session.pop('session_id', None)
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        logger.error(f"Error in reset endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    try:
        stats = {
            'feedback': feedback_system.get_feedback_statistics(),
            'interactions': interaction_logger.get_statistics(),
            'graph': system_components['graph_retriever'].graph_builder.get_statistics(),
            'vector_store': system_components['graph_retriever'].vector_store.get_statistics()
        }
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in statistics endpoint: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
