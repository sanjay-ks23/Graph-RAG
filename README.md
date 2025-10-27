# Graph RAG Psychotherapy Chatbot 🌟

A culturally-aware therapeutic chatbot for Indian children and adolescents (ages 0-18) using Graph RAG architecture with Gemma 3n E2B-it model.

## 🎯 Overview

This system combines **Graph-based Retrieval Augmented Generation (Graph RAG)** with a fine-tuned language model to provide empathetic, contextual, and culturally-sensitive therapeutic support for children. It goes beyond simple retrieval by building a knowledge graph from therapy books to understand relationships between therapeutic concepts.

## ✨ Key Features

### Core Capabilities
- **Graph RAG Architecture**: Knowledge graphs built from therapy books to understand relationships, not just retrieve chunks
- **Deep Contextual Understanding**: Comprehends therapeutic concepts through entity extraction and relationship mapping
- **Conversation Memory Management**: Maintains full chat history within Gemma 3n's context window
- **Self-Learning System**: Adapts and improves over time through feedback mechanisms
- **Cultural Adaptation**: Indian demographic-specific considerations (family values, academic pressure, festivals)
- **Empathetic Interaction**: Therapeutic communication style designed for children

### Technical Features
- **Hybrid Retrieval**: Combines vector similarity search with graph-based reasoning
- **Entity Extraction**: Identifies therapeutic techniques, emotions, behaviors, cognitive patterns, coping strategies
- **Relationship Mapping**: Understands how concepts relate (treats, causes, appropriate_for_age, etc.)
- **Crisis Detection**: Identifies crisis keywords and provides appropriate support resources
- **Session Management**: Maintains separate conversations with automatic cleanup
- **Feedback Loop**: Collects user feedback to improve responses over time

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Flask Web Interface                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Chat Service Layer                        │
│  • Conversation Memory  • Context Assembly  • Response Gen   │
└─────┬──────────────────────────────────────────────┬────────┘
      │                                               │
┌─────▼─────────────────┐                 ┌──────────▼─────────┐
│  Graph RAG Retrieval  │                 │   Gemma 3n E2B-it  │
│  • Vector Search      │                 │   LLM Generation   │
│  • Graph Traversal    │                 │                    │
│  • Context Reasoning  │                 │                    │
└─────┬─────────────────┘                 └────────────────────┘
      │
┌─────▼──────────────┐  ┌──────────────┐  ┌─────────────────┐
│  Knowledge Graph   │  │ Vector Store │  │ Embedding Model │
│  (NetworkX)        │  │  (FAISS)     │  │ (EmbeddingGemma)│
└────────────────────┘  └──────────────┘  └─────────────────┘
```

### Module Structure

1. **Data Ingestion & Preprocessing** (`src/ingestion/`)
   - Document loading (PDF, TXT, EPUB, DOCX)
   - Text chunking with overlap
   - Metadata extraction

2. **Graph Construction** (`src/graph/`)
   - Entity extraction using NLP and patterns
   - Relationship identification
   - Knowledge graph building with NetworkX

3. **Embedding & Indexing** (`src/embedding/`, `src/vector_store/`)
   - EmbeddingGemma for semantic embeddings
   - FAISS vector store for similarity search
   - Efficient indexing and retrieval

4. **Graph-based Retrieval** (`src/retrieval/`)
   - Hybrid vector + graph retrieval
   - Graph traversal for context expansion
   - Therapeutic context assembly

5. **LLM Integration** (`src/llm/`)
   - Gemma 3n E2B-it model
   - Context-aware prompt construction
   - Response generation

6. **Conversation Management** (`src/conversation/`)
   - Multi-turn conversation memory
   - User profile extraction
   - Session management

7. **Feedback & Learning** (`src/feedback/`)
   - Feedback collection
   - Interaction logging
   - Pattern analysis for improvement

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM recommended
- HuggingFace account (for model access)

### Quick Start

1. **Clone the repository**
```bash
cd /home/sanj-ai/Documents/SlateMate/Graph-RAG
```

2. **Set up environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

3. **Configure environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Add your HuggingFace token if needed
```

4. **Add therapy books**
```bash
# Place your therapy books in the data directory
mkdir -p data/therapy_books
# Copy PDF, TXT, EPUB, or DOCX files to data/therapy_books/
```

5. **Index documents**
```bash
# Build knowledge graph and vector store
python index_documents.py
```

6. **Run the application**
```bash
# Start Flask server
python app.py

# Or use the convenience script
chmod +x run.sh
./run.sh
```

7. **Access the chatbot**
```
Open your browser and navigate to: http://localhost:5000
```

## 📖 Usage

### Indexing Therapy Books

```bash
# Index all books in data/therapy_books/
python index_documents.py

# Force reindexing
python index_documents.py --force-reindex

# Use custom config
python index_documents.py --config custom_config.yaml
```

### Running the Chat Application

```bash
# Development mode
FLASK_DEBUG=True python app.py

# Production mode
FLASK_ENV=production python app.py
```

### API Endpoints

- `POST /api/chat` - Send a message and get response
- `POST /api/feedback` - Submit feedback for a response
- `POST /api/reset` - Reset current conversation
- `GET /api/statistics` - Get system statistics

### Example API Usage

```python
import requests

# Send a chat message
response = requests.post('http://localhost:5000/api/chat', json={
    'message': 'I feel anxious about my exams'
})
print(response.json()['response'])

# Submit feedback
requests.post('http://localhost:5000/api/feedback', json={
    'message_id': 'msg_123',
    'user_message': 'I feel anxious',
    'assistant_response': '...',
    'rating': 5,
    'feedback_text': 'Very helpful!'
})
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Model settings**: LLM and embedding model parameters
- **Graph construction**: Entity types, relationship types, confidence thresholds
- **Retrieval**: Top-k values, similarity thresholds, graph depth
- **Conversation**: History length, context window, summarization
- **Therapeutic settings**: Age range, cultural context, crisis keywords

## 🧠 How It Works

### 1. Document Processing
- Therapy books are loaded and chunked into manageable pieces
- Each chunk is processed for entity extraction and relationship identification

### 2. Knowledge Graph Construction
- Entities (techniques, emotions, behaviors, etc.) become nodes
- Relationships (treats, causes, relates_to, etc.) become edges
- Graph captures semantic relationships between therapeutic concepts

### 3. Embedding Generation
- All chunks are embedded using EmbeddingGemma (300M parameters)
- Embeddings stored in FAISS index for fast similarity search

### 4. Query Processing
When a user sends a message:
1. **Vector Retrieval**: Find similar chunks using semantic search
2. **Graph Retrieval**: Extract relevant entities and traverse graph
3. **Context Assembly**: Combine chunks + graph knowledge
4. **Memory Integration**: Add conversation history
5. **Response Generation**: Gemma 3n generates empathetic response

### 5. Self-Learning
- User feedback is collected and analyzed
- Patterns identified for system improvement
- Interaction logs used for future enhancements

## 📊 System Statistics

Access statistics via API or logs:
- Number of indexed documents and chunks
- Graph size (nodes and edges)
- Vector store size
- Interaction counts
- Feedback metrics

## 🔒 Safety Features

- **Crisis Detection**: Identifies keywords related to self-harm, abuse, etc.
- **Resource Provision**: Provides helpline numbers (Childline 1098, AASRA)
- **Age-Appropriate**: Adjusts language based on user age
- **Cultural Sensitivity**: Respects Indian family values and contexts

## 🛠️ Development

### Project Structure
```
Graph-RAG/
├── app.py                      # Flask application
├── index_documents.py          # Indexing script
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── src/
│   ├── ingestion/             # Document loading
│   ├── graph/                 # Graph construction
│   ├── embedding/             # Embedding generation
│   ├── vector_store/          # Vector storage
│   ├── retrieval/             # Hybrid retrieval
│   ├── llm/                   # LLM integration
│   ├── conversation/          # Memory management
│   ├── chat/                  # Chat service
│   ├── feedback/              # Learning system
│   ├── pipeline/              # Indexing pipeline
│   └── utils/                 # Utilities
├── templates/
│   └── index.html             # Web interface
└── data/
    ├── therapy_books/         # Input books
    ├── graph_db.pkl           # Knowledge graph
    ├── vector_store.index     # FAISS index
    └── feedback.jsonl         # Feedback data
```

### Adding New Features

1. **New Entity Types**: Edit `config.yaml` → `graph.entity_types`
2. **New Relationships**: Edit `config.yaml` → `graph.relationship_types`
3. **Custom Prompts**: Modify `src/chat/chat_service.py` → `_create_system_prompt()`
4. **Additional Models**: Update `src/llm/` and `src/embedding/`

## 🤝 Contributing

This is a specialized therapeutic system. When contributing:
- Maintain empathetic and child-friendly language
- Respect cultural sensitivities
- Follow ethical AI guidelines for mental health applications
- Test thoroughly with diverse scenarios

## 📝 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

This chatbot is designed to provide supportive conversations and is **NOT a replacement for professional mental health care**. For serious mental health concerns, please consult qualified mental health professionals.

**Emergency Resources (India)**:
- Childline: 1098 (24/7 helpline for children)
- AASRA: 91-22-27546669 (24x7 crisis helpline)
- iCall: 91-22-25521111 (Mon-Sat, 8am-10pm)

## 🙏 Acknowledgments

- **Gemma Models**: Google's Gemma 3n E2B-it and EmbeddingGemma
- **Libraries**: Transformers, Sentence-Transformers, NetworkX, FAISS, Flask
- **Therapeutic Knowledge**: Based on evidence-based therapy literature

## 📧 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Built with ❤️ for the wellbeing of children and adolescents**
