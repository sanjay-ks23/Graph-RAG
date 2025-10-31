# Graph RAG Psychotherapy Chatbot

A culturally-aware therapeutic chatbot for Indian children and adolescents using Graph RAG architecture with Gemma 3n E2B-it model.

## Overview

This system combines Graph-based Retrieval Augmented Generation with a fine-tuned language model to provide empathetic, contextual, and culturally-sensitive therapeutic support for children. It builds knowledge graphs from therapy literature to understand relationships between therapeutic concepts.


**Module Structure**

1. **Data Ingestion** (`src/ingestion/`) - Document loading, text chunking, metadata extraction
2. **Graph Construction** (`src/graph/`) - Entity extraction, relationship identification, knowledge graph building
3. **Embedding & Indexing** (`src/embedding/`, `src/vector_store/`) - Semantic embeddings, FAISS vector store
4. **Graph-based Retrieval** (`src/retrieval/`) - Hybrid vector and graph retrieval, context assembly
5. **LLM Integration** (`src/llm/`) - Gemma 3n E2B-it model, prompt construction, response generation
6. **Conversation Management** (`src/conversation/`) - Multi-turn memory, user profile extraction, session management
7. **Feedback & Learning** (`src/feedback/`) - Feedback collection, interaction logging, pattern analysis

## Memory Management

**CUDA Configuration**

The system includes automatic CUDA cache management. For limited GPU memory:

```yaml
# config.yaml
models:
  llm:
    device: "cuda"      # Keep LLM on GPU
  embedding:
    device: "cpu"       # Move embeddings to CPU
    batch_size: 32
```

**Environment Setup**

```bash
# .env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Memory Optimization**
- Automatic cache clearing at critical points
- Reduced batch sizes for embedding generation
- FP16 precision for CUDA operations
- Memory-efficient model loading


## Installation

**Prerequisites**
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM recommended

**Setup**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Configure environment (optional)
cp .env.example .env

# Add therapy books
mkdir -p data/therapy_books
# Copy PDF, TXT, EPUB, or DOCX files to data/therapy_books/

# Index documents
python index_documents.py

# Run application
python app.py
```

Access at `http://localhost:5000`

## Usage

**Indexing**

```bash
python index_documents.py                    # Index all books
python index_documents.py --force-reindex    # Force reindexing
python index_documents.py --config custom_config.yaml
```

**Running**

```bash
FLASK_DEBUG=True python app.py      # Development
FLASK_ENV=production python app.py  # Production
```

**API Endpoints**

- `POST /api/chat` - Send message and get response
- `POST /api/feedback` - Submit feedback
- `POST /api/reset` - Reset conversation
- `GET /api/statistics` - Get system statistics

**Example**

```python
import requests

response = requests.post('http://localhost:5000/api/chat', 
    json={'message': 'I feel anxious about my exams'})
print(response.json()['response'])
```

## Configuration

Edit `config.yaml` to customize model settings, graph construction, retrieval parameters, conversation management, and therapeutic settings.

## How It Works

**Document Processing**
- Therapy books are chunked and processed for entity extraction and relationship identification
- Entities become graph nodes, relationships become edges
- Chunks are embedded using EmbeddingGemma and stored in FAISS index

**Query Processing**
1. Vector retrieval finds similar chunks using semantic search
2. Graph retrieval extracts relevant entities and traverses relationships
3. Context assembly combines chunks and graph knowledge
4. Memory integration adds conversation history
5. Gemma 3n generates empathetic response

**Self-Learning**
- User feedback is collected and analyzed for system improvement


## Project Structure

```
Graph-RAG/
├── app.py                  # Flask application
├── index_documents.py      # Indexing script
├── config.yaml             # Configuration
├── requirements.txt        # Dependencies
├── src/
│   ├── ingestion/         # Document loading
│   ├── graph/             # Graph construction
│   ├── embedding/         # Embedding generation
│   ├── vector_store/      # Vector storage
│   ├── retrieval/         # Hybrid retrieval
│   ├── llm/               # LLM integration
│   ├── conversation/      # Memory management
│   ├── chat/              # Chat service
│   ├── feedback/          # Learning system
│   ├── pipeline/          # Indexing pipeline
│   └── utils/             # Utilities
├── templates/
│   └── index.html         # Web interface
└── data/
    ├── therapy_books/     # Input books
    ├── graph_db.pkl       # Knowledge graph
    ├── vector_store.index # FAISS index
    └── feedback.jsonl     # Feedback data
```



