# System Architecture - Graph RAG Psychotherapy Chatbot

## Overview

This document provides a detailed technical architecture of the Graph RAG-based psychotherapy chatbot system.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                             │
│                    (Flask Web Application)                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      CHAT SERVICE LAYER                           │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Session Mgmt   │  │ Memory Mgmt  │  │ Crisis Detection   │  │
│  └────────────────┘  └──────────────┘  └────────────────────┘  │
└────────────┬─────────────────────────────────────┬──────────────┘
             │                                      │
             ▼                                      ▼
┌────────────────────────────────┐    ┌───────────────────────────┐
│    RETRIEVAL & REASONING       │    │   GENERATION (LLM)        │
│  ┌──────────────────────────┐ │    │  ┌─────────────────────┐ │
│  │  Graph RAG Retriever     │ │    │  │  Gemma 3n E2B-it    │ │
│  │  • Vector Search         │ │    │  │  • Context Assembly │ │
│  │  • Graph Traversal       │ │    │  │  • Response Gen     │ │
│  │  • Context Assembly      │ │    │  └─────────────────────┘ │
│  └──────────────────────────┘ │    └───────────────────────────┘
└────────────┬───────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
│  Knowledge  │  │ Vector Store │  │ Embedding Model │
│    Graph    │  │   (FAISS)    │  │ (EmbeddingGemma)│
│ (NetworkX)  │  │              │  │                 │
└─────────────┘  └──────────────┘  └─────────────────┘
     ▲                 ▲                    ▲
     │                 │                    │
     └─────────────────┴────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  INDEXING PIPELINE      │
          │  • Document Loading     │
          │  • Entity Extraction    │
          │  • Graph Construction   │
          │  • Embedding Generation │
          └─────────────────────────┘
                       ▲
                       │
          ┌────────────▼────────────┐
          │   DATA SOURCES          │
          │   (Therapy Books)       │
          │   PDF, TXT, EPUB, DOCX  │
          └─────────────────────────┘
```

## Component Details

### 1. Data Ingestion & Preprocessing Module

**Location**: `src/ingestion/`

**Components**:
- `DocumentLoader`: Loads documents from multiple formats
- `TextChunker`: Splits documents into overlapping chunks

**Flow**:
```
Therapy Books → DocumentLoader → Raw Text
                                    ↓
                              TextChunker
                                    ↓
                         Chunks (512 tokens, 50 overlap)
```

**Key Features**:
- Multi-format support (PDF, TXT, EPUB, DOCX)
- Intelligent chunking with overlap for context preservation
- Metadata extraction and tracking

### 2. Graph Construction Module

**Location**: `src/graph/`

**Components**:
- `EntityExtractor`: Extracts therapeutic entities using NLP + patterns
- `GraphBuilder`: Constructs and manages knowledge graph

**Entity Types**:
- Therapeutic techniques (CBT, mindfulness, etc.)
- Emotions (anxiety, sadness, anger, etc.)
- Behaviors (avoidance, aggression, etc.)
- Cognitive patterns (negative thoughts, catastrophizing, etc.)
- Coping strategies (breathing, grounding, etc.)
- Developmental stages (toddler, adolescent, etc.)
- Cultural context (family values, academic pressure, etc.)

**Relationship Types**:
- `treats`: Technique treats emotion/behavior
- `causes`: Behavior causes emotion
- `relates_to`: General relationship
- `appropriate_for_age`: Technique suitable for age group
- `cultural_adaptation`: Cultural consideration

**Graph Structure**:
```
Node: {
  id: "entity_type:text",
  text: "cognitive behavioral therapy",
  type: "therapeutic_technique",
  confidence: 0.8,
  frequency: 15,
  sources: ["book1.pdf", "book2.pdf"],
  chunks: ["chunk_1", "chunk_5"]
}

Edge: {
  source: "therapeutic_technique:cbt",
  target: "emotion:anxiety",
  type: "treats",
  confidence: 0.9
}
```

### 3. Embedding & Indexing Module

**Location**: `src/embedding/`, `src/vector_store/`

**Components**:
- `EmbeddingModel`: Generates embeddings using EmbeddingGemma
- `FAISSVectorStore`: Stores and retrieves embeddings

**Embedding Pipeline**:
```
Text Chunks → EmbeddingGemma (300M) → 768-dim vectors
                                           ↓
                                    FAISS IVF Index
                                           ↓
                                  Fast Similarity Search
```

**FAISS Configuration**:
- Index Type: IVF (Inverted File Index)
- Dimension: 768
- Distance Metric: L2
- Normalization: L2 normalized embeddings

### 4. Graph-based Retrieval Module

**Location**: `src/retrieval/`

**Component**: `GraphRetriever`

**Retrieval Strategy**:

1. **Vector Retrieval**:
   - Encode query using EmbeddingGemma
   - Search FAISS index for top-k similar chunks
   - Apply similarity threshold filtering

2. **Graph Retrieval**:
   - Extract entities from query
   - Find matching nodes in graph
   - Traverse graph to depth N
   - Collect related nodes and relationships

3. **Context Assembly**:
   - Combine vector results + graph knowledge
   - Rank and deduplicate
   - Format for LLM consumption

**Retrieval Flow**:
```
User Query
    ↓
┌───┴────┐
│ Vector │ → Top-K Chunks (similarity > threshold)
│ Search │
└────────┘
    ↓
┌───┴────┐
│ Entity │ → Extract query entities
│Extract │
└────────┘
    ↓
┌───┴────┐
│ Graph  │ → Find nodes + traverse neighbors
│Traverse│
└────────┘
    ↓
┌───┴────┐
│Context │ → Chunks + Graph Knowledge + Relationships
│Assembly│
└────────┘
```

### 5. LLM Integration Module

**Location**: `src/llm/`

**Component**: `GemmaModel`

**Model**: Gemma 3n E2B-it (google/gemma-2-2b-it)

**Configuration**:
- Max Length: 8192 tokens
- Temperature: 0.7
- Top-p: 0.9
- Top-k: 40
- Context Window: 6144 tokens (reserved for input)

**Chat Format**:
```
<start_of_turn>user
{system_prompt}
<end_of_turn>
<start_of_turn>user
{context_prompt}
<end_of_turn>
<start_of_turn>user
{user_message_1}
<end_of_turn>
<start_of_turn>model
{assistant_response_1}
<end_of_turn>
<start_of_turn>user
{user_message_2}
<end_of_turn>
<start_of_turn>model
```

### 6. Conversation Management Module

**Location**: `src/conversation/`

**Components**:
- `ConversationMemory`: Manages conversation history
- `SessionManager`: Handles multiple user sessions

**Memory Structure**:
```
ConversationMemory {
  history: deque(maxlen=10),  // Last 10 turns
  summary: str,               // Summarized older context
  user_profile: {
    age: int,
    emotions_mentioned: [str],
    ...
  }
}
```

**Features**:
- Automatic summarization when history exceeds threshold
- User profile extraction (age, emotions, concerns)
- Token-aware context window management
- Session timeout and cleanup

### 7. Feedback & Learning Module

**Location**: `src/feedback/`

**Components**:
- `FeedbackSystem`: Collects and analyzes feedback
- `InteractionLogger`: Logs all interactions

**Learning Pipeline**:
```
User Feedback → Storage (JSONL)
                     ↓
              Pattern Analysis
                     ↓
         Identify Common Themes
                     ↓
    Update System (Future: Fine-tuning)
```

**Feedback Data**:
```json
{
  "timestamp": "2024-10-27T20:47:00",
  "session_id": "uuid",
  "user_message": "...",
  "assistant_response": "...",
  "rating": 4.5,
  "feedback_text": "helpful",
  "metadata": {...}
}
```

### 8. Chat Service (Orchestration)

**Location**: `src/chat/`

**Component**: `ChatService`

**Orchestration Flow**:
```
User Message
    ↓
Crisis Detection
    ↓
Retrieve Context (Graph RAG)
    ↓
Get Conversation History
    ↓
Build Prompt (System + Context + History + Message)
    ↓
Generate Response (Gemma)
    ↓
Add Crisis Support (if needed)
    ↓
Store in Memory
    ↓
Log Interaction
    ↓
Return Response
```

**Therapeutic Principles**:
- Active listening and validation
- Age-appropriate language
- Cultural sensitivity (Indian context)
- Strengths-based approach
- Safety-first (crisis detection)

## Data Flow

### Indexing Phase

```
1. Load Documents
   └─> DocumentLoader reads PDF/TXT/EPUB/DOCX

2. Chunk Documents
   └─> TextChunker creates 512-token chunks with 50-token overlap

3. Extract Entities
   └─> EntityExtractor identifies therapeutic concepts
   └─> Pattern matching + NLP (spaCy)

4. Build Graph
   └─> GraphBuilder creates nodes and edges
   └─> NetworkX MultiDiGraph

5. Generate Embeddings
   └─> EmbeddingGemma encodes all chunks
   └─> 768-dimensional vectors

6. Index Vectors
   └─> FAISSVectorStore creates IVF index
   └─> Stores metadata mapping

7. Persist
   └─> Save graph (pickle)
   └─> Save FAISS index
   └─> Save metadata
```

### Query Phase

```
1. Receive User Message
   └─> Flask API endpoint

2. Get/Create Session
   └─> SessionManager retrieves ConversationMemory

3. Detect Crisis
   └─> Check for crisis keywords

4. Retrieve Context
   └─> Vector search (FAISS)
   └─> Graph traversal (NetworkX)
   └─> Combine results

5. Get Therapeutic Context
   └─> Age-appropriate techniques
   └─> Emotion-specific strategies
   └─> Cultural considerations

6. Build Prompt
   └─> System prompt (therapeutic guidelines)
   └─> Context prompt (retrieved knowledge)
   └─> Conversation history
   └─> Current message

7. Generate Response
   └─> Gemma 3n E2B-it
   └─> Token management
   └─> Temperature-based sampling

8. Post-process
   └─> Add crisis support if needed
   └─> Format response

9. Store & Log
   └─> Update conversation memory
   └─> Log interaction
   └─> Return to user
```

## Scalability Considerations

### Current Implementation
- Single-server Flask application
- In-memory graph (NetworkX)
- Local FAISS index
- Session-based memory

### Future Enhancements
- **Graph Database**: Migrate to Neo4j for large-scale graphs
- **Distributed Vector Store**: Use Milvus or Pinecone
- **Model Serving**: Deploy with TensorRT or vLLM
- **Caching**: Redis for session management
- **Load Balancing**: Multiple Flask workers with Gunicorn
- **Async Processing**: Celery for background tasks

## Performance Optimization

### Indexing
- Batch processing for embeddings
- Parallel entity extraction
- Incremental graph updates

### Retrieval
- FAISS IVF for fast approximate search
- Graph traversal depth limiting
- Result caching

### Generation
- KV-cache for faster generation
- Quantization (FP16/INT8)
- Batch inference when possible

## Security & Privacy

### Data Protection
- No persistent user data storage
- Session-based memory only
- Automatic session cleanup

### Safety Measures
- Crisis keyword detection
- Helpline resource provision
- Age-appropriate content filtering

### Ethical Considerations
- Transparent limitations (not a replacement for therapy)
- Cultural sensitivity
- Child safety focus

## Monitoring & Logging

### Metrics Tracked
- Number of interactions
- Response times
- Retrieval quality (chunks/nodes used)
- User feedback ratings
- Crisis detections

### Logs
- Application logs (INFO/ERROR)
- Interaction logs (JSONL)
- Feedback logs (JSONL)

## Technology Stack

### Core
- **Python 3.8+**
- **PyTorch** (model inference)
- **Transformers** (Gemma models)
- **Sentence-Transformers** (EmbeddingGemma)

### Graph & Vector
- **NetworkX** (knowledge graph)
- **FAISS** (vector search)
- **NumPy** (numerical operations)

### NLP
- **spaCy** (entity extraction)
- **NLTK** (text processing)

### Web
- **Flask** (web framework)
- **Flask-CORS** (cross-origin support)

### Data Processing
- **PyPDF2** (PDF parsing)
- **ebooklib** (EPUB parsing)
- **python-docx** (DOCX parsing)

### Utilities
- **PyYAML** (configuration)
- **jsonlines** (logging)
- **tqdm** (progress bars)

---

**Last Updated**: October 2024
