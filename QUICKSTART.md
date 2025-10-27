# Quick Start Guide

Get your Graph RAG Psychotherapy Chatbot running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU with CUDA (optional but recommended)

## Installation Steps

### 1. Set Up Environment

```bash
# Navigate to project directory
cd /home/sanj-ai/Documents/SlateMate/Graph-RAG

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file (optional)
nano .env  # or use your preferred editor
```

**Important**: If you need to access gated models on HuggingFace, add your token to `.env`:
```
HF_TOKEN=your_huggingface_token_here
```

### 4. Add Therapy Books

```bash
# Create directory for books
mkdir -p data/therapy_books

# Add your therapy books (PDF, TXT, EPUB, or DOCX)
# Example:
# cp /path/to/your/therapy_book.pdf data/therapy_books/
```

**Note**: For testing, you can add any text-based psychology or therapy books. The system will extract therapeutic concepts automatically.

### 5. Index Documents

```bash
# Build knowledge graph and vector store
python index_documents.py
```

This will:
- Load all documents from `data/therapy_books/`
- Extract therapeutic entities and relationships
- Build knowledge graph
- Generate embeddings
- Create vector index

**Expected time**: 2-10 minutes depending on number of books and hardware.

### 6. Run the Application

```bash
# Start Flask server
python app.py
```

Or use the convenience script:
```bash
chmod +x run.sh
./run.sh
```

### 7. Access the Chatbot

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the therapeutic chatbot interface!

## Testing the System

### Sample Conversations

Try these example messages:

1. **Expressing anxiety**:
   ```
   I'm feeling really anxious about my exams coming up
   ```

2. **Sharing emotions**:
   ```
   I feel sad and don't know why
   ```

3. **Seeking help**:
   ```
   I'm 14 years old and having trouble sleeping because I worry a lot
   ```

4. **Cultural context**:
   ```
   My parents expect me to get top marks and I feel so much pressure
   ```

### Expected Behavior

The chatbot should:
- ✅ Respond empathetically
- ✅ Use age-appropriate language
- ✅ Validate feelings
- ✅ Provide therapeutic insights
- ✅ Show cultural awareness
- ✅ Display context information (chunks and graph nodes used)

## Troubleshooting

### Issue: "No documents found"

**Solution**: Add therapy books to `data/therapy_books/` directory and run `python index_documents.py`

### Issue: CUDA out of memory

**Solution**: Edit `config.yaml` and change:
```yaml
models:
  llm:
    device: "cpu"
  embedding:
    device: "cpu"
```

### Issue: Model download fails

**Solution**: 
1. Check internet connection
2. Add HuggingFace token to `.env` if using gated models
3. Try downloading manually:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
```

### Issue: spaCy model not found

**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### Issue: Port 5000 already in use

**Solution**: Change port in `.env`:
```
PORT=8000
```

Then run:
```bash
python app.py
```

## Next Steps

### Customize the System

1. **Adjust therapeutic approach**: Edit `src/chat/chat_service.py` → `_create_system_prompt()`

2. **Add more entity types**: Edit `config.yaml` → `graph.entity_types`

3. **Modify retrieval settings**: Edit `config.yaml` → `retrieval` section

4. **Change model parameters**: Edit `config.yaml` → `models` section

### Monitor Performance

Access statistics:
```bash
curl http://localhost:5000/api/statistics
```

View logs:
```bash
tail -f logs/graph_rag.log
```

### Add More Books

```bash
# Add new books to directory
cp new_book.pdf data/therapy_books/

# Reindex
python index_documents.py --force-reindex
```

## API Usage

### Send a message

```python
import requests

response = requests.post('http://localhost:5000/api/chat', json={
    'message': 'I feel anxious about school'
})

print(response.json()['response'])
```

### Submit feedback

```python
requests.post('http://localhost:5000/api/feedback', json={
    'message_id': 'unique_id',
    'user_message': 'I feel anxious',
    'assistant_response': 'Response text...',
    'rating': 5,
    'feedback_text': 'Very helpful!'
})
```

### Reset conversation

```python
requests.post('http://localhost:5000/api/reset')
```

## Production Deployment

For production use:

1. **Disable debug mode**:
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

2. **Use production server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. **Set up HTTPS** (recommended for production)

4. **Configure proper logging and monitoring**

## Getting Help

- Check `README.md` for detailed documentation
- Review `ARCHITECTURE.md` for system design
- Check logs in `logs/graph_rag.log`
- Review configuration in `config.yaml`

## Important Reminders

⚠️ **This is NOT a replacement for professional mental health care**

✅ **Always encourage users to seek professional help for serious concerns**

🔒 **Ensure data privacy and security in production deployments**

---

**You're all set! Start chatting with your therapeutic AI companion! 🌟**
