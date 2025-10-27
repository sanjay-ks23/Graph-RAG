#!/bin/bash

# Graph RAG Psychotherapy Chatbot - Run Script

echo "======================================"
echo "Graph RAG Psychotherapy Chatbot"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration!"
fi

# Check if data is indexed
if [ ! -f "data/graph_db.pkl" ] || [ ! -f "data/vector_store.index" ]; then
    echo ""
    echo "======================================"
    echo "No indexed data found!"
    echo "======================================"
    echo "Please add therapy books to data/therapy_books/ directory"
    echo "Then run: python index_documents.py"
    echo ""
    read -p "Do you want to index now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python index_documents.py
    fi
fi

# Run Flask app
echo ""
echo "======================================"
echo "Starting Flask Application..."
echo "======================================"
echo "Access the chatbot at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python app.py
