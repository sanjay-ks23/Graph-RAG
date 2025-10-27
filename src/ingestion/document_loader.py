"""Document loading and preprocessing module"""

import os
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from docx import Document
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentLoader:
    """Load documents from various formats"""
    
    def __init__(self, books_directory: str):
        self.books_directory = Path(books_directory)
        self.supported_formats = {'.pdf', '.txt', '.epub', '.docx'}
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the books directory"""
        documents = []
        
        if not self.books_directory.exists():
            logger.warning(f"Books directory {self.books_directory} does not exist")
            return documents
        
        for file_path in self.books_directory.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.load_document(str(file_path))
                    if doc:
                        documents.append(doc)
                        logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a single document"""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self._load_pdf(file_path)
        elif suffix == '.txt':
            return self._load_txt(file_path)
        elif suffix == '.epub':
            return self._load_epub(file_path)
        elif suffix == '.docx':
            return self._load_docx(file_path)
        else:
            logger.warning(f"Unsupported format: {suffix}")
            return None
    
    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Load PDF document"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return {
            'source': file_path,
            'content': text,
            'format': 'pdf',
            'metadata': {'pages': len(pdf_reader.pages)}
        }
    
    def _load_txt(self, file_path: str) -> Dict[str, Any]:
        """Load text document"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return {
            'source': file_path,
            'content': text,
            'format': 'txt',
            'metadata': {}
        }
    
    def _load_epub(self, file_path: str) -> Dict[str, Any]:
        """Load EPUB document"""
        book = epub.read_epub(file_path)
        text = ""
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n"
        
        return {
            'source': file_path,
            'content': text,
            'format': 'epub',
            'metadata': {}
        }
    
    def _load_docx(self, file_path: str) -> Dict[str, Any]:
        """Load DOCX document"""
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        return {
            'source': file_path,
            'content': text,
            'format': 'docx',
            'metadata': {}
        }


class TextChunker:
    """Chunk text into manageable pieces"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces"""
        text = document['content']
        chunks = []
        
        # Split by sentences first
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'source': document['source'],
                    'chunk_id': len(chunks),
                    'metadata': document.get('metadata', {})
                })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.chunk_overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'source': document['source'],
                'chunk_id': len(chunks),
                'metadata': document.get('metadata', {})
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str], 
                               overlap_words: int) -> List[str]:
        """Get sentences for overlap"""
        overlap = []
        word_count = 0
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= overlap_words:
                overlap.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        return overlap
