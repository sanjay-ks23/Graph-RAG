"""Self-learning and feedback system"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeedbackSystem:
    """Collect and process user feedback for self-learning"""
    
    def __init__(self, feedback_storage: str = "data/feedback.jsonl",
                 quality_threshold: float = 4.0,
                 update_frequency: int = 100):
        self.feedback_storage = Path(feedback_storage)
        self.quality_threshold = quality_threshold
        self.update_frequency = update_frequency
        
        self.feedback_storage.parent.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.total_feedback = 0
        self.positive_feedback = 0
        self.negative_feedback = 0
    
    def record_feedback(self, session_id: str, message_id: str,
                       user_message: str, assistant_response: str,
                       rating: float = None, feedback_text: str = None,
                       metadata: Dict[str, Any] = None):
        """Record user feedback"""
        
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'message_id': message_id,
            'user_message': user_message,
            'assistant_response': assistant_response,
            'rating': rating,
            'feedback_text': feedback_text,
            'metadata': metadata or {}
        }
        
        # Write to file
        with jsonlines.open(self.feedback_storage, mode='a') as writer:
            writer.write(feedback_entry)
        
        # Update statistics
        self.total_feedback += 1
        if rating:
            if rating >= self.quality_threshold:
                self.positive_feedback += 1
            else:
                self.negative_feedback += 1
        
        logger.info(f"Feedback recorded: rating={rating}")
        
        # Check if update needed
        if self.total_feedback % self.update_frequency == 0:
            self._trigger_learning_update()
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        if self.total_feedback == 0:
            return {
                'total_feedback': 0,
                'positive_rate': 0.0,
                'negative_rate': 0.0,
                'average_rating': 0.0
            }
        
        return {
            'total_feedback': self.total_feedback,
            'positive_feedback': self.positive_feedback,
            'negative_feedback': self.negative_feedback,
            'positive_rate': self.positive_feedback / self.total_feedback,
            'negative_rate': self.negative_feedback / self.total_feedback
        }
    
    def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent feedback entries"""
        if not self.feedback_storage.exists():
            return []
        
        feedback_list = []
        with jsonlines.open(self.feedback_storage) as reader:
            for entry in reader:
                feedback_list.append(entry)
        
        return feedback_list[-limit:]
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback to identify patterns"""
        recent_feedback = self.get_recent_feedback(limit=100)
        
        if not recent_feedback:
            return {}
        
        # Analyze common issues in negative feedback
        negative_patterns = []
        positive_patterns = []
        
        for entry in recent_feedback:
            rating = entry.get('rating')
            feedback_text = entry.get('feedback_text', '').lower()
            
            if rating and rating < self.quality_threshold and feedback_text:
                negative_patterns.append(feedback_text)
            elif rating and rating >= self.quality_threshold and feedback_text:
                positive_patterns.append(feedback_text)
        
        # Extract common words
        def extract_common_words(texts, top_k=10):
            from collections import Counter
            import re
            
            words = []
            for text in texts:
                words.extend(re.findall(r'\w+', text))
            
            # Filter common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            words = [w for w in words if w not in stop_words and len(w) > 3]
            
            return Counter(words).most_common(top_k)
        
        analysis = {
            'total_analyzed': len(recent_feedback),
            'negative_count': len(negative_patterns),
            'positive_count': len(positive_patterns),
            'common_negative_themes': extract_common_words(negative_patterns, 5),
            'common_positive_themes': extract_common_words(positive_patterns, 5)
        }
        
        return analysis
    
    def _trigger_learning_update(self):
        """Trigger learning update based on feedback"""
        logger.info(f"Learning update triggered after {self.update_frequency} interactions")
        
        # Analyze patterns
        analysis = self.analyze_feedback_patterns()
        
        # Log insights
        logger.info(f"Feedback analysis: {analysis}")
        
        # In a full implementation, this would:
        # 1. Fine-tune the model on high-quality interactions
        # 2. Update retrieval weights based on feedback
        # 3. Adjust graph relationships based on usage patterns
        # 4. Update prompt templates based on what works
        
        return analysis


class InteractionLogger:
    """Log all interactions for analysis and learning"""
    
    def __init__(self, log_path: str = "data/interactions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_interaction(self, session_id: str, user_message: str,
                       assistant_response: str, context_used: Dict[str, Any],
                       metadata: Dict[str, Any] = None):
        """Log an interaction"""
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'user_message': user_message,
            'assistant_response': assistant_response,
            'context_used': context_used,
            'metadata': metadata or {}
        }
        
        with jsonlines.open(self.log_path, mode='a') as writer:
            writer.write(interaction)
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a session"""
        if not self.log_path.exists():
            return []
        
        interactions = []
        with jsonlines.open(self.log_path) as reader:
            for entry in reader:
                if entry.get('session_id') == session_id:
                    interactions.append(entry)
        
        return interactions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interaction statistics"""
        if not self.log_path.exists():
            return {'total_interactions': 0}
        
        total = 0
        sessions = set()
        
        with jsonlines.open(self.log_path) as reader:
            for entry in reader:
                total += 1
                sessions.add(entry.get('session_id'))
        
        return {
            'total_interactions': total,
            'unique_sessions': len(sessions),
            'avg_interactions_per_session': total / len(sessions) if sessions else 0
        }
