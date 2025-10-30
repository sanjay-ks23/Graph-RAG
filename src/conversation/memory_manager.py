"""Conversation memory and context management"""

import re
import time
from typing import List, Dict, Any
from collections import deque
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

EMOTIONS = ['anxious', 'worried', 'scared', 'angry', 'sad', 'happy', 
            'excited', 'frustrated', 'lonely', 'confused']
AGE_PATTERNS = [
    r'i am (\d+) years? old',
    r"i'm (\d+) years? old",
    r'(\d+) years? old',
]

class ConversationMemory:
    """Manage conversation history and context"""
    
    def __init__(self, max_history_turns: int = 10, 
                 context_window_tokens: int = 6144,
                 summarization_threshold: int = 8):
        self.max_history_turns = max_history_turns
        self.context_window_tokens = context_window_tokens
        self.summarization_threshold = summarization_threshold
        
        self.history = deque(maxlen=max_history_turns)
        self.summary = None
        self.user_profile = {}
    
    def add_turn(self, user_message: str, assistant_message: str, 
                 metadata: Dict[str, Any] = None):
        """Add a conversation turn"""
        turn = {
            'user': user_message,
            'assistant': assistant_message,
            'metadata': metadata or {}
        }
        self.history.append(turn)
        
        # Update user profile
        self._update_user_profile(turn)
        
        # Check if summarization needed
        if len(self.history) >= self.summarization_threshold:
            self._summarize_if_needed()
    
    def get_context_messages(self, token_counter) -> List[Dict[str, str]]:
        """Get conversation history as messages within token limit"""
        messages = []
        total_tokens = 0
        
        # Add summary if exists
        if self.summary:
            summary_msg = {
                'role': 'system',
                'content': f"Previous conversation summary: {self.summary}"
            }
            summary_tokens = token_counter(summary_msg['content'])
            if summary_tokens < self.context_window_tokens // 4:
                messages.append(summary_msg)
                total_tokens += summary_tokens
        
        # Add recent history (reverse order)
        for turn in reversed(self.history):
            user_msg = {'role': 'user', 'content': turn['user']}
            assistant_msg = {'role': 'assistant', 'content': turn['assistant']}
            
            turn_tokens = token_counter(turn['user']) + token_counter(turn['assistant'])
            
            if total_tokens + turn_tokens > self.context_window_tokens:
                break
            
            messages.insert(0, assistant_msg)
            messages.insert(0, user_msg)
            total_tokens += turn_tokens
        
        return messages
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile extracted from conversation"""
        return self.user_profile
    
    def update_user_profile(self, key: str, value: Any):
        """Update user profile"""
        self.user_profile[key] = value
    
    def _update_user_profile(self, turn: Dict[str, Any]):
        """Extract and update user information from conversation"""
        user_message = turn['user'].lower()
        
        self._extract_age(user_message)
        self._extract_emotions(user_message)
    
    def _extract_age(self, message: str):
        """Extract age from message"""
        for pattern in AGE_PATTERNS:
            match = re.search(pattern, message)
            if match:
                age = int(match.group(1))
                if 0 <= age <= 18:
                    self.user_profile['age'] = age
                    break
    
    def _extract_emotions(self, message: str):
        """Extract emotions from message"""
        mentioned_emotions = [e for e in EMOTIONS if e in message]
        if mentioned_emotions:
            if 'emotions_mentioned' not in self.user_profile:
                self.user_profile['emotions_mentioned'] = []
            self.user_profile['emotions_mentioned'].extend(mentioned_emotions)
            self.user_profile['emotions_mentioned'] = \
                self.user_profile['emotions_mentioned'][-10:]
    
    def _summarize_if_needed(self):
        """Create summary of older conversations"""
        # Simple summarization: extract key points from older turns
        if len(self.history) >= self.summarization_threshold:
            older_turns = list(self.history)[:self.summarization_threshold // 2]
            
            key_points = []
            for turn in older_turns:
                # Extract user concerns
                if any(word in turn['user'].lower() for word in 
                      ['worried', 'anxious', 'scared', 'problem', 'help']):
                    key_points.append(f"User expressed: {turn['user'][:100]}")
            
            if key_points:
                self.summary = " | ".join(key_points[:3])
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        self.summary = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            'num_turns': len(self.history),
            'has_summary': self.summary is not None,
            'user_profile': self.user_profile
        }


class SessionManager:
    """Manage multiple conversation sessions"""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions = {}
        self.session_timeout_minutes = session_timeout_minutes
    
    def get_or_create_session(self, session_id: str, 
                             **memory_kwargs) -> ConversationMemory:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'memory': ConversationMemory(**memory_kwargs),
                'created_at': self._get_timestamp(),
                'last_active': self._get_timestamp()
            }
        else:
            self.sessions[session_id]['last_active'] = self._get_timestamp()
        
        return self.sessions[session_id]['memory']
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = self._get_timestamp()
        timeout_seconds = self.session_timeout_minutes * 60
        
        expired = []
        for session_id, session_data in self.sessions.items():
            if current_time - session_data['last_active'] > timeout_seconds:
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
            logger.info(f"Expired session removed: {session_id}")
    
    @staticmethod
    def _get_timestamp() -> float:
        """Get current timestamp"""
        return time.time()
