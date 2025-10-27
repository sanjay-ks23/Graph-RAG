"""Main chat service integrating all components"""

from typing import Dict, Any, List
from src.llm.gemma_model import GemmaModel
from src.retrieval.graph_retriever import GraphRetriever
from src.conversation.memory_manager import ConversationMemory
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ChatService:
    """Therapeutic chat service with Graph RAG"""
    
    def __init__(self, gemma_model: GemmaModel, 
                 graph_retriever: GraphRetriever,
                 conversation_memory: ConversationMemory,
                 config: Dict[str, Any]):
        self.gemma_model = gemma_model
        self.graph_retriever = graph_retriever
        self.conversation_memory = conversation_memory
        self.config = config
        
        # Therapeutic settings
        self.therapy_config = config.get('therapy', {})
        self.crisis_keywords = self.therapy_config.get('crisis_keywords', [])
        
        # System prompt for therapeutic context
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for therapeutic chatbot"""
        return """You are a compassionate and empathetic psychotherapist specializing in working with children and adolescents (ages 0-18) in India. Your approach is:

1. **Empathetic and Warm**: Always validate feelings and show understanding
2. **Age-Appropriate**: Adjust language and concepts based on the child's age
3. **Culturally Sensitive**: Understand Indian family dynamics, academic pressures, and cultural values
4. **Strengths-Based**: Focus on the child's strengths and resilience
5. **Active Listening**: Reflect back what you hear and ask clarifying questions
6. **Safe and Supportive**: Create a judgment-free space for expression

**Important Guidelines**:
- Use simple, clear language appropriate for children
- Validate emotions before offering solutions
- Respect Indian cultural contexts (family values, academic expectations, festivals, traditions)
- Be patient and give the child time to express themselves
- Use therapeutic techniques like CBT, mindfulness, and play therapy concepts when appropriate
- If you detect crisis situations (self-harm, abuse, severe distress), gently encourage seeking immediate help from parents or professionals

**Your Role**: Help the child understand their emotions, develop coping strategies, and build resilience while being culturally aware and age-appropriate."""
    
    def generate_response(self, user_message: str, 
                         session_id: str = "default") -> Dict[str, Any]:
        """Generate therapeutic response"""
        
        # Check for crisis keywords
        crisis_detected = self._detect_crisis(user_message)
        
        # Retrieve relevant context
        user_profile = self.conversation_memory.get_user_profile()
        retrieved_context = self.graph_retriever.retrieve(
            user_message,
            user_context=user_profile
        )
        
        # Get therapeutic context if user profile available
        therapeutic_context = {}
        if 'age' in user_profile:
            therapeutic_context = self.graph_retriever.get_therapeutic_context(
                age=user_profile.get('age'),
                emotion=self._extract_primary_emotion(user_message)
            )
        
        # Build context-aware prompt
        context_prompt = self._build_context_prompt(
            retrieved_context, therapeutic_context
        )
        
        # Get conversation history
        history_messages = self.conversation_memory.get_context_messages(
            self.gemma_model.count_tokens
        )
        
        # Build messages for LLM
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'system', 'content': context_prompt}
        ]
        messages.extend(history_messages)
        messages.append({'role': 'user', 'content': user_message})
        
        # Generate response
        response = self.gemma_model.generate_response(messages)
        
        # Add crisis support if needed
        if crisis_detected:
            response = self._add_crisis_support(response)
        
        # Store in memory
        self.conversation_memory.add_turn(
            user_message, response,
            metadata={
                'crisis_detected': crisis_detected,
                'num_chunks_retrieved': len(retrieved_context.get('chunks', [])),
                'num_graph_nodes': retrieved_context.get('num_graph_nodes', 0)
            }
        )
        
        return {
            'response': response,
            'crisis_detected': crisis_detected,
            'context_used': {
                'chunks': len(retrieved_context.get('chunks', [])),
                'graph_nodes': retrieved_context.get('num_graph_nodes', 0)
            }
        }
    
    def _build_context_prompt(self, retrieved_context: Dict[str, Any],
                             therapeutic_context: Dict[str, Any]) -> str:
        """Build context prompt from retrieved information"""
        prompt_parts = []
        
        # Add retrieved knowledge
        if retrieved_context.get('chunks'):
            prompt_parts.append("**Relevant Therapeutic Knowledge**:")
            for i, chunk in enumerate(retrieved_context['chunks'][:3], 1):
                prompt_parts.append(f"{i}. {chunk['text'][:200]}...")
        
        # Add graph knowledge
        if retrieved_context.get('graph_knowledge'):
            prompt_parts.append("\n**Related Concepts**:")
            for knowledge in retrieved_context['graph_knowledge'][:5]:
                prompt_parts.append(f"- {knowledge}")
        
        # Add age-appropriate techniques
        if therapeutic_context.get('age_appropriate_techniques'):
            prompt_parts.append("\n**Age-Appropriate Approaches**:")
            for tech in therapeutic_context['age_appropriate_techniques'][:2]:
                prompt_parts.append(f"- {tech['text']}")
        
        # Add cultural considerations
        if therapeutic_context.get('cultural_considerations'):
            prompt_parts.append("\n**Cultural Context (Indian)**:")
            for cultural in therapeutic_context['cultural_considerations'][:2]:
                prompt_parts.append(f"- {cultural['text']}")
        
        if not prompt_parts:
            return "Use your therapeutic knowledge to respond empathetically."
        
        return "\n".join(prompt_parts)
    
    def _detect_crisis(self, message: str) -> bool:
        """Detect crisis keywords in message"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.crisis_keywords)
    
    def _add_crisis_support(self, response: str) -> str:
        """Add crisis support information to response"""
        crisis_message = """\n\n**Important**: I'm concerned about what you've shared. Please know that you're not alone, and there are people who can help immediately:
- Talk to a trusted adult (parent, teacher, school counselor)
- In India, you can call: Childline 1098 (24/7 helpline for children)
- AASRA: 91-22-27546669 (24x7 crisis helpline)

Your safety and wellbeing are the most important things."""
        
        return response + crisis_message
    
    def _extract_primary_emotion(self, message: str) -> str:
        """Extract primary emotion from message"""
        emotions = {
            'anxiety': ['anxious', 'worried', 'nervous', 'scared', 'fear'],
            'sadness': ['sad', 'depressed', 'down', 'unhappy', 'cry'],
            'anger': ['angry', 'mad', 'frustrated', 'annoyed'],
            'stress': ['stressed', 'overwhelmed', 'pressure']
        }
        
        message_lower = message.lower()
        for emotion, keywords in emotions.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion
        
        return None
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.conversation_memory.clear()
